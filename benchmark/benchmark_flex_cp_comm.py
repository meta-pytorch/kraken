import gc
import os
import random
import logging
import triton

import click
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
    context_parallel_unshard,
)
from torch.distributed.tensor.experimental._context_parallel._cp_custom_ops import (
    flex_cp_allgather,
)
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)

from kraken.comm import (
    FlexCPMaskedGather,
)

# Sequence dimension in [B, H, S, D] tensor layout
SEQ_DIM = 2


def prepare_batch(
    batch_size: int, seqlen: int, doc_len_mean: int, doc_len_std: int
) -> torch.Tensor:
    """Generate a batch tensor with document boundary markers.

    Creates a tensor of shape [batch_size, seqlen] where documents are
    represented as sequences of zeros followed by a 1 separator.
    """
    total_needed = batch_size * seqlen

    # Generate document lengths until we have enough to fill the tensor
    generated_doc_lens = []
    current_sum = 0
    while current_sum <= total_needed:
        doc_len = max(1, int(random.gauss(doc_len_mean, doc_len_std)))
        generated_doc_lens.append(doc_len)
        current_sum += doc_len + 1  # +1 for the separator

    # Create and fill the tensor
    tensor = torch.zeros(batch_size, seqlen, dtype=torch.int32)
    flat_view = tensor.view(-1)

    idx = 0
    for doc_len in generated_doc_lens:
        if idx >= total_needed:
            break
        # Skip doc_len positions (already zeros)
        idx += doc_len
        # Place separator if within bounds
        if idx < total_needed:
            flat_view[idx] = 1
            idx += 1

    return tensor


def get_document_mask_mod(batch: torch.Tensor, eos_id: int) -> _mask_mod_signature:
    """Creates a document mask that prevents attention across document boundaries.

    Args:
        batch: Input batch tensor with shape [b, s]
        eos_id: End-of-sequence token ID that marks document boundaries

    Returns:
        A mask modifier function that implements document-level masking.
    """
    # batch is [b, s] shape
    eos_mask = batch == eos_id
    eos_mask[:, -1] = True
    cumulative_mask = torch.cumsum(torch.where(eos_mask, 1, 0), dim=1)
    sequence_indices = torch.zeros_like(cumulative_mask, dtype=torch.int32)
    sequence_indices[:, 1:] = cumulative_mask[:, :-1]

    def document_mask(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return sequence_indices[b, q_idx] == sequence_indices[b, kv_idx]

    return document_mask


@nvshmem.requires_nvshmem
@triton.jit
def barrier_all_kernel():
    nvshmem.barrier_all()


def barrier_all():
    barrier_all_kernel[(1,)](num_ctas=1)


def cp_shard(
    mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_mask: BlockMask,
    load_balancer_type: str,
) -> tuple[tuple[torch.Tensor, ...], BlockMask]:
    cp_world_size = mesh.size(0)
    seqlen = inputs[0].shape[SEQ_DIM]

    # Create load balancer based on type
    if load_balancer_type == "headtail":
        load_balancer = _HeadTailLoadBalancer(seqlen, cp_world_size, "cuda")
    elif load_balancer_type == "ptrr":
        load_balancer = _PTRRLoadBalancer(attention_mask, cp_world_size)
    else:  # "none" or empty string
        load_balancer = None

    inputs = _context_parallel_shard(
        mesh=mesh,
        buffers=inputs,
        seq_dims=tuple(2 for _ in inputs),
        load_balancer=load_balancer,
    )

    # BlockMask, has shape, [B, H, Q, KV], and we can only shard
    # on the Q seq dimension, not KV.
    MASK_Q_SEQ_DIM = 2
    mask = _context_parallel_shard(
        mesh=mesh,
        buffers=[attention_mask],
        seq_dims=[MASK_Q_SEQ_DIM],
        load_balancer=load_balancer,
    )[0]

    return inputs, mask, load_balancer


@record
def benchmark(
    batch_size: int,
    nheads: int,
    kv_nheads: int,
    seqlen: int,
    dimension: int,
    doc_len_mean: int,
    doc_len_std: int,
    load_balancer_type: str,
    num_layers: int,
    num_iterations: int,
    cp_size: int,
):
    """Benchmark FlexAttention with flex_cp_allgather vs mask-aware communication.

    Args:
        batch_size: Batch size
        nheads: Number of query attention heads
        kv_nheads: Number of key/value attention heads (for GQA)
        seqlen: Sequence length
        dimension: Dimension per head
        doc_len_mean: Mean document length
        doc_len_std: Standard deviation of document length
        load_balancer_type: Type of load balancer
        num_layers: Number of layers to simulate per iteration (like transformer layers)
        num_iterations: Number of outer iterations with different random inputs
    """
    gc.disable()

    world_size = dist.get_world_size()
    enable_gqa = nheads != kv_nheads
    dtype = torch.bfloat16

    # Setup that only needs to happen once (outside iteration loop)
    tp_size = world_size // cp_size
    world_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("world",)
    )
    device_mesh = world_mesh._unflatten(0, (cp_size, tp_size), ("cp", "tp"))["cp"]
    pg_name = c10d._get_process_group_name(device_mesh.get_group())

    # Compile functions once with max-autotune for best performance
    compiled_create_block_mask = torch.compile(
        create_block_mask, dynamic=False, fullgraph=True
    )
    compiled_flex_attention = torch.compile(
        flex_attention,
        dynamic=False,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
    )

    # Create FlexCPMaskedGather instance for mask-aware communication
    masked_gatherer = FlexCPMaskedGather()

    # Storage for all timing results across iterations
    all_allgather_comm_times = []
    all_mask_aware_comm_times = []
    all_flex_attn_times = []
    all_total_allgather_times = []
    all_total_mask_aware_times = []
    all_valid_blocks = []

    # Run num_iterations + 1 times (first iteration is warmup, ignored)
    total_runs = num_iterations + 1

    for iteration in range(total_runs):
        is_warmup = iteration == 0

        # Generate new random input for each iteration
        batch_tensor = prepare_batch(batch_size, seqlen, doc_len_mean, doc_len_std)
        batch_tensor = batch_tensor.cuda()

        # Create document masking
        eos_id = 1
        document_mask_mod = get_document_mask_mod(batch_tensor, eos_id)

        # Create block mask
        block_mask = compiled_create_block_mask(
            document_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device="cuda",
        )

        # Create Q, K, V tensors
        q = torch.randn(
            batch_size, nheads, seqlen, dimension, device="cuda", dtype=dtype
        )
        k = torch.randn(
            batch_size, kv_nheads, seqlen, dimension, device="cuda", dtype=dtype
        )
        v = torch.randn(
            batch_size, kv_nheads, seqlen, dimension, device="cuda", dtype=dtype
        )

        # Shard Q, K, V and mask
        (q_shard, k_shard, v_shard), mask_shard, load_balancer = cp_shard(
            mesh=device_mesh,
            inputs=(q, k, v),
            attention_mask=block_mask,
            load_balancer_type=load_balancer_type,
        )

        # Pre-gather for compute-only benchmark
        k_gathered, v_gathered = flex_cp_allgather(k_shard, v_shard, SEQ_DIM, pg_name)

        # Create CUDA events for timing
        # For communication: single event pair for entire loop (mean only)
        allgather_comm_start = torch.cuda.Event(enable_timing=True)
        allgather_comm_end = torch.cuda.Event(enable_timing=True)
        mask_aware_comm_start = torch.cuda.Event(enable_timing=True)
        mask_aware_comm_end = torch.cuda.Event(enable_timing=True)

        # For others: per-operation events (median ± std)
        flex_attn_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(num_layers)
        ]
        total_allgather_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(num_layers)
        ]
        total_mask_aware_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(num_layers)
        ]

        torch.cuda.synchronize()

        # 1. AllGather communication loop (measure entire loop)
        allgather_comm_start.record()
        for _ in range(num_layers):
            kg, vg = flex_cp_allgather(k_shard, v_shard, SEQ_DIM, pg_name)
        allgather_comm_end.record()
        torch.cuda.synchronize()
        dist.barrier()
        logging.info("Benchmark Allgather Done")

        # 2. MaskAware communication loop (measure entire loop)
        mask_aware_comm_start.record()
        for _ in range(num_layers):
            km, vm = masked_gatherer.gather(
                k_shard, v_shard, mask_shard, seqlen, device_mesh
            )
        mask_aware_comm_end.record()
        torch.cuda.synchronize()
        dist.barrier()
        logging.info("Benchmark MaskedGather is Done")

        # 3. FlexAttention compute loop (per-operation measurement)
        for layer in range(num_layers):
            flex_attn_events[layer][0].record()
            _ = compiled_flex_attention(
                q_shard,
                k_gathered,
                v_gathered,
                block_mask=mask_shard,
                enable_gqa=enable_gqa,
            )
            flex_attn_events[layer][1].record()
        torch.cuda.synchronize()
        dist.barrier()
        logging.info("Benchmark FlexAttention Done")

        # 4. Total AllGather (comm + compute) loop (per-operation measurement)
        for layer in range(num_layers):
            total_allgather_events[layer][0].record()
            kg2, vg2 = flex_cp_allgather(k_shard, v_shard, SEQ_DIM, pg_name)
            _ = compiled_flex_attention(
                q_shard,
                kg2,
                vg2,
                block_mask=mask_shard,
                enable_gqa=enable_gqa,
            )
            total_allgather_events[layer][1].record()
        torch.cuda.synchronize()
        dist.barrier()
        logging.info("Benchmark Allgather + FlexAttention Done")

        # 5. Total MaskAware (comm + compute) loop (per-operation measurement)
        for layer in range(num_layers):
            total_mask_aware_events[layer][0].record()
            km2, vm2 = masked_gatherer.gather(
                k_shard, v_shard, mask_shard, seqlen, device_mesh
            )
            _ = compiled_flex_attention(
                q_shard,
                km2,
                vm2,
                block_mask=mask_shard,
                enable_gqa=enable_gqa,
            )
            total_mask_aware_events[layer][1].record()
        torch.cuda.synchronize()
        dist.barrier()
        logging.info("Benchmark MaskedGather + FlexAttention Done")

        # Skip warmup iteration
        if is_warmup:
            continue

        # Collect timing results
        # Communication: mean per operation from entire loop measurement
        allgather_comm_mean = (
            allgather_comm_start.elapsed_time(allgather_comm_end) / num_layers
        )
        mask_aware_comm_mean = (
            mask_aware_comm_start.elapsed_time(mask_aware_comm_end) / num_layers
        )
        all_allgather_comm_times.append(allgather_comm_mean)
        all_mask_aware_comm_times.append(mask_aware_comm_mean)

        # Others: per-operation times
        for layer in range(num_layers):
            all_flex_attn_times.append(
                flex_attn_events[layer][0].elapsed_time(flex_attn_events[layer][1])
            )
            all_total_allgather_times.append(
                total_allgather_events[layer][0].elapsed_time(
                    total_allgather_events[layer][1]
                )
            )
            all_total_mask_aware_times.append(
                total_mask_aware_events[layer][0].elapsed_time(
                    total_mask_aware_events[layer][1]
                )
            )

        # Collect valid blocks for this iteration (for statistics)
        blocks, valid_blocks, block_size = (
            FlexCPMaskedGather._get_required_blocks_from_mask(mask_shard, seqlen)
        )
        all_valid_blocks.append(valid_blocks.float().mean().item())

    # Compute local statistics for this rank
    def compute_stats(times):
        t = torch.tensor(times)
        return t.median().item(), t.std().item()

    def compute_mean(times):
        t = torch.tensor(times)
        return t.mean().item()

    # Communication: mean only (measured as entire loop)
    local_allgather_comm_mean = compute_mean(all_allgather_comm_times)
    local_mask_aware_comm_mean = compute_mean(all_mask_aware_comm_times)

    # Others: median ± std (per-operation measurement)
    local_flex_attn_median, local_flex_attn_std = compute_stats(all_flex_attn_times)
    local_total_allgather_median, local_total_allgather_std = compute_stats(
        all_total_allgather_times
    )
    local_total_mask_aware_median, local_total_mask_aware_std = compute_stats(
        all_total_mask_aware_times
    )

    # Gather stats from all ranks using all_gather_object
    local_stats = {
        "allgather_comm_mean": local_allgather_comm_mean,
        "mask_aware_comm_mean": local_mask_aware_comm_mean,
        "flex_attn_median": local_flex_attn_median,
        "flex_attn_std": local_flex_attn_std,
        "total_allgather_median": local_total_allgather_median,
        "total_allgather_std": local_total_allgather_std,
        "total_mask_aware_median": local_total_mask_aware_median,
        "total_mask_aware_std": local_total_mask_aware_std,
    }

    all_stats = [None for _ in range(world_size)]
    dist.all_gather_object(all_stats, local_stats, group=dist.group.WORLD)

    # Use max values across all ranks (slowest rank determines wall-clock time)
    allgather_comm_mean = max(s["allgather_comm_mean"] for s in all_stats)
    mask_aware_comm_mean = max(s["mask_aware_comm_mean"] for s in all_stats)
    flex_attn_median = max(s["flex_attn_median"] for s in all_stats)
    flex_attn_std = max(s["flex_attn_std"] for s in all_stats)
    total_allgather_median = max(s["total_allgather_median"] for s in all_stats)
    total_allgather_std = max(s["total_allgather_std"] for s in all_stats)
    total_mask_aware_median = max(s["total_mask_aware_median"] for s in all_stats)
    total_mask_aware_std = max(s["total_mask_aware_std"] for s in all_stats)

    avg_valid_blocks = sum(all_valid_blocks) / len(all_valid_blocks)
    total_blocks = seqlen // block_size
    block_reduction = (1 - avg_valid_blocks / total_blocks) * 100

    # Report statistics
    if dist.get_rank() == 0:
        print("\nBenchmark Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Q heads: {nheads}")
        print(f"  KV heads: {kv_nheads}")
        print(f"  GQA enabled: {enable_gqa}")
        print(f"  Sequence length: {seqlen}")
        print(f"  Dimension per head: {dimension}")
        print(f"  World size: {world_size}")
        print(f"  Load balancer: {load_balancer_type}")
        print(f"  Doc len mean: {doc_len_mean}")
        print(f"  Doc len std: {doc_len_std}")
        print(f"  Num layers: {num_layers}")
        print(f"  Num iterations: {num_iterations}")
        print(f"  Total samples: {num_iterations * num_layers}")

        print("\nMask Statistics:")
        print(f"  Total KV blocks: {total_blocks}")
        print(f"  Avg valid blocks: {avg_valid_blocks:.1f}")
        print(f"  Block reduction: {block_reduction:.1f}%")

        def fmt(median, std):
            return f"{median:.3f} ± {std:.3f}"

        print("\nTiming Breakdown (ms):")
        print(f"  AllGather Communication:    {allgather_comm_mean:.3f} (mean)")
        print(f"  MaskAware Communication:    {mask_aware_comm_mean:.3f} (mean)")
        print(
            f"  FlexAttention Compute:      {fmt(flex_attn_median, flex_attn_std)} (median ± std)"
        )
        print(
            f"  Total AllGather:            {fmt(total_allgather_median, total_allgather_std)} (median ± std)"
        )
        print(
            f"  Total MaskAware:            {fmt(total_mask_aware_median, total_mask_aware_std)} (median ± std)"
        )

        comm_speedup = allgather_comm_mean / mask_aware_comm_mean
        total_speedup = total_allgather_median / total_mask_aware_median

        print("\nSpeedups:")
        print(f"  Communication speedup: {comm_speedup:.2f}x")
        print(f"  Total speedup:         {total_speedup:.2f}x")


def test(
    batch_size: int,
    nheads: int,
    kv_nheads: int,
    seqlen: int,
    dimension: int,
    doc_len_mean: int,
    doc_len_std: int,
    load_balancer_type: str,
):
    """Test FlexAttention with flex_cp_allgather and mask-aware communication.

    Validates that both CP attention approaches produce the same results as
    non-CP attention by running all and comparing outputs after unsharding.
    """
    world_size = dist.get_world_size()
    dtype = torch.bfloat16

    # Setup
    device_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("cp",)
    )
    pg_name = c10d._get_process_group_name(device_mesh.get_group())

    # Create input batch
    batch_tensor = prepare_batch(batch_size, seqlen, doc_len_mean, doc_len_std)
    batch_tensor = batch_tensor.cuda()

    # Create document masking
    eos_id = 1
    document_mask_mod = get_document_mask_mod(batch_tensor, eos_id)

    # Compile create_block_mask
    compiled_create_block_mask = torch.compile(
        create_block_mask, dynamic=False, fullgraph=True
    )

    block_mask = compiled_create_block_mask(
        document_mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=seqlen,
        KV_LEN=seqlen,
        device="cuda",
    )

    # Create Q, K, V tensors
    q = torch.randn(batch_size, nheads, seqlen, dimension, device="cuda", dtype=dtype)
    k = torch.randn(
        batch_size, kv_nheads, seqlen, dimension, device="cuda", dtype=dtype
    )
    v = torch.randn(
        batch_size, kv_nheads, seqlen, dimension, device="cuda", dtype=dtype
    )

    # Shard Q, K, V and mask
    (q_shard, k_shard, v_shard), mask_shard, load_balancer = cp_shard(
        mesh=device_mesh,
        inputs=(q, k, v),
        attention_mask=block_mask,
        load_balancer_type=load_balancer_type,
    )

    # Check if GQA is enabled
    enable_gqa = nheads != kv_nheads

    # Create FlexCPMaskedGather instance for mask-aware communication
    masked_gatherer = FlexCPMaskedGather()

    # Compile flex_attention with max-autotune for best performance
    compiled_flex_attention = torch.compile(
        flex_attention,
        dynamic=False,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
    )

    # Run baseline (non-CP) flex_attention
    if dist.get_rank() == 0:
        print("Running baseline (non-CP) attention...")
    baseline_output = compiled_flex_attention(
        q, k, v, block_mask=block_mask, enable_gqa=enable_gqa
    )

    # Run CP attention with flex_cp_allgather
    if dist.get_rank() == 0:
        print("Running Context Parallel attention with flex_cp_allgather...")
    k_gathered, v_gathered = flex_cp_allgather(k_shard, v_shard, SEQ_DIM, pg_name)
    cp_output_allgather = compiled_flex_attention(
        q_shard, k_gathered, v_gathered, block_mask=mask_shard, enable_gqa=enable_gqa
    )

    # Run CP attention with mask-aware communication
    if dist.get_rank() == 0:
        print("Running Context Parallel attention with mask-aware communication...")
    k_gathered_masked, v_gathered_masked = masked_gatherer.gather(
        k_shard, v_shard, mask_shard, seqlen, device_mesh
    )
    cp_output_masked = compiled_flex_attention(
        q_shard,
        k_gathered_masked,
        v_gathered_masked,
        block_mask=mask_shard,
        enable_gqa=enable_gqa,
    )

    # Unshard CP outputs
    if dist.get_rank() == 0:
        print("Unsharding CP outputs...")
    cp_output_allgather_full = context_parallel_unshard(
        device_mesh,
        buffers=[cp_output_allgather],
        seq_dims=[SEQ_DIM],
        load_balancer=load_balancer,
    )[0]

    cp_output_masked_full = context_parallel_unshard(
        device_mesh,
        buffers=[cp_output_masked],
        seq_dims=[SEQ_DIM],
        load_balancer=load_balancer,
    )[0]

    # Validate outputs match
    if dist.get_rank() == 0:
        print("Validating outputs...")

        # Check flex_cp_allgather vs baseline
        torch.testing.assert_close(
            baseline_output, cp_output_allgather_full, atol=1e-3, rtol=1e-3
        )
        print("\n✓ flex_cp_allgather output matches baseline!")
        print(f"  Output shape: {baseline_output.shape}")
        print(
            "  Max absolute difference: "
            f"{(baseline_output - cp_output_allgather_full).abs().max().item():.6f}"
        )

        # Check mask-aware vs baseline
        torch.testing.assert_close(
            baseline_output, cp_output_masked_full, atol=1e-3, rtol=1e-3
        )
        print("\n✓ mask_aware_comm output matches baseline!")
        print(
            "  Max absolute difference: "
            f"{(baseline_output - cp_output_masked_full).abs().max().item():.6f}"
        )

        print("\n✓ ALL TESTS PASSED!")


@click.command()
@click.option("--batch_size", default=1)
@click.option("--nheads", default=32)
@click.option("--kv_nheads", default=8)
@click.option("--seqlen", default=65536)
@click.option("--dimension", default=4096)
@click.option("--doc_len_mean", default=16384)
@click.option("--doc_len_std", default=4096)
@click.option(
    "--load_balancer",
    default="headtail",
    type=click.Choice(["headtail", "ptrr", "none"]),
    help="Load balancer type: headtail, ptrr, or none",
)
@click.option("--layers", default=32, help="Number of layers per iteration")
@click.option(
    "--iterations",
    default=20,
    help="Number of iterations with different random inputs",
)
@click.option(
    "--cp_size", default=1, help="Context parallel size (outer mesh dimension)"
)
@click.option("--test_mode", default=False, is_flag=True)
def main(
    batch_size: int,
    nheads: int,
    kv_nheads: int,
    seqlen: int,
    dimension: int,
    doc_len_mean: int,
    doc_len_std: int,
    load_balancer: str,
    layers: int,
    iterations: int,
    cp_size: int,
    test_mode: bool,
):
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 benchmark_flex_cp_comm.py
    """
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(42)
    random.seed(42)
    logging.info("init_process_group is done")
    dist.barrier()

    if test_mode:
        test(
            batch_size,
            nheads,
            kv_nheads,
            seqlen,
            dimension // nheads,
            doc_len_mean,
            doc_len_std,
            load_balancer,
        )
    else:
        benchmark(
            batch_size,
            nheads,
            kv_nheads,
            seqlen,
            dimension // nheads,
            doc_len_mean,
            doc_len_std,
            load_balancer,
            layers,
            iterations,
            cp_size,
        )
    logging.info("Before destroy process group")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

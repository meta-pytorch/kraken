import random

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _HeadTailLoadBalancer,
)
from torch.nn.attention.flex_attention import _mask_mod_signature, create_block_mask
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

from kraken.comm import (
    FlexCPMaskedGather,
    get_required_blocks,
)


def copy_valid_blocks(
    blocks: torch.Tensor,
    valid_blocks: torch.Tensor,
    sharded_k: torch.Tensor,
    sharded_v: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    block_size: int = 128,
    cp_world_size: int = 2,
) -> None:
    """
    Copy valid blocks from sharded K/V tensors to output K/V tensors based on block indices.

    This implementation avoids CUDA host-device synchronization by not using
    nonzero(). All operations stay on GPU with pre-allocated index tensors.

    Memory-efficient version: Only creates index tensors for dimensions we
    actually index into, uses slicing for H and hidden dimensions.

    Args:
        blocks: (B, KV) tensor containing block indices. Each row contains
                block numbers, where only the first `valid_blocks[i]` entries
                are valid.
        valid_blocks: (B,) tensor indicating number of valid blocks for each
                      batch element.
        sharded_k: Source K tensor with shape (cp_world_size, B, H, KV_per_rank,
                   hidden). Context parallel sharded tensor.
        sharded_v: Source V tensor with shape (cp_world_size, B, H, KV_per_rank,
                   hidden). Context parallel sharded tensor.
        output_k: Destination K tensor with shape (B, H, KV_total, hidden).
        output_v: Destination V tensor with shape (B, H, KV_total, hidden).
        block_size: Size of each block in elements (default: 128).
        cp_world_size: Context parallel world size. Must be >= 2
                       (default: 2).

    Example:
        >>> cp_world_size = 4
        >>> B, KV, H, hidden = 2, 8, 32, 128
        >>> blocks = torch.tensor(
        ...     [[0, 5, 10, 17, 17, 17, 17, 17], [2, 7, 17, 17, 17, 17, 17, 17]]
        ... )
        >>> valid_blocks = torch.tensor([3, 2])
        >>> KV_per_rank = 512  # 2048 // 4
        >>> sharded_k = torch.randn(4, 2, 32, 512, 128)
        >>> sharded_v = torch.randn(4, 2, 32, 512, 128)
        >>> output_k = torch.zeros(2, 32, 2048, 128)
        >>> output_v = torch.zeros(2, 32, 2048, 128)
        >>> copy_valid_blocks(
        ...     blocks,
        ...     valid_blocks,
        ...     sharded_k,
        ...     sharded_v,
        ...     output_k,
        ...     output_v,
        ...     cp_world_size=4,
        ... )
    """
    B_blocks, KV = blocks.shape
    device = blocks.device

    # Context parallel sharded case
    cp_ws, B, H, KV_per_rank, hidden = sharded_k.shape
    _, _, KV_total, _ = output_k.shape
    blocks_per_rank = KV_per_rank // block_size

    # Pre-compute index combinations
    batch_idx = torch.arange(B, device=device)[:, None, None]
    kv_block_idx = torch.arange(KV, device=device)[None, :, None]
    offset_idx = torch.arange(block_size, device=device)[None, None, :]

    # Compute block numbers and derive rank info: (B, KV, 1)
    block_nums = blocks[:, :, None]
    rank_nums = block_nums // blocks_per_rank  # Which rank
    rank_block_nums = block_nums % blocks_per_rank  # Block within rank

    # Compute indices for output: (B, KV, block_size)
    kv_indices_output = block_nums * block_size + offset_idx

    # Compute indices for sharded: (B, KV, block_size)
    kv_indices_sharded = rank_block_nums * block_size + offset_idx

    # Create validity mask: (B, KV, block_size)
    valid_mask = kv_block_idx < valid_blocks[:, None, None]
    valid_mask = valid_mask.expand(B, KV, block_size)

    # Broadcast all indices: (B, KV, block_size)
    batch_broadcast = batch_idx.expand(B, KV, block_size)
    rank_broadcast = rank_nums.expand(B, KV, block_size)

    # Flatten for indexing: (B * KV * block_size,)
    batch_flat = batch_broadcast.reshape(-1)
    rank_flat = rank_broadcast.reshape(-1)
    kv_sharded_flat = kv_indices_sharded.reshape(-1)
    kv_output_flat = kv_indices_output.reshape(-1)
    mask_flat = valid_mask.reshape(-1)

    # Perform copy for both K and V with rank dimension
    output_k[batch_flat[mask_flat], :, kv_output_flat[mask_flat], :] = sharded_k[
        rank_flat[mask_flat], batch_flat[mask_flat], :, kv_sharded_flat[mask_flat], :
    ]
    output_v[batch_flat[mask_flat], :, kv_output_flat[mask_flat], :] = sharded_v[
        rank_flat[mask_flat], batch_flat[mask_flat], :, kv_sharded_flat[mask_flat], :
    ]


def copy_valid_blocks_simple(
    blocks: torch.Tensor,
    valid_blocks: torch.Tensor,
    sharded_k: torch.Tensor,
    sharded_v: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    block_size: int = 128,
    cp_world_size: int = 2,
) -> None:
    """
    Simple loop-based version for clarity and debugging.

    This is functionally equivalent to copy_valid_blocks() but uses explicit
    loops instead of vectorized operations. Useful for understanding the logic
    or debugging.

    Args:
        blocks: (B, KV) tensor containing block indices.
        valid_blocks: (B,) tensor indicating number of valid blocks for each
                      batch element.
        sharded_k: Source K tensor with shape (cp_world_size, B, H, KV_per_rank,
                   hidden). Context parallel sharded tensor.
        sharded_v: Source V tensor with shape (cp_world_size, B, H, KV_per_rank,
                   hidden). Context parallel sharded tensor.
        output_k: Destination K tensor with shape (B, H, KV_total, hidden).
        output_v: Destination V tensor with shape (B, H, KV_total, hidden).
        block_size: Size of each block in elements (default: 128).
        cp_world_size: Context parallel world size. Must be >= 2
                       (default: 2).
    """
    # Context parallel sharded case
    cp_ws, B, H, KV_per_rank, hidden = sharded_k.shape
    blocks_per_rank = KV_per_rank // block_size

    for batch_idx in range(B):
        num_valid = valid_blocks[batch_idx].item()
        for block_idx in range(num_valid):
            block_num = blocks[batch_idx, block_idx].item()

            # Calculate which rank and local block index
            rank = block_num // blocks_per_rank
            rank_block_idx = block_num % blocks_per_rank

            # Destination indices in output
            start_output = block_num * block_size
            end_output = start_output + block_size

            # Source indices in sharded (local to rank)
            start_sharded = rank_block_idx * block_size
            end_sharded = start_sharded + block_size

            # Copy K: output_k[b, :, block*128:(block+1)*128, :] =
            #         sharded_k[rank, b, :, rank_block*128:(rank_block+1)*128, :]
            output_k[batch_idx, :, start_output:end_output, :].copy_(
                sharded_k[rank, batch_idx, :, start_sharded:end_sharded, :]
            )
            # Copy V: output_v[b, :, block*128:(block+1)*128, :] =
            #         sharded_v[rank, b, :, rank_block*128:(rank_block+1)*128, :]
            output_v[batch_idx, :, start_output:end_output, :].copy_(
                sharded_v[rank, batch_idx, :, start_sharded:end_sharded, :]
            )


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


def cp_shard(
    mesh,
    inputs: tuple[torch.Tensor, ...],
    attention_mask,
):
    """Shard inputs and attention mask using context parallel."""
    cp_world_size = mesh.size(0)

    # load_balancer = _PTRRLoadBalancer(attention_mask, cp_world_size)
    load_balancer = _HeadTailLoadBalancer(inputs[0].shape[2], cp_world_size, "cuda")
    # load_balancer = None

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


@instantiate_parametrized_tests
class TestContextParallelComm(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42)
        random.seed(42)

    @skip_if_lt_x_gpu(4)
    def test_copy_valid_blocks_with_block_mask(self):
        """
        Merged test that combines get_required_blocks with copy_valid_blocks.
        Uses realistic FlexAttention block masks to get valid blocks,
        then tests FlexCPMaskedGather against Python implementations.
        """
        self._init_process()

        # Fixed test settings - same as test_required_blocks
        batch_size = 2
        seqlen = 8192 * 8
        doc_len_mean = 1024 * 16
        doc_len_std = 2048
        nheads = 1
        dimension = 128
        dtype = torch.bfloat16

        # Create device mesh for context parallel
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )

        # Create input batch with document boundaries
        batch_tensor = prepare_batch(batch_size, seqlen, doc_len_mean, doc_len_std)
        batch_tensor = batch_tensor.cuda()

        # Create document masking
        eos_id = 1  # The separator value from prepare_batch
        document_mask_mod = get_document_mask_mod(batch_tensor, eos_id)

        # Create block mask
        compiled_create_block_mask = torch.compile(create_block_mask)
        block_mask = compiled_create_block_mask(
            document_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device="cuda",
        )

        # Create dummy Q, K, V tensors for sharding
        q = torch.randn(
            batch_size, nheads, seqlen, dimension, device="cuda", dtype=dtype
        )
        k = torch.randn(
            batch_size, nheads, seqlen, dimension, device="cuda", dtype=dtype
        )
        v = torch.randn(
            batch_size, nheads, seqlen, dimension, device="cuda", dtype=dtype
        )

        # Shard inputs and mask
        (q_shard, k_shard, v_shard), mask_shard, load_balancer = cp_shard(
            mesh=device_mesh, inputs=(q, k, v), attention_mask=block_mask
        )

        # Gather all shards from all ranks for Python reference implementations
        cp_world_size = self.world_size
        k_shard_list = [torch.zeros_like(k_shard) for _ in range(cp_world_size)]
        v_shard_list = [torch.zeros_like(v_shard) for _ in range(cp_world_size)]

        dist.all_gather(k_shard_list, k_shard)
        dist.all_gather(v_shard_list, v_shard)

        # Stack them for Python implementations
        k_shard_gathered = torch.stack(k_shard_list, dim=0)
        v_shard_gathered = torch.stack(v_shard_list, dim=0)

        # Extract BlockMask components for Python implementations
        kv_indices = mask_shard.kv_indices
        kv_num_blocks = mask_shard.kv_num_blocks
        full_kv_indices = mask_shard.full_kv_indices
        full_kv_num_blocks = mask_shard.full_kv_num_blocks

        # Get required blocks
        blocks, max_kv_blocks = get_required_blocks(
            kv_indices, kv_num_blocks, full_kv_indices, full_kv_num_blocks
        )

        block_size = block_mask.BLOCK_SIZE[0]
        B = batch_size
        H = nheads
        KV_total = seqlen
        hidden = dimension

        # Count valid blocks per batch
        sentinel = max_kv_blocks + 1
        valid_blocks = (blocks != sentinel).sum(dim=1).to(torch.int32)

        # Create output tensors for Python implementations
        output_k_vec = torch.zeros(
            B, H, KV_total, hidden, device=self.device, dtype=dtype
        )
        output_k_simple = torch.zeros(
            B, H, KV_total, hidden, device=self.device, dtype=dtype
        )
        output_v_vec = torch.zeros(
            B, H, KV_total, hidden, device=self.device, dtype=dtype
        )
        output_v_simple = torch.zeros(
            B, H, KV_total, hidden, device=self.device, dtype=dtype
        )

        # Test Python implementations
        copy_valid_blocks(
            blocks,
            valid_blocks,
            k_shard_gathered,
            v_shard_gathered,
            output_k_vec,
            output_v_vec,
            block_size,
            cp_world_size,
        )
        copy_valid_blocks_simple(
            blocks,
            valid_blocks,
            k_shard_gathered,
            v_shard_gathered,
            output_k_simple,
            output_v_simple,
            block_size,
            cp_world_size,
        )

        # Test FlexCPMaskedGather (uses Triton kernel internally)
        gatherer = FlexCPMaskedGather()
        output_k_triton, output_v_triton = gatherer.gather(
            k_shard, v_shard, mask_shard, seqlen, device_mesh
        )

        # Verify Python implementations match each other
        assert torch.allclose(output_k_vec, output_k_simple), (
            "K: Vectorized and simple implementations don't match!"
        )
        assert torch.allclose(output_v_vec, output_v_simple), (
            "V: Vectorized and simple implementations don't match!"
        )

        # Verify FlexCPMaskedGather produces the same result (for valid blocks only)
        KV_per_rank = KV_total // cp_world_size
        blocks_per_rank = KV_per_rank // block_size
        for batch_idx in range(B):
            for block_idx in range(valid_blocks[batch_idx].item()):
                block_num = blocks[batch_idx, block_idx].item()
                assert block_num != sentinel, (
                    f"Unexpected sentinel value at batch {batch_idx}, "
                    f"block_idx {block_idx}"
                )

                rank = block_num // blocks_per_rank
                rank_block_idx = block_num % blocks_per_rank

                start_b = block_num * block_size
                end_b = start_b + block_size
                start_a = rank_block_idx * block_size
                end_a = start_a + block_size

                # Check K
                expected_k = k_shard_gathered[rank, batch_idx, :, start_a:end_a, :]
                actual_k_vec = output_k_vec[batch_idx, :, start_b:end_b, :]
                actual_k_triton = output_k_triton[batch_idx, :, start_b:end_b, :]
                assert torch.allclose(actual_k_vec, expected_k), (
                    f"K vec: Batch {batch_idx}, Block {block_num} not copied correctly!"
                )
                assert torch.allclose(actual_k_triton, expected_k), (
                    f"K triton: Batch {batch_idx}, Block {block_num} not copied correctly!"
                )

                # Check V
                expected_v = v_shard_gathered[rank, batch_idx, :, start_a:end_a, :]
                actual_v_vec = output_v_vec[batch_idx, :, start_b:end_b, :]
                actual_v_triton = output_v_triton[batch_idx, :, start_b:end_b, :]
                assert torch.allclose(actual_v_vec, expected_v), (
                    f"V vec: Batch {batch_idx}, Block {block_num} not copied correctly!"
                )
                assert torch.allclose(actual_v_triton, expected_v), (
                    f"V triton: Batch {batch_idx}, Block {block_num} not copied correctly!"
                )

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_flex_cp_masked_gather_class(self):
        """
        Test the FlexCPMaskedGather class interface.
        Verifies that the class produces the same result as Python reference
        implementations and that repeated calls work correctly.
        """
        self._init_process()

        # Fixed test settings
        batch_size = 2
        seqlen = 8192 * 8
        doc_len_mean = 1024 * 16
        doc_len_std = 2048
        nheads = 1
        dimension = 128
        dtype = torch.bfloat16

        # Create device mesh for context parallel
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )

        # Create input batch with document boundaries
        batch_tensor = prepare_batch(batch_size, seqlen, doc_len_mean, doc_len_std)
        batch_tensor = batch_tensor.cuda()

        # Create document masking
        eos_id = 1
        document_mask_mod = get_document_mask_mod(batch_tensor, eos_id)

        # Create block mask
        compiled_create_block_mask = torch.compile(create_block_mask)
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
            batch_size, nheads, seqlen, dimension, device="cuda", dtype=dtype
        )
        v = torch.randn(
            batch_size, nheads, seqlen, dimension, device="cuda", dtype=dtype
        )

        # Shard inputs and mask
        (q_shard, k_shard, v_shard), mask_shard, load_balancer = cp_shard(
            mesh=device_mesh, inputs=(q, k, v), attention_mask=block_mask
        )

        # Test using the FlexCPMaskedGather class
        gatherer = FlexCPMaskedGather()
        k_gathered, v_gathered = gatherer.gather(
            k_shard, v_shard, mask_shard, seqlen, device_mesh
        )

        # Verify output shapes
        B, H, KV_per_rank, hidden = k_shard.shape
        assert k_gathered.shape == (B, H, seqlen, hidden), (
            f"Expected k_gathered shape {(B, H, seqlen, hidden)}, "
            f"got {k_gathered.shape}"
        )
        assert v_gathered.shape == (B, H, seqlen, hidden), (
            f"Expected v_gathered shape {(B, H, seqlen, hidden)}, "
            f"got {v_gathered.shape}"
        )

        # Gather all shards for Python reference implementation
        cp_world_size = self.world_size
        k_shard_list = [torch.zeros_like(k_shard) for _ in range(cp_world_size)]
        v_shard_list = [torch.zeros_like(v_shard) for _ in range(cp_world_size)]
        dist.all_gather(k_shard_list, k_shard)
        dist.all_gather(v_shard_list, v_shard)
        k_shard_gathered = torch.stack(k_shard_list, dim=0)
        v_shard_gathered = torch.stack(v_shard_list, dim=0)

        # Get required blocks
        kv_indices = mask_shard.kv_indices
        kv_num_blocks = mask_shard.kv_num_blocks
        full_kv_indices = mask_shard.full_kv_indices
        full_kv_num_blocks = mask_shard.full_kv_num_blocks
        block_size = block_mask.BLOCK_SIZE[0]
        num_total_blocks = seqlen // block_size

        blocks, max_kv_blocks = get_required_blocks(
            kv_indices,
            kv_num_blocks,
            full_kv_indices,
            full_kv_num_blocks,
            num_total_blocks,
        )
        sentinel = max_kv_blocks + 1
        valid_blocks = (blocks != sentinel).sum(dim=1).to(torch.int32)

        # Create Python reference output
        output_k_ref = torch.zeros(
            B, H, seqlen, hidden, device=self.device, dtype=dtype
        )
        output_v_ref = torch.zeros(
            B, H, seqlen, hidden, device=self.device, dtype=dtype
        )
        copy_valid_blocks(
            blocks,
            valid_blocks,
            k_shard_gathered,
            v_shard_gathered,
            output_k_ref,
            output_v_ref,
            block_size,
            cp_world_size,
        )

        # Verify FlexCPMaskedGather outputs match reference for valid blocks
        for batch_idx in range(B):
            for block_idx in range(valid_blocks[batch_idx].item()):
                block_num = blocks[batch_idx, block_idx].item()
                start = block_num * block_size
                end = start + block_size

                assert torch.allclose(
                    k_gathered[batch_idx, :, start:end, :],
                    output_k_ref[batch_idx, :, start:end, :],
                ), (
                    f"FlexCPMaskedGather K output doesn't match at batch {batch_idx}, "
                    f"block {block_num}!"
                )
                assert torch.allclose(
                    v_gathered[batch_idx, :, start:end, :],
                    output_v_ref[batch_idx, :, start:end, :],
                ), (
                    f"FlexCPMaskedGather V output doesn't match at batch {batch_idx}, "
                    f"block {block_num}!"
                )

        # Test that calling gather again reuses symmetric memory and produces same result
        k_gathered2, v_gathered2 = gatherer.gather(
            k_shard, v_shard, mask_shard, seqlen, device_mesh
        )

        # Compare valid blocks only
        for batch_idx in range(B):
            for block_idx in range(valid_blocks[batch_idx].item()):
                block_num = blocks[batch_idx, block_idx].item()
                start = block_num * block_size
                end = start + block_size

                assert torch.allclose(
                    k_gathered[batch_idx, :, start:end, :],
                    k_gathered2[batch_idx, :, start:end, :],
                ), (
                    f"Second gather call produced different K at batch {batch_idx}, "
                    f"block {block_num}!"
                )
                assert torch.allclose(
                    v_gathered[batch_idx, :, start:end, :],
                    v_gathered2[batch_idx, :, start:end, :],
                ), (
                    f"Second gather call produced different V at batch {batch_idx}, "
                    f"block {block_num}!"
                )

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()

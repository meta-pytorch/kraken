import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .._ptx_utils import wait_gmem_barrier
from ..comm import _copy_engine_all_gather_w_progress


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    ret["flops8"] = 2.0 * M * N * K
    if "c_desc" in args:
        bytes_per_elem = args["c_desc"].base.element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret["bytes"] = bytes_per_elem * (M * K + N * K)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def _matmul_kernel_tma_persistent_w_progress(
    a_shared_desc,
    a_desc,
    b_desc,
    c_desc,
    progress_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_BLOCK_SIZE_M: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Persistent Triton kernel for matrix multiplication with progress waiting.

    This kernel performs matrix multiplication (`C = A @ B`) in a persistent manner.
    It waits for chunks of the `A` matrix to be gathered from other ranks by
    monitoring a `progress_ptr` before consuming them.
    """

    dtype = tl.float8e4nv if FP8_OUTPUT else tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am_src = 0
    offs_bn = 0
    current_a_desc = a_desc

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            NUM_COMM_BLOCKS = M // COMM_BLOCK_SIZE_M
            NUM_COMM_BLOCKS_PER_RANK = NUM_COMM_BLOCKS // WORLD_SIZE
            NUM_PID_M_PER_COMM_BLOCK = COMM_BLOCK_SIZE_M // BLOCK_SIZE_M

            # Pivot tile_id so that M tiles are processed in their ready order.
            # This pivot preserves the prior swizzling.
            pid_m = (pid_m + NUM_PID_M_PER_COMM_BLOCK * RANK) % num_pid_m

            comm_block_id = pid_m // NUM_PID_M_PER_COMM_BLOCK
            if comm_block_id // NUM_COMM_BLOCKS_PER_RANK == RANK:
                # Read from the local a_shard
                offs_am_src = (pid_m * BLOCK_SIZE_M) % COMM_BLOCK_SIZE_M
                current_a_desc = a_shared_desc
            else:
                # Wait for and read from a_shard copied from remote ranks
                wait_gmem_barrier(
                    progress_ptr + comm_block_id,
                    expect=1,
                    sem="acquire",
                    scope="gpu",
                    op="ld",
                )
                offs_am_src = pid_m * BLOCK_SIZE_M
                current_a_desc = a_desc

        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = ki * BLOCK_SIZE_K

        a = current_a_desc.load([offs_am_src, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            c_desc.store([pid_m * BLOCK_SIZE_M, offs_bn], c)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


_tma_desc_cache = {}


def _create_2d_tma_descriptor(tensor: torch.Tensor, block_dim1: int, block_dim0: int) -> TensorDescriptor:
    global _tma_desc_cache
    block_shape = (int(block_dim1), int(block_dim0))
    key = (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        block_shape,
        tensor.dtype,
        tensor.device,
    )
    desc = _tma_desc_cache.get(key)
    if desc is None:
        desc = TensorDescriptor.from_tensor(tensor, block_shape)
        _tma_desc_cache[key] = desc
    return desc


def _matmul_w_progress(
    a: torch.Tensor,
    a_shared: torch.Tensor,
    b: torch.Tensor,
    progress: torch.Tensor,
    configs: dict,
) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K2 == K

    bT = b.T
    if not bT.is_contiguous():
        raise ValueError("b.T must be contiguous")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    desc_a_shared = _create_2d_tma_descriptor(
        a_shared,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
    )

    desc_a = _create_2d_tma_descriptor(
        a,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
    )
    desc_bt = _create_2d_tma_descriptor(
        bT,
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
    )
    desc_c = _create_2d_tma_descriptor(
        c,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
    )

    configs["NUM_SMS"] = torch.cuda.get_device_properties(
        a.device
    ).multi_processor_count

    grid = lambda META: (  # noqa: E731
        min(
            configs["NUM_SMS"],
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    _matmul_kernel_tma_persistent_w_progress[grid](
        desc_a_shared,
        desc_a,
        desc_bt,
        desc_c,
        progress,
        M,
        N,
        K,
        BLOCK_SIZE_M=configs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs["GROUP_SIZE_M"],
        COMM_BLOCK_SIZE_M=configs["COMM_BLOCK_SIZE_M"],
        RANK=configs["RANK"],
        WORLD_SIZE=configs["WORLD_SIZE"],
        FP8_OUTPUT=a.dtype == torch.float8_e4m3fn,
        NUM_SMS=configs["NUM_SMS"],
        num_stages=configs["num_stages"],
        num_warps=configs["num_warps"],
    )

    return c


def all_gather_matmul(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    a_out: torch.Tensor | None = None,
    progress: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a fused all-gather and matrix multiplication operation.

    This function first performs an all-gather operation on the `a_shared` tensor
    to construct the full `A` matrix. It then performs a matrix multiplication
    `C = A @ B`. The all-gather is performed by the copy engine on a separate
    stream, and the matrix multiplication is performed by a persistent Triton
    kernel that waits for the data to be gathered.

    Args:
        a_shared (torch.Tensor): The local shard of the `A` matrix. This must
            be a symmetric tensor.
        b (torch.Tensor): The `B` matrix.
        a_out (torch.Tensor | None, optional): The output tensor for the
            all-gathered `A` matrix. If None, a new tensor is created.
        progress (torch.Tensor | None, optional): A tensor for tracking the
            progress of the all-gather operation. If None, a new tensor is
            created.
        **kwargs: Additional keyword arguments for kernel configuration:
            splits_per_rank (int, optional): The number of splits for the
                all-gather operation. Defaults to 1.
            block_size_m (int, optional): The block size for the M dimension.
                Defaults to 128.
            block_size_n (int, optional): The block size for the N dimension.
                Defaults to 256.
            block_size_k (int, optional): The block size for the K dimension.
                Defaults to 64.
            group_size_m (int, optional): The group size for the M dimension.
                Defaults to 4.
            num_stages (int, optional): The number of stages for the matmul
                kernel. Defaults to 3.
            num_warps (int, optional): The number of warps for the matmul
                kernel. Defaults to 8.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the all-gathered
            `A` matrix and the result of the matrix multiplication `C`.
    """
    configs = {
        "SPLITS_PER_RANK": kwargs.get("splits_per_rank", 1),
        "BLOCK_SIZE_M": kwargs.get("block_size_m", 128),
        "BLOCK_SIZE_N": kwargs.get("block_size_n", 256),
        "BLOCK_SIZE_K": kwargs.get("block_size_k", 64),
        "GROUP_SIZE_M": kwargs.get("group_size_m", 4),
        "num_stages": kwargs.get("num_stages", 3),
        "num_warps": kwargs.get("num_warps", 8),
    }

    symm_mem_hdl = symm_mem.rendezvous(a_shared, group=dist.group.WORLD)

    a_shape = list(a_shared.shape)
    a_shape[0] *= symm_mem_hdl.world_size

    configs["RANK"] = symm_mem_hdl.rank
    configs["WORLD_SIZE"] = symm_mem_hdl.world_size
    if (
        configs["SPLITS_PER_RANK"]
        * configs["WORLD_SIZE"]
        * configs["BLOCK_SIZE_M"]
        * configs["GROUP_SIZE_M"]
        > a_shape[0]
    ):
        configs["GROUP_SIZE_M"] = 1
        configs["SPLITS_PER_RANK"] = 1

    configs["COMM_BLOCK_SIZE_M"] = (
        a_shape[0] // configs["WORLD_SIZE"] // configs["SPLITS_PER_RANK"]
    )
    assert (
        configs["COMM_BLOCK_SIZE_M"]
        % (configs["BLOCK_SIZE_M"] * configs["GROUP_SIZE_M"])
        == 0
    )

    if a_out is None:
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)

    if progress is None:
        progress = torch.zeros(
            symm_mem_hdl.world_size * configs["SPLITS_PER_RANK"],
            dtype=torch.uint32,
            device=a_shared.device,
        )
    else:
        progress.fill_(0)  # Reset progress to 0.

    # Perform all-gather using the copy engine on a backend stream.
    backend_stream = _copy_engine_all_gather_w_progress(
        a_out, a_shared, progress, configs["SPLITS_PER_RANK"]
    )

    # Perform matrix multiplication on gathered a, which waits for signal of completion for each chunk of a.
    c = _matmul_w_progress(a_out, a_shared, b, progress, configs)

    torch.cuda.current_stream().wait_stream(backend_stream)

    return a_out, c

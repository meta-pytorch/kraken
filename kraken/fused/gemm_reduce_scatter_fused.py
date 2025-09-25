import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .. import _ptx_utils as ptx_utils


@triton.jit
def gemm_reduce_scatter_kernel(
    a_ptr,
    b_ptr,
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_symm_m: tl.constexpr,
    stride_symm_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Fused GEMM + Reduce-Scatter kernel.
    Computes C = A @ B locally, then performs reduce-scatter across ranks.
    The result is scattered along the M dimension.
    """

    # 1. Program ID and Tiling Calculation
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. Local GEMM computation
    # We do A @ B and C gets stored in rank's symm mem buffer

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # GEMM Computation
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # This is the full C matrix
    c_local = accumulator.to(a_ptr.dtype.element_ty)

    # Get this rank's buffer in the symmetric memory space
    my_buffer_ptr = buf_tuple[rank]

    # Calculate where to store this tile in the buffer
    offs_symm_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_symm_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create pointers to the symmetric memory location
    symm_ptrs = (
        my_buffer_ptr
        + stride_symm_m * offs_symm_m[:, None]
        + stride_symm_n * offs_symm_n[None, :]
    )

    # Store the C in the rank's symmetric memory buffer
    mask_mn = (offs_symm_m[:, None] < M) & (offs_symm_n[None, :] < N)
    tl.store(symm_ptrs, c_local, mask=mask_mn)

    # synchronize
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True
    )

    # Reduce Scatter logic: For each tile in the rank's assigned row slice (along M),
    # sum corresponding tiles from all ranks' buffers and store the reduced tile directly in the local output.
    # This is to avoid full global sum on any rank.

    # Compute the size of each scattered slice (rows per rank)
    M_scatter = M // world_size

    # Figure out rank's assigned row range in the global output
    my_scatter_start_row = rank * M_scatter
    my_scatter_end_row = (rank + 1) * M_scatter

    # Get the starting row of the current tile this block is handling
    current_tile_start_row = pid_m * BLOCK_SIZE_M

    # If the program's tile falls into scattered output slice for this rank
    if (current_tile_start_row >= my_scatter_start_row) and (
        current_tile_start_row < my_scatter_end_row
    ):
        # This block is responsible for computing a tile of the final output

        # Reduce results from all ranks for this specific tile
        acc_reduce = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in tl.static_range(world_size):
            buffer_rank_ptr = buf_tuple[i]
            remote_tile_ptrs = (
                buffer_rank_ptr
                + stride_symm_m * offs_symm_m[:, None]
                + stride_symm_n * offs_symm_n[None, :]
            )
            c_block = tl.load(remote_tile_ptrs, mask=mask_mn, other=0.0)
            acc_reduce += c_block

        # Calculate offset into the local scattered output tensor
        offs_out_m = (
            current_tile_start_row - my_scatter_start_row + tl.arange(0, BLOCK_SIZE_M)
        )

        offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        output_ptrs = (
            output_ptr
            + stride_out_m * offs_out_m[:, None]
            + stride_out_n * offs_out_n[None, :]
        )

        mask_out = (offs_out_m[:, None] < M_scatter) & (offs_out_n[None, :] < N)

        # Store the reduced tile to the output and cast to orig dtype
        tl.store(output_ptrs, acc_reduce.to(output_ptr.dtype.element_ty), mask=mask_out)

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )


def gemm_reduce_scatter(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Fused GEMM + Reduce-Scatter operation.
    Computes C = A @ B on each rank, then performs reduce-scatter to sum results
    and scatter them along the M dimension.
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    Returns:
        Output matrix of shape (M / world_size, N) containing the reduce-scattered result.
    """

    assert a.shape[1] == b.shape[0], (
        "Inner dimensions must match for matrix multiplication"
    )
    M, K = a.shape
    _, N = b.shape

    group = kwargs.get("group", dist.group.WORLD)
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    assert M % world_size == 0, (
        f"M dimension ({M}) must be divisible by world_size ({world_size})"
    )

    # Configuration stuff
    BLOCK_SIZE_M = kwargs.get("BLOCK_SIZE_M", 64)
    BLOCK_SIZE_N = kwargs.get("BLOCK_SIZE_N", 64)
    BLOCK_SIZE_K = kwargs.get("BLOCK_SIZE_K", 32)
    GROUP_SIZE_M = kwargs.get("GROUP_SIZE_M", 8)
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 3)
    assert a.dtype == b.dtype, "Input tensors must have the same dtype"
    assert a.dtype == torch.float32, "Only float32 is supported for now"

    M_scatter = M // world_size
    # Create output tensor for the scattered result
    output = torch.empty((M_scatter, N), dtype=a.dtype, device=a.device)

    # Create a symmetric buffer for the GEMM results
    gemm_buffer = symm_mem.empty((M, N), dtype=a.dtype, device=a.device)
    symm_mem_hdl = symm_mem.rendezvous(gemm_buffer, group=group)

    # Create buffer tuple for all ranks
    buf_list = [
        symm_mem_hdl.get_buffer(i, tuple((M, N)), a.dtype)
        for i in range(symm_mem_hdl.world_size)
    ]
    buf_tuple = tuple(buf_list)

    # Launch kernel
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    gemm_reduce_scatter_kernel[grid](
        a,
        b,
        buf_tuple,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_symm_m=gemm_buffer.stride(0),
        stride_symm_n=gemm_buffer.stride(1),
        stride_out_m=output.stride(0),
        stride_out_n=output.stride(1),
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

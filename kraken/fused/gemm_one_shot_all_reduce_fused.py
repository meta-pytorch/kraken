import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .. import _ptx_utils as ptx_utils


@triton.jit
def gemm_one_shot_all_reduce_kernel(
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
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused GEMM + All-Reduce kernel.
    Computes C = A @ B locally, then performs all-reduce across ranks.
    """

    # Get program IDs and tile indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B for local gemm
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulator for GEMM
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # GEMM computation
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c_local = acc.to(tl.float32)

    # Write local GEMM result to symmetric memory
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Get this rank's buffer from the tuple
    my_buffer_ptr = buf_tuple[rank]

    # Compute addresses in symmetric buffer for each element and store tile result into symmem
    c_ptrs = my_buffer_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c_local, mask=mask)

    # Synchronize before all-reduce
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True
    )

    # All-reduce: sum results from all ranks
    acc_reduce = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in tl.static_range(world_size):
        buffer_rank = buf_tuple[i]
        c_ptrs = (
            buffer_rank + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        )
        c_block = tl.load(c_ptrs, mask=mask, other=0.0)
        acc_reduce += c_block

    # Store final result
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    output_ptrs = (
        output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    )
    tl.store(output_ptrs, acc_reduce, mask=mask)
    # Final synchronization
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )


def gemm_one_shot_all_reduce(
    a: torch.Tensor, b: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Fused GEMM + All-Reduce operation.
    Computes C = A @ B on each rank, then performs all-reduce to sum results.
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    Returns:
        Output matrix of shape (M, N) containing the all-reduced result
    """

    assert a.shape[1] == b.shape[0], (
        "Inner dimensions must match for matrix multiplication"
    )

    M, K = a.shape
    K, N = b.shape
    group = kwargs.get("group", None)
    group = dist.group.WORLD if group is None else group

    # Configuration
    BLOCK_SIZE_M = kwargs.get("BLOCK_SIZE_M", 64)
    BLOCK_SIZE_N = kwargs.get("BLOCK_SIZE_N", 64)
    BLOCK_SIZE_K = kwargs.get("BLOCK_SIZE_K", 64)
    GROUP_SIZE_M = kwargs.get("GROUP_SIZE_M", 8)
    num_warps = kwargs.get("num_warps", 4)
    num_stages = kwargs.get("num_stages", 3)

    output = torch.empty((M, N), dtype=torch.float32, device=a.device)

    symm_mem_hdl = symm_mem.get_symm_mem_workspace(
        group.group_name, min_size=M * N * output.element_size()
    )

    buf_list = [
        symm_mem_hdl.get_buffer(i, [M, N], torch.float32, 0)
        for i in range(symm_mem_hdl.world_size)
    ]
    buf_tuple = tuple(buf_list)
    gemm_buffer = buf_list[symm_mem_hdl.rank]

    # Launch kernel
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    gemm_one_shot_all_reduce_kernel[grid](
        a,
        b,
        buf_tuple,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        gemm_buffer.stride(0),
        gemm_buffer.stride(1),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
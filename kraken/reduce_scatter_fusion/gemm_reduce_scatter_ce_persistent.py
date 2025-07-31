import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import triton
import triton.language as tl
from cuda.bindings import driver

from .._ptx_utils import get_flat_tid, send_signal


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_desc_ptr" in args:
        bytes_per_elem = args["c_desc_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K)
    return ret


_tma_desc_cache = {}


def _create_2d_tma_descriptor(ptr, dim1, dim0, block_dim1, block_dim0, element_size):
    global _tma_desc_cache
    key = (ptr, dim1, dim0, block_dim1, block_dim0, element_size)
    if key in _tma_desc_cache:
        return _tma_desc_cache[key]
    desc = triton.tools.experimental_descriptor.create_2d_tma_descriptor(
        ptr,
        dim1,
        dim0,
        block_dim1,
        block_dim0,
        element_size,
    )
    _tma_desc_cache[key] = desc
    return desc


@triton.jit(launch_metadata=_matmul_launch_metadata)
def _gemm_producer_persistent_kernel(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    progress_ptr,
    signal_pad_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
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
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    M_per_rank = M // WORLD_SIZE

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pid_m_offset = (RANK + 1) * M_per_rank // BLOCK_SIZE_M

    for _ in range(k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # Pivot tile_id so that M tiles are processed in communication order.
            # This pivot preserves the prior swizzling.
            pid_m = (pid_m + pid_m_offset) % num_pid_m

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(
            a_desc_ptr, [offs_am, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)
            tl._experimental_descriptor_store(c_desc_ptr, c, [offs_am, offs_bn])

            # calculate progress and send signals to corresponding ranks
            scatter_start = offs_am // M_per_rank
            scatter_end = (offs_am + BLOCK_SIZE_M - 1) // M_per_rank
            scatter_end = min(scatter_end, WORLD_SIZE - 1)

            for rank in range(scatter_start, scatter_end + 1):
                m_start = M_per_rank * rank
                m_end = M_per_rank * (rank + 1) - 1
                tiled_m_start = m_start // BLOCK_SIZE_M
                tiled_m_end = m_end // BLOCK_SIZE_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                val = tl.atomic_add(progress_ptr + rank, 1, sem="release", scope="gpu")
                if val == tiled_m_size * num_pid_n - 1:
                    send_addr = signal_pad_ptr + rank
                    if get_flat_tid() == 0:
                        send_signal(send_addr, "release")

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def copy_engine_scatter(
    inp: torch.Tensor,
    output: torch.Tensor,  # Must be symmetric tensor
    signal_pad: torch.Tensor,
    group: dist.ProcessGroup | None = None,
):
    assert output.is_contiguous()
    M, N = inp.shape

    symm_mem_hdl = symm_mem.get_symm_mem_workspace(
        group.group_name, min_size=output.numel() * output.element_size()
    )

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size
    M_per_rank = M // world_size

    # copy gemm tiles to corresponding ranks
    stream = torch.cuda.current_stream()
    for step in range(world_size):
        remote_rank = (rank + step + 1) % world_size

        # wait signal from gemm kernel
        signal_pad_ptr = signal_pad.data_ptr()
        signal_ele_size = signal_pad.element_size()
        wait_addr = signal_pad_ptr + signal_ele_size * remote_rank
        driver.cuStreamWaitValue32(
            stream.cuda_stream,
            wait_addr,
            1,
            driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )

        offset = rank * M_per_rank * N
        remote_buf = symm_mem_hdl.get_buffer(
            remote_rank, [M_per_rank, N], inp.dtype, offset
        )
        src_buf = inp[remote_rank * M_per_rank : (remote_rank + 1) * M_per_rank, :]
        remote_buf.copy_(src_buf)


def gemm_producer_w_progress(
    a: torch.Tensor,
    b: torch.Tensor,
    gemm_out: torch.Tensor,
    progress: torch.Tensor,
    signal_pad: torch.Tensor,
    configs: dict,
    group: dist.ProcessGroup | None = None,
):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Inner dimensions must match for matrix multiplication"
    assert a.dtype == b.dtype, "Input dtypes must match"

    bT = b.T

    desc_a = _create_2d_tma_descriptor(
        a.data_ptr(),
        M,
        K,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
        a.element_size(),
    )
    desc_bt = _create_2d_tma_descriptor(
        bT.data_ptr(),
        N,
        K,
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        bT.element_size(),
    )
    desc_c = _create_2d_tma_descriptor(
        gemm_out.data_ptr(),
        M,
        N,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        gemm_out.element_size(),
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

    group = dist.group.WORLD if group is None else group

    _gemm_producer_persistent_kernel[grid](
        desc_a,
        desc_bt,
        desc_c,
        progress,
        signal_pad,
        M,
        N,
        K,
        BLOCK_SIZE_M=configs["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs["GROUP_SIZE_M"],
        RANK=configs["RANK"],
        WORLD_SIZE=configs["WORLD_SIZE"],
        FP8_OUTPUT=a.dtype == torch.float8_e4m3fn,
        NUM_SMS=configs["NUM_SMS"],
        num_stages=configs["num_stages"],
        num_warps=configs["num_warps"],
    )


@triton.jit
def _reduce_persistent_kernel(
    in_desc_ptr,  # [M, N]
    out_desc_ptr,  # [M_per_rank, N]
    M_per_rank,
    N,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 256,
    BLOCK_SIZE_N: tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n
    for tile_id in range(pid, total_tiles, num_pid):
        tile_id_m = tile_id // num_tiles_n
        tile_id_n = tile_id % num_tiles_n
        cur_rank = (RANK + 1) % WORLD_SIZE
        accum = tl._experimental_descriptor_load(
            in_desc_ptr,
            [
                tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank,
                tile_id_n * BLOCK_SIZE_N,
            ],
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
            tl.bfloat16,
        )
        for i in range(1, WORLD_SIZE):
            cur_rank = (i + RANK + 1) % WORLD_SIZE
            data = tl._experimental_descriptor_load(
                in_desc_ptr,
                [
                    tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank,
                    tile_id_n * BLOCK_SIZE_N,
                ],
                [BLOCK_SIZE_M, BLOCK_SIZE_N],
                tl.bfloat16,
            )
            accum += data

        tl._experimental_descriptor_store(
            out_desc_ptr, accum, [tile_id_m * BLOCK_SIZE_M, tile_id_n * BLOCK_SIZE_N]
        )


def reduce(
    inp: torch.Tensor,  # scatter_out with shape [M, N]
    output: torch.Tensor,  # [M_per_rank, N]
    configs: dict,
):
    M, N = inp.shape
    M_per_rank = M // configs["WORLD_SIZE"]
    assert output.shape[0] == M_per_rank and M % configs["WORLD_SIZE"] == 0

    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 64

    in_desc_ptr = _create_2d_tma_descriptor(
        inp.data_ptr(),
        M,
        N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        inp.element_size(),
    )
    out_desc_ptr = _create_2d_tma_descriptor(
        output.data_ptr(),
        M_per_rank,
        N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        output.element_size(),
    )

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M_per_rank, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _reduce_persistent_kernel[grid](
        in_desc_ptr,
        out_desc_ptr,
        M_per_rank,
        N,
        RANK=configs["RANK"],
        WORLD_SIZE=configs["WORLD_SIZE"],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4,
    )

    return output


def gemm_reduce_scatter_ce_persistent(
    a: torch.Tensor,
    b: torch.Tensor,
    reduce_op: str = "sum",  # only support sum for now
    group: dist.ProcessGroup | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    M = a.shape[0]
    N = b.shape[1]

    group = dist.group.WORLD if group is None else group
    gemm_out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    symm_mem_hdl = symm_mem.get_symm_mem_workspace(
        group.group_name, min_size=M * N * a.element_size()
    )
    scatter_out = symm_mem_hdl.get_buffer(symm_mem_hdl.rank, [M, N], a.dtype, 0)
    world_size = symm_mem_hdl.world_size

    assert M % world_size == 0
    M_per_rank = M // world_size
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    backend_stream.wait_stream(torch.cuda.current_stream())

    output = torch.empty((M_per_rank, N), dtype=a.dtype, device=a.device)

    configs = {
        "BLOCK_SIZE_M": kwargs.get("block_size_m", 128),
        "BLOCK_SIZE_N": kwargs.get("block_size_n", 256),
        "BLOCK_SIZE_K": kwargs.get("block_size_k", 64),
        "GROUP_SIZE_M": kwargs.get("group_size_m", 8),
        "num_stages": kwargs.get("num_stages", 3),
        "num_warps": kwargs.get("num_warps", 8),
    }
    configs["RANK"] = symm_mem_hdl.rank
    configs["WORLD_SIZE"] = world_size

    progress = torch.zeros(world_size, dtype=torch.uint32, device=a.device)
    signal_pad = torch.zeros(world_size, dtype=torch.uint32, device=a.device)

    gemm_producer_w_progress(a, b, gemm_out, progress, signal_pad, configs)

    with backend_stream:
        copy_engine_scatter(gemm_out, scatter_out, signal_pad, group)

    torch.cuda.current_stream().wait_stream(backend_stream)
    symm_mem_hdl.barrier()

    reduce(scatter_out, output, configs)

    return output

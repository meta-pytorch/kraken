import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
# import triton.tools.experimental_descriptor

from .._ptx_utils import wait_gmem_barrier
from .copy_engine_all_gather import copy_engine_all_gather_w_progress


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    ret["flops8"] = 2.0 * M * N * K
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret["bytes"] = bytes_per_elem * (M * K + N * K)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def _matmul_kernel_tma_persistent_w_progress(
    a_shared_desc_ptr,
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
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
    Slightly modified from the sm90 tma persistent Triton tutorial.
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
    a_ptr = a_desc_ptr

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
                a_ptr = a_shared_desc_ptr
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
                a_ptr = a_desc_ptr

        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = ki * BLOCK_SIZE_K

        a = tl._experimental_descriptor_load(
            a_ptr, [offs_am_src, offs_k], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr, [offs_bn, offs_k], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
        )
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)

            tl._experimental_descriptor_store(
                c_desc_ptr, c, [pid_m * BLOCK_SIZE_M, offs_bn]
            )
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


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

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    desc_a_shared = _create_2d_tma_descriptor(
        a_shared.data_ptr(),
        a_shared.shape[0],
        K,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_K"],
        a_shared.element_size(),
    )

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
        b.element_size(),
    )
    desc_c = _create_2d_tma_descriptor(
        c.data_ptr(),
        M,
        N,
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        c.element_size(),
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

    backend_stream = copy_engine_all_gather_w_progress(
        a_out, a_shared, progress, configs["SPLITS_PER_RANK"]
    )

    c = _matmul_w_progress(a_out, a_shared, b, progress, configs)

    torch.cuda.current_stream().wait_stream(backend_stream)

    return a_out, c

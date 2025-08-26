import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem

import triton  # @manual
import triton.language as tl  # @manual


def check_if_nvshmem_available():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).major >= 8 and symm_mem.is_nvshmem_available()


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


@triton.jit(launch_metadata=_matmul_launch_metadata)
def _gemm_producer_persistent_kernel(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    gemm_out_data_ptr,
    progress_ptr,
    symm_mem_ptrs_ptr,
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

        a = a_desc_ptr.load([offs_am, offs_k])
        b = b_desc_ptr.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            c = accumulator.to(dtype)
            c_desc_ptr.store([offs_am, offs_bn], c)

            remote_rank = offs_am // M_per_rank
            val = tl.atomic_add(
                progress_ptr + remote_rank, 1, sem="release", scope="gpu"
            )
            # tiled_m_size * num_pid_n  = num of tiles in a slice

            if val == M_per_rank / BLOCK_SIZE_M * num_pid_n - 1:
                remote_buffer_ptr_int64 = tl.load(symm_mem_ptrs_ptr + remote_rank)
                # remote_signal_pad_ptr_int64 = tl.load(signal_pad_ptr_tensor + remote_rank)
                # tl.device_print("remote_buffer_ptr_int64: ", remote_buffer_ptr_int64)
                # dest_ptr = remote_buffer_ptr_int64 + RANK * M_per_rank * N * 2
                dest_ptr = remote_buffer_ptr_int64
                #source_ptr = gemm_out_data_ptr + remote_rank * M_per_rank * N * 2
                source_ptr = gemm_out_data_ptr
                tl.device_print("BEFORE NVSHMEM!!!", val)
                tl.device_print("From rank ", RANK)
                tl.device_print("To rank ", remote_rank)
                tl.device_print("To pointer ", remote_buffer_ptr_int64)
                # NVSHMEM_SIGNAL_SET = 0
                # nvshmem.putmem_signal_block(
                #     dest_ptr, source_ptr, 8, remote_signal_pad_ptr_int64, 1, NVSHMEM_SIGNAL_SET, remote_rank)
                nvshmem.putmem_block_extern_wrapper(dest_ptr, source_ptr, 8, remote_rank)
                # tl.device_print("AFTER NVSHMEM!!!")
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # remote_rank = offs_am // M_per_rank
            # remote_buffer_ptr_int64 = tl.load(symm_mem_ptrs_ptr + remote_rank)
            # remote_buffer_ptr = remote_buffer_ptr_int64.to(tl.pointer_type(dtype))
            # block_ptr = tl.make_block_ptr(
            #     base=remote_buffer_ptr,  # int64 base address
            #     shape=(M, N),  # full matrix shape
            #     strides=(N, 1),  # row-major
            #     offsets=(
            #         M_per_rank * RANK - M_per_rank * remote_rank + offs_am,
            #         offs_bn,
            #     ),  # tile start
            #     block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),  # tile size
            #     order=(1, 0),  # row-major
            # )
            # tl.store(block_ptr, c)
            # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


def gemm_producer_w_progress(
    a: torch.Tensor,
    b: torch.Tensor,
    gemm_out: torch.Tensor,
    progress: torch.Tensor,
    symm_mem_ptr_tensor,
    configs: dict,
    group: dist.ProcessGroup | None = None,
    extern_libs: object | None = None,
):
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Inner dimensions must match for matrix multiplication"
    assert a.dtype == b.dtype, "Input dtypes must match"

    bT = b.T

    desc_a = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
        a, [configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_K"]]
    )
    desc_bt = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
        bT,
        [configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"]],
    )
    desc_c = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
        gemm_out,
        [configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"]],
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
        gemm_out.data_ptr(),
        progress,
        symm_mem_ptr_tensor,
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
        extern_libs=extern_libs,
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
        accum = in_desc_ptr.load(
            [tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N]
        )
        for i in range(1, WORLD_SIZE):
            cur_rank = (i + RANK + 1) % WORLD_SIZE
            data = in_desc_ptr.load(
                [
                    tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank,
                    tile_id_n * BLOCK_SIZE_N,
                ]
            )
            accum += data

        out_desc_ptr.store([tile_id_m * BLOCK_SIZE_M, tile_id_n * BLOCK_SIZE_N], accum)


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
    in_desc_ptr = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
        inp, [BLOCK_SIZE_M, BLOCK_SIZE_N]
    )
    out_desc_ptr = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
        output, [BLOCK_SIZE_M, BLOCK_SIZE_N]
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


def triton_fused_matmul_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused GEMM + Reduce-Scatter with overlapped GEMM and Scatter operation.
    Computes C = A @ B on each rank, then performs reduce-scatter to sum results
    and scatter them along the M dimension.
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N)
    Returns:
        Output matrix of shape (M / world_size, N) containing the reduce-scattered result.
    """
    assert (
        a.shape[1] == b.shape[0]
    ), "Inner dimensions must match for matrix multiplication"

    M, N = a.shape[0], b.shape[1]

    # Initialize NVSHMEM device library
    nvshmem_lib = nvshmem.enable_triton()
    assert (symm_mem.is_nvshmem_available()), "NVSHMEM is not available"

    # Use the global process group if no specific group is provided, otherwise use the given group
    group = dist.group.WORLD if group is None else group
    # Get the total number of processes/GPUs in the distributed group
    world_size = dist.get_world_size(group)
    # Get the current process's rank (ID) within the distributed group (0 to world_size-1)
    rank = dist.get_rank(group)

    assert (
        M % world_size == 0
    ), f"M dimension ({M}) must be divisible by world_size ({world_size})"

    # Ensure the matrix can be evenly divided among all processes
    assert M % world_size == 0

    # Create output tensor for the scatter result
    M_per_rank = M // world_size
    output = torch.empty((M_per_rank, N), dtype=a.dtype, device=a.device)

    # configurations for GEMM heurisitcs etc
    configs = {
        "BLOCK_SIZE_M": kwargs.get("block_size_m", 64),
        "BLOCK_SIZE_N": kwargs.get("block_size_n", 256),
        "BLOCK_SIZE_K": kwargs.get("block_size_k", 64),
        "GROUP_SIZE_M": kwargs.get("group_size_m", 8),
        "num_stages": kwargs.get("num_stages", 3),
        "num_warps": kwargs.get("num_warps", 8),
    }
    configs["RANK"] = rank
    configs["WORLD_SIZE"] = world_size

    assert (
        (M / world_size) % configs["BLOCK_SIZE_M"] == 0
    ), f"M_per_rank dimension ({M / world_size}) must be divisible by BLOCK_SIZE_M ({configs["BLOCK_SIZE_M"]})"

    # Create symmetric buffer for GEMM output
    symm_mem_hdl = symm_mem.get_symm_mem_workspace(
        group.group_name, min_size=M * N * a.element_size()
    )
    # Create an array of raw data pointers to the symmetric memory buffers on each rank
    symm_mem_ptrs = []
    scatter_out = None
    for rank in range(world_size):
        buf = symm_mem_hdl.get_buffer(rank, [M, N], a.dtype, 0)
        symm_mem_ptrs.append(buf.data_ptr())
        if rank == symm_mem_hdl.rank:
            scatter_out = buf
    symm_mem_ptr_tensor = torch.tensor(symm_mem_ptrs, dtype=torch.int64, device=a.device)

    # group_name = dist.group.WORLD.group_name
    # scatter_out = None
    # symm_mem.enable_symm_mem_for_group(group_name)
    # symm_mem_tensor = symm_mem.empty(M * N, dtype=a.dtype, device=a.device)
    # symm_mem_hdl = symm_mem.rendezvous(symm_mem_tensor, group=group_name)
    # symm_mem_hdl.barrier()
    # symm_mem_ptr_tensor = torch.empty(world_size, dtype=torch.int64, device=a.device)
    # signal_pad_ptr_tensor = torch.empty(world_size, dtype=torch.int64, device=a.device)
    # for rank in range(world_size):
    #     symm_mem_ptr_tensor[rank] = symm_mem_hdl.buffer_ptrs[rank]
    #     signal_pad_ptr_tensor[rank] = symm_mem_hdl.signal_pad_ptrs[rank]
    #     if rank == symm_mem_hdl.rank:
    #         scatter_out = symm_mem_ptr_tensor[rank]

    # group_name = dist.group.WORLD.group_name
    # symm_mem.enable_symm_mem_for_group(group_name)
    # symm_mem_tensor = symm_mem.empty(M * N, dtype=a.dtype, device=a.device)
    # symm_mem_hdl = symm_mem.rendezvous(symm_mem_tensor, group=group_name)

    # local_ptr = torch.tensor([symm_mem_hdl.buffer_ptrs[symm_mem_hdl.rank]], dtype=torch.int64, device=a.device)
    # local_signal_pad_ptr = torch.tensor([symm_mem_hdl.signal_pad_ptrs[symm_mem_hdl.rank]], dtype=torch.int64, device=a.device)
    # all_ptrs = [torch.empty_like(local_ptr) for _ in range(world_size)]
    # all_signal_pad_ptrs = [torch.empty_like(local_signal_pad_ptr) for _ in range(world_size)]
    # dist.all_gather(all_ptrs, local_ptr)
    # dist.all_gather(all_signal_pad_ptrs, local_signal_pad_ptr)
    # symm_mem_ptr_tensor = torch.cat(all_ptrs)
    # signal_pad_ptr_tensor = torch.cat(all_signal_pad_ptrs)
    # scatter_out = symm_mem_ptr_tensor[symm_mem_hdl.rank]

    gemm_out = symm_mem.empty((M, N), dtype=a.dtype, device=a.device)
    progress = torch.zeros(world_size, dtype=torch.uint32, device=a.device)

    print("symm_mem_ptr_tensor: ", symm_mem_ptr_tensor)
    gemm_producer_w_progress(a, b, gemm_out, progress, symm_mem_ptr_tensor, configs, extern_libs=nvshmem_lib)
    symm_mem_hdl.barrier()

    # Communication is now fused into the GEMM kernel, no separate copy engine needed

    reduce(scatter_out, output, configs)

    return output

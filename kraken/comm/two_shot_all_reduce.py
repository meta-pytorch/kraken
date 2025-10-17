import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .. import _logging as log
from .. import _ptx_utils as ptx_utils


@triton.jit
def two_shot_all_reduce_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    output_ptr,
    numel,
    stride_per_program: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_ptr = tl.multiple_of(input_ptr, 16)
    output_ptr = tl.multiple_of(output_ptr, 16)
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # Copy the input to the symmetric memory buffer.
    # Each PID needs to perform copy for every BLOCK_SIZE * world_size elements.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    block_start = pid * stride_per_program
    while block_start < numel:
        offsets = block_start + tl.arange(0, tl.constexpr(stride_per_program))
        mask = offsets < numel
        val = tl.load(input_ptr + offsets, mask=mask)
        tl.store(buffer_ptr + offsets, val, mask=mask)
        block_start += tl.num_programs(axis=0) * stride_per_program

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Two-shot allreduce
    block_start = pid * stride_per_program
    while block_start < numel:
        offsets = block_start + rank * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            val = tl.load(buffer_ptr + offsets, mask=mask).to(tl.float32)
            acc += val

        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            tl.store(buffer_ptr + offsets, acc, mask=mask)

        block_start += tl.num_programs(axis=0) * stride_per_program

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Copy the result from the symmetric memory buffer to the output.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    block_start = pid * stride_per_program
    while block_start < numel:
        offsets = block_start + tl.arange(0, stride_per_program)
        mask = offsets < numel
        val = tl.load(buffer_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(output_ptr + offsets, val, mask=mask)
        block_start += tl.num_programs(axis=0) * stride_per_program

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )


def two_shot_all_reduce(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Perform a two-shot all-reduce operation using symmetric memory.

    output = all_reduce(input)

    Args:
        tensor (torch.Tensor): The input tensor to be reduced. Must be of dtype
            torch.bfloat16 and 128-bit aligned.
        **kwargs: Additional keyword arguments for kernel configuration:
            max_num_blocks (int, optional): The maximum number of blocks to launch.
            num_warps (int, optional): The number of warps per block.
            BLOCK_SIZE (int, optional): The BLOCK_SIZE parameter for the kernel.

    Returns:
        torch.Tensor: The output tensor containing the reduced result.
    """
    config = {
        "max_num_blocks": kwargs.get("max_num_blocks", 24),
        "num_warps": kwargs.get("num_warps", 32),
        "BLOCK_SIZE": kwargs.get("BLOCK_SIZE", 2048),
    }

    # Create symmetric memory buffer for two-shot operation
    symm_mem_buffer = symm_mem.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=dist.group.WORLD)
    output = torch.empty_like(tensor)

    world_size = symm_mem_hdl.world_size
    assert config["BLOCK_SIZE"] % world_size == 0, (
        "BLOCK_SIZE must be divisible by world_size for two-shot all-reduce"
    )

    num_blocks = min(
        triton.cdiv(tensor.numel(), config["BLOCK_SIZE"] * world_size),
        config["max_num_blocks"]
    )

    kernel = two_shot_all_reduce_kernel[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        tensor,
        output,
        numel=tensor.numel(),
        stride_per_program=config["BLOCK_SIZE"] * world_size,
        rank=symm_mem_hdl.rank,
        world_size=world_size,
        BLOCK_SIZE=config["BLOCK_SIZE"],
        num_warps=config["num_warps"],
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

    return output

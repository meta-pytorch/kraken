"""
This module implements a Triton kernel for two-shot all-reduce with bias addition
Torch symmetric memory interface. According to the benchmark, this
two-shot all-reduce starts to outperform one-shot all-reduce when the input size
is larger than ~150KB and starts to lose to NCCL (without fusion) when the input
size is larger than ~15MB.


NOTE: bias is the same across ranks for this use case as the workload is for inference.
"""

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .. import _logging as log
from .. import _ptx_utils as ptx_utils


@triton.jit
def two_shot_all_reduce_bias_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    output_ptr,
    numel,
    has_bias: tl.constexpr,
    stride_per_program: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_ptr = tl.multiple_of(input_ptr, 16)
    output_ptr = tl.multiple_of(output_ptr, 16)
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    # Note: Triton complains this is not a constexpr, but it is :(
    # stride_per_program = BLOCK_SIZE * world_size

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

        # NOTE: Doing this between two shots is feasible because the bias is the
        # same across all ranks.
        if has_bias:
            bias_ptr = tl.multiple_of(bias_ptr, 16)
            acc += tl.load(bias_ptr + offsets, mask=mask).to(tl.float32)

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

    # Ensure that subsequent kernels do not corrupt the data before this kernel
    # completes loading from the symmetric memory.
    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )


def two_shot_all_reduce_bias(
    symm_mem_input: torch.Tensor,
    input_tensor: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    max_num_blocks: int = 24,
    BLOCK_SIZE: int = 2048,
    group: dist.ProcessGroup | None = None,
):
    """
    Perform a two-shot all-reduce operation with bias addition using symmetric memory.

    output = all_reduce(input)
    output = output + bias if bias is not None else output

    NOTE: bias is the same across ranks for this use case as the workload is for inference.

    Args:
        symm_mem_input (torch.Tensor): The symmetric memory buffer.
        input_tensor (torch.Tensor): The input tensor to be reduced. Must be of dtype
            torch.bfloat16 and 128-bit aligned.
        bias (torch.Tensor | None): The bias tensor to be added to the reduced
            input. If None, no bias is added.
        output (torch.Tensor): The output tensor to store the result.
        max_num_blocks (int, optional): The maximum number of blocks to launch.
        BLOCK_SIZE (int, optional): The BLOCK_SIZE parameter for the kernel.
        group (dist.ProcessGroup | None, optional): The process group to use for
            the all-reduce operation. If None, the default process group will be
            used.

    Returns:
        torch.Tensor: The output tensor containing the reduced result with bias added.
    """

    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_input, group=group)

    world_size = symm_mem_hdl.world_size
    num_blocks = min(
        triton.cdiv(input_tensor.numel(), BLOCK_SIZE * world_size), max_num_blocks
    )
    rank = symm_mem_hdl.rank

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert input_tensor.numel() % 8 == 0, (
        "The number of elements must be 128-bit aligned."
    )
    assert BLOCK_SIZE % world_size == 0

    num_warps = 32

    kernel = two_shot_all_reduce_bias_kernel[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        input_tensor,
        bias,
        output,
        numel=input_tensor.numel(),
        has_bias=bias is not None,
        stride_per_program=BLOCK_SIZE * world_size,
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

    return output

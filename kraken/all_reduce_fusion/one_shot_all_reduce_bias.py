"""
This module demonstrates the usage of fusing one_shot all_reduce with bias addition,
which is used by certain models, using PyTorch symmetric memory interface.

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
def one_shot_all_reduce_bias_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    output_ptr,
    numel,
    has_bias: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One-shot all-reduce operation with optional bias addition on the input.

    Args:
        symm_mem_buffer_ptrs: Pointer to the symmetric memory buffer pointers.
        symm_mem_signal_pad_ptrs: Pointer to the signal pad pointers for synchronization.
        input_ptr: Pointer to the input tensor data.
        bias_ptr: Pointer to the bias tensor data.
        output_ptr: Pointer to the output tensor data.
        numel: The total number of elements in the input tensor to be processed.
        has_bias (tl.constexpr): Flag indicating whether a bias is present.
        rank (tl.constexpr): The rank of the current device in the symm_mem group.
        world_size (tl.constexpr): Total number of devices in the symm_mem group.
        BLOCK_SIZE (tl.constexpr): The size of each block for processing.

    Returns:
        None
    """

    pid = tl.program_id(axis=0)
    input_ptr = tl.multiple_of(input_ptr, 16)
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # Copy the input to the symmetric memory buffer.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    block_start = pid * BLOCK_SIZE
    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        val = tl.load(input_ptr + offsets, mask=mask)
        tl.store(buffer_ptr + offsets, val, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequenceMemAccess=True,
    )

    block_start = pid * BLOCK_SIZE
    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        if has_bias:
            bias_ptr = tl.multiple_of(bias_ptr, 16)
            acc = tl.load(bias_ptr + offsets, mask=mask).to(tl.float32)
        else:
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        # One-shot all-reduce
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            val = tl.load(buffer_ptr + offsets, mask=mask).to(tl.float32)
            acc += val

        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )


def one_shot_all_reduce_bias(
    symm_mem_buffer: torch.Tensor,
    input_tensor: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    max_num_blocks: int = 24,
    BLOCK_SIZE: int = 4096,
    group: dist.ProcessGroup | None = None,
) -> None:
    """
    One-shot all-reduce operation with optional bias addition on the input.

    output = all_reduce(input)
    output = output + bias if bias is not None else output

    NOTE: that bias is assumed to be the same as this use case is for inference.

    This kernel uses a persistent execution style, launching up to 24 blocks,
    with blocks iterating over the input tensor until all elements are
    processed.

    Args:
        symm_mem_buffer (torch.Tensor): The symmetric memory buffer.
        input_tensor (torch.Tensor): The input tensor to be reduced. Must be of dtype
            torch.bfloat16 and 128-bit aligned.
        bias (torch.Tensor | None): The bias tensor to be added to the reduced
            input. If None, no bias is added.
        output (torch.Tensor): The tensor where the result of the all-reduce
            operation is stored.
        group (dist.ProcessGroup | None, optional): The process group to use for
            the all-reduce operation. If None, the default process group will be
            used.
        max_num_blocks (int, optional): The maximum number of blocks to launch.
        BLOCK_SIZE (int, optional): The BLOCK_SIZE parameter for the kernel.

    Returns:
        None
    """

    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group=group)
    if symm_mem_hdl is None:
        raise ValueError("symm_mem_buffer much be a valid symmetric memory tensor.")
    num_blocks = min(triton.cdiv(input_tensor.numel(), BLOCK_SIZE), max_num_blocks)

    assert input_tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert input_tensor.numel() % 8 == 0, (
        "The number of elements must be 128-bit aligned."
    )

    num_warps = 32

    kernel = one_shot_all_reduce_bias_kernel[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        input_tensor,
        bias,
        output,
        numel=input_tensor.numel(),
        has_bias=bias is not None,
        world_size=symm_mem_hdl.world_size,
        rank=symm_mem_hdl.rank,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

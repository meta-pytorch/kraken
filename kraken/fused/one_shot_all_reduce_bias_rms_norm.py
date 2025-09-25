"""
This file demonstrate using PyTorch symmetric memory interface to fuse
one-shot all_reduce with bias addition and RMSNorm for a particular use case.

NOTE: bias and w the same across ranks for this use case as the
workload is for inference.
"""

import math
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from triton.language.math import rsqrt as tl_rsqrt

from .. import _logging as log
from .. import _ptx_utils as ptx_utils


@triton.jit
def one_shot_all_reduce_bias_rms_norm_kernel(
    symm_mem_buffer_ptrs,
    symm_mem_signal_pad_ptrs,
    input_ptr,
    bias_ptr,
    w_ptr,
    y_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    bt_stride: tl.constexpr,
    h_stride: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    bt_idx = row_idx // H
    h_idx = row_idx % H
    variance = tl.zeros([1], dtype=tl.float32)
    col_offsets = tl.arange(0, triton.next_power_of_2(D))
    mask = col_offsets < D

    input_ptr = input_ptr.to(tl.pointer_type(tl.bfloat16))
    input_ptr = tl.multiple_of(input_ptr, 16)
    bias_ptr = bias_ptr.to(tl.pointer_type(tl.bfloat16))
    bias_ptr = tl.multiple_of(bias_ptr, 16)
    y_ptr = y_ptr.to(tl.pointer_type(tl.bfloat16))
    y_ptr = tl.multiple_of(y_ptr, 16)

    offset = bt_idx * bt_stride + h_idx * h_stride
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # Copy the input, x, to the symmetric memory buffer.
    row = tl.load(input_ptr + offset + col_offsets, mask=mask)
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    tl.store(buffer_ptr + offset + col_offsets, row, mask=mask)

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # Allreduce + bias
    row = tl.load(
        bias_ptr + offset + col_offsets,
        mask=mask,
    ).to(tl.float32)
    for i in range(world_size):
        buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        x = tl.load(
            buffer_ptr + offset + col_offsets,
            mask=mask,
        ).to(tl.float32)
        row += x

    # The regular RMSNorm
    variance = tl.sum(row * row, axis=0) / D
    rstd = tl_rsqrt(variance + eps)

    w = tl.load(w_ptr + col_offsets, mask=mask).to(tl.float32)
    tl.store(y_ptr + offset + col_offsets, row * rstd * w, mask=mask)

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )


def one_shot_all_reduce_bias_rms_norm(
    symm_mem_input: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1.0e-5,
    BLOCK_SIZE: int = 1024,
    group: dist.ProcessGroup | None = None,
) -> None:
    """This function performs a one_shot all-reduce, bias addition and RMSNorm.

    dist.all_reduce(x)
    x = x + bias
    y = F.rms_norm(x, x.shape[-1], w, eps)

    Args:
        symm_mem_input (torch.Tensor): The symmetric memory buffer.
        x (torch.Tensor): The input tensor to be reduced.
        bias (torch.Tensor): The bias tensor to be added to the reduced input.
        w (torch.Tensor): The weights tensor for RMS normalization.
        y (torch.Tensor): The output tensor to store the result.
        eps (float, optional): The epsilon value for RMSNorm. Default is 1.0e-5.
        BLOCK_SIZE (int, optional): The BLOCK_SIZE parameter for the kernel.
        group (dist.ProcessGroup, optional): The process group for allreduce.
            Default is None which uses the WORLD process group.

    Returns:
        torch.Tensor: The resulting tensor after all-reduce, bias addition, and
        RMS normalization.
    """
    D = x.shape[-1]

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.stride() == x.stride(), (str(x.stride()), str(y.stride()))
    assert w.is_contiguous()
    assert w.shape == (D,)
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert y.shape == x.shape

    num_blocks = math.prod(x.shape[:-1])
    num_warps = 32
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_input, group=group)

    kernel = one_shot_all_reduce_bias_rms_norm_kernel[(num_blocks,)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        x,
        bias,
        w,
        y,
        eps,
        D=D,
        H=1,
        bt_stride=D,
        h_stride=0,
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

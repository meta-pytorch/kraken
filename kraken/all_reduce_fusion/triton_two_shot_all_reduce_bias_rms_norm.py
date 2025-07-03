"""
This file demonstrates using PyTorch symmetric memory interface to fuse
two-shot all_reduce with bias addition and RMSNorm for a particular use case.

There is no significant performance gain from this fusion, as each block must
handle data in a row-aligned manner, which is not the ideal case for two-shot
all-reduce.

NOTE: bias and w are the same across ranks for this use case as the workload
is for inference.
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
def two_shot_all_reduce_bias_rms_norm_kernel(
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
    size_per_rank: tl.constexpr,
    rows_per_block: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64) * rows_per_block
    bt_idx = row_idx // H
    h_idx = row_idx % H
    col_offsets = tl.arange(0, triton.next_power_of_2(D))

    # Each block has to compute the RMSNorm one row per time, and
    # the row size is D.
    mask = col_offsets < D

    input_ptr = tl.multiple_of(input_ptr, 16)
    bias_ptr = tl.multiple_of(bias_ptr, 16)
    y_ptr = tl.multiple_of(y_ptr, 16)

    offset = bt_idx * bt_stride + h_idx * h_stride
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # Copy the input, x, to the symmetric memory buffer.
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    for i in tl.static_range(rows_per_block):
        row = tl.load(input_ptr + offset + i * D + col_offsets, mask=mask)
        tl.store(buffer_ptr + offset + i * D + col_offsets, row, mask=mask)

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequenceMemAccess=True,
    )

    # Two shot allreduce
    local_rank_offsets = (
        offset
        + size_per_rank * rank
        + tl.arange(0, triton.next_power_of_2(size_per_rank))
    )
    local_rank_mask = local_rank_offsets < (offset + size_per_rank * (rank + 1))

    # Bias addition, this is feasible because the bias is the same across ranks.
    acc = tl.load(bias_ptr + local_rank_offsets, mask=local_rank_mask).to(tl.float32)
    for remote_rank in range(world_size):
        buffer_ptr = tl.load(buffer_ptrs + remote_rank).to(tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        val = tl.load(buffer_ptr + local_rank_offsets, mask=local_rank_mask).to(
            tl.float32
        )
        acc += val

    for remote_rank in range(world_size):
        buffer_ptr = tl.load(buffer_ptrs + remote_rank).to(tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        tl.store(buffer_ptr + local_rank_offsets, acc, mask=local_rank_mask)

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequenceMemAccess=True,
    )

    # The regular RMSNorm
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    for i in tl.static_range(rows_per_block):
        row = tl.load(buffer_ptr + offset + i * D + col_offsets, mask=mask).to(
            tl.float32
        )
        variance = tl.sum(row * row, axis=0) / D
        rstd = tl_rsqrt(variance + eps)

        w = tl.load(w_ptr + col_offsets, mask=mask).to(tl.float32)
        tl.store(y_ptr + offset + i * D + col_offsets, row * rstd * w, mask=mask)

    ptx_utils.symm_mem_sync(
        symm_mem_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )


def triton_two_shot_all_reduce_bias_rms_norm(
    symm_mem_input: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1.0e-5,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Performs two-shot all-reduce, bias addition and RMSNorm in a fused manner.

    This function executes the following operations:
    dist.all_reduce(x)
    x = x + bias
    y = F.rms_norm(x, x.shape[-1], w, eps)

    Args:
        symm_mem_input (torch.Tensor): The symmetric memory buffer for
            communication.
        x (torch.Tensor): The input tensor to be reduced.
        bias (torch.Tensor): The bias tensor to be added to the reduced input.
        w (torch.Tensor): The weights tensor for RMS normalization.
        y (torch.Tensor): The output tensor to store the result.
        eps (float, optional): The epsilon value for RMSNorm. Defaults to
            1.0e-5.
        group (dist.ProcessGroup, optional): The process group for allreduce.
            Defaults to None which uses the WORLD process group.

    Returns:
        None: The result is stored in the output tensor y.
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

    total_rows = math.prod(x.shape[:-1])

    # We only support certain total_rows to just demonstrate the idea.
    BLOCK_SIZE = 1024
    if total_rows < 2:
        rows_per_block = 1
    elif total_rows <= 32:
        rows_per_block = 2
    elif total_rows <= 64:
        rows_per_block = 4
    else:
        rows_per_block = 8

    num_blocks = total_rows // rows_per_block

    num_warps = 32
    group = group or dist.group.WORLD
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_input, group=group)
    world_size = symm_mem_hdl.world_size
    rank = symm_mem_hdl.rank

    size_per_rank = rows_per_block * D // world_size

    # assert total_rows % rows_per_block == 0
    assert size_per_rank * world_size == rows_per_block * D
    assert size_per_rank % 16 == 0

    kernel = two_shot_all_reduce_bias_rms_norm_kernel[(num_blocks,)](
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
        size_per_rank=size_per_rank,
        rows_per_block=rows_per_block,
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    dump_kernel = os.environ.get("SYMM_DUMP_KERNEL", "0") == "1"
    if dump_kernel and torch.distributed.get_rank() == 0:
        log.log_triton_kernel(kernel)

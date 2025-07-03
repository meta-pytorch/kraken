import math

import torch
import triton
import triton.language as tl
from triton.language.math import rsqrt as tl_rsqrt


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    eps: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    bt_stride: tl.constexpr,
    h_stride: tl.constexpr,
    H: tl.constexpr,
):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    bt_idx = row_idx // H
    h_idx = row_idx % H
    variance = tl.zeros([1], dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    offset = bt_idx * bt_stride + h_idx * h_stride
    row_start_ptr = x_ptr + offset
    out_start_ptr = y_ptr + offset
    input_ptrs = row_start_ptr + col_offsets
    output_ptrs = out_start_ptr + col_offsets

    row = tl.load(input_ptrs, mask=mask, other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )
    variance = tl.sum(row * row, axis=0) / D
    rstd = tl_rsqrt(variance + eps)

    w = tl.load(w_ptr + col_offsets, mask=mask, eviction_policy="evict_first").to(
        tl.float32
    )
    tl.store(output_ptrs, row * rstd * w, mask=mask)


def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    y = torch.empty_like(x)
    D = x.shape[-1]
    assert w.is_contiguous()
    assert w.shape == (D,)
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert y.shape == x.shape

    num_blocks = math.prod(x.shape[:-1])
    _rms_norm_kernel[(num_blocks,)](
        x,
        y,
        w,
        eps,
        D,
        BLOCK_SIZE=triton.next_power_of_2(D),
        bt_stride=D,
        h_stride=0,
        H=1,
        num_warps=32,
    )
    return y

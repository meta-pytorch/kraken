"""
This module implements a Triton kernel for one-shot all-reduce.
This kernel performs an all-reduce operation on a Torch symmetric memory tensor distributed across
multiple devices. According to benchmark results, one-shot all reduce outperforms NCCL ring reduce
for small message sizes (<~400KB on a 8xH100 system with NVSwitch).
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .. import _ptx_utils as ptx_utils


@triton.jit
def one_shot_all_reduce_kernel(
    buf_tuple,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Synchronize blocks with matching block_id across all participating devices before starting.
    # This ensures that all previous memory operations are visible.
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True
    )

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.bfloat16)

        # Iteratively load from each rank's buffer and accumulate. `static_range` unrolls the loop at compile time, enabling efficient iteration over `buf_tuple`.
        for i in tl.static_range(world_size):
            buffer_rank = buf_tuple[i]
            x = tl.load(buffer_rank + offsets, mask=mask)
            acc += x
        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    # Synchronize all participating devices after the reduction is complete.
    # Subsequent kernel cannot overwrite the symmetric memory buffer until all devices reach this point.
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )


def one_shot_all_reduce(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Perform a one-shot all-reduce operation using symmetric memory.

    output = all_reduce(input)

    Args:
        tensor (torch.Tensor): The input tensor to be reduced. Must be of dtype torch.bfloat16 and 128-bit aligned.
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
        "BLOCK_SIZE": kwargs.get("BLOCK_SIZE", 8192),
    }

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert tensor.numel() % 8 == 0, "The number of elements must be 128-bit aligned."
    assert config["BLOCK_SIZE"] % (config["num_warps"] * 32) == 0, (
        "BLOCK_SIZE must be a multiple of num_warps * 32"
    )

    num_blocks = min(
        triton.cdiv(tensor.numel(), config["BLOCK_SIZE"]), config["max_num_blocks"]
    )

    symm_mem_hdl = symm_mem.rendezvous(tensor, group=dist.group.WORLD)
    output = torch.empty_like(tensor)

    # Get the buffer pointers for each rank from the symmetric memory handle, and pass them as a tuple to the triton kernel.
    buf_list = [
        symm_mem_hdl.get_buffer(i, tuple(tensor.shape), tensor.dtype)
        for i in range(symm_mem_hdl.world_size)
    ]
    buf_tuple = tuple(buf_list)

    # symm_mem_hdl.signal_pad_ptrs_dev: An array of pointers pointing to signal_pads for each rank.
    # A signal pad is a memory region used for synchronization between devices.
    # `symm_mem_sync` kernel uses these signal pads to implement a cross-device barrier to ensure memory visibility of symmetric memory tensors.
    one_shot_all_reduce_kernel[(num_blocks, 1, 1)](
        buf_tuple,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        numel=tensor.numel(),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=config["BLOCK_SIZE"],
        num_warps=config["num_warps"],
    )

    return output

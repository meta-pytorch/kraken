import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def _copy_engine_all_gather_w_progress(
    output: torch.Tensor,
    inp: torch.Tensor,  # Must be symmetric tensor
    progress: torch.Tensor,
    splits_per_rank: int,
    backend_stream: torch.cuda.Stream | None = None,
) -> torch.cuda.Stream:
    backend_stream = symm_mem._get_backend_stream(priority=-1)
    assert inp.is_contiguous()

    symm_mem_hdl = symm_mem.rendezvous(inp, group=dist.group.WORLD)
    assert symm_mem_hdl is not None

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size

    assert inp.numel() % splits_per_rank == 0
    assert progress.numel() >= world_size * splits_per_rank

    output_shape = list(inp.shape)
    output_shape[0] *= world_size
    assert list(output.shape) == output_shape, (list(output.shape), output_shape)

    # Split the output tensor into chunks for each rank and split.
    chunks = output.chunk(world_size * splits_per_rank)

    # Synchronize all ranks before starting the copy operations.
    # This ensures any previous operations on the symmetric memory tensor are completed.
    symm_mem_hdl.barrier()
    backend_stream.wait_stream(torch.cuda.current_stream())

    # Perform the all-gather operation on the backend stream.
    with torch.cuda.stream(backend_stream):
        # Iterate through source rank and splits of the source rank.
        for step in range(world_size):
            src_rank = (rank + step + 1) % world_size
            for split_id in range(splits_per_rank):
                src_buf = symm_mem_hdl.get_buffer(
                    src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
                )
                # Copy data from the source buffer to the corresponding output chunk using copy engine.
                chunks[src_rank * splits_per_rank + split_id].copy_(src_buf)
                # Signal the completion of the copy for this chunk in progress tensor.
                # cuStreamWriteValue32 issues a system level fence before the write
                symm_mem_hdl.stream_write_value32(
                    progress,
                    offset=src_rank * splits_per_rank + split_id,
                    val=1,
                )

        # Synchronize all ranks after all copy operations are issued.
        # This ensures all copy operations are completed before proceeding.
        symm_mem_hdl.barrier()

    return backend_stream


def all_gather_w_progress(
    a_shared: torch.Tensor,
    a_out: torch.Tensor | None = None,
    progress: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Performs an all-gather operation using the copy engine and tracks progress.

    This function gathers data from all ranks into a single output tensor. It uses
    the copy engine for the data transfer and a progress tensor to signal the
    completion of each chunk copy in the progress tensor.
    The operation is performed on a backend CUDA stream.

    Args:
        a_shared (torch.Tensor): The input tensor, which must be a symmetric tensor.
            Each rank provides its shard of the data in this tensor.
        a_out (torch.Tensor, optional): The output tensor to store the gathered data.
        progress (torch.Tensor, optional): A tensor to track the progress of the copy
            operations. Its size should be at least `world_size * splits_per_rank`.
            Initially, all elements should be zero. After a chunk is copied,
            the corresponding element is set to 1.
        splits_per_rank (int): The number of splits (chunks) per rank.
        backend_stream (torch.cuda.Stream, optional): A background CUDA stream for
            the copy engine operations. If not provided, a new stream is created.

    Returns:
        torch.Tensor: The output tensor containing the gathered data from all ranks.
    """
    configs = {
        "SPLITS_PER_RANK": kwargs.get("splits_per_rank", 1),
    }

    symm_mem_hdl = symm_mem.rendezvous(a_shared, group=dist.group.WORLD)

    a_shape = list(a_shared.shape)
    a_shape[0] *= symm_mem_hdl.world_size

    configs["RANK"] = symm_mem_hdl.rank
    configs["WORLD_SIZE"] = symm_mem_hdl.world_size

    configs["COMM_BLOCK_SIZE_M"] = (
        a_shape[0] // configs["WORLD_SIZE"] // configs["SPLITS_PER_RANK"]
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

    backend_stream = _copy_engine_all_gather_w_progress(
        a_out, a_shared, progress, configs["SPLITS_PER_RANK"]
    )

    torch.cuda.current_stream().wait_stream(backend_stream)

    return a_out

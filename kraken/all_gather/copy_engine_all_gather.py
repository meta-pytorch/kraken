import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def copy_engine_all_gather_w_progress(
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

    chunks = output.chunk(world_size * splits_per_rank)

    symm_mem_hdl.barrier()
    backend_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(backend_stream):
        for step in range(0, world_size):
            src_rank = (rank + step + 1) % world_size
            for split_id in range(splits_per_rank):
                src_buf = symm_mem_hdl.get_buffer(
                    src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
                )
                chunks[src_rank * splits_per_rank + split_id].copy_(src_buf)
                # cuStreamWriteValue32 issues a system level fence before the write
                symm_mem_hdl.stream_write_value32(
                    progress,
                    offset=src_rank * splits_per_rank + split_id,
                    val=1,
                )
        symm_mem_hdl.barrier()

    return backend_stream

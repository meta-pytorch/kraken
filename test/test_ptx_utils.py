import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import triton
import triton.language as tl

from kraken._ptx_utils import symm_mem_sync, wait_gmem_barrier, arrive_gmem_barrier


@instantiate_parametrized_tests
class PTXSymmMemBarrier(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        # world_size > 2 is needed to verify accumulation order
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    @triton.jit
    def barrier_test_kernel(
        signal_pad_ptrs,
        rank: tl.constexpr,
        world_size: tl.constexpr,
    ):
        symm_mem_sync(signal_pad_ptrs, None, rank, world_size)

    @skip_if_lt_x_gpu(4)
    def test_symm_mem_barrier(self):
        self._init_process()
        t = symm_mem.empty(4096, device=self.device)
        symm_mem_hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)

        self.barrier_test_kernel[(32, 1, 1)](
            symm_mem_hdl.signal_pad_ptrs_dev,
            rank=symm_mem_hdl.rank,
            world_size=symm_mem_hdl.world_size,
        )

        signal_pad = symm_mem_hdl.get_signal_pad(symm_mem_hdl.rank)
        assert signal_pad.eq(0).all().item()

        dist.destroy_process_group()


class PTXGmemBarrier(TestCase):
    @triton.jit
    def gmem_barrier_test_kernel(
        signal_pad_ptrs,
        NUM_PRODUCERS: tl.constexpr,
        NUM_CONSUMERS: tl.constexpr,
    ):
        bid = tl.program_id(0)
        if bid < NUM_PRODUCERS:
            arrive_gmem_barrier(
                signal_pad_ptrs + bid,
                update=1,
                sem="release",
                scope="gpu",
                op="atomic_xchg",
            )
            pass
        else:
            consumer_id = bid - NUM_PRODUCERS
            producer_id = consumer_id // (NUM_CONSUMERS // NUM_PRODUCERS)
            wait_gmem_barrier(
                signal_pad_ptrs + producer_id,
                expect=1,
                sem="acquire",
                scope="gpu",
                op="ld",
            )

    def test_gmem_barrier_arrive_wait(self):
        signal_pad = torch.zeros(64, device="cuda", dtype=torch.int32).contiguous()
        num_producers = 8
        num_consumers = 32

        self.gmem_barrier_test_kernel[(num_producers + num_consumers, 1, 1)](
            signal_pad,
            NUM_PRODUCERS=num_producers,
            NUM_CONSUMERS=num_consumers,
        )

        expect = torch.zeros_like(signal_pad)
        expect[:num_producers] = 1

        torch.testing.assert_close(signal_pad, expect)


if __name__ == "__main__":
    run_tests()

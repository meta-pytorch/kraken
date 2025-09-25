import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import kraken


@instantiate_parametrized_tests
class TritonAllGatherTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        # world_size > 2 is needed to verify accumulation order
        return torch.cuda.device_count()

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

    @skip_if_lt_x_gpu(4)
    def test_all_gather_w_progress(self):
        self._init_process()
        group_name = dist.group.WORLD.group_name
        a_shared = symm_mem.empty(
            (1024, 1024),
            dtype=torch.bfloat16,
            device=self.device,
        ).normal_()
        symm_mem_hdl = symm_mem.rendezvous(a_shared, group_name)

        progress = torch.zeros(
            symm_mem_hdl.world_size,
            dtype=torch.uint32,
            device=self.device,
        )

        golden_a = a_shared.clone()
        a_gathered = fc.all_gather_tensor(golden_a, 0, "0")

        a_out = kraken.comm.all_gather_w_progress(a_shared, progress=progress)

        torch.testing.assert_close(a_out, a_gathered)
        assert torch.all(progress != 0)

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()

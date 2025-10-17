import os
import sys

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
)

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import kraken


@instantiate_parametrized_tests
class TritonAllReduceTest(MultiProcessTestCase):
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

    @skip_if_lt_x_gpu(4)
    def test_one_shot(self):
        self._init_process()
        group_name = dist.group.WORLD.group_name
        input_tensor = symm_mem.empty(
            (1024, 1024),
            dtype=torch.bfloat16,
            device=self.device,
        )
        input_tensor = input_tensor.normal_()
        symm_mem.rendezvous(input_tensor, group_name)

        result = kraken.comm.one_shot_all_reduce(input_tensor)

        golden = input_tensor.clone()
        dist.all_reduce(golden)

        torch.testing.assert_close(result, golden, rtol=1e-1, atol=1e-1)

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_two_shot(self):
        self._init_process()
        group_name = dist.group.WORLD.group_name
        input_tensor = symm_mem.empty(
            (1024, 1024),
            dtype=torch.bfloat16,
            device=self.device,
        )
        input_tensor = input_tensor.normal_()
        symm_mem.rendezvous(input_tensor, group_name)

        result = kraken.comm.two_shot_all_reduce(input_tensor)

        golden = input_tensor.clone()
        dist.all_reduce(golden)

        torch.testing.assert_close(result, golden, rtol=1e-1, atol=1e-1)

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()

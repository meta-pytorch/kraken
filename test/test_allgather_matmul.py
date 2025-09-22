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
class TritonAllGatherMatmulTest(MultiProcessTestCase):
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
    def test_all_gather_matmul(self):
        self._init_process()
        M = 4096
        N = 6656
        K = 16384

        group_name = dist.group.WORLD.group_name
        a_shared = symm_mem.empty(
            (M // self.world_size, K),
            dtype=torch.bfloat16,
            device=self.device,
        ).normal_()
        symm_mem.rendezvous(a_shared, group_name)
        bT = torch.randn(
            (K, N), device=self.device, dtype=torch.bfloat16
        ).T.contiguous()
        b = bT.T

        ag, c = kraken.all_gather_fusion.all_gather_matmul(a_shared, b)

        golden_a = a_shared.clone()
        ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_matmul(
            golden_a, [b], gather_dim=0, group_name=group_name
        )

        torch.testing.assert_close(c, mm_golden[0], rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(ag, ag_golden)

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()

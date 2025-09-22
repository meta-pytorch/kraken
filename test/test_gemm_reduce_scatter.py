from datetime import timedelta
import os
import sys

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

# Adjust the path to import the kernel from the 'kraken' project directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kraken.reduce_scatter_fusion import (
    gemm_reduce_scatter,
    gemm_reduce_scatter_ce_persistent,
)


@instantiate_parametrized_tests
class TritonGemmReduceScatterTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        # Use at least 4 GPUs to properly test collective ops
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
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=10), dist.group.WORLD
        )
        torch.manual_seed(42 + self.rank)

    def _get_expected_result(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        group_name = dist.group.WORLD.group_name

        # use torch symm mem's fused_matmul_reduce_scatter impl for testing
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            a, b, "sum", scatter_dim=0, group_name=group_name
        )

    @skip_if_lt_x_gpu(4)
    def test_gemm_reduce_scatter(self):
        self._init_process()
        M, N, K = 512, 256, 128
        a = torch.randn((M, K), dtype=torch.float32, device=self.device)
        b = torch.randn((K, N), dtype=torch.float32, device=self.device)

        result = gemm_reduce_scatter(a, b)
        expected = self._get_expected_result(a, b)

        # Compare results
        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_gemm_reduce_scatter_rank_specific(self):
        """Test with rank-specific values to verify correct reduction and scattering."""
        self._init_process()
        M, N, K = 512, 256, 128

        # Each rank contributes `rank + 1` to each element of its `a` matrix.
        rank_multiplier = self.rank + 1
        a = torch.full((M, K), rank_multiplier, dtype=torch.float32, device=self.device)
        b = torch.ones((K, N), dtype=torch.float32, device=self.device)

        result = gemm_reduce_scatter(a, b)
        expected = self._get_expected_result(a, b)

        # The values should match here
        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_gemm_reduce_scatter_ce_persistent(self):
        self._init_process()
        M, N, K = 8192, 4096, 14336
        a = torch.randn((M, K), dtype=torch.bfloat16, device=self.device)
        b = torch.randn((N, K), dtype=torch.bfloat16, device=self.device).t()

        result = gemm_reduce_scatter_ce_persistent(a, b)

        gemm_out = torch.matmul(a, b)
        expected = torch.empty(
            (M // self.world_size, N), device="cuda", dtype=torch.bfloat16
        )
        torch.distributed.reduce_scatter_tensor(
            expected, gemm_out, group=dist.group.WORLD
        )

        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()

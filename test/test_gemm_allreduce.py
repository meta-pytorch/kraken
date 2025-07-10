import os
import sys
from datetime import timedelta

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
class TritonGemmAllReduceTest(MultiProcessTestCase):
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
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=10), dist.group.WORLD
        )
        torch.manual_seed(42 + self.rank)

    @skip_if_lt_x_gpu(4)
    def test_gemm_all_reduce(self):
        self._init_process()
        M, N, K = 512, 256, 128

        # create a and b local tensors
        a = torch.empty((M, K), dtype=torch.float32, device=self.device).normal_()
        b = torch.empty((K, N), dtype=torch.float32, device=self.device).normal_()

        # calculate result for our fused kernel
        result = kraken.all_reduce_fusion.gemm_one_shot_all_reduce_fused(a, b)

        # expected value
        expected = torch.matmul(a, b)
        dist.all_reduce(expected)

        # compare result and expected
        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_gemm_all_reduce_square(self):
        self._init_process()
        M, N, K = 256, 256, 256

        # create a and b local tensors
        a = torch.empty((M, K), dtype=torch.float32, device=self.device).normal_()
        b = torch.empty((K, N), dtype=torch.float32, device=self.device).normal_()

        # calculate result for our fused kernel
        result = kraken.all_reduce_fusion.gemm_one_shot_all_reduce_fused(a, b)

        # expected value
        expected = torch.matmul(a, b)
        dist.all_reduce(expected)

        # compare result and expected
        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_rank_specific_values_all_reduce(self):
        """Test with rank-specific values to verify all-reduce accumulation"""
        self._init_process()
        M, N, K = 32, 32, 32

        # Each rank contributes (rank + 1) to the final sum
        # This makes it easy to verify all-reduce worked correctly
        rank_multiplier = self.rank + 1
        a = torch.ones((M, K), dtype=torch.float32, device=self.device) * rank_multiplier
        b = torch.ones((K, N), dtype=torch.float32, device=self.device)

        result = kraken.all_reduce_fusion.gemm_one_shot_all_reduce_fused(a, b)

        # Expected: sum of all rank contributions
        # rank 0: 1*K, rank 1: 2*K, rank 2: 3*K, rank 3: 4*K
        # Total = K * (1+2+3+4) = K * 10
        expected_sum = K * sum(range(1, self.world_size + 1))  # K * 10 = K * 10
        expected = torch.full((M, N), expected_sum, dtype=torch.float32, device=self.device)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        dist.destroy_process_group()

if __name__ == "__main__":
    run_tests()

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

from kraken.all_reduce_fusion import (
    rms_norm,
    one_shot_all_reduce_bias_rms_norm,
    two_shot_all_reduce_bias_rms_norm,
)


@instantiate_parametrized_tests
class TritonAllReduceBiasRMSNormTest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
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
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=10), dist.group.WORLD
        )
        torch.manual_seed(42 + self.rank)

    def _nccl_all_reduce_bias_rms_norm(
        self, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor
    ) -> None:
        dist.all_reduce(x)
        return rms_norm(x + bias, w)

    @skip_if_lt_x_gpu(4)
    def test_one_shot_bias_rms_norm(self):
        self._init_process()

        symm_mem_buffer = symm_mem.empty(
            (1024, 1024),
            dtype=torch.bfloat16,
            device=self.device,
        )
        symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD)

        for b in [1, 2, 4, 8, 16, 32, 64]:
            torch.manual_seed(42 + self.rank + b)
            input_tensor = torch.randn(
                b, 5120, device=self.device, dtype=torch.bfloat16
            )
            # ensure the bias to be the same acorss rank
            torch.manual_seed(42 + b)
            bias = torch.randn(b, 5120, device=self.device, dtype=torch.bfloat16)
            w = torch.randn(5120, device=self.device, dtype=torch.bfloat16)
            y = torch.empty_like(input_tensor)
            one_shot_all_reduce_bias_rms_norm(
                symm_mem_buffer, input_tensor, bias, w, y
            )
            baseline = self._nccl_all_reduce_bias_rms_norm(
                input_tensor.clone(), w.clone(), bias.clone()
            )

            torch.testing.assert_close(y, baseline, rtol=4e-2, atol=4e-2)
            dist.barrier()

        dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    def test_two_shot_bias_rms_norm(self):
        self._init_process()

        symm_mem_buffer = symm_mem.empty(
            (1024, 1024),
            dtype=torch.bfloat16,
            device=self.device,
        )
        symm_mem.rendezvous(symm_mem_buffer, dist.group.WORLD)

        for b in [1, 2, 4, 8, 16, 32, 64]:
            torch.manual_seed(42 + self.rank + b)
            input_tensor = torch.randn(
                b, 5120, device=self.device, dtype=torch.bfloat16
            )
            # ensure the bias to be the same acorss rank
            torch.manual_seed(42 + b)
            bias = torch.randn(b, 5120, device=self.device, dtype=torch.bfloat16)
            w = torch.randn(5120, device=self.device, dtype=torch.bfloat16)
            y = torch.empty_like(input_tensor)
            two_shot_all_reduce_bias_rms_norm(
                symm_mem_buffer, input_tensor, bias, w, y
            )
            baseline = self._nccl_all_reduce_bias_rms_norm(
                input_tensor.clone(), w.clone(), bias.clone()
            )

            torch.testing.assert_close(y, baseline, rtol=4e-2, atol=4e-2)
            dist.barrier()

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()

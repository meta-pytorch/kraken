import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import functools
import itertools
import os
import sys

from tabulate import tabulate
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Add the kraken directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import kraken
from kraken._logging import benchmark_with_event


def torch_symm_mem_ag_mm(a_shared, b):
    a_gathered, c = torch.ops.symm_mem.fused_all_gather_matmul(
        a_shared, [b], gather_dim=0, group_name=dist.group.WORLD.group_name
    )
    return a_gathered, c[0]


def nccl_mem_ag_mm(a_shared, b):
    from torch.distributed._functional_collectives import all_gather_tensor

    a_gathered = all_gather_tensor(a_shared, 0, "0")
    return a_gathered, torch.matmul(a_gathered, b)


@dataclass(frozen=True)
class ExperimentConfig:
    shape: tuple[int, int, int]
    dtype: torch.dtype
    backends: list[str]
    baseline_backend: str
    device: torch.device

    def asdict(self):
        # Convert the dataclass instance to a dictionary
        d = asdict(self)
        d.pop("backends", None)
        d.pop("device", None)
        d.pop("baseline_backend", None)
        return d


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: dict[str, float]  # backend -> time in us

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = self.results
        return {**dict1, **dict2}


def generate_experiment_configs(
    dtype: torch.dtype,
    M: list[int],
    N: list[int],
    K: list[int],
    backends: list[str],
    device: torch.device,
) -> list[ExperimentConfig]:
    # Generate cross config shapes from M, N, K lists
    shapes = list(itertools.product(M, N, K))

    all_configs = []
    for shape in shapes:
        all_configs.append(
            ExperimentConfig(
                shape=shape,
                dtype=dtype,
                backends=backends,
                baseline_backend=backends[0],
                device=device,
            )
        )

    return all_configs


def get_single_backend_fn(backend: str):
    if backend == "nccl":
        return nccl_mem_ag_mm
    if backend == "torch_symm_mem":
        return torch_symm_mem_ag_mm
    if backend == "triton":
        return kraken.all_gather.triton_all_gather_matmul
    raise NotImplementedError(backend)


def clone_symm_mem_tensor(tensor: torch.Tensor) -> torch.Tensor:
    symm_mem_tensor = symm_mem.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    symm_mem.rendezvous(symm_mem_tensor, dist.group.WORLD.group_name)
    symm_mem_tensor.copy_(tensor)
    return symm_mem_tensor


def run_experiment(config: ExperimentConfig) -> dict[str, float]:
    M, N, K = config.shape
    a = symm_mem.empty(
        (M, K),
        dtype=config.dtype,
        device=config.device,
    ).normal_()
    b = torch.randn((K, N), device=config.device, dtype=config.dtype).T.contiguous().T
    symm_mem.rendezvous(a, dist.group.WORLD.group_name)

    input_tensors = {backend: clone_symm_mem_tensor(a) for backend in config.backends}
    gloden_inp = clone_symm_mem_tensor(a)

    gloden_o = get_single_backend_fn(config.baseline_backend)(gloden_inp, b)

    results = {}
    for backend in config.backends:
        fn = get_single_backend_fn(backend)
        inp = input_tensors[backend]

        test_o = fn(inp, b)
        torch.testing.assert_close(test_o[1], gloden_o[1], atol=1e-1, rtol=1e-1)

        target_fn = functools.partial(fn, inp, b)
        results[backend] = benchmark_with_event(target_fn, flush_l2=True)

    return results


def print_results(results: list[Experiment], save_path: str | None = None):
    table_data = defaultdict(list)

    for experiment in results:
        baseline_time = experiment.results[experiment.config.baseline_backend]
        min_time = float("inf")
        best_backend = experiment.config.baseline_backend
        backends = experiment.config.backends
        for key, value in experiment.asdict().items():
            if key in backends:
                if value < min_time:
                    min_time = value
                    best_backend = key
                table_data[key].append(value)
            else:
                table_data[key].append(value)
        table_data[f"Speedup over {experiment.config.baseline_backend}"].append(
            baseline_time / min_time
        )
        table_data["Best Backend"].append(best_backend)
    print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

    if save_path is not None:
        with open(save_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=table_data.keys())
            writer.writeheader()
            for i in range(len(next(iter(table_data.values())))):
                row = {k: v[i] for k, v in table_data.items()}
                writer.writerow(row)
        print(f"\nResults saved to {save_path}")


def main(args):
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    torch.manual_seed(42 + local_rank)

    results = []
    configs = generate_experiment_configs(
        args.dtype, args.M, args.N, args.K, args.backend, device
    )
    for config in configs:
        results.append(
            Experiment(
                config,
                run_experiment(config),
            )
        )
    if dist.get_rank() == 0:
        print_results(results, args.save_path)
    dist.destroy_process_group()


def shape_input_type(s):
    try:
        M, N, K = map(int, s.split(","))
        return M, N, K
    except Exception as e:
        raise argparse.ArgumentTypeError("Heads must be Hq,Hkv") from e


if __name__ == "__main__":
    help_str = """
Run with torchrun
torchrun \
--nnodes 1 --nproc-per-node 8 \
--rdzv-backend c10d --rdzv-endpoint localhost:0 \
--no_python python3 \
benchmark/benchmark_all_gather_matmul.py
"""

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run sweep over sizes for Allreduce. " + help_str
    )

    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        choices=[
            "nccl",
            "torch_symm_mem",
            "triton",
        ],
        default=["nccl", "torch_symm_mem", "triton"],
        help="Backend to use for AllGather Matmul. Use first backend as baseline. ",
    )

    parser.add_argument(
        "-M",
        type=shape_input_type,
        nargs="+",
        default=[2**x for x in range(7, 11)],
        help="matmul shapes: (M, N, K). (M, K) @ (K, N) -> (M, N)",
    )

    parser.add_argument(
        "-N",
        type=shape_input_type,
        nargs="+",
        default=[6656],
        help="matmul shapes: (M, N, K). (M, K) @ (K, N) -> (M, N)",
    )

    parser.add_argument(
        "-K",
        type=shape_input_type,
        nargs="+",
        default=[2**x for x in range(12, 15)],
        help="matmul shapes: (M, N, K). (M, K) @ (K, N) -> (M, N)",
    )

    parser.add_argument("-dtype", type=str, help="dtype", default="bfloat16")
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save the results JSON file (optional)",
        default=None,
    )

    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    if "LOCAL_RANK" not in os.environ:
        print(
            "Error: LOCAL_RANK environment variable is not defined. Are you running with torchrun? "
        )
        print(help_str)
        sys.exit(1)

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except ValueError:
        print(
            "Error: LOCAL_RANK environment variable must be a valid integer. Are you running with torchrun? "
        )
        print(help_str)
        sys.exit(1)
    main(args)

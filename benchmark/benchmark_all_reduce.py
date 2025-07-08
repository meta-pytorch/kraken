import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import functools
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


def symm_mem_multimem_all_reduce(msg):
    return torch.ops.symm_mem.multimem_all_reduce_(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def symm_mem_one_shot_all_reduce(msg):
    return torch.ops.symm_mem.one_shot_all_reduce(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def symm_mem_two_shot_all_reduce(msg):
    return torch.ops.symm_mem.two_shot_all_reduce_(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def nccl_ring(msg):
    dist.all_reduce(msg)
    return msg


def formatt_large_number(num: int) -> str:
    if num >= 2**30:
        return f"{num / 2**30:.0f}g"
    if num >= 2**20:
        return f"{num / 2**20:.0f}m"
    if num >= 2**10:
        return f"{num / 2**10:.0f}k"
    return str(num)


@dataclass(frozen=True)
class ExperimentConfig:
    shape: tuple[int]
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

        formated_size = [formatt_large_number(num) for num in self.shape]
        d["shape"] = f"({', '.join(formated_size)})"
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
    dtype: torch.dtype, sizes: list[int], backends: list[str], device: torch.device
) -> list[ExperimentConfig]:
    all_configs = []
    for sz in sizes:
        all_configs.append(
            ExperimentConfig(
                shape=(sz,),
                dtype=dtype,
                backends=backends,
                baseline_backend=backends[0],
                device=device,
            )
        )

    return all_configs


def get_single_backend_fn(backend: str):
    if backend == "dist_multimem":
        return symm_mem_multimem_all_reduce
    if backend == "dist_1shot":
        return symm_mem_one_shot_all_reduce
    if backend == "dist_2shot":
        return symm_mem_two_shot_all_reduce
    if backend == "triton_1shot":
        return kraken.all_reduce.one_shot_all_reduce
    if backend == "nccl":
        return nccl_ring
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
    input_tensor = symm_mem.empty(
        config.shape,
        dtype=config.dtype,
        device=config.device,
    )
    symm_mem.rendezvous(input_tensor, dist.group.WORLD.group_name)
    input_tensor = input_tensor.normal_()
    input_tensors = {
        backend: clone_symm_mem_tensor(input_tensor) for backend in config.backends
    }
    gloden_inp = clone_symm_mem_tensor(input_tensor)

    gloden_o = get_single_backend_fn(config.baseline_backend)(gloden_inp)

    results = {}
    for backend in config.backends:
        fn = get_single_backend_fn(backend)
        inp = input_tensors[backend]
        target_fn = functools.partial(fn, inp)
        test_o = target_fn()
        torch.testing.assert_close(test_o, gloden_o, atol=1e-1, rtol=1e-1)

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
    configs = generate_experiment_configs(args.dtype, args.size, args.backend, device)
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


if __name__ == "__main__":
    help_str = """
Run with torchrun
torchrun \
--nnodes 1 --nproc-per-node 8 \
--rdzv-backend c10d --rdzv-endpoint localhost:0 \
--no_python python3 \
benchmark/benchmark_all_reduce.py
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
            "1shot",
            "dist_multimem",
            "dist_1shot",
            "dist_2shot",
        ],
        default=["nccl", "1shot", "dist_multimem"],
        help="Backend to use for AllReduce. Use first backend as baseline. ",
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs="+",
        default=[2**exp for exp in range(12, 21)],
        help="Tensor lengths",
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

import functools
import gc
import os

import click
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from kraken import _logging as log
from kraken.all_reduce_fusion import (
    triton_one_shot_all_reduce_bias as one_shot_all_reduce_bias,
)


def triton_one_shot_all_reduce_bias(
    x: torch.Tensor, bias: torch.Tensor, symm_mem_input: torch.Tensor
) -> torch.Tensor:
    y = torch.empty_like(x)
    one_shot_all_reduce_bias(symm_mem_input, x, bias, y)
    return y


def c10d_one_shot_all_reduce_bias_copy_out(
    x: torch.Tensor, bias: torch.Tensor, symm_mem_input: torch.Tensor
) -> torch.Tensor:
    y = torch.empty_like(x)
    torch.ops.symm_mem.one_shot_all_reduce_copy_out(
        symm_mem_input,
        x,
        "sum",
        dist.group.WORLD.group_name,
        y,
    )
    y.add_(bias)
    return y


def nccl_all_reduce_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(x)
    return x + bias


def create_benchmarks(
    b: int, t: int, d_size: int, device: torch.device, dtype: torch.dtype
):
    all_functions = {
        "nccl_ring": nccl_all_reduce_bias,
        "c10d_one_shot": c10d_one_shot_all_reduce_bias_copy_out,
        "triton_one_shot_bias_fusion": triton_one_shot_all_reduce_bias,
    }

    all_benchmarks = {}
    x = torch.randn(b, t, d_size, dtype=dtype, device=device)

    # Ensure bias to be the same across ranks
    torch.manual_seed(42)
    bias = torch.randn(b, t, d_size, dtype=dtype, device=device)

    for k, v in all_functions.items():
        if k != "nccl_ring":
            symm_mem_input = symm_mem.empty(b, t, d_size, dtype=dtype, device=device)
            symm_mem.rendezvous(symm_mem_input, dist.group.WORLD.group_name)
            all_benchmarks[k] = functools.partial(
                v, x=x.clone(), bias=bias.clone(), symm_mem_input=symm_mem_input
            )
        else:
            all_benchmarks[k] = functools.partial(v, x=x.clone(), bias=bias.clone())

    return all_benchmarks


@torch.no_grad()
def benchmark(device: torch.device, b: int, t: int, d_size: int) -> dict[str, float]:
    """
    Note that bias are the same across all ranks for this workload.

    dist.all_reduce(x)
    y = x + bias
    """

    gc.disable()
    all_benchmarks = create_benchmarks(b, t, d_size, device, torch.bfloat16)
    torch.cuda.synchronize()

    results = {}
    for k, v in all_benchmarks.items():
        runtime_us = log.benchmark_with_event(v, benchmark_iters=200, flush_l2=True)
        results[k] = runtime_us

    result_string = "\t".join([f"{k}: {v:.2f} us " for k, v in results.items()])
    if dist.get_rank() == 0:
        print(
            f"b: {b} \t"
            f"t: {t} \t"
            f"d: {d_size} \t"
            f"bytes: {b * t * d_size * torch.bfloat16.itemsize} \t"
            f"{result_string}"
        )

    return results


@click.command()
@click.option("--max_b", default=512)
@click.option("--max_t", default=1)
@click.option("--d_size", default=5120)
def main(max_b: int = 512, max_t: int = 1, d_size: int = 5120):
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 benchmark_all_reduce_bias.py
    """
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(42 + local_rank)

    # Assuming the input is [B, T, D] where T is 1
    # This is just one pattern
    d_size = 5120

    if dist.get_rank() == 0:
        print("Benchmarking ...")

    comm_bytes = []
    runtime_results = []
    b_sizes = [2**k for k in range(max_b.bit_length()) if 2**k <= max_b]
    t_sizes = [2**k for k in range(max_t.bit_length()) if 2**k <= max_t]
    for b in b_sizes:
        for t in t_sizes:
            result = benchmark(device, b, t=t, d_size=d_size)
            total_bytes = b * t * d_size * torch.bfloat16.itemsize
            comm_bytes.append(f"b:{b} t:{t} d:{d_size} bytes:{total_bytes}")

            runtime_results.append([])
            for v in result.values():
                runtime_results[-1].append(v)

    experiments = list(result.keys())
    if dist.get_rank() == 0:
        log.plot_experiment_comparison(
            comm_bytes, experiments, runtime_results, "benchmark_all_reduce_bias.png"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

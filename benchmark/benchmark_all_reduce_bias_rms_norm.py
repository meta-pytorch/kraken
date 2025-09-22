import functools
import gc
import os

import click
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import kraken
from kraken import _logging as log


def one_shot_all_reduce_bias_rms_norm(x, bias, rms_weight, symm_mem_input):
    y = torch.empty_like(x)
    kraken.all_reduce_fusion.one_shot_all_reduce_bias_rms_norm(symm_mem_input, x, bias, rms_weight, y)
    return y


def one_shot_all_reduce_bias_with_rms_norm(x, bias, rms_weight, symm_mem_input):
    y = torch.empty_like(x)
    kraken.all_reduce_fusion.one_shot_all_reduce_bias(symm_mem_input, x, bias, y)
    return kraken.all_reduce_fusion.rms_norm(y, rms_weight)


def two_shot_all_reduce_bias_rms_norm(x, bias, rms_weight, symm_mem_input):
    y = torch.empty_like(x)
    kraken.all_reduce_fusion.two_shot_all_reduce_bias_rms_norm(symm_mem_input, x, bias, rms_weight, y)
    return y


def two_shot_all_reduce_bias_with_rms_norm(x, bias, rms_weight, symm_mem_input):
    y = torch.empty_like(x)
    kraken.all_reduce_fusion.two_shot_all_reduce_bias(symm_mem_input, x, bias, y)
    return kraken.all_reduce_fusion.rms_norm(y, rms_weight)


def nccl_all_reduce_bias_rms_norm(x, bias, rms_weight):
    dist.all_reduce(x)
    y = x + bias
    return kraken.all_reduce_fusion.rms_norm(y, rms_weight)


def create_benchmarks(b, t, d_size, device, dtype):
    x = torch.randn(b, t, d_size, dtype=dtype, device=device)

    # Ensure that bias and w are the same across ranks
    torch.manual_seed(42)
    w = torch.randn(d_size, dtype=dtype, device=device)
    bias = torch.randn(b, t, d_size, dtype=dtype, device=device)

    all_functions = {
        "nccl_ring": nccl_all_reduce_bias_rms_norm,
        "one_shot_bias_fused + rms_norm": one_shot_all_reduce_bias_with_rms_norm,
        "two_shot_bias_fused + rms_norm": two_shot_all_reduce_bias_with_rms_norm,
        "one_shot_bias_rms_norm_fused": one_shot_all_reduce_bias_rms_norm,
        "two_shot_bias_rms_norm_fused": two_shot_all_reduce_bias_rms_norm,
    }
    all_benchmarks = {}
    for k, v in all_functions.items():
        if k == "nccl_ring":
            all_benchmarks[k] = functools.partial(
                v, x=x.clone(), bias=bias.clone(), rms_weight=w.clone()
            )
        else:
            symm_mem_input = symm_mem.empty(b, t, d_size, dtype=dtype, device=device)
            symm_mem.rendezvous(symm_mem_input, dist.group.WORLD.group_name)
            all_benchmarks[k] = functools.partial(
                v,
                x=x.clone(),
                bias=bias.clone(),
                rms_weight=w.clone(),
                symm_mem_input=symm_mem_input,
            )

    return all_benchmarks


@torch.no_grad()
def benchmark(device: torch.device, b: int, t: int, d_size: int) -> dict[str, float]:
    """
    NOTE: bias and w are the same across all ranks for this workload.

    dist.all_reduce(x)
    y = x + bias
    y = rms_norm(y, w)
    """
    gc.disable()

    all_benchmarks = create_benchmarks(b, t, d_size, device, torch.bfloat16)
    torch.cuda.synchronize()

    results = {}
    for k, v in all_benchmarks.items():
        runtime = log.benchmark_with_event(v, benchmark_iters=200, flush_l2=True)
        results[k] = runtime

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
    --no_python python3 benchmark_all_reduce_bias_rms_norm.py
    """
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(42 + local_rank)

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
            comm_bytes,
            experiments,
            runtime_results,
            "benchmark_all_reduce_bias_rms_norm.png",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

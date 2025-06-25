import atexit
import tempfile

import matplotlib.pyplot as plt
import torch.distributed as dist

triton_kernels = {}


def log_triton_kernel(kernel):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    def on_exit():
        print("PTX files:")
        for kernel in triton_kernels:
            f = tempfile.NamedTemporaryFile(dir="/tmp", delete=False)
            f.write(kernel.asm["ptx"].encode("utf-8"))
            print(f"+- {kernel.name}: {f.name}")

    if len(triton_kernels) == 0:
        atexit.register(on_exit)

    if kernel not in triton_kernels:
        triton_kernels[kernel] = None


def plot_experiment_comparison(
    sizes: list[str], experiments: list[str], data: list[list[float]], filename: str
):
    """
    Plots and saves a line chart comparing mechanisms' running times
    across experiment settings.

    Args:
        sizes: The input sizes.
        experiments: The names of the experiments.
        data: The data to plot.
        filename: The filename to save the plot to.
    """

    # Prepare X-axis labels
    x_pos = range(len(sizes))
    plt.figure(figsize=(12, 6))
    for i, experiment in enumerate(experiments):
        plt.plot(x_pos, [row[i] for row in data], marker="o", label=experiment)

    plt.xticks(x_pos, sizes, rotation=30, ha="right")
    plt.ylabel("Running Time (us)")
    plt.xlabel("Sizes")
    plt.title("Experiments")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as {filename}")

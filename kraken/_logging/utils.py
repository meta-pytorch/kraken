import torch.distributed as dist

triton_kernels = {}


def log_triton_kernel(kernel):
    import atexit
    import tempfile

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

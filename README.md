# Kraken

[**🎯 Features**](#-features) | [**🚀 Getting Started**](#-getting-started) | [**💻 Usage**](#-usage) | [**Benchmarks**](#-benchmarks) | [**🤝 Contributing**](#-contributing) | [**⚖️ License**](#️-license)

#### A Triton library of Symmetric Memory operators and examples.

</div>
This repository aims to be a cookbook for developing distributed AI models using Triton and PyTorch's symmetric memory capabilities. 

This is NOT intended to be a "framework" or "library" - it is intended to provide some high-performance Triton implementations with in-kernel communication for developers to hack on :) Please copy-paste and fork as you desire.


In additional to that, it includes a set of benchmarks to help researchers and developers explore and evaluate their implmentations. 

Our initial kernels are adapted from the [Symmetric Memory Recipes](https://github.com/yifuwang/symm-mem-recipes) by Yifu Wang.

## 🎯 Features
- Receipe for high-performance Triton implementations of `all_gather`, `all_reduce`, and `reduce_scatter`.
- Comm-comp fused kernels such as `gemm_one_shot_all_reduce_fused` for increased efficiency.
- A suite of benchmarks to measure and compare the performance of different comm + comp implementations.
- PTX utilities for synchronization primitives not yet supported by Triton. 

## 🚀 Getting Started
### Prerequisites
- PyTorch (version 2.6.0 or higher)
- Triton (version 3.3.0 or higher)
- Python (version 3.10 or higher)

### Installation
```bash
git clone https://github.com/meta-pytorch/kraken
cd kraken
pip install -e . -r requirements.txt
```

## 💻 Usage
Rather than a rigid framework, Kraken is a hands-on tutorial: developers can embed its techniques into xformers, FlashAttention, TorchInductor-generated kernels—or any custom Triton code. 

There are two ways of using Kraken kernels: 


You can import and use the Kraken kernels in your own PyTorch projects. Here is an example of how to use the `one_shot_all_reduce` kernel:

```python
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import kraken
import os

# local_rank is needed for device placement, and can be received from the environment
local_rank = int(os.environ["LOCAL_RANK"])

# Create and initialize a symmetric memory tensor
# See blog: https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/279 for symmetric memory details. 
a_shared = symm_mem.empty(
        (4096, 4096), 
        dtype=torch.bfloat16, 
        device=f"cuda:{local_rank}",
    )
symm_mem.rendezvous(a_shared, group=dist.group.WORLD)
a_shared = a_shared.normal_()

# Call one_shot_all_reduce kernel from kraken. 
a = kraken.one_shot_all_reduce(a_shared)
```

Alternatively, you can build your own custom kernels by leveraging Kraken's low-level primitives. This allows you to create highly optimized kernels tailored to your specific needs. We provide PTX implementations of low-level primitives in `kraken._ptx_utils`.

Here's an example of how to use `kraken._ptx_utils.symm_mem_sync` to synchronize blocks with matching `block_id` across participating devices in a custom kernel. This is often necessary before and after accessing symmetric memory tensors.

```python
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import triton
import triton.language as tl

import kraken
import os

@triton.jit
def custom_distributed_kernel(
    a_shared_ptrs,
    a_signal_pad_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    # Synchronizes blocks with matching block_id across participating devices.
    # Ensures that all writes to a_shared from previous kernels across all devices
    #  are visible to the current kernel:
    kraken._ptx_utils.symm_mem_sync(
        a_signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=True,
    )
    ...  # access a_shared via a_shared_ptrs.

# Create and initialize a symmetric memory tensor
local_rank = int(os.environ["LOCAL_RANK"])
a_shared = symm_mem.empty((4096, 4096), dtype=torch.bfloat16, device=f"cuda:{local_rank}")
symm_mem_hdl = symm_mem.rendezvous(a_shared, group=dist.group.WORLD)

# Define the grid for kernel launch. For simplicity, we use a single thread block.
grid = (1,)

# Call custom kernel
custom_distributed_kernel[grid](
    symm_mem_hdl.buffer_ptrs_dev,
    symm_mem_hdl.signal_pad_ptrs_dev,
    rank=symm_mem_hdl.rank,
    world_size=symm_mem_hdl.world_size,
)
```


## 📁 Structure
Kraken is organized for easy hacking of distributed Triton kernel: 

### Example Kernels
#### `kraken.all_gather_fusion`
- `all_gather_matmul`
#### `kraken.all_reduce_fusion`
- `rms_norm`,
- `gemm_one_shot_all_reduce_fused`
-  `one_shot_all_reduce_bias`
- `one_shot_all_reduce_bias_rms_norm`
- `two_shot_all_reduce_bias`
- `two_shot_all_reduce_bias_rms_norm`
- `one_shot_all_reduce`
#### `kraken.reduce_scatter_fusion`
- `gemm_reduce_scatter`
- `gemm_reduce_scatter_ce_persistent`


### Inline PTX Utils
`kraken._ptx_utils` provides inline ptx implementation of memory barrier synchorinzations that are not natively supported by triton. 



### Benchmarks
Kraken includes a set of benchmarks in `benchmarks/` to evaluate the performance of its kernels. You can run them as follows:

```bash
torchrun --nnodes 1 --nproc-per-node 8 \
--rdzv-backend c10d --rdzv-endpoint localhost:0 --no_python python3 \
benchmark/benchmark_all_reduce.py \
--backend nccl,triton_1shot,dist_1shot
# ... and so on for other benchmarks
```

Run with `--help` to see configurable benchmark arguments for setting backends, dtype, shape etc. to profile. 
```bash
python benchmark/benchmark_all_reduce.py --help
```


## 🤝 Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more details on how to contribute to the project.

## ⚖️ License
Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository.
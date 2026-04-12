#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# torchrun --nproc-per-node 4 --standalone ring_attention.py

"""
Ring attention using NVSHMEM Triton kernels.

This example demonstrates how to create a Triton kernel that gets key and values tensors
from neighbors in a ring topology, and interleaves the communication with attention computation.
"""

import triton.language as tl

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
from torch._inductor.runtime.triton_compat import triton


@triton.jit
def nvshmem_ring_attention_kernel(
    query_ptr,
    temp_key_ptr,
    key_ptr,
    temp_value_ptr,
    value_ptr,
    attention_ptr,
    n_tokens: tl.constexpr,
    hid_dim: tl.constexpr,
    my_pe: tl.constexpr,
    n_pes: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    nelems = BLOCK_SIZE_M * hid_dim

    # # Online attention initialization
    # m = tl.zeros((n_tokens,), dtype=tl.bfloat16)
    # d = tl.full((n_tokens,), 1.0, dtype=tl.bfloat16)

    a = tl.zeros((BLOCK_SIZE_M, hid_dim), dtype=tl.float32)

    q_block_ptr = tl.make_block_ptr(
        base=query_ptr,
        shape=(n_tokens, hid_dim),
        strides=(hid_dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, hid_dim),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=temp_key_ptr,
        shape=(n_tokens, hid_dim),
        strides=(hid_dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, hid_dim),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=temp_value_ptr,
        shape=(n_tokens, hid_dim),
        strides=(hid_dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, hid_dim),
        order=(1, 0),
    )
    a_block_ptr = tl.make_block_ptr(
        base=attention_ptr,
        shape=(n_tokens, hid_dim),
        strides=(hid_dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, hid_dim),
        order=(1, 0),
    )

    for peer_dist in tl.static_range(1, n_pes):
        peer = (my_pe + peer_dist) % n_pes
        nvshmem.get(temp_key_ptr, key_ptr, nelems, peer)
        nvshmem.get(temp_value_ptr, value_ptr, nelems, peer)

        query = tl.load(q_block_ptr)
        key = tl.load(k_block_ptr)
        key_t = tl.trans(key)
        value = tl.load(v_block_ptr)

        # Calculate dot product score
        s = tl.dot(query, key_t)
        a = tl.dot(s, value.cast(tl.float32), a)

        # Online algorithm
        # # Update max score
        # old_m = m
        # m = tl.maximum(s, old_m)
        # # Update denominator
        # a = tl.exp(old_m - m)
        # b = tl.exp(s - m)
        # d_scaled = d * a
        # d = d_scaled + b
        # # Calculate attention
        # a = a * d_scaled / d + b / d * temp_value

    a = a.to(tl.bfloat16)
    tl.store(a_block_ptr, a)


def ring_attention_example():
    """
    Example demonstrating ring attention.
    """
    # Initialize distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Set up NVSHMEM
    symm_mem.set_backend("NVSHMEM")
    nvshmem_lib = nvshmem.enable_triton()
    group_name = dist.distributed_c10d._get_default_group().group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    
    # Configuration
    n_tokens = 16
    hid_dim = 256
    shape = (n_tokens, hid_dim)
    dtype = torch.bfloat16
    
    # Create symmetric tensors
    # Each rank fills with its own rank value
    key = symm_mem.empty(shape, dtype=dtype, device=device).fill_(rank)
    temp_key = symm_mem.empty(shape, dtype=dtype, device=device).fill_(-1)
    value = symm_mem.empty(shape, dtype=dtype, device=device).fill_(rank)
    temp_value = symm_mem.empty(shape, dtype=dtype, device=device).fill_(-1)

    # Query is local tensor
    query = torch.empty(shape, dtype=dtype, device=device).copy_(
        torch.randn(shape, dtype=dtype, device=device)
    )
    attention = torch.zeros(shape, dtype=dtype, device=device)
    
    # Rendezvous
    symm_mem.rendezvous(key, group=group_name)
    symm_mem.rendezvous(temp_key, group=group_name)
    symm_mem.rendezvous(value, group=group_name)
    symm_mem.rendezvous(temp_value, group=group_name)
    
    # Synchronize before operations
    dist.barrier()
    
    # Execute ring attention
    nvshmem_ring_attention_kernel[(1,)](
        query,
        temp_key, key,
        temp_value, value,
        attention,
        n_tokens, hid_dim,
        rank, world_size,
        BLOCK_SIZE_M=n_tokens,
        extern_libs=nvshmem_lib,
    )
    
    # Synchronize after operations
    dist.barrier()
    
    # Verify results
    print(f"Rank {rank}: attention = {attention}")
    print(f"Rank {rank}: Ring attention completed successfully!")


if __name__ == "__main__":
    if not symm_mem.is_nvshmem_available():
        print("NVSHMEM not available")
        exit(0)
    
    dist.init_process_group()

    print("Running ring attention example...")
    ring_attention_example()

    dist.destroy_process_group()

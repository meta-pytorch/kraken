import contextlib
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
import triton
import triton.language as tl

from .. import _ptx_utils as ptx_utils

# Check if NVSHMEM backend is enabled via environment variable
_NVSHMEM_ENV = os.environ.get("TORCH_SYMMMEM", "").upper() == "NVSHMEM"


def init_nvshmem() -> None:
    """Initialize NVSHMEM backend for symmetric memory.

    When TORCH_SYMMMEM=NVSHMEM is set, this function verifies that NVSHMEM
    is available and attempts to set it as the backend.

    Raises:
        RuntimeError: If NVSHMEM is not available.

    Note:
        It is okay if set_backend() fails (e.g., if called multiple times
        from multiple FlexCPMaskedGather instances).
    """
    print("Calling init_nvshmem")
    if not symm_mem.is_nvshmem_available():
        raise RuntimeError("NVSHMEM is not available but TORCH_SYMMMEM=NVSHMEM is set")

    print("Using NVSHMEM backend for symmetric memory")
    with contextlib.suppress(RuntimeError):
        symm_mem.set_backend("NVSHMEM")


def get_required_blocks(
    kv_indices: torch.Tensor,
    kv_num_blocks: torch.Tensor,
    full_kv_indices: torch.Tensor | None,
    full_kv_num_blocks: torch.Tensor | None,
    num_total_blocks: int | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Extract unique required KV block indices from BlockMask sparse representation.

    Uses a scatter-based approach instead of unique() for better performance.
    Scatter is O(n) vs unique's O(n log n) due to sorting.

    Args:
        kv_indices: Block indices from BlockMask, shape [B, H, Q, KV]
        kv_num_blocks: Number of valid blocks per query, shape [B, H, Q]
        full_kv_indices: Optional full block indices, shape [B, H, Q, KV]
        full_kv_num_blocks: Optional number of valid full blocks, shape [B, H, Q]
        num_total_blocks: Optional total number of blocks (seqlen // block_size).
                          If not provided, will be computed from max index (slower).

    Returns:
        Tuple of (required_blocks, max_kv_blocks) where required_blocks has
        shape [B, max_kv_blocks] containing sorted unique block indices.
    """
    B, H, Q, KV = kv_indices.shape
    device = kv_indices.device
    dtype = kv_indices.dtype
    sentinel = KV + 1

    # Create range for masking valid entries
    kv_range = torch.arange(KV, device=device).view(1, 1, 1, KV)

    # Get valid mask for kv_indices
    valid_mask = kv_range < kv_num_blocks.unsqueeze(-1)  # [B, H, Q, KV]

    # Combine with full_kv_indices if present
    if full_kv_indices is not None:
        full_valid_mask = kv_range < full_kv_num_blocks.unsqueeze(-1)
        # Concatenate masks and indices
        valid_mask = torch.cat([valid_mask, full_valid_mask], dim=-1)
        kv_indices = torch.cat([kv_indices, full_kv_indices], dim=-1)

    # Determine buffer size for boolean mask
    if num_total_blocks is None:
        # Fallback: compute from max index (triggers GPU sync)
        num_total_blocks = int(kv_indices.max().item()) + 1

    # Vectorized batched scatter approach
    # Flatten to [B, total_elements]
    total_elements = H * Q * kv_indices.shape[-1]
    valid_indices_flat = kv_indices.reshape(B, total_elements)  # [B, total]
    valid_mask_flat = valid_mask.reshape(B, total_elements)  # [B, total]

    # Create boolean mask for all batches: [B, num_total_blocks]
    block_masks = torch.zeros(B, num_total_blocks, dtype=torch.bool, device=device)

    # For each batch, scatter True only at valid positions
    # We need to loop here because scatter_ with False values would overwrite True values
    for b in range(B):
        valid_idx = valid_indices_flat[b][valid_mask_flat[b]].long()
        block_masks[b].scatter_(0, valid_idx, True)

    # Vectorized extraction of sorted indices using argsort
    # Convert bool mask to int and sort descending to get True positions first
    # Then the sorted indices give us the block indices in order
    block_masks_int = block_masks.int()  # [B, num_total_blocks]

    # Use argsort on negated mask to get True positions first (stable sort)
    # argsort on -block_masks_int puts 1s (True) before 0s (False)
    sorted_indices = torch.argsort(
        -block_masks_int, dim=1, stable=True
    )  # [B, num_total_blocks]

    # Get number of True values per batch
    num_valid = block_masks.sum(dim=1)  # [B]

    # Create output tensor with sentinel
    ret = torch.full((B, KV), sentinel, device=device, dtype=dtype)

    # Create mask for valid positions (up to KV elements)
    pos_range = torch.arange(KV, device=device).unsqueeze(0)  # [1, KV]
    valid_pos_mask = pos_range < num_valid.unsqueeze(1)  # [B, KV]

    # Take first KV positions from sorted_indices
    first_kv_indices = sorted_indices[:, :KV]  # [B, KV]

    # Copy only valid positions
    ret = torch.where(valid_pos_mask, first_kv_indices.to(dtype), ret)

    return ret, KV


@triton.jit
def copy_valid_blocks_kernel(
    sharded_k_tuple,
    sharded_v_tuple,
    output_k_ptr,
    output_v_ptr,
    blocks_ptr,
    valid_blocks_ptr,
    signal_pad_ptrs,
    B: tl.constexpr,
    H: tl.constexpr,
    KV_per_rank: tl.constexpr,
    KV_total: tl.constexpr,
    hidden: tl.constexpr,
    block_size: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    """
    Triton kernel for copying valid blocks from sharded K/V tensors to output K/V tensors.
    Uses symmetric memory to access remote buffers directly.

    Each program copies a tile of (BLOCK_H, block_size, BLOCK_HIDDEN) from one
    valid block. Grid is over (batch, kv_block, h_tiles, hidden_tiles).
    """
    # Synchronize before accessing remote memory
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasSubsequentMemAccess=True
    )

    pid = tl.program_id(axis=0)

    # Calculate grid dimensions
    h_tiles = tl.cdiv(H, BLOCK_H)
    hidden_tiles = tl.cdiv(hidden, BLOCK_HIDDEN)
    tiles_per_kv_block = h_tiles * hidden_tiles
    total_tiles_per_batch = max_kv_blocks * tiles_per_kv_block

    # Decompose program ID
    batch_idx = pid // total_tiles_per_batch
    remaining = pid % total_tiles_per_batch
    kv_block_idx = remaining // tiles_per_kv_block
    tile_idx = remaining % tiles_per_kv_block
    h_tile_idx = tile_idx // hidden_tiles
    hidden_tile_idx = tile_idx % hidden_tiles

    if batch_idx >= B:
        # Still need to synchronize even if we exit early
        ptx_utils.symm_mem_sync(
            signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
        )
        return

    # Load number of valid blocks for this batch
    num_valid = tl.load(valid_blocks_ptr + batch_idx)
    if kv_block_idx >= num_valid:
        # Still need to synchronize even if we exit early
        ptx_utils.symm_mem_sync(
            signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
        )
        return

    # Load the block number
    block_num = tl.load(blocks_ptr + batch_idx * max_kv_blocks + kv_block_idx)

    # Calculate rank and local block index
    blocks_per_rank = KV_per_rank // block_size
    src_rank = block_num // blocks_per_rank
    rank_block_idx = block_num % blocks_per_rank

    # Compute H and hidden indices for this tile
    h_start = h_tile_idx * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    hidden_start = hidden_tile_idx * BLOCK_HIDDEN
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = hidden_offsets < hidden

    # Create 2D mask
    mask_2d = h_mask[:, None] & hidden_mask[None, :]

    # Copy all KV positions in the block for this (H, hidden) tile
    for kv_offset in range(block_size):
        # Compute source indices in sharded tensors
        # sharded_k/v: (B, H, KV_per_rank, hidden) - per-rank shape
        kv_pos_src = rank_block_idx * block_size + kv_offset

        # Compute destination indices in output tensors
        # output_k/v: (B, H, KV_total, hidden)
        kv_pos_dst = block_num * block_size + kv_offset

        dst_k_ptrs = (
            output_k_ptr
            + batch_idx * (H * KV_total * hidden)
            + h_offsets[:, None] * (KV_total * hidden)
            + kv_pos_dst * hidden
            + hidden_offsets[None, :]
        )

        dst_v_ptrs = (
            output_v_ptr
            + batch_idx * (H * KV_total * hidden)
            + h_offsets[:, None] * (KV_total * hidden)
            + kv_pos_dst * hidden
            + hidden_offsets[None, :]
        )

        # Load from the correct rank's buffer and store
        # Use static_range to iterate over ranks and find the matching one
        for i in tl.static_range(world_size):
            if src_rank == i:
                sharded_k_ptr = sharded_k_tuple[i]
                sharded_v_ptr = sharded_v_tuple[i]

                # Compute pointers for 2D load (H, hidden) - K
                src_k_ptrs = (
                    sharded_k_ptr
                    + batch_idx * (H * KV_per_rank * hidden)
                    + h_offsets[:, None] * (KV_per_rank * hidden)
                    + kv_pos_src * hidden
                    + hidden_offsets[None, :]
                )

                # Compute pointers for 2D load (H, hidden) - V
                src_v_ptrs = (
                    sharded_v_ptr
                    + batch_idx * (H * KV_per_rank * hidden)
                    + h_offsets[:, None] * (KV_per_rank * hidden)
                    + kv_pos_src * hidden
                    + hidden_offsets[None, :]
                )

                # Load and store K
                vals_k = tl.load(src_k_ptrs, mask=mask_2d)
                tl.store(dst_k_ptrs, vals_k, mask=mask_2d)

                # Load and store V
                vals_v = tl.load(src_v_ptrs, mask=mask_2d)
                tl.store(dst_v_ptrs, vals_v, mask=mask_2d)

    # Synchronize after accessing remote memory
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs, None, rank, world_size, hasPreviousMemAccess=True
    )


@nvshmem.requires_nvshmem
@triton.jit
def copy_valid_blocks_kernel_nvshmem(
    sharded_k_ptr,
    sharded_v_ptr,
    output_k_ptr,
    output_v_ptr,
    blocks_ptr,
    valid_blocks_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    KV_per_rank: tl.constexpr,
    KV_total: tl.constexpr,
    hidden: tl.constexpr,
    block_size: tl.constexpr,
    my_pe: tl.constexpr,
    n_pes: tl.constexpr,
    max_kv_blocks: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    """
    NVSHMEM Triton kernel for copying valid blocks using nvshmem.get().

    Uses nvshmem.get to fetch data directly into the output buffer,
    avoiding temp buffer race conditions between parallel programs.
    """
    pid = tl.program_id(axis=0)

    # Calculate grid dimensions
    h_tiles = tl.cdiv(H, BLOCK_H)
    hidden_tiles = tl.cdiv(hidden, BLOCK_HIDDEN)
    tiles_per_kv_block = h_tiles * hidden_tiles
    total_tiles_per_batch = max_kv_blocks * tiles_per_kv_block

    # Decompose program ID
    batch_idx = pid // total_tiles_per_batch
    remaining = pid % total_tiles_per_batch
    kv_block_idx = remaining // tiles_per_kv_block
    tile_idx = remaining % tiles_per_kv_block
    h_tile_idx = tile_idx // hidden_tiles
    hidden_tile_idx = tile_idx % hidden_tiles

    if batch_idx >= B:
        return

    # Load number of valid blocks for this batch
    num_valid = tl.load(valid_blocks_ptr + batch_idx)
    if kv_block_idx >= num_valid:
        return

    # Load the block number
    block_num = tl.load(blocks_ptr + batch_idx * max_kv_blocks + kv_block_idx)

    # Calculate rank and local block index
    blocks_per_rank = KV_per_rank // block_size
    src_pe = block_num // blocks_per_rank
    rank_block_idx = block_num % blocks_per_rank

    # Compute H and hidden indices for this tile
    h_start = h_tile_idx * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    hidden_start = hidden_tile_idx * BLOCK_HIDDEN
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = hidden_offsets < hidden

    # Create 2D mask
    mask_2d = h_mask[:, None] & hidden_mask[None, :]

    # Copy all KV positions in the block for this (H, hidden) tile
    for kv_offset in range(block_size):
        # Compute source indices in sharded tensors
        kv_pos_src = rank_block_idx * block_size + kv_offset

        # Compute destination indices in output tensors
        kv_pos_dst = block_num * block_size + kv_offset

        if src_pe != my_pe:
            # For remote PE, use nvshmem.get to fetch directly to output
            # Fetch each row (h value) separately since data is strided
            for h_idx in tl.static_range(BLOCK_H):
                h_val = h_start + h_idx
                if h_val < H:
                    # Source: contiguous BLOCK_HIDDEN elements
                    src_offset = (
                        batch_idx * (H * KV_per_rank * hidden)
                        + h_val * (KV_per_rank * hidden)
                        + kv_pos_src * hidden
                        + hidden_start
                    )

                    # Destination: directly in output buffer
                    dst_offset = (
                        batch_idx * (H * KV_total * hidden)
                        + h_val * (KV_total * hidden)
                        + kv_pos_dst * hidden
                        + hidden_start
                    )

                    # Fetch BLOCK_HIDDEN elements directly to output
                    nvshmem.get(
                        output_k_ptr,
                        sharded_k_ptr,
                        BLOCK_HIDDEN,
                        src_pe,
                    )
                    """
                    nvshmem.get(
                        output_v_ptr + dst_offset,
                        sharded_v_ptr + src_offset,
                        BLOCK_HIDDEN,
                        src_pe,
                    )
                    """
        else:
            # Load directly from local buffer
            src_k_ptrs = (
                sharded_k_ptr
                + batch_idx * (H * KV_per_rank * hidden)
                + h_offsets[:, None] * (KV_per_rank * hidden)
                + kv_pos_src * hidden
                + hidden_offsets[None, :]
            )
            src_v_ptrs = (
                sharded_v_ptr
                + batch_idx * (H * KV_per_rank * hidden)
                + h_offsets[:, None] * (KV_per_rank * hidden)
                + kv_pos_src * hidden
                + hidden_offsets[None, :]
            )
            vals_k = tl.load(src_k_ptrs, mask=mask_2d)
            vals_v = tl.load(src_v_ptrs, mask=mask_2d)

            # Compute destination pointers
            dst_k_ptrs = (
                output_k_ptr
                + batch_idx * (H * KV_total * hidden)
                + h_offsets[:, None] * (KV_total * hidden)
                + kv_pos_dst * hidden
                + hidden_offsets[None, :]
            )
            dst_v_ptrs = (
                output_v_ptr
                + batch_idx * (H * KV_total * hidden)
                + h_offsets[:, None] * (KV_total * hidden)
                + kv_pos_dst * hidden
                + hidden_offsets[None, :]
            )

            # Store K and V
            tl.store(dst_k_ptrs, vals_k, mask=mask_2d)
            tl.store(dst_v_ptrs, vals_v, mask=mask_2d)


def _copy_valid_blocks_triton(
    blocks: torch.Tensor,
    valid_blocks: torch.Tensor,
    sharded_k: torch.Tensor,
    sharded_v: torch.Tensor,
    output_k: torch.Tensor,
    output_v: torch.Tensor,
    block_size: int = 128,
    group_name: str | None = None,
) -> None:
    """
    Triton-accelerated version of copy_valid_blocks using symmetric memory.

    This implementation uses Triton kernels with symmetric memory for direct
    remote buffer access across ranks. When TORCH_SYMMMEM=NVSHMEM is set,
    uses NVSHMEM backend with nvshmem.get() operations.

    Args:
        blocks: (B, KV) tensor containing block indices.
        valid_blocks: (B,) tensor indicating number of valid blocks for each
                      batch element.
        sharded_k: Local K tensor on symmetric memory with shape
                   (B, H, KV_per_rank, hidden).
        sharded_v: Local V tensor on symmetric memory with shape
                   (B, H, KV_per_rank, hidden).
        output_k: Destination K tensor with shape (B, H, KV_total, hidden).
        output_v: Destination V tensor with shape (B, H, KV_total, hidden).
        block_size: Size of each block in elements (default: 128).
        group_name: Process group name for symmetric memory operations.
                   If None, uses WORLD group.

    Note:
        sharded_k and sharded_v must be allocated using symm_mem.empty() and
        registered with symm_mem.rendezvous() before calling this function.
    """
    # Get symmetric memory handles for K and V
    if group_name is None:
        group_name = dist.group.WORLD.group_name
    symm_mem_hdl_k = symm_mem.rendezvous(sharded_k, group_name)
    symm_mem_hdl_v = symm_mem.rendezvous(sharded_v, group_name)

    world_size = symm_mem_hdl_k.world_size
    rank = symm_mem_hdl_k.rank

    B, H, KV_per_rank, hidden = sharded_k.shape
    _, _, KV_total, _ = output_k.shape
    _, max_kv_blocks = blocks.shape

    # Kernel configuration - tile sizes for H and hidden dimensions
    BLOCK_H = min(8, H)
    BLOCK_HIDDEN = min(128, hidden)

    # Calculate grid size
    h_tiles = triton.cdiv(H, BLOCK_H)
    hidden_tiles = triton.cdiv(hidden, BLOCK_HIDDEN)
    tiles_per_kv_block = h_tiles * hidden_tiles
    total_tiles_per_batch = max_kv_blocks * tiles_per_kv_block
    num_programs = B * total_tiles_per_batch

    if _NVSHMEM_ENV:
        copy_valid_blocks_kernel_nvshmem[(num_programs,)](
            sharded_k,
            sharded_v,
            output_k,
            output_v,
            blocks,
            valid_blocks,
            B=B,
            H=H,
            KV_per_rank=KV_per_rank,
            KV_total=KV_total,
            hidden=hidden,
            block_size=block_size,
            my_pe=rank,
            n_pes=world_size,
            max_kv_blocks=max_kv_blocks,
            BLOCK_H=BLOCK_H,
            BLOCK_HIDDEN=BLOCK_HIDDEN,
        )
    else:
        # Use default symmetric memory kernel
        # Get remote buffer pointers for all ranks - K
        k_buf_list = [
            symm_mem_hdl_k.get_buffer(i, tuple(sharded_k.shape), sharded_k.dtype)
            for i in range(world_size)
        ]
        k_buf_tuple = tuple(k_buf_list)

        # Get remote buffer pointers for all ranks - V
        v_buf_list = [
            symm_mem_hdl_v.get_buffer(i, tuple(sharded_v.shape), sharded_v.dtype)
            for i in range(world_size)
        ]
        v_buf_tuple = tuple(v_buf_list)

        # Launch kernel with symmetric memory buffers
        copy_valid_blocks_kernel[(num_programs,)](
            k_buf_tuple,
            v_buf_tuple,
            output_k,
            output_v,
            blocks,
            valid_blocks,
            symm_mem_hdl_k.signal_pad_ptrs_dev,
            B=B,
            H=H,
            KV_per_rank=KV_per_rank,
            KV_total=KV_total,
            hidden=hidden,
            block_size=block_size,
            rank=rank,
            world_size=world_size,
            max_kv_blocks=max_kv_blocks,
            BLOCK_H=BLOCK_H,
            BLOCK_HIDDEN=BLOCK_HIDDEN,
        )


class FlexCPMaskedGather:
    """
    Mask-aware communication for Context Parallel FlexAttention.

    This class manages symmetric memory buffers for efficient KV gathering
    based on attention masks. Instead of gathering all K/V blocks from all
    ranks (like flex_cp_allgather), it only fetches the required blocks
    based on the attention mask, reducing communication when many KV blocks
    are masked out.

    The class handles:
    - Lazy allocation of symmetric memory buffers
    - Automatic reallocation when tensor shapes change
    - Extraction of required blocks from BlockMask
    - Efficient copying using Triton kernels with symmetric memory

    Example:
        >>> gatherer = FlexCPMaskedGather()
        >>> k_gathered, v_gathered = gatherer.gather(
        ...     k_shard, v_shard, mask_shard, seqlen, device_mesh
        ... )
    """

    def __init__(self):
        """Initialize the FlexCPMaskedGather instance.

        Symmetric memory buffers are not allocated here; they are lazily
        allocated on the first call to gather() based on input tensor shapes.

        If TORCH_SYMMMEM=NVSHMEM is set, initializes the NVSHMEM backend.

        Raises:
            RuntimeError: If TORCH_SYMMMEM=NVSHMEM but NVSHMEM is not available.
        """
        print("FlexCPMaskedGather.__init__()")
        self._k_symm: torch.Tensor | None = None
        self._v_symm: torch.Tensor | None = None
        self._compiled_get_required_blocks = None

        # Cache for block extraction results (reused when mask is unchanged)
        self._cached_mask = None
        self._cached_seqlen: int | None = None
        self._cached_blocks: torch.Tensor | None = None
        self._cached_valid_blocks: torch.Tensor | None = None
        self._cached_block_size: int | None = None

        # Initialize NVSHMEM if requested
        if _NVSHMEM_ENV:
            init_nvshmem()
        symm_mem.enable_symm_mem_for_group(
            dist.distributed_c10d._get_process_group_name(dist.group.WORLD)
        )
        dist.barrier()

    def _ensure_symm_memory(
        self,
        sharded_k: torch.Tensor,
        sharded_v: torch.Tensor,
        group_name: str,
    ) -> None:
        """Ensure symmetric memory buffers are allocated, copied, and rendezvous'd.

        This method handles all symmetric memory setup:
        1. Allocates buffers if needed (or reallocates if shape/dtype changes)
        2. Copies input tensors to symmetric memory
        3. Calls rendezvous to make memory accessible across ranks

        Args:
            sharded_k: K tensor shard with shape (B, H, KV_per_rank, hidden)
            sharded_v: V tensor shard with shape (B, H, KV_per_rank, hidden)
            group_name: Process group name for symmetric memory initialization
        """
        shape = sharded_k.shape
        dtype = sharded_k.dtype
        device = sharded_k.device

        # Check if we need to allocate or reallocate
        need_alloc = (
            self._k_symm is None
            or self._k_symm.shape != shape
            or self._k_symm.dtype != dtype
        )

        if need_alloc:
            # Enable symmetric memory for the group before allocation
            symm_mem.enable_symm_mem_for_group(group_name)

            self._k_symm = symm_mem.empty(shape, dtype=dtype, device=device)
            self._v_symm = symm_mem.empty(shape, dtype=dtype, device=device)

        # Rendezvous to make symmetric memory accessible across ranks
        symm_mem.rendezvous(self._k_symm, group_name)
        symm_mem.rendezvous(self._v_symm, group_name)

        # Copy input tensors to symmetric memory
        self._k_symm.copy_(sharded_k)
        self._v_symm.copy_(sharded_v)

    @staticmethod
    def _get_required_blocks_from_mask(
        mask_shard,
        seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Extract required block indices from a sharded BlockMask.

        Args:
            mask_shard: Sharded BlockMask from context parallel sharding
            seqlen: Total sequence length across all ranks

        Returns:
            Tuple of (blocks, valid_blocks, block_size) where:
            - blocks: (B, max_kv_blocks) tensor of required block indices
            - valid_blocks: (B,) tensor of valid block counts per batch
            - block_size: Size of each attention block
        """
        kv_indices = mask_shard.kv_indices
        kv_num_blocks = mask_shard.kv_num_blocks
        full_kv_indices = mask_shard.full_kv_indices
        full_kv_num_blocks = mask_shard.full_kv_num_blocks
        block_size = mask_shard.BLOCK_SIZE[0]
        num_total_blocks = seqlen // block_size

        blocks, max_kv_blocks = get_required_blocks(
            kv_indices,
            kv_num_blocks,
            full_kv_indices,
            full_kv_num_blocks,
            num_total_blocks,
        )

        sentinel = max_kv_blocks + 1
        valid_blocks = (blocks != sentinel).sum(dim=1).to(torch.int32)

        return blocks, valid_blocks, block_size

    def gather(
        self,
        sharded_k: torch.Tensor,
        sharded_v: torch.Tensor,
        mask_shard,
        seqlen: int,
        mesh: "torch.distributed.device_mesh.DeviceMesh",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V tensors using mask-aware communication.

        This method only fetches the required KV blocks based on the attention
        mask, using symmetric memory for direct GPU-to-GPU access.

        Args:
            sharded_k: K tensor shard with shape (B, H, KV_per_rank, hidden)
            sharded_v: V tensor shard with shape (B, H, KV_per_rank, hidden)
            mask_shard: Sharded BlockMask from context parallel sharding
            seqlen: Total sequence length across all ranks
            mesh: DeviceMesh for the context parallel group

        Returns:
            Tuple of (k_gathered, v_gathered) with shape (B, H, KV_total, hidden)
        """
        B, H, KV_per_rank, hidden = sharded_k.shape
        dtype = sharded_k.dtype
        device = sharded_k.device
        group = mesh.get_group()
        group_name = group.group_name

        # Ensure symmetric memory is allocated, copied, and rendezvous'd
        self._ensure_symm_memory(sharded_k, sharded_v, group_name)

        # Get required blocks from mask (use cache if mask is unchanged)
        if mask_shard is self._cached_mask and seqlen == self._cached_seqlen:
            blocks = self._cached_blocks
            valid_blocks = self._cached_valid_blocks
            block_size = self._cached_block_size
        else:
            blocks, valid_blocks, block_size = self._get_required_blocks_from_mask(
                mask_shard, seqlen
            )
            # Update cache
            self._cached_mask = mask_shard
            self._cached_seqlen = seqlen
            self._cached_blocks = blocks
            self._cached_valid_blocks = valid_blocks
            self._cached_block_size = block_size

        # Allocate output tensors
        output_k = torch.empty(B, H, seqlen, hidden, device=device, dtype=dtype)
        output_v = torch.empty(B, H, seqlen, hidden, device=device, dtype=dtype)

        # Copy only required blocks using Triton kernel with symmetric memory
        _copy_valid_blocks_triton(
            blocks,
            valid_blocks,
            self._k_symm,
            self._v_symm,
            output_k,
            output_v,
            block_size,
            group_name,
        )

        return output_k, output_v

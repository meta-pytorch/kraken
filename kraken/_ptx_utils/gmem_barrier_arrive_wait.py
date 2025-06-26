import triton
import triton.language as tl


@triton.jit
def arrive_gmem_barrier(
    addr,
    update: tl.constexpr = 1,  # set the lock value to
    sem: tl.constexpr = "release",
    scope: tl.constexpr = "gpu",
    op: tl.constexpr = "atomic_xchg",
    skip_sync: tl.constexpr = False,
):
    tl.static_assert(
        op == "atomic_xchg",
        "Currently only support atomic_xchg wait on gmem_barriers. ",
    )

    if not skip_sync:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )
    return tl.atomic_xchg(addr, update, sem=sem, scope=scope)


@triton.jit
def wait_gmem_barrier(
    addr,
    expect: tl.constexpr = 1,  # wait until lock is set to expect
    update: tl.constexpr = 0,  # update the lock once it is aquired.
    sem: tl.constexpr = "acquire",
    scope: tl.constexpr = "gpu",
    op: tl.constexpr = "ld",
    skip_sync: tl.constexpr = False,
):
    """
    Wait for a global memory barrier to reach the expected state.

    This function implements a spin-wait loop that continuously checks a memory location
    until it reaches the expected value, providing synchronization across GPU threads.

    Args:
        addr: Memory address of the barrier to wait on (Must be a scalar)
        expect: Expected value to wait for (default: 1)
        update: Update the barrier with once acquired (default: 0)
        sem: Memory semantics for the atomic operation (default: "acquire")
        scope: Scope of the atomic operation. Options: "gpu", "sys" (default: "gpu")
        op: Atomic operation type (default: "ld", currently only supported option)
    """
    tl.static_assert(
        op == "ld" and update == 0, "Currently only support ld wait on gmem_barriers. "
    )
    # TODO(joydddd): add support for cas barriers.

    tl.static_assert(addr.type.is_ptr(), "Barrier address must be a scalar.")
    # TODO(joydddd): add wait_gmem_multi_barrier. (each thread waits on a different barrier).

    # Spin-wait loop:
    #   Uses atomic_add with update=0 for ld.global.{sem}.{scope}
    #   Triton generates smem broadcasting of tl.atomic_add return value in ptx,
    #   but it is optimized away by ptxas in SASS, hence no performance overhead.
    while tl.atomic_add(addr, update, sem=sem, scope=scope) != expect:
        pass

    if not skip_sync:
        tl.inline_asm_elementwise(
            "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )
    # tl.debug_barrier() cause significant performance loss. (Perhaps breaks triton prefetching?)

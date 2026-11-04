"""NVIDIA shared transaction barriers; compile checked, run on sm_120a only."""

from krn.frontend.numerics import todo, u32, u64


def init(bar: u64, count: u32) -> None:
    """Start the barrier `bar`, an element of a shared u64 array, expecting `count` arrivals per phase."""
    todo()


def arrive(bar: u64) -> u64:
    """Arrive once at `bar`; the result is the phase token."""
    return todo()


def arrive_expect_tx(bar: u64, bytes: u32) -> u64:
    """Arrive once and add `bytes` of asynchronous copy traffic that must land before the phase completes (sm_90)."""
    return todo()


def wait_parity(bar: u64, parity: u32) -> None:
    """Spin until the phase of the given parity completes; trap instead of hanging after about 2^26 tries (sm_90)."""
    todo()

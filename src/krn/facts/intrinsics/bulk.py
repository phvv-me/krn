"""NVIDIA bulk copies; compile checked, run on sm_120a only."""

from krn.frontend.numerics import Scalar, todo, u32, u64


def copy[T: Scalar](dst: T, src: T, nbytes: u32, bar: u64) -> None:
    """Copy `nbytes` (a multiple of 16) from `src` to the shared `dst`; `bar` completes when they landed (sm_90)."""
    todo()

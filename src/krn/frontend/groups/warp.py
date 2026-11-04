"""The warp: the lockstep group of lanes, 32 on NVIDIA and Apple and 64 on AMD CDNA.

`size()` is a capture-time constant of the target, so one source covers both widths, and masks are always 64-bit
(`Mask`). A `mask` left out means every lane of the warp, and a `width` left out means the whole warp; a smaller power
of two makes sub-warp segments, as in CUDA's `__shfl_xor_sync(mask, x, lane_mask, width)`. Bodies are signature-only:
the lowering per target is in `catalog.py`, and the sub-warp groups of `cg` are built from these functions.
"""

from ..memory import Tensor
from ..numerics import Integer, Scalar, i32, todo, u32
from ..scopes import Mask


def size() -> int:
    """Lanes per warp, a capture-time constant of the target."""
    return todo()


def lane() -> u32:
    """Linear block thread rank modulo the warp size."""
    return todo()


def index() -> u32:
    """Index of this warp inside its block."""
    return todo()


def sync(mask: Mask | None = None) -> None:
    """A barrier for the lanes in `mask`."""
    return todo()


def active_mask() -> Mask:
    """The lanes that are executing this instruction now."""
    return todo()


def reserve(counter: Tensor[i32, int], wants: bool) -> i32:
    """Warp-aggregated append: one atomic per warp hands each wanting lane a slot, and -1 to the others."""
    return todo()


def ballot(predicate: bool, mask: Mask | None = None) -> Mask:
    """A mask with bit `i` set where lane `i` of `mask` has `predicate` true."""
    return todo()


def all(predicate: bool, mask: Mask | None = None) -> bool:
    return todo()


def any(predicate: bool, mask: Mask | None = None) -> bool:
    return todo()


def eq(predicate: bool, mask: Mask | None = None) -> bool:
    """Whether `predicate` has the same value in every lane of `mask`."""
    return todo()


def shfl[T: Scalar](x: T, src: int, width: int | None = None, mask: Mask | None = None) -> T:
    """Read `src % width` within this lane's segment; width defaults to the warp size."""
    return todo()


def shfl_up[T: Scalar](x: T, delta: int, width: int | None = None, mask: Mask | None = None) -> T:
    """The `x` of lane `lane - delta`; a source outside the width segment returns its own `x`."""
    return todo()


def shfl_down[T: Scalar](x: T, delta: int, width: int | None = None, mask: Mask | None = None) -> T:
    """The `x` of lane `lane + delta`; a source outside the width segment returns its own `x`."""
    return todo()


def shfl_xor[T: Scalar](x: T, lane_mask: int, width: int | None = None, mask: Mask | None = None) -> T:
    """Butterfly within width segments; a source outside the segment returns this lane's `x`."""
    return todo()


def match_any[T: Scalar](x: T, mask: Mask | None = None) -> Mask:
    """The lanes of `mask` whose `x` equals the `x` of this lane."""
    return todo()


def match_all[T: Scalar](x: T, mask: Mask | None = None) -> tuple[Mask, bool]:
    """`(mask, True)` when every lane of `mask` has the same `x`, `(0, False)` otherwise."""
    return todo()


def sum[T: Scalar](x: T) -> T:
    """One REDUX on sm_80 and later for integers, a DPP tree on AMD, simd_sum on Apple."""
    return todo()


def exclusive_sum[T: Integer](x: T) -> T:
    return todo()


def argmax[T: Scalar](value: T, index: u32) -> tuple[T, u32]:
    """Largest value and the lowest index that holds it, over the lanes."""
    return todo()


def lanemask_lt() -> Mask:
    """Mask of the lanes below the calling lane."""
    return todo()


def lanemask_le() -> Mask:
    """Mask of the lanes at or below the calling lane."""
    return todo()


def lanemask_eq() -> Mask:
    """Mask of the lanes equal to the calling lane."""
    return todo()


def lanemask_ge() -> Mask:
    """Mask of the lanes at or above the calling lane."""
    return todo()


def lanemask_gt() -> Mask:
    """Mask of the lanes above the calling lane."""
    return todo()

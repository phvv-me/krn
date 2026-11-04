"""Memory views and tiles. A Tile lives in registers or shared memory and its layout is inferred, never written.

A tensor is indexed like a numpy array. Scalar indices read and write one element (the SIMT path), and slices read and
write tiles, so `c[rows, cols] = a[rows, k : k + bk] @ b[k : k + bk, cols]` needs no load or store call. A slice
extent must be a compile-time constant (`lo : lo + C`, or `block.index().x.tile(C)`). Elements past the edge of a tensor
are *invalid lanes*: they read as zero, take part in nothing (a reduction skips them, an elementwise operation keeps
them invalid, a matrix product counts them as zero) and are not stored, so there is no mask to forget. Scalar indexing
is typed up to rank 3 in this sketch.

A bare int axis or parameter is a runtime value; compile-time values come only from literals, target facts and
Const fields.

Shape-carrying classes use classic invariant TypeVars. PEP 695 parameters get an inferred variance, and a covariant
shape lets ty widen a contraction dimension to a union (K = 64 | 32) and accept a mismatch. Invariance makes `mma`
reject mismatched dims in both checkers. Elementwise operations take any shape through a method-scoped `*S`, so
broadcasting stays legal and is checked by the compiler. `int` means a runtime dimension and a Literal pins it.
"""

from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from enum import StrEnum, auto
from types import CapsuleType
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, TypeVarTuple, overload

from krn.data.formats import BlockFormat

from ._ops import TensorOps, TileOps
from .numerics import (
    FloatingPoint,
    Idx,
    Integer,
    Scalar,
    Selector,
    bf16,
    f8e4m3fn,
    f8e5m2,
    f16,
    f32,
    i32,
    todo,
    u8,
)
from .records import AUTO, Auto, Const, struct

if TYPE_CHECKING:
    import numpy as np

T = TypeVar("T", bound=Scalar | bool)
F = TypeVar("F", bound=BlockFormat)
Shape = TypeVarTuple("Shape")


class Tile(TileOps[T, *Shape]):
    """A block-cooperative value. Element access exists only through the operations below."""

    dtype: type[T]

    def __add__[*S](self, other: Tile[T, *S] | T | float) -> Tile[T, *Shape]:
        return todo()

    def __sub__[*S](self, other: Tile[T, *S] | T | float) -> Tile[T, *Shape]:
        return todo()

    def __mul__[*S](self, other: Tile[T, *S] | T | float) -> Tile[T, *Shape]:
        return todo()

    def __truediv__[*S](self, other: Tile[T, *S] | T | float) -> Tile[T, *Shape]:
        return todo()

    def __neg__(self) -> Tile[T, *Shape]:
        return todo()

    def transpose[M, N](self: Tile[T, M, N]) -> Tile[T, N, M]:
        """The transpose of a matrix tile: element (i, j) of the result is element (j, i) of the operand."""
        return todo()

    def __lt__[*S](self, other: Tile[T, *S] | T | float) -> Tile[bool, *Shape]:
        """Elementwise comparisons give bool tiles, which `where` and `&` consume."""
        return todo()

    def __le__[*S](self, other: Tile[T, *S] | T | float) -> Tile[bool, *Shape]:
        return todo()

    def __gt__[*S](self, other: Tile[T, *S] | T | float) -> Tile[bool, *Shape]:
        return todo()

    def __ge__[*S](self, other: Tile[T, *S] | T | float) -> Tile[bool, *Shape]:
        return todo()

    def map(self, fn: Callable[[T], T]) -> Tile[T, *Shape]:
        """Apply a scalar device function (a `@kernel` function or an `@intrinsic`) to every element."""
        return todo()

    def quantize[Q: BlockFormat](self, fmt: Q) -> QTile[Q, *Shape]:
        """Convert to a block-scaled format, computing the scales from the data (dynamic quantization)."""
        return todo()

    def __radd__(self, other: T | float) -> Tile[T, *Shape]:
        return todo()

    def __rsub__(self, other: T | float) -> Tile[T, *Shape]:
        return todo()

    def __rmul__(self, other: T | float) -> Tile[T, *Shape]:
        return todo()

    def __rtruediv__(self, other: T | float) -> Tile[T, *Shape]:
        return todo()

    @overload
    def __matmul__[M, N, K](self: Tile[f16, M, K], other: Tile[f16, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[bf16, M, K], other: Tile[bf16, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f32, M, K], other: Tile[f32, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f16, M, K], other: Tile[bf16, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[bf16, M, K], other: Tile[f16, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f16, M, K], other: Tile[f32, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f32, M, K], other: Tile[f16, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[bf16, M, K], other: Tile[f32, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f32, M, K], other: Tile[bf16, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f8e4m3fn, M, K], other: Tile[f8e4m3fn, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f8e4m3fn, M, K], other: Tile[f8e5m2, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f8e5m2, M, K], other: Tile[f8e4m3fn, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[M, N, K](self: Tile[f8e5m2, M, K], other: Tile[f8e5m2, K, N]) -> Tile[f32, M, N]: ...
    @overload
    def __matmul__[F: FloatingPoint, M, N, K](
        self: Tile[F, M, K], other: Tile[F, K, N]
    ) -> Tile[FloatingPoint, M, N]: ...
    def __matmul__[M, N, K](self, other: Tile[T, K, N]) -> Tile[f32, M, N] | Tile[FloatingPoint, M, N]:
        """Product with the default accumulator: float32 for fp8, f16, bf16 and f32.

        `acc += a @ b` compiles to one multiply-accumulate on the matrix unit of the target. A bare `a @ b` is the same
        product into a fresh float32 tile, so `s = q @ k.transpose()` needs no zeros.
        """
        return todo()

    @overload
    def to[U: Scalar](self, dtype: type[U]) -> Tile[U, *Shape]: ...
    @overload
    def to(self, dtype: Selector) -> Tile[FloatingPoint, *Shape]: ...
    def to[U: Scalar](self, dtype: type[U] | Selector) -> Tile[U, *Shape] | Tile[FloatingPoint, *Shape]:
        """Convert every element. A selector such as `T.fp.min` converts to whatever the target resolves it to."""
        return todo()

    def mma[A: Scalar, M, N, K](
        self: Tile[T, M, K],
        other: Tile[T, K, N],
        acc: Tile[A, M, N],
        *,
        promote_every: int | None = None,
    ) -> Tile[A, M, N]:
        """Matrix multiply-accumulate with a declared accumulator: acc + self @ other.

        The instruction is picked per (operand format, accumulator, target): wgmma or tcgen05 on NVIDIA, MFMA on
        AMD, a cooperative tensor on Apple, an emulated decode-then-multiply where no native form exists.
        `promote_every` moves the partial sum into float32 registers every that many elements of K, which Hopper
        needs for FP8 (a 13-bit fused total) and Blackwell does not.
        """
        return todo()


class QTile(Generic[F, *Shape]):
    """A tile of block-scaled codes. Only `@`, `mma` and `dequantize` consume it."""

    fmt: F

    def __matmul__[M, N, K](self: QTile[F, M, K], other: QTile[F, K, N]) -> Tile[f32, M, N]:
        """Block-scaled product with float32 accumulation. Both operands carry the same format F."""
        return todo()

    def dequantize[U: Scalar](self, dtype: type[U]) -> Tile[U, *Shape]:
        return todo()

    def mma[M, N, K](
        self: QTile[F, M, K],
        other: QTile[F, K, N],
        acc: Tile[f32, M, N],
        *,
        promote_every: int | None = None,
    ) -> Tile[f32, M, N]:
        """Block-scaled multiply-accumulate. Both operands carry the same format F, or the checker rejects it."""
        return todo()


class Tensor(TensorOps[T, *Shape]):
    """A strided view of global memory. Scalar indices are the SIMT path, slices are the tile path."""

    shape: tuple[int, ...]
    dtype: type[T]

    def to_host(self) -> np.ndarray:
        """Copy to host storage after the caller has ordered pending writes."""
        raise NotImplementedError

    @overload
    def __getitem__(self, index: Idx) -> T: ...
    @overload
    def __getitem__(self, index: tuple[Idx, Idx]) -> T: ...
    @overload
    def __getitem__(self, index: tuple[Idx, Idx, Idx]) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Tile[T, *Shape]: ...
    @overload
    def __getitem__(self, index: tuple[Idx, slice] | tuple[slice, Idx]) -> Tile[T, int]: ...
    @overload
    def __getitem__(self, index: tuple[slice, slice]) -> Tile[T, *Shape]: ...
    @overload
    def __getitem__(self, index: tuple[Idx, slice, slice]) -> Tile[T, int, int]: ...
    def __getitem__(self, index: Idx | slice | tuple[Idx | slice, ...]) -> T | Tile[T, int] | Tile[T, *Shape]:
        """Slices keep the axes of the tensor, so a tile of `Tensor[f16, M, K]` is a `Tile[f16, M, K]`."""
        return todo()

    @overload
    def __setitem__(self, index: Idx | tuple[Idx, Idx] | tuple[Idx, Idx, Idx], value: T | int | float) -> None: ...
    @overload
    def __setitem__(self, index: tuple[Idx, slice] | tuple[slice, Idx], value: Tile[T, int]) -> None: ...
    @overload
    def __setitem__(self, index: slice | tuple[slice, slice], value: Tile[T, *Shape]) -> None: ...
    @overload
    def __setitem__(self, index: tuple[Idx, slice, slice], value: Tile[T, int, int]) -> None: ...
    def __setitem__(
        self,
        index: Idx | slice | tuple[Idx | slice, ...],
        value: T | int | float | Tile[T, int] | Tile[T, *Shape],
    ) -> None:
        return todo()


class SupportsDLPack(Protocol):
    """The DLPack exchange standard that torch, jax, numpy and cupy implement, the one place a Protocol is right."""

    def __dlpack__(self, *, stream: int | None = None) -> CapsuleType: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...


def from_dlpack[T: Scalar](source: SupportsDLPack, dtype: type[T]) -> Tensor[T, int, int]:
    """Wrap a framework tensor without copying. The dtype is checked against the exporter's."""
    return todo()


type Vector[T: Scalar | bool, N = int] = Tensor[T, N]
type Matrix[T: Scalar, R = int, C = int] = Tensor[T, R, C]
type Tensor3[T: Scalar, A = int, B = int, C = int] = Tensor[T, A, B, C]


@struct
class QMatrix[F: BlockFormat]:
    """Packed codes and scales of a matrix whose blocks run along its contraction axis."""

    fmt: Const[F]
    shape: tuple[int, int]
    codes: Tensor[u8, int, int]
    scales: Tensor[u8, int, int]
    tensor_scale: Tensor[f32, int]

    def __getitem__(self, index: tuple[slice, slice]) -> QTile[F, int, int]:
        """The codes and scales of a tile, not decoded: only `@`, `mma` and `dequantize` read them."""
        return todo()


class Space(StrEnum):
    """Where a tile lives. The default is the cheapest place the target offers; a kernel names another on purpose."""

    REGISTERS = auto()
    SHARED = auto()
    TENSOR_MEMORY = auto()  # per-SM accumulator memory of Blackwell datacenter parts, a capture error elsewhere


@overload
def zeros[U: Scalar, *S](dtype: type[U], *extent: int, space: Space = Space.REGISTERS) -> Tile[U, *S]: ...
@overload
def zeros[*S](dtype: Selector, *extent: int, space: Space = Space.REGISTERS) -> Tile[FloatingPoint, *S]: ...
def zeros[U: Scalar, *S](
    dtype: type[U] | Selector, *extent: int, space: Space = Space.REGISTERS
) -> Tile[U, *S] | Tile[FloatingPoint, *S]:
    """A tile of zeros. The dtype may be a type or a selector such as `acc(f16)` or `fp.min`."""
    return todo()


@overload
def full[U: Scalar, *S](dtype: type[U], value: float, *extent: int) -> Tile[U, *S]: ...
@overload
def full[*S](dtype: Selector, value: float, *extent: int) -> Tile[FloatingPoint, *S]: ...
def full[U: Scalar, *S](dtype: type[U] | Selector, value: float, *extent: int) -> Tile[U, *S] | Tile[FloatingPoint, *S]:
    return todo()


class Pending[T]:
    """The result of an asynchronous copy or multiply that has been issued and not yet awaited.

    Inside an `async def` kernel, `await pending` is the wait that makes the data usable: `cp.async.wait_group` on
    Ampere and Ada, an mbarrier wait on Hopper and Blackwell (TMA), a `s_waitcnt` on AMD, a threadgroup barrier on
    Apple, nothing on a CPU. The compiler sinks the await as late as the data dependences allow, which is the
    software pipeline.
    """

    def __await__(self) -> Generator[None, None, T]:
        return todo()


class Ring[T: Scalar, *S]:
    """A ring of tile-sized slots in shared memory that connects a producer role to a consumer role.

    `await ring.put(tile)` waits for a free slot and issues the copy into it; `await ring.get()` waits for a full slot
    and returns its tile. On Hopper and Blackwell the two waits are mbarrier phases (the "empty" and "full" barriers
    of a TMA pipeline), on AMD and Ampere they are `s_waitcnt` and `cp.async.wait_group` counters, on a CPU a plain
    queue of stack tiles. The programmer writes the producer and the consumer; the phase parity bookkeeping that Gluon,
    TLX and Mosaic GPU leave in user code is the ring's.
    """

    async def put(self, tile: Tile[T, *S]) -> None:
        return todo()

    async def get(self) -> Tile[T, *S]:
        return todo()


def ring[T: Scalar](dtype: type[T], stages: int | Auto, rows: int, cols: int) -> Ring[T, int, int]:
    """Allocate a ring of `stages` slots of `rows` x `cols` elements in shared memory (`AUTO`: from the target)."""
    return todo()


def copy[T: Scalar, *S](source: Tile[T, *S]) -> Pending[Tile[T, *S]]:
    """Issue the load of a slice without waiting for it: `a_next = copy(a[rows, ks])`, then `await a_next`."""
    return todo()


def maximum[T: Scalar, *S](a: Tile[T, *S], b: Tile[T, *S]) -> Tile[T, *S]:
    """Elementwise maximum. Invalid lanes (outside the tensor) stay invalid and never win."""
    return todo()


def minimum[T: Scalar, *S](a: Tile[T, *S], b: Tile[T, *S]) -> Tile[T, *S]:
    return todo()


@dataclass(frozen=True)
class Uniform:
    """Every warp loads and computes: a multi-stage ring of asynchronous copies (cp.async, direct-to-LDS)."""


@dataclass(frozen=True)
class WarpSpecialized:
    """Producer warps move tiles, consumer warps multiply: TMA with barriers on sm_90a and sm_100a."""

    producers: int = 1
    consumers: int = 2


@dataclass(frozen=True)
class PingPong:
    """Two groups of waves alternate between loading and multiplying, which hides latency where there is no TMA."""

    groups: int = 2


type Schedule = Uniform | WarpSpecialized | PingPong

UNIFORM = Uniform()


def where[*S, E: Scalar](
    condition: Tile[bool, *S], x: Tile[E, *S] | int | float, y: Tile[E, *S] | int | float
) -> Tile[E, *S]:
    """Elementwise `x` where the condition holds and `y` elsewhere; the operands broadcast like `+`."""
    return todo()


@overload
def arange(extent: int, axis: int = 0, *, rank: Literal[1] = 1, start: int | Integer = 0) -> Tile[i32, int]: ...
@overload
def arange(extent: int, axis: int = 0, *, rank: Literal[2], start: int | Integer = 0) -> Tile[i32, int, int]: ...
def arange(
    extent: int, axis: int = 0, *, rank: int = 1, start: int | Integer = 0
) -> Tile[i32, int] | Tile[i32, int, int]:
    """An int32 tile of rank `rank` that is `extent` long along `axis` and 1 elsewhere, holding `start` plus the index
    along `axis`. `arange(bq, 0, rank=2, start=q0)` is a column of query positions, `arange(bk, 1, rank=2)` a row of key
    positions, and their comparison a causal mask."""
    return todo()


def chunks(extent: int, tile: int) -> Iterator[slice]:
    """The tiles of an axis in order, as slices, with no overlap of loads and compute: `for cs in chunks(N, bn)`."""
    return todo()


def pipelined(
    extent: int, tile: int, *, stages: int | Auto = AUTO, schedule: Schedule | Auto = AUTO
) -> Iterator[slice]:
    """Software-pipeline a loop over the tiles of an axis, yielding each tile as a slice: `for ks in pipelined(K, bk)`.

    Loads of tile i+stages-1 overlap the compute of tile i. `AUTO` (the default) lets the compiler choose the depth
    from the shared memory of the target and the tile sizes in use, and the schedule from what the target offers
    (`Uniform` where there is no bulk copy, `WarpSpecialized` where there is). A value you pass is a request, not a
    hint: the compiler rejects a schedule the target cannot run (`WarpSpecialized` needs `Feature.BULK_COPY`).
    """
    return todo()

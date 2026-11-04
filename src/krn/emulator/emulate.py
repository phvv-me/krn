"""A Python emulator of the typed IR, for looking inside a kernel with the tools every Python programmer has.

The kernel's IR becomes Python source (one generator per thread), parsed, and given the line numbers of the user's file
before it is compiled under the user's file name. So `breakpoint()` in the kernel body is Python's own: `pdb` lists
the kernel as written, `p` shows its locals (`_block_id` and `_thread_id` name the thread), `n` steps by source line, a
traceback names the line, and `PYTHONBREAKPOINT=krn.compiler.debug.record` records the locals for an agent instead of
stopping. Memory is checked the way `compute-sanitizer --tool memcheck` would: an out-of-bounds read or write raises
at the line that did it, and `assert` raises with the block and the thread.

Threads of a block run as coroutines that switch only at a barrier or a warp collective, so those are real:
`block.sync()` waits for every live thread, `warp.shfl_xor(...)` and the other collectives exchange the lanes'
values, and atomics are atomic because nothing else runs between their two halves. A collective that some lanes skip
and others wait on raises `Deadlock` naming the threads and their lines, as `compute-sanitizer --tool synccheck`
reports it.

It is a debugging tool, not a backend: it is slow (a Python generator per thread), it follows the semantics of the IR
and not the rounding of a GPU. Kernels with `grid.sync()` schedule all blocks together, keeping each block's shared
memory separate. Differential tests run kernels through it and through the C backend and compare.
"""

import ast
import inspect
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from itertools import product
from pathlib import Path
from types import GeneratorType

import numpy as np

from krn import Kernel as KernelFunction
from krn import catalog
from krn.compiler import narrow
from krn.compiler.capture import INTRINSICS, CallSpan, capture
from krn.compiler.dtypes import BF16, BOOL, I8X4, I16X2, PY_FLOAT, PY_INT, U8X4, U16X2, Arithmetic, Dtype, Kind
from krn.compiler.emit import dimensions
from krn.compiler.errors import CaptureError
from krn.compiler.ir import (
    Accumulate,
    Arange,
    Assert,
    Assign,
    Atomic,
    Binary,
    Breakpoint,
    Cast,
    Const,
    Dim,
    Effect,
    Expr,
    Fn,
    For,
    If,
    Intrinsic,
    Kernel,
    Load,
    Math,
    Op,
    Param,
    ParamType,
    Print,
    Reduce,
    Reduction,
    Select,
    Stmt,
    Store,
    StructType,
    Sync,
    TensorRef,
    TensorType,
    TileAssign,
    TileLoad,
    TileStore,
    Transpose,
    Unary,
    Var,
    While,
    Zeros,
)
from krn.compiler.matrix import FP8, FP8_TARGETS
from krn.compiler.validity import can_be_invalid, children, masked_tiles, walk
from krn.facts.targets import Target
from krn.frontend.atomic import AtomicOp
from krn.frontend.geometry import Geometry
from krn.runtime.cpu import NUMPY
from krn.runtime.cpu import Value as Argument
from krn.runtime.runtime import Argument as HostArgument
from krn.runtime.runtime import _consts, _dtypes, _structs, _value

log = logging.getLogger(__name__)

NUMPY_DTYPES = {
    **NUMPY,
    **dict.fromkeys(narrow.FORMATS, np.dtype("uint8")),
    **dict.fromkeys(narrow.FNUZ_DTYPES, np.dtype("uint8")),
    BOOL: np.dtype("bool"),
    PY_INT: np.dtype("int64"),
    PY_FLOAT: np.dtype("float64"),
    I8X4: np.dtype("uint32"),
    U8X4: np.dtype("uint32"),
    I16X2: np.dtype("uint32"),
    U16X2: np.dtype("uint32"),
}
NARROW = {narrow.NAMES[fmt]: fmt for fmt in narrow.FORMATS.values()}  # by the name the emitted text carries
AXES = ("x", "y", "z")
GEOMETRY = ("_bx", "_by", "_bz", "_tx", "_ty", "_tz", "_gx", "_gy", "_gz", "_bdx", "_bdy", "_bdz")

type Cells = tuple[int, ...]
type Value = bool | int | float | np.generic


@dataclass(frozen=True, slots=True)
class BlockRequest:
    """A thread waits at a barrier of its block; `count`, `all` and `any` bring a vote."""

    op: str
    vote: bool = False


@dataclass(frozen=True, slots=True)
class WarpRequest:
    """A lane calls a warp collective on the lanes of `mask`."""

    op: str
    mask: int
    operands: tuple[Value, ...]
    site: str = ""


type Request = BlockRequest | WarpRequest


class Deadlock(RuntimeError):
    """Threads wait for a collective that the others will never reach."""


def np_dtype(dtype: Dtype) -> np.dtype:
    return np.dtype("float32") if dtype == BF16 else NUMPY_DTYPES[dtype]


def _bfloat16(value: np.ndarray) -> np.ndarray:
    """Round float32 lanes to bfloat16, ties to even, retaining their float32 host storage and quieting NaNs."""
    array = np.asarray(value, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    quiet = (bits | np.uint32(0x00400000)) & np.uint32(0xFFFF0000)
    return np.where(np.isnan(array), quiet, rounded).view(np.float32)


def narrow_encode(name: str, x: Value) -> np.generic | np.ndarray:
    """The code of float32 `x` in the narrow format `name`: the table reference, which the compiled text matches."""
    return narrow.reference_encode(NARROW[name], x)[()]


def narrow_decode(name: str, codes: Value) -> np.generic | np.ndarray:
    return narrow.reference_decode(NARROW[name], codes)


def wrap(x: int, bits: int, signed: bool) -> int:
    """`x` in two's complement of `bits` bits."""
    x &= (1 << bits) - 1
    return x - (1 << bits) if signed and x >> (bits - 1) else x


def divide(a: int, b: int) -> int:
    """Integer division toward zero; a zero divisor gives zero on every backend."""
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def remainder(a: int, b: int) -> int:
    """C remainder: the sign of the dividend."""
    return a - b * divide(a, b) if b else 0


def shift(a: int, count: int, bits: int, right: bool) -> int:
    """A defined shift evaluates its operands once, including atomic expressions."""
    if not 0 <= count < bits:
        return 0
    return a >> count if right else a << count


def integer_tile(
    a: np.ndarray | np.generic, b: np.ndarray | np.generic, operation: str, bits: int, signed: bool
) -> np.ndarray:
    left = np.asarray(a, dtype=f"{'int' if signed else 'uint'}{bits}")
    right = np.asarray(b, dtype=left.dtype) if operation in ("div", "mod") else np.asarray(b)
    left, right = np.broadcast_arrays(left, right)
    out = np.zeros_like(left)
    if operation in ("div", "mod"):
        function = np.floor_divide if operation == "div" else np.remainder
        return function(left, right, where=right != 0, out=out)
    function = np.left_shift if operation == "shl" else np.right_shift
    count = np.asarray(np.clip(right, 0, bits - 1), dtype=left.dtype)
    return function(left, count, where=(right >= 0) & (right < bits), out=out)


def read(name: str, array: np.ndarray, *cell: int) -> Value:
    """A checked element read; an index outside the array raises at the line that made it."""
    if len(cell) != array.ndim or any(not 0 <= c < n for c, n in zip(cell, array.shape, strict=True)):
        raise IndexError(f"read of `{name}`{list(cell)} is outside shape {array.shape}")
    value = array[cell]
    return int(value) if array.dtype.kind in "iu" else bool(value) if array.dtype.kind == "b" else value


def write(name: str, array: np.ndarray, value: Value, *cell: int) -> None:
    if len(cell) != array.ndim or any(not 0 <= c < n for c, n in zip(cell, array.shape, strict=True)):
        raise IndexError(f"write of `{name}`{list(cell)} is outside shape {array.shape}")
    array[cell] = value


def async_copy(
    dst_name: str,
    dst: np.ndarray,
    dst_cell: Cells,
    src_name: str,
    src: np.ndarray,
    src_cell: Cells,
    count: int,
) -> None:
    """Synchronous refinement of a contiguous copy, with the usual checked element accesses."""
    write(dst_name, dst, read(src_name, src, *src_cell), *dst_cell)
    src_start, dst_start = (
        int(np.ravel_multi_index(src_cell, src.shape)),
        int(np.ravel_multi_index(dst_cell, dst.shape)),
    )
    for offset in range(1, count):
        value = read(src_name, src.reshape(-1), src_start + offset)
        write(dst_name, dst.reshape(-1), value, dst_start + offset)


def regions(
    shape: tuple[int, ...], starts: Cells, extents: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """The part of a tile that overlaps the array, as slices of the array and of the tile."""
    lows = [max(s, 0) for s in starts]
    highs = [max(min(s + e, n), lo) for s, e, n, lo in zip(starts, extents, shape, lows, strict=True)]
    inside = tuple(slice(lo, hi) for lo, hi in zip(lows, highs, strict=True))
    local = tuple(slice(lo - s, hi - s) for s, lo, hi in zip(starts, lows, highs, strict=True))
    return inside, local


def load_tile(array: np.ndarray, starts: Cells, extents: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """A tile read; elements outside the array read as zero (`tile_valid` says which ones those are)."""
    tile = np.zeros(extents, dtype)
    inside, local = regions(array.shape, starts, extents)
    tile[local] = array[inside]
    return tile


def tile_valid(array: np.ndarray, starts: Cells, extents: tuple[int, ...]) -> np.ndarray:
    """Which lanes of a tile read at `starts` lie inside the array."""
    valid = np.zeros(extents, bool)
    valid[regions(array.shape, starts, extents)[1]] = True
    return valid


def store_tile(
    array: np.ndarray, starts: Cells, extents: tuple[int, ...], tile: np.ndarray, valid: np.ndarray | None = None
) -> None:
    """A tile write of `tile` at `extents`, which adds the squeezed axes back; elements outside the array and invalid
    lanes are skipped."""
    values = tile.reshape(extents)
    inside, local = regions(array.shape, starts, extents)
    if valid is None:
        array[inside] = values[local]
    else:
        keep = np.broadcast_to(valid, tile.shape).reshape(extents)[local]
        target = array[inside]
        target[keep] = values[local][keep]


def initial(op: Reduction, dtype: np.dtype) -> int | float:
    """What a reduction returns for a lane with nothing valid under it: the identity of `op`."""
    low, high = (-np.inf, np.inf) if dtype.kind == "f" else (np.iinfo(dtype).min, np.iinfo(dtype).max)
    return {Reduction.MAX: low, Reduction.MIN: high, Reduction.SUM: 0, Reduction.PROD: 1}[op]


def reduce_valid(op: Reduction, x: np.ndarray, valid: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    """`op` over the valid lanes of `x` only; a mean divides by their number, so none gives NaN without a warning."""
    where = np.broadcast_to(valid, x.shape)
    if op is Reduction.MEAN:
        total = np.sum(x, axis=axis, keepdims=keepdims, where=where, initial=0)
        count = np.sum(where, axis=axis, keepdims=keepdims).astype(x.dtype)
        with np.errstate(divide="ignore", invalid="ignore"):
            return total / count
    return REDUCE[op](x, axis=axis, keepdims=keepdims, where=where, initial=initial(op, x.dtype))


def mean(x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    """The mean of every lane along `axis`, all of them valid."""
    return np.sum(x, axis=axis, keepdims=keepdims) / x.shape[axis]


def any_valid(valid: np.ndarray, shape: tuple[int, ...], axis: int, keepdims: bool) -> np.ndarray:
    """Whether any lane along `axis` is valid: the validity of a reduction."""
    return np.broadcast_to(valid, shape).any(axis=axis, keepdims=keepdims)


def accumulate(
    acc: np.ndarray,
    lhs: np.ndarray,
    rhs: np.ndarray,
    lhs_valid: np.ndarray | None,
    rhs_valid: np.ndarray | None,
    acc_valid: np.ndarray | None,
) -> None:
    """`acc += lhs @ rhs` as the scalar fallback computes it: one fused multiply-add per product, in k order, with
    invalid lanes counted as zero; a row or column with no valid lane stays invalid."""
    left = lhs if lhs_valid is None else np.where(lhs_valid, lhs, 0)
    right = rhs if rhs_valid is None else np.where(rhs_valid, rhs, 0)
    left, right = left.astype(np.float32), right.astype(np.float32)
    for k in range(left.shape[1]):
        acc[...] = fused(left[:, k, None], right[None, k, :], acc)
    if acc_valid is not None:
        if lhs_valid is not None:
            acc_valid &= any_valid(lhs_valid, lhs.shape, 1, True)
        if rhs_valid is not None:
            acc_valid &= any_valid(rhs_valid, rhs.shape, 0, True)


def plain(value: Value) -> int | float:
    """A Python number for `value`, which may be a numpy scalar."""
    return value.item() if isinstance(value, np.generic) else value


def atomic(op: AtomicOp, name: str, array: np.ndarray, value: Value, compare: Value | None, *cell: int) -> Value:
    """Apply `op` to one element and return the old value."""
    old = read(name, array, *cell)
    swapped = value if old == compare else old
    result = swapped if op is AtomicOp.CAS else op.apply(plain(old), plain(value))
    if array.dtype.kind in "iu":
        result = wrap(int(result), array.dtype.itemsize * 8, array.dtype.kind == "i")
    write(name, array, result, *cell)
    return old


def show(value: Value) -> str:
    return f"{float(value):g}" if isinstance(value, np.floating) else str(value)


def fused(
    a: np.ndarray | np.generic, b: np.ndarray | np.generic, c: np.ndarray | np.generic
) -> np.ndarray | np.generic:
    """A single rounding in the input dtype, including for scalar numpy values.

    Finite float32 operands take a vector route that is exact: the product is exact in float64, and TwoSum leaves the
    rounding error of the sum, which decides a round-to-odd float64 result. Rounding that to float32 is the single
    rounding, since 53 >= 24 + 2. Other operands use the reference rounding one element at a time.
    """
    dtype = np.asarray(a).dtype
    if dtype == np.float32 and all(np.isfinite(np.asarray(v)).all() for v in (a, b, c)):
        product = np.asarray(a, np.float64) * np.asarray(b, np.float64)
        total = product + np.asarray(c, np.float64)
        virtual = total - product
        error = (product - (total - virtual)) + (np.asarray(c, np.float64) - virtual)
        odd = (total.view(np.int64) & 1).astype(bool)
        total = np.where((error != 0) & ~odd, np.nextafter(total, np.copysign(np.inf, error)), total)
        with np.errstate(over="ignore"):  # a result beyond the float32 range rounds to infinity, as IEEE says
            return total.astype(dtype)[()]
    reference = catalog.BY_NAME["fma"].reference
    assert reference is not None
    function = np.vectorize(lambda x, y, z: reference(x, y, z, bits=dtype.itemsize * 8, kind="f"), otypes=[dtype])
    result = function(a, b, c)
    return result[()] if result.ndim == 0 else result


ERF = np.vectorize(math.erf, otypes=[np.float64])
MATH: dict[Fn, Callable[..., np.ndarray | np.generic]] = {
    Fn.EXP: np.exp, Fn.EXP2: np.exp2, Fn.EXP10: lambda x: np.power(np.asarray(10, x.dtype), x), Fn.EXPM1: np.expm1,
    Fn.LOG: np.log, Fn.LOG2: np.log2, Fn.LOG10: np.log10, Fn.LOG1P: np.log1p, Fn.SQRT: np.sqrt, Fn.CBRT: np.cbrt,
    Fn.RSQRT: lambda x: 1 / np.sqrt(x), Fn.RECIPROCAL: lambda x: 1 / x, Fn.SIN: np.sin, Fn.COS: np.cos,
    Fn.TANH: np.tanh,
    Fn.SIGMOID: lambda x: 1 / (1 + np.exp(-x)), Fn.FLOOR: np.floor, Fn.CEIL: np.ceil, Fn.ROUND: np.rint,
    Fn.TRUNC: np.trunc, Fn.ABS: np.abs, Fn.RELU: lambda x: np.maximum(x, 0), Fn.SILU: lambda x: x / (1 + np.exp(-x)),
    Fn.SOFTPLUS: lambda x: np.logaddexp(0, x), Fn.MAXIMUM: np.maximum, Fn.MINIMUM: np.minimum, Fn.POW: np.power,
    Fn.CLIP: np.clip, Fn.FMA: fused,
    Fn.ERF: lambda x: ERF(x).astype(np.asarray(x).dtype),
    Fn.GELU: lambda x: 0.5 * x * (1 + ERF(x * 0.7071067811865476)).astype(x.dtype),
}  # fmt: skip
REDUCE: dict[Reduction, Callable[..., np.ndarray]] = {
    Reduction.MAX: np.max,
    Reduction.MIN: np.min,
    Reduction.SUM: np.sum,
    Reduction.PROD: np.prod,
    Reduction.MEAN: mean,
}


def intrinsic(name: str, bits: int, kind: str, result_bits: int, result_kind: str, *operands: Value) -> Value:
    """Run the standard-library oracle, rounding only at the IR result boundary."""
    reference = catalog.BY_NAME[name].reference
    assert reference is not None
    numbers = [float(x) if isinstance(x, (float, np.floating)) else int(x) for x in operands]
    result = reference(*numbers, bits=bits, kind=kind)
    assert not isinstance(result, tuple)
    if result_kind == "float":
        if result_bits == 16:
            return np.float32(result) if kind == "bf" else np.float16(result)
        return np.float32(result) if result_bits == 32 else np.float64(result)
    if result_kind == "bool":
        return bool(result)
    return wrap(int(result), result_bits, result_kind == "sint")


def arange_tile(shape: tuple[int, ...], axis: int, start: int) -> np.ndarray:
    """The integers `start` onward along `axis` of a tile of `shape`, the same along every other axis."""
    ramp = np.arange(start, start + shape[axis], dtype=np.int32)
    return np.broadcast_to(ramp.reshape([-1 if a == axis else 1 for a in range(len(shape))]), shape).copy()


HELPERS: dict[str, object] = {
    "np": np, "time": time, "wrap": wrap, "intrinsic": intrinsic, "divide": divide, "remainder": remainder,
    "arange_tile": arange_tile,
    "read": read, "write": write, "async_copy": async_copy, "shift": shift, "integer_tile": integer_tile,
    "load_tile": load_tile, "store_tile": store_tile, "tile_valid": tile_valid, "reduce_valid": reduce_valid,
    "any_valid": any_valid, "accumulate": accumulate, "atomic": atomic, "show": show, "MATH": MATH, "REDUCE": REDUCE,
    "AtomicOp": AtomicOp, "Fn": Fn, "Reduction": Reduction, "BlockRequest": BlockRequest, "WarpRequest": WarpRequest,
    "bfloat16": _bfloat16, "narrow_encode": narrow_encode, "narrow_decode": narrow_decode,
}  # fmt: skip


def leaf(path: tuple[str, ...]) -> str:
    return "_".join(path)


class PythonEmitter:
    """IR to Python source. Every emitted line remembers the line of the user's file it came from."""

    def __init__(self, kernel: Kernel, warp: int) -> None:
        self.kernel, self.warp = kernel, warp
        self.masked = masked_tiles(kernel)
        self.lines: list[tuple[str, int]] = []
        self.depth, self.origin = 1, kernel.span.line

    def emit(self, parameters: Sequence[str]) -> tuple[str, list[int]]:
        self.lines.append((f"def {self.kernel.name}({', '.join(parameters)}):", self.kernel.span.line))
        self.put("_block_id, _thread_id = (_bx, _by, _bz), (_tx, _ty, _tz)")
        self.put("if False:\n        yield")  # the function is a generator even when the kernel never waits
        self.body(self.kernel.body)
        text = "\n".join(line for line, _ in self.lines)
        return text, [origin for line, origin in self.lines for _ in line.split("\n")]

    def put(self, text: str) -> None:
        self.lines.append(("    " * self.depth + text, self.origin))

    def body(self, statements: Sequence[Stmt]) -> None:
        before = len(self.lines)
        for statement in statements:
            self.origin = statement.span.line
            getattr(self, f"stmt_{type(statement).__name__}")(statement)
        if len(self.lines) == before:
            self.put("pass")

    def nested(self, header: str, statements: Sequence[Stmt], origin: int) -> None:
        self.origin = origin
        self.put(header)
        self.depth += 1
        self.body(statements)
        self.depth -= 1

    def stmt_Assign(self, s: Assign) -> None:
        self.put(f"{s.var.name} = {self.val(s.value)}")

    def stmt_Store(self, s: Store) -> None:
        cells = ", ".join(self.val(i) for i in s.index)
        self.put(f"write({leaf(s.tensor.path)!r}, {leaf(s.tensor.path)}, {self.val(s.value)}, {cells})")

    def stmt_TileAssign(self, s: TileAssign) -> None:
        self.put(f"{s.var.name}[...] = {self.val(s.value)}")
        if s.var.name in self.masked:
            self.put(f"{s.var.name}__v[...] = {self.valid(s.value) or 'True'}")

    def stmt_TileStore(self, s: TileStore) -> None:
        starts = ", ".join(f"int({self.val(sl.start)})" for sl in s.slices)
        extents, value, valid = tuple(sl.extent for sl in s.slices), self.val(s.value), self.valid(s.value)
        self.put(f"store_tile({leaf(s.tensor.path)}, ({starts},), {extents}, {value}, {valid})")

    def stmt_Accumulate(self, s: Accumulate) -> None:
        acc_valid = f"{s.acc.name}__v" if s.acc.name in self.masked else None
        operands = f"{s.acc.name}, {self.val(s.lhs)}, {self.val(s.rhs)}, {self.valid(s.lhs)}, {self.valid(s.rhs)}"
        self.put(f"accumulate({operands}, {acc_valid})")

    def valid(self, e: Expr) -> str | None:
        """A numpy boolean expression for the valid lanes of the tile `e`; None when every lane is (see `validity`)."""
        if not e.shape or not can_be_invalid(e, self.masked):
            return None
        return getattr(self, f"valid_{type(e).__name__}", self.valid_elementwise)(e)

    def valid_elementwise(self, e: Expr) -> str | None:
        parts = [part for child in children(e) if (part := self.valid(child))]
        return f"({' & '.join(parts)})" if parts else None

    def valid_TileLoad(self, e: TileLoad) -> str:
        starts = ", ".join(f"int({self.val(sl.start)})" for sl in e.slices)
        extents = tuple(sl.extent for sl in e.slices)
        return f"tile_valid({leaf(e.tensor.path)}, ({starts},), {extents}).reshape({e.shape})"

    def valid_Transpose(self, e: Transpose) -> str:
        return f"np.swapaxes({self.valid(e.operand)}, 0, 1)"

    def valid_Var(self, e: Var) -> str:
        return f"{e.name}__v"

    def valid_Reduce(self, e: Reduce) -> str:
        return f"any_valid({self.valid(e.operand)}, {e.operand.shape}, {e.axis}, {e.keepdims})"

    def stmt_For(self, s: For) -> None:
        self.nested(
            f"for {s.var.name} in range({self.val(s.lo)}, {self.val(s.hi)}, {s.step}):",
            s.body,
            s.span.line,
        )

    def stmt_While(self, s: While) -> None:
        self.nested(f"while {self.val(s.cond)}:", s.body, s.span.line)

    def stmt_If(self, s: If) -> None:
        self.nested(f"if {self.val(s.cond)}:", s.then, s.span.line)
        if s.orelse:
            self.nested("else:", s.orelse, s.span.line)

    def stmt_Sync(self, s: Sync) -> None:
        self.put('yield BlockRequest("sync")')

    def stmt_Effect(self, s: Effect) -> None:
        self.put(self.val(s.value))

    def stmt_Breakpoint(self, s: Breakpoint) -> None:
        self.put("breakpoint()")

    def stmt_Print(self, s: Print) -> None:
        self.put(f"print({', '.join(repr(p) if isinstance(p, str) else f'show({self.val(p)})' for p in s.parts)})")

    def stmt_Assert(self, s: Assert) -> None:
        self.put(f"if not ({self.val(s.cond)}):")
        self.put(f"    raise AssertionError({s.text!r} + f' (block {{_bx}}, thread {{_tx}})')")

    def val(self, e: Expr) -> str:
        return getattr(self, f"val_{type(e).__name__}")(e)

    def val_Const(self, e: Const) -> str:
        if e.dtype == BOOL:
            return repr(bool(e.value))
        if e.dtype.is_float:
            return f"np.{np_dtype(e.dtype).name}({e.value!r})"
        return repr(e.value)

    def val_Var(self, e: Var) -> str:
        return e.name

    def val_Param(self, e: Param) -> str:
        return leaf(e.path)

    def val_Dim(self, e: Dim) -> str:
        return leaf(e.tensor.dim(e.axis))

    def val_Load(self, e: Load) -> str:
        return f"read({leaf(e.tensor.path)!r}, {leaf(e.tensor.path)}, {', '.join(self.val(i) for i in e.index)})"

    def val_Atomic(self, e: Atomic) -> str:
        cells = ", ".join(self.val(i) for i in e.index)
        compare = self.val(e.compare) if e.compare is not None else "None"
        name = leaf(e.tensor.path)
        return f"atomic(AtomicOp.{e.op.name}, {name!r}, {name}, {self.val(e.value)}, {compare}, {cells})"

    def val_Intrinsic(self, e: Intrinsic) -> str:
        name, operands = e.name, [self.val(a) for a in e.args]
        if name.startswith("cluster."):
            raise NotImplementedError("the emulator runs one block at a time and clusters need a cluster launch")
        if (scoped := catalog.BY_NAME[name].atomic) is not None:
            target = e.args[0]
            assert isinstance(target, Load)
            cells = ", ".join(self.val(i) for i in target.index)
            tensor = leaf(target.tensor.path)
            compare = operands[1] if scoped[0] is AtomicOp.CAS else "None"
            return f"atomic(AtomicOp.{scoped[0].name}, {tensor!r}, {tensor}, {operands[-1]}, {compare}, {cells})"
        if name.startswith("cache.st"):
            target = e.args[0]
            assert isinstance(target, Load)
            tensor = leaf(target.tensor.path)
            cells = ", ".join(self.val(i) for i in target.index)
            return f"write({tensor!r}, {tensor}, {operands[1]}, {cells})"
        if name.startswith("cache.prefetch_"):
            return "None"
        if name == "async_copy.copy":
            dst, src, count = e.args
            assert isinstance(dst, Load) and isinstance(src, Load)
            dst_name, src_name = leaf(dst.tensor.path), leaf(src.tensor.path)
            dst_cell = "(" + "".join(f"{self.val(i)}, " for i in dst.index) + ")"
            src_cell = "(" + "".join(f"{self.val(i)}, " for i in src.index) + ")"
            return (
                f"async_copy({dst_name!r}, {dst_name}, {dst_cell}, {src_name!r}, {src_name}, {src_cell}, "
                f"{self.val(count)})"
            )
        if name in ("async_copy.commit", "async_copy.wait"):
            return "None"
        if catalog.BY_NAME[name].reference is not None and not name.startswith("warp."):
            operand = e.args[0].dtype
            kind = {Kind.UINT: "u", Kind.SINT: "s", Kind.FLOAT: "f"}.get(operand.kind, "u")
            kind = "bf" if operand == BF16 else kind
            return (
                f"intrinsic({name!r}, {operand.bits}, {kind!r}, {e.dtype.bits}, {e.dtype.kind.value!r}, "
                f"{', '.join(operands)})"
            )
        scope, _, rest = name.partition(".")
        if rest[-2:-1] == "." and rest[-1] in AXES:
            field_, axis = rest[:-2], rest[-1]
            return {
                "thread.index": "_t",
                "block.index": "_b",
                "block.dim": "_bd",
                "grid.dim": "_g",
            }[f"{scope}.{field_}"] + axis
        spec = INTRINSICS.get(name)
        full = (1 << self.warp) - 1
        thread_rank = "(_tx + _ty * _bdx + _tz * _bdx * _bdy)"
        block_rank = "(_bx + _by * _gx + _bz * _gx * _gy)"
        block_threads = "(_bdx * _bdy * _bdz)"
        match name:
            case "block.thread_rank":
                return thread_rank
            case "block.num_threads":
                return block_threads
            case "grid.block_rank":
                return block_rank
            case "grid.num_blocks":
                return "(_gx * _gy * _gz)"
            case "grid.thread_rank":
                return f"({block_rank} * {block_threads} + {thread_rank})"
            case "grid.num_threads":
                return f"(_gx * _gy * _gz * {block_threads})"
            case "warp.lane":
                return f"({thread_rank} % {self.warp})"
            case "warp.index":
                return f"({thread_rank} // {self.warp})"
            case "warp.lanemask_lt" | "warp.lanemask_le" | "warp.lanemask_eq" | "warp.lanemask_ge" | "warp.lanemask_gt":
                lane = f"({thread_rank} % {self.warp})"
                below, through = f"((1 << {lane}) - 1)", f"((2 << {lane}) - 1)"
                return {
                    "lt": below,
                    "le": through,
                    "eq": f"(1 << {lane})",
                    "ge": f"({full} ^ {below})",
                    "gt": f"({full} ^ {through})",
                }[rest.removeprefix("lanemask_")]
            case "block.sync" | "block.count" | "block.all" | "block.any":
                vote = f", bool({operands[0]})" if operands else ""
                return f"(yield BlockRequest({rest!r}{vote}))"
            case "grid.sync":
                return '(yield BlockRequest("grid_sync"))'
            case "popcount" | "reverse_bits" | "leading_zeros" | "first_set":
                return self.bits(name, e, operands[0])
            case "thread.clock" | "thread.clock64" | "thread.globaltimer":
                return f"(time.perf_counter_ns() & {(1 << e.dtype.bits) - 1})"
            case "sleep":
                return "None"
        if name.startswith("thread.fence_"):
            return "None"
        if spec is not None and scope == "warp":
            mask = operands[spec.mask] if spec.mask is not None and len(operands) > spec.mask else repr(full)
            data = [o for i, o in enumerate(operands) if i != spec.mask]
            site = (
                str((str(e.span), *(str(caller) for caller in e.span.callers)))
                if isinstance(e.span, CallSpan)
                else str(e.span)
            )
            request = f"(yield WarpRequest({rest!r}, {mask}, ({''.join(f'{d}, ' for d in data)}), {site!r}))"
            return f"wrap({request}, 32, {e.dtype.kind is Kind.SINT})" if rest.startswith("reduce_") else request
        raise NotImplementedError(f"the emulator has no semantics for `{name}`")

    def bits(self, name: str, e: Intrinsic, x: str) -> str:
        width, signed = e.args[0].dtype.bits, e.args[0].dtype.kind is Kind.SINT
        masked = f"(int({x}) & {(1 << width) - 1})"
        match name:
            case "popcount":
                return f"{masked}.bit_count()"
            case "leading_zeros":
                return f"({width} - {masked}.bit_length())"
            case "first_set":
                return f"({masked} & -{masked}).bit_length()"
            case _:
                return f"wrap(int(format({masked}, '0{width}b')[::-1], 2), {width}, {signed})"

    def val_Unary(self, e: Unary) -> str:
        x, dtype = self.val(e.operand), e.dtype
        if e.op is Op.NOT:
            return f"(not {x})"
        if e.shape or not dtype.is_int:
            return f"({'-' if e.op is Op.NEG else '~'}{x})"
        return f"wrap({'-' if e.op is Op.NEG else '~'}{x}, {dtype.bits}, {dtype.kind is Kind.SINT})"

    def val_Binary(self, e: Binary) -> str:
        a, b, dtype = self.val(e.lhs), self.val(e.rhs), e.lhs.dtype
        if e.op in (Op.LAND, Op.LOR):
            return f"({a} {'and' if e.op is Op.LAND else 'or'} {b})"
        if e.shape and dtype.is_int and e.op in (Op.DIV, Op.MOD, Op.SHL, Op.SHR):
            return f"integer_tile({a}, {b}, {e.op.name.lower()!r}, {dtype.bits}, {dtype.kind is Kind.SINT})"
        compare = e.op in (Op.LT, Op.LE, Op.GT, Op.GE, Op.EQ, Op.NE)
        if compare or e.shape or not (dtype.is_int or dtype == PY_INT):
            return f"({a} {e.op.value} {b})" if e.op is not Op.DIV or not e.shape else f"({a} / {b})"
        signed, bits = dtype.kind is not Kind.UINT, dtype.bits or 64
        match e.op:
            case Op.DIV:
                return f"wrap(divide({a}, {b}), {bits}, {signed})"
            case Op.MOD:
                return f"remainder({a}, {b})"
            case Op.SHL | Op.SHR:
                return f"wrap(shift({a}, {b}, {bits}, {e.op is Op.SHR}), {bits}, {signed})"
            case Op.AND | Op.OR | Op.XOR:
                return f"({a} {e.op.value} {b})"
            case _:
                return f"wrap({a} {e.op.value} {b}, {bits}, {signed})"

    def val_Math(self, e: Math) -> str:
        return f"MATH[Fn.{e.fn.name}]({', '.join(self.val(a) for a in e.args)})"

    def val_Cast(self, e: Cast) -> str:
        x, dtype = self.val(e.operand), e.dtype
        if narrow.is_narrow(e.operand.dtype):  # a code decodes to float32 first, then encodes last, as the C text does
            x = f"narrow_decode({narrow.NAMES[narrow.FORMATS[e.operand.dtype]]!r}, {x})"
        if narrow.is_narrow(dtype):
            return f"narrow_encode({narrow.NAMES[narrow.FORMATS[dtype]]!r}, {x})"
        if dtype == BF16:
            return f"bfloat16({x})"
        if e.shape:
            return f"{x}.astype(np.{np_dtype(dtype).name})"
        if dtype.is_int:
            return f"wrap(int({x}), {dtype.bits}, {dtype.kind is Kind.SINT})"
        if dtype.kind is Kind.PACKED:  # a packed word is its 32 bits, reinterpreted from an integer or another word
            return f"wrap(int({x}), 32, False)"
        if dtype.is_float:
            return f"np.{np_dtype(dtype).name}({x})"
        if dtype == BOOL:
            return f"bool({x})"
        return f"int({x})" if dtype == PY_INT else f"float({x})"  # a Python number: a literal or a runtime dimension

    def val_Select(self, e: Select) -> str:
        if e.shape:
            return f"np.where({self.val(e.cond)}, {self.val(e.then)}, {self.val(e.orelse)})"
        return f"({self.val(e.then)} if {self.val(e.cond)} else {self.val(e.orelse)})"

    def val_Transpose(self, e: Transpose) -> str:
        return f"np.swapaxes({self.val(e.operand)}, 0, 1)"

    def val_Arange(self, e: Arange) -> str:
        return f"arange_tile({e.shape}, {e.axis}, int({self.val(e.start)}))"

    def val_Zeros(self, e: Zeros) -> str:
        return f"np.zeros({e.shape}, np.{np_dtype(e.dtype).name})"

    def val_Reduce(self, e: Reduce) -> str:
        if (valid := self.valid(e.operand)) is None:
            op = f"REDUCE[Reduction.{e.op.name}]({self.val(e.operand)}, axis={e.axis}, keepdims={e.keepdims})"
        else:
            op = f"reduce_valid(Reduction.{e.op.name}, {self.val(e.operand)}, {valid}, {e.axis}, {e.keepdims})"
        return f"{op}.astype(np.{np_dtype(e.dtype).name})"

    def val_TileLoad(self, e: TileLoad) -> str:
        starts = ", ".join(f"int({self.val(sl.start)})" for sl in e.slices)
        extents = tuple(sl.extent for sl in e.slices)
        load = f"load_tile({leaf(e.tensor.path)}, ({starts},), {extents}, np.{np_dtype(e.dtype).name})"
        return f"{load}.reshape({e.shape})"


@dataclass(slots=True)
class Thread:
    index: int
    generator: GeneratorType[Request, Value | None, None]
    request: Request | None = None
    reply: Value | None = None
    done: bool = False
    block_id: int = 0

    def advance(self) -> None:
        """Run the thread to its next request, giving it the reply to the last one."""
        try:
            self.request = self.generator.send(self.reply)
        except StopIteration:
            self.done, self.request = True, None
        self.reply = None


@dataclass(slots=True)
class Emulated:
    """A kernel ready to run: the IR, the Python source it became and the function compiled from it."""

    kernel: Kernel
    source: str
    function: Callable[..., GeneratorType[Request, Value | None, None]]
    parameters: tuple[str, ...]
    warp: int
    tensors: dict[str, TensorRef] = field(default_factory=dict)

    def run(self, launch: Geometry, values: Sequence[Argument]) -> None:
        """Run every block of `launch` on the arguments; arrays are updated in place."""
        arguments, masked = self.arguments(values), masked_tiles(self.kernel)
        gx, gy, gz, threads = dimensions(self.kernel, launch)
        together = 'BlockRequest("grid_sync")' in self.source
        pending: list[Thread] = []
        with np.errstate(all="ignore"):
            for bz, by, bx in product(range(gz), range(gy), range(gx)):
                shared = {r.path[0]: np.zeros(r.extent, np_dtype(r.type.dtype)) for r in self.kernel.shared}
                tiles = {v.name: np.zeros(v.shape, np_dtype(v.dtype)) for v in self.kernel.tiles}
                tiles |= {f"{v.name}__v": np.zeros(v.shape, bool) for v in self.kernel.tiles if v.name in masked}
                fixed = {"_bx": bx, "_by": by, "_bz": bz, "_gx": gx, "_gy": gy, "_gz": gz}
                fixed |= {"_bdx": threads, "_bdy": 1, "_bdz": 1, "_ty": 0, "_tz": 0}
                block = [
                    Thread(
                        t,
                        self.function(**arguments, **shared, **tiles, **fixed, _tx=t),
                        block_id=bx + by * gx + bz * gx * gy,
                    )
                    for t in range(threads)
                ]
                if together:
                    pending.extend(block)
                else:
                    self.schedule(block)
            if together:
                self.schedule(pending)

    def arguments(self, values: Sequence[Argument]) -> dict[str, Argument]:
        """The kernel's parameters by the names the emitted function uses: arrays, numbers, struct fields."""
        out: dict[str, Argument] = {}

        def bind(path: tuple[str, ...], kind: ParamType, value: Argument) -> None:
            match kind, value:
                case Dtype(), int() | float():
                    out[leaf(path)] = np_dtype(kind).type(value) if kind.is_float else int(value)
                case TensorType(), np.ndarray():
                    out[leaf(path)] = value
                    for axis, size in enumerate(value.shape):
                        out[leaf((*path[:-1], f"{path[-1]}_n{axis}"))] = size
                case StructType(), Mapping():
                    for name, field_kind in kind.fields:
                        bind((*path, name), field_kind, value[name])

        for (name, kind), value in zip(self.kernel.params, values, strict=True):
            bind((name,), kind, value)
        return out

    def schedule(self, block: list[Thread]) -> None:
        """Advance every thread to its next request, answer the requests that can be answered, repeat until all end."""
        while live := [t for t in block if not t.done]:
            for t in live:
                if t.request is None:
                    t.advance()
            live = [t for t in block if not t.done]
            progressed = False
            if live and all(isinstance(t.request, BlockRequest) and t.request.op == "grid_sync" for t in live):
                for t in live:
                    t.request = None
                progressed = True
            else:
                for block_id in sorted({t.block_id for t in live}):
                    group = [t for t in live if t.block_id == block_id]
                    progressed = self.answer_warps(group) or self.answer_block(group) or progressed
            if live and not progressed and all(t.request for t in live):
                raise Deadlock(self.describe_wait(live))

    def answer_block(self, live: list[Thread]) -> bool:
        requests = [t.request for t in live]
        if not all(isinstance(r, BlockRequest) for r in requests):
            return False
        ops = {r.op for r in requests if isinstance(r, BlockRequest)}
        if "grid_sync" in ops:
            return False
        if len(ops) != 1:
            raise Deadlock(f"threads of a block wait at different barriers: {sorted(ops)}")
        votes = [r.vote for r in requests if isinstance(r, BlockRequest)]
        result = {"sync": None, "count": sum(votes), "all": all(votes), "any": any(votes)}[ops.pop()]
        for t in live:
            t.reply, t.request = result, None
        return True

    def answer_warps(self, live: list[Thread]) -> bool:
        progressed = False
        for w in sorted({t.index // self.warp for t in live}):
            lanes = {t.index % self.warp: t for t in live if t.index // self.warp == w}
            asking = {lane: t.request for lane, t in lanes.items() if isinstance(t.request, WarpRequest)}
            alive_mask = sum(1 << lane for lane in lanes)
            groups = {(r.op, r.mask & alive_mask, r.site if r.op == "active_mask" else "") for r in asking.values()}
            answered: set[int] = set()
            for op, mask, site in sorted(groups):
                members = (
                    {lane for lane, request in asking.items() if request.op == op and request.site == site}
                    if op == "active_mask"
                    else {lane for lane in lanes if mask >> lane & 1}
                )
                if not members or members & answered or not members <= asking.keys():
                    continue
                if not all(asking[lane].op == op and asking[lane].mask & alive_mask == mask for lane in members):
                    continue
                replies = collective(
                    op,
                    {lane: asking[lane].operands for lane in members},
                    members if op == "active_mask" else set(lanes),
                )
                for lane in members:
                    lanes[lane].reply, lanes[lane].request = replies[lane], None
                progressed = True
                answered.update(members)
        return progressed

    def describe_wait(self, live: list[Thread]) -> str:
        where = [
            f"thread {t.index} waits at {t.request} on line {t.generator.gi_frame.f_lineno}"
            for t in live
            if t.generator.gi_frame
        ]
        return "no thread can continue: " + "; ".join(where[:8])


def collective(op: str, operands: Mapping[int, tuple[Value, ...]], alive: set[int]) -> dict[int, Value | None]:
    """What each member lane of a warp gets back from a collective, given the operands of all members."""
    members = set(operands)
    match op:
        case "reduce_add" | "reduce_min" | "reduce_max" | "reduce_and" | "reduce_or" | "reduce_xor":
            values = [int(plain(v[0])) for v in operands.values()]
            if op == "reduce_add":
                result = sum(values)
            elif op in ("reduce_min", "reduce_max"):
                result = (min if op == "reduce_min" else max)(values)
            else:
                result = values[0]
                for value in values[1:]:
                    result = (
                        result & value
                        if op == "reduce_and"
                        else result | value
                        if op == "reduce_or"
                        else result ^ value
                    )
            return dict.fromkeys(members, result)
        case "sync":
            return dict.fromkeys(members)
        case "active_mask":
            return dict.fromkeys(members, sum(1 << lane for lane in alive))
        case "ballot":
            return dict.fromkeys(members, sum(1 << lane for lane, (value, *_) in operands.items() if value))
        case "all":
            return dict.fromkeys(members, all(bool(v) for v, *_ in operands.values()))
        case "any":
            return dict.fromkeys(members, any(bool(v) for v, *_ in operands.values()))
        case "eq":
            return dict.fromkeys(members, len({bool(v) for v, *_ in operands.values()}) == 1)
        case "match_any":
            return {
                lane: sum(1 << other for other, (v, *_) in operands.items() if v == mine)
                for lane, (mine, *_) in operands.items()
            }
        case "match_all":
            equal = len({plain(v) for v, *_ in operands.values()}) == 1
            return dict.fromkeys(members, sum(1 << lane for lane in members) if equal else 0)
        case "shfl" | "shfl_up" | "shfl_down" | "shfl_xor":
            out: dict[int, Value | None] = {}
            for lane, (value, amount, segment) in operands.items():
                delta, size = int(plain(amount)), int(plain(segment))
                start = lane // size * size
                src = {
                    "shfl": start + delta % size,
                    "shfl_up": lane - delta,
                    "shfl_down": lane + delta,
                    "shfl_xor": lane ^ delta,
                }[op]
                out[lane] = operands[src][0] if src in operands and start <= src < start + size else value
            return out
    raise NotImplementedError(f"the emulator has no semantics for the warp collective `{op}`")


def build(kernel: Kernel, target: Target) -> Emulated:
    """Compile the IR to a Python function whose line table is the user's file."""
    for s in walk(kernel.body):
        if isinstance(s, Accumulate) and s.lhs.dtype in FP8:
            raise CaptureError(s.span, f"the emulator has no fp8 decode; NVIDIA forms exist on {FP8_TARGETS}")
    narrow.refuse(kernel, target)
    emitter = PythonEmitter(kernel, target.warp_size)
    leaves = [name for argument, kind in kernel.params for name in parameter_names(argument, kind)]
    shared = [r.path[0] for r in kernel.shared]
    tiles = [v.name for v in kernel.tiles]
    tiles += [f"{name}__v" for name in sorted(masked_tiles(kernel))]
    parameters = (*leaves, *shared, *tiles, *GEOMETRY)
    text, origin = emitter.emit(parameters)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt | ast.expr | ast.arg):
            node.lineno = origin[node.lineno - 1]
            node.end_lineno = max(node.lineno, origin[(node.end_lineno or node.lineno) - 1])
    defined: dict[str, Callable[..., GeneratorType[Request, Value | None, None]]] = {}
    exec(compile(tree, kernel.span.file, "exec"), dict(HELPERS), defined)
    return Emulated(kernel, text, defined[kernel.name], tuple(parameters), target.warp_size)


def parameter_names(name: str, kind: ParamType, prefix: tuple[str, ...] = ()) -> list[str]:
    path = (*prefix, name)
    match kind:
        case Dtype():
            return [leaf(path)]
        case TensorType():
            return [leaf(path), *(leaf((*prefix, f"{name}_n{axis}")) for axis in range(kind.rank))]
        case StructType():
            return [n for field_name, field_kind in kind.fields for n in parameter_names(field_name, field_kind, path)]
    return []


def emulate(
    function: KernelFunction,
    launch: Geometry,
    *arguments: HostArgument,
    target: Target | None = None,
) -> None:
    """Run a `@kernel` function in the emulator. Arrays in `arguments` are updated in place.

    The target is an NVIDIA one by default, so warp collectives are available; pass another to emulate its warp width.
    """
    chosen = target or Target.of("sm_90a")
    names = tuple(inspect.signature(function.fn).parameters)
    consts = {
        name: _consts(arg)
        for name, arg in zip(names, arguments, strict=True)
        if is_dataclass(arg) and not isinstance(arg, type)
    }
    ir = capture(
        Path(inspect.getsourcefile(function.fn) or ""),
        function.name,
        consts=consts,
        dtypes=_dtypes(names, arguments),
        structs=_structs(names, arguments),
        target=chosen,
        arithmetic=Arithmetic.STRICT,
    )
    build(ir, chosen).run(launch, [_value(a) for a in arguments])


__all__ = ["Deadlock", "Emulated", "build", "emulate"]

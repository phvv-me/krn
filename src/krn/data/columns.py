"""Columns in the layout that libcudf and Arrow share, so a column can be handed to either without a copy.

A fixed width `Column[F]` is `values` and `valid`. A variable width `Lists[F]` adds `offsets` (`i32`, `rows + 1`
entries, absolute positions in `values`); strings are `Lists[u8]`. Validity is `u32` words, bit `row % 32` of word
`row // 32`, least significant bit first, 1 for a value that is not null. It is always allocated, all ones for the
rows of a column with no nulls, so a kernel never branches on its absence. A null value is undefined, and the
offsets of a null list are equal. Every host buffer has a positive multiple of 64 bytes of capacity, zero past the
logical size; kernels read `rows`, never the length of a buffer.
"""

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum, auto
from typing import NamedTuple, overload

import numpy as np

from krn import Vector, atomic, cg, items, kernel, struct
from krn.arithmetic import align_up, ceildiv
from krn.frontend.numerics import widens_to
from krn.frontend.types import Scalar, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

type Argument = int | np.ndarray | Mapping[str, Argument]
type Launch = Callable[[str, int, list[Argument]], None]

ALIGNMENT = 64
FAN_IN = 64
CHUNK = 256


@kernel
def bit_mask(bit: u32) -> u32:
    """`1 << bit` as five constant shifts, because the sketch types a shift by `int` counts only."""
    mask = u32(1)
    for step in (16, 8, 4, 2, 1):
        if bit >= step:
            mask <<= step
            bit -= step
    return mask


@struct
class Column[F: Scalar]:
    values: Vector[F]
    valid: Vector[u32]
    rows: u32

    def is_valid(self, row: u32) -> bool:
        return (self.valid[row // 32] & bit_mask(row % 32)) != 0


@struct
class Lists[F: Scalar](Column[F]):
    offsets: Vector[i32]


@struct
class Output[F: Scalar](Column[F]):
    """Distinct operand names avoid duplicate C typedefs without changing the layout."""


@struct
class Other[F: Scalar](Column[F]):
    pass


@struct
class Condition[F: Scalar](Column[F]):
    pass


@struct
class Partials[F: Scalar](Column[F]):
    pass


@struct
class ListsOutput[F: Scalar](Lists[F]):
    pass


class ColumnArrays(NamedTuple):
    """The host buffers of a `Column`: the field names are the struct's, so `_asdict()` is a launch argument."""

    values: np.ndarray
    valid: np.ndarray
    rows: int


class ListsArrays(NamedTuple):
    """The host buffers of a `Lists`; `values` is the child, and strings are `Lists[u8]` with `chars` as `values`."""

    values: np.ndarray
    valid: np.ndarray
    rows: int
    offsets: np.ndarray


def padded(array: np.ndarray) -> np.ndarray:
    """A copy of `array` in a buffer whose capacity is a positive multiple of 64 bytes, zero past `array`."""
    per_block = ALIGNMENT // array.itemsize
    buffer = np.zeros(max(per_block, align_up(array.size, per_block)), array.dtype)
    buffer[: array.size] = array
    return buffer


def empty(size: int, dtype: np.typing.DTypeLike) -> np.ndarray:
    """A zeroed buffer of `size` logical elements with the padded capacity."""
    return padded(np.zeros(size, dtype))


def validity_words(valid: np.ndarray) -> np.ndarray:
    """The `u32` validity words of a boolean array, padded; the bits past its length are zero."""
    return padded(np.packbits(valid, bitorder="little")).view("<u4").astype(np.uint32)


def validity_bits(words: np.ndarray, rows: int) -> np.ndarray:
    """The boolean array of the first `rows` validity bits."""
    return np.unpackbits(words.astype("<u4").view(np.uint8), bitorder="little", count=rows).astype(bool)


def offsets_array(lengths: np.ndarray) -> np.ndarray:
    """The `i32` offsets of rows with these lengths, `rows + 1` entries then the last offset up to the capacity."""
    total = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    if total[-1] > np.iinfo(np.int32).max:
        raise OverflowError("offsets are i32, so a child holds at most 2**31 - 1 values")
    offsets = padded(total.astype(np.int32))
    offsets[len(total) :] = total[-1]
    return offsets


@kernel
def mark(valid: Vector[u32], row: u32, flag: bool) -> None:
    if flag:
        atomic.or_(valid, row // 32, bit_mask(row % 32))
    else:
        atomic.and_(valid, row // 32, ~bit_mask(row % 32))


@kernel
def copy_range[A: Scalar, B: Scalar](src: Column[A], out: Output[B], start: u32, base: u32, count: u32) -> None:
    for i in items(count):
        out.values[base + i] = src.values[start + i].to(out.values.dtype)
        mark(out.valid, base + i, src.is_valid(start + i))


@kernel
def gather_values[A: Scalar, I: Scalar, B: Scalar](src: Column[A], index: Vector[I], out: Output[B]) -> None:
    for i in items(out.rows):
        j = u64(index[i])
        good = False
        if j < u64(src.rows):
            good = src.is_valid(u32(j))
            out.values[i] = src.values[j].to(out.values.dtype)
        mark(out.valid, i, good)


@kernel
def scatter_values[A: Scalar, I: Scalar, B: Scalar](src: Column[A], index: Vector[I], out: Output[B]) -> None:
    for i in items(src.rows):
        j = u32(index[i])
        out.values[j] = src.values[i].to(out.values.dtype)
        mark(out.valid, j, src.is_valid(i))


@kernel
def fill_values[B: Scalar](out: Output[B], value: Vector[B], valid: u32, start: u32, count: u32) -> None:
    for i in items(count):
        out.values[start + i] = value[0]
        mark(out.valid, start + i, valid != 0)


@kernel
def choose_values[C: Scalar, A: Scalar, B: Scalar, D: Scalar](
    cond: Condition[C], a: Column[A], b: Other[B], out: Output[D]
) -> None:
    for i in items(out.rows):
        good = False
        if cond.is_valid(i):
            if cond.values[i] != 0:
                good = a.is_valid(i)
                out.values[i] = a.values[i].to(out.values.dtype)
            else:
                good = b.is_valid(i)
                out.values[i] = b.values[i].to(out.values.dtype)
        mark(out.valid, i, good)


@kernel
def null_flags[A: Scalar](src: Column[A], out: Output[u8]) -> None:
    for i in items(src.rows):
        out.values[i] = u8(not src.is_valid(i))
        mark(out.valid, i, True)


@kernel
def replace_nulls[A: Scalar, B: Scalar](src: Column[A], value: Vector[B], out: Output[B]) -> None:
    for i in items(src.rows):
        if src.is_valid(i):
            out.values[i] = src.values[i].to(out.values.dtype)
        else:
            out.values[i] = value[0]
        mark(out.valid, i, True)


@kernel
def fold(src: Column, out: Column, j: u32, op) -> None:
    total = out.values[j]
    seen = False
    i = j
    while i < src.rows:
        if src.is_valid(i):
            v = src.values[i].to(out.values.dtype)
            total = op(total, v) if seen else v
            seen = True
        i += out.rows
    out.values[j] = total
    mark(out.valid, j, seen)


@kernel
def fold_sum[A: Scalar, B: Scalar](src: Column[A], out: Output[B]) -> None:
    for j in items(out.rows):
        fold(src, out, j, cg.plus)


@kernel
def smaller(a, b):
    """Comparison selection also supports f16, which capture's min intrinsic currently refuses."""
    return a if a < b else b


@kernel
def larger(a, b):
    return a if a > b else b


@kernel
def fold_min[A: Scalar, B: Scalar](src: Column[A], out: Output[B]) -> None:
    for j in items(out.rows):
        fold(src, out, j, smaller)


@kernel
def fold_max[A: Scalar, B: Scalar](src: Column[A], out: Output[B]) -> None:
    for j in items(out.rows):
        fold(src, out, j, larger)


@kernel
def fold_count[A: Scalar](src: Column[A], out: Output[u64]) -> None:
    for j in items(out.rows):
        total = u64(0)
        i = j
        while i < src.rows:
            if src.is_valid(i):
                total += 1
            i += out.rows
        out.values[j] = total
        mark(out.valid, j, total != 0)


@kernel
def first_equal[A: Scalar](src: Column[A], value: Vector[A], out: Output[i64]) -> None:
    for j in items(out.rows):
        first = i64(0)
        seen = False
        i = j
        while i < src.rows:
            if src.is_valid(i) and src.values[i] == value[0]:
                if not seen:
                    first = i64(i)
                seen = True
            i += out.rows
        out.values[j] = first
        mark(out.valid, j, seen)


@kernel
def scan_chunk(src: Column, out: Column, totals: Column, c: u32, chunk: u32, op) -> None:
    total = out.values[c * chunk]
    seen = False
    for i in range(c * chunk, min((c + 1) * chunk, src.rows)):
        good = src.is_valid(u32(i))
        if good:
            v = src.values[i].to(out.values.dtype)
            total = op(total, v) if seen else v
            seen = True
            out.values[i] = total
        mark(out.valid, u32(i), good)
    totals.values[c] = total
    mark(totals.valid, c, seen)


@kernel
def scan_sum[A: Scalar, B: Scalar](src: Column[A], out: Output[B], totals: Partials[B], chunk: u32) -> None:
    for c in items(totals.rows):
        scan_chunk(src, out, totals, c, chunk, cg.plus)


@kernel
def scan_min[A: Scalar, B: Scalar](src: Column[A], out: Output[B], totals: Partials[B], chunk: u32) -> None:
    for c in items(totals.rows):
        scan_chunk(src, out, totals, c, chunk, smaller)


@kernel
def scan_max[A: Scalar, B: Scalar](src: Column[A], out: Output[B], totals: Partials[B], chunk: u32) -> None:
    for c in items(totals.rows):
        scan_chunk(src, out, totals, c, chunk, larger)


@kernel
def scan_offset(out: Column, prefix: Column, i: u32, chunk: u32, op) -> None:
    """A null chunk has no identity; look back to the last valid scanned total."""
    c = i // chunk
    if c > 0 and out.is_valid(i):
        previous = c - 1
        good = prefix.is_valid(previous)
        while previous > 0 and not good:
            previous -= 1
            good = prefix.is_valid(previous)
        if good:
            out.values[i] = op(prefix.values[previous], out.values[i])


@kernel
def scan_add_sum[B: Scalar](out: Output[B], prefix: Column[B], chunk: u32) -> None:
    for i in items(out.rows):
        scan_offset(out, prefix, i, chunk, cg.plus)


@kernel
def scan_add_min[B: Scalar](out: Output[B], prefix: Column[B], chunk: u32) -> None:
    for i in items(out.rows):
        scan_offset(out, prefix, i, chunk, smaller)


@kernel
def scan_add_max[B: Scalar](out: Output[B], prefix: Column[B], chunk: u32) -> None:
    for i in items(out.rows):
        scan_offset(out, prefix, i, chunk, larger)


@kernel
def selected_flags[A: Scalar](mask: Column[A], out: Output[u32]) -> None:
    for i in items(mask.rows):
        out.values[i] = u32(mask.is_valid(i) and mask.values[i] != 0)
        mark(out.valid, i, True)


@kernel
def valid_flags[A: Scalar](src: Column[A], out: Output[u32]) -> None:
    for i in items(src.rows):
        out.values[i] = u32(src.is_valid(i))
        mark(out.valid, i, True)


@kernel
def selected_rows(flags: Column[u32], offsets: Vector[i32], index: Vector[i32]) -> None:
    for i in items(flags.rows):
        if flags.values[i] != 0:
            index[offsets[i]] = i32(i)


@kernel
def gather_lengths[A: Scalar, I: Scalar](src: Lists[A], index: Vector[I], lengths: Output[i32]) -> None:
    for i in items(lengths.rows):
        j = u64(index[i])
        length = i32(0)
        good = False
        if j < u64(src.rows):
            good = src.is_valid(u32(j))
        if good:
            length = src.offsets[j + 1] - src.offsets[j]
        lengths.values[i] = length
        mark(lengths.valid, i, True)


@kernel
def gather_lists[A: Scalar, I: Scalar, B: Scalar](src: Lists[A], index: Vector[I], out: ListsOutput[B]) -> None:
    """One thread per row; a warp per long row is deferred until measured."""
    for i in items(out.rows):
        j = u64(index[i])
        good = False
        if j < u64(src.rows):
            good = src.is_valid(u32(j))
            if good:
                start = src.offsets[j]
                for k in range(src.offsets[j + 1] - start):
                    out.values[out.offsets[i] + k] = src.values[start + k].to(out.values.dtype)
        mark(out.valid, i, good)


@kernel
def copy_lists[A: Scalar, B: Scalar](
    src: Lists[A], out: ListsOutput[B], start: u32, count: u32, base_row: u32, base_value: i32
) -> None:
    """One thread per row, shifting offsets and converting child values; host supplies the final offset."""
    for i in items(count):
        row = start + i
        begin = src.offsets[row]
        dest = base_value + begin - src.offsets[start]
        out.offsets[base_row + i] = dest
        for k in range(src.offsets[row + 1] - begin):
            out.values[dest + k] = src.values[begin + k].to(out.values.dtype)
        mark(out.valid, base_row + i, src.is_valid(row))


type Number = int | float | bool | np.number | np.bool_
type Values = Sequence[Number | None] | np.ndarray
type Arrays = ColumnArrays | ListsArrays

_TYPES = dict(
    zip(
        map(np.dtype, ("u1", "i1", "u2", "i2", "f2", "u4", "i4", "f4", "u8", "i8", "f8")),
        (u8, i8, u16, i16, f16, u32, i32, f32, u64, i64, f64),
        strict=True,
    )
)


def _dtype(dtype: np.typing.DTypeLike) -> np.dtype:
    result = np.dtype(dtype)
    if result.kind == "b":
        result = np.dtype("u1")
    if result not in _TYPES:
        raise TypeError(f"unsupported column dtype {result}")
    return result


def common_type(*dtypes: np.typing.DTypeLike) -> np.dtype:
    """First lossless common dtype in the column ladder; incomparable types require an explicit cast."""
    sources = [_dtype(dtype) for dtype in dtypes]
    if not sources:
        raise ValueError("common_type needs at least one dtype")
    for dtype, scalar in _TYPES.items():
        if all(widens_to(_TYPES[source], scalar) for source in sources):
            return dtype
    raise TypeError(f"no lossless common type for {', '.join(map(str, sources))}; cast one explicitly")


def column_arrays(values: Values, dtype: np.typing.DTypeLike | None = None) -> ColumnArrays:
    """Build a nullable column; bools use u8 storage and an all-null inferred column uses f64."""
    valid = np.array([v is not None for v in values], dtype=bool)
    inferred = (
        values.dtype if isinstance(values, np.ndarray) else np.asarray([v for v in values if v is not None]).dtype
    )
    if dtype is not None:
        inferred = dtype
    result = new_column(len(values), _dtype(inferred))
    result.values[: result.rows] = np.asarray([0 if v is None else v for v in values], result.values.dtype)
    result.valid[:] = validity_words(valid)
    return result


def column_values(c: ColumnArrays) -> list[int | float | None]:
    """Read logical rows; the bytes of null values are undefined."""
    return [
        v.item() if good else None for v, good in zip(c.values[: c.rows], validity_bits(c.valid, c.rows), strict=True)
    ]


def new_column(rows: int, dtype: np.typing.DTypeLike) -> ColumnArrays:
    """Allocate zeroed values and all-null validity in the requested dtype."""
    if rows < 0:
        raise ValueError("rows must be nonnegative")
    return ColumnArrays(empty(rows, _dtype(dtype)), empty(ceildiv(rows, 32), np.uint32), rows)


def lists_arrays(rows: Sequence[Sequence[Number] | np.ndarray | None], dtype: np.typing.DTypeLike) -> ListsArrays:
    """Build nullable lists with non-null children of dtype; null lists have equal adjacent offsets."""
    offsets = offsets_array(np.array([0 if row is None else len(row) for row in rows], np.int64))
    out = _new_lists(len(rows), dtype, offsets)
    out.values[: int(offsets[len(rows)])] = np.array(
        [v for row in rows if row is not None for v in row], out.values.dtype
    )
    out.valid[:] = validity_words(np.array([row is not None for row in rows], bool))
    return out


def lists_values(c: ListsArrays) -> list[list[int | float] | None]:
    """Read logical lists, preserving nulls and valid empty lists."""
    return [
        c.values[c.offsets[i] : c.offsets[i + 1]].tolist() if good else None
        for i, good in enumerate(validity_bits(c.valid, c.rows))
    ]


def _new_lists(rows: int, dtype: np.typing.DTypeLike, offsets: np.ndarray) -> ListsArrays:
    return ListsArrays(empty(int(offsets[rows]), _dtype(dtype)), empty(ceildiv(rows, 32), np.uint32), rows, offsets)


def _run(launch: Launch, name: str, rows: int, *args: Argument | Arrays) -> None:
    if rows:
        launch(
            name,
            rows,
            [arg._asdict() if isinstance(arg, (ColumnArrays, ListsArrays)) else arg for arg in args],
        )


def cast(launch: Launch, c: ColumnArrays, dtype: np.typing.DTypeLike) -> ColumnArrays:
    """Explicitly convert values, preserving nulls; float overflow to integers is undefined."""
    out = new_column(c.rows, dtype)
    _run(launch, "copy_range", c.rows, c, out, 0, 0, c.rows)
    return out


@overload
def slice_rows(launch: Launch, c: ColumnArrays, start: int | None, stop: int | None) -> ColumnArrays: ...
@overload
def slice_rows(launch: Launch, c: ListsArrays, start: int | None, stop: int | None) -> ListsArrays: ...
def slice_rows(launch: Launch, c: Arrays, start: int | None, stop: int | None) -> Arrays:
    """Clip Python slice bounds, preserving dtype and nulls."""
    first, last, _ = slice(start, stop).indices(c.rows)
    count = max(0, last - first)
    if isinstance(c, ListsArrays):
        offsets = offsets_array(np.diff(c.offsets[first : first + count + 1]))
        out_lists = _new_lists(count, c.values.dtype, offsets)
        _run(launch, "copy_lists", count, c, out_lists, first, count, 0, 0)
        return out_lists
    out = new_column(count, c.values.dtype)
    _run(launch, "copy_range", out.rows, c, out, first, 0, out.rows)
    return out


@overload
def head(launch: Launch, c: ColumnArrays, count: int = 5) -> ColumnArrays: ...
@overload
def head(launch: Launch, c: ListsArrays, count: int = 5) -> ListsArrays: ...
def head(launch: Launch, c: Arrays, count: int = 5) -> Arrays:
    """First count rows, preserving dtype and nulls; negative count omits trailing rows."""
    return slice_rows(launch, c, None, count)


@overload
def tail(launch: Launch, c: ColumnArrays, count: int = 5) -> ColumnArrays: ...
@overload
def tail(launch: Launch, c: ListsArrays, count: int = 5) -> ListsArrays: ...
def tail(launch: Launch, c: Arrays, count: int = 5) -> Arrays:
    """Last count rows, preserving dtype and nulls; negative count omits leading rows."""
    return slice_rows(launch, c, -count if count else c.rows, None)


@overload
def concatenate(launch: Launch, columns: Sequence[ColumnArrays]) -> ColumnArrays: ...
@overload
def concatenate(launch: Launch, columns: Sequence[ListsArrays]) -> ListsArrays: ...
def concatenate(launch: Launch, columns: Sequence[Arrays]) -> Arrays:
    """Append columns, preserving nulls and widening explicitly to their lossless common dtype."""
    dtype = common_type(*(c.values.dtype for c in columns))
    rows = sum(c.rows for c in columns)
    if isinstance(columns[0], ListsArrays):
        if not all(isinstance(c, ListsArrays) for c in columns):
            raise TypeError("concatenate requires matching column layouts")
        offsets = offsets_array(
            np.concatenate([np.diff(c.offsets[: c.rows + 1]) for c in columns if isinstance(c, ListsArrays)])
        )
        lists = _new_lists(rows, dtype, offsets)
        base_row, base_value = 0, 0
        for c in columns:
            if isinstance(c, ListsArrays):
                _run(launch, "copy_lists", c.rows, c, lists, 0, c.rows, base_row, base_value)
                base_row += c.rows
                base_value += int(c.offsets[c.rows])
        return lists
    if not all(isinstance(c, ColumnArrays) for c in columns):
        raise TypeError("concatenate requires matching column layouts")
    out = new_column(rows, dtype)
    base = 0
    for c in columns:
        _run(launch, "copy_range", c.rows, c, out, 0, base, c.rows)
        base += c.rows
    return out


def _indices(index: Sequence[int] | np.ndarray) -> np.ndarray:
    result = np.asarray(index) if len(index) else np.array([], dtype=np.int32)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise TypeError("indices must be a one-dimensional integer array")
    return result


@overload
def gather(launch: Launch, c: ColumnArrays, index: Sequence[int] | np.ndarray) -> ColumnArrays: ...
@overload
def gather(launch: Launch, c: ListsArrays, index: Sequence[int] | np.ndarray) -> ListsArrays: ...
def gather(launch: Launch, c: Arrays, index: Sequence[int] | np.ndarray) -> Arrays:
    """Take rows in index order; negative, out-of-range and source-null rows yield nulls of the source dtype."""
    indices = _indices(index)
    if isinstance(c, ListsArrays):
        lengths = new_column(len(indices), np.int32)
        _run(launch, "gather_lengths", lengths.rows, c, padded(indices), lengths)
        lists = _new_lists(lengths.rows, c.values.dtype, exclusive_scan(launch, lengths))
        _run(launch, "gather_lists", lists.rows, c, padded(indices), lists)
        return lists
    out = new_column(len(indices), c.values.dtype)
    _run(launch, "gather_values", out.rows, c, padded(indices), out)
    return out


def scatter(launch: Launch, src: ColumnArrays, index: Sequence[int] | np.ndarray, target: ColumnArrays) -> ColumnArrays:
    """Copy target then overwrite unique in-range indices, converting to target dtype and copying source validity."""
    indices = _indices(index)
    if len(indices) != src.rows or np.any(indices < 0) or np.any(indices >= target.rows):
        raise ValueError("scatter requires one in-range index per source row")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("scatter targets must be unique")
    out = cast(launch, target, target.values.dtype)
    _run(launch, "scatter_values", src.rows, src, padded(indices), out)
    return out


def fill(
    launch: Launch, c: ColumnArrays, value: Number | None, start: int = 0, stop: int | None = None
) -> ColumnArrays:
    """Copy and fill clipped rows in the column dtype; None clears validity."""
    out = cast(launch, c, c.values.dtype)
    first, last, _ = slice(start, stop).indices(c.rows)
    _run(
        launch,
        "fill_values",
        max(0, last - first),
        out,
        padded(np.array([0 if value is None else value], c.values.dtype)),
        int(value is not None),
        first,
        max(0, last - first),
    )
    return out


def _operand(value: ColumnArrays | Number | None, rows: int, adopted: np.dtype | None) -> ColumnArrays:
    """A column as it is; a scalar broadcast to a column. A Python literal adopts the dtype of the other operand."""
    match value:
        case ColumnArrays():
            if value.rows != rows:
                raise ValueError("operands must have equal row counts")
            return value
        case None | bool() | int() | float() if adopted is not None and not isinstance(value, np.generic):
            return column_arrays([value] * rows, adopted)
        case _:
            return column_arrays([value] * rows, None if value is None else np.asarray(value).dtype)


def copy_if_else(
    launch: Launch,
    cond: ColumnArrays,
    a: ColumnArrays | Number | None,
    b: ColumnArrays | Number | None,
) -> ColumnArrays:
    """Nonzero chooses a, zero b, null condition null; the result dtype is the lossless common dtype of a and b.

    A Python literal takes the dtype of the other operand (the weak literal rule); a numpy scalar keeps its own.
    """
    if cond.values.dtype.kind not in "iu":
        raise TypeError("condition must have integer storage")
    left_dtype = a.values.dtype if isinstance(a, ColumnArrays) else None
    right_dtype = b.values.dtype if isinstance(b, ColumnArrays) else None
    left, right = _operand(a, cond.rows, right_dtype), _operand(b, cond.rows, left_dtype)
    out = new_column(cond.rows, common_type(left.values.dtype, right.values.dtype))
    _run(launch, "choose_values", out.rows, cond, left, right, out)
    return out


def is_null(launch: Launch, c: ColumnArrays) -> ColumnArrays:
    """A never-null u8 column, one exactly where the input is null."""
    out = new_column(c.rows, np.uint8)
    _run(launch, "null_flags", c.rows, c, out)
    return out


def fill_null(launch: Launch, c: ColumnArrays, value: Number) -> ColumnArrays:
    """Replace nulls; Python scalars use the column dtype, numpy scalars choose a lossless common dtype."""
    dtype = common_type(c.values.dtype, value.dtype) if isinstance(value, (np.number, np.bool_)) else c.values.dtype
    out = new_column(c.rows, dtype)
    _run(launch, "replace_nulls", c.rows, c, padded(np.array([value], dtype)), out)
    return out


class Reduction(StrEnum):
    SUM = auto()
    MIN = auto()
    MAX = auto()
    COUNT = auto()
    MEAN = auto()
    ANY = auto()
    ALL = auto()
    ARG_MIN = auto()
    ARG_MAX = auto()


class Scan(StrEnum):
    SUM = auto()
    MIN = auto()
    MAX = auto()


def _accumulator(dtype: np.dtype, acc: np.typing.DTypeLike | None) -> np.dtype:
    if acc is not None:
        return _dtype(acc)
    return np.dtype({"i": "i8", "u": "u8", "f": "f8" if dtype.itemsize == 8 else "f4"}[dtype.kind])


def _fold(launch: Launch, c: ColumnArrays, name: str, dtype: np.dtype) -> ColumnArrays:
    while True:
        out = new_column(max(1, ceildiv(c.rows, FAN_IN)), dtype)
        _run(launch, name, out.rows, c, out)
        if out.rows == 1:
            return out
        c = out
        if name == "fold_count":
            name = "fold_sum"


def reduce(
    launch: Launch, c: ColumnArrays, kind: Reduction, acc: np.typing.DTypeLike | None = None
) -> int | float | bool | None:
    """Skip nulls; sum/count/any/all of nothing are 0/0/False/True, other results None.

    Sum defaults to i64/u64 for integers and f32/f64 for floats; acc overrides it. Extrema and their first indices
    retain input precision. Mean divides sum by count in Python floats. Float extrema exclude NaNs.
    """
    if not c.rows:
        return {
            Reduction.SUM: 0,
            Reduction.COUNT: 0,
            Reduction.ANY: False,
            Reduction.ALL: True,
        }.get(kind)
    match kind:
        case Reduction.MEAN:
            count = reduce(launch, c, Reduction.COUNT)
            total = reduce(launch, c, Reduction.SUM, acc)
            return float(total) / int(count) if total is not None and count else None
        case Reduction.ANY | Reduction.ALL:
            if c.values.dtype.kind != "u":
                raise TypeError(
                    "any and all are defined for bool (u8) and unsigned integer columns; cast or compare first"
                )
            value = reduce(launch, c, Reduction.MAX if kind is Reduction.ANY else Reduction.MIN)
            return bool(value) if value is not None else kind is Reduction.ALL
        case Reduction.ARG_MIN | Reduction.ARG_MAX:
            value = reduce(launch, c, Reduction.MIN if kind is Reduction.ARG_MIN else Reduction.MAX)
            if value is None:
                return None
            matches = new_column(max(1, ceildiv(c.rows, FAN_IN)), np.int64)
            _run(
                launch,
                "first_equal",
                matches.rows,
                c,
                padded(np.array([value], c.values.dtype)),
                matches,
            )
            return reduce(launch, matches, Reduction.MIN)
        case Reduction.COUNT:
            name, dtype = "fold_count", np.dtype("u8")
        case Reduction.SUM:
            name, dtype = "fold_sum", _accumulator(c.values.dtype, acc)
        case Reduction.MIN | Reduction.MAX:
            name, dtype = f"fold_{kind.value}", c.values.dtype
        case _:
            raise ValueError(f"unknown reduction {kind}")
    out = _fold(launch, c, name, dtype)
    if not out.valid[0] & 1:
        return 0 if kind in (Reduction.SUM, Reduction.COUNT) else None
    return out.values[0].item()


def null_count(launch: Launch, c: ColumnArrays) -> int:
    """Number of null rows, computed by the shared non-null count reduction."""
    count = reduce(launch, c, Reduction.COUNT)
    return c.rows - int(count) if count is not None else c.rows


def scan(
    launch: Launch,
    c: ColumnArrays,
    kind: Scan,
    acc: np.typing.DTypeLike | None = None,
    chunk: int = CHUNK,
) -> ColumnArrays:
    """Inclusive scan skipping nulls and keeping their rows null; sum uses the reduction accumulator defaults.

    Min/max retain the input dtype. Three collective-free kernels recurse over chunk totals.
    """
    if chunk < 2:
        raise ValueError("chunk must be at least two")
    dtype = _accumulator(c.values.dtype, acc) if kind is Scan.SUM else c.values.dtype
    out = new_column(c.rows, dtype)
    totals = new_column(ceildiv(c.rows, chunk), dtype)
    _run(launch, f"scan_{kind.value}", totals.rows, c, out, totals, chunk)
    if totals.rows > 1:
        prefix = scan(launch, totals, kind, dtype, chunk)
        _run(launch, f"scan_add_{kind.value}", out.rows, out, prefix, chunk)
    return out


def exclusive_scan(launch: Launch, lengths: ColumnArrays | np.ndarray) -> np.ndarray:
    """Padded i32 offsets [0, inclusive lengths], using the shared scan; lengths are non-null and nonnegative."""
    c = column_arrays(lengths) if isinstance(lengths, np.ndarray) else lengths
    if c.values.dtype.kind not in "iu" or not validity_bits(c.valid, c.rows).all() or np.any(c.values[: c.rows] < 0):
        raise ValueError("lengths must be non-null nonnegative integers")
    if np.any(c.values[: c.rows] > np.iinfo(np.int32).max):
        raise OverflowError("offsets are i32, so a child holds at most 2**31 - 1 values")
    inclusive = scan(launch, c, Scan.SUM, np.uint64)
    if c.rows and inclusive.values[c.rows - 1] > np.iinfo(np.int32).max:
        raise OverflowError("offsets are i32, so a child holds at most 2**31 - 1 values")
    offsets = empty(c.rows + 1, np.int32)
    offsets[1 : c.rows + 1] = inclusive.values[: c.rows]
    offsets[c.rows + 1 :] = offsets[c.rows]
    return offsets


def _selection(launch: Launch, c: Arrays, flags: ColumnArrays) -> Arrays:
    offsets = exclusive_scan(launch, flags)
    index = empty(int(offsets[flags.rows]), np.int32)
    _run(launch, "selected_rows", flags.rows, flags, offsets, index)
    return gather(launch, c, index[: int(offsets[flags.rows])])


@overload
def apply_boolean_mask(launch: Launch, c: ColumnArrays, mask: ColumnArrays) -> ColumnArrays: ...
@overload
def apply_boolean_mask(launch: Launch, c: ListsArrays, mask: ColumnArrays) -> ListsArrays: ...
def apply_boolean_mask(launch: Launch, c: Arrays, mask: ColumnArrays) -> Arrays:
    """Keep non-null nonzero mask rows in order, preserving source dtype and nulls."""
    if c.rows != mask.rows or mask.values.dtype.kind not in "iu":
        raise ValueError("mask must be an integer column with the same row count")
    flags = new_column(c.rows, np.uint32)
    _run(launch, "selected_flags", c.rows, mask, flags)
    return _selection(launch, c, flags)


@overload
def drop_nulls(launch: Launch, c: ColumnArrays) -> ColumnArrays: ...
@overload
def drop_nulls(launch: Launch, c: ListsArrays) -> ListsArrays: ...
def drop_nulls(launch: Launch, c: Arrays) -> Arrays:
    """Keep valid rows in their original order and dtype."""
    flags = new_column(c.rows, np.uint32)
    _run(launch, "valid_flags", c.rows, c, flags)
    return _selection(launch, c, flags)

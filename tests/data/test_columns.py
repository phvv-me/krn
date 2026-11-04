"""One operation table and independent nullable references on C, the emulator, Metal and CUDA."""

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Items, targets
from krn.compiler.api import capture, emit
from krn.compiler.dtypes import BF16, F16, F32, Dtype
from krn.compiler.source import Module
from krn.data import columns as col
from krn.data.columns import Argument, ColumnArrays, Launch
from krn.data.strings import as_lists, as_strings, strings_arrays, strings_values
from krn.emulator.emulate import Emulated
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from krn.runtime.cpu import CPUProgram
from krn.runtime.runtime import DTYPES as DEVICE_DTYPES
from tests.conftest import build_kernel
from tests.helpers.devices import CUDARun, MetalRun, bindings, fresh

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

try:
    from krn.runtime import metal
except ImportError:
    metal = None

FILE = Path(__file__).resolve().parents[2] / "src" / "krn" / "data" / "columns.py"
DTYPES = tuple(map(np.dtype, ("u1", "i1", "u2", "i2", "u4", "i4", "u8", "i8", "f2", "f4", "f8")))
SIZES = (0, 1, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 65537)
type Binding = tuple[tuple[str, Dtype], ...]
type Check = Callable[[Launch, bool], None]


@cache
def c_program(name: str, binding: Binding) -> CPUProgram:
    return build_kernel(FILE, name, dtypes=dict(binding))


def c_launch(name: str, rows: int, arguments: list[Argument]) -> None:
    c_program(name, tuple(bindings(FILE, name, arguments).items()))(*arguments, launch=Items(rows))


@cache
def emulator_program(name: str, binding: Binding) -> Emulated:
    target = targets.Target.of("sm_90a")
    return emulate(capture(FILE, name, target=target, dtypes=dict(binding)), target)


def emulator_launch(name: str, rows: int, arguments: list[Argument]) -> None:
    emulator_program(name, tuple(bindings(FILE, name, arguments).items())).run(Items(rows), arguments)


def assert_column(actual: ColumnArrays, expected: Sequence[int | float | None], dtype: np.typing.DTypeLike) -> None:
    assert actual.rows == len(expected) and actual.values.dtype == np.dtype(dtype)
    good = np.array([v is not None for v in expected], bool)
    np.testing.assert_array_equal(actual.valid, col.validity_words(good))
    wanted = np.array([v for v in expected if v is not None], dtype)
    np.testing.assert_array_equal(actual.values[: actual.rows][good].view("u1"), wanted.view("u1"))
    for buffer in (actual.values, actual.valid):
        assert buffer.nbytes > 0 and buffer.nbytes % 64 == 0


def corpus(dtype: np.dtype, small: bool, large: bool = False):
    rng = np.random.default_rng(309)
    sizes = SIZES[:8] if small or not large else SIZES
    for n in sizes:
        for density in (0, 0.1, 0.5, 1):
            values = rng.integers(-4 if dtype.kind in "if" else 0, 8, n).astype(dtype)
            if dtype.kind == "f":
                values *= np.array(0.25, dtype)
            valid = rng.random(n) >= density
            rows = [v.item() if good else None for v, good in zip(values, valid, strict=True)]
            yield col.column_arrays(rows, dtype), rows


def check_cast(launch: Launch, small: bool, source: np.dtype, target: np.dtype) -> None:
    values = np.array(
        [0.25, 1.5, 7.75, 17, 31, 63, 127] if source.kind == "f" else [0, 1, 7, 17, 31, 63, 127],
        source,
    )
    c = col.column_arrays([None, *values, None], source)
    assert_column(col.cast(launch, c, target), [None, *values.astype(target).tolist(), None], target)


def check_copy(launch: Launch, small: bool, dtype: np.dtype) -> None:
    c = col.column_arrays([None, 0, 1, 7, None], dtype)
    for start, stop in ((None, None), (-4, -1), (-200, 200), (4, 1), (0, 0)):
        assert_column(col.slice_rows(launch, c, start, stop), col.column_values(c)[start:stop], dtype)
    for count in (0, 1, 50, -2):
        assert_column(col.head(launch, c, count), col.column_values(c)[:count], dtype)
        assert_column(col.tail(launch, c, count), col.column_values(c)[-count:] if count else [], dtype)
    for c, rows in corpus(dtype, small):
        assert_column(col.cast(launch, c, dtype), rows, dtype)
    if dtype.kind == "f":
        values = np.array([0, -0.0, np.nan, np.inf, -np.inf], dtype)
        c = col.column_arrays(values)
        assert_column(col.cast(launch, c, dtype), values.tolist(), dtype)
        assert_column(
            col.gather(launch, c, [4, 3, 2, 1, 0]),
            np.take(values, [4, 3, 2, 1, 0]).tolist(),
            dtype,
        )


def check_gather(launch: Launch, small: bool, dtype: np.dtype) -> None:
    for c, rows in corpus(dtype, small, large=True):
        index = np.array([*range(c.rows - 1, -1, -1), -1, c.rows], dtype=np.int32)
        expected = [rows[i] if 0 <= i < len(rows) else None for i in index]
        assert_column(col.gather(launch, c, index), expected, dtype)
    c = col.column_arrays([1, None, 3], "i4")
    for index_dtype in ("i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"):
        assert_column(col.gather(launch, c, np.array([2, 1, 0, 127], index_dtype)), [3, None, 1, None], "i4")
    assert_column(col.gather(launch, c, np.array([2**64 - 1], "u8")), [None], "i4")


def check_scatter(launch: Launch, small: bool) -> None:
    for dtype in ("u1", "i4", "f4"):
        target = col.column_arrays([1, 2, None, 4, None], dtype)
        src = col.column_arrays([None, 9, 8], "i2")
        assert_column(col.scatter(launch, src, [0, 2, 4], target), [None, 2, 9, 4, 8], dtype)
        assert col.column_values(target) == [1, 2, None, 4, None]
        for index in ([-1, 2, 3], [0, 2, 5], [0, 0, 3]):
            with pytest.raises(ValueError):
                col.scatter(launch, src, index, target)


def check_choose(launch: Launch, small: bool) -> None:
    cond = col.column_arrays([0, 1, None, -2, 0, 1], "i4")
    a, b = (
        col.column_arrays([1, None, 3, 4, 5, 6], "f2"),
        col.column_arrays([7, 8, 9, 10, None, 12], "f4"),
    )
    assert_column(col.copy_if_else(launch, cond, a, b), [7, None, None, 4, None, 6], "f4")
    assert_column(col.copy_if_else(launch, cond, np.float16(2), b), [7, 2, None, 2, None, 2], "f4")
    assert_column(col.copy_if_else(launch, cond, 2, a), [1, 2, None, 2, 5, 2], "f2")
    assert_column(col.copy_if_else(launch, cond, 2.5, a), [1, 2.5, None, 2.5, 5, 2.5], "f2")
    assert_column(col.copy_if_else(launch, cond, a, None), [None, None, None, 4, None, 6], "f2")
    assert_column(col.copy_if_else(launch, cond, 2, 3), [3, 2, None, 2, 3, 2], "i8")
    assert_column(col.copy_if_else(launch, col.column_arrays([True, False]), 2, 3), [2, 3], "i8")


def check_choose_f64(launch: Launch, small: bool) -> None:
    """A numpy scalar keeps its dtype, so a float64 scalar with an f16 column widens to f64 (no Metal)."""
    cond = col.column_arrays([0, 1, None, -2, 0, 1], "i4")
    a = col.column_arrays([1, None, 3, 4, 5, 6], "f2")
    assert_column(col.copy_if_else(launch, cond, np.float64(2), a), [1, 2, None, 2, 5, 2], "f8")


def check_nulls(launch: Launch, small: bool) -> None:
    for dtype in ("u1", "i4", "f2", "f4"):
        for c, rows in corpus(np.dtype(dtype), small):
            assert_column(col.is_null(launch, c), [int(v is None) for v in rows], "u1")
            assert_column(col.fill_null(launch, c, 3), [3 if v is None else v for v in rows], dtype)
            for value in (None, 3):
                expected = list(rows)
                expected[1:-1] = [value] * len(expected[1:-1])
                assert_column(col.fill(launch, c, value, 1, -1), expected, dtype)
    c = col.column_arrays([1, None], "f2")
    assert_column(col.fill_null(launch, c, np.float32(2.5)), [1, 2.5], "f4")


def check_concatenate(launch: Launch, small: bool, pair: tuple[str, str, str]) -> None:
    a, b, out = pair
    left, right = col.column_arrays([1, None, 3], a), col.column_arrays([None, 5], b)
    assert_column(
        col.concatenate(launch, [left, right, col.column_arrays([], a)]),
        [1, None, 3, None, 5],
        out,
    )
    with pytest.raises(TypeError, match=r"uint64.*int64.*cast one explicitly"):
        col.concatenate(launch, [col.column_arrays([1], "u8"), col.column_arrays([2], "i8")])


def check_reductions(launch: Launch, small: bool, dtype: np.dtype) -> None:
    for c, rows in corpus(dtype, small, large=True):
        valid = np.array([v for v in rows if v is not None], dtype)
        acc = np.dtype(
            "f8" if dtype == np.dtype("f8") else "f4" if dtype.kind == "f" else "u8" if dtype.kind == "u" else "i8"
        )
        for kind, op in (
            (col.Reduction.SUM, np.add),
            (col.Reduction.MIN, np.minimum),
            (col.Reduction.MAX, np.maximum),
        ):
            actual = col.reduce(launch, c, kind)
            expected = (
                op.reduce(valid, dtype=acc if kind is col.Reduction.SUM else dtype).item()
                if len(valid)
                else 0
                if kind is col.Reduction.SUM
                else None
            )
            assert actual == expected
    c = col.column_arrays([None, 3, 1, None, 1, 7, 7], dtype)
    for kind, expected in (
        (col.Reduction.COUNT, 5),
        (col.Reduction.MEAN, 3.8),
        (col.Reduction.ARG_MIN, 2),
        (col.Reduction.ARG_MAX, 5),
    ):
        assert col.reduce(launch, c, kind) == expected
    assert col.null_count(launch, c) == 2
    for kind in (col.Reduction.ANY, col.Reduction.ALL):
        if dtype.kind == "u":
            assert col.reduce(launch, c, kind) is True
        else:
            with pytest.raises(TypeError, match="bool"):
                col.reduce(launch, c, kind)
    for rows in ([], [None], [None] * 65):
        c = col.column_arrays(rows, "u4")
        for kind in col.Reduction:
            expected = {
                col.Reduction.SUM: 0,
                col.Reduction.COUNT: 0,
                col.Reduction.ANY: False,
                col.Reduction.ALL: True,
            }.get(kind)
            assert col.reduce(launch, c, kind) == expected
    c = col.column_arrays([True, None, False])
    assert col.reduce(launch, c, col.Reduction.ANY) is True
    assert col.reduce(launch, c, col.Reduction.ALL) is False
    c = col.column_arrays([None, *([5] * 70), -2, -2, 9, 9], "i4")
    assert col.reduce(launch, c, col.Reduction.ARG_MIN) == 71
    assert col.reduce(launch, c, col.Reduction.ARG_MAX) == 73


def check_precision(launch: Launch, small: bool, acc: str) -> None:
    values = np.random.default_rng(51).normal(size=65 if small else 1025).astype("f2")
    c = col.column_arrays(values)
    result = col.reduce(launch, c, col.Reduction.SUM, acc)
    assert result is not None
    assert abs(result - np.sum(values, dtype=np.float64)) <= np.finfo(acc).eps * np.abs(values.astype("f8")).sum() * 64
    assert col.reduce(launch, c, col.Reduction.SUM, acc) == result
    out = col.scan(launch, c, col.Scan.SUM, acc=acc, chunk=4)
    expected = np.cumsum(values, dtype=np.float64)
    error = np.finfo(acc).eps * np.cumsum(np.abs(values.astype("f8")), dtype=np.float64) * 64
    assert np.all(np.abs(out.values[: out.rows] - expected) <= error)
    np.testing.assert_array_equal(out.valid, col.validity_words(np.ones(c.rows, bool)))


def check_scans(launch: Launch, small: bool, dtype: np.dtype) -> None:
    for c, rows in corpus(dtype, small, large=True):
        valid = np.array([v for v in rows if v is not None], dtype)
        for kind, op in (
            (col.Scan.SUM, np.add),
            (col.Scan.MIN, np.minimum),
            (col.Scan.MAX, np.maximum),
        ):
            out = col.scan(launch, c, kind)
            accumulated = op.accumulate(valid, dtype=out.values.dtype)
            it = iter(accumulated.tolist())
            expected = [next(it) if v is not None else None for v in rows]
            assert_column(out, expected, out.values.dtype)
    for chunk in (4, 7):
        for kind in col.Scan:
            for rows in (
                [None] * 15 + [2, None, 3] + [None] * 20 + [1] + [None] * 25 + [4],
                [None] * 65,
                [1] + [None] * 63 + [2],
            ):
                c = col.column_arrays(rows, "i4")
                out, default = col.scan(launch, c, kind, chunk=chunk), col.scan(launch, c, kind)
                assert_column(out, col.column_values(default), default.values.dtype)
                repeat = col.scan(launch, c, kind, chunk=chunk)
                np.testing.assert_array_equal(out.values, repeat.values)
                np.testing.assert_array_equal(out.valid, repeat.valid)
    lengths = np.array([0, 2, 0, 3, 1, 0, 5], np.int32)
    np.testing.assert_array_equal(col.exclusive_scan(launch, lengths), col.offsets_array(lengths))
    np.testing.assert_array_equal(col.exclusive_scan(launch, np.array([], "i4")), col.offsets_array(np.array([], "i4")))
    with pytest.raises(ValueError):
        col.scan(launch, col.column_arrays([1]), col.Scan.SUM, chunk=1)
    for bad in (col.column_arrays([None], "i4"), np.array([-1], "i4"), np.array([1.5], "f4")):
        with pytest.raises(ValueError):
            col.exclusive_scan(launch, bad)
    with pytest.raises(OverflowError):
        col.exclusive_scan(launch, np.array([2**31 - 1, 1], "i8"))


def check_filter(launch: Launch, small: bool) -> None:
    rng = np.random.default_rng(772)
    for dtype in ("u1", "i4", "f4"):
        for c, rows in corpus(np.dtype(dtype), small, large=True):
            choices = rng.choice([-2, 0, 1, 3], c.rows)
            good = rng.random(c.rows) > 0.25
            mask = col.column_arrays([int(v) if ok else None for v, ok in zip(choices, good, strict=True)], "i4")
            indices = np.flatnonzero(good & (choices != 0))
            assert_column(col.apply_boolean_mask(launch, c, mask), [rows[i] for i in indices], dtype)
            assert_column(col.drop_nulls(launch, c), [v for v in rows if v is not None], dtype)


def assert_lists(
    actual: col.ListsArrays,
    expected: Sequence[Sequence[int | float] | None],
    dtype: np.typing.DTypeLike,
) -> None:
    assert actual.rows == len(expected) and actual.values.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(actual.valid, col.validity_words(np.array([r is not None for r in expected], bool)))
    np.testing.assert_array_equal(
        actual.offsets,
        col.offsets_array(np.array([0 if r is None else len(r) for r in expected], "i4")),
    )
    assert col.lists_values(actual) == expected
    for buffer in (actual.values, actual.valid, actual.offsets):
        assert buffer.nbytes > 0 and buffer.nbytes % 64 == 0


def check_lists(launch: Launch, small: bool) -> None:
    for dtype in ("u1", "i2", "f2", "f4"):
        for n in (0, 1, 33, 65) if small else SIZES[:-1]:
            rows = [None if i % 5 == 0 else [i % 7] * (i % 4) for i in range(n)]
            c = col.lists_arrays(rows, dtype)
            assert_lists(c, rows, dtype)
            for index in ([], [*range(n - 1, -1, -1), -1, n]):
                expected = [rows[i] if 0 <= i < n else None for i in index]
                assert_lists(col.gather(launch, c, np.array(index, "i4")), expected, dtype)
            for start, stop in ((None, None), (-4, -1), (-200, 200), (4, 1)):
                assert_lists(col.slice_rows(launch, c, start, stop), rows[start:stop], dtype)
            assert_lists(col.head(launch, c, 2), rows[:2], dtype)
            assert_lists(col.tail(launch, c, 0), [], dtype)
            assert_lists(col.tail(launch, c, -2), rows[2:], dtype)
            assert_lists(col.concatenate(launch, [c, c]), rows + rows, dtype)
            mask = col.column_arrays([None if i % 7 == 0 else i % 2 for i in range(n)], "u1")
            assert_lists(
                col.apply_boolean_mask(launch, c, mask),
                [row for i, row in enumerate(rows) if i % 7 != 0 and i % 2],
                dtype,
            )
            assert_lists(col.drop_nulls(launch, c), [r for r in rows if r is not None], dtype)
    a, b = col.lists_arrays([[1], None, []], "u1"), col.lists_arrays([[2, 3]], "i1")
    assert_lists(col.concatenate(launch, [a, b]), [[1], None, [], [2, 3]], "i2")
    a, b = col.lists_arrays([[1], None, []], "f2"), col.lists_arrays([[2, 3]], "f4")
    assert_lists(col.concatenate(launch, [a, b]), [[1], None, [], [2, 3]], "f4")
    long = list(range(65 if small else 1025))
    assert_lists(
        col.gather(launch, col.lists_arrays([None, [], long], "i4"), [2, 1, 0, 2]),
        [long, [], None, long],
        "i4",
    )


def check_strings(launch: Launch, small: bool) -> None:
    rows = [None, "", "é🙂", "日本語", "a\x00b", "ß", None] * (1 if small else 10)
    lists = as_lists(strings_arrays(rows))
    indices = np.array([3, 2, 0, 1, -1, len(rows), 4, 5], "i4")
    out = as_strings(col.gather(launch, lists, indices))
    assert strings_values(out) == [
        rows[i].encode() if 0 <= i < len(rows) and rows[i] is not None else None for i in indices
    ]
    combined = as_strings(col.concatenate(launch, [lists, col.head(launch, lists, 3)]))
    assert strings_values(combined) == [r.encode() if r is not None else None for r in rows + rows[:3]]
    mask = col.column_arrays([i % 2 for i in range(len(rows))], "u1")
    out = as_strings(col.apply_boolean_mask(launch, lists, mask))
    assert strings_values(out) == [r.encode() if r is not None else None for r in rows[1::2]]


def check_bool(launch: Launch, small: bool) -> None:
    c = col.column_arrays([True, None, False, True, None])
    assert_column(col.cast(launch, c, "u1"), [1, None, 0, 1, None], "u1")
    assert_column(col.gather(launch, c, [3, 2, 1, -1]), [1, 0, None, None], "u1")
    assert col.reduce(launch, c, col.Reduction.SUM) == 2
    assert col.reduce(launch, c, col.Reduction.MIN) == 0
    assert col.reduce(launch, c, col.Reduction.MAX) == 1
    for kind, expected in (
        (col.Scan.SUM, [1, None, 1, 2, None]),
        (col.Scan.MIN, [1, None, 0, 0, None]),
        (col.Scan.MAX, [1, None, 1, 1, None]),
    ):
        out = col.scan(launch, c, kind, chunk=2)
        assert_column(out, expected, "u8" if kind is col.Scan.SUM else "u1")


@dataclass(frozen=True)
class Operation:
    name: str
    kernels: tuple[str, ...]
    check: Check


OPERATIONS = [
    *(
        Operation(f"cast-{a}-{b}", ("copy_range",), partial(check_cast, source=a, target=b))
        for a in DTYPES
        for b in DTYPES
    ),
    *(Operation(f"copy-{d}", ("copy_range",), partial(check_copy, dtype=d)) for d in DTYPES),
    *(Operation(f"gather-{d}", ("gather_values",), partial(check_gather, dtype=d)) for d in DTYPES),
    Operation("scatter", ("scatter_values",), check_scatter),
    Operation("choose", ("choose_values",), check_choose),
    Operation("choose-f64", (), check_choose_f64),
    Operation("nulls", ("null_flags", "replace_nulls", "fill_values"), check_nulls),
    *(
        Operation(f"concatenate-{a}-{b}", (), partial(check_concatenate, pair=(a, b, out)))
        for a, b, out in (("u1", "i1", "i2"), ("f2", "f4", "f4"), ("i4", "f4", "f8"))
    ),
    *(
        Operation(
            f"reductions-{d}",
            ("fold_sum", "fold_min", "fold_max", "fold_count", "first_equal"),
            partial(check_reductions, dtype=d),
        )
        for d in DTYPES
    ),
    *(Operation(f"precision-{acc}", (), partial(check_precision, acc=acc)) for acc in ("f4", "f8")),
    *(
        Operation(
            f"scans-{d}",
            ("scan_sum", "scan_min", "scan_max", "scan_add_sum", "scan_add_min", "scan_add_max"),
            partial(check_scans, dtype=d),
        )
        for d in DTYPES
    ),
    Operation("filter", ("selected_flags", "valid_flags", "selected_rows"), check_filter),
    Operation("lists", ("gather_lengths", "gather_lists", "copy_lists"), check_lists),
    Operation("strings", (), check_strings),
    Operation("bool", (), check_bool),
]


@pytest.mark.parametrize("op", OPERATIONS, ids=lambda op: op.name)
def test_c(op: Operation) -> None:
    op.check(c_launch, False)


@pytest.mark.parametrize("op", OPERATIONS, ids=lambda op: op.name)
def test_emulator(op: Operation) -> None:
    op.check(emulator_launch, True)


@pytest.fixture(scope="module")
def metal_launch() -> Launch:
    if metal is None:
        pytest.skip("Metal runtime is unavailable on this host")
    backend = metal
    try:
        gpu = backend.MetalDevice(Target.of("apple_m4"))
    except (OSError, backend.MetalError) as error:
        pytest.skip(f"no Metal device: {error}")

    @cache
    def program(name: str, binding: Binding) -> MetalRun:
        return MetalRun(gpu, capture(FILE, name, target=gpu.target, dtypes=dict(binding)))

    def launch(name: str, rows: int, arguments: list[Argument]) -> None:
        binding = tuple(bindings(FILE, name, arguments).items())
        if any((dtype.bits == 64 and dtype.is_float) or dtype == BF16 for _, dtype in binding):
            pytest.skip("Metal has no f64 or bf16 storage or arithmetic")
        program(name, binding).run(Items(rows), arguments)

    return launch


@pytest.fixture(scope="module")
def card_launch(gpu: CUDADevice) -> Launch:
    card = gpu

    @cache
    def program(name: str, binding: Binding) -> CUDARun:
        return CUDARun(card, capture(FILE, name, target=card.target, dtypes=dict(binding)))

    def launch(name: str, rows: int, arguments: list[Argument]) -> None:
        program(name, tuple(bindings(FILE, name, arguments).items())).run(Items(rows), arguments)

    return launch


@pytest.mark.parametrize("op", OPERATIONS, ids=lambda op: op.name)
def test_metal(op: Operation, metal_launch: Launch) -> None:
    op.check(metal_launch, False)


@pytest.mark.parametrize("op", OPERATIONS, ids=lambda op: op.name)
def test_card(op: Operation, card_launch: Launch) -> None:
    op.check(card_launch, False)


def representative_binding(name: str, dtype: np.dtype) -> dict[str, Dtype]:
    binding: dict[str, Dtype] = {}
    for arg in Module.read(FILE).kernel(name).args.args:
        match arg.annotation:
            case ast.Subscript(value=ast.Name(id=base), slice=ast.Name(id=element)):
                chosen: np.dtype = (
                    np.dtype("i4")
                    if element == "I"
                    else dtype
                    if element in "ABCD"
                    else np.dtype({"u8": "u1", "u32": "u4", "u64": "u8", "i32": "i4", "i64": "i8"}[element])
                )
                binding[arg.arg if base == "Vector" else f"{arg.arg}.values"] = DEVICE_DTYPES[chosen]
    return binding


@pytest.mark.parametrize("arch", ("sm_120a", "gfx942"))
@pytest.mark.parametrize("dtype", ("f2", "f4", "i8", "u1"))
@pytest.mark.parametrize("name", sorted({k for op in OPERATIONS for k in op.kernels}))
def test_render(name: str, dtype: str, arch: str) -> None:
    target = targets.Target.of(arch)
    kernel = capture(FILE, name, dtypes=representative_binding(name, np.dtype(dtype)), target=target)
    source = emit(kernel, target).source
    assert name in source and "__global__" in source


def test_a_half_copy_on_cuda_moves_the_bits_and_keeps_nan_payloads() -> None:
    target = targets.Target.of("sm_120a")
    dtypes = {"src.values": F16, "out.values": F16}
    source = emit(capture(FILE, "copy_range", dtypes=dtypes, target=target), target).source
    assert "out.values[(long long)((base + i))] = src.values[(long long)((start + i))];" in source


def test_emulator_bf16() -> None:
    values = np.array([0.1, 1.01, -2.03, 7.1], "f4")
    src, rounded_src, out = (
        col.column_arrays(values),
        col.new_column(len(values), "f4"),
        col.new_column(1, "f4"),
    )
    target = targets.Target.of("sm_90a")
    convert = capture(FILE, "copy_range", dtypes={"src.values": F32, "out.values": BF16}, target=target)
    emulate(convert, target).run(Items(src.rows), [src._asdict(), rounded_src._asdict(), 0, 0, src.rows])
    kernel = capture(FILE, "fold_sum", dtypes={"src.values": BF16, "out.values": F32}, target=target)
    emulate(kernel, target).run(Items(1), [rounded_src._asdict(), out._asdict()])
    bits = values.view("u4")
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view("f4")
    assert_column(rounded_src, rounded.tolist(), "f4")
    assert_column(out, [np.sum(rounded, dtype=np.float32).item()], "f4")


class Bf16(np.ndarray):
    """float32 host storage that a launch binds as bf16: numpy has no bfloat16 and only the emulator carries one."""


def bf16_column(values: np.ndarray, valid: np.ndarray) -> ColumnArrays:
    bits = values.astype("f4").view("u4")
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view("f4")
    c = col.column_arrays([v.item() if good else None for v, good in zip(rounded, valid, strict=True)], "f4")
    return c._replace(values=c.values.view(Bf16))


def bf16_launch(name: str, rows: int, arguments: list[Argument]) -> None:
    binding = bindings(FILE, name, arguments)
    names = [parameter.arg for parameter in Module.read(FILE).kernel(name).args.args]
    for parameter, argument in zip(names, arguments, strict=True):
        if isinstance(argument, dict) and isinstance(argument["values"], Bf16):
            binding[f"{parameter}.values"] = BF16
    emulator_program(name, tuple(binding.items())).run(Items(rows), arguments)


def test_emulator_bf16_inputs_across_the_library() -> None:
    rng = np.random.default_rng(23)
    values, valid = rng.normal(size=70).astype("f4"), rng.random(70) > 0.2
    c = bf16_column(values, valid)
    rows = col.column_values(c)
    present = np.array([v for v in rows if v is not None], "f4")
    total = col.reduce(bf16_launch, c, col.Reduction.SUM, "f4")
    assert total is not None and abs(total - present.sum(dtype=np.float64)) <= 1e-5 * np.abs(present).sum()
    assert col.reduce(bf16_launch, c, col.Reduction.MIN) == present.min()
    assert col.reduce(bf16_launch, c, col.Reduction.MAX) == present.max()
    assert col.reduce(bf16_launch, c, col.Reduction.COUNT) == len(present)
    out = col.scan(bf16_launch, c, col.Scan.MAX, chunk=4)
    expected = iter(np.maximum.accumulate(present).tolist())
    assert_column(out, [None if v is None else next(expected) for v in rows], "f4")
    index = np.array([69, 0, 5, -1, 70], "i4")
    assert_column(
        col.gather(bf16_launch, c, index),
        [rows[i] if 0 <= i < len(rows) else None for i in index],
        "f4",
    )
    other = col.column_arrays(np.arange(70, dtype="f4"))
    cond = col.column_arrays(np.arange(70) % 2, "u1")
    chosen = col.copy_if_else(bf16_launch, cond, c, other)
    assert_column(
        chosen,
        [rows[i] if i % 2 and rows[i] is not None else (None if i % 2 else float(i)) for i in range(70)],
        "f4",
    )
    assert col.column_values(col.fill_null(bf16_launch, c, 0.0)) == [0.0 if v is None else v for v in rows]


def test_kernel_table_is_complete() -> None:
    kernels = {
        fn.name
        for fn in ast.parse(FILE.read_text()).body
        if isinstance(fn, ast.FunctionDef)
        and any(isinstance(d, ast.Name) and d.id == "kernel" for d in fn.decorator_list)
    }
    entries = {name for op in OPERATIONS for name in op.kernels}
    reachable = set(entries)
    functions = {fn.name: fn for fn in ast.parse(FILE.read_text()).body if isinstance(fn, ast.FunctionDef)}
    pending = list(entries)
    while pending:
        for call in ast.walk(functions[pending.pop()]):
            if isinstance(call, ast.Name):
                name = call.id
                if name in functions and name not in reachable:
                    reachable.add(name)
                    pending.append(name)
    assert kernels <= reachable
    assert entries <= kernels


def test_host_contract() -> None:
    for values in ([], [None], [True, False], np.array([True, False])):
        c = col.column_arrays(values)
        assert c.values.dtype == (np.dtype("f8") if not len(values) or values[0] is None else np.dtype("u1"))
    assert col.common_type("f2", "i2") == np.dtype("f4")
    c = col.column_arrays([1, None], "i4")
    args = [fresh(c._asdict())]
    assert args[0] is not c
    with pytest.raises(ValueError):
        col.new_column(-1, "i4")
    with pytest.raises(ValueError):
        col.common_type()
    with pytest.raises(TypeError):
        col.column_arrays([1], "c8")

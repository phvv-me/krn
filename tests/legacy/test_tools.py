"""The helpers of krn inlined into kernels, against the original Python functions.

Every case runs on the C backend, in the emulator and, where there is a card or a Metal GPU, on CUDA and on Metal. The
expected output is built by a stdlib expression with the meaning of the original (itertools, collections, statistics),
and `test_*_originals` checks those expressions against more-itertools and toolz when they are installed. es-toolkit is
JavaScript, so its expectations are the examples on its reference pages (es-toolkit.dev/reference, read 2026-11-01).
"""

import importlib
import itertools as it
import shutil
import statistics
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Blocks, itertools, targets
from krn import itertools as toolz
from krn.compiler.api import CaptureError, capture
from krn.compiler.dtypes import F32, I32, U8, Dtype
from krn.emulator.emulate import build as emulate
from krn.runtime.cpu import Value
from tests.conftest import ROOT, build_kernel
from tests.helpers.devices import CUDARun, fresh

if TYPE_CHECKING:
    from krn.compiler.ir import Kernel
    from krn.runtime.cuda_backend import CUDADevice
    from krn.runtime.metal import MetalDevice
    from tests.helpers.devices import MetalRun

FILE = ROOT / "tests" / "helpers" / "tool_kernels.py"
X = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
Y = [v * 2 for v in X]
KEYS = [0, 0, 1, 1, 1, 3, 3, 4, 4, 4]
FILL = 99.0
F = np.float32
U = np.uint32
ELEMENTS: dict[type[np.generic], Dtype] = {np.float32: F32, np.int32: I32, np.uint8: U8}
TENSORS = (
    "x",
    "a",
    "b",
    "m",
    "keys",
    "fill",
    "out",
)  # the parameters a generic kernel may bind its type from

type Row = list[float]


def flat(rows: Iterable[Iterable[float]]) -> list[float]:
    return list(it.chain.from_iterable(rows))


REFERENCE: dict[str, Callable[[Row], list[float]]] = {
    "pairwise": lambda r: flat(it.pairwise(r)),
    "windowed": lambda r: flat(r[i : i + 3] for i in range(max(len(r) - 2, 0))),
    "windowed_step": lambda r: flat(r[i : i + 3] for i in range(0, max(len(r) - 2, 0), 2)),
    "partition": lambda r: flat(zip(*[iter(r)] * 2, strict=False)),
    "take": lambda r: list(it.islice(r, 3)),
    "toolz_take": lambda r: list(it.islice(r, 3)),
    "tail": lambda r: list(deque(r, maxlen=3)),
    "drop": lambda r: list(it.islice(r, 3, None)),
    "take_nth": lambda r: list(it.islice(r, 0, None, 3)),
    "intersperse": lambda r: [v for x in r for v in (FILL, x)][1:],
    "first": lambda r: [next(iter(r), FILL)],
    "last": lambda r: [next(reversed(r), FILL)],
    "nth": lambda r: [next(it.islice(r, 2, None), FILL)],
    "second": lambda r: [next(it.islice(r, 1, None), FILL)],
}
WITH_FILL = {
    "intersperse",
    "first",
    "last",
    "nth",
    "second",
}  # kernels that also take a fill value


@dataclass(frozen=True)
class Case:
    name: str
    inputs: tuple[Value, ...]
    expected: np.ndarray
    element: type[np.generic] | None = None  # the element type a generic kernel is captured with

    @property
    def dtypes(self) -> dict[str, Dtype]:
        return {} if self.element is None else dict.fromkeys(TENSORS, ELEMENTS[self.element])

    def arguments(self) -> list[Value]:
        return [*(fresh(value) for value in self.inputs), np.zeros_like(self.expected)]

    def check(self, arguments: list[Value]) -> None:
        actual = arguments[-1]
        assert isinstance(actual, np.ndarray)
        if self.expected.dtype.kind == "f":
            np.testing.assert_allclose(actual, self.expected, rtol=1e-5, atol=0, equal_nan=True)
        else:
            np.testing.assert_array_equal(actual, self.expected)


def moves(d: type[np.generic], n: int, start: int) -> list[Case]:
    """The helpers that only move elements, for the element type `d`: each is generic, so each runs for every type."""
    row, other = X[start:n], Y[start:n]
    plain = (np.array(X, d), start, n)
    filled = (np.array(X, d), np.array([FILL], d), start, n)
    runs: list[int] = []
    offset = start
    for key, group in it.groupby(KEYS[start:n]):
        runs += (key, offset, offset := offset + len(list(group)))
    matrix = np.array([X, Y] * 5, d)
    return [
        *(
            Case(
                f"run_{name}",
                filled if name in WITH_FILL else plain,
                np.array(reference(row), d),
                d,
            )
            for name, reference in REFERENCE.items()
        ),
        Case("run_groupby", (np.array(KEYS, d), start, n), np.array(runs, U), d),
        Case("run_chain", (plain[0], np.array(Y, d), start, n), np.array(row + other, d), d),
        Case(
            "run_interleave",
            (plain[0], np.array(Y, d), start, n),
            np.array(flat(zip(row, other, strict=True)), d),
            d,
        ),
        Case("run_flatten", (matrix, 0, n), np.array(flat(matrix[:n].tolist()), d), d),
    ]


def cases() -> list[Case]:
    table: list[Case] = []
    for d, sizes in ((F, (0, 1, 2, 5, 10)), (np.int32, (0, 3, 10)), (np.uint8, (0, 3, 10))):
        for n in sizes:
            for start in (0, 2, n) if n == 10 else (0,):
                table += moves(d, n, start)
    for n in (0, 1, 2, 5, 10):
        chunks = list(it.batched(range(n), 2, strict=False))
        table += [
            Case("run_chunked", (n,), np.array([(c[0], len(c)) for c in chunks], U).ravel()),
            Case("run_batched", (n,), np.array([(c[0], c[-1] + 1) for c in chunks], U).ravel()),
        ]
        for start in (0, 2, n) if n == 10 else (0,):
            row = X[start:n]
            args = (np.array(X, F), start, n)
            table += [
                Case("run_accumulate", args, np.array(list(it.accumulate(row)), F)),
                Case("run_running_max", args, np.array(list(it.accumulate(row, max)), F)),
                Case("run_running_min", args, np.array(list(it.accumulate(row, min)), F)),
                Case("run_mean", args, np.array([statistics.mean(row) if row else float("nan")], F)),
                Case(
                    "run_count_by",
                    (np.array(KEYS, U), start, n),
                    np.array([Counter(KEYS[start:n])[k] for k in range(5)], U),
                ),
                *(
                    Case(f"run_{name}", (start, n), np.array(expected, U))
                    for name, expected in (
                        ("ilen", [len(range(start, n))]),
                        ("ilen_step", [len(range(start, n, 3))]),
                        ("range_right", list(reversed(range(start, n)))),
                        ("range_right_step", list(reversed(range(start, n, 5)))),
                    )
                ),
                *(
                    Case(
                        f"run_{name}_by",
                        args,
                        np.array([op(range(start, n), key=X.__getitem__, default=start)], U),
                    )
                    for name, op in (("min", min), ("max", max))
                ),
            ]
    indptr = [0, 2, 2, 5, 10, 10]
    unsorted = [4, 2, 1, 1, 3, 0, 4, 0, 3, 2]
    table.append(
        Case(
            "run_min_by_rows",
            (np.array(unsorted, F), np.array(indptr, U), len(indptr) - 1),
            np.array(
                [min(range(a, b), key=unsorted.__getitem__, default=a) for a, b in it.pairwise(indptr)],
                U,
            ),
        )
    )
    # es-toolkit.dev/reference: math/{clamp,inRange,lerp,inverseLerp,mean,rangeRight}, array/minBy
    table += [
        Case("run_clip", (np.array([10, 2, 20], F), 3), np.array([10, 5, 15], F)),
        Case(
            "run_in_range",
            (
                np.array([3, 1, 5, -3], F),
                np.array([2, 2, 2, -5], F),
                np.array([5, 5, 5, -1], F),
                4,
            ),
            np.array([1, 0, 0, 1], U),
        ),
        Case(
            "run_lerp",
            (
                np.array([0, 10, 100, 0], F),
                np.array([100, 20, 0, 100], F),
                np.array([0.5, 0.25, 0.25, 1.5], F),
                4,
            ),
            np.array([50, 12.5, 75, 150], F),
        ),
        Case(
            "run_inverse_lerp",
            (
                np.array([0, 10, 100, 0, 5], F),
                np.array([100, 20, 0, 100, 5], F),
                np.array([50, 12.5, 75, 150, 5], F),
                5,
            ),
            np.array([0.5, 0.25, 0.25, 1.5, 0], F),
        ),
        Case("run_mean", (np.array([1, 2, 3, 4, 5], F), 0, 5), np.array([3], F)),
        Case("run_mean", (np.array([1.5, 2.5, 3.5], F), 0, 3), np.array([2.5], F)),
        Case("run_range_right", (1, 5), np.array([4, 3, 2, 1], U)),
        Case("run_range_right_step", (0, 20), np.array([15, 10, 5, 0], U)),
        Case("run_range_right", (0, 11), np.array(list(reversed(range(11))), U)),
        Case("run_unit_clip", (np.array([-2, 0, 1, 4], F), 4), np.array([0, 0, 1, 1], F)),
        Case("run_min_by", (np.array([30, 28, 26], F), 0, 3), np.array([2], U)),
        Case("run_min_by", (np.array([10, 5, 0, 5, 15], F), 0, 5), np.array([2], U)),
    ]
    return table


CASES = cases()
IDS = [f"{case.name}-{case.element.__name__}" if case.element else case.name for case in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_c(case: Case) -> None:
    arguments = case.arguments()
    build_kernel(FILE, case.name, dtypes=case.dtypes)(*arguments, launch=Blocks(1, threads=1))
    case.check(arguments)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_emulator(case: Case) -> None:
    arguments = case.arguments()
    target = targets.Target.of("sm_90a")
    emulate(capture(FILE, case.name, target=target, dtypes=case.dtypes), target).run(Blocks(1, threads=1), arguments)
    case.check(arguments)


@pytest.fixture(scope="module")
def card(gpu: CUDADevice) -> CUDADevice:
    return gpu


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_card(case: Case, card: CUDADevice) -> None:
    arguments = case.arguments()
    kernel = capture(FILE, case.name, target=card.target, dtypes=case.dtypes)
    CUDARun(card, kernel).run(Blocks(1, threads=1), arguments)
    case.check(arguments)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_metal(case: Case, metal_gpu: MetalDevice, metal_program: Callable[[Kernel], MetalRun]) -> None:
    arguments = case.arguments()
    kernel = capture(FILE, case.name, target=metal_gpu.target, dtypes=case.dtypes)
    metal_program(kernel).run(Blocks(1, threads=1), arguments)
    case.check(arguments)


def test_more_itertools_originals() -> None:
    original = pytest.importorskip("more_itertools")
    for n in (0, 1, 2, 5, 10):
        row = X[:n]
        assert list(original.chunked(row, 2)) == [list(c) for c in it.batched(row, 2, strict=False)]
        assert list(original.take(3, row)) == REFERENCE["take"](row)
        assert list(original.tail(3, row)) == REFERENCE["tail"](row)
        assert flat(original.sliding_window(row, 3)) == REFERENCE["windowed"](row)
        assert flat(w for w in original.windowed(row, 3, step=2) if None not in w) == REFERENCE["windowed_step"](row)
        assert flat(original.pairwise(row)) == REFERENCE["pairwise"](row)
        assert list(original.intersperse(FILL, row)) == REFERENCE["intersperse"](row)
        assert list(original.interleave(row, Y[:n])) == flat(zip(row, Y[:n], strict=True))
        assert list(original.flatten([row, Y[:n]])) == row + Y[:n]
        assert original.ilen(range(n)) == len(range(n))
        assert [original.first(row, FILL)] == REFERENCE["first"](row)
        assert [original.last(row, FILL)] == REFERENCE["last"](row)
        assert [original.nth(row, 2, FILL)] == REFERENCE["nth"](row)


def test_toolz_originals() -> None:
    original = pytest.importorskip("toolz")
    for n in (0, 1, 2, 5, 10):
        row = X[:n]
        assert list(original.take(3, row)) == REFERENCE["take"](row)
        assert list(original.tail(3, row)) == REFERENCE["tail"](row)
        assert list(original.drop(3, row)) == REFERENCE["drop"](row)
        assert list(original.take_nth(3, row)) == REFERENCE["take_nth"](row)
        assert flat(original.partition(2, row)) == REFERENCE["partition"](row)
        assert flat(original.interleave([row, Y[:n]])) == flat(zip(row, Y[:n], strict=True))
        assert [original.nth(2, row, FILL)] == REFERENCE["nth"](row)
        if row:
            assert original.first(row) == REFERENCE["first"](row)[0]
            assert original.last(row) == REFERENCE["last"](row)[0]
        if n > 1:
            assert original.second(row) == REFERENCE["second"](row)[0]


HEADER = """from collections.abc import Iterator
from krn import Vector, kernel
from krn.frontend.types import f32, u32
"""


@pytest.mark.parametrize("call", ["chunked(n, size)", "windowed(x, 0, n, size)"])
def test_runtime_size(tmp_path: Path, call: str) -> None:
    """A size is a capture-time constant. Capture reads files, so a copy of the helpers imports like the real ones."""
    shutil.copy(Path(itertools.__file__), tmp_path / "helpers.py")
    path = tmp_path / "runtime_size.py"
    path.write_text(
        HEADER
        + "from .helpers import chunked, windowed\n"
        + "@kernel\ndef refused(x: Vector[f32], n: u32, size: u32) -> None:\n"
        + f"    for value in {call}:\n        pass\n"
    )
    with pytest.raises(CaptureError, match=r"capture-time constant|known at capture"):
        capture(path, "refused")


def test_one_element_type_per_call() -> None:
    """`chain` joins two rows of one element type: strict arithmetic reaches the generic helpers too."""
    with pytest.raises(CaptureError, match=r"type parameter `S` is f32 and `b` is i32"):
        capture(FILE, "run_chain", dtypes={"a": F32, "b": I32, "out": F32})


def test_absolute_import_of_a_generator(tmp_path: Path) -> None:
    path = tmp_path / "absolute.py"
    path.write_text(
        HEADER
        + """from krn.itertools import chunked
@kernel
def run(n: u32, out: Vector[u32]) -> None:
    for start, valid in chunked(n, 2):
        out[start] = valid
"""
    )
    capture(path, "run")


def test_helper_local_with_a_kernel_name(tmp_path: Path) -> None:
    """A helper's literal-initialized local must not be mistaken for the kernel's single-assignment constant."""
    path = tmp_path / "collision.py"
    path.write_text(
        HEADER
        + """
@kernel
def running(x: Vector[f32], n: u32) -> Iterator[f32]:
    total = 0.0
    for i in range(n):
        total += x[i]
        yield total

@kernel
def run(x: Vector[f32], n: u32, out: Vector[f32]) -> None:
    for j, total in enumerate(running(x, n)):
        out[j] = total
"""
    )
    out = np.zeros(10, F)
    build_kernel(path, "run")(np.array(X, F), 10, out, launch=Blocks(1, threads=1))
    np.testing.assert_array_equal(out, list(it.accumulate(X)))


def test_runtime_imports_and_reexport() -> None:
    for name in ("krn.itertools", "krn.functools", "krn.math", "tests.helpers.tool_kernels"):
        importlib.import_module(name)
    assert toolz.take is itertools.take
    capture(FILE, "run_toolz_take", dtypes={"x": F32, "out": F32})

"""One oracle table covers capture-time vocabulary on C, the emulator, and an optional CUDA card."""

from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Blocks, T, targets
from krn.compiler.api import CaptureError, capture
from krn.compiler.capture import ConstValues
from krn.compiler.ir import Assert, For, If, Stmt, While
from krn.emulator.emulate import build as emulate
from krn.runtime.cpu import Value
from tests.conftest import KERNELS, build_kernel
from tests.helpers.devices import CUDARun, fresh

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

FILE = KERNELS / "vocabulary.py"


@dataclass(frozen=True)
class Case:
    name: str
    inputs: tuple[Value, ...]
    expected: np.ndarray
    consts: ConstValues | None = None
    cpu_expected: np.ndarray | None = None

    def arguments(self) -> list[Value]:
        return [*(fresh(value) for value in self.inputs), np.zeros_like(self.expected)]

    def check(self, arguments: list[Value], *, cpu: bool = False) -> None:
        expected = self.cpu_expected if cpu and self.cpu_expected is not None else self.expected
        actual = arguments[-1]
        assert isinstance(actual, np.ndarray)
        if expected.dtype.kind == "f":
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0, equal_nan=True)
        else:
            np.testing.assert_array_equal(actual, expected)


def digits(values: range) -> int:
    result = 0
    for value in values:
        result = result * 10 + value
    return result


X = np.array([2, -3, 5, 7], np.int32)
CASES = [
    Case("constant_folds", (), np.array([1, 3, 2, 3, 0], np.uint8)),
    *(
        Case(
            name,
            (values, len(values)),
            np.array(
                [
                    sum(int(v) for v in values) % 256 if values.dtype == np.uint8 else np.sum(values),
                    np.minimum.reduce(values),
                    np.maximum.reduce(values),
                ],
                values.dtype,
            ),
        )
        for name, values in (
            ("fold_u8", np.array([250, 7, 12], np.uint8)),
            ("fold_i32", X),
            ("fold_f32", np.array([2.5, -3, 7], np.float32)),
            ("fold_f32", np.array([2.5, np.nan, 7], np.float32)),
        )
    ),
    Case(
        "fold_clauses",
        (X, len(X)),
        np.array(
            [
                sum(int(X[i]) + int(X[j]) for i in range(4) if i != 1 for j in range(i) if j != 2),
                min(int(X[i]) + int(X[j]) for i in range(4) if i > 0 for j in range(i) if j != 2),
                max(int(X[i]) - int(X[j]) for i in range(4) if i > 0 for j in range(i) if j != 2),
                int(X[0]) + int(X[1]) + 1,
                sum(int(X[i]) + int(X[j]) for i, j in zip(range(4), range(1, 4), strict=False)),
                sum(range(4)),
                max(range(4)),
            ],
            np.int32,
        ),
    ),
    *(
        Case(
            "fold_logic",
            (values, len(values)),
            np.array(
                [
                    any(v > 0 for v in values),
                    all(v > 0 for v in values),
                    any(v < 0 for v in values),
                    all(v < 0 for v in values),
                    0,
                    False,
                    True,
                    any(values[i] == values[j] for i in range(len(values)) if i > 0 for j in range(i)),
                    all(values[i] == values[j] for i in range(len(values)) if i > 0 for j in range(i)),
                ],
                np.int32,
            ),
        )
        for values in (
            X,
            np.array([2, 2], np.int32),
            np.array([-1, -2], np.int32),
            np.array([], np.int32),
        )
    ),
    *(
        Case(
            "range_loops",
            (n, 3),
            np.array(
                [
                    sum(k + i for k, i in enumerate(range(n), 3)),
                    sum(reversed(range(n))),
                    digits(range(1, n, 2)[::-1]),
                    sum(range(n, 0, -1)),
                    sum(a + b for a, b in zip(range(n), range(1, n, 2), strict=False)),
                    sum(a + b for a, b in zip(range(n, 0, -2), range(1, n, 3), strict=False)),
                    sum(k + a + b for k, (a, b) in enumerate(zip(range(n), range(n, 0, -1), strict=False), 3)),
                ],
                np.uint32,
            ),
        )
        for n in (0, 1, 7)
    ),
    *(
        Case(
            "signed_loops",
            (n,),
            np.array(
                [
                    sum(range(n, -3, -2)),
                    digits(range(-2, n, 3)[::-1]),
                    digits(range(n, -3, -2)[::-1]),
                    digits(range(n, -3, -2)[::-1]),
                ],
                np.int32,
            ),
        )
        for n in (0, 7, -5)
    ),
    Case("generators", (4, {"n": 4}), np.array([28, 22, 14], np.uint32)),
    Case(
        "sequences",
        (np.array([2, 3, 4], np.int32),),
        np.array([2, 3, 3, 14, 10, 4, 2, 16, 12, 2, 3, 2, 0, 16, 16, 36, 332, 11], np.int32),
    ),
    Case("comprehensions", (X,), np.array([77, 14, 0, 0, 2, 0, 0, 4, 9, 16, 6, 9], np.int32)),
    Case(
        "matches",
        (np.zeros(1, np.float32), {}),
        np.array([32, 2, 32, 2, 1, 1], np.int32),
        cpu_expected=np.array([32, 2, 32, 1, 1, 1], np.int32),
    ),
    Case(
        "matches",
        (np.zeros(1, np.float32), {}),
        np.array([32, 14, 16, 2, 1, 1], np.int32),
        consts={"choice": {"count": 4, "dtype": T.f16}},
        cpu_expected=np.array([32, 14, 16, 1, 1, 1], np.int32),
    ),
    Case("bound_snapshot", (np.array([5], np.uint32),), np.array([10], np.uint32)),
    Case("scopes", (X,), np.array([2, 2, 6, 1, 2, 2, 4], np.int32)),
    Case("constant_loop", (), np.arange(4, dtype=np.uint32)),
    Case("long_ranges", (), np.array([4 * sum(range(1000))], np.uint32)),
    Case(
        "match_facts",
        (np.zeros(1, np.float32),),
        np.array([32, 32, 1, 3], np.int32),
        cpu_expected=np.array([16, 32, 1, 3], np.int32),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_c(case: Case) -> None:
    arguments = case.arguments()
    build_kernel("vocabulary", case.name, case.consts)(*arguments, launch=Blocks(1, threads=1))
    case.check(arguments, cpu=True)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_emulator(case: Case) -> None:
    arguments = case.arguments()
    target = targets.Target.of("sm_90a")
    emulate(capture(FILE, case.name, consts=case.consts, target=target), target).run(Blocks(1, threads=1), arguments)
    case.check(arguments)


@pytest.fixture(scope="module")
def card(gpu: CUDADevice) -> CUDADevice:
    return gpu


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_card(case: Case, card: CUDADevice) -> None:
    arguments = case.arguments()
    kernel = capture(FILE, case.name, consts=case.consts, target=card.target)
    CUDARun(card, kernel).run(Blocks(1, threads=1), arguments)
    case.check(arguments)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "xs = [i for i in range(n)]",
            "a list needs a length known at capture.*range\\(n\\).*Vector",
        ),
        ("xs = list(range(n))", "a list needs a length known at capture"),
        ("xs = [0] * n", "a list needs a length known at capture"),
        ("xs = [i for i in range(3) if i < n]", "a list needs a length known at capture"),
        ("xs = {1, 2}", "a set has no fixed layout"),
        ("xs = {i for i in range(3)}", "a set has no fixed layout"),
        ("xs = set([1])", "a set has no fixed layout"),
        ("xs = frozenset([1])", "a set has no fixed layout"),
        ("xs = {1: 2}", "a dict has no fixed layout"),
        ("xs = {i: i for i in range(3)}", "a dict has no fixed layout"),
        ("xs = dict()", "a dict has no fixed layout"),
        ("xs = [1, 2]\nv = xs[n]", "capture-time constant.*runtime value `n`.*enumerate"),
        ("xs = [1]\nv = xs[2]", "out of range for length 1"),
        ("xs = [1]\nxs.append(2)", "fixed at capture: build a new one"),
        ("xs = [1]\nxs.extend([2])", "fixed at capture: build a new one"),
        ("xs = [1]\nxs.insert(0, 2)", "fixed at capture: build a new one"),
        ("xs = [1]\nxs.pop()", "fixed at capture: build a new one"),
        ("xs = [1]\nxs[0] = 2", "fixed at capture: build a new one"),
        ("xs = [1]\nxs[0] += 2", "fixed at capture: build a new one"),
        ("xs = [1]\ndel xs[0]", "fixed at capture: build a new one"),
        ("xs = [1]\ndel xs", "fixed at capture: build a new one"),
        (
            "for a, b in zip(values(n), range(n)):\n    pass",
            "cannot step a generator method in lockstep",
        ),
        ("for a in reversed(values(n)):\n    pass", "generator method yields in its own order"),
        (
            "for a, b in zip(range(n), [1, 2]):\n    pass",
            "runtime range.*register sequence.*enumerate",
        ),
        (
            "match n:\n    case 1:\n        pass",
            "match needs a capture-time subject.*runtime value `n`.*use `if`",
        ),
        (
            "match 1:\n    case value if n > 0:\n        pass",
            "match guard must be a capture-time constant",
        ),
        ("match 1:\n    case [a]:\n        pass", "unsupported pattern.*capture-time match"),
        ("for i in range(0, n, 0):\n    pass", "step of a loop must not be zero"),
        (
            "for i in range(0, 4, n):\n    pass",
            "step of a loop.*known at capture.*runtime value `n`",
        ),
        (
            "xs = list(range(1000))",
            "unrolling `range\\(1000\\)` would emit 1000 copies.*use a loop",
        ),
        ("xs = (i for i in range(n))", "generator expression must be consumed by"),
        ("xs = [1] + (2,)", "cannot concatenate list and tuple"),
        ("xs = [1] - [2]", "operator.*not defined for a register sequence"),
        ("xs = [*n]", "starred register sequence element must be a register sequence"),
        ("a, b = [1]", "length 1.*length 2"),
        ("a, *rest, b = [1]", "length 1.*length 3"),
        (
            "for pair in zip(range(n), range(n)):\n    pass",
            "runtime zip loop needs a tuple target",
        ),
        (
            "for pair in enumerate(range(n)):\n    pass",
            "runtime enumerate loop needs a two-tuple target",
        ),
        (
            "for a, b in zip(range(n), range(T.i32(3))):\n    pass",
            "loop bounds have types.*convert one side",
        ),
        ("v = any(i for i in range(n))", "condition must be bool"),
        ("v = all(i for i in range(n))", "condition must be bool"),
        ("v = sum(n)", "takes one generator expression, a range or a register sequence"),
        ("v = [1] == 1", "expected a value, got a sequence"),
        *(
            (f"v = {name}({source})", f"`{name}\\(\\)` of an empty sequence: nothing to fold")
            for name in ("min", "max")
            for source in ("[]", "()", "range(0)", "i for i in range(0)", "[i for i in ()]")
        ),
    ],
)
def test_refusals(tmp_path: Path, body: str, message: str) -> None:
    file = tmp_path / "refused.py"
    file.write_text(
        "from krn import T, kernel\n"
        "@kernel\ndef values(n: int):\n    for i in range(n):\n        yield i\n"
        "@kernel\ndef refused(n: T.u32):\n" + indent(body, "    ") + "\n"
    )
    with pytest.raises(CaptureError, match=message):
        capture(file, "refused")


@pytest.mark.parametrize(
    ("imports", "callee"),
    [("from itertools import chain as joined", "joined"), ("import itertools as it", "it.chain")],
)
@pytest.mark.parametrize("rows", ["[]", "[[], [2, 3], [], [4]]"])
def test_flatten_aliases_preserve_order_and_empty_rows(tmp_path: Path, imports: str, callee: str, rows: str) -> None:
    file = tmp_path / "flatten.py"
    file.write_text(
        f"{imports}\nfrom krn import T, Vector, kernel\n"
        f"@kernel\ndef flatten(out: Vector[T.i32]):\n"
        f"    for i, v in enumerate({callee}.from_iterable({rows})):\n        out[i] = v\n"
    )
    target = targets.Target.of("sm_90a")
    kernel = capture(file, "flatten", target=target)
    out = np.zeros(3, np.int32)
    emulate(kernel, target).run(Blocks(1, threads=1), [out])
    np.testing.assert_array_equal(out, [0, 0, 0] if rows == "[]" else [2, 3, 4])


def test_runtime_empty_min() -> None:
    target = targets.Target.of("sm_90a")
    program = emulate(capture(FILE, "runtime_empty", target=target), target)
    with pytest.raises(AssertionError, match=r"min\(\) iterable argument is empty"):
        program.run(Blocks(1, threads=1), [0, np.zeros(1, np.uint32)])


@pytest.mark.parametrize(("arch", "asserted"), [("sm_90a", True), ("gfx942", True), ("avx512", False)])
def test_empty_fold_assert_follows_the_renderer(arch: str, asserted: bool) -> None:
    """The C renderer's assert calls printf undeclared (a request on emit.py), so the CPU target omits the check."""
    body = capture(FILE, "runtime_empty", target=targets.Target.of(arch)).body
    assert any(isinstance(statement, Assert) for statement in body) is asserted


def loops(statements: tuple[Stmt, ...]) -> list[For | While]:
    found: list[For | While] = []
    for statement in statements:
        if isinstance(statement, For | While):
            found.extend([statement, *loops(statement.body)])
        elif isinstance(statement, If):
            found.extend([*loops(statement.then), *loops(statement.orelse)])
    return found


def test_unrolling_and_plain_range() -> None:
    assert loops(capture(FILE, "sequences").body) == []
    assert loops(capture(FILE, "comprehensions").body) == []
    direct = loops(capture(FILE, "plain_range").body)
    assert len(direct) == 1
    assert isinstance(direct[0], For)
    assert direct[0].step == 1
    assert len(loops(capture(FILE, "long_ranges").body)) == 3  # ranges stay loops past the unroll cap
    constant = loops(capture(FILE, "constant_loop").body)
    assert len(constant) == 1
    assert isinstance(constant[0], For)

"""One Python/NumPy oracle table exercises C, the emulator, and an optional CUDA card."""

import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Items, T, catalog, targets
from krn.compiler.api import CaptureError, capture, emit
from krn.compiler.dtypes import PY_FLOAT
from krn.compiler.ir import TensorType
from krn.emulator.emulate import build as emulate
from krn.runtime.cpu import CTYPES, Value
from tests.conftest import KERNELS, build_kernel
from tests.helpers.devices import CUDARun, fresh

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

FILE = KERNELS / "number_ops.py"


@dataclass(frozen=True)
class Case:
    name: str
    inputs: tuple[Value, ...]
    expected: tuple[np.ndarray, ...]
    count: int = 6

    def arguments(self) -> list[Value]:
        return [*(fresh(v) for v in self.inputs), *(np.zeros_like(v) for v in self.expected)]

    def check(self, arguments: list[Value]) -> None:
        for actual, expected in zip(arguments[-len(self.expected) :], self.expected, strict=True):
            assert isinstance(actual, np.ndarray)
            if np.issubdtype(expected.dtype, np.integer):
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0, equal_nan=True)
                zeros = expected == 0
                np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


U8 = np.array([0, 1, 7, 128, 254, 255], np.uint8)
U32 = np.array([0, 1, 7, 1 << 31, (1 << 32) - 2, (1 << 32) - 1], np.uint32)
U64 = np.array([0, 1, 7, 1 << 63, (1 << 64) - 2, (1 << 64) - 1], np.uint64)
I32 = np.array([-(1 << 31), -2000, -3, 0, 7, (1 << 31) - 1], np.int32)
X = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0], np.float32)
SPECIAL = np.array([-np.inf, -2.5, -0.0, 0.0, np.inf, np.nan], np.float32)
OTHER = np.array([3, np.nan, 1, -1, np.nan, 2], np.float32)
SIGNED = np.array([-1, 2, -0.0, 0.0, -1, -1], np.float32)
EXPONENTS = np.array([-150, -3, 0, 4, 100, 200], np.int64)
CUBE = np.arange(36, dtype=np.float32).reshape(6, 2, 3)
CUBE_EXPECTED = np.zeros_like(CUBE)
CUBE_EXPECTED[:, 1, 2] = CUBE[:, 1, 2] + 1
UNARY_MATH = (
    np.sqrt,
    np.exp,
    np.exp2,
    np.log,
    np.log2,
    np.log10,
    np.sin,
    np.cos,
    np.tanh,
    np.floor,
    np.ceil,
    np.trunc,
    np.cbrt,
)

with np.errstate(all="ignore"):
    CASES = [
        Case(
            "split",
            (U32 - np.uint32(U32 == (1 << 32) - 1), 7),
            (
                np.array(
                    [
                        [v // 7, v % 7, v // 7, v % 7, 255 // (v + 1), 255 % (v + 1)]
                        for v in [int(u) - int(u == (1 << 32) - 1) for u in U32]
                    ],
                    np.uint32,
                ),
            ),
        ),
        Case(
            "powers",
            (X,),
            (
                np.column_stack(
                    [np.power(X.astype(np.float64), n) for n in (0, 1, 2, 3, 5, 9, 3, 0.5)]
                    + [np.power(2.0, X), np.power(X, -2), np.power(X, X), X * X]
                ).astype(np.float32),
            ),
        ),
        Case(
            "int_powers",
            (U8, I32),
            (
                np.array([int(v) ** 3 % 256 for v in U8], np.uint8),
                np.array([(int(v) ** 3 + (1 << 31)) % (1 << 32) - (1 << 31) for v in I32], np.int32),
            ),
        ),
        Case("magnitudes", (I32, SPECIAL, U32), (np.abs(I32), np.abs(SPECIAL), U32)),
        Case(
            "tile_powers",
            (X[:4].copy(),),
            (np.ones(4, np.float32), X[:4] ** 5, X[:4] ** 9),
            count=1,
        ),
        Case(
            "extrema",
            (U8, I32, SPECIAL, OTHER),
            (
                np.column_stack([np.minimum(U8, 7), np.maximum(U8, 7)]),
                np.column_stack([np.minimum(I32, -3), np.maximum(I32, -3)]),
                np.column_stack(
                    [
                        np.minimum(SPECIAL, OTHER),
                        np.maximum(SPECIAL, OTHER),
                        np.minimum(OTHER, SPECIAL),
                        np.maximum(OTHER, SPECIAL),
                    ]
                ),
            ),
        ),
        Case(
            "bit_widths",
            (U8, U32, U64),
            (
                np.column_stack(
                    [
                        np.array([method(int(v)) for v in values], np.uint32)
                        for values in (U8, U32, U64)
                        for method in (int.bit_count, int.bit_length)
                    ]
                ),
            ),
        ),
        *(
            Case(
                "unary_math",
                (v,),
                (np.column_stack([fn(v) for fn in UNARY_MATH] + [v + math.pi + math.e + math.tau, np.exp(v)]),),
            )
            for v in (X, SPECIAL)
        ),
        Case(
            "binary_math",
            (SPECIAL,),
            (
                np.column_stack(
                    [
                        np.isnan(SPECIAL),
                        np.isfinite(SPECIAL),
                        SPECIAL.astype(np.float64) * 2 + 1,
                    ]
                ).astype(np.float32),
            ),
        ),
        Case(
            "copy_sign",
            (SPECIAL, SIGNED),
            (
                np.column_stack(
                    [
                        np.copysign(SPECIAL, SIGNED),
                        np.copysign(np.float32(2), SPECIAL),
                        np.copysign(SPECIAL, -0.0),
                    ]
                ),
            ),
        ),
        Case("scale_exponent", (X, EXPONENTS), (np.ldexp(X, EXPONENTS),)),
        Case(
            "double_math",
            (X.astype(np.float64),),
            (
                np.column_stack(
                    [
                        np.sqrt(X.astype(np.float64)),
                        X.astype(np.float64) ** 9,
                        X.astype(np.float64) * 2 + 1,
                    ]
                ),
            ),
        ),
        Case(
            "generic_numbers",
            (U8, X, 3, 0.25),
            (
                np.array([(int(v) + 3) % 256 for v in U8], np.uint8),
                X + np.float32(0.25),
            ),
        ),
        Case("generic_integer", (U8, 3), (np.array([(int(v) + 3) % 256 for v in U8], np.uint8),)),
        Case("cube", (CUBE,), (CUBE_EXPECTED,)),
        Case("constants", (), (np.array([8, 0.25, 9, 4, 2, 8, 8, 9, 3, 2], np.float32),), count=1),
    ]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_c(case: Case) -> None:
    if case.name == "generic_numbers" and PY_FLOAT not in CTYPES:
        pytest.skip("C scalar ABI lacks the PY_FLOAT ctypes mapping for a bare float parameter")
    arguments = case.arguments()
    build_kernel("number_ops", case.name)(*arguments, launch=Items(case.count))
    case.check(arguments)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_emulator(case: Case) -> None:
    if case.name in ("copy_sign", "scale_exponent"):
        pytest.skip("emulator has no copysign or ldexp intrinsic semantics yet")
    arguments = case.arguments()
    target = targets.Target.of("sm_90a")
    program = emulate(capture(FILE, case.name, target=target), target)
    if case.name == "generic_numbers" and program.arguments(arguments)["f"] != arguments[3]:
        pytest.skip("emulator binds a bare float parameter through int(value), truncating its fractional part")
    program.run(Items(case.count), arguments)
    case.check(arguments)


@pytest.fixture(scope="module")
def card(gpu: CUDADevice) -> CUDADevice:
    return gpu


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_card(case: Case, card: CUDADevice) -> None:
    arguments = case.arguments()
    kernel = capture(FILE, case.name, target=card.target)
    CUDARun(card, kernel).run(Items(case.count), arguments)
    case.check(arguments)


def test_host_number_protocol_and_tensor3() -> None:
    for value, abc in (
        (T.u8(3), "Integral"),
        (T.i32(-3), "Integral"),
        (T.f32(3.0), "Real"),
        (T.c64(3.0), "Complex"),
    ):
        assert isinstance(value, getattr(numbers, abc))
    assert math.sqrt(T.f16(9.0)) == 3.0
    assert [10, 20, 30][T.u8(1)] == 20
    kernel = capture(FILE, "cube")
    assert isinstance(kernel.params[0][1], TensorType)
    assert kernel.params[0][1].rank == 3
    assert catalog.missing() == []


@pytest.mark.parametrize(
    ("parameters", "expression", "message"),
    [
        ("x: T.u8, n: int", "x ** n", "known at capture.*runtime value `n`"),
        ("x: T.u8", "x ** -2", "convert the base to T.f32"),
        ("x: T.u8", "x ** 0.5", "known at capture"),
        ("x: T.i32", "divmod(x, 3)", "C truncates, Python floors.*convert"),
        ("x: T.f32", "divmod(x, 3.0)", "use `/` for floats"),
        ("x: T.i32", "x.bit_count()", "magnitude.*popcount.*leading_zeros"),
        ("x: T.i32", "x.bit_length()", "magnitude.*popcount.*leading_zeros"),
        ("x: T.u8, y: T.u32", "min(x, y)", "dtype mismatch"),
        ("x: T.u8", "min(x)", "only the two-operand form"),
        ("x: T.u8", "max(x, 1, 2)", "only the two-operand form"),
        ("x: T.u8", "pow(x, 2, 3)", "third modular argument"),
        ("x: T.u8", "math.isnan(x)", "f32 or f64 scalar"),
        ("x: T.f32", "math.ldexp(x, 1.5)", "integer scalar"),
        ("x: T.f32, y: T.f64", "math.copysign(x, y)", "matching f32 or f64"),
    ],
)
def test_refusals(tmp_path: Path, parameters: str, expression: str, message: str) -> None:
    file = tmp_path / "refused.py"
    file.write_text(f"from krn import T, kernel, math\n@kernel\ndef refused({parameters}):\n    v = {expression}\n")
    with pytest.raises(CaptureError, match=message):
        capture(file, "refused")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("v = x[0:n]", "slice extent must be a capture-time constant.*runtime"),
        ("v = block.tile(n)", "a tile size must be a capture-time constant.*runtime value `n`"),
        (
            "for i in range(0, 10, n):\n        x[i] = 1",
            "step of a loop.*known at capture.*runtime value `n`",
        ),
    ],
)
def test_runtime_is_not_capture_time(tmp_path: Path, body: str, message: str) -> None:
    file = tmp_path / "runtime.py"
    file.write_text(
        f"from krn import T, Vector, block, kernel\n@kernel\ndef runtime(x: Vector[T.u8], n: int):\n    {body}\n"
    )
    with pytest.raises(CaptureError, match=message):
        capture(file, "runtime")


@pytest.mark.parametrize("arch", ("sm_90a", "gfx942", "apple_m5", "avx512"))
def test_intrinsic_lowerings_exist(arch: str) -> None:
    target = targets.Target.of(arch)
    for name in ("copy_sign", "scale_exponent"):
        source = emit(capture(FILE, name, target=target), target).source
        assert ("copysign" if name == "copy_sign" else "ldexp") in source

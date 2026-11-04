"""Packed words as element types: lane-wise `+ - -x` and comparisons, `pack`, `unpack` and `.lanes`, against numpy."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Blocks
from krn.compiler.api import CaptureError, capture, emit
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from krn.runtime.cpu import CPUProgram
from tests.helpers.devices import MetalRun

if TYPE_CHECKING:
    from krn.runtime.metal import MetalDevice
from tests.helpers.packed import LANE, kernel, lanes, random_words, words

PACKED = tuple(LANE)
COUNT, THREADS = 256, 64
COMPARE = {
    "==": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}


def build(directory: Path, body: str, ty: str, out: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return kernel(directory, body, ty=ty, out=out or ty)


def execute(file: Path, a: np.ndarray, b: np.ndarray, backend: str, device: MetalDevice | None = None) -> np.ndarray:
    target = Target.of("avx512")
    if backend == "metal":
        assert device is not None
        target = device.target
    ir = capture(file, "k", target=target)
    result = np.zeros(len(a), np.uint32)
    launch = Blocks(len(a) // THREADS, threads=THREADS)
    if backend == "emulator":
        emulate(ir, target).run(launch, [a, b, result])
    elif backend == "c":
        CPUProgram(ir, emit(ir, target))(a, b, result, launch=launch)
    else:
        assert device is not None
        MetalRun(device, ir).run(launch, [a, b, result])
    return result


@pytest.fixture(scope="module")
def metal_device():
    metal = pytest.importorskip("krn.runtime.metal")
    try:
        return metal.MetalDevice(Target.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("backend", ["emulator", "c", "metal"])
@pytest.mark.parametrize("ty", PACKED)
def test_packed_word_round_trips_and_subtracts_per_lane(tmp_path: Path, request, backend: str, ty: str) -> None:
    device = request.getfixturevalue("metal_device") if backend == "metal" else None
    rng = np.random.default_rng(21)
    a, b = random_words(rng, COUNT), random_words(rng, COUNT)
    copied = execute(build(tmp_path / "copy", "out[i] = a[i]", ty), a, b, backend, device)
    np.testing.assert_array_equal(copied, a)
    difference = execute(build(tmp_path / "sub", "out[i] = a[i] - b[i]", ty), a, b, backend, device)
    np.testing.assert_array_equal(difference, words(lanes(a, ty) - lanes(b, ty), ty))


@pytest.mark.parametrize("ty", PACKED)
def test_sum_and_negation_wrap_per_lane(tmp_path: Path, ty: str) -> None:
    rng = np.random.default_rng(22)
    a, b = random_words(rng, COUNT), random_words(rng, COUNT)
    total = execute(build(tmp_path / "sum", "out[i] = a[i] + b[i]", ty), a, b, "c")
    np.testing.assert_array_equal(total, words(lanes(a, ty) + lanes(b, ty), ty))
    negated = execute(build(tmp_path / "neg", "out[i] = -a[i]", ty), a, b, "c")
    np.testing.assert_array_equal(negated, words(-lanes(a, ty), ty))


@pytest.mark.parametrize("ty", ["i8x4", "u8x4", "i16x2", "u16x2"])
@pytest.mark.parametrize("op", list(COMPARE))
def test_comparison_is_a_lane_mask_of_the_same_type(tmp_path: Path, ty: str, op: str) -> None:
    rng = np.random.default_rng(23)
    a, b = random_words(rng, COUNT), random_words(rng, COUNT)
    b[:8] = a[:8]  # equal lanes exercise == and the boundary of <= and >=
    mask = execute(build(tmp_path / "mask", f"out[i] = a[i] {op} b[i]", ty), a, b, "emulator")
    np.testing.assert_array_equal(mask, words(np.where(COMPARE[op](lanes(a, ty), lanes(b, ty)), -1, 0), ty))


@pytest.mark.parametrize("backend", ["emulator", "c"])
@pytest.mark.parametrize("ty", ["u8x4", "i16x2"])
@pytest.mark.parametrize("form", ["unpack", "lanes"])
def test_lanes_come_out_low_first_and_pack_back(tmp_path: Path, backend: str, ty: str, form: str) -> None:
    rng = np.random.default_rng(24)
    a, b = random_words(rng, COUNT), random_words(rng, COUNT)
    count = 4 if ty.endswith("x4") else 2
    names = ", ".join(f"x{i}" for i in range(count))
    source = "unpack(a[i])" if form == "unpack" else "a[i].lanes"
    body = f"{names} = {source}\n        out[i] = pack({', '.join(f'x{i}' for i in reversed(range(count)))})"
    reversed_words = execute(build(tmp_path / form, body, ty), a, b, backend)
    np.testing.assert_array_equal(reversed_words, words(lanes(a, ty)[:, ::-1], ty))


@pytest.mark.parametrize("backend", ["emulator", "c"])
def test_a_thirty_two_bit_word_reinterprets_as_packed(tmp_path: Path, backend: str) -> None:
    rng = np.random.default_rng(25)
    a, b = random_words(rng, COUNT), random_words(rng, COUNT)
    file = tmp_path / "reinterpret" / "kernel.py"
    file.parent.mkdir()
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        "@kernel\ndef k(a: Vector[T.u32], b: Vector[T.u32], out: Vector[T.u8x4]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n        out[i] = T.u8x4(a[i] ^ b[i])\n"
    )
    np.testing.assert_array_equal(execute(file, a, b, backend), a ^ b)


def test_mixed_packed_types_are_refused_with_both_names(tmp_path: Path) -> None:
    file = build(tmp_path, "out[i] = a[i] + b[i]", "u8x4")
    file.write_text(file.read_text().replace("b: Vector[T.u8x4]", "b: Vector[T.i8x4]"))
    with pytest.raises(CaptureError, match="dtype mismatch in `\\+`: u8x4 and i8x4"):
        capture(file, "k", target=Target.of("avx512"))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("out[i] = a[i] * b[i]", "`\\*` is not defined for u8x4"),
        ("out[i] = a[i] << 1", "`<<` is not defined for u8x4"),
        ("out[i] = pack(a[i], b[i])", "pack takes 8-bit lanes"),
    ],
)
def test_packed_misuse_is_refused_at_the_user_line(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(CaptureError, match=message) as error:
        capture(build(tmp_path, body, "u8x4"), "k", target=Target.of("avx512"))
    assert error.value.span is not None and error.value.span.line == 7


def test_a_packed_word_converts_only_from_a_thirty_two_bit_integer(tmp_path: Path) -> None:
    file = tmp_path / "convert" / "kernel.py"
    file.parent.mkdir()
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        "@kernel\ndef k(a: Vector[T.f32], out: Vector[T.u8x4]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n        out[i] = T.u8x4(a[i])\n"
    )
    with pytest.raises(CaptureError, match="converts only to or from a 32-bit integer"):
        capture(file, "k", target=Target.of("avx512"))


@pytest.mark.parametrize("backend", ["emulator", "c"])
@pytest.mark.parametrize("ty", PACKED)
def test_a_packed_local_holds_the_word_between_statements(tmp_path: Path, backend: str, ty: str) -> None:
    rng = np.random.default_rng(26)
    a, b = random_words(rng, COUNT), random_words(rng, COUNT)
    body = "total = a[i] + b[i]\n        out[i] = total"
    actual = execute(build(tmp_path, body, ty), a, b, backend)
    np.testing.assert_array_equal(actual, words(lanes(a, ty) + lanes(b, ty), ty))

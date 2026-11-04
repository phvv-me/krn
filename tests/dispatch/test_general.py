"""The general functions of `krn.math` against numpy, on the emulator, the C backend and the Apple GPU, and the
software-fallback warning they raise on GPU targets."""

import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn import Blocks
from krn.compiler.api import CaptureError, capture, emit
from krn.compiler.errors import TargetError
from krn.compiler.ir import Kernel
from krn.compiler.performance import PerformanceWarning
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from krn.math import abs_diff, add_sat, maximum, pack, unpack
from krn.runtime.cpu import CPUProgram
from krn.runtime.settings import configure
from tests.helpers.devices import MetalRun

if TYPE_CHECKING:
    from krn.runtime.metal import MetalDevice
from tests.helpers.packed import LANE, NUMPY, kernel, lanes, random_values, random_words, words

PACKED = tuple(LANE)
PLAIN = ("i8", "u8", "i16", "u16", "i32", "u32")
OPS = ("add_sat", "maximum", "abs_diff")
COUNT, THREADS = 256, 64


def out_type(op: str, ty: str) -> str:
    """abs_diff of a signed type is the unsigned type of the same width; the other operations keep the type."""
    return "u" + ty[1:] if op == "abs_diff" and ty[0] == "i" else ty


def packed_oracle(op: str, ty: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x, y = lanes(a, ty), lanes(b, ty)
    bits = 8 if ty.endswith("x4") else 16
    low, high = (0, (1 << bits) - 1) if ty[0] == "u" else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    match op:
        case "add_sat":
            result = np.clip(x + y, low, high)
        case "maximum":
            result = np.maximum(x, y)
        case _:
            result = np.abs(x - y)
    return words(result.ravel(), ty)


def plain_oracle(op: str, ty: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x, y = a.astype(np.int64), b.astype(np.int64)
    info = np.iinfo(np.dtype(NUMPY[ty]))
    match op:
        case "add_sat":
            result = np.clip(x + y, info.min, info.max)
        case "maximum":
            result = np.maximum(x, y)
        case _:
            result = np.abs(x - y)
    return result.astype(NUMPY[out_type(op, ty)])


def edge_words(ty: str) -> np.ndarray:
    """Words holding both ends of the lane range, so the saturation and wrap edges are always exercised."""
    info = np.iinfo(LANE[ty])
    count = 4 if ty.endswith("x4") else 2
    rows = np.array([[info.min, info.max, 0, info.max][:count], [info.max, info.min, -1, 1][:count]])
    return words(rows.ravel(), ty)


def execute(
    ir: Kernel, target: Target, backend: str, device: MetalDevice | None, values: list[np.ndarray], count: int
) -> None:
    """Run `ir` over `count` threads on the backend, writing the arrays of `values` in place."""
    threads = Blocks(count // THREADS, threads=THREADS)
    if backend == "emulator":
        emulate(ir, target).run(threads, values)
    elif backend == "c":
        CPUProgram(ir, emit(ir, target))(*values, launch=threads)
    else:
        assert device is not None
        MetalRun(device, ir).run(threads, values)


def run(
    directory: Path, body: str, ty: str, out: str, a, b, backend: str, device: MetalDevice | None = None
) -> np.ndarray:
    target = Target.of("avx512")
    if backend == "metal":
        assert device is not None
        target = device.target
    ir = capture(kernel(directory, body, ty=ty, out=out), "k", target=target)
    result = np.zeros(len(a), NUMPY.get(out, np.uint32))
    execute(ir, target, backend, device, [a, b, result], len(a))
    return result


@pytest.mark.parametrize("backend", ["emulator", "c", "metal"])
@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("op", OPS)
def test_packed_general_matches_numpy(tmp_path: Path, request, backend: str, ty: str, op: str) -> None:
    device = request.getfixturevalue("metal_device") if backend == "metal" else None
    rng = np.random.default_rng(11)
    a, b = random_words(rng, COUNT), random_words(rng, COUNT)
    a[:2], b[:2] = edge_words(ty), edge_words(ty)[::-1]
    actual = run(tmp_path, f"out[i] = {op}(a[i], b[i])", ty, out_type(op, ty), a, b, backend, device)
    np.testing.assert_array_equal(actual, packed_oracle(op, ty, a, b))


@pytest.mark.parametrize("backend", ["emulator", "c", "metal"])
@pytest.mark.parametrize("ty", PLAIN)
@pytest.mark.parametrize("op", OPS)
def test_plain_general_matches_numpy(tmp_path: Path, request, backend: str, ty: str, op: str) -> None:
    device = request.getfixturevalue("metal_device") if backend == "metal" else None
    rng = np.random.default_rng(12)
    a, b = random_values(rng, ty, COUNT), random_values(rng, ty, COUNT)
    actual = run(tmp_path, f"out[i] = {op}(a[i], b[i])", ty, out_type(op, ty), a, b, backend, device)
    np.testing.assert_array_equal(actual, plain_oracle(op, ty, a, b))


@pytest.mark.parametrize("backend", ["emulator", "c"])
@pytest.mark.parametrize("op", ["maximum", "abs_diff"])
def test_float_general_matches_numpy(tmp_path: Path, backend: str, op: str) -> None:
    rng = np.random.default_rng(13)
    a, b = random_values(rng, "f32", COUNT), random_values(rng, "f32", COUNT)
    expected = np.maximum(a, b) if op == "maximum" else np.abs(a - b)
    actual = run(tmp_path, f"out[i] = {op}(a[i], b[i])", "f32", "f32", a, b, backend)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("ty", ["u8x4", "i16x2"])
def test_three_operand_maximum_on_packed_words(tmp_path: Path, ty: str) -> None:
    rng = np.random.default_rng(14)
    a, b = random_words(rng, COUNT), random_words(rng, COUNT)
    actual = run(tmp_path, "out[i] = maximum(a[i], b[i], a[i])", ty, ty, a, b, "c")
    np.testing.assert_array_equal(actual, packed_oracle("maximum", ty, packed_oracle("maximum", ty, a, b), a))


def test_three_operand_maximum_on_plain_integers(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    a, b = random_values(rng, "i32", COUNT), random_values(rng, "i32", COUNT)
    actual = run(tmp_path, "out[i] = maximum(a[i], b[i], a[i])", "i32", "i32", a, b, "c")
    np.testing.assert_array_equal(actual, np.maximum(np.maximum(a, b), a))


@pytest.mark.parametrize("ty", ["u8x4", "i8x4", "u16x2", "i16x2"])
def test_nvidia_packed_add_sat_is_one_simd_instruction(tmp_path: Path, ty: str) -> None:
    target = Target.of("sm_120a")
    text = emit(capture(kernel(tmp_path, "out[i] = add_sat(a[i], b[i])", ty=ty, out=ty), "k", target=target), target)
    assert f"__vadd{'ss' if ty[0] == 'i' else 'us'}{ty[-1]}(" in text.source


def test_float_add_sat_is_refused_with_the_candidates(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = add_sat(a[i], b[i])", ty="f32", out="f32")
    with pytest.raises(TargetError) as error:
        capture(file, "k", target=Target.of("sm_120a"))
    assert error.value.feature == "add_sat"
    assert error.value.target == "sm_120a"
    assert "candidates:" in error.value.reason and "i8x4" in error.value.reason
    assert error.value.instead


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("maximum(a[i])", "takes 2 or 3 operands"),
        ("add_sat(a[i], b[i], a[i])", "takes 2 operands"),
        ("add_sat(a[i], y=b[i])", "takes 2 operands"),
    ],
)
def test_general_functions_refuse_the_wrong_number_of_operands(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(CaptureError, match=message):
        capture(kernel(tmp_path, f"out[i] = {body}", ty="i32", out="i32"), "k", target=Target.of("avx512"))


def test_packed_operands_of_mixed_types_are_refused(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = maximum(a[i], b[i])", ty="u8x4", out="u8x4")
    file.write_text(file.read_text().replace("b: Vector[T.u8x4]", "b: Vector[T.i8x4]"))
    with pytest.raises(CaptureError, match="one type"):
        capture(file, "k", target=Target.of("avx512"))


ATOMIC_KERNEL = (
    "from krn import T, Vector, atomic, grid, kernel\n"
    "from krn.math import abs_diff, add_sat, maximum\n"
    "@kernel\ndef k(a: Vector[T.u32], counter: Vector[T.u32], out: Vector[T.u32]) -> None:\n"
    "    i = grid.thread_rank()\n    if i < out.shape[0]:\n        out[i] = {call}\n"
)
ATOMIC_CALLS = (
    "add_sat(atomic.add(counter, 0, 1), a[i])",
    "maximum(atomic.add(counter, 0, 1), a[i])",
    "maximum(a[i], atomic.add(counter, 0, 1), a[i])",
    "abs_diff(atomic.add(counter, 0, 1), a[i])",
)


@pytest.mark.parametrize("backend", ["emulator", "c", "metal"])
@pytest.mark.parametrize("call", ATOMIC_CALLS)
def test_an_atomic_operand_runs_once_per_thread(tmp_path: Path, request, backend: str, call: str) -> None:
    """Each thread takes one old value, so the values are a permutation of 0..COUNT-1 and the counter ends at COUNT."""
    device = request.getfixturevalue("metal_device") if backend == "metal" else None
    target = device.target if device is not None else Target.of("avx512")
    file = tmp_path / "kernel.py"
    file.write_text(ATOMIC_KERNEL.format(call=call))
    ir = capture(file, "k", target=target)
    a, counter, out = np.zeros(COUNT, np.uint32), np.zeros(1, np.uint32), np.zeros(COUNT, np.uint32)
    execute(ir, target, backend, device, [a, counter, out], COUNT)
    np.testing.assert_array_equal(np.sort(out), np.arange(COUNT, dtype=np.uint32))
    assert counter[0] == COUNT


TILE_KERNEL = (
    "from krn import T, Vector, block, kernel\n"
    "from krn.math import abs_diff, add_sat, maximum\n"
    "@kernel\ndef k(a: Vector[T.{ty}], b: Vector[T.{ty}], out: Vector[T.{ty}]) -> None:\n"
    "    tile = block.tile(16)\n    out[tile] = {body}\n"
)


@pytest.mark.parametrize(
    ("ty", "body"),
    [
        ("u8x4", "a[tile] + b[tile]"),
        ("u8x4", "add_sat(a[tile], b[tile])"),
        ("i32", "maximum(a[tile], b[tile])"),
        ("i32", "abs_diff(a[tile], b[tile])"),
        ("i32", "add_sat(a[tile], b[tile])"),
    ],
)
def test_tile_operands_of_packed_and_general_functions_are_refused(tmp_path: Path, ty: str, body: str) -> None:
    file = tmp_path / "kernel.py"
    file.write_text(TILE_KERNEL.format(ty=ty, body=body))
    with pytest.raises(TargetError, match="packed vector operations are scalar-per-thread for now") as error:
        capture(file, "k", target=Target.of("avx512"))
    assert error.value.target == "avx512"
    assert error.value.instead == "index the tile element and call per element"


def test_the_top_level_maximum_still_takes_float_tiles(tmp_path: Path) -> None:
    file = tmp_path / "kernel.py"
    file.write_text(
        "from krn import T, Vector, block, kernel, maximum\n"
        "@kernel\ndef k(a: Vector[T.f32], b: Vector[T.f32], out: Vector[T.f32]) -> None:\n"
        "    tile = block.tile(16)\n    out[tile] = maximum(a[tile], b[tile])\n"
    )
    capture(file, "k", target=Target.of("avx512"))


def software_fallbacks(caught, name: str) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, PerformanceWarning) and name in str(w.message)]


def capture_recording(file: Path, target: Target):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capture(file, "k", target=target)
    return caught


def test_apple_packed_add_sat_is_metal_native_and_silent(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = add_sat(a[i], add_sat(b[i], a[i]))", ty="u8x4", out="u8x4")
    assert not software_fallbacks(capture_recording(file, Target.of("apple_m4")), "add_sat")


def test_amd_packed_add_sat_warns_once_per_kernel_and_function(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = add_sat(a[i], add_sat(b[i], a[i]))", ty="u8x4", out="u8x4")
    [message] = software_fallbacks(capture_recording(file, Target.of("gfx942")), "add_sat")
    assert "add_sat on u8x4 has no native instruction on gfx942" in message
    assert "instruction sequence; native on Apple GPUs and NVIDIA GPUs" in message


def test_nvidia_native_packed_is_silent(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = add_sat(a[i], b[i])", ty="u8x4", out="u8x4")
    assert not software_fallbacks(capture_recording(file, Target.of("sm_120a")), "add_sat")


def test_cpu_never_warns_about_software_sequences(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = add_sat(a[i], b[i])", ty="i32", out="i32")
    assert not software_fallbacks(capture_recording(file, Target.of("avx512")), "add_sat")


def test_plain_integer_add_sat_warns_when_no_gpu_has_it_native(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = add_sat(a[i], b[i])", ty="i32", out="i32")
    [message] = software_fallbacks(capture_recording(file, Target.of("sm_120a")), "add_sat")
    assert "add_sat on i32 has no native instruction on sm_120a" in message
    assert "not native on any GPU target" in message


def test_short_sequences_are_silent(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = maximum(a[i], b[i])", ty="i32", out="i32")
    assert not software_fallbacks(capture_recording(file, Target.of("sm_120a")), "maximum")


def test_warning_can_be_disabled(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = add_sat(a[i], b[i])", ty="i32", out="i32")
    with configure(**{"warn.warn_software_fallback": False}):
        assert not software_fallbacks(capture_recording(file, Target.of("sm_120a")), "add_sat")


def test_minimum_instruction_setting_moves_the_threshold(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = add_sat(a[i], b[i])", ty="u8x4", out="u8x4")
    assert software_fallbacks(capture_recording(file, Target.of("gfx942")), "add_sat")
    with configure(**{"warn.software_fallback_min_instructions": 1000}):
        assert not software_fallbacks(capture_recording(file, Target.of("gfx942")), "add_sat")


HELPER = (
    "from krn import T\nfrom krn.math import add_sat\n\n"
    "def {name}(x: T.i32, y: T.i32) -> T.i32:\n    return add_sat(x, y)\n"
)
CALLER = (
    "from krn import T, Vector, grid, kernel\n"
    "from .helpers import saturate\n"
    "@kernel\ndef k(a: Vector[T.i32], b: Vector[T.i32], out: Vector[T.i32]) -> None:\n"
    "    i = grid.thread_rank()\n    if i < out.shape[0]:\n        out[i] = saturate(a[i], b[i])\n"
)


def test_helpers_from_two_modules_warn_once_for_one_kernel(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text(HELPER.format(name="first"))
    (tmp_path / "second.py").write_text(HELPER.format(name="second"))
    file = tmp_path / "kernel.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        "from .first import first\nfrom .second import second\n"
        "@kernel\ndef k(a: Vector[T.i32], b: Vector[T.i32], out: Vector[T.i32]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n        out[i] = second(first(a[i], b[i]), b[i])\n"
    )
    [message] = software_fallbacks(capture_recording(file, Target.of("sm_120a")), "add_sat")
    assert message.startswith("k on sm_120a: add_sat on i32")


def test_unrelated_kernels_with_one_helper_each_warn(tmp_path: Path) -> None:
    """Both kernels are named `k` and sit on the same line, in different files: neither suppresses the other."""
    (tmp_path / "helpers.py").write_text(HELPER.format(name="saturate"))
    files = [tmp_path / "one.py", tmp_path / "two.py"]
    for file in files:
        file.write_text(CALLER)
    messages = [software_fallbacks(capture_recording(file, Target.of("sm_120a")), "add_sat") for file in files]
    assert [len(found) for found in messages] == [1, 1]


def test_apple_abs_diff_is_metal_vector_code(tmp_path: Path) -> None:
    file = kernel(tmp_path, "out[i] = abs_diff(a[i], b[i])", ty="i16x2", out="u16x2")
    source = emit(capture(file, "k", target=Target.of("apple_m4")), Target.of("apple_m4")).source
    assert "metal::absdiff(x, y)" in source and "as_type<short2>" in source


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (add_sat, (1, 2)),
        (abs_diff, (1, 2)),
        (maximum, (1, 2)),
        (maximum, (1, 2, 3)),
        (pack, (1, 2, 3, 4)),
        (unpack, (1,)),
    ],
)
def test_host_stubs_only_describe_the_signature(function, args) -> None:
    with pytest.raises(NotImplementedError):
        function(*args)


LANE_WORD = st.one_of(
    st.sampled_from([0, 1, 0x7F7F7F7F, 0x80808080, 0xFFFFFFFF, 0x80008000, 0x7FFF7FFF]), st.integers(0, 2**32 - 1)
)


@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("op", OPS)
@settings(max_examples=20, deadline=None)
@given(
    a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT),
    b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT),
)
def test_packed_general_matches_numpy_over_all_lanes(ty: str, op: str, a: list[int], b: list[int]) -> None:
    left, right = np.array(a, np.uint32), np.array(b, np.uint32)
    with TemporaryDirectory() as directory:
        emulated_dir, composed_dir = Path(directory, "emulated"), Path(directory, "composed")
        emulated_dir.mkdir(), composed_dir.mkdir()
        actual = run(emulated_dir, f"out[i] = {op}(a[i], b[i])", ty, out_type(op, ty), left, right, "emulator")
        composed = run(composed_dir, f"out[i] = {op}(a[i], b[i])", ty, out_type(op, ty), left, right, "c")
    expected = packed_oracle(op, ty, left, right)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(composed, expected)

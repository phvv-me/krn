"""Float32 atomic add and subtract keep subnormals: the native instruction flushes them on global memory, so a GPU
runs a compare-and-swap loop and warns once per kernel and target. Float64 and integer atomics keep the native
instruction."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from krn import Blocks
from krn.compiler import performance
from krn.compiler.api import capture
from krn.compiler.lower import emit
from krn.compiler.performance import PerformanceWarning
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from krn.runtime.cpu import CPUProgram
from krn.runtime.settings import configure
from tests.helpers.devices import CUDARun, compare

pytestmark = pytest.mark.filterwarnings("ignore:.*compare-and-swap loop")  # the warning has its own tests below
HEADER = "from krn import T, Vector, atomic, grid, kernel, shared_array\n"
LAUNCH = Blocks(1, threads=1)
C_TARGET = Target.of("avx512")
TINY = float(np.finfo(np.float32).tiny)
ONE_ULP = 2.0**-149  # the smallest positive float32 subnormal
GLOBAL_RUN = "@kernel\ndef run(x: Vector[T.f32], out: Vector[T.f32]) -> None:\n    atomic.{op}(out, 0, x[0])\n"
SHARED_RUN = (
    "@kernel\ndef run(x: Vector[T.f32], out: Vector[T.f32]) -> None:\n"
    "    cell = shared_array(T.f32, 1)\n    cell[0] = 0.0\n    atomic.{op}(cell, 0, x[0])\n    out[0] = cell[0]\n"
)
CASES = [
    ("add", 0.0, ONE_ULP, ONE_ULP),  # a subnormal added into zero
    ("add", ONE_ULP, ONE_ULP, 2 * ONE_ULP),  # subnormal operand and subnormal result
    ("sub", 2 * ONE_ULP, ONE_ULP, ONE_ULP),
]


def write(directory: Path, body: str) -> Path:
    file = directory / "atomics.py"
    file.write_text(HEADER + body)
    return file


def warnings_of(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, PerformanceWarning)]


NOFTZ = ("sm_90a", "sm_120a")
LOOP_CASES = [  # every float32 add and sub loops, except the global add that these lower to one noftz instruction
    (arch, op, space)
    for arch in ("sm_80", "sm_90a", "sm_120a")
    for op in ("add", "sub")
    for space in ("global", "shared")
    if not (arch in NOFTZ and op == "add" and space == "global")
]


@pytest.mark.parametrize(("arch", "op", "space"), LOOP_CASES)
def test_float32_add_and_sub_run_a_loop(tmp_path: Path, arch: str, op: str, space: str) -> None:
    target = Target.of(arch)
    template = GLOBAL_RUN if space == "global" else SHARED_RUN
    text = emit(capture(write(tmp_path, template.format(op=op)), "run", target=target), target).source
    assert "atomicCAS" in text and "__float_as_uint" in text
    assert "atomicAdd(" not in text and "noftz" not in text


@pytest.mark.parametrize("arch", NOFTZ)
def test_global_float32_add_is_one_noftz_instruction(tmp_path: Path, arch: str) -> None:
    target = Target.of(arch)
    text = emit(capture(write(tmp_path, GLOBAL_RUN.format(op="add")), "run", target=target), target).source
    assert "atom.global.add.noftz.f32" in text
    assert "atomicCAS" not in text and "atomicAdd(" not in text
    assert "_atomic_add_f32_noftz(&out[" in text


@pytest.mark.parametrize("arch", NOFTZ)
def test_the_noftz_add_warns_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arch: str) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    target = Target.of(arch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit(capture(write(tmp_path, GLOBAL_RUN.format(op="add")), "run", target=target), target)
    assert not warnings_of(caught)


@pytest.mark.parametrize(
    ("dtype", "op", "native"),
    [("f64", "add", "atomicAdd(p, v)"), ("f64", "sub", "atomicAdd(p, -v)"), ("i32", "sub", "atomicSub(p, v)")],
)
def test_float64_and_integer_atomics_keep_the_native_instruction(
    tmp_path: Path, dtype: str, op: str, native: str
) -> None:
    target = Target.of("sm_90a")
    body = f"@kernel\ndef run(x: Vector[T.{dtype}], out: Vector[T.{dtype}]) -> None:\n    atomic.{op}(out, 0, x[0])\n"
    text = emit(capture(write(tmp_path, body), "run", target=target), target).source
    assert native in text
    assert "atomicCAS" not in text


def test_the_loop_warns_once_per_kernel_and_target_and_can_be_silenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    other = "@kernel\ndef other(x: Vector[T.f32], out: Vector[T.f32]) -> None:\n    atomic.sub(out, 0, x[0])\n"
    file = write(tmp_path, GLOBAL_RUN.format(op="add") + other + "    atomic.add(out, 0, x[0])\n")
    target, later = Target.of("sm_80"), Target.of("sm_120a")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit(capture(file, "run", target=target), target)
        emit(capture(file, "run", target=target), target)
        emit(capture(file, "other", target=target), target)
        emit(capture(file, "other", target=later), later)
    messages = warnings_of(caught)
    assert [m.split(":")[0] for m in messages] == ["run on sm_80", "other on sm_80", "other on sm_120a"]
    assert "a float32 atomic runs a compare-and-swap loop" in messages[0]
    assert "warn.warn_strict_atomic_cas" in messages[0]
    monkeypatch.setattr(performance, "_seen", set())
    with configure(**{"warn.warn_strict_atomic_cas": False}), warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        emit(capture(file, "run", target=target), target)
    assert not warnings_of(quiet)


@pytest.mark.parametrize(
    "body",
    [
        "@kernel\ndef run(x: Vector[T.f64], out: Vector[T.f64]) -> None:\n    atomic.add(out, 0, x[0])\n",
        GLOBAL_RUN.format(op="max"),
    ],
    ids=["float64", "float32-max"],
)
def test_atomics_that_do_not_flush_do_not_warn(tmp_path: Path, body: str) -> None:
    target = Target.of("sm_90a")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit(capture(write(tmp_path, body), "run", target=target), target)
    assert not warnings_of(caught)


@pytest.mark.parametrize(("op", "start", "operand", "expected"), CASES)
def test_one_writer_keeps_subnormal_operand_and_result_on_the_c_backend(
    tmp_path: Path, op: str, start: float, operand: float, expected: float
) -> None:
    ir = capture(write(tmp_path, GLOBAL_RUN.format(op=op)), "run", target=C_TARGET)
    x = np.array([operand], np.float32)
    actual, oracle = np.array([start], np.float32), np.array([start], np.float32)
    with np.errstate(all="ignore"):
        emulate(ir, C_TARGET).run(LAUNCH, [x, oracle])
        CPUProgram(ir, emit(ir, C_TARGET))(x, actual, launch=LAUNCH)
    want = np.array([expected], np.float32)
    assert 0 < abs(want[0]) < TINY
    assert actual.view(np.uint32)[0] == oracle.view(np.uint32)[0] == want.view(np.uint32)[0]


@pytest.mark.parametrize("arch", ["sm_80", "sm_90a", "sm_120a"])
def test_the_loop_compiles_with_nvrtc(nvrtc, tmp_path: Path, arch: str) -> None:
    target = Target.of(arch)
    source = emit(capture(write(tmp_path, SHARED_RUN.format(op="sub")), "run", target=target), target).source
    assert nvrtc.compile(source, arch, "cubin").data[:4] == b"\x7fELF"


@pytest.mark.parametrize("arch", NOFTZ)
def test_the_noftz_add_compiles_with_nvrtc(nvrtc, tmp_path: Path, arch: str) -> None:
    target = Target.of(arch)
    source = emit(capture(write(tmp_path, GLOBAL_RUN.format(op="add")), "run", target=target), target).source
    assert nvrtc.compile(source, arch, "cubin").data[:4] == b"\x7fELF"


CONTENDED_RUN = (
    "@kernel\ndef run(x: Vector[T.f32], out: Vector[T.f32]) -> None:\n    atomic.add(out, 0, x[grid.thread_rank()])\n"
)
CONTENDED_LAUNCH = Blocks(64, threads=256)


def test_contended_float32_adds_keep_subnormals_on_the_card(gpu, tmp_path: Path) -> None:
    """16384 threads add a signed power of two (1, 2, 4 or 8 ulps) into one word. Every partial sum is an exact
    subnormal, so the card must return the exact sum bit for bit, in any order."""
    k = np.random.default_rng(0).choice([-1, 1], 16384) * 2 ** (np.arange(16384) % 4)
    x, out = (k * ONE_ULP).astype(np.float32), np.zeros(1, np.float32)
    file = write(tmp_path, CONTENDED_RUN)
    _, actual = compare(gpu.nvidia, lambda kernel: CUDARun(gpu, kernel), "run", CONTENDED_LAUNCH, [x, out], file)
    want = np.array([np.ldexp(float(k.sum()), -149)], np.float32)
    assert 0 < abs(want[0]) < TINY
    assert actual[1].view(np.uint32)[0] == want.view(np.uint32)[0]


@pytest.mark.parametrize(("op", "start", "operand", "expected"), CASES)
def test_one_thread_keeps_a_subnormal_on_the_card(
    gpu, tmp_path: Path, op: str, start: float, operand: float, expected: float
) -> None:
    file = write(tmp_path, GLOBAL_RUN.format(op=op))
    x, out = np.array([operand], np.float32), np.array([start], np.float32)
    _, actual = compare(gpu.nvidia, lambda kernel: CUDARun(gpu, kernel), "run", LAUNCH, [x, out], file)
    assert actual[1].view(np.uint32)[0] == np.array([expected], np.float32).view(np.uint32)[0]

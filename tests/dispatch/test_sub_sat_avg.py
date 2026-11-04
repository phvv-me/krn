"""`sub_sat`, `avg` and the lane tests `all(a OP b)` and `any(a OP b)` against lane-by-lane Python, on the emulator, the
C backend and the Apple GPU, and the NVIDIA and Metal spellings they lower to."""

import warnings
from functools import cache
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
from krn.runtime.cpu import CPUProgram
from tests.helpers.devices import CUDARun, MetalRun

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice
    from krn.runtime.metal import MetalDevice
    from krn.runtime.nvrtc import Nvidia as Nvrtc

COUNT = 64
PACKED = ("i8x4", "u8x4", "i16x2", "u16x2")
PLAIN = ("i8", "u8", "i16", "u16", "i32", "u32")
NUMPY = {"i8": np.int8, "u8": np.uint8, "i16": np.int16, "u16": np.uint16, "i32": np.int32, "u32": np.uint32}
RELATIONS = ("==", "!=", "<", "<=", ">", ">=")
LANE_WORD = st.one_of(
    st.sampled_from([0, 1, 0x7F7F7F7F, 0x80808080, 0xFFFFFFFF, 0x80008000, 0x7FFF7FFF]), st.integers(0, 2**32 - 1)
)


def lanes_of(word: int, ty: str) -> list[int]:
    """The lanes of a word as the Python integers they hold, signed for an `i` type, low lane first."""
    bits = 8 if ty.endswith("x4") else 16
    values = [(word >> (bits * i)) & ((1 << bits) - 1) for i in range(32 // bits)]
    if ty[0] == "i":
        values = [v - (1 << bits) if v >> (bits - 1) else v for v in values]
    return values


def word_of(values: list[int], ty: str) -> int:
    """The word holding the lane values, each wrapped to its lane width."""
    bits = 8 if ty.endswith("x4") else 16
    return sum((v & ((1 << bits) - 1)) << (bits * i) for i, v in enumerate(values))


def lane_range(ty: str) -> tuple[int, int]:
    bits = 8 if ty.endswith("x4") else 16
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if ty[0] == "i" else (0, (1 << bits) - 1)


def lane_oracle(op: str, ty: str, a: int, b: int) -> int:
    """`sub_sat` clamps each lane difference to the lane range; `avg` rounds each lane mean half up."""
    low, high = lane_range(ty)
    values = [
        min(max(x - y, low), high) if op == "sub_sat" else (x + y + 1) >> 1
        for x, y in zip(lanes_of(a, ty), lanes_of(b, ty), strict=True)
    ]
    return word_of(values, ty)


def lane_test_oracle(name: str, relation: str, ty: str, a: int, b: int) -> int:
    """`all` is true when every lane compares true, `any` when one does."""
    compare = {"==": int.__eq__, "!=": int.__ne__, "<": int.__lt__, "<=": int.__le__, ">": int.__gt__, ">=": int.__ge__}
    results = [compare[relation](x, y) for x, y in zip(lanes_of(a, ty), lanes_of(b, ty), strict=True)]
    return int(all(results) if name == "all" else any(results))


def kernel_file(directory: Path, body: str, ty: str, out: str) -> Path:
    file = directory / "kernel.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        "from krn.math import avg, sub_sat\n"
        f"@kernel\ndef k(a: Vector[T.{ty}], b: Vector[T.{ty}], out: Vector[T.{out}]) -> None:\n"
        f"    i = grid.thread_rank()\n    if i < out.shape[0]:\n        {body}\n"
    )
    return file


@cache
def compiled(body: str, ty: str, out: str, arch: str) -> Kernel:
    """The IR of one kernel, captured once: the hypothesis examples of a case reuse it."""
    with TemporaryDirectory() as directory:
        return capture(kernel_file(Path(directory), body, ty, out), "k", target=Target.of(arch))


def run(ir: Kernel, backend: str, values: list[np.ndarray], device: MetalDevice | None = None) -> None:
    """Run `ir` over `COUNT` threads on the backend, writing the last array of `values`."""
    threads = Blocks(1, threads=COUNT)
    if backend == "emulator":
        emulate(ir, Target.of("avx512")).run(threads, values)
    elif backend == "c":
        CPUProgram(ir, emit(ir, Target.of("avx512")))(*values, launch=threads)
    else:
        assert device is not None
        MetalRun(device, ir).run(threads, values)


def precision_warnings(file: Path, arch: str) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capture(file, "k", target=Target.of(arch))
    return [str(w.message) for w in caught if issubclass(w.category, PerformanceWarning)]


def check_packed_op(backend: str, device: MetalDevice | None, ty: str, op: str, a, b) -> None:
    arch = device.target.arch if device is not None else "avx512"
    ir = compiled(f"out[i] = {op}(a[i], b[i])", ty, ty, arch)
    result = np.zeros(COUNT, np.uint32)
    run(ir, backend, [np.array(a, np.uint32), np.array(b, np.uint32), result], device)
    expected = [lane_oracle(op, ty, x, y) for x, y in zip(a, b, strict=True)]
    np.testing.assert_array_equal(result, np.array(expected, np.uint32))


def check_lane_test(backend: str, device: MetalDevice | None, ty: str, name: str, relation: str, a, b) -> None:
    arch = device.target.arch if device is not None else "avx512"
    body = f"if {name}(a[i] {relation} b[i]):\n            out[i] = 1\n        else:\n            out[i] = 0"
    result = np.zeros(COUNT, np.uint32)
    run(compiled(body, ty, "u32", arch), backend, [np.array(a, np.uint32), np.array(b, np.uint32), result], device)
    expected = [lane_test_oracle(name, relation, ty, x, y) for x, y in zip(a, b, strict=True)]
    np.testing.assert_array_equal(result, np.array(expected, np.uint32))


@pytest.mark.parametrize("backend", ["emulator", "c"])
@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("op", ["sub_sat", "avg"])
@settings(max_examples=12, deadline=None)
@given(a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT), b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT))
def test_packed_sub_sat_and_avg_match_lane_oracle(backend: str, ty: str, op: str, a, b) -> None:
    check_packed_op(backend, None, ty, op, a, b)


@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("op", ["sub_sat", "avg"])
@settings(max_examples=12, deadline=None)
@given(a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT), b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT))
def test_packed_sub_sat_and_avg_match_lane_oracle_on_metal(metal_device: MetalDevice, ty: str, op: str, a, b) -> None:
    check_packed_op("metal", metal_device, ty, op, a, b)


@pytest.mark.parametrize("backend", ["emulator", "c"])
@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("name", ["all", "any"])
@pytest.mark.parametrize("relation", RELATIONS)
@settings(max_examples=8, deadline=None)
@given(a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT), b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT))
def test_lane_tests_match_lane_oracle(backend: str, ty: str, name: str, relation: str, a, b) -> None:
    check_lane_test(backend, None, ty, name, relation, a, b)


@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("name", ["all", "any"])
@pytest.mark.parametrize("relation", RELATIONS)
@settings(max_examples=8, deadline=None)
@given(a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT), b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT))
def test_lane_tests_match_lane_oracle_on_metal(
    metal_device: MetalDevice, ty: str, name: str, relation: str, a, b
) -> None:
    check_lane_test("metal", metal_device, ty, name, relation, a, b)


@pytest.mark.parametrize("backend", ["emulator", "c"])
@pytest.mark.parametrize("ty", PLAIN)
@pytest.mark.parametrize("op", ["sub_sat", "avg"])
def test_plain_sub_sat_and_avg_match_numpy(backend: str, ty: str, op: str) -> None:
    rng = np.random.default_rng(21)
    info = np.iinfo(NUMPY[ty])
    a = rng.integers(info.min, info.max, COUNT, endpoint=True, dtype=np.int64)
    b = rng.integers(info.min, info.max, COUNT, endpoint=True, dtype=np.int64)
    a[:4], b[:4] = (info.min, info.max, info.max, info.min), (info.max, info.min, info.min, info.min)
    left, right = a.astype(NUMPY[ty]), b.astype(NUMPY[ty])
    if op == "sub_sat":
        expected = np.clip(left.astype(np.int64) - right, info.min, info.max)
    else:
        expected = (left.astype(np.int64) + right + 1) >> 1
    result = np.zeros(COUNT, NUMPY[ty])
    run(compiled(f"out[i] = {op}(a[i], b[i])", ty, ty, "avx512"), backend, [left, right, result])
    np.testing.assert_array_equal(result, expected.astype(NUMPY[ty]))


@pytest.mark.parametrize("ty", ["u8x4", "i8x4", "u16x2", "i16x2"])
@pytest.mark.parametrize("op", ["sub_sat", "avg"])
def test_nvidia_sub_sat_is_saturating_simd_and_unsigned_avg_is_vavgu(tmp_path: Path, ty: str, op: str) -> None:
    target = Target.of("sm_120a")
    text = emit(capture(kernel_file(tmp_path, f"out[i] = {op}(a[i], b[i])", ty, ty), "k", target=target), target)
    lanes, signed = ty[-1], ty[0] == "i"
    if op == "sub_sat":
        assert f"__vsub{'ss' if signed else 'us'}{lanes}(" in text.source
    elif signed:
        assert "_family_avg_signed(" in text.source and "__vavgs" not in text.source
    else:
        assert f"__vavgu{lanes}(" in text.source


def test_nvidia_signed_avg_is_composed_and_warns(tmp_path: Path) -> None:
    file = kernel_file(tmp_path, "out[i] = avg(a[i], b[i])", "i8x4", "i8x4")
    [message] = [text for text in precision_warnings(file, "sm_120a") if "avg on i8x4" in text]
    assert "has no native instruction on sm_120a" in message and "native on Apple GPUs" in message


def test_nvidia_lane_test_is_the_set_instruction(tmp_path: Path) -> None:
    target = Target.of("sm_120a")
    body = "if all(a[i] == b[i]):\n            out[i] = 1\n        else:\n            out[i] = 0"
    text = emit(capture(kernel_file(tmp_path, body, "u8x4", "u32"), "k", target=target), target)
    assert "__vseteq4(" in text.source


def test_apple_avg_and_sub_sat_are_metal_calls(tmp_path: Path) -> None:
    target = Target.of("apple_m4")
    average = kernel_file(tmp_path, "out[i] = avg(a[i], b[i])", "u8x4", "u8x4")
    average = emit(capture(average, "k", target=target), target)
    assert "metal::rhadd(x, y)" in average.source and "as_type<uchar4>" in average.source
    (tmp_path / "subtract").mkdir()
    subtract = kernel_file(tmp_path / "subtract", "out[i] = sub_sat(a[i], b[i])", "i16x2", "i16x2")
    text = emit(capture(subtract, "k", target=target), target)
    assert "metal::subsat(x, y)" in text.source


def test_lane_test_of_a_plain_comparison_is_refused(tmp_path: Path) -> None:
    body = "if all(a[i] == b[i]):\n            out[i] = 1\n        else:\n            out[i] = 0"
    with pytest.raises(CaptureError, match="takes a comparison of two packed words of one type"):
        capture(kernel_file(tmp_path, body, "i32", "u32"), "k", target=Target.of("avx512"))


@pytest.mark.parametrize("name", ["sub_sat", "avg"])
def test_float_operands_are_refused_with_the_candidates(tmp_path: Path, name: str) -> None:
    file = kernel_file(tmp_path, f"out[i] = {name}(a[i], b[i])", "f32", "f32")
    with pytest.raises(TargetError, match="candidates"):
        capture(file, "k", target=Target.of("sm_120a"))


# CUDADevice.to_device types a host array by its numpy dtype, so a packed parameter receives a uint32 tensor and
# cuda_backend.flatten refuses it; strict, so the mark goes once the runtime types packed uploads from the kernel.
PACKED_ON_CARD = pytest.mark.xfail(strict=True, raises=TypeError, reason="packed uploads are typed as uint32")


@pytest.mark.parametrize("arch", ["sm_80", "sm_90a", "sm_120a"])
@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("op", ["sub_sat", "avg"])
def test_general_packed_compiles_with_nvrtc(tmp_path: Path, nvrtc: Nvrtc, arch: str, ty: str, op: str) -> None:
    target = Target.of(arch)
    ir = capture(kernel_file(tmp_path, f"out[i] = {op}(a[i], b[i])", ty, ty), "k", target=target)
    assert nvrtc.compile(emit(ir, target).source, arch, "cubin").data[:4] == b"\x7fELF"


@PACKED_ON_CARD
@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("op", ["sub_sat", "avg"])
@settings(max_examples=6, deadline=None)
@given(a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT), b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT))
def test_general_packed_runs_on_the_card(gpu: CUDADevice, ty: str, op: str, a, b) -> None:
    ir = compiled(f"out[i] = {op}(a[i], b[i])", ty, ty, gpu.target.arch)
    result = np.zeros(COUNT, np.uint32)
    CUDARun(gpu, ir).run(Blocks(1, threads=COUNT), [np.array(a, np.uint32), np.array(b, np.uint32), result])
    expected = [lane_oracle(op, ty, x, y) for x, y in zip(a, b, strict=True)]
    np.testing.assert_array_equal(result, np.array(expected, np.uint32))


@PACKED_ON_CARD
@pytest.mark.parametrize("ty", PACKED)
@pytest.mark.parametrize("name", ["all", "any"])
@pytest.mark.parametrize("relation", RELATIONS)
@settings(max_examples=6, deadline=None)
@given(a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT), b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT))
def test_lane_tests_run_on_the_card(gpu: CUDADevice, ty: str, name: str, relation: str, a, b) -> None:
    body = f"if {name}(a[i] {relation} b[i]):\n            out[i] = 1\n        else:\n            out[i] = 0"
    result = np.zeros(COUNT, np.uint32)
    ir = compiled(body, ty, "u32", gpu.target.arch)
    CUDARun(gpu, ir).run(Blocks(1, threads=COUNT), [np.array(a, np.uint32), np.array(b, np.uint32), result])
    expected = [lane_test_oracle(name, relation, ty, x, y) for x, y in zip(a, b, strict=True)]
    np.testing.assert_array_equal(result, np.array(expected, np.uint32))

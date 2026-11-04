"""MMA changes accumulation order relative to the sequential FMA oracle.

The absolute forward-error bound scales with |A| @ |B|; a fixed relative tolerance near a canceled result is wrong.
BF16 device storage uses uint16 words until the runtime has a public BF16 tensor constructor.
FP8 has no emulator or C backend decode: device bytes are checked against independent 256-entry float64 tables.
"""

import logging
import re
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Tiles, targets
from krn.compiler.api import CaptureError, build, capture, emit
from krn.compiler.cuda import NvidiaMma
from krn.compiler.dtypes import BF16, fp
from krn.compiler.errors import TargetError
from krn.compiler.hip import AMDMFMA
from krn.compiler.ir import Accumulate, For
from krn.compiler.matrix import MatrixPlan, pipeline_stages, warp_grid
from krn.compiler.msl import AppleSimdgroup
from krn.compiler.validity import walk
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from krn.runtime.compile_options import CompileOptions
from krn.runtime.cpu import CLANG, NUMPY
from tests.helpers.devices import MetalRun
from tests.helpers.text import words

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

ARCHES = ("sm_80", "sm_86", "sm_89", "sm_90a", "sm_120a")


def fused_loop(source: str) -> bool:
    """The scalar fallback accumulates each product with the renderer's fma (`fmaf`, `fma`, `precise::fma`)."""
    return re.search(r"_s = [\w:]*fma(f)?\(", source) is not None


def source_of(
    path: Path,
    bm: int = 32,
    bn: int = 32,
    bk: int = 16,
    dtype: str = "f16",
    rhs: str | None = None,
    output: str = "f32",
    stages: int = 0,
    generic: bool = False,
    epilogue: str = "acc",
) -> Path:
    loop = (
        f"range(0, a.shape[1], {bk})"
        if generic
        else f"pipelined(a.shape[1], {bk}, stages={stages})"
        if stages
        else f"pipelined(a.shape[1], {bk})"
    )
    slices = f"slice(ks, ks + {bk})" if generic else "ks"
    path.write_text(
        "from krn import Matrix, block, kernel, pipelined, zeros\n"
        "from krn.frontend.types import f16, bf16, f32, f8e4m3fn, f8e5m2\n"
        f"@kernel\ndef gemm(a: Matrix[{dtype}], b: Matrix[{rhs or dtype}], c: Matrix[{output}]) -> None:\n"
        f"    rows, cols = block.tile({bm}, {bn})\n"
        f"    acc = zeros(f32, {bm}, {bn})\n"
        f"    for ks in {loop}:\n"
        f"        acc += a[rows, {slices}] @ b[{slices}, cols]\n"
        f"    c[rows, cols] = ({epilogue}).to({output})\n"
    )
    path.write_text(path.read_text().replace("fp(4, 3)", "f8e4m3fn").replace("fp(5, 2)", "f8e5m2"))
    return path


@pytest.mark.parametrize(
    ("bm", "bn", "grid"),
    [(128, 128, (2, 2)), (128, 64, (2, 2)), (128, 256, (2, 4)), (16, 16, (1, 2))],
)
def test_warp_tiling(bm: int, bn: int, grid: tuple[int, int]) -> None:
    assert warp_grid(bm, bn) == grid


@pytest.mark.parametrize(
    ("stage_bytes", "requested", "want"),
    [
        (18944, 0, 2),
        (27136, 0, 3),
        (10496, 0, 4),
        (52224, 0, 0),
        (2816, 3, 3),
        (2816, 10, 0),
        (2816, 1, 0),
    ],
)
def test_stage_budget(stage_bytes: int, requested: int, want: int) -> None:
    assert pipeline_stages(stage_bytes, requested, 99 * 1024) == want


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("dtype", ("f16", "bf16"))
@pytest.mark.parametrize("bk", (8, 16, 24, 32))
def test_decision_and_emission(tmp_path: Path, arch: str, dtype: str, bk: int) -> None:
    target = targets.Nvidia.of(arch)
    kernel = capture(source_of(tmp_path / "gemm.py", bk=bk, dtype=dtype), "gemm", target=target)
    statement = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    plan = NvidiaMma().plan(statement, target)
    assert isinstance(plan, MatrixPlan)
    assert plan.instruction.k == (16 if bk % 16 == 0 else 8)
    unit = emit(kernel, target)
    assert plan.instruction.spelling in unit.source
    assert all(token not in unit.source for token in ("wgmma", "tcgen05", "tmem", "_s += "))
    assert "float* acc" not in unit.source
    stage_bytes = 2 * (32 * (bk + 8) + bk * (32 + 8))
    assert unit.shared_bytes == stage_bytes * pipeline_stages(stage_bytes, 0, target.shared_bytes)
    assert "cp.async.cg.shared.global" in unit.source
    assert unit.threads == 128
    loop = next(s for s in kernel.body if isinstance(s, For))
    assert loop.pipelined and loop.stages == 0
    assert "mma.sync" not in emit(kernel, targets.CPU.of("avx512")).source


@pytest.mark.parametrize(
    ("bm", "a", "b", "reason"),
    [
        (8, "f16", "f16", "divisible"),
        (32, "f16", "f32", "f16 x f32"),
        (32, "f32", "f32", "f32 x f32"),
    ],
)
def test_fallback_reason(tmp_path: Path, bm: int, a: str, b: str, reason: str) -> None:
    target = targets.Nvidia.of("sm_80")
    kernel = capture(source_of(tmp_path / "gemm.py", bm=bm, dtype=a, rhs=b), "gemm", target=target)
    statement = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    refusal = NvidiaMma().plan(statement, target)
    assert isinstance(refusal, str) and reason in refusal and "sm_80 and later" in refusal
    assert fused_loop(emit(kernel, target).source)
    assert "sm_70" in str(NvidiaMma().plan(statement, replace(target, arch="sm_70")))


FP8_PAIRS = [(a, b) for a in ("fp(4, 3)", "fp(5, 2)") for b in ("fp(4, 3)", "fp(5, 2)")]


@pytest.mark.parametrize("a,b", FP8_PAIRS)
@pytest.mark.parametrize("arch", ("sm_89", "sm_120a"))
def test_fp8_emission(tmp_path: Path, a: str, b: str, arch: str) -> None:
    target = targets.Nvidia.of(arch)
    kernel = capture(source_of(tmp_path / "fp8.py", bk=32, dtype=a, rhs=b), "gemm", target=target)
    statement = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    plan = NvidiaMma().plan(statement, target)
    assert isinstance(plan, MatrixPlan) and plan.fp8
    unit = emit(kernel, target)
    assert plan.instruction.spelling in unit.source and "unsigned char* _t" in unit.source
    assert "cp.async" not in unit.source and ".trans" not in unit.source
    assert unit.shared_bytes == 2 * 32 * (32 + 16)


@pytest.mark.parametrize("arch", ("sm_80", "sm_86", "avx512", "apple_m4", "gfx942", "gfx950"))
@pytest.mark.parametrize("bk", (16, 32))
def test_fp8_no_scalar_fallback(tmp_path: Path, arch: str, bk: int) -> None:
    target = targets.Target.of(arch)
    kernel = capture(source_of(tmp_path / "fp8.py", bk=bk, dtype="fp(4, 3)"), "gemm", target=target)
    with pytest.raises(CaptureError, match="sm_89, sm_90a, sm_100a, sm_120a"):
        emit(kernel, target)


def test_fp8_invalid_tile_and_expression(tmp_path: Path) -> None:
    target = targets.Nvidia.of("sm_120a")
    file = source_of(tmp_path / "fp8.py", bk=16, dtype="fp(4, 3)")
    kernel = capture(file, "gemm", target=target)
    with pytest.raises(CaptureError, match="BK by 32"):
        emit(kernel, target)
    with pytest.raises(CaptureError, match="emulator has no fp8 decode"):
        emulate(kernel, target)
    file = source_of(tmp_path / "expression.py", bk=32, dtype="fp(4, 3)")
    file.write_text(file.read_text().replace("a[rows, ks]", "a[rows, ks].to(f8e5m2)"))
    with pytest.raises(CaptureError, match="direct TileLoad"):
        capture(file, "gemm", target=target)


@pytest.mark.parametrize("a,b", FP8_PAIRS)
@pytest.mark.parametrize("arch", ("sm_89", "sm_90a", "sm_100a", "sm_120a"))
def test_fp8_compile(
    tmp_path: Path,
    compile_cubin: Callable[[str, tuple[str, ...]], bytes],
    a: str,
    b: str,
    arch: str,
) -> None:
    target = targets.Nvidia.of(arch)
    kernel = capture(source_of(tmp_path / "fp8.py", bk=32, dtype=a, rhs=b), "gemm", target=target)
    assert compile_cubin(emit(kernel, target).source, CompileOptions.resolve(target).flags())[:4] == b"\x7fELF"


def fp8_table(exponent: int, mantissa: int) -> np.ndarray:
    result = np.zeros(256, np.float64)
    bias = 2 ** (exponent - 1) - 1
    for code in range(256):
        fraction = code & (2**mantissa - 1)
        power = (code >> mantissa) & (2**exponent - 1)
        value = (
            fraction / 2**mantissa * 2.0 ** (1 - bias)
            if power == 0
            else (1 + fraction / 2**mantissa) * 2.0 ** (power - bias)
        )
        if exponent == 4 and code & 0x7F == 0x7F:
            value = np.nan
        elif exponent == 5 and power == 31:
            value = np.nan if fraction else np.inf
        result[code] = -value if code & 128 else value
    return result


@pytest.mark.parametrize("a,b", FP8_PAIRS)
@pytest.mark.parametrize("m,n,k,bm,bn", ((50, 38, 70, 32, 32), (19, 7, 33, 16, 8), (65, 19, 0, 32, 24)))
def test_fp8_card(tmp_path: Path, gpu: CUDADevice, a: str, b: str, m: int, n: int, k: int, bm: int, bn: int) -> None:
    if gpu.nvidia.sm < 89:
        pytest.skip("fp8 MMA needs sm_89 or later")
    kernel = capture(source_of(tmp_path / "fp8.py", bm, bn, 32, a, b), "gemm", target=gpu.nvidia)
    rng = np.random.default_rng(81)
    raw = [
        rng.integers(0, 112, shape, dtype=np.uint8) | (rng.integers(0, 2, shape, dtype=np.uint8) << 7)
        for shape in ((m, k), (k, n))
    ]
    decoded = [
        fp8_table(4, 3)[x] if dtype == "fp(4, 3)" else fp8_table(5, 2)[x] for x, dtype in zip(raw, (a, b), strict=True)
    ]
    tensors = [gpu.to_device(x) for x in (*raw, np.zeros((m, n), np.float32))]
    for tensor, dtype in zip(tensors[:2], (a, b), strict=True):
        tensor.element = fp(4, 3) if dtype == "fp(4, 3)" else fp(5, 2)
    program = gpu.program(kernel)
    launch = replace(Tiles((m, n), (bm, bn)).blocks, threads=max(128, program.unit.threads))
    program.call(program.geometry(launch), tensors).enqueue(gpu.default_stream)
    agree(tensors[2].to_host(), decoded[0], decoded[1], k, "f32")


@pytest.mark.parametrize("a,b", (("f16", "bf16"), ("bf16", "f16")))
@pytest.mark.parametrize("bk", (8, 16, 24))
def test_tf32_emission(tmp_path: Path, a: str, b: str, bk: int) -> None:
    target = targets.Nvidia.of("sm_80")
    kernel = capture(source_of(tmp_path / "tf32.py", bk=bk, dtype=a, rhs=b), "gemm", target=target)
    statement = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    plan = NvidiaMma().plan(statement, target)
    assert isinstance(plan, MatrixPlan) and "exact via tf32" in plan.description
    source = emit(kernel, target).source
    assert plan.instruction.spelling in source and "cp.async" not in source and "ldmatrix" not in source


@pytest.mark.parametrize("a,b", (("f16", "bf16"), ("bf16", "f16")))
@pytest.mark.parametrize("m,n,k,bk", ((50, 38, 70, 16), (19, 7, 33, 8), (16, 8, 0, 8), (65, 19, 25, 24)))
def test_tf32_card(tmp_path: Path, gpu: CUDADevice, a: str, b: str, m: int, n: int, k: int, bk: int) -> None:
    if gpu.nvidia.sm < 80:
        pytest.skip("tf32 MMA needs sm_80 or later")
    kernel = capture(source_of(tmp_path / "tf32.py", 32, 32, bk, a, b), "gemm", target=gpu.nvidia)
    rng = np.random.default_rng(51)
    values = [rng.uniform(-1, 1, shape).astype(np.float32) for shape in ((m, k), (k, n))]
    raw = [bf16_words(x) if dtype == "bf16" else x.astype(np.float16) for x, dtype in zip(values, (a, b), strict=True)]
    decoded = [
        bf16_values(x) if dtype == "bf16" else x.astype(np.float32) for x, dtype in zip(raw, (a, b), strict=True)
    ]
    tensors = [gpu.to_device(x) for x in (*raw, np.zeros((m, n), np.float32))]
    for tensor, dtype in zip(tensors[:2], (a, b), strict=True):
        if dtype == "bf16":
            tensor.element = BF16
    program = gpu.program(kernel)
    launch = replace(Tiles((m, n), (32, 32)).blocks, threads=max(128, program.unit.threads))
    program.call(program.geometry(launch), tensors).enqueue(gpu.default_stream)
    agree(tensors[2].to_host(), decoded[0], decoded[1], k, "f32")


@pytest.mark.parametrize("arch", ("sm_80", "sm_86", "sm_120a"))
@pytest.mark.parametrize("a,b", (("f16", "bf16"), ("bf16", "f16")))
def test_tf32_compile(
    tmp_path: Path,
    compile_cubin: Callable[[str, tuple[str, ...]], bytes],
    a: str,
    b: str,
    arch: str,
) -> None:
    target = targets.Nvidia.of(arch)
    kernel = capture(source_of(tmp_path / "tf32.py", bk=24, dtype=a, rhs=b), "gemm", target=target)
    assert compile_cubin(emit(kernel, target).source, CompileOptions.resolve(target).flags())[:4] == b"\x7fELF"


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("dtype", ("f16", "bf16"))
@pytest.mark.parametrize("bk", (8, 16))
def test_compile(
    tmp_path: Path,
    compile_cubin: Callable[[str, tuple[str, ...]], bytes],
    arch: str,
    dtype: str,
    bk: int,
) -> None:
    target = targets.Nvidia.of(arch)
    kernel = capture(source_of(tmp_path / "gemm.py", bk=bk, dtype=dtype), "gemm", target=target)
    assert compile_cubin(emit(kernel, target).source, CompileOptions.resolve(target).flags())[:4] == b"\x7fELF"


def bf16_words(array: np.ndarray) -> np.ndarray:
    bits = array.astype(np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_values(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize(
    ("m", "n", "k", "bm", "bn", "bk"),
    [
        (50, 38, 70, 32, 32, 16),
        (130, 70, 100, 32, 32, 16),
        (16, 8, 8, 16, 8, 8),
        (257, 129, 33, 32, 32, 16),
        (64, 64, 64, 32, 32, 16),
        (33, 17, 56, 32, 32, 24),
        (130, 257, 33, 128, 256, 32),
        (16, 8, 0, 16, 8, 8),
    ],
)
@pytest.mark.parametrize("dtype", ("f16", "bf16"))
@pytest.mark.parametrize("output", ("f16", "f32"))
@pytest.mark.parametrize("stages", (0, 2, 3, 4))
def test_card(
    tmp_path: Path,
    gpu: CUDADevice,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    dtype: str,
    output: str,
    stages: int,
) -> None:
    if gpu.nvidia.sm < 80:
        pytest.skip("mma f16/bf16 needs sm_80 or later")
    file = source_of(tmp_path / "gemm.py", bm, bn, bk, dtype, output=output, stages=stages, generic=stages == 0)
    try:
        kernel = capture(file, "gemm", target=gpu.nvidia)
        program = gpu.program(kernel)
    except TargetError as error:
        if "shared-memory bytes" not in str(error):
            raise
        pytest.skip(f"this card refuses the stage buffers, as it should: {error}")
    rng = np.random.default_rng(0)
    a, b = (rng.uniform(-1, 1, shape).astype(np.float32) for shape in ((m, k), (k, n)))
    aw, bw = (bf16_words(x) if dtype == "bf16" else x.astype(np.float16) for x in (a, b))
    a, b = (bf16_values(x) if dtype == "bf16" else x.astype(np.float32) for x in (aw, bw))
    tensors = [gpu.to_device(x) for x in (aw, bw, np.zeros((m, n), np.float16 if output == "f16" else np.float32))]
    if dtype == "bf16":
        for tensor in tensors[:2]:
            tensor.element = BF16
    launch = replace(Tiles((m, n), (bm, bn)).blocks, threads=max(program.unit.threads, 128))
    program.call(program.geometry(launch), tensors).enqueue(gpu.default_stream)
    actual = tensors[2].to_host()
    agree(actual, a, b, k, output)
    # The C ABI represents BF16 differently; test it through explicit f32 inputs of the same exact values.
    cpu = targets.CPU.of("avx512")
    # The C backend has no asynchronous copy pipeline, so its oracle runs the same loop without stages.
    plain = source_of(tmp_path / "plain.py", bm, bn, bk, dtype, output=output, stages=0, generic=True)
    oracle_kernel = (
        capture(plain, "gemm", dtypes={"a": BF16, "b": BF16}, target=cpu)
        if dtype == "bf16"
        else capture(plain, "gemm", target=cpu)
    )
    result = np.zeros((m, n), actual.dtype)
    arguments = [a if dtype == "bf16" else aw, b if dtype == "bf16" else bw, result]
    emulate(oracle_kernel, cpu).run(Tiles((m, n), (bm, bn)), arguments)
    agree(result, a, b, k, output)
    if dtype == "bf16":
        file = source_of(
            tmp_path / "cpu.py",
            bm,
            bn,
            bk,
            "f32",
            output=output,
            stages=0,
            generic=True,
        )
        oracle_kernel = capture(file, "gemm", target=cpu)
    result.fill(0)
    build(oracle_kernel, cpu)(*arguments, launch=Tiles((m, n), (bm, bn)))
    agree(result, a, b, k, output)


def agree(actual: np.ndarray, a: np.ndarray, b: np.ndarray, k: int, output: str) -> None:
    reference = a.astype(np.float64) @ b.astype(np.float64)
    bound = 8 * 2**-24 * (k + 8) * (np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64)))
    if output == "f16":
        bound += np.abs(reference.astype(np.float16).astype(np.float64) - reference)
        bound += np.spacing(np.abs(reference).astype(np.float16))
    assert np.all(np.abs(actual.astype(np.float64) - reference) <= bound)


@pytest.mark.parametrize("epilogue", ("acc * 2.0 + 1.0", "acc - acc.sum(axis=1, keepdims=True)"))
def test_home_reads(tmp_path: Path, epilogue: str) -> None:
    target = targets.Nvidia.of("sm_80")
    file = source_of(tmp_path / "gemm.py", epilogue=epilogue)
    kernel = capture(file, "gemm", target=target)
    unit = emit(kernel, target)
    spilled = "sum" in epilogue
    assert ("float* acc" in unit.source) == spilled
    stage_bytes = 2 * (32 * 24 + 16 * 40)
    assert unit.shared_bytes == stage_bytes * 4 + (32 * 33 * 4 if spilled else 0)
    cpu = targets.CPU.of("avx512")
    a, b = np.ones((32, 16), np.float16), np.ones((16, 32), np.float16)
    out = np.zeros((32, 32), np.float32)
    build(kernel, cpu)(a, b, out, launch=Tiles(out.shape, (32, 32)))
    assert np.all(out == (-496 if spilled else 33))


@pytest.mark.parametrize("epilogue", ("acc * 2.0 + 1.0", "acc - acc.sum(axis=1, keepdims=True)"))
def test_home_reads_on_card(tmp_path: Path, gpu: CUDADevice, epilogue: str) -> None:
    if gpu.nvidia.sm < 80:
        pytest.skip("mma f16/bf16 needs sm_80 or later")
    kernel = capture(source_of(tmp_path / "gemm.py", epilogue=epilogue), "gemm", target=gpu.nvidia)
    tensors = [
        gpu.to_device(x)
        for x in (
            np.ones((32, 16), np.float16),
            np.ones((16, 32), np.float16),
            np.zeros((32, 32), np.float32),
        )
    ]
    program = gpu.program(kernel)
    launch = replace(Tiles((32, 32), (32, 32)).blocks, threads=max(program.unit.threads, 128))
    program.call(program.geometry(launch), tensors).enqueue(gpu.default_stream)
    assert np.all(tensors[2].to_host() == (-496 if "sum" in epilogue else 33))


def test_home_stages_another_product(tmp_path: Path, gpu: CUDADevice) -> None:
    if gpu.nvidia.sm < 80:
        pytest.skip("mma f16/bf16 needs sm_80 or later")
    file = source_of(tmp_path / "gemm.py", bm=32, bn=16, bk=16, generic=True)
    file.write_text(
        file.read_text()
        .replace("    for ks", "    other = zeros(f32, 32, 16)\n    for ks")
        .replace(
            "    c[rows, cols]",
            "        other += acc.to(f16) @ b[slice(ks, ks + 16), cols]\n    c[rows, cols]",
        )
        .replace("(acc).to", "(other).to")
    )
    kernel = capture(file, "gemm", target=gpu.nvidia)
    tensors = [
        gpu.to_device(x)
        for x in (
            np.ones((32, 16), np.float16),
            np.ones((16, 16), np.float16),
            np.zeros((32, 16), np.float32),
        )
    ]
    program = gpu.program(kernel)
    assert "float* acc" not in program.unit.source
    launch = replace(Tiles((32, 16), (32, 16)).blocks, threads=max(program.unit.threads, 128))
    program.call(program.geometry(launch), tensors).enqueue(gpu.default_stream)
    assert np.all(tensors[2].to_host() == 256)


def test_generic_expression_operand(tmp_path: Path, gpu: CUDADevice) -> None:
    if gpu.nvidia.sm < 80:
        pytest.skip("mma f16/bf16 needs sm_80 or later")
    file = source_of(tmp_path / "gemm.py", generic=True)
    file.write_text(file.read_text().replace("a[rows, slice(ks, ks + 16)]", "(a[rows, slice(ks, ks + 16)] + f16(0.0))"))
    kernel = capture(file, "gemm", target=gpu.nvidia)
    tensors = [
        gpu.to_device(x)
        for x in (
            np.ones((32, 16), np.float16),
            np.ones((16, 32), np.float16),
            np.zeros((32, 32), np.float32),
        )
    ]
    program = gpu.program(kernel)
    assert "mma.sync" in program.unit.source and "cp.async" not in program.unit.source
    launch = replace(Tiles((32, 32), (32, 32)).blocks, threads=max(program.unit.threads, 128))
    program.call(program.geometry(launch), tensors).enqueue(gpu.default_stream)
    assert np.all(tensors[2].to_host() == 16)


@pytest.mark.parametrize(
    ("m", "n", "k", "bm", "bn", "bk"),
    [
        (50, 38, 70, 32, 32, 16),
        (130, 70, 100, 32, 32, 16),
        (16, 8, 8, 16, 8, 8),
        (257, 129, 33, 32, 32, 16),
    ],
)
def test_oracles(tmp_path: Path, m: int, n: int, k: int, bm: int, bn: int, bk: int) -> None:
    cpu = targets.CPU.of("avx512")
    kernel = capture(source_of(tmp_path / "gemm.py", bm, bn, bk), "gemm", target=cpu)
    rng = np.random.default_rng(1)
    a, b = (rng.uniform(-1, 1, shape).astype(np.float16) for shape in ((m, k), (k, n)))
    emulated, compiled = (np.zeros((m, n), np.float32) for _ in range(2))
    emulate(kernel, cpu).run(Tiles((m, n), (bm, bn)), [a, b, emulated])
    build(kernel, cpu)(a, b, compiled, launch=Tiles((m, n), (bm, bn)))
    agree(emulated, a, b, k, "f32")
    agree(compiled, a, b, k, "f32")


@pytest.mark.parametrize(
    ("arch", "bm", "bn", "bk", "threads"),
    [
        ("apple_m4", 16, 16, 8, 128),
        ("apple_m4", 32, 32, 16, 128),
        ("gfx942", 16, 16, 16, 64),
        ("gfx950", 32, 32, 32, 128),
    ],
)
def test_other_matrix_units(tmp_path: Path, arch: str, bm: int, bn: int, bk: int, threads: int) -> None:
    target = targets.Target.of(arch)
    kernel = capture(source_of(tmp_path / "gemm.py", bm, bn, bk), "gemm", target=target)
    s = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    plan = (AppleSimdgroup() if isinstance(target, targets.Apple) else AMDMFMA()).plan(s, target)
    assert isinstance(plan, MatrixPlan)
    assert plan.threads == threads and plan.warp_size == target.warp_size
    unit = emit(kernel, target, lineinfo=False)
    assert unit.threads == threads and plan.instruction.spelling in unit.source
    assert "cp.async" not in unit.source and "_s += " not in unit.source
    assert str(threads) in unit.source
    if isinstance(target, targets.Apple):
        assert not AppleSimdgroup.owned_fragments and not AppleSimdgroup.async_copy
        assert "simdgroup_store(_f_acc" in unit.source and "simdgroup_load(_f_acc" in unit.source
        assert "threadgroup float* acc" in unit.source
        assert unit.shared_bytes == bm * bn * 4 + 2 * (bm * (bk + 8) + bk * (bn + 8))
        assert "transpose" not in unit.source
    else:
        assert AMDMFMA.owned_fragments and not AMDMFMA.async_copy
        assert "float* acc" not in unit.source and "__builtin_trap()" in unit.source


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_amd_k8_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture, arch: str) -> None:
    target = targets.Target.of(arch)
    kernel = capture(source_of(tmp_path / "gemm.py", 16, 16, 8), "gemm", target=target)
    with caplog.at_level(logging.INFO, logger="krn.compiler.emit"):
        unit = emit(kernel, target)
    assert fused_loop(unit.source) and "__builtin_amdgcn_mfma" not in unit.source
    assert all(token in caplog.text for token in ("gfx942", "gfx950", "16x16x8", "divisible by 16"))


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
def test_apple_stage_and_mirror_budget(tmp_path: Path, caplog: pytest.LogCaptureFixture, dtype: str) -> None:
    target = targets.Apple.of("apple_m4")
    kernel = capture(source_of(tmp_path / "gemm.py", 64, 64, 64, dtype), "gemm", target=target)
    with caplog.at_level(logging.INFO, logger="krn.compiler.emit"):
        unit = emit(kernel, target)
    assert fused_loop(unit.source) and "simdgroup_multiply_accumulate" not in unit.source
    assert "operand stages and accumulator mirror" in caplog.text and "32768" in caplog.text


@pytest.mark.parametrize("bm,bn,bk", ((8, 8, 8), (16, 16, 8), (32, 32, 16)))
def test_apple_bfloat_uses_raw_words_and_native_matrix_stages(tmp_path: Path, bm: int, bn: int, bk: int) -> None:
    target = targets.Apple.of("apple_m4")
    kernel = capture(source_of(tmp_path / "gemm.py", bm, bn, bk, "bf16"), "gemm", target=target)
    s = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    plan = AppleSimdgroup().plan(s, target)
    assert isinstance(plan, MatrixPlan) and plan.instruction.operands == (BF16, BF16)
    unit = emit(kernel, target, lineinfo=False)
    assert unit.shared_bytes == 4 * bm * bn + 2 * (bm * (bk + 8) + bk * (bn + 8))
    assert all(
        token in unit.source
        for token in (
            "const device ushort* a",
            "const device ushort* b",
            "threadgroup bfloat* _t",
            "as_type<bfloat>(a[",
            "as_type<bfloat>(b[",
            "simdgroup_matrix<bfloat, 8, 8> _a",
            "simdgroup_float8x8 _f_acc",
            "simdgroup_multiply_accumulate(_f_acc",
        )
    )
    assert "_s += " not in unit.source


@pytest.mark.parametrize("bm,bn,bk", ((4, 16, 8), (16, 4, 8), (16, 16, 4)))
@pytest.mark.parametrize("dtype", ("f16", "bf16"))
def test_apple_matrix_tiles_require_eight_element_granules(
    tmp_path: Path, bm: int, bn: int, bk: int, dtype: str
) -> None:
    target = targets.Apple.of("apple_m4")
    kernel = capture(source_of(tmp_path / "gemm.py", bm, bn, bk, dtype), "gemm", target=target)
    s = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    reason = AppleSimdgroup().plan(s, target)
    assert isinstance(reason, str) and "BM divisible by 8, BN by 8, BK by 8" in reason
    source = emit(kernel, target, lineinfo=False).source
    assert fused_loop(source) and "simdgroup_multiply_accumulate" not in source


@pytest.mark.parametrize("bm,bn", ((16, 16), (32, 64), (64, 32), (64, 128), (128, 128), (128, 256)))
def test_mfma_coordinates_cover_each_element_once(tmp_path: Path, bm: int, bn: int) -> None:
    """Model of AMD's documented layout, not an AMD hardware test."""
    target = targets.AMD.of("gfx942")
    kernel = capture(source_of(tmp_path / "gemm.py", bm, bn, 16), "gemm", target=target)
    statement = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    plan = AMDMFMA().plan(statement, target)
    assert isinstance(plan, MatrixPlan)
    expressions = AMDMFMA().coordinates(plan)
    coordinates = [
        tuple(
            eval(
                e.replace("threadIdx.x", str(lane)).replace("/", "//"),
                {},
                {"_mt": mt, "_nt": nt, "_fr": fr},
            )
            for e in expressions
        )
        for lane in range(plan.threads)
        for mt in range(plan.mt)
        for nt in range(plan.nt)
        for fr in range(4)
    ]
    assert len(coordinates) == len(set(coordinates)) == bm * bn
    assert set(coordinates) == {(r, c) for r in range(bm) for c in range(bn)}


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
def test_mfma_documented_layout_model(tmp_path: Path, dtype: str) -> None:
    """Evaluate emitted load/ownership indices against documented MFMA semantics; this is not hardware.

    AMD's matrix-cores-cdna blog, __builtin_amdgcn_mfma_f32_16x16x16f16 example, verifies
    A=(lane%16, lane//16*4+j), B=(lane//16*4+j, lane%16), D=B's coordinates.
    """
    target = targets.AMD.of("gfx942")
    kernel = capture(source_of(tmp_path / "gemm.py", 16, 16, 16, dtype), "gemm", target=target)
    statement = next(s for s in walk(kernel.body) if isinstance(s, Accumulate))
    plan = AMDMFMA().plan(statement, target)
    assert isinstance(plan, MatrixPlan)
    unit = AMDMFMA()
    ai, bi = unit.operand_indices(plan)
    product = "\n".join(unit.product("acc", plan, "a_stage", "b_stage"))
    assert ai in product and bi in product
    rng = np.random.default_rng(217)
    raw = [rng.uniform(-1, 1, (16, 16)).astype(np.float32) for _ in range(2)]
    a, b = [bf16_values(bf16_words(x)) if dtype == "bf16" else x.astype(np.float16) for x in raw]
    stages = [np.pad(x, ((0, 0), (0, 8)), constant_values=np.nan).ravel() for x in (a, b)]
    av, bv = (np.zeros((64, 4), np.float64) for _ in range(2))
    for lane in range(64):
        for j in range(4):
            names = {"_lane": lane, "_wr": 0, "_wc": 0, "_mt": 0, "_nt": 0, "_kk": 0, "_j": j}
            aidx, bidx = (eval(e.replace("/", "//"), {}, names) for e in (ai, bi))
            av[lane, j], bv[lane, j] = stages[0][aidx + j], stages[1][bidx]
            assert av[lane, j] == a[lane % 16, lane // 16 * 4 + j]
            assert bv[lane, j] == b[lane // 16 * 4 + j, lane % 16]
    actual = np.zeros((16, 16), np.float32)
    expressions = unit.coordinates(plan)
    for lane in range(64):
        for fr in range(4):
            row, col = (
                eval(
                    e.replace("threadIdx.x", str(lane)).replace("/", "//"),
                    {},
                    {"_mt": 0, "_nt": 0, "_fr": fr},
                )
                for e in expressions
            )
            actual[row, col] = sum(av[row + 16 * (k // 4), k % 4] * bv[col + 16 * (k // 4), k % 4] for k in range(16))
    agree(actual, a, b, 16, "f32")


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
@pytest.mark.parametrize("dtype", ("f16", "bf16"))
@pytest.mark.parametrize("bk", (16, 32))
def test_mfma_compiles_to_amdgpu(tmp_path: Path, arch: str, dtype: str, bk: int) -> None:
    probe = subprocess.run([CLANG[0], "--print-targets"], capture_output=True, text=True, timeout=30, check=False)
    if "amdgcn" not in probe.stdout:
        pytest.skip("this clang has no AMDGPU backend")
    target = targets.AMD.of(arch)
    kernel = capture(source_of(tmp_path / "gemm.py", 32, 32, bk, dtype), "gemm", target=target)
    source = emit(kernel, target, lineinfo=False).source
    unit, assembly = tmp_path / "unit.hip", tmp_path / "unit.s"
    unit.write_text(source)
    command = [
        CLANG[0],
        "-x",
        "hip",
        f"--offload-arch={arch}",
        "--offload-device-only",
        "-nogpulib",
        "-nogpuinc",
        "-I",
        str(Path(__file__).resolve().parents[1] / "hip_shim"),
        "-O2",
        "-S",
        "-o",
        str(assembly),
        str(unit),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    mnemonic = "v_mfma_f32_16x16x16_f16" if dtype == "f16" else "v_mfma_f32_16x16x16_bf16"
    found = sorted(set(words(assembly.read_text())) & {mnemonic, mnemonic + "_1k"})
    assert found
    logging.getLogger(__name__).info("%s %s BK=%d: %s", arch, dtype, bk, ", ".join(found))


@pytest.mark.parametrize(
    "m,n,k,bm,bn,bk",
    [
        (50, 38, 70, 32, 32, 16),
        (130, 70, 100, 32, 32, 16),
        (16, 8, 8, 16, 8, 8),
        (257, 129, 33, 32, 32, 16),
        (20, 17, 24, 16, 16, 8),
        (16, 8, 0, 16, 8, 8),
    ],
)
@pytest.mark.parametrize("epilogue", ("acc", "acc * 2.0 + 1.0", "acc - acc.sum(axis=1, keepdims=True)"))
@pytest.mark.parametrize("dtype", ("f16", "bf16"))
def test_apple_matrix_on_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    epilogue: str,
    dtype: str,
) -> None:
    metal = pytest.importorskip("krn.runtime.metal", reason="PyObjC and Metal are macOS only")
    try:
        gpu = metal.MetalDevice(Target.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))
    if dtype == "bf16":
        if gpu.target.arch != "apple_m4":
            pytest.skip("bf16 matrix validation targets the M4")
        # Metal binds raw buffers; its shared C validator has no public bf16 storage constructor yet.
        monkeypatch.setitem(NUMPY, BF16, np.dtype("uint16"))
    kernel = capture(
        source_of(tmp_path / "gemm.py", bm, bn, bk, dtype, epilogue=epilogue),
        "gemm",
        target=gpu.target,
    )
    rng = np.random.default_rng(34)
    values = [rng.uniform(-1, 1, shape).astype(np.float32) for shape in ((m, k), (k, n))]
    raw = [bf16_words(x) if dtype == "bf16" else x.astype(np.float16) for x in values]
    a, b = [bf16_values(x) if dtype == "bf16" else x.astype(np.float32) for x in raw]
    actual, expected = (np.zeros((m, n), np.float32) for _ in range(2))
    program = MetalRun(gpu, kernel)
    assert "simdgroup_multiply_accumulate" in program.program.unit.source
    launch = replace(Tiles((m, n), (bm, bn)).blocks, threads=max(128, program.program.unit.threads))
    program.run(launch, [*raw, actual])
    emulate(kernel, gpu.target).run(Tiles((m, n), (bm, bn)), [a, b, expected])
    if epilogue == "acc":
        agree(actual, a, b, k, "f32")
    else:
        assert np.allclose(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("epilogue", ("acc * 2.0 + 1.0", "acc - acc.sum(axis=1, keepdims=True)"))
def test_apple_opaque_reads_and_writes_use_the_mirror(tmp_path: Path, epilogue: str) -> None:
    target = targets.Apple.of("apple_m4")
    file = source_of(tmp_path / "gemm.py", 16, 16, 8, epilogue="acc")
    file.write_text(file.read_text().replace("    c[rows, cols]", f"    acc = {epilogue}\n    c[rows, cols]"))
    unit = emit(capture(file, "gemm", target=target), target, lineinfo=False)
    source = unit.source
    product = source.index("simdgroup_multiply_accumulate")
    spill = source.index("simdgroup_store(_f_acc", product)
    fill = source.index("simdgroup_load(_f_acc", spill)
    store = source.index("c[", fill)
    assert product < spill < fill < store
    assert "_fr" not in source
    assert "threadgroup float* acc" in source


@pytest.mark.parametrize("arch", ("apple_m4", "gfx942", "gfx950"))
def test_matrix_stages_preserve_half_bits_for_expressions(tmp_path: Path, arch: str) -> None:
    target = targets.Target.of(arch)
    file = source_of(tmp_path / "gemm.py", 32, 32, 16)
    file.write_text(file.read_text().replace("a[rows, ks]", "(a[rows, ks] + f16(0.0))"))
    source = emit(capture(file, "gemm", target=target), target, lineinfo=False).source
    assert "_s += " not in source
    if isinstance(target, targets.Apple):
        assert "threadgroup half* _t" in source and "= (half)(" in source
    else:
        assert "__builtin_bit_cast(unsigned short, (_Float16)" in source


def test_apple_counts_all_matrix_stages_before_choosing_fragments(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = targets.Apple.of("apple_m4")
    file = source_of(tmp_path / "gemm.py", 32, 32, 64)
    product = "        acc += a[rows, ks] @ b[ks, cols]"
    file.write_text(file.read_text().replace(product, "\n".join([product] * 3)))
    kernel = capture(file, "gemm", target=target)
    with caplog.at_level(logging.INFO, logger="krn.compiler.emit"):
        unit = emit(kernel, target)
    assert "simdgroup_multiply_accumulate" not in unit.source and fused_loop(unit.source)
    assert "mirrors and other shared buffers" in caplog.text
    assert unit.shared_bytes <= target.shared_bytes


def test_memory_limited_fma_does_not_update_an_aliased_operand_in_place(tmp_path: Path) -> None:
    target = targets.Apple.of("apple_m4")
    file = source_of(tmp_path / "gemm.py", 64, 64, 64)
    file.write_text(
        file.read_text().replace(
            "    c[rows, cols]",
            "    acc += acc.to(f16) @ b[slice(0, 64), cols]\n    c[rows, cols]",
        )
    )
    kernel = capture(file, "gemm", target=target)
    with pytest.raises(CaptureError, match="shared memory"):
        emit(kernel, target)

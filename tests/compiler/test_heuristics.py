"""Each backend choice changes source, with real compiler and device checks where available."""

import json
import logging
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Const, Vector, atomic, block, grid, items, kernel, shared_array, struct
from krn.arithmetic import ceildiv
from krn.compiler.api import capture
from krn.compiler.cuda import CUDARenderer
from krn.compiler.emit import Emitter, Renderer, Unit, dimensions
from krn.compiler.heuristics import DEFAULTS, Decisions, Extent, Inlining, aliased, decide
from krn.compiler.hip import HIPRenderer
from krn.compiler.ir import Kernel
from krn.compiler.msl import LANGUAGE_VERSION, MSLRenderer
from krn.facts.targets import AMD, CPU, Apple, Nvidia, Target
from krn.frontend.geometry import Blocks, Geometry, Items, Tiles
from krn.frontend.types import Scalar, f32
from krn.runtime.compile_options import CompileOptions
from krn.runtime.cpu import CLANG, CPUProgram, CPURenderer
from krn.runtime.runtime import CPUDevice
from krn.runtime.settings import Settings
from tests.helpers.devices import CUDARun, MetalRun

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice
    from krn.runtime.metal import MetalDevice

THIS = Path(__file__)
ARCHES = ("avx512", "sm_90a", "sm_120a", "gfx942", "apple_m4")


@kernel
def update(x: Vector[f32], y: Vector[f32]) -> None:
    for i in items(x.shape[0]):
        y[i] = y[i] * 2 + x[i]


@kernel
def copied[S: Scalar](x: Vector[S], y: Vector[S]) -> None:
    for i in items(x.shape[0]):
        y[i] = x[i]


@kernel
def activated(x: Vector[f32], y: Vector[f32]) -> None:
    tile = block.tile(16)
    y[tile] = x[tile].silu()


@kernel
def shifted(value: f32) -> f32:
    return value + 1


@kernel
def called(x: Vector[f32], y: Vector[f32]) -> None:
    for i in items(x.shape[0]):
        y[i] = max(shifted(x[i]), 0.0)


@kernel
def tiled(x: Vector[f32], y: Vector[f32]) -> None:
    tile = block.tile(16)
    y[tile] = x[tile] + 1


@kernel
def counted(x: Vector[f32], y: Vector[f32]) -> None:
    for i in items(x.shape[0]):
        value = x[i]
        for _k in range(4):
            value = value * 2
        while value < 1:
            value = value + 1
        y[i] = value


@kernel
def accumulated(x: Vector[f32], y: Vector[f32]) -> None:
    for i in items(x.shape[0]):
        atomic.add(y, 0, x[i])


@kernel
def synchronized(x: Vector[f32], y: Vector[f32]) -> None:
    grid.sync()
    for i in items(x.shape[0]):
        y[i] = x[i]


@kernel
def shared_read(x: Vector[f32], y: Vector[f32]) -> None:
    scratch = shared_array(f32, 16)
    for i in items(x.shape[0]):
        scratch[0] = x[i]
        y[i] = scratch[0]


@struct
class Inputs:
    v: Vector[f32]


@kernel
def from_struct(s: Inputs, y: Vector[f32]) -> None:
    for i in items(s.v.shape[0]):
        y[i] = s.v[i]


@struct
class Config:
    width: Const[int] = 16
    scale: Const[float] = 1.0
    enabled: Const[bool] = False


@struct
class Required:
    width: Const[int]


def renderer(target: Target, decisions: Decisions | None = None) -> Renderer:
    match target:
        case Nvidia():
            return CUDARenderer(target, decisions)
        case AMD():
            return HIPRenderer(target, decisions)
        case Apple():
            return MSLRenderer(target, decisions)
        case CPU():
            return CPURenderer(target, decisions)
    raise AssertionError(target)


def rendered(
    name: str = "update",
    arch: str = "sm_90a",
    env: Mapping[str, str] | None = None,
    *,
    overlap: bool | None = None,
    launch: Geometry | None = None,
) -> tuple[Kernel, Decisions, Unit]:
    target = Target.of(arch)
    ir = capture(THIS, name, target=target)
    decisions = decide(ir, target, Settings.from_environ(env or {}), aliased=overlap, launch=launch)
    return ir, decisions, Emitter(ir, renderer(target, decisions)).emit()


@pytest.mark.parametrize("arch", ARCHES)
def test_default_rules_and_launch_geometry(arch: str) -> None:
    ir, decisions, unit = rendered(arch=arch)
    assert decisions.unroll.value == DEFAULTS.unroll
    assert decisions.maxregs.value is None
    assert decisions.const.value and not decisions.restrict.value
    assert dimensions(ir, Items(1025), decisions) == (ceildiv(1025, 256 * DEFAULTS.unroll), 1, 1, 256)
    assert rendered("tiled", arch)[1].block.value == (1 if arch == "avx512" else 128)
    _, explicit, source = rendered(arch=arch, launch=Blocks(3, threads=64))
    assert explicit.block.value == 64
    assert dimensions(ir, Blocks(3, threads=64), explicit) == (3, 1, 1, 64)
    assert source.source and unit.source
    with pytest.raises(ValueError, match="tile product"):
        explicit.floor(128)
    assert decisions.floor(512).block.value == 512
    assert decisions.floor(1) is decisions


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "size,expected", [(64, Inlining.INLINE), (65, Inlining.PLAIN), (512, Inlining.PLAIN), (513, Inlining.NEVER)]
)
def test_helper_size_rule(arch: str, size: int, expected: Inlining) -> None:
    _, decisions, _ = rendered(arch=arch)
    assert decisions.helper(size) is expected


def test_extent_conflicts() -> None:
    read, write = Extent(0, 16, False), Extent(8, 24, True)
    assert aliased([read, write]) and write.conflicts(read)
    assert not aliased([read, read]) and not aliased([read, Extent(16, 32, True)])
    assert not aliased([])


@pytest.mark.parametrize("separate_buffers", [False, True])
def test_c_compiles_and_runs_disjoint_and_overlapping_arguments(separate_buffers: bool) -> None:
    ir, _, unit = rendered(arch="avx512")
    x = np.arange(1025, dtype=np.float32)
    y = np.ones_like(x) if separate_buffers else x
    expected = y.copy() * 2 + x.copy()
    CPUProgram(ir, unit)(x, y, launch=Items(len(x)))
    np.testing.assert_array_equal(y, expected)


def test_c_helper_rule_compiles_and_runs() -> None:
    ir, _, unit = rendered("activated", "avx512")
    x = np.linspace(-1, 1, 37, dtype=np.float32)
    y = np.empty_like(x)
    CPUProgram(ir, unit)(x, y, launch=Tiles(x.shape, (16,)))
    np.testing.assert_allclose(y, x / (1 + np.exp(-x)), rtol=1e-6)


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a"])
def test_cuda_variants_compile_when_nvrtc_is_available(
    compile_cubin: Callable[[str, tuple[str, ...]], bytes], arch: str
) -> None:
    source = rendered("activated", arch, launch=Blocks(2, threads=64))[2].source
    assert compile_cubin(source, CompileOptions.resolve(Target.of(arch)).flags())[:4] == b"\x7fELF"


def test_cuda_runs_when_a_card_is_available(gpu: CUDADevice) -> None:
    ir = capture(THIS, "update", target=gpu.nvidia)
    x, y = np.arange(37, dtype=np.float32), np.ones(37, dtype=np.float32)
    CUDARun(gpu, ir).run(Items(len(x)), [x, y])
    np.testing.assert_array_equal(y, x + 2)


def test_metal_rules_compile_and_run_when_available(metal_gpu: MetalDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    metal = pytest.importorskip("krn.runtime.metal")
    ir, decisions, _ = rendered(
        "activated",
        "apple_m4",
        launch=Blocks(2, threads=64),
    )
    monkeypatch.setattr(metal_gpu, "program", lambda ir: metal.MetalProgram(metal_gpu, ir, decisions))
    program = MetalRun(metal_gpu, ir)
    x, y = np.zeros(37, dtype=np.float32), np.ones(37, dtype=np.float32)
    program.run(Blocks(3, threads=64), [x, y])
    np.testing.assert_array_equal(y, x)


@pytest.mark.parametrize("name", ["activated", "counted"])
def test_msl_rule_syntax_against_shim(tmp_path: Path, name: str) -> None:
    unit = tmp_path / "heuristics.metal"
    unit.write_text(
        rendered(
            name,
            "apple_m4",
            launch=Blocks(2, threads=64),
        )[2].source
    )
    done = subprocess.run(
        [
            CLANG[0],
            "-fsyntax-only",
            "-x",
            "c++",
            "-std=c++17",
            "-Wno-unknown-attributes",
            "-I",
            str(THIS.resolve().parents[1] / "msl_shim"),
            str(unit),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_hip_variant_compiles_if_clang_has_amdgpu(tmp_path: Path) -> None:
    backends = subprocess.run([CLANG[0], "--print-targets"], capture_output=True, text=True, timeout=30, check=True)
    if "amdgcn" not in backends.stdout:
        pytest.skip("this clang has no AMDGPU backend")
    unit = tmp_path / "heuristics.hip"
    unit.write_text(
        rendered(
            "activated",
            "gfx942",
            launch=Blocks(2, threads=64),
        )[2].source
    )
    command = [
        CLANG[0],
        "-x",
        "hip",
        "--offload-arch=gfx942",
        "--offload-device-only",
        "-nogpulib",
        "-nogpuinc",
        "-I",
        str(THIS.resolve().parents[1] / "hip_shim"),
        "-S",
        "-o",
        str(tmp_path / "unit.s"),
        str(unit),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr


def test_metal_qualifier_probe_when_available(metal_gpu: MetalDevice) -> None:
    metal = pytest.importorskip("Metal")
    options = metal.MTLCompileOptions.new()
    options.setLanguageVersion_(getattr(metal, f"MTLLanguageVersion{LANGUAGE_VERSION[0]}_{LANGUAGE_VERSION[1]}"))
    source = """#include <metal_stdlib>
using namespace metal;
inline __attribute__((always_inline)) float forced(float v) { return v; }
__attribute__((noinline)) float separate(float v) { return v; }
[[max_total_threads_per_threadgroup(64)]] kernel void probe(
    device float* __restrict x [[buffer(0)]], uint i [[thread_position_in_grid]]
) {
    #pragma unroll 4
    for (int j = 0; j < 4; ++j) x[i] = separate(forced(x[i]));
}
"""
    library, error = metal_gpu.core.newLibraryWithSource_options_error_(source, options, None)
    assert library is not None, error
    logging.getLogger(__name__).info("Metal accepted __restrict, helper qualifiers, launch bound and unroll pragma")


def test_explanation_reports_helpers_callees_and_does_not_launch(device: CPUDevice) -> None:
    x = np.arange(17, dtype=np.float32)
    original = x.copy()
    explanation = device.launcher(called).explain(x, x)
    np.testing.assert_array_equal(x, original)
    assert "x and y" in explanation.variant
    assert explanation.space == ("x: items(x.shape[0])",)
    assert explanation.geometry["threads"] == DEFAULTS.element_threads
    assert any(
        "maximum" in name and size > 0 and "inline" in qualifier for name, size, qualifier, _ in explanation.helpers
    )
    assert ("shifted", ("callee",)) in explanation.roles
    assert "call site" in str(explanation)
    assert json.loads(json.dumps(explanation.to_json())) == explanation.to_json()
    explicit = device.launcher(called).explain(x, x, launch=Blocks(3, threads=64))
    assert explicit.geometry["grid"] == [3, 1, 1] and explicit.geometry["threads"] == 64


def test_cooperative_inferred_grid_is_capped(gpu: CUDADevice) -> None:
    if not gpu.cooperative:
        pytest.skip("device does not support cooperative launches")
    x = np.arange(1 << 22, dtype=np.float32)
    a, b = gpu.to_device(x), gpu.to_device(np.empty_like(x))
    launcher = gpu.launcher(synchronized)
    explanation = launcher.explain(a, b)
    assert explanation.cap and "resident capacity" in explanation.cap
    launcher.submit(a, b).wait(60)
    np.testing.assert_array_equal(b.to_host(), x)
    explicit = gpu.launcher(synchronized, Items(len(x))).explain(a, b)
    assert explicit.cap is None

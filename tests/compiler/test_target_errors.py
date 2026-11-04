"""Explicit target choices fail with a reason and an alternative; unset choices retain target defaults."""

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from krn import BackendCompile, Cache, Debug, Run, T, configure
from krn.cache.tuning import TuningDb
from krn.compiler.api import capture
from krn.compiler.artifacts import inspect_kernel
from krn.compiler.debug import Artifact
from krn.compiler.dtypes import BOOL, F64, U32, U64, Dtype
from krn.compiler.emit import check_shared
from krn.compiler.errors import Span, TargetError
from krn.compiler.lower import emit
from krn.compiler.msl import MSLRenderer
from krn.facts import catalog
from krn.facts.targets import Apple, Target
from krn.frontend.atomic import AtomicOp
from krn.lib import lib
from krn.runtime.compile_options import CompileOptions, Dialect
from krn.runtime.cpu import compile_c
from krn.runtime.settings import BACKENDS, Settings
from tests.compiler.test_heuristics import THIS, rendered, update
from tests.compiler.test_tensor_cores import source_of as matrix_source
from tests.runtime.test_metal_backend import boundary as boundary

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice, CUDATensor

ARCHES = ("avx512", "sm_90a", "gfx942", "apple_m4")


def diagnostic(error: TargetError, feature: str, arch: str) -> None:
    assert feature in error.feature and arch == error.target
    assert error.reason.strip() and error.instead.strip()
    message = str(error)
    assert feature in message and arch in message
    assert f"{error.feature} is not available on {arch}: " in message
    assert f"{error.reason.rstrip('.')}. Instead: {error.instead.rstrip('.')}." in message


@pytest.mark.parametrize(
    "feature,value,arch,name",
    [
        *(("KRN_WORKERS", "2", arch, "update") for arch in ("sm_90a", "gfx942", "apple_m4")),
        ("KRN_CACHE_DIR", "/tmp/krn-target-errors", "apple_m4", "update"),
    ],
)
def test_explicit_setting_and_unset_default(feature: str, value: str, arch: str, name: str) -> None:
    with pytest.raises(TargetError) as caught:
        rendered(name, arch, {feature: value})
    diagnostic(caught.value, {"KRN_WORKERS": "run.workers", "KRN_CACHE_DIR": "cache.dir"}[feature], arch)
    assert rendered(name, arch)[2].source


@pytest.mark.parametrize("arch", ARCHES)
def test_every_settings_field_declares_backend_behavior(arch: str) -> None:
    for item in fields(Settings):
        assert "backends" in item.metadata, item.name
        assert set(item.metadata["backends"]) <= set(BACKENDS)
    assert rendered(arch=arch)[2].source


OPTION_REFUSALS = [
    *(("CXX_STD", "c++2c", arch) for arch in ("sm_90a", "apple_m4")),
    ("CXX_STD", "c++17", "avx512"),
    ("CXX_STD", "c++14", "apple_m4"),
    *(("OPT", level, "sm_90a") for level in ("1", "2", "3", "s")),
    *(("OPT", level, "apple_m4") for level in ("0", "1", "3", "s", "fast-compile")),
    ("MARCH", "apple_m4", "apple_m4"),
    ("LINEINFO", "1", "apple_m4"),
    ("DEBUG", "1", "apple_m4"),
]


@pytest.mark.parametrize("name,value,arch", OPTION_REFUSALS)
@pytest.mark.parametrize("qualified", (False, True))
def test_compile_option_refusals_and_other_backend_inertness(name: str, value: str, arch: str, qualified: bool) -> None:
    target = Target.of(arch)
    backend = Dialect.of(target).value
    field = {"CXX_STD": "std"}.get(name, name.lower())
    if name in {"LINEINFO", "DEBUG"}:
        snapshot = Settings(debug=Debug(debug=value == "1") if field == "debug" else Debug(lineinfo=value == "1"))
        feature = f"debug.{field}"
    else:
        feature = f"compile.{backend}.{field}"
        change = configure(**{feature: value})
        snapshot = change.chosen
    with pytest.raises(TargetError) as caught:
        CompileOptions.resolve(target, {}, settings=snapshot)
    diagnostic(caught.value, feature, arch)
    assert CompileOptions.resolve(target, {}, settings=Settings()).flags()


@pytest.mark.parametrize("arch", ARCHES)
def test_honored_explicit_choices_and_precedence(arch: str) -> None:
    target = Target.of(arch)
    assert CompileOptions.resolve(target, {"KRN_DEBUG": "0", "KRN_DEBUG_LINEINFO": "0"}).flags()
    for mode in ("auto", "never"):
        assert rendered(
            arch=arch,
            env={
                "KRN_INLINE": mode,
                "KRN_INLINE_LIMIT": "2",
                "KRN_CONST": mode,
                "KRN_RESTRICT": mode,
                "KRN_UNROLL": "4",
            },
        )[2].source


@pytest.mark.parametrize("arch", ("sm_90a", "gfx942"))
def test_max_standard_is_a_target_default(arch: str) -> None:
    assert CompileOptions.resolve(Target.of(arch), {}, std="max").flags()


@pytest.mark.parametrize("arch", ARCHES)
def test_every_missing_catalog_lowering_has_a_target_diagnostic(arch: str) -> None:
    target = Target.of(arch)
    for entry in catalog.ENTRIES:
        if not entry.lowering(target):
            with pytest.raises(TargetError) as caught:
                catalog.spell(entry.name, target)
            diagnostic(caught.value, entry.name, arch)
    assert rendered(arch=arch)[2].source


@pytest.mark.parametrize("arch", ARCHES)
def test_shared_memory_limit(arch: str) -> None:
    target = Target.of(arch)
    ir = capture(THIS, "update", target=target)
    with pytest.raises(TargetError) as caught:
        check_shared(ir, target, target.shared_bytes + 1)
    diagnostic(caught.value, "shared memory", arch)
    check_shared(ir, target, 0)
    assert emit(ir, target).source


@pytest.mark.parametrize("arch", ("apple_m1", "apple_m2", "apple_m3", "apple_m4", "apple_m5"))
@pytest.mark.parametrize("intrinsic", (False, True))
def test_atomic_refusals_keep_source_location(tmp_path: Path, arch: str, intrinsic: bool) -> None:
    path = tmp_path / "atomic.py"
    name = "intrinsic" if intrinsic else "atomic"
    call = "atomic_system.add(x[0], 1)" if intrinsic else "atomic.add(x, 0, 1)"
    imports = "from krn.facts.intrinsics import atomic_system\n" if intrinsic else ""
    path.write_text(
        "from krn import T, Vector, atomic, kernel\n" + imports + f"@kernel\ndef run(x: Vector[T.u64]):\n    {call}\n"
    )
    target = Target.of(arch)
    with pytest.raises(TargetError) as caught:
        capture(path, "run", target=target)
    diagnostic(caught.value, "atomic", arch)
    assert caught.value.span is not None
    assert str(path) in str(caught.value)
    supported = tmp_path / f"{name}_supported.py"
    supported.write_text(path.read_text().replace("T.u64", "T.u32"))
    assert emit(capture(supported, "run", target=target), target).source


@pytest.mark.parametrize("dtype", (BOOL, F64, U64), ids=str)
def test_metal_atomic_renderer_checks_the_plan_before_type_lowering(dtype: Dtype) -> None:
    renderer = MSLRenderer(Apple.of("apple_m4"))
    with pytest.raises(TargetError) as caught:
        renderer.atomic(AtomicOp.ADD, dtype, "p", "v", None)
    diagnostic(caught.value, "atomic.add", "apple_m4")
    assert renderer.atomic(AtomicOp.ADD, U32, "p", "v", None)


@pytest.mark.parametrize("arch", ("sm_80", "gfx942", "apple_m4", "avx512"))
def test_fp8_matrix_refusals(tmp_path: Path, arch: str) -> None:
    target = Target.of(arch)
    source = matrix_source(tmp_path / "fp8.py", bk=32, dtype="fp(4, 3)")
    with pytest.raises(TargetError) as caught:
        emit(capture(source, "gemm", target=target), target)
    diagnostic(caught.value, "fp8 matrix product", arch)
    source = matrix_source(tmp_path / "f16.py", bk=32, dtype="f16")
    assert emit(capture(source, "gemm", target=target), target).source


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("stages", (1, 2, 9))
def test_explicit_pipeline_depth(tmp_path: Path, arch: str, stages: int) -> None:
    target = Target.of(arch)
    source = matrix_source(tmp_path / "stages.py", stages=stages)
    if arch == "sm_90a" and stages == 2:
        assert emit(capture(source, "gemm", target=target), target).source
    else:
        with pytest.raises(TargetError) as caught:
            emit(capture(source, "gemm", target=target), target)
        diagnostic(caught.value, "pipelined(stages)", arch)
    source = matrix_source(tmp_path / "auto.py")
    assert emit(capture(source, "gemm", target=target), target).source


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("schedule", ("WarpSpecialized", "PingPong"))
def test_explicit_pipeline_schedule(tmp_path: Path, arch: str, schedule: str) -> None:
    target = Target.of(arch)
    source = matrix_source(tmp_path / "schedule.py")
    text = source.read_text()
    source.write_text(
        f"from krn import {schedule}\n"
        + text.replace("pipelined(a.shape[1], 16)", f"pipelined(a.shape[1], 16, schedule={schedule}())")
    )
    with pytest.raises(TargetError) as caught:
        capture(source, "gemm", target=target)
    diagnostic(caught.value, schedule, arch)
    source = matrix_source(tmp_path / "auto.py")
    assert emit(capture(source, "gemm", target=target), target).source


@pytest.mark.parametrize("arch", ARCHES)
def test_explicit_tile_storage(tmp_path: Path, arch: str) -> None:
    target = Target.of(arch)
    source = matrix_source(tmp_path / "storage.py")
    source.write_text(source.read_text().replace("zeros(f32, 32, 32)", "zeros(f32, 32, 32, space=1)"))
    with pytest.raises(TargetError) as caught:
        capture(source, "gemm", target=target)
    diagnostic(caught.value, "zeros(space)", arch)
    source = matrix_source(tmp_path / "default.py")
    assert emit(capture(source, "gemm", target=target), target).source


@pytest.mark.parametrize("tool", ("print(x[0])", "assert x[0] > 0", "breakpoint()"))
def test_metal_debug_tools(tmp_path: Path, tool: str) -> None:
    source = tmp_path / "tools.py"
    prefix = "from krn import kernel, Vector, T\n@kernel\ndef run(x: Vector[T.f32]):\n"
    source.write_text(prefix + f"    {tool}\n")
    target = Target.of("apple_m4")
    with pytest.raises(TargetError) as caught:
        emit(capture(source, "run", target=target), target)
    diagnostic(caught.value, tool.split("(")[0].split()[0], target.arch)
    source = tmp_path / "without_tools.py"
    source.write_text(prefix + "    x[0] = 1\n")
    assert emit(capture(source, "run", target=target), target).source


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("artifact", (Artifact.PTX, Artifact.SASS, Artifact.ASM))
def test_unavailable_binary_artifacts(arch: str, artifact: Artifact) -> None:
    target = Target.of(arch)
    ir = capture(THIS, "update", target=target)
    with pytest.raises(TargetError) as caught:
        inspect_kernel(ir, target, artifacts=frozenset({artifact}))
    diagnostic(caught.value, "KRN_DEBUG_KEEP", arch)
    assert inspect_kernel(ir, target).source


@pytest.mark.parametrize(
    "library,arch",
    [
        ("cublas", "apple_m4"),
        ("cublaslt", "gfx942"),
        ("accelerate", "sm_90a"),
        ("mps", "avx512"),
        ("hipblaslt", "apple_m4"),
        ("mkl", "sm_90a"),
        ("onednn", "apple_m4"),
        ("cusparse", "gfx942"),
    ],
)
@pytest.mark.parametrize("per_op", (False, True))
def test_vendor_library_choice(library: str, arch: str, per_op: bool, tmp_path: Path) -> None:
    mode = lib.Mode.AUTO if per_op else lib.Mode(library)
    per_operation = {lib.Op.MATMUL: lib.Mode(library)} if per_op else {}
    dispatcher = lib.Dispatcher(
        Target.of(arch), {lib.Op.MATMUL: []}, TuningDb(tmp_path / "tuning.json"), mode, per_operation, ""
    )
    problem = lib.Problem(lib.Op.MATMUL, (), (16, 16, 16), lib.LibOptions.of(dispatcher.target))
    with pytest.raises(TargetError) as caught:
        dispatcher.choose(lib.MatmulCandidate, problem, capturing=False)
    diagnostic(caught.value, "lib.", arch)
    assert "auto" in caught.value.instead
    assert rendered(arch=arch)[2].source


def test_optional_span_preserves_compiler_format(tmp_path: Path) -> None:
    path = tmp_path / "kernel.py"
    path.write_text("value\n")
    error = TargetError("feature", "arch", "reason", "alternative", span=Span(str(path), 1, 2))
    assert f"{path}:1:2: error:" in str(error) and "value" in str(error) and "^" in str(error)
    diagnostic(error, "feature", "arch")
    diagnostic(TargetError("feature", "arch", "reason.", "alternative."), "feature", "arch")


def test_cpu_compiler_options_and_cache_are_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(**{"compile.cpu.opt": "3"})
    path, _ = compile_c("void target_error_probe(void) {}", cache_dir=tmp_path / "cache")
    assert path.is_file() and path.parent == tmp_path / "cache"


@pytest.mark.parametrize(
    "variable,value", [("KRN_CACHE_DIR", "/tmp/krn-target-errors"), ("KRN_METAL_OPT", "s"), ("KRN_METAL_DEBUG", "1")]
)
def test_metal_validates_before_device_allocation(
    boundary, monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    module, _, _ = boundary
    feature = {"KRN_CACHE_DIR": "cache.dir", "KRN_METAL_OPT": "compile.metal.opt", "KRN_METAL_DEBUG": "debug.debug"}[
        variable
    ]
    configure(**{feature: True if variable == "KRN_METAL_DEBUG" else value})
    with pytest.raises(TargetError) as caught:
        module.MetalDevice(Target.of("apple_m4"))
    diagnostic(caught.value, feature, "apple_m4")
    configure(cache=Cache(), debug=Debug(), **{"compile.metal": BackendCompile()})
    assert module.MetalDevice(Target.of("apple_m4")).target.arch == "apple_m4"


def test_metal_graph_arguments_are_not_silently_replayed(boundary) -> None:
    module, _, _ = boundary
    device = module.MetalDevice(Apple.of("apple_m4"), Settings())
    recorded: list[int] = []
    graph = device.graph(lambda values: values.append(1), recorded)
    graph(recorded)
    with pytest.raises(TargetError) as caught:
        graph([])
    diagnostic(caught.value, "graph replay", "apple_m4")
    device.settings = Settings(run=Run(graphs=False))
    other: list[int] = []
    device.graph(lambda values: values.append(1), recorded)(other)
    assert other == [1]


def test_cuda_graph_arguments_are_not_silently_replayed(gpu: CUDADevice) -> None:
    tensor = gpu.empty(T.f32, 1)

    def body(value: CUDATensor) -> None:
        gpu.launcher(update)(value, value)

    graph = gpu.graph(body, tensor)
    graph(tensor)
    with pytest.raises(TargetError) as caught:
        graph(gpu.empty(T.f32, 1))
    diagnostic(caught.value, "graph replay", gpu.target.arch)


APPLE_GPUS = ("apple_m1", "apple_m2", "apple_m3", "apple_m4", "apple_m5")
FLOAT64_SITES = {  # the parameters and the statement of a kernel that names a float64 at one site
    "parameter": ("a: Vector[T.f64], out: Vector[T.f32]", "out[i] = T.f32(a[i])"),
    "constant": ("out: Vector[T.f32]", "out[i] = T.f32(T.f64(2.0))"),
    "value": ("a: Vector[T.f32], out: Vector[T.f32]", "out[i] = T.f32(T.f64(a[i]) * 2.0)"),
}


def float64_kernel(directory: Path, site: str) -> Path:
    parameters, statement = FLOAT64_SITES[site]
    file = directory / "float64.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n@kernel\n"
        f"def k({parameters}) -> None:\n    i = grid.thread_rank()\n    if i < out.shape[0]:\n        {statement}\n"
    )
    return file


@pytest.mark.parametrize("arch", APPLE_GPUS)
@pytest.mark.parametrize("site", FLOAT64_SITES)
def test_float64_on_apple_is_refused_at_capture_with_the_four_parts(tmp_path: Path, arch: str, site: str) -> None:
    with pytest.raises(TargetError) as error:
        capture(float64_kernel(tmp_path, site), "k", target=Target.of(arch))
    assert error.value.feature == "f64" and error.value.target == arch
    assert error.value.reason == "Apple GPUs have no 64-bit floating point"
    assert error.value.instead == "use f32, or run on a CPU or NVIDIA target"


@pytest.mark.parametrize("arch", ["avx512", "sm_90a"])
@pytest.mark.parametrize("site", FLOAT64_SITES)
def test_float64_is_captured_on_cpu_and_nvidia(tmp_path: Path, arch: str, site: str) -> None:
    assert capture(float64_kernel(tmp_path, site), "k", target=Target.of(arch)).name == "k"

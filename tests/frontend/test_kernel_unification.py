"""Capture classifies one decorator by use: entry point, callee, configured or folded.

Every kernel here lives in this file, so the launcher captures it from source and the tests capture it by name. Each
kernel runs on the C backend, in the emulator, on the Mac's GPU and on a CUDA card, and the results must be identical.
"""

import importlib.util
import inspect
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from operator import methodcaller
from pathlib import Path
from typing import TYPE_CHECKING, assert_type
from unittest.mock import patch

import numpy as np
import pytest

import krn
from krn import Const, Items, Kernel, Vector, configure, grid, items, kernel, struct
from krn.compiler.api import CaptureError, CPUDevice, CPUTensor, build, capture, emit
from krn.compiler.capture import ConstValues, roles
from krn.compiler.dtypes import U32, Dtype
from krn.compiler.ir import Kernel as Ir
from krn.compiler.source import Role
from krn.emulator.emulate import build as emulated
from krn.facts.targets import CPU, Target
from krn.frontend.geometry import Geometry
from krn.frontend.types import u32
from krn.runtime.compile_options import CompileOptions
from krn.runtime.cpu import CLANG, Value
from krn.runtime.runtime import DTYPES
from krn.runtime.settings import Settings
from tests.compiler.test_hip import CLANG_HAS_AMDGPU
from tests.helpers.devices import CUDARun, MetalRun

THIS = Path(__file__)

if TYPE_CHECKING:
    from krn.data.sparse import lane_sum
    from krn.itertools import (
        chain,
        drop,
        first,
        interleave,
        last,
        nth,
        pairwise,
        partition,
        second,
        tail,
        take,
        take_nth,
    )

    def generic_callee_types[F: krn.T.FloatingPoint](x: Vector[F], value: F) -> None:
        assert_type(lane_sum(value), F)
        assert_type(first(x, 0, 1, value), F)
        assert_type(last(x, 0, 1, value), F)
        assert_type(nth(x, 0, 2, 1, value), F)
        assert_type(second(x, 0, 2, value), F)
        assert_type(pairwise(x, 0, 2), Iterator[tuple[F, F]])
        assert_type(chain(x, 0, 2, x, 0, 2), Iterator[F])
        assert_type(interleave(x, x, 0, 2), Iterator[F])
        assert_type(take(x, 0, 2, 1), Iterator[F])
        assert_type(tail(x, 0, 2, 1), Iterator[F])
        assert_type(drop(x, 0, 2, 1), Iterator[F])
        assert_type(take_nth(x, 0, 2, 1), Iterator[F])
        assert_type(partition(x, 0, 2, 2), Iterator[tuple[F, ...]])


@dataclass
class Backend:
    name: str
    target: Target
    execute: Callable[[Ir, Geometry, list[Value]], None]

    def run(
        self,
        name: str,
        launch: Geometry,
        arguments: list[Value],
        consts: ConstValues | None = None,
        **arrays: np.ndarray,
    ) -> None:
        """Capture `name` for this backend with the element types of the named arrays, then run it on `arguments`."""
        dtypes: dict[str, Dtype] = {parameter: DTYPES[array.dtype] for parameter, array in arrays.items()}
        self.execute(
            capture(THIS, name, target=self.target, dtypes=dtypes, consts=consts),
            launch,
            arguments,
        )


@pytest.fixture(params=["cpu", "emulator", "metal", "cuda"])
def backend(request: pytest.FixtureRequest) -> Backend:
    match request.param:
        case "cpu":
            cpu = CPU.of("avx512")
            return Backend("cpu", cpu, lambda ir, launch, args: build(ir, cpu)(*args, launch=launch))
        case "emulator":
            target = Target.of("sm_90a")
            return Backend("emulator", target, lambda ir, launch, args: emulated(ir, target).run(launch, args))
        case "metal":
            metal = pytest.importorskip("krn.runtime.metal", reason="PyObjC and the Metal framework are macOS only")
            try:
                gpu = metal.MetalDevice(Target.of("apple_m4"))
            except metal.MetalError as error:
                pytest.skip(str(error))
            return Backend(
                "metal",
                gpu.target,
                lambda ir, launch, args: MetalRun(gpu, ir).run(launch, args),
            )
    card = request.getfixturevalue("gpu")
    return Backend("cuda", card.nvidia, lambda ir, launch, args: CUDARun(card, ir).run(launch, args))


@kernel
def nonzero(v: Vector[u32], i: u32) -> bool:
    return v[i] != 0


@kernel
def nonzero_loaded(v: Vector[u32], i: u32) -> bool:
    entry = v[i]
    return entry != 0


@kernel
def checked_load(v: Vector[u32], i: u32, reads: Vector[u32]) -> u32:
    reads[0] += 1
    value = v[i]
    return value


@kernel
def guarded_loads(v: Vector[u32], out: Vector[u32], reads: Vector[u32]) -> None:
    for t in items(1):
        n = u32(v.shape[0])
        i = u32(0)
        while i < n and checked_load(v, i, reads) != 0:
            i += 1
        out[t] = i
        out[1] = u32(i == n or checked_load(v, i, reads) != 0)
        out[2] = u32(0) if i == n else checked_load(v, i, reads)
        out[3] = checked_load(v, i, reads) if i < n else u32(0)
        out[4] = u32(i < n and checked_load(v, i, reads) != 0 and checked_load(v, i, reads) != 0)
        out[5] = u32(i == n or checked_load(v, i, reads) != 0 or checked_load(v, i, reads) != 0)


@pytest.mark.parametrize("n", [0, 1, 7])
def test_guarded_helpers_never_load_beyond_the_extent(backend: Backend, n: int) -> None:
    v, out, reads = np.ones(n, np.uint32), np.full(6, 99, np.uint32), np.zeros(1, np.uint32)
    stop = 3 if n == 7 else n
    if stop < n:
        v[stop] = 0
    backend.run("guarded_loads", Items(1), [v, out, reads], v=v, out=out, reads=reads)
    np.testing.assert_array_equal(out, [stop, int(stop == n), 0, 0, 0, int(stop == n)])
    assert reads[0] == (n if stop == n else stop + 7)


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a", "gfx942", "gfx950", "apple_m4"])
def test_guarded_helpers_render_for_each_gpu(arch: str) -> None:
    target = Target.of(arch)
    source = emit(capture(THIS, "guarded_loads", target=target), target).source
    assert "if (short_circuit" in source
    assert "if (" in source and "conditional" in source


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a"])
def test_guarded_helpers_compile_with_nvrtc(compile_cubin: Callable[[str, tuple[str, ...]], bytes], arch: str) -> None:
    target = Target.of(arch)
    source = emit(capture(THIS, "guarded_loads", target=target), target).source
    assert compile_cubin(source, CompileOptions.resolve(target).flags())[:4] == b"\x7fELF"


@pytest.mark.skipif(not CLANG_HAS_AMDGPU, reason="this clang has no AMDGPU backend")
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_guarded_helpers_compile_to_amdgpu(tmp_path: Path, arch: str) -> None:
    target = Target.of(arch)
    unit = tmp_path / "guarded.hip"
    unit.write_text(emit(capture(THIS, "guarded_loads", target=target), target).source)
    done = subprocess.run(
        [
            CLANG[0],
            "-x",
            "hip",
            f"--offload-arch={arch}",
            "--offload-device-only",
            "-nogpulib",
            "-nogpuinc",
            "-I",
            str(THIS.resolve().parents[1] / "hip_shim"),
            "-S",
            "-o",
            str(tmp_path / "guarded.s"),
            str(unit),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def test_guarded_helpers_type_check_as_msl(tmp_path: Path) -> None:
    target = Target.of("apple_m4")
    unit = tmp_path / "guarded.metal"
    unit.write_text(emit(capture(THIS, "guarded_loads", target=target), target).source)
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


@kernel
def run_length(v: Vector[u32], n: u32, out: Vector[u32]) -> None:
    for t in items(1):
        i = u32(0)
        while i < n and nonzero(v, i):
            i += 1
        out[t] = i


@kernel
def run_length_loaded(v: Vector[u32], n: u32, out: Vector[u32]) -> None:
    for t in items(1):
        i = u32(0)
        while i < n and nonzero_loaded(v, i):
            i += 1
        out[t] = i


@kernel
def run_length_alone(v: Vector[u32], out: Vector[u32]) -> None:
    for t in items(1):
        i = u32(0)
        while nonzero_loaded(v, i):
            i += 1
        out[t] = i


RUNS = [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [7, 7, 7, 7, 0]]


def first_zero(values: list[int]) -> int:
    return next((i for i, entry in enumerate(values) if entry == 0), len(values))


@pytest.mark.parametrize("values", RUNS, ids=str)
@pytest.mark.parametrize("name", ["run_length", "run_length_loaded"])
def test_a_helper_in_a_while_condition_is_evaluated_on_every_iteration(
    backend: Backend, name: str, values: list[int]
) -> None:
    """Reevaluate a helper on every iteration, guarding its loads at the extent."""
    v, out = np.array(values, np.uint32), np.full(1, 99, np.uint32)
    backend.run(name, Items(1), [v, len(values), out], v=v, out=out)
    assert out[0] == first_zero(values)


@pytest.mark.parametrize("values", RUNS, ids=str)
def test_a_helper_alone_in_a_while_condition_is_evaluated_on_every_iteration(
    backend: Backend, values: list[int]
) -> None:
    v, out = np.array([*values, 0], np.uint32), np.full(1, 99, np.uint32)
    backend.run("run_length_alone", Items(1), [v, out], v=v, out=out)
    assert out[0] == first_zero(values)


@struct
class Scale:
    factor: Const[int] = 3


@kernel
def configured_increment(s: Scale, x: Vector[u32], y: Vector[u32]) -> None:
    for i in items(x.shape[0]):
        y[i] = x[i] * s.factor + steps(8)


def test_default_configuration_uses_environment_but_explicit_configuration_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(**{"run.config": {"configured_increment.factor": "7"}})
    cpu = CPUDevice(CPU.of("avx512"), Settings.from_environ())
    try:
        with patch.object(krn.host.Device, "current", return_value=cpu):
            x = cpu.to_device(np.arange(7, dtype=np.uint32))
            outputs = [cpu.empty(u32, 7) for _ in range(3)]
            configured_increment(x, outputs[0])
            configured_increment[Scale(2)](x, outputs[1])
            configured_increment(Scale(5), x, outputs[2])
            cpu.default_stream.record().wait(60)
            for output, factor in zip(outputs, (7, 2, 5), strict=True):
                np.testing.assert_array_equal(output.array, x.array * factor + 3)
    finally:
        cpu.close()


def test_kernel_explain_reports_every_capture_role_without_launching(device: CPUDevice) -> None:
    x, y = np.arange(7, dtype=np.uint32), np.full(7, 99, np.uint32)
    explanation = configured_increment.explain(x, y)
    assert ("configured_increment", ("configured", "entry")) in explanation.roles
    assert ("steps", ("callee", "folded")) in explanation.roles
    assert "roles configured_increment" in str(explanation)
    assert "configured, entry" in str(explanation)
    np.testing.assert_array_equal(y, np.full(7, 99, np.uint32))
    assert configured_increment[Scale(2)].explain(x, y).kernel == "configured_increment"
    assert configured_increment.explain(x, y, launch=Items(1)).geometry["inference"] == "explicit launch"


@pytest.mark.parametrize("legacy", [Items(1), krn.Blocks(1), krn.Tiles((1,), (1,)), (Scale(), Items(1))])
def test_brackets_reject_every_legacy_geometry_form(
    legacy: Geometry | tuple[Scale, Geometry],
) -> None:
    with pytest.raises(TypeError) as error:
        methodcaller("__getitem__", legacy)(configured_increment)
    assert "brackets take a configuration" in str(error.value) and "launch=" in str(error.value)


@kernel
def axpy_at(s: Scale, x: Vector[u32], y: Vector[u32]) -> None:
    for i in items(x.shape[0]):
        y[i] = y[i] + x[i] * s.factor


@kernel
def twice_axpy(s: Scale, x: Vector[u32], y: Vector[u32]) -> None:
    axpy_at(s, x, y)
    axpy_at(s, x, y)


@kernel
def steps(width: int) -> int:
    return width.bit_length() - 1


@kernel
def folded_steps(n: u32, out: Vector[u32]) -> None:
    for i in items(1):
        value = n
        for k in range(steps(8)):
            value += k
        out[i] = value


RUNTIME_STEPS_SOURCE = """
@kernel
def runtime_steps(n: u32, out: Vector[u32]) -> None:
    for i in items(1):
        out[i] = u32(steps(n))
"""


@struct
class Offset:
    base: u32

    def shifted(self, x: u32) -> u32:
        return self.base + x

    def doubled(self, x: u32) -> u32:
        return x + x


@kernel
def offset_values(o: Offset, x: Vector[u32], y: Vector[u32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = o.doubled(o.shifted(x[i]))


@pytest.mark.parametrize("n", [1, 37, 300])
@pytest.mark.parametrize("factor", [1, 3, 7])
def test_a_function_used_as_entry_callee_and_configured_kernel_gives_identical_results(
    backend: Backend, n: int, factor: int
) -> None:
    rng = np.random.default_rng(n)
    x, y0 = rng.integers(0, 1000, size=(2, n), dtype=np.uint32)
    for name, repetitions in (("axpy_at", 1), ("twice_axpy", 2)):
        y = y0.copy()
        backend.run(name, Items(n), [{}, x, y], consts={"s": {"factor": factor}}, x=x, y=y)
        np.testing.assert_array_equal(y, y0 + repetitions * factor * x)


@pytest.mark.parametrize("n", [1, 37, 300])
@pytest.mark.parametrize("factor", [1, 3, 7])
def test_configuration_in_arguments_and_brackets_launches_the_same_kernel(
    device: CPUDevice, n: int, factor: int
) -> None:
    rng = np.random.default_rng(n)
    x, y0 = rng.integers(0, 1000, size=(2, n), dtype=np.uint32)
    ordinary, configured, twice = (y0.copy() for _ in range(3))
    axpy_at(Scale(factor), x, ordinary, launch=Items(n))
    axpy_at[Scale(factor)](x, configured)
    twice_axpy[Scale(factor)](x, twice)
    device.default_stream.record().wait(60)
    np.testing.assert_array_equal(ordinary, y0 + factor * x)
    np.testing.assert_array_equal(configured, ordinary)
    np.testing.assert_array_equal(twice, y0 + 2 * factor * x)


def test_roles_follow_each_use_and_callee_capture_does_not_change_the_entry_body() -> None:
    dtypes = {"x": U32, "y": U32}
    before = capture(THIS, "axpy_at", dtypes=dtypes)
    assert roles(THIS, "axpy_at", dtypes=dtypes) == {"axpy_at": {Role.ENTRY, Role.CONFIGURED}}
    capture(THIS, "twice_axpy", dtypes=dtypes)
    assert roles(THIS, "twice_axpy", dtypes=dtypes) == {
        "twice_axpy": {Role.ENTRY, Role.CONFIGURED},
        "axpy_at": {Role.CALLEE, Role.CONFIGURED},
    }
    assert capture(THIS, "axpy_at", dtypes=dtypes).body == before.body


@pytest.mark.parametrize("n", [8, 1024])
@pytest.mark.parametrize("name", ["folded_steps", "runtime_steps"])
def test_only_constant_calls_fold_and_bounded_loops_produce_exact_results(
    backend: Backend, tmp_path: Path, name: str, n: int
) -> None:
    path = THIS
    if name == "runtime_steps":
        assert isinstance(steps, Kernel)
        path = tmp_path / "runtime_steps.py"
        # Capture preserves u32 through an int helper; static checkers require a conversion that changes its dtype.
        path.write_text(
            "from krn import Vector, items, kernel\nfrom krn.frontend.types import u32\n"
            + inspect.getsource(steps.fn)
            + RUNTIME_STEPS_SOURCE
        )
    expected_roles = {Role.CALLEE, Role.FOLDED} if name == "folded_steps" else {Role.CALLEE}
    assert roles(path, name, target=backend.target) == {
        name: {Role.ENTRY},
        "steps": expected_roles,
    }
    out = np.zeros(1, np.uint32)
    backend.execute(capture(path, name, target=backend.target), Items(1), [n, out])
    np.testing.assert_array_equal(out, [n + 3 if name == "folded_steps" else n.bit_length() - 1])


def test_all_decorator_aliases_return_the_same_kernel_object_type() -> None:
    for name in ("device", "template", "comptime"):
        assert not hasattr(krn, name)
    assert isinstance(steps, Kernel)
    wrapped = kernel(steps.fn)
    assert isinstance(wrapped, Kernel)
    assert wrapped.fn is steps.fn
    assert wrapped.name == steps.name


ALIAS_SOURCE = """
from krn import Const, Vector, grid, kernel, struct
from krn.frontend.types import u32

@struct
class Scale:
    factor: Const[int] = 3

@spelling
def doubled(x: u32) -> u32:
    return x + x

@spelling
def steps(width: int) -> int:
    return width.bit_length() - 1

@spelling
def configured(s: Scale, x: Vector[u32], y: Vector[u32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = doubled(x[i]) * s.factor + steps(8)
"""


@pytest.mark.parametrize("spelling", ["device", "template", "comptime", "kernel"])
def test_decorator_spelling_preserves_emitted_c_and_real_launcher_results(
    tmp_path: Path, device: CPUDevice, spelling: str
) -> None:
    path = tmp_path / f"alias_{spelling}.py"
    path.write_text(ALIAS_SOURCE.replace("@spelling", f"@{spelling}"))
    if spelling != "kernel":
        with pytest.raises(CaptureError) as error:
            capture(path, "configured", consts={"s": {"factor": 7}}, target=device.target)
        assert "write @kernel" in str(error.value)
        return
    ir = capture(path, "configured", consts={"s": {"factor": 7}}, target=device.target)
    assert emit(ir, device.target, lineinfo=False).entry == "configured"
    assert roles(path, "configured") == {
        "configured": {Role.ENTRY, Role.CONFIGURED},
        "doubled": {Role.CALLEE},
        "steps": {Role.CALLEE, Role.FOLDED},
    }
    spec = importlib.util.spec_from_file_location(f"alias_{spelling}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = np.random.default_rng(37).integers(0, 1000, 37, dtype=np.uint32)
    y = np.zeros_like(x)
    module.configured(module.Scale(7), x, y, launch=Items(len(x)))
    device.default_stream.record().wait(60)
    np.testing.assert_array_equal(y, 14 * x + 3)


@pytest.mark.parametrize("n", [1, 37, 300])
def test_decorated_and_undecorated_struct_methods_are_callees(backend: Backend, n: int) -> None:
    x = np.random.default_rng(n).integers(0, 1000, n, dtype=np.uint32)
    y = np.zeros_like(x)
    assert roles(THIS, "offset_values", target=backend.target) == {
        "offset_values": {Role.ENTRY},
        "Offset.shifted": {Role.CALLEE},
        "Offset.doubled": {Role.CALLEE},
    }
    backend.run("offset_values", Items(n), [{"base": 7}, x, y], x=x, y=y)
    np.testing.assert_array_equal(y, 2 * (x + 7))


def test_calling_a_kernel_on_the_host_launches_it(device: CPUDevice) -> None:
    x = CPUTensor[u32, int](np.arange(37, dtype=np.uint32))
    y = CPUTensor[u32, int](np.zeros(37, np.uint32))
    axpy_at(Scale(), x, y, launch=Items(37))
    device.default_stream.record().wait(60)
    np.testing.assert_array_equal(y.array, 3 * x.array)

"""The Metal boundary faked at the PyObjC module, including the failure paths unavailable on this host."""

import asyncio
import sys
import threading
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Blocks, Items, T, kernel
from krn.compiler.api import capture
from krn.compiler.dtypes import F32, I32
from krn.compiler.errors import TargetError
from krn.compiler.ir import StructType, TensorType
from krn.facts.targets import CPU, Apple
from krn.runtime import api
from krn.runtime.runtime import CPUDevice
from krn.runtime.settings import Cache, Run, Settings, configure
from tests.compiler.test_tensor_cores import source_of as matrix_source
from tests.helpers import example_kernels
from tests.helpers.example_kernels import saxpy

if TYPE_CHECKING:
    from krn.runtime import metal


class Buffer:
    def __init__(self, length: int) -> None:
        self.data = bytearray(length)

    def contents(self):
        return SimpleNamespace(as_buffer=lambda length: memoryview(self.data)[:length])


class Command:
    def __init__(self) -> None:
        self.encoded: list[tuple] = []
        self.failure: str | None = None
        self.completed = threading.Event()
        self.completed.set()
        self.committed = False

    def computeCommandEncoder(self):
        return SimpleNamespace(
            setComputePipelineState_=lambda pipeline: self.encoded.append(("pipeline", pipeline)),
            setBuffer_offset_atIndex_=lambda *args: self.encoded.append(("buffer", *args)),
            setBytes_length_atIndex_=lambda *args: self.encoded.append(("bytes", *args)),
            dispatchThreadgroups_threadsPerThreadgroup_=lambda *args: self.encoded.append(("dispatch", *args)),
            endEncoding=lambda: self.encoded.append(("end",)),
        )

    def encodeSignalEvent_value_(self, *args) -> None:
        self.encoded.append(("signal", *args))

    def encodeWaitForEvent_value_(self, *args) -> None:
        self.encoded.append(("wait", *args))

    def commit(self) -> None:
        self.committed = True

    def waitUntilCompleted(self) -> None:
        assert self.completed.wait(10)

    def error(self) -> str | None:
        return self.failure

    def GPUEndTime(self) -> float:
        return 0.25

    def status(self) -> int:
        return 4 if self.completed.is_set() else 2


class Core:
    def __init__(self) -> None:
        self.unified = True
        self.family = True
        self.allocate = True
        self.compile = True
        self.pipeline = True
        self.builds = 0
        self.commands: list[Command] = []
        self.versions: list[int] = []
        self.math_modes: list[int] = []
        self.fp_functions: list[int] = []
        self.limit = 1024

    def name(self) -> str:
        return "Apple M4 Pro"

    def hasUnifiedMemory(self) -> bool:
        return self.unified

    def supportsFamily_(self, family: int) -> bool:
        assert family == 1009
        return self.family

    def newSharedEvent(self):
        return SimpleNamespace()

    def newCommandQueue(self):
        return SimpleNamespace(commandBuffer=self.command)

    def command(self) -> Command:
        command = Command()
        self.commands.append(command)
        return command

    def newBufferWithLength_options_(self, length: int, options: int) -> Buffer | None:
        assert options == 0
        return Buffer(length) if self.allocate else None

    def newLibraryWithSource_options_error_(self, source: str, options, error):
        assert "kernel void" in source and error is None
        self.builds += 1
        library = SimpleNamespace(newFunctionWithName_=lambda name: name)
        return (library, None) if self.compile else (None, "compile failed")

    def newComputePipelineStateWithFunction_error_(self, function: str, error):
        assert function and error is None
        pipeline = SimpleNamespace(maxTotalThreadsPerThreadgroup=lambda: self.limit, threadExecutionWidth=lambda: 32)
        return (pipeline, None) if self.pipeline else (None, "pipeline failed")


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch):
    core = Core()
    fake = SimpleNamespace(
        MTLCreateSystemDefaultDevice=lambda: core,
        MTLResourceStorageModeShared=0,
        MTLLanguageVersion3_2=196610,
        MTLMathModeSafe=0,
        MTLMathFloatingPointFunctionsPrecise=1,
        MTLCompileOptions=SimpleNamespace(
            new=lambda: SimpleNamespace(
                setLanguageVersion_=core.versions.append,
                setMathMode_=core.math_modes.append,
                setMathFloatingPointFunctions_=core.fp_functions.append,
            )
        ),
        MTLSizeMake=lambda *size: size,
    )
    # Inject before importing as well, so these tests run where PyObjC is not installed.
    existing = sys.modules.get("krn.runtime.metal")
    monkeypatch.setitem(sys.modules, "Metal", fake)
    module = import_module("krn.runtime.metal")
    monkeypatch.setattr(module, "Metal", fake)
    yield module, core, fake
    if existing is None:
        sys.modules.pop("krn.runtime.metal", None)


@pytest.fixture
def device(boundary) -> metal.MetalDevice:
    module, _, _ = boundary
    return module.MetalDevice(Apple.of("apple_m4"), Settings())


@dataclass(frozen=True, kw_only=True)
class Record:
    value: int


@kernel
def no_arguments() -> None:
    pass


def test_shared_storage_and_allocation(device: metal.MetalDevice, boundary) -> None:
    module, core, _ = boundary
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    tensor = device.to_device(source)
    assert tensor.shape == (3, 2) and tensor.strides == (8, 4)
    assert tensor.dtype is T.f32
    assert tensor.nbytes == 24
    assert np.array_equal(tensor.to_host(), source)
    np.frombuffer(tensor.buffer.contents().as_buffer(tensor.nbytes), np.float32)[0] = 9
    assert tensor.array[0, 0] == 9
    assert np.shares_memory(np.asarray(tensor), tensor.array)
    assert np.asarray(tensor, dtype=np.float64).dtype == np.float64
    assert repr(tensor) == "MetalTensor(float32[3, 2])"
    assert device.empty(T.f32, 0).shape == (0,)
    assert device.empty(T.fp.min, 2).dtype is T.f16
    with pytest.raises(ValueError, match="negative"):
        device.empty(T.f32, -1)
    with pytest.raises(TypeError, match="no representation"):
        device.to_device(np.array([1], dtype=np.complex64))
    core.allocate = False
    with pytest.raises(module.MetalError, match="allocate"):
        device.empty(T.f32, 1)


def test_backend_selection_and_missing_hardware(boundary, monkeypatch: pytest.MonkeyPatch) -> None:
    module, core, fake = boundary
    backend = module.MetalBackend()
    target = Apple.of("apple_m4")
    assert backend.device(target) is backend.device(target)
    assert isinstance(module.BACKEND, api.Backend)
    with pytest.raises(TypeError, match="Apple target"):
        backend.device(CPU.of("avx512"))
    with pytest.raises(TypeError, match="Apple target"):
        module.MetalDevice(CPU.of("avx512"))
    for unified, family in ((False, True), (True, False)):
        core.unified, core.family = unified, family
        with pytest.raises(module.MetalError, match="Apple9"):
            module.MetalDevice(target)
    monkeypatch.setattr(fake, "MTLCreateSystemDefaultDevice", lambda: None)
    with pytest.raises(module.MetalError, match="no Metal"):
        module.MetalDevice(target)


def test_packing_cache_and_geometry(device: metal.MetalDevice, boundary, tmp_path) -> None:
    module, core, _ = boundary
    x = device.to_device(np.arange(17, dtype=np.float32))
    launcher = device.launcher(saxpy, Items(17))
    assert device.launcher(saxpy, Items(17)) is launcher
    variant = launcher.plan((T.f32(2), x, x))
    program = variant.program
    assert device.program(program.ir) is device.program(program.ir)
    assert core.builds == 2 and core.versions == [196610, 196610]
    assert core.math_modes == [0, 0] and core.fp_functions == [1, 1]
    call = program.call(program.geometry(Items(17)), [2, x, x])
    for invalid in ((-1, 1, 1, 32), (1, 1, 1, 0)):
        with pytest.raises(ValueError, match="nonnegative"):
            program.call(invalid, [2, x, x])
    packed = np.frombuffer(call.scalars, program.layout)[0]
    assert packed["a"] == 2 and packed["x_n0"] == 17 and packed["y_n0"] == 17
    call.enqueue(device.default_stream)
    encoded = core.commands[-1].encoded
    assert encoded[1][1] is x.buffer and encoded[2][1] is x.buffer
    assert encoded[-2] == ("dispatch", (1, 1, 1), (256, 1, 1))
    assert core.commands[-1].committed
    assert module.native(np.int32(3)) == 3
    assert module.native(Record(value=5)) == {"value": 5}
    assert module.native(T.f32(3)) == 3
    assert module.native(x) is x
    with pytest.raises(TypeError, match="Metal kernel"):
        module.native(np.ones(2))
    with pytest.raises(TypeError, match="match"):
        program.call((1, 1, 1, 64), [2, device.empty(T.i32, 17), x])
    with pytest.raises(TypeError, match="expected"):
        program.call((1, 1, 1, 64), [2, 3, x])
    buffers = {}
    scalars = np.zeros((), np.dtype([("r_value", np.int32)]))
    program._bind("r", StructType("Record", (("value", I32),)), {"value": 7}, buffers, scalars)
    assert scalars["r_value"] == 7
    program._bind("r", StructType("Empty", ()), {}, buffers, scalars)
    with pytest.raises(TypeError, match="match"):
        program._bind("r", TensorType(F32, 2), x, buffers, scalars)
    before = len(core.commands)
    program.call((0, 1, 1, 64), [2, x, x]).enqueue(device.default_stream)
    assert len(core.commands) == before
    other = module.MetalDevice(Apple.of("apple_m4"))
    with pytest.raises(ValueError, match="launch device"):
        call.enqueue(other.default_stream)
    with pytest.raises(ValueError, match="pipeline limit"):
        program.geometry(Blocks(1, threads=2048))
    program.unit = replace(program.unit, threads=128)
    with pytest.raises(ValueError, match="tile product"):
        program.geometry(Blocks(1, threads=64))
    assert launcher.extent(x, True).high - launcher.extent(x, True).low == x.nbytes
    with pytest.raises(TypeError, match="Metal tensor"):
        launcher.extent(np.ones(2), True)
    with pytest.raises(ValueError, match="geometry cannot be inferred"):
        device.launcher(saxpy).geometry(variant, (2, x, x))
    device.settings = Settings(cache=Cache(dir=tmp_path))
    with pytest.raises(TargetError, match=r"cache.dir"):
        device.program(program.ir, replace(program.decisions, settings=device.settings))
    device.settings = Settings()
    assert device.program(program.ir, replace(program.decisions, settings=device.settings)) is program
    assert not tuple(tmp_path.iterdir())


def test_compilation_failures(device: metal.MetalDevice, boundary) -> None:
    module, core, _ = boundary
    ir = capture(example_kernels.__file__, "saxpy", target=device.target)
    core.compile = False
    with pytest.raises(module.MetalError, match="compile failed"):
        device.program(ir)
    core.compile, core.pipeline = True, False
    with pytest.raises(module.MetalError, match="pipeline failed"):
        device.program(ir)
    core.pipeline = True
    assert device.program(ir) is device.program(ir)


def test_bound_graph_streams_and_events(device: metal.MetalDevice, boundary) -> None:
    module, core, _ = boundary
    x = device.to_device(np.arange(8, dtype=np.float32))
    launcher = device.launcher(saxpy, Blocks(1, threads=64))
    bound = launcher.bind(T.f32(2), x, x)
    bound()
    bound.rebind(T.f32(4), x, x)
    bound.submit().wait(5)
    launcher.submit(T.f32(2), x, x).wait(5)
    calls = []

    def body(tensor, *, scale=2):
        calls.append(1)
        launcher(T.f32(scale), tensor, tensor)

    before = len(core.commands)
    graph = device.graph(body, x, scale=2)
    assert calls == [1] and len(core.commands) == before
    assert not device.default_stream.capturing
    graph.submit(x, scale=2).wait(5)
    assert calls == [1]
    for args, kwargs in (((x,), {"scale": 3}), ((device.empty(T.f32, 8),), {"scale": 2}), ((x,), {})):
        with pytest.raises(TargetError, match="graph replay"):
            graph(*args, **kwargs)
    assert len(calls) == 1
    device.settings = Settings(run=Run(graphs=False))
    device.graph(body, x)(x)
    assert len(calls) == 2
    device.settings = Settings()

    def nested() -> None:
        device.graph(body, x)

    with pytest.raises(RuntimeError, match="nested"):
        device.graph(nested)
    assert not device.default_stream.capturing

    def record_during_capture() -> None:
        device.default_stream.record()

    def allocate_during_capture() -> None:
        device.empty(T.f32, 1)

    for forbidden in (record_during_capture, allocate_during_capture):
        with pytest.raises(RuntimeError, match="launches only"):
            device.graph(forbidden)

    def another_stream() -> None:
        with device.stream():
            bound()

    with pytest.raises(RuntimeError, match="one stream"):
        device.graph(another_stream)
    stream = device.stream()
    assert stream.describe()["type"] == "MetalStream"
    assert device.launch_stream() is device.default_stream
    with stream:
        assert device.launch_stream() is stream
        launcher(T.f32(2), x, x)
    first = stream.record()
    first.wait(5)
    device.default_stream.wait(first)
    assert core.commands[-1].encoded[0] == ("wait", first.shared, first.value)
    last = device.default_stream.record()
    last.wait(5)
    assert last.elapsed_since(first) == 0
    with pytest.raises(TypeError, match="same device"):
        last.elapsed_since(module.CPUEvent())
    with pytest.raises(TypeError, match="same device"):
        stream.wait(module.CPUEvent())
    with CPUDevice(CPU.of("avx512")).stream(), pytest.raises(TypeError, match="another device"):
        device.launch_stream()
    with module.MetalDevice(device.target).stream(), pytest.raises(TypeError, match="another device"):
        device.launch_stream()
    blocker = core.command()
    blocker.completed.clear()
    stream.commands.append(blocker)
    pending = stream.record()
    with pytest.raises(RuntimeError, match="completed"):
        pending.elapsed_since(first)
    with pytest.raises(TimeoutError):
        pending.wait(0)
    blocker.completed.set()
    pending.wait(5)

    async def finish() -> None:
        await pending
        await device.synchronize()

    asyncio.run(finish())
    assert not stream.commands
    failed = core.command()
    failed.failure = "device failed"
    stream.commands.append(failed)
    with pytest.raises(module.MetalError, match="device failed"):
        stream.record().wait(5)


def test_empty_argument_pack_and_default_backend(device: metal.MetalDevice, boundary, monkeypatch) -> None:
    module, _, _ = boundary
    launcher = device.launcher(no_arguments, Blocks(1, threads=32))
    launcher()
    assert launcher.encode(()).scalars == b""
    configure(**{"run.target": "apple_m4"})
    monkeypatch.setattr(module.BACKEND, "device", lambda target: device)
    assert api.Device.current() is device
    configure(**{"run.target": None})
    monkeypatch.setattr(api, "System", SimpleNamespace(detect=lambda: SimpleNamespace(target=device.target)))
    assert api.Device.current() is device


@pytest.mark.parametrize("epilogue", ("acc", "acc * 2.0 + 1.0", "acc - acc.sum(axis=1, keepdims=True)"))
def test_bfloat_matrix_storage_at_the_boundary(device: metal.MetalDevice, boundary, tmp_path, epilogue: str) -> None:
    _, core, _ = boundary
    device.settings = Settings()
    ir = capture(
        matrix_source(tmp_path / "gemm.py", 16, 16, 8, "bf16", epilogue=epilogue), "gemm", target=device.target
    )
    program = device.program(ir)
    raw = np.full((16, 8), 0x3F80, np.uint16)
    a, b = device.to_device(raw), device.to_device(raw.T)
    out = device.empty(T.f32, 16, 16)
    call = program.call((1, 1, 1, 128), [a, b, out])
    call.enqueue(device.default_stream)
    assert call.buffers == (a, b, out)
    assert np.array_equal(a.array, raw)
    typed = device.empty(T.bf16, 16, 8)
    typed.array[...] = raw
    assert typed.numpy == np.dtype(np.uint16)
    program.call((1, 1, 1, 128), [typed, b, out])
    with pytest.raises(TypeError, match="does not match"):
        program.call((1, 1, 1, 128), [device.empty(T.f16, 16, 8), b, out])
    assert core.math_modes == [0] and core.fp_functions == [1]


def test_sqrt_compilation_at_the_boundary(device: metal.MetalDevice, boundary) -> None:
    _, core, _ = boundary
    ir = capture(Path(__file__).with_name("test_metal.py"), "square_root", target=device.target)
    program = device.program(ir)
    assert "precise::sqrt(" in program.unit.source and core.math_modes == [0] and core.fp_functions == [1]
    x = device.to_device(np.arange(513, dtype=np.float32))
    y = device.empty(T.f32, 513)
    program.call(program.geometry(Items(513)), [x, y]).enqueue(device.default_stream)
    assert core.commands[-1].encoded[1][1] is x.buffer
    assert core.commands[-1].encoded[2][1] is y.buffer

import ctypes
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from krn import configure, host, lib, targets
from krn.compiler.api import Arithmetic, CPUDevice, CPUProgram, build, capture
from krn.compiler.capture import ConstValues
from krn.compiler.dtypes import Dtype
from krn.hw import system
from krn.hw.devices import intel
from krn.runtime import settings as settings_state
from krn.runtime.nvrtc import Nvidia as Nvrtc
from tests.helpers.devices import MetalRun

if TYPE_CHECKING:
    from krn.compiler.ir import Kernel
    from krn.runtime.cuda_backend import CUDADevice
    from krn.runtime.metal import MetalDevice

pytest_plugins = ["tests._internals.compat.plugin"]  # records the measured hardware runs when `--compat-out` is given

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples"
SKETCH = ROOT / "experiments" / "sketch"
KERNELS = Path(__file__).resolve().parent / "kernels"
NVRTC_LIB = Path(os.environ.get("KRN_NVRTC_LIB", "/nonexistent"))


def pytest_configure() -> None:
    """The runtime tests exercise the CPU backend, so every machine runs them on a CPU target (`KRN_TARGET` wins)."""
    configure(**{"run.target": settings_state.current_settings().run.target or "avx512"})


@pytest.fixture
def device() -> CPUDevice:
    current = host.Device.current()
    assert isinstance(current, CPUDevice)
    return current


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch):
    """A fresh dispatcher for each mode: the environment is read once, when the first call builds it."""

    def use(mode: str) -> None:
        configure(**{"lib.mode": mode, "lib.tune": "off"})
        lib._dispatcher.cache_clear()

    previous = settings_state.current_settings()
    yield use
    monkeypatch.setattr(settings_state, "_current", previous)
    lib._dispatcher.cache_clear()


@pytest.fixture(scope="session")
def accelerate():
    """The `lib_accelerate` module, or a skip where the framework cannot be loaded. It is not a file on a current macOS
    (it lives in the dynamic linker's cache), so the only test of its presence is loading it."""
    module = pytest.importorskip("krn.lib.lib_accelerate")
    try:
        ctypes.CDLL(module.Accelerate.path)
    except OSError:
        pytest.skip("Accelerate is a macOS framework")
    return module


@pytest.fixture(scope="session")
def nvrtc() -> Nvrtc:
    """NVRTC 13 from the directory `scripts/activate.sh` exports as KRN_NVRTC_LIB (the nvidia wheel or a toolkit)."""
    if not (NVRTC_LIB / "libnvrtc.so.13").exists():
        pytest.skip("NVRTC 13 not found: run through `mb run` on Linux or export KRN_NVRTC_LIB")
    return Nvrtc(NVRTC_LIB)


@pytest.fixture(scope="session")
def gpu() -> CUDADevice:
    """The device of the CUDA backend for the first card. Skips without the cuda-python wheels or without a card."""
    backend = pytest.importorskip("krn.runtime.cuda_backend", reason="cuda-core and cuda-bindings are not installed")
    if (target := next((t for t in system.System.detect().targets if t.vendor == "nvidia"), None)) is None:
        pytest.skip("no CUDA card")
    return backend.BACKEND.device(target)


@pytest.fixture(scope="session")
def metal_gpu() -> MetalDevice:
    """This Mac's GPU through Metal. Skips off macOS (no PyObjC) and where the system has no Metal device."""
    metal = pytest.importorskip("krn.runtime.metal", reason="PyObjC and the Metal framework are macOS only")
    try:
        return metal.MetalDevice(targets.Apple.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="session")
def metal_program(metal_gpu: MetalDevice) -> Callable[[Kernel], MetalRun]:
    """A captured kernel compiled for this Mac's GPU by the system: `metal_program(kernel).run(launch, arrays)`."""
    return lambda kernel: MetalRun(metal_gpu, kernel)


@pytest.fixture(scope="session")
def compile_cubin() -> Callable[[str, tuple[str, ...]], bytes]:
    """NVRTC through the backend, which needs the cuda-python wheels and no card."""
    backend = pytest.importorskip("krn.runtime.cuda_backend", reason="cuda-core and cuda-bindings are not installed")
    return backend.compile_cubin


def kernel_path(file: str) -> Path:
    """A kernel file of the tests, else of the sketches."""
    if (path := KERNELS / f"{file}.py").exists():
        return path
    if not SKETCH.is_dir():
        pytest.skip("local sketches are absent")
    return SKETCH / f"{file}.py"


def build_kernel(
    file: str | Path,
    name: str,
    consts: ConstValues | None = None,
    *,
    dtypes: dict[str, Dtype] | None = None,
    arch: str = "avx512",
    warp_size: int = 16,
    arithmetic: Arithmetic = Arithmetic.STRICT,
) -> CPUProgram:
    """Capture, lower and compile one kernel for a CPU target: `build_kernel("gemm", "gemm", {"t": {"bm": 16}})`."""
    path = file if isinstance(file, Path) else kernel_path(file)
    target = replace(targets.CPU.of(arch), warp_size=warp_size)
    return build(
        capture(path, name, consts=consts, dtypes=dtypes, target=target, arithmetic=arithmetic),
        target,
    )


@pytest.fixture(autouse=True)
def no_debugger(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `breakpoint()` in an emulated kernel must never open pdb and wait for input during a test run."""
    monkeypatch.setenv("PYTHONBREAKPOINT", "0")


@pytest.fixture(autouse=True)
def optional_native_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenVINO's native plugin discovery aborts in the sandbox; opt into it with the other host probes."""
    if os.environ.get("KRN_HARDWARE") != "1":
        monkeypatch.setattr(intel, "openvino", None)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if not SKETCH.is_dir():
        missing = pytest.mark.skip(reason="local sketches are absent")
        for item in items:
            if isinstance(item, pytest.Function) and "SKETCH" in vars(item.module):
                item.add_marker(missing)
    if os.environ.get("KRN_HARDWARE") != "1":
        skip = pytest.mark.skip(reason="set KRN_HARDWARE=1 to probe the host")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings_state, "_current", settings_state.current_settings())

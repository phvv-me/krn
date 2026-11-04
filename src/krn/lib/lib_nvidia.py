"""What the NVIDIA vendor adapters (`lib_cublaslt`, `lib_cusparse`) share.

Tensors and streams are read through the two open CUDA interfaces, `__cuda_array_interface__` and `__cuda_stream__`,
which our backend, `cuda.core` and torch all implement, so an adapter never imports a backend. Scratch memory is
taken on the default stream and synchronized before anyone uses it, inside `relaxed_capture`: that is legal while
another stream is being captured (measured on an RTX 5080 with cuSPARSE 12.8), and the pointer stays valid in the
graph. Planning is not: a plan made while a capture is open leaves the library unable to enqueue into it, so an adapter
plans before and says in `capturable` whether it did.
"""

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Protocol, TypedDict, runtime_checkable

from krn.frontend.memory import Tensor, Vector
from krn.frontend.numerics import FloatingPoint, bf16, f16, f32, f64, u8
from krn.runtime.api import Device, Stream

try:
    from cuda.bindings import driver as cu
    from cuda.bindings.runtime import cudaDataType
except ImportError:
    cu = cudaDataType = None

_DATA_TYPES: dict[type[FloatingPoint], str] = {
    f16: "CUDA_R_16F",
    bf16: "CUDA_R_16BF",
    f32: "CUDA_R_32F",
    f64: "CUDA_R_64F",
}


def data_type(dtype: type[FloatingPoint]) -> int:
    if cudaDataType is None:
        raise ImportError("cuda-bindings is unavailable")
    return getattr(cudaDataType, _DATA_TYPES[dtype])


class ArrayInterface(TypedDict):
    """The keys of `__cuda_array_interface__` that the adapters read."""

    data: tuple[int, bool]
    shape: tuple[int, ...]


@runtime_checkable
class CUDAArray(Protocol):
    """Any CUDA array: the open standard of Numba, CuPy and torch, which our `CUDATensor` implements."""

    @property
    def __cuda_array_interface__(self) -> ArrayInterface: ...


@runtime_checkable
class CUDAStream(Protocol):
    """Any CUDA stream: the protocol of `cuda.core` and torch's `ExternalStream`, which our `CUDAStream` implements."""

    def __cuda_stream__(self) -> tuple[int, int]: ...


def address(tensor: Tensor) -> int:
    """The device address of the first element."""
    if not isinstance(tensor, CUDAArray):
        raise TypeError(f"{type(tensor).__name__} is not CUDA device memory")
    return tensor.__cuda_array_interface__["data"][0]


def handle(stream: Stream) -> int:
    """The CUstream of a stream."""
    if not isinstance(stream, CUDAStream):
        raise TypeError(f"{type(stream).__name__} is not a CUDA stream")
    return stream.__cuda_stream__()[1]


def unavailable(version: Callable[[], int]) -> str | None:
    """Why a library or a CUDA device cannot be used."""
    try:
        version()
        status, count = driver().cuDeviceGetCount()
        _check(status)
        if not count:
            return "no CUDA device is available"
    except (ImportError, OSError, RuntimeError) as error:
        return str(error)
    return None


def driver():
    """The CUDA driver API, initialized: cuInit returns at once after the first call."""
    if cu is None:
        raise ImportError("cuda-bindings and a CUDA driver are unavailable")
    _check(cu.cuInit(0)[0])
    return cu


def _check(status: int) -> None:
    if status:
        raise RuntimeError(f"CUDA driver returned {status}")


@contextmanager
def relaxed_capture() -> Generator[None]:
    """Let the calling thread make calls that a stream capture forbids (allocation, synchronization), then restore."""
    api = driver()
    status, mode = api.cuThreadExchangeStreamCaptureMode(api.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED)
    _check(status)
    try:
        yield
    finally:
        _check(api.cuThreadExchangeStreamCaptureMode(mode)[0])


def synchronize(stream: Stream) -> None:
    """Block the calling thread until the work queued on `stream` has run."""
    _check(driver().cuStreamSynchronize(handle(stream))[0])


def allocate(device: Device, nbytes: int) -> Vector[u8]:
    """Device memory that is ready for any stream: made on the default stream, then waited for. Stream-ordered
    allocation and the wait are not allowed on the capturing thread, so both run in a relaxed window, where they
    take no part in the capture."""
    with relaxed_capture(), device.default_stream:
        scratch = device.empty(u8, max(nbytes, 1))
        synchronize(device.default_stream)
    return scratch


class Scratch:
    """The workspace of one stream, grown to the largest request. A buffer that was outgrown stays alive, because
    work already queued, or recorded in a graph, may still hold its address."""

    def __init__(self, device: Device) -> None:
        self.device, self.retired = device, []
        self.buffer = allocate(device, 0)

    def at_least(self, nbytes: int) -> int:
        """The address of a buffer of at least `nbytes`."""
        if self.buffer.shape[0] < nbytes:
            self.retired.append(self.buffer)
            self.buffer = allocate(self.device, nbytes)
        return address(self.buffer)


class PerThread[T]:
    """One resource per CUDA context and thread, made on first use."""

    def __init__(self, make: Callable[[], T]) -> None:
        self._make, self._local = make, threading.local()

    def __call__(self) -> T:
        context = 0
        if cu is not None:
            status, current = driver().cuCtxGetCurrent()
            _check(status)
            context = int(current)
        try:
            resources = self._local.resources
        except AttributeError:
            resources = self._local.resources = {}
        try:
            return resources[context]
        except KeyError:
            resource = resources[context] = self._make()
            return resource

"""Frontend slice of krn.gpu: capture a kernel from its source, check it, lower it and run it.

Nothing registers the CPU backend: `host.Device.current()` finds a CPU target, imports `krn.runtime.runtime` and reads
its `BACKEND`, so `saxpy(a, x, y, launch=Items(n))` runs.
"""

from krn.runtime.cpu import CPUProgram
from krn.runtime.runtime import CPUBackend, CPUDevice, CPUEvent, CPUGraph, CPUStream, CPUTensor

from .artifacts import inspect_kernel
from .capture import capture
from .dtypes import Arithmetic
from .emit import Unit
from .errors import CaptureError
from .ir import Kernel
from .lower import build, emit

__all__ = [
    "Arithmetic",
    "CPUBackend",
    "CPUDevice",
    "CPUEvent",
    "CPUGraph",
    "CPUProgram",
    "CPUStream",
    "CPUTensor",
    "CaptureError",
    "Kernel",
    "Unit",
    "build",
    "capture",
    "emit",
    "inspect_kernel",
]

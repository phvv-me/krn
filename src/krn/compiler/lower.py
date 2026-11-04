"""From captured IR to a translation unit and, on the host CPU, to a compiled program."""

from krn.facts.targets import AMD, CPU, Apple, Nvidia, Target
from krn.runtime.cpu import CPUProgram, CPURenderer
from krn.runtime.settings import current_settings, retained

from .cuda import CUDARenderer
from .emit import Emitter, Unit
from .heuristics import decide
from .hip import HIPRenderer
from .ir import Kernel
from .msl import MSLRenderer


@retained
def emit(kernel: Kernel, target: Target, *, lineinfo: bool | None = None) -> Unit:
    """The translation unit of `kernel` for `target`: CUDA C++ for Nvidia, HIP C++ for AMD, MSL for Apple, C for a CPU.

    With `lineinfo` (default: `KRN_DEBUG_LINEINFO`) every statement is preceded by a `#line` directive naming the Python
    file and line, so debuggers and profilers report the kernel's own source.
    """
    snapshot = current_settings()
    decisions = decide(kernel, target, snapshot)
    info = snapshot.debug.lineinfo if lineinfo is None else lineinfo
    match target:
        case Nvidia():
            return Emitter(kernel, CUDARenderer(target, decisions), lineinfo=info).emit()
        case AMD():
            return Emitter(kernel, HIPRenderer(target, decisions), lineinfo=info).emit()
        case Apple():
            return Emitter(kernel, MSLRenderer(target, decisions), lineinfo=info).emit()
        case CPU():
            return Emitter(kernel, CPURenderer(target, decisions), lineinfo=info).emit()
    raise NotImplementedError(f"no renderer for target {target.name}")


@retained
def build(kernel: Kernel, target: CPU) -> CPUProgram:
    """Emit C for the host and compile it with clang."""
    return CPUProgram(kernel, emit(kernel, target))

"""NVRTC stand-ins that record the option list of each compile, so the flags are checked without NVRTC, and the kernels
the NVRTC tests compile.

`recording_nvidia` fakes the ctypes library behind `nvrtc.Nvidia`; `record_bindings` fakes `cuda.bindings.nvrtc`, which
the runtime's `compile_result` calls through its module.
"""

from types import ModuleType

import pytest

from krn.runtime.nvrtc import Nvidia as Nvrtc

SOURCE = 'extern "C" __global__ void kernel(float* out) { out[0] = 1.0f; }'
ADVERSARIAL = """
extern "C" __device__ float f(float a, float b, float c) { return a * b + c; }
extern "C" __global__ void kernel(float* out, float a, float b, float c) { out[0] = f(a, b, c); }
"""
CONFLICTS = (
    "--use_fast_math",
    "-use_fast_math",
    "--use-fast-math",
    "--fmad=true",
    "-fmad",
    "-fmad=true",
    "--fmad true",
    "--ftz=true",
    "-ftz",
    "--ftz true",
    "--prec-div=false",
    "-prec-div=false",
    "--prec-div false",
    "--prec-sqrt=false",
    "-prec_sqrt false",
)


class _CtypesLibrary:
    """The three NVRTC entry points of a compile through ctypes."""

    def __init__(self) -> None:
        self.received: list[tuple[str, ...]] = []

    def nvrtcCreateProgram(self, *_) -> int:
        return 0

    def nvrtcCompileProgram(self, _program, count, options) -> int:
        self.received.append(tuple(options[index].decode() for index in range(count)))
        return 0

    def nvrtcDestroyProgram(self, *_) -> None:
        return None


class _BindingsLibrary:
    """The entry points of `cuda.bindings.nvrtc` that `compile_result` calls, returning success tuples."""

    def __init__(self) -> None:
        self.received: list[tuple[str, ...]] = []

    def nvrtcCreateProgram(self, *_) -> tuple[int, int]:
        return 0, 1

    def nvrtcCompileProgram(self, _program, count, options) -> tuple[int]:
        self.received.append(tuple(option.decode() for option in options))
        return (0,)

    def nvrtcGetProgramLogSize(self, _program) -> tuple[int, int]:
        return 0, 0

    def nvrtcGetProgramLog(self, _program, _text) -> tuple[int]:
        return (0,)

    def nvrtcGetCUBINSize(self, _program) -> tuple[int, int]:
        return 0, 0

    def nvrtcGetCUBIN(self, _program, _cubin) -> tuple[int]:
        return (0,)

    def nvrtcDestroyProgram(self, *_) -> None:
        return None


def recording_nvidia(monkeypatch: pytest.MonkeyPatch) -> tuple[Nvrtc, list[tuple[str, ...]]]:
    """An Nvidia whose compiler records; no library is loaded. The list keeps the options of each compile."""
    library = _CtypesLibrary()
    nvidia = Nvrtc.__new__(Nvrtc)
    monkeypatch.setattr(nvidia, "nvrtc", library, raising=False)
    monkeypatch.setattr(nvidia, "_read", lambda *_: b"")
    monkeypatch.setattr(nvidia, "_read_log", lambda *_: "")
    return nvidia, library.received


def record_bindings(monkeypatch: pytest.MonkeyPatch, backend: ModuleType) -> list[tuple[str, ...]]:
    """Replaces the `nvrtc` of the CUDA backend module with a recorder and returns the list of its compiles."""
    library = _BindingsLibrary()
    monkeypatch.setattr(backend, "nvrtc", library)
    return library.received

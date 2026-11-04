"""NVRTC and nvJitLink through ctypes, so compile experiments need no GPU, no driver and no cuda-python wheel.

The libraries come from NVIDIA's `nvidia-cu13` wheel, which a torch install already carries. Pass its directory.
"""

import ctypes
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal

type Kind = Literal["ptx", "cubin", "ltoir"]


class NvJitLinkInput(IntEnum):
    """From nvJitLink.h."""

    CUBIN = 1
    PTX = 2
    LTOIR = 3


@dataclass(frozen=True)
class Compiled:
    data: bytes
    log: str
    seconds: float


class NvidiaError(RuntimeError):
    pass


class Nvidia:
    """The two compiler libraries of one CUDA 13 install."""

    def __init__(self, lib_dir: Path) -> None:
        self.lib_dir = lib_dir
        started = time.perf_counter()
        self.nvrtc = ctypes.CDLL(str(lib_dir / "libnvrtc.so.13"))
        self.nvjitlink = ctypes.CDLL(str(lib_dir / "libnvJitLink.so.13"))
        self.load_seconds = time.perf_counter() - started
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(self.nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)), "nvrtcVersion")
        self.nvrtc_version = (major.value, minor.value)

    def _check(self, status: int, what: str, log: str = "") -> None:
        if status:
            raise NvidiaError(f"{what} failed with status {status}\n{log}")

    def compile(self, source: str, arch: str, kind: Kind, options: tuple[str, ...] = ()) -> Compiled:
        """Compile CUDA C++ for `arch` (sm_90a) to PTX, a cubin or LTO-IR."""
        flags = [f"--gpu-architecture={arch}" if kind != "ltoir" else f"-arch={arch}", "-default-device", *options]
        if kind == "ltoir":
            flags.append("-dlto")
        prog = ctypes.c_void_p()
        started = time.perf_counter()
        self._check(
            self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"unit.cu", 0, None, None), "create"
        )
        try:
            encoded = (ctypes.c_char_p * len(flags))(*(f.encode() for f in flags))
            status = self.nvrtc.nvrtcCompileProgram(prog, len(flags), encoded)
            log = self._read_log(prog)
            self._check(status, "nvrtcCompileProgram", log)
            getter = {"ptx": "PTX", "cubin": "CUBIN", "ltoir": "LTOIR"}[kind]
            data = self._read(prog, f"nvrtcGet{getter}Size", f"nvrtcGet{getter}")
            return Compiled(data, log, time.perf_counter() - started)
        finally:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    def _read(self, prog: ctypes.c_void_p, size_fn: str, get_fn: str) -> bytes:
        size = ctypes.c_size_t()
        self._check(getattr(self.nvrtc, size_fn)(prog, ctypes.byref(size)), size_fn)
        buffer = ctypes.create_string_buffer(size.value)
        self._check(getattr(self.nvrtc, get_fn)(prog, buffer), get_fn)
        return buffer.raw

    def _read_log(self, prog: ctypes.c_void_p) -> str:
        return self._read(prog, "nvrtcGetProgramLogSize", "nvrtcGetProgramLog").rstrip(b"\0").decode(errors="replace")

    def link(self, arch: str, inputs: list[tuple[NvJitLinkInput, bytes]], *, lto: bool = True) -> Compiled:
        """Link LTO-IR, or relocatable cubins with `lto=False`, into one cubin with nvJitLink."""
        flags = [f"-arch={arch}", *(["-lto"] if lto else [])]
        encoded = (ctypes.c_char_p * len(flags))(*(f.encode() for f in flags))
        handle = ctypes.c_void_p()
        started = time.perf_counter()
        self._check(self.nvjitlink.nvJitLinkCreate(ctypes.byref(handle), len(flags), encoded), "nvJitLinkCreate")
        try:
            for index, (kind, data) in enumerate(inputs):
                status = self.nvjitlink.nvJitLinkAddData(handle, int(kind), data, len(data), f"in{index}".encode())
                self._check(status, "nvJitLinkAddData")
            status = self.nvjitlink.nvJitLinkComplete(handle)
            if status:
                size = ctypes.c_size_t()
                self.nvjitlink.nvJitLinkGetErrorLogSize(handle, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                self.nvjitlink.nvJitLinkGetErrorLog(handle, log)
                raise NvidiaError(
                    f"nvJitLinkComplete failed with status {status}\n{log.value.decode(errors='replace')}"
                )
            size = ctypes.c_size_t()
            self._check(self.nvjitlink.nvJitLinkGetLinkedCubinSize(handle, ctypes.byref(size)), "cubin size")
            cubin = ctypes.create_string_buffer(size.value)
            self._check(self.nvjitlink.nvJitLinkGetLinkedCubin(handle, cubin), "cubin")
            return Compiled(cubin.raw, "", time.perf_counter() - started)
        finally:
            self.nvjitlink.nvJitLinkDestroy(ctypes.byref(handle))

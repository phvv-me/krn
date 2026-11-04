import ctypes
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from krn.compiler.errors import TargetError
from krn.hw.libraries import Libraries, LibraryNotFound, LoadedLibrary, SonameLoader
from krn.profile import Activity
from krn.profile.providers.amd.tracer import ROCTxLoader, ROCTxTracer
from krn.profile.providers.apple import tracer as apple
from krn.profile.providers.cpu.tracer import CPUTracer
from krn.profile.providers.intel.tracer import IntelTracer
from krn.profile.providers.nvidia import bindings


def test_apple_intervals_missing_extra_and_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Signposter:
        Category = SimpleNamespace(PointsOfInterest="points")

        def __init__(self, subsystem: str, category: str) -> None:
            calls.extend((subsystem, category))

        def begin_interval(self, name: str):
            calls.append(name)
            return lambda _: calls.append("end")

    monkeypatch.setattr(apple, "Signposter", Signposter)
    monkeypatch.setattr(apple, "system_key", lambda: "Darwin")
    assert apple.SignpostTracer.is_available()
    apple.SignpostTracer().start("outer")()
    assert calls == ["org.krn.profile", "points", "outer", "end"]
    monkeypatch.setattr(apple, "Signposter", None)
    with pytest.raises(TargetError, match=r"krn\[perf\]"):
        apple.SignpostTracer().start("missing")
    monkeypatch.setitem(sys.modules, "os_signpost", None)
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        assert runpy.run_module(apple.__name__)["Signposter"] is None


def test_amd_loader_and_intel_refuse_unsupported_native_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    handle = SimpleNamespace(roctxRangeStartA=Mock(return_value=7), roctxRangeStop=Mock(side_effect=calls.append))
    monkeypatch.setattr(Libraries, "load", lambda _: SimpleNamespace(handle=handle))
    monkeypatch.setattr(Libraries, "available", lambda _: True)
    assert ROCTxLoader().candidates("roctx") == ("libroctx64.so",)
    assert ROCTxTracer.is_available() and IntelTracer.is_available() and CPUTracer.is_available()
    ROCTxTracer().start("outer")()
    assert calls == [7]
    for backend in (ROCTxTracer(), IntelTracer(), CPUTracer()):
        with pytest.raises(TargetError):
            backend.collect(Activity.MEMCPY)
    with pytest.raises(TargetError, match="tested tracing"):
        IntelTracer().start("range")
    loaded = LoadedLibrary(name="roctx", path=Path("/fake/libroctx64.so"), handle=ctypes.CDLL(None), source="fake")
    monkeypatch.setattr(SonameLoader, "load", lambda *_: loaded)
    assert ROCTxLoader().load("roctx") is loaded

    def missing(self, name: str) -> LoadedLibrary:
        raise LibraryNotFound(name, ("libroctx64.so",), ("fake",), "old hint")

    monkeypatch.setattr(SonameLoader, "load", missing)
    with pytest.raises(LibraryNotFound, match="ROCm"):
        ROCTxLoader().load("roctx")


def test_optional_binding_import_success_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    nvtx = ModuleType("nvtx")
    cupti = ModuleType("cupti")
    monkeypatch.setattr(cupti, "cupti", ModuleType("cupti.cupti"), raising=False)
    monkeypatch.setitem(sys.modules, "nvtx", nvtx)
    monkeypatch.setitem(sys.modules, "cupti", cupti)
    assert runpy.run_path(bindings.__file__)["has_nvtx"]
    assert runpy.run_path(bindings.__file__)["has_cupti"]
    cuda = ModuleType("cuda")
    cuda_bindings = ModuleType("cuda.bindings")
    monkeypatch.setattr(cuda_bindings, "driver", ModuleType("cuda.bindings.driver"), raising=False)
    monkeypatch.setattr(cuda_bindings, "runtime", ModuleType("cuda.bindings.runtime"), raising=False)
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", cuda_bindings)
    assert runpy.run_path(bindings.__file__)["has_cuda"]

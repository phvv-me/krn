import _ctypes
import ctypes
import runpy
import sys
from dataclasses import FrozenInstanceError, replace
from importlib.metadata import PackagePath
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn.hw import insights
from krn.hw import libraries as lib
from krn.hw.devices import nvidia
from krn.hw.devices.base import ReadingDevice
from krn.hw.enums import Vendor
from krn.hw.system import System


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    lib.Libraries.clear_cache()


def test_platforms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert lib.known_libraries() == {} and lib.search_directories() == () and lib.nvidia_filenames("nvml") == ()
    assert lib.search_directories(system="Darwin") == () and lib.nvidia_filenames("nvml", system="Darwin") == ()
    assert lib.nvidia_filenames("custom", system="Linux") == ("libcustom.so",)
    assert lib._nvidia_wheels("nvrtc") == ("nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu12")
    assert "metal" in lib.known_libraries(system="Darwin")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.setenv("HIP_PATH", str(tmp_path / "hip"))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "ld"))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    assert tmp_path / "lib" in lib.search_directories(system="Linux")
    monkeypatch.setattr(lib, "system_key", lambda: "Linux")
    assert lib.SonameLoader().candidates("amd_smi")[0] == "libamd_smi.so"
    assert lib._canonical("c") == "libsystem"
    monkeypatch.setattr(lib.NvidiaLoader, "names", frozenset({"nvJitLink"}))
    assert lib._canonical("nvjitlink") == "nvJitLink"


@pytest.mark.skipif(sys.platform != "darwin", reason="loads the macOS dyld shared cache")
def test_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib, "system_key", lambda: "Darwin")
    loader = lib.FrameworkLoader()
    path = Path("/usr/lib/libSystem.B.dylib")
    assert loader.find("libsystem") == path
    loaded = lib.Libraries.load("c")
    assert loaded.path == path and loaded.source == "framework"
    assert isinstance(loaded.handle, ctypes.CDLL)
    assert lib.Libraries.load("libsystem") is loaded
    with pytest.raises(FrozenInstanceError):
        setattr(loaded, "name".lower(), "changed")
    monkeypatch.setattr(lib, "dyld_find", Mock(side_effect=ValueError))
    assert loader.find("libsystem") is None
    monkeypatch.setattr(lib, "dyld_find", lambda *args, **kwargs: "/other/libSystem.B.dylib")
    assert loader.find("libsystem") is None


def test_soname(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # An extension module is a shared object on every platform, so an exact path loads the same way everywhere.
    path = _ctypes.__file__
    assert path and isinstance(lib.SonameLoader().load(path).handle, ctypes.CDLL)
    found = tmp_path / "libamd_smi.so"
    found.touch()
    monkeypatch.setattr(lib, "system_key", lambda: "Linux")
    monkeypatch.setattr(lib, "search_directories", lambda **kwargs: (tmp_path,))
    assert lib.Libraries.find("amd_smi") == found
    missing = str(tmp_path / "missing.so")
    assert lib.Libraries.find(missing) is None
    call = Mock(side_effect=OSError("broken"))
    monkeypatch.setattr(lib.ctypes, "CDLL", call)
    assert not lib.Libraries.available(missing)
    with pytest.raises(lib.LibraryNotFound) as caught:
        lib.Libraries.load(missing)
    assert caught.value.names == (missing,) and missing in str(caught.value)
    assert call.call_count == 1
    lib.Libraries.clear_cache()
    assert not lib.Libraries.available("amd_smi")
    with pytest.raises(lib.LibraryNotFound) as caught:
        lib.Libraries.load("amd_smi")
    assert all(place in str(caught.value) for place in ("libamd_smi.so", str(found), "wheel metadata: amdsmi"))
    assert lib.Libraries.find("unknown") is None
    assert not lib.Libraries.available("unknown")
    monkeypatch.setattr(lib.ctypes, "CDLL", Mock(return_value=SimpleNamespace()))
    assert lib.Libraries.load(str(found)).source == "soname"


def test_wheel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lib, "system_key", lambda: "Linux")
    file = PackagePath("lib/libamd_smi.so")
    file.dist = Mock(locate_file=lambda path: tmp_path / path)
    path = tmp_path / file
    path.parent.mkdir()
    path.touch()
    monkeypatch.setattr(lib.metadata, "files", lambda name: [PackagePath("unrelated.txt"), file])
    loader = lib.WheelLoader()
    assert loader.find("amd_smi") == path
    monkeypatch.setattr(lib.ctypes, "CDLL", Mock(return_value=SimpleNamespace()))
    assert loader.load("amd_smi").source == "wheel"

    def load(candidate: str) -> SimpleNamespace:
        if candidate != str(path):
            raise OSError("absent")
        return SimpleNamespace()

    monkeypatch.setattr(lib.ctypes, "CDLL", load)
    assert lib.Libraries.load("amd_smi").source == "wheel"
    assert lib.Libraries.find("amd_smi") == path
    monkeypatch.setattr(lib.metadata, "files", lambda name: None)
    assert loader.find("amd_smi") is None
    with pytest.raises(lib.LibraryNotFound, match="wheel metadata: amdsmi"):
        loader.load("amd_smi")
    monkeypatch.setattr(lib.metadata, "files", Mock(side_effect=lib.metadata.PackageNotFoundError))
    assert loader.find("amd_smi") is None


def test_nvidia(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Driver libraries (NVML, libcuda) are unknown to pathfinder and load by exact soname.
    monkeypatch.setattr(lib, "pathfinder", None)
    monkeypatch.setattr(lib.WheelLoader, "candidates", lambda self, name: ())  # hosts may ship NVIDIA wheels
    monkeypatch.setattr(lib.ctypes, "CDLL", Mock(side_effect=OSError("no driver")))
    assert not lib.Libraries.available("nvml")
    monkeypatch.setattr(lib, "system_key", lambda: "Linux")
    monkeypatch.setattr(lib, "search_directories", lambda **kwargs: (tmp_path,))
    path = tmp_path / "libnvidia-ml.so.1"
    path.touch()
    assert lib.NvidiaLoader().find("nvml") == path
    assert lib.NvidiaLoader().find("cuda") is None
    assert lib.NvidiaLoader().find("nvrtc") is None
    monkeypatch.setattr(lib.ctypes, "CDLL", Mock(return_value=SimpleNamespace()))
    lib.Libraries.clear_cache()
    assert lib.Libraries.load("nvml").source == "soname"
    assert lib.Libraries.available("nvml")
    load = Mock(return_value=SimpleNamespace(abs_path=str(path), was_already_loaded_from_elsewhere=True))
    fake = SimpleNamespace(
        load_nvidia_dynamic_lib=load,
        SUPPORTED_NVIDIA_LIBNAMES=("nvrtc", "new_library"),
        DynamicLibNotFoundError=LookupError,
        DynamicLibUnknownError=ValueError,
        DynamicLibNotAvailableError=RuntimeError,
    )
    monkeypatch.setattr(lib, "pathfinder", fake)
    lib.Libraries.clear_cache()
    assert lib.Libraries.load("nvrtc").source == "cuda.pathfinder"
    assert lib.Libraries.available("nvrtc")
    load.assert_called_once_with("nvrtc")
    for error in (LookupError, ValueError, RuntimeError):
        load.side_effect = error("toolkit absent")
        with pytest.raises(lib.LibraryNotFound, match="toolkit absent"):
            lib.NvidiaLoader().load("nvrtc")
    load.side_effect = None
    monkeypatch.setattr(lib.ctypes, "CDLL", Mock(side_effect=OSError("invalid binary")))
    with pytest.raises(lib.LibraryNotFound, match="invalid binary"):
        lib.NvidiaLoader().load("nvrtc")
    module = ModuleType("cuda")
    module.__dict__["pathfinder"] = fake
    monkeypatch.setitem(sys.modules, "cuda", module)
    monkeypatch.setattr(lib.LibraryLoader, "_registry", list(lib.LibraryLoader._registry))
    namespace = runpy.run_module("krn.hw.libraries")
    assert namespace["NvidiaLoader"].names == frozenset({"nvml", "cuda", *fake.SUPPORTED_NVIDIA_LIBNAMES})
    monkeypatch.setattr(lib.ctypes, "CDLL", Mock(return_value=SimpleNamespace()))
    assert namespace["Libraries"].load("new_library").source == "cuda.pathfinder"


@given(
    st.lists(st.text(alphabet="abcxyz", min_size=1, max_size=12), min_size=1, max_size=8, unique=True),
    st.booleans(),
)
def test_find_never_loads_and_failures_list_every_place(names: list[str], exists: bool) -> None:
    # A fresh local monkeypatch context keeps every generated example isolated.
    with pytest.MonkeyPatch.context() as patch, TemporaryDirectory() as directory:
        lib.Libraries.clear_cache()
        places = tuple(f"{directory}/{name}.so" for name in names)
        if exists:
            Path(places[0]).touch()
        patch.setattr(lib.SonameLoader, "candidates", lambda self, name: places)
        call = Mock(side_effect=OSError("absent"))
        patch.setattr(lib.ctypes, "CDLL", call)
        assert lib.Libraries.find("/missing/library.so") == (Path(places[0]) if exists else None)
        call.assert_not_called()
        with pytest.raises(lib.LibraryNotFound) as caught:
            lib.Libraries.load("/missing/library.so")
        assert caught.value.names == places
        assert all(place in str(caught.value) for place in places)
        assert not lib.Libraries.available("/missing/library.so")
        assert call.call_count == len(places)


def test_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib.LibraryLoader, "_registry", [])

    class CustomLoader(lib.LibraryLoader):
        names = frozenset({"custom"})
        source = "custom"

        def find(self, name: str) -> Path | None:
            return Path("/custom")

        def load(self, name: str) -> lib.LoadedLibrary:
            return lib.LoadedLibrary(name=name, path=Path("/custom"), handle=ctypes.CDLL(None), source=self.source)

    class AbstractLoader(lib.LibraryLoader):
        pass

    assert lib.LibraryLoader.implementations() == (CustomLoader,)
    assert lib.Libraries.find("custom") == Path("/custom")
    assert lib.Libraries.load("custom").source == "custom"
    assert lib.isabstract(AbstractLoader)


def test_missing_library_insights(monkeypatch: pytest.MonkeyPatch, inventory: System) -> None:
    gpu = ReadingDevice(vendor=Vendor.NVIDIA, name="GPU", index=0, source="wgpu")
    system = replace(inventory, devices=(gpu,))
    missing = lib.LibraryNotFound("nvml", ("nvml",), ("/driver/libnvidia-ml.so.1",), "Install krn[cuda] and the driver")
    monkeypatch.setattr(lib.Libraries, "load", Mock(side_effect=missing))
    assert "/driver/libnvidia-ml.so.1" in insights.missing_sources(system)[0].message
    monkeypatch.setattr(nvidia, "core", SimpleNamespace())
    monkeypatch.setattr(nvidia, "system", SimpleNamespace())
    monkeypatch.setattr(lib.Libraries, "available", lambda name: False)
    assert "nvml" in (nvidia.NvidiaProbe().available().reason or "")
    monkeypatch.setattr(lib.Libraries, "load", lambda name: SimpleNamespace())
    assert "binding" in insights.missing_sources(system)[0].message
    assert not insights.missing_sources(replace(system, devices=(replace(gpu, vendor=Vendor.UNKNOWN),)))
    assert not insights.missing_sources(replace(system, devices=(replace(gpu, source="cuda"),)))


def test_native_loads_are_centralized() -> None:
    for path in Path("src").rglob("*.py"):
        if path != Path("src/krn/hw/libraries.py"):
            assert "ctypes.CDLL" not in path.read_text()
        assert "find_library" not in path.read_text()

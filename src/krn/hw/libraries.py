"""Cached native library discovery and loading without detection commands."""

import ctypes
import os
from abc import ABC, abstractmethod
from ctypes.macholib.dyld import dyld_find
from dataclasses import dataclass, replace
from functools import cache
from importlib import metadata
from importlib.util import find_spec
from inspect import isabstract
from pathlib import Path
from typing import ClassVar

try:
    from cuda import pathfinder
except ImportError:
    pathfinder = None

from patos import value_dispatch

from .exceptions import HardwareError
from .platforms import otherwise, system_key

type LibraryHandle = ctypes.CDLL


@dataclass(frozen=True, kw_only=True)
class LoadedLibrary:
    name: str
    path: Path
    handle: ctypes.CDLL
    source: str


class LibraryNotFound(HardwareError):
    """All attempted names and search locations, including loader errors."""

    def __init__(self, name: str, names: tuple[str, ...], places: tuple[str, ...], fix: str) -> None:
        self.name, self.names, self.places = name, names, places
        super().__init__(f"Library {name!r} unavailable; names: {', '.join(names)}; tried: {'; '.join(places)}", fix)


@dataclass(frozen=True, kw_only=True)
class LibrarySpec:
    loader: str
    filenames: tuple[str, ...]
    distributions: tuple[str, ...] = ()
    fix: str = "Install the system runtime or its Python runtime wheel"


@value_dispatch(kind="system")
def known_libraries() -> dict[str, LibrarySpec]:
    return {}


@known_libraries.register("Darwin")
def _mac_libraries() -> dict[str, LibrarySpec]:
    return {
        name.lower(): LibrarySpec(
            loader="framework",
            filenames=(f"/System/Library/Frameworks/{name}.framework/{name}",),
            fix="Restore the macOS system frameworks",
        )
        for name in ("IOKit", "CoreFoundation", "Metal", "Accelerate")
    } | {
        "libsystem": LibrarySpec(loader="framework", filenames=("/usr/lib/libSystem.B.dylib",)),
        "ioreport": LibrarySpec(loader="framework", filenames=("/usr/lib/libIOReport.dylib",)),
    }


@known_libraries.register("Linux")
def _linux_libraries() -> dict[str, LibrarySpec]:
    return {
        "amd_smi": LibrarySpec(
            loader="soname",
            filenames=("libamd_smi.so",),
            distributions=("amdsmi", "rocm-sdk-core"),
            fix='Install ROCm and pip install "krn[hip]"',
        ),
        "hip": LibrarySpec(
            loader="soname",
            filenames=("libamdhip64.so",),
            distributions=("rocm-sdk-core",),
            fix='Install ROCm and pip install "krn[hip]"',
        ),
        "level_zero": LibrarySpec(
            loader="soname",
            filenames=("libze_loader.so.1", "libze_loader.so"),
            distributions=("intel-sycl-rt",),
            fix="Install the Level Zero loader (level-zero, intel-level-zero-gpu or libze1), "
            'then pip install "krn[intel]"',
        ),
        "drm_amdgpu": LibrarySpec(loader="soname", filenames=("libdrm_amdgpu.so.1",), fix="Install libdrm"),
        "opencl": LibrarySpec(
            loader="soname",
            filenames=("libOpenCL.so.1", "libOpenCL.so"),
            distributions=("intel-opencl-rt",),
            fix="Install an OpenCL ICD loader and the Intel OpenCL driver (intel-compute-runtime), "
            'then pip install "krn[intel]"',
        ),
        "libsystem": LibrarySpec(loader="soname", filenames=("libc.so.6",)),
    } | _nvidia_specs("Linux")


@value_dispatch(kind="system")
def nvidia_filenames(name: str) -> tuple[str, ...]:
    return ()


@nvidia_filenames.register("Linux")
def _linux_nvidia_filenames(name: str) -> tuple[str, ...]:
    return {
        "nvml": ("libnvidia-ml.so.1",),
        "cuda": ("libcuda.so.1",),
        "cublas": ("libcublas.so.13", "libcublas.so.12", "libcublas.so"),
        "cusparse": ("libcusparse.so.12", "libcusparse.so"),
        "nvrtc": ("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"),
        "nvjitlink": ("libnvJitLink.so.13", "libnvJitLink.so.12", "libnvJitLink.so"),
    }.get(name.lower(), (f"lib{name}.so",))


otherwise(nvidia_filenames, "Darwin")


def _nvidia_specs(system: str) -> dict[str, LibrarySpec]:
    return {
        name: LibrarySpec(
            loader="cuda.pathfinder",
            filenames=nvidia_filenames(name, system=system),
            distributions=_nvidia_wheels(name),
            fix='Install the NVIDIA driver and pip install "krn[cuda]"',
        )
        for name in NvidiaLoader.names
    }


def _nvidia_wheels(name: str) -> tuple[str, ...]:
    return {
        "nvml": (),
        "cuda": (),
        "nvrtc": ("nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu12"),
        "cublaslt": ("nvidia-cublas", "nvidia-cublas-cu12"),
    }.get(name.lower(), (f"nvidia-{name.lower()}", f"nvidia-{name.lower()}-cu12"))


def _spec(name: str) -> LibrarySpec:
    return known_libraries(system=system_key()).get(name, LibrarySpec(loader="cuda.pathfinder", filenames=()))


def _canonical(name: str) -> str:
    if Path(name).is_absolute():
        return name
    aliases = {"c": "libsystem", "amdhip64": "hip", "ze_loader": "level_zero"}
    normalized = aliases.get(name.lower(), name.lower())
    return next((key for key in NvidiaLoader.names if key.lower() == normalized), normalized)


@value_dispatch(kind="system")
def search_directories() -> tuple[Path, ...]:
    return ()


@search_directories.register("Linux")
def _linux_directories() -> tuple[Path, ...]:
    roots = tuple(Path(os.environ[key]) for key in ("ROCM_PATH", "HIP_PATH", "CUDA_PATH") if os.environ.get(key))
    return (
        *(Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p),
        *(root / part for root in (*roots, *sorted(Path("/opt").glob("rocm*"))) for part in ("lib", "lib64")),
        Path("/usr/local/cuda/lib64"),
        Path("/usr/lib/wsl/lib"),
        Path("/opt/intel/oneapi/compiler/latest/lib"),
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/lib"),
        Path("/lib64"),
        *(Path("/usr/lib") / arch for arch in ("x86_64-linux-gnu", "aarch64-linux-gnu")),
    )


otherwise(search_directories, "Darwin")


class LibraryLoader(ABC):
    """Self-registering strategies; subclasses declare the neutral names they serve."""

    names: ClassVar[frozenset[str]] = frozenset()
    source: ClassVar[str]
    _registry: ClassVar[list[type[LibraryLoader]]] = []

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        LibraryLoader._registry.append(cls)

    @classmethod
    def implementations(cls) -> tuple[type[LibraryLoader], ...]:
        return tuple(impl for impl in cls._registry if issubclass(impl, cls) and not isabstract(impl))

    @abstractmethod
    def find(self, name: str) -> Path | None:
        """Locate without executing native code."""

    @abstractmethod
    def load(self, name: str) -> LoadedLibrary:
        """Load or raise LibraryNotFound with every attempted location."""


class SonameLoader(LibraryLoader):
    names = frozenset({"amd_smi", "hip", "level_zero", "drm_amdgpu", "opencl", "libsystem"})
    source = "soname"

    def candidates(self, name: str) -> tuple[str, ...]:
        if Path(name).is_absolute():
            return (name,)
        filenames = _spec(name).filenames
        return (
            *filenames,
            *(str(root / file) for root in search_directories(system=system_key()) for file in filenames),
        )

    def find(self, name: str) -> Path | None:
        return next((Path(p) for p in self.candidates(name) if Path(p).is_absolute() and Path(p).is_file()), None)

    def load(self, name: str) -> LoadedLibrary:
        tried = []
        candidates = self.candidates(name)
        for candidate in candidates:
            try:
                handle = ctypes.CDLL(candidate)
            except OSError as error:
                tried.append(f"{candidate}: {error}")
                continue
            return LoadedLibrary(name=name, path=Path(candidate), handle=handle, source=self.source)
        raise LibraryNotFound(name, candidates, tuple(tried), _spec(name).fix)


class FrameworkLoader(SonameLoader):
    names = frozenset({"iokit", "corefoundation", "metal", "libsystem", "ioreport", "accelerate"})
    source = "framework"

    def candidates(self, name: str) -> tuple[str, ...]:
        return _spec(name).filenames

    def find(self, name: str) -> Path | None:
        # dyld also locates entries in the shared cache with no file on disk.
        for candidate in self.candidates(name):
            try:
                found = dyld_find(candidate, env={})
            except ValueError:
                continue
            if found == candidate:
                return Path(found)
        return None


class WheelLoader(SonameLoader):
    names = SonameLoader.names
    source = "wheel"

    def candidates(self, name: str) -> tuple[str, ...]:
        paths = []
        spec = _spec(name)
        for distribution in spec.distributions:
            try:
                files = metadata.files(distribution)
            except metadata.PackageNotFoundError:
                continue
            for file in files or ():
                if file.name in spec.filenames:
                    paths.append(str(file.locate()))
        return tuple(paths)

    def load(self, name: str) -> LoadedLibrary:
        try:
            return super().load(name)
        except LibraryNotFound as error:
            locations = tuple(f"wheel metadata: {distribution}" for distribution in _spec(name).distributions)
            raise LibraryNotFound(name, error.names, (*locations, *error.places), error.fix) from error


class NvidiaLoader(SonameLoader):
    # Pathfinder knows the CUDA Toolkit libraries only; the driver ones (NVML, libcuda) load by exact soname.
    names = frozenset({"nvml", "cuda"}) | (
        frozenset(pathfinder.SUPPORTED_NVIDIA_LIBNAMES)
        if pathfinder is not None
        else frozenset({"cublas", "cusparse", "nvrtc", "nvjitlink"})
    )
    source = "cuda.pathfinder"

    def candidates(self, name: str) -> tuple[str, ...]:
        filenames = nvidia_filenames(name, system=system_key())
        return (
            *filenames,
            *(str(root / file) for root in search_directories(system=system_key()) for file in filenames),
            *WheelLoader().candidates(name),
        )

    def load(self, name: str) -> LoadedLibrary:
        if pathfinder is None or name not in pathfinder.SUPPORTED_NVIDIA_LIBNAMES:
            return replace(super().load(name), source="soname")
        location = f"cuda.pathfinder.load_nvidia_dynamic_lib({name!r})"
        fix = 'Install the NVIDIA driver and pip install "krn[cuda]"'
        try:
            loaded = pathfinder.load_nvidia_dynamic_lib(name)
        except (
            pathfinder.DynamicLibNotFoundError,
            pathfinder.DynamicLibUnknownError,
            pathfinder.DynamicLibNotAvailableError,
        ) as error:
            raise LibraryNotFound(name, (name,), (location, str(error)), fix) from error
        try:
            handle = ctypes.CDLL(loaded.abs_path)
        except OSError as error:
            raise LibraryNotFound(name, (name,), (location, str(loaded.abs_path), str(error)), fix) from error
        return LoadedLibrary(name=name, path=Path(loaded.abs_path), handle=handle, source=self.source)


class Libraries:
    """Process-wide flyweights; failures persist until clear_cache is called."""

    @staticmethod
    def _strategies(name: str) -> tuple[LibraryLoader, ...]:
        if Path(name).is_absolute():
            return (SonameLoader(),)
        return tuple(
            impl()
            for impl in LibraryLoader.implementations()
            if name in impl.names
            and (name not in known_libraries(system=system_key()) or impl.source in {_spec(name).loader, "wheel"})
        )

    @staticmethod
    @cache
    def _result(name: str) -> LoadedLibrary | LibraryNotFound:
        errors = []
        for loader in Libraries._strategies(name):
            try:
                return loader.load(name)
            except LibraryNotFound as error:
                errors.append(error)
        return LibraryNotFound(
            name,
            tuple(n for e in errors for n in e.names) or (name,),
            tuple(p for e in errors for p in e.places) or (f"registry: {system_key()}",),
            errors[0].fix if errors else "Register a LibraryLoader for this name and platform",
        )

    @staticmethod
    def load(name: str) -> LoadedLibrary:
        result = Libraries._result(_canonical(name))
        if isinstance(result, LibraryNotFound):
            raise result
        return result

    @staticmethod
    @cache
    def _find(name: str) -> Path | None:
        return next((path for loader in Libraries._strategies(name) if (path := loader.find(name)) is not None), None)

    @staticmethod
    def find(name: str) -> Path | None:
        return Libraries._find(_canonical(name))

    @staticmethod
    def available(name: str) -> bool:
        return isinstance(Libraries._result(_canonical(name)), LoadedLibrary)

    @staticmethod
    def clear_cache() -> None:
        Libraries._result.cache_clear()
        Libraries._find.cache_clear()


def importable(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except ImportError, ValueError:
        return False


def available_libraries(system: str) -> frozenset[str]:
    found = {name for name in ("cublas", "cusparse") if Libraries.available(name)}
    if importable("MetalPerformanceShaders"):
        found.add("MetalPerformanceShaders")
    if "accelerate" in known_libraries(system=system) and Libraries.available("accelerate"):
        found.add("Accelerate")
    return frozenset(found)

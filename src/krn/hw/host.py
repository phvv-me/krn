"""Host inventory from Python libraries, with explicit platform dispatch."""

import os
import platform
import sys
import sysconfig
from dataclasses import dataclass

try:
    import psutil
except ImportError:
    psutil = None

try:
    import archspec.cpu as archspec
except ImportError:
    archspec = None

from patos import value_dispatch

from . import apple_native, native
from .devices.apple import AppleProbe
from .enums import Vendor
from .platforms import otherwise, system_key
from .topology import detect_topology


@value_dispatch(kind="system")
def os_description() -> tuple[str, str]:
    raise NotImplementedError("pass system=: Darwin and Linux are described")


@os_description.register("Linux")
def _linux_description() -> tuple[str, str]:
    try:
        info = platform.freedesktop_os_release()
    except OSError:
        info = {}
    return info.get("NAME", "Linux"), info.get("VERSION_ID", platform.release())


@os_description.register("Darwin")
def _mac_description() -> tuple[str, str]:
    return "macOS", platform.mac_ver()[0]


@dataclass(frozen=True, kw_only=True)
class OS:
    system: str
    release: str
    distribution: str
    distribution_version: str
    architecture: str
    rosetta: bool = False

    @classmethod
    def detect(cls) -> OS:
        distribution, version = os_description(system=system_key())
        return cls(
            system=system_key(),
            release=platform.release(),
            distribution=distribution,
            distribution_version=version,
            architecture=platform.machine(),
        )


@value_dispatch(kind="system")
def cpu_identity() -> tuple[str, Vendor, frozenset[str], str | None, int | None]:
    return platform.processor() or platform.machine(), Vendor.UNKNOWN, frozenset(), None, None


@cpu_identity.register("Linux")
def _linux_cpu() -> tuple[str, Vendor, frozenset[str], str | None, int | None]:
    if archspec is None:
        return cpu_identity()
    try:
        cpu = archspec.host()
        vendor = {
            "GenuineIntel": Vendor.INTEL,
            "AuthenticAMD": Vendor.AMD,
            "Intel": Vendor.INTEL,
            "AMD": Vendor.AMD,
        }.get(cpu.vendor, Vendor.UNKNOWN)
        return cpu.name, vendor, frozenset(cpu.features), cpu.name, cpu.generation
    except OSError, ValueError, RuntimeError:
        return cpu_identity()


@cpu_identity.register("Darwin")
def _mac_cpu() -> tuple[str, Vendor, frozenset[str], str | None, int | None]:
    try:
        device = next(iter(AppleProbe().discover()))
        return device.name, device.vendor, frozenset(), None, None
    except StopIteration, RuntimeError, OSError, AttributeError:
        return cpu_identity()


@value_dispatch(kind="system")
def core_split() -> tuple[int | None, int | None]:
    return None, None


@core_split.register("Darwin")
def _mac_split() -> tuple[int | None, int | None]:
    return apple_native.performance_cores()


otherwise(core_split, "Linux")


@value_dispatch(kind="system")
def frequency() -> float | None:
    return None


@frequency.register("Linux")
def _cpu_frequency() -> float | None:
    if psutil is None:
        return None
    try:
        reading = psutil.cpu_freq()
        return reading.max if reading is not None else None
    except OSError, NotImplementedError:
        return None


otherwise(frequency, "Darwin")


@value_dispatch(kind="system")
def caches() -> tuple[tuple[str, int], ...]:
    return ()


@caches.register("Linux")
def _linux_caches() -> tuple[tuple[str, int], ...]:
    topology = detect_topology(system="Linux")
    return topology.caches if topology.unavailable_reason is None else native.cpu_caches()


otherwise(caches, "Darwin")


@dataclass(frozen=True, kw_only=True)
class CPU:
    name: str
    vendor: Vendor = Vendor.UNKNOWN
    architecture: str
    logical_cores: int
    physical_cores: int | None = None
    performance_cores: int | None = None
    efficiency_cores: int | None = None
    features: frozenset[str] = frozenset()
    caches: tuple[tuple[str, int], ...] = ()
    max_mhz: float | None = None
    microarchitecture: str | None = None
    generation: int | None = None

    @classmethod
    def detect(cls) -> CPU:
        name, vendor, features, microarchitecture, generation = cpu_identity(system=system_key())
        performance, efficiency = core_split(system=system_key())
        return cls(
            name=name,
            vendor=vendor,
            architecture=platform.machine(),
            logical_cores=(psutil.cpu_count(logical=True) if psutil else os.cpu_count()) or 1,
            physical_cores=psutil.cpu_count(logical=False) if psutil else None,
            performance_cores=performance,
            efficiency_cores=efficiency,
            features=features,
            caches=caches(system=system_key()),
            max_mhz=frequency(system=system_key()),
            microarchitecture=microarchitecture,
            generation=generation,
        )


@value_dispatch(kind="system")
def cgroup_limits() -> tuple[float | None, int | None]:
    return None, None


@cgroup_limits.register("Linux")
def _linux_limits() -> tuple[float | None, int | None]:
    return native.cgroup_limits()


otherwise(cgroup_limits, "Darwin")


def cpu_affinity() -> frozenset[int] | None:
    if psutil is None:
        return None
    try:
        return frozenset(psutil.Process().cpu_affinity())
    except AttributeError, OSError, psutil.Error, NotImplementedError:
        return None


@dataclass(frozen=True, kw_only=True)
class Limits:
    cpu_affinity: frozenset[int] | None = None
    usable_cpus: float
    cpu_quota: float | None = None
    memory_bytes: int | None = None
    masks: tuple[tuple[str, str], ...] = ()

    @classmethod
    def detect(cls) -> Limits:
        affinity = cpu_affinity()
        quota, memory = cgroup_limits(system=system_key())
        count = float(len(affinity) if affinity is not None else os.process_cpu_count() or 1)
        return cls(
            cpu_affinity=affinity,
            usable_cpus=min(count, quota) if quota is not None else count,
            cpu_quota=quota,
            memory_bytes=memory,
            masks=tuple(
                (key, os.environ[key])
                for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK")
                if key in os.environ
            ),
        )


@dataclass(frozen=True, kw_only=True)
class Memory:
    total: int | None
    page_size: int
    available: int | None = None
    swap_total: int | None = None
    swap_used: int | None = None
    cgroup_limit: int | None = None

    @classmethod
    def detect(cls, limits: Limits) -> Memory:
        memory = psutil.virtual_memory() if psutil else None
        try:
            swap = psutil.swap_memory() if psutil else None
        except OSError, NotImplementedError:
            swap = None
        total = memory.total if memory else None
        limit = limits.memory_bytes
        return cls(
            total=total,
            available=memory.available if memory else None,
            page_size=os.sysconf("SC_PAGE_SIZE"),
            swap_total=swap.total if swap else None,
            swap_used=swap.used if swap else None,
            cgroup_limit=limit if limit is not None and total is not None and limit < total else None,
        )


@dataclass(frozen=True, kw_only=True)
class Volume:
    path: str
    total: int
    free: int
    used_fraction: float

    @classmethod
    def detect(cls, path: str = "/") -> Volume:
        if psutil is None:
            raise OSError("psutil unavailable")
        usage = psutil.disk_usage(path)
        return cls(
            path=path, total=usage.total, free=usage.free, used_fraction=usage.used / usage.total if usage.total else 0
        )

    @classmethod
    def detect_all(cls) -> tuple[Volume, ...]:
        if psutil is None:
            return ()
        result = []
        for path in dict.fromkeys(["/", *(p.mountpoint for p in psutil.disk_partitions())]):
            try:
                result.append(cls.detect(path))
            except OSError:
                continue
        return tuple(result)


@dataclass(frozen=True, kw_only=True)
class PythonRuntime:
    version: str
    implementation: str
    free_threaded: bool
    gil_enabled: bool
    executable: str

    @classmethod
    def detect(cls) -> PythonRuntime:
        return cls(
            version=platform.python_version(),
            implementation=platform.python_implementation(),
            free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            gil_enabled=sys._is_gil_enabled(),
            executable=sys.executable,
        )

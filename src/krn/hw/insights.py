"""Small actionable rules over an inventory and fresh readings."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .devices.base import Device, Telemetry
from .enums import DeviceKind, Severity, Vendor
from .libraries import Libraries, LibraryNotFound

if TYPE_CHECKING:
    from .system import System

_BINDINGS = {
    Vendor.NVIDIA: ("nvml",),
    Vendor.AMD: ("amd_smi",),
    Vendor.APPLE: ("iokit",),
    Vendor.INTEL: ("level_zero", "opencl"),
}
_EXTRAS = {Vendor.NVIDIA: "cuda", Vendor.AMD: "hip", Vendor.APPLE: "metal", Vendor.INTEL: "intel"}
_NATIVE_SOURCES = {"drm", "sysctl", "wgpu"}
# Throttling needs a thermal or power-brake reason, a power cap under load, or a temperature near the slowdown.
# HW_SLOWDOWN is the umbrella bit that some boards set at idle, so it counts only through the bits it covers.
_THERMAL_REASONS = frozenset({"HW_THERMAL_SLOWDOWN", "SW_THERMAL_SLOWDOWN", "HW_POWER_BRAKE_SLOWDOWN"})
_BUSY_PERCENT = 50
_NEAR_SLOWDOWN_C = 5


@dataclass(frozen=True, kw_only=True)
class Insight:
    code: str
    severity: Severity
    subject: str
    message: str
    fix: str


def _gpus(system: System) -> tuple[Device, ...]:
    """GPUs this process can use; a device hidden by a runtime mask raises no warning."""
    return tuple(d for d in system.visible_devices if d.kind == DeviceKind.GPU)


def _missing(names: tuple[str, ...]) -> tuple[LibraryNotFound, ...]:
    errors = []
    for name in names:
        try:
            Libraries.load(name)
        except LibraryNotFound as error:
            errors.append(error)
    return tuple(errors)


def missing_sources(system: System) -> tuple[Insight, ...]:
    """One warning per GPU naming every runtime library that is missing, or the Python binding it lacks."""
    result = []
    for device in _gpus(system):
        if device.vendor not in _BINDINGS:
            continue
        missing = _missing(_BINDINGS[device.vendor])
        if missing:
            message = "; ".join(str(error) for error in missing)
            fix = "; ".join(dict.fromkeys(error.fix for error in missing))
        elif device.source in _NATIVE_SOURCES:
            message = "GPU detected without its compute binding"
            fix = f'pip install "krn[{_EXTRAS[device.vendor]}]"'
        else:
            continue
        result.append(
            Insight(code="missing-source", severity=Severity.WARNING, subject=device.name, message=message, fix=fix)
        )
    return tuple(result)


def _throttling(live: Telemetry) -> tuple[str, ...]:
    """The conditions that show throttling now; each names the reading that triggered it."""
    reasons = set(live.throttle_reasons or ())
    conditions = [f"{reason} active" for reason in sorted(reasons & _THERMAL_REASONS)]
    utilization = live.utilization_percent or 0
    if "SW_POWER_CAP" in reasons and utilization >= _BUSY_PERCENT:
        conditions.append(f"SW_POWER_CAP active at {utilization:g}% utilization")
    temperature, slowdown = live.temperature_c, live.slowdown_temperature_c
    if temperature is not None and slowdown is not None and temperature >= slowdown - _NEAR_SLOWDOWN_C:
        conditions.append(
            f"{temperature:g} C is within {_NEAR_SLOWDOWN_C} C of the {slowdown:g} C slowdown temperature"
        )
    return tuple(conditions)


def device_health(system: System) -> tuple[Insight, ...]:
    result = []
    for device in _gpus(system):
        live = device.telemetry()
        conditions = _throttling(live)
        if conditions:
            result.append(
                Insight(
                    code="gpu-throttling",
                    severity=Severity.WARNING,
                    subject=device.name,
                    message="GPU throttling: " + "; ".join(conditions),
                    fix="Reduce load and check cooling and power limits",
                )
            )
        if (
            device.memory.total
            and live.memory_used_bytes is not None
            and live.memory_used_bytes > 0.9 * device.memory.total
        ):
            result.append(
                Insight(
                    code="gpu-memory",
                    severity=Severity.WARNING,
                    subject=device.name,
                    message="GPU memory is over 90% used",
                    fix="Release unused device allocations",
                )
            )
    return tuple(result)


def process_limits(system: System) -> tuple[Insight, ...]:
    limited = system.limits.usable_cpus < system.cpu.logical_cores or system.memory.cgroup_limit is not None
    return (
        (
            Insight(
                code="process-limits",
                severity=Severity.INFO,
                subject="process",
                message="CPU affinity or cgroup limits reduce available resources",
                fix="Size workloads to process limits or request a larger allocation",
            ),
        )
        if limited
        else ()
    )


def visibility(system: System) -> tuple[Insight, ...]:
    return (
        (
            Insight(
                code="hidden-devices",
                severity=Severity.INFO,
                subject="process",
                message="Device visibility masks hide detected devices",
                fix="Review CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES and ZE_AFFINITY_MASK",
            ),
        )
        if len(system.visible_devices) < len(system.devices)
        else ()
    )


def python_build(system: System) -> tuple[Insight, ...]:
    return (
        Insight(
            code="python-build",
            severity=Severity.INFO,
            subject="Python",
            message=f"Free-threaded build: {system.python.free_threaded}; GIL enabled: {system.python.gil_enabled}",
            fix="Use a free-threaded build with the GIL disabled for parallel Python workloads",
        ),
    )


def rosetta(system: System) -> tuple[Insight, ...]:
    return (
        (
            Insight(
                code="rosetta",
                severity=Severity.WARNING,
                subject="Python",
                message="Python runs under Rosetta",
                fix="Use an arm64 Python environment",
            ),
        )
        if system.os.rosetta
        else ()
    )


def disk_space(system: System) -> tuple[Insight, ...]:
    return tuple(
        Insight(
            code="disk-space",
            severity=Severity.WARNING,
            subject=v.path,
            message="Less than 10% disk space is free",
            fix="Remove unused caches or choose another volume",
        )
        for v in system.volumes
        if v.free < 0.1 * v.total
    )


def software_renderers(system: System) -> tuple[Insight, ...]:
    return tuple(
        Insight(
            code="software-renderer",
            severity=Severity.INFO,
            subject=d.name,
            message="Software renderer on the CPU; it is not a GPU and not a compute target",
            fix="Select a hardware GPU adapter for compute",
        )
        for d in system.devices
        if d.kind == DeviceKind.SOFTWARE
    )


def unsupported_devices(system: System) -> tuple[Insight, ...]:
    return tuple(
        Insight(
            code="unsupported-target",
            severity=Severity.WARNING,
            subject=d.name,
            message=f"No compiler target for {d.architecture or 'this device'}",
            fix="Select a supported GPU or use a supported CPU backend",
        )
        for d in _gpus(system)
        if d.target is None
    )


RULES: tuple[Callable[[System], tuple[Insight, ...]], ...] = (
    missing_sources,
    device_health,
    process_limits,
    visibility,
    python_build,
    rosetta,
    disk_space,
    software_renderers,
    unsupported_devices,
)

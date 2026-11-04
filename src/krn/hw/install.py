"""Hardware-gated optional dependency recommendations."""

from dataclasses import dataclass
from typing import Literal

from patos import value_dispatch

from .enums import DeviceKind, Vendor
from .system import System

type Extra = Literal["cuda", "cuda-libs", "metal", "hip", "refs", "cuda-dx", "cccl", "intel", "probe"]
EXTRAS: tuple[Extra, ...] = ("cuda", "cuda-libs", "metal", "hip", "refs", "cuda-dx", "cccl", "intel", "probe")
OPTIONAL_EXTRAS: tuple[Extra, ...] = ("refs", "cuda-dx", "cccl")


@dataclass(frozen=True, kw_only=True)
class InstallPlan:
    extras: tuple[Extra, ...]
    optional_extras: tuple[Extra, ...]
    must_not_install: tuple[Extra, ...]
    command: str
    notes: tuple[str, ...] = ()


@value_dispatch(kind="system")
def _selected(machine: System) -> tuple[Extra, ...]:
    raise NotImplementedError("pass system=: Darwin and Linux select extras")


@_selected.register("Linux")
def _compute_extras(system: System) -> tuple[Extra, ...]:
    selected: list[Extra] = []
    vendors = system.by_vendor
    if vendors[Vendor.NVIDIA]:
        selected.extend(("cuda", "cuda-libs"))
    if system.os.architecture == "x86_64" and any(device.kind is DeviceKind.GPU for device in vendors[Vendor.AMD]):
        selected.append("hip")
    if any(device.kind in {DeviceKind.GPU, DeviceKind.NPU} for device in vendors[Vendor.INTEL]):
        selected.append("intel")
    return tuple(selected)


@_selected.register("Darwin")
def _mac_extras(system: System) -> tuple[Extra, ...]:
    return ("metal",) if system.os.architecture == "arm64" and system.by_vendor[Vendor.APPLE] else ()


def install_plan(system: System) -> InstallPlan:
    selected = _selected(system, system=system.os.system)
    optional: tuple[Extra, ...] = ("probe", *OPTIONAL_EXTRAS) if "cuda" in selected else ("probe",)
    requirement = f'"krn[{",".join(selected)}]"' if selected else '"krn"'
    return InstallPlan(
        extras=tuple(selected),
        optional_extras=optional,
        must_not_install=tuple(e for e in EXTRAS if e not in selected and e not in optional),
        command=f"pip install {requirement}",
        notes=("HIP wheels require x86_64; CUDA supports aarch64.",) if system.os.architecture == "aarch64" else (),
    )

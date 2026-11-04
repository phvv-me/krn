"""What a native cell needs from the machine, from the vendor documentation, with the page each fact came from.

A fact is written only where a documented page states it. A target the documents do not cover has no `Requires`, and its
cell says nothing about the machine it needs.
"""

from dataclasses import dataclass

from krn.facts.targets import AMD, CPU, Apple, Nvidia, Target

PTX_NOTES = "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes"
TOOLKIT_DRIVERS = "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html"
METAL_FEATURES = "https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf"
ROCM_MATRIX = "https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html"
INTRINSICS = "https://software.intel.com/sites/landingpage/IntrinsicsGuide/"
AMX = "https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-amx.html"
NVIDIA_SOURCE = f"{PTX_NOTES} {TOOLKIT_DRIVERS}"


@dataclass(frozen=True, kw_only=True)
class Requires:
    """The machine a native cell needs. Each field is empty where the documentation does not say.

    `ptx` is the PTX ISA version that introduced the target, `toolkit` the CUDA release that shipped that ISA, and
    `driver` the minimum driver of that release (the driver branch where the table gives no version). `gpu` is an Apple
    GPU family or an AMD architecture, and `cpu` the instruction-set extensions of an x86 target.
    """

    gpu: str = ""
    cpu: str = ""
    ptx: str = ""
    toolkit: str = ""
    driver: str = ""
    os: str = ""
    metal: str = ""
    rocm: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not any(getattr(self, name) for name in ("gpu", "cpu", "ptx", "toolkit", "driver", "os", "metal", "rocm")):
            raise ValueError("a Requires names at least one fact")

    def summary(self) -> str:
        """The facts on one line, in the order a reader checks them: toolkit, driver, ISA, GPU, CPU."""
        parts = [
            f"CUDA {self.toolkit}" if self.toolkit else "",
            f"driver {self.driver}" if self.driver else "",
            f"PTX ISA {self.ptx}" if self.ptx else "",
            f"macOS {self.os}" if self.os else "",
            f"Metal {self.metal}" if self.metal else "",
            f"ROCm {self.rocm}" if self.rocm else "",
            self.gpu,
            self.cpu,
        ]
        return ", ".join(part for part in parts if part)


def _nvidia(ptx: str, toolkit: str, driver: str) -> Requires:
    return Requires(ptx=ptx, toolkit=toolkit, driver=driver, source=NVIDIA_SOURCE)


# PTX ISA release notes give the PTX version that added each target; the toolkit table gives the driver of each CUDA
# release. sm_100a's CUDA 12.7 row is not in the table, so its driver is the branch the PTX ISA table names (R565).
NVIDIA: dict[str, Requires] = {
    "sm_80": _nvidia("7.0", "11.0", "450.80.02"),
    "sm_86": _nvidia("7.1", "11.1", "455.23"),
    "sm_89": _nvidia("7.8", "11.8", "520.61.05"),
    "sm_90a": _nvidia("8.0", "12.0", "525.60.13"),
    "sm_100a": _nvidia("8.6", "12.7", "R565 branch"),
    "sm_103a": _nvidia("8.8", "12.9", "575.51.03"),
    "sm_120a": _nvidia("8.7", "12.8", "570.26"),
    "sm_121a": _nvidia("8.8", "12.9", "575.51.03"),
}
# Apple's feature table lists each family's GPUs (Apple7 M1, Apple8 M2, Apple9 M3 and M4); no table row gives a macOS or
# Metal Shading Language minimum, so an Apple cell names its family only. M5 (Apple10) is not in the table.
APPLE: dict[int, Requires] = {family: Requires(gpu=f"Apple{family}", source=METAL_FEATURES) for family in (7, 8, 9)}
AMD_ARCH: dict[str, Requires] = {arch: Requires(gpu=arch, source=ROCM_MATRIX) for arch in ("gfx942", "gfx950")}
CPU_ISA: dict[str, Requires] = {
    "avx2": Requires(cpu="AVX2", source=INTRINSICS),
    "avx512": Requires(cpu="AVX-512F", source=INTRINSICS),
    "spr": Requires(cpu="AMX-TILE, AMX-BF16, AMX-INT8", source=f"{AMX} {INTRINSICS}"),
}


def requirement(target: Target) -> Requires | None:
    """What a native cell on `target` needs, or None where the documentation does not state it."""
    match target:
        case Nvidia():
            return NVIDIA.get(target.arch)
        case Apple():
            return APPLE.get(target.family)
        case AMD():
            return AMD_ARCH.get(target.arch)
        case CPU():
            return CPU_ISA.get(target.arch)
    return None

"""Cited dense operations per compute unit per cycle; FMA counts as two ops.

These are ceilings, not achievable kernel rates. Tensor rates use dense matrices
and the fastest supported accumulation mode; no structured sparsity multiplier.
An absent dtype is unknown, not zero or an assertion of lack of support.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, kw_only=True)
class ArchitectureRates:
    operations_per_cycle: Mapping[str, float]
    reference: str

    def __post_init__(self) -> None:
        self.__dict__["operations_per_cycle"] = MappingProxyType(dict(self.operations_per_cycle))


# GA100: 64 FP32 lanes, 32 FP64 lanes; tensor FP64 doubles the latter.
# https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
_GA100 = ArchitectureRates(
    operations_per_cycle={"f64": 128, "f32": 128, "tf32": 1024, "f16": 2048, "bf16": 2048, "int8": 4096},
    reference="https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/",
)
# GA102: 128 FP32 lanes, 2 FP64 lanes, 512 dense FP16 FMAs/SM/cycle (table 6).
# https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf
_GA102 = ArchitectureRates(
    # TF32/BF16 rates differ between GeForce and professional GA102 SKUs.
    operations_per_cycle={"f64": 4, "f32": 256, "f16": 1024, "int8": 2048},
    reference="https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf",
)
# GH100: 128 FP32 lanes, 64 FP64 lanes; FP64 tensor rate 256 ops/cycle/SM.
# https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
_GH100 = ArchitectureRates(
    operations_per_cycle={"f64": 256, "f32": 256, "tf32": 2048, "f16": 4096, "bf16": 4096, "fp8": 8192, "int8": 8192},
    reference="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/",
)
# RTX Blackwell: appendix specifications; 128 FP32 lanes/SM, dense tensor rates.
# FP16 and FP8 here use FP16 accumulation; FP32 accumulation can have lower throughput.
# https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
_GB20X = ArchitectureRates(
    # Tensor rates require the product tier, not just compute capability.
    operations_per_cycle={"f32": 256},
    reference="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf",
)
# CDNA3: vendor's FLOPS/CLOCK/CU table, including FP64 matrix instructions.
# https://rocmdocs.amd.com/en/develop/reference/gpu-arch/mi300.html
_CDNA3 = ArchitectureRates(
    operations_per_cycle={"f64": 256, "f32": 256, "tf32": 1024, "f16": 2048, "bf16": 2048, "fp8": 4096, "int8": 4096},
    reference="https://rocmdocs.amd.com/en/develop/reference/gpu-arch/mi300.html",
)

RATES: Mapping[str, ArchitectureRates | None] = MappingProxyType(
    {
        "sm_80": _GA100,
        "sm_86": _GA102,
        "sm_89": None,  # Ada accumulation-mode rates not yet audited.
        "sm_90a": _GH100,
        "sm_100a": None,  # Datacenter Blackwell rates not yet audited.
        "sm_103a": None,
        "sm_120a": _GB20X,
        "sm_121a": None,  # Do not assume GB10 has GB20x throughput.
        "gfx942": _CDNA3,
        "gfx950": None,  # CDNA4 rates not yet audited.
        # Apple publishes SKU bandwidth, but no cited per-core/cycle FMA table here.
        "apple_m1": None,
        "apple_m2": None,
        "apple_m3": None,
        "apple_m4": None,
        "apple_m5": None,
        # ISA width alone does not establish FMA ports, frequency, or heterogeneous core mix.
        "avx2": None,
        "avx512": None,
        "spr": None,
    }
)

# Exact SKU only: family membership alone does not establish a memory bus width.
# https://support.apple.com/en-gb/121554 (M4 Pro technical specifications).
MEMORY_BANDWIDTH_GBS: Mapping[str, float] = MappingProxyType({"Apple M4 Pro": 273.0})

# The cited appendices give exact dense rates for these products; professional
# cards of the same architecture can have different accumulation throughput.
# Prefixes NVIDIA and GeForce are removed from probe identity before lookup.
DEVICE_RATES: Mapping[tuple[str, str], ArchitectureRates] = MappingProxyType(
    {
        ("sm_86", "RTX 3090"): ArchitectureRates(
            operations_per_cycle={"f64": 4, "f32": 256, "tf32": 256, "f16": 1024, "bf16": 512, "int8": 2048},
            reference=_GA102.reference,  # GA102 whitepaper, appendix A, table 9.
        ),
        ("sm_120a", "RTX 5080"): ArchitectureRates(
            operations_per_cycle={"f32": 256, "tf32": 256, "f16": 1024, "bf16": 512, "fp8": 2048, "int8": 2048},
            reference=_GB20X.reference,  # RTX Blackwell whitepaper, appendix B, table 4.
        ),
    }
)

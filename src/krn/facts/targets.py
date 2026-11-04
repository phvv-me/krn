"""Targets as data: what each machine reachable from Tsukuba can do, queried while a kernel is captured.

A `Target` is a frozen value that exists before capture, so every attribute is a constant in the kernel body and a
branch on it folds away. `Target.of("sm_100a")` builds one from the tables below. The tables are a first draft from
the study notes (the capability matrix in notes/round4 is the source to fix them against); they are real code, unlike
the kernel bodies of the sketch.

    sm_90a, sm_100a, sm_120a   NVIDIA Hopper, datacenter Blackwell, consumer Blackwell (GB10 is sm_121a)
    gfx942, gfx950             AMD MI300A (Sirius) and MI355
    apple_m1 ... apple_m5      Apple GPUs
    avx2, avx512, spr          x86 CPUs, spr is Sapphire Rapids (Miyabi-C, Pegasus hosts) with AMX
"""

from dataclasses import dataclass
from enum import StrEnum, auto
from functools import cache
from typing import ClassVar, Self

from krn.data.formats import BlockFormat, FloatFormat, IntFormat, Special


class Feature(StrEnum):
    """What a kernel may need from a target. `requires(...)` and `target.has(...)` are checked at capture time."""

    SHARED_MEMORY = auto()  # block-shared memory (a stack tile on CPUs)
    WARP_SHUFFLE = auto()
    WARP_VOTE = auto()
    ATOMICS_64 = auto()
    FLOAT64 = auto()
    TENSOR_CORE = auto()  # some native matrix unit exists: mma.sync, wgmma, tcgen05, MFMA, simdgroup_matrix, AMX
    ASYNC_COPY = auto()  # global to shared copy that does not stall the issuing thread (cp.async, direct-to-LDS)
    BULK_COPY = auto()  # descriptor driven bulk copies (TMA)
    ASYNC_MMA = auto()  # matrix multiply issued by one thread, reading shared memory (wgmma, tcgen05)
    GRID_SYNC = auto()  # a barrier across the grid, through a cooperative launch (CUDA 9, HIP): `grid.sync()`
    TENSOR_MEMORY = auto()  # accumulators live in per-SM tensor memory, not registers (tcgen05 on sm_100a, sm_103a)
    CLUSTERS = auto()  # thread block clusters with distributed shared memory
    BLOCK_SCALED_MMA = auto()  # MX and NVFP4 products without a software decode


class Dtype(StrEnum):
    """Capability tokens for element formats. Storage means load, store and convert; mma means a native multiply."""

    F64 = auto()
    F32 = auto()
    TF32 = auto()
    F16 = auto()
    BF16 = auto()
    F8E4M3FN = auto()
    F8E4M3FNUZ = auto()
    F8E5M2 = auto()
    F8E5M2FNUZ = auto()
    F6 = auto()
    F4E2M1 = auto()
    I8 = auto()
    I4 = auto()

    @classmethod
    def of(cls, element: FloatFormat | IntFormat) -> Dtype:
        """The token of an element format. The mapping covers the formats the sketch names."""
        match element:
            case IntFormat(bits=8):
                return cls.I8
            case IntFormat(bits=4):
                return cls.I4
            case FloatFormat(exp_bits=11, man_bits=52):
                return cls.F64
            case FloatFormat(exp_bits=8, man_bits=23):
                return cls.F32
            case FloatFormat(exp_bits=5, man_bits=10):
                return cls.F16
            case FloatFormat(exp_bits=8, man_bits=7):
                return cls.BF16
            case FloatFormat(exp_bits=4, man_bits=3, special=Special.FNUZ):
                return cls.F8E4M3FNUZ
            case FloatFormat(exp_bits=4, man_bits=3):
                return cls.F8E4M3FN
            case FloatFormat(exp_bits=5, man_bits=2, special=Special.FNUZ):
                return cls.F8E5M2FNUZ
            case FloatFormat(exp_bits=5, man_bits=2):
                return cls.F8E5M2
            case FloatFormat(exp_bits=2, man_bits=1):
                return cls.F4E2M1
            case FloatFormat(exp_bits=2, man_bits=3) | FloatFormat(exp_bits=3, man_bits=2):
                return cls.F6
            case _:
                raise ValueError(f"no capability token for {element}")


@dataclass(frozen=True, slots=True)
class Target:
    """What a kernel is compiled for.

    `warp_size` is the lockstep width: 32 on NVIDIA, 64 on AMD CDNA, 32 on Apple (read from the pipeline when it is
    not a constant of the architecture), and the number of float32 lanes of a vector register on a CPU.
    """

    vendor: ClassVar[str] = "any"

    arch: str
    warp_size: int
    shared_bytes: int
    features: frozenset[Feature]
    storage: frozenset[Dtype]
    mma: frozenset[Dtype]
    scaled: frozenset[str] = frozenset()  # block-scaled formats the matrix unit takes with their scales, by format name
    # The operations that flush float32 subnormal operands or results here, by the names `compiler.numerics` gives
    # them. Empty where IEEE denormals hold, None where the target is unmeasured (then nothing is claimed either way).
    denormal_flushes: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.warp_size & (self.warp_size - 1):
            raise ValueError(f"{self.arch}: warp_size {self.warp_size} is not a power of two")
        if missing := self.mma - self.storage:
            raise ValueError(f"{self.arch}: mma dtypes {sorted(missing)} are not storage dtypes")

    @property
    def name(self) -> str:
        return self.vendor

    def has(self, feature: Feature) -> bool:
        return feature in self.features

    def supports(self, element: FloatFormat | IntFormat) -> bool:
        """Can the target hold and convert this element format?"""
        return Dtype.of(element) in self.storage

    def mma_native(self, fmt: BlockFormat | FloatFormat | IntFormat) -> bool:
        """Does the matrix unit multiply this format natively?

        Scales finer than one MMA step along K (MX and NVFP4 blocks of 16 or 32) need a block-scaled instruction that
        takes that very format (`scaled`). A
        coarse scale (a float32 per 1 x 128 block) is applied to the partial sums outside the unit, so the element
        format alone decides.
        """
        if isinstance(fmt, BlockFormat):
            fine_scale = fmt.scale is not None and fmt.block[1] <= 32
            return Dtype.of(fmt.elem) in self.mma and (not fine_scale or fmt.name in self.scaled)
        return Dtype.of(fmt) in self.mma

    @classmethod
    def of(cls, arch: str) -> Self:
        """The target of an architecture name such as `sm_100a`, `gfx942`, `apple_m5` or `spr`.

        Called on a vendor class it also checks the vendor: `CPU.of("sm_90a")` raises, and `CPU.of("spr")` is a `CPU`.
        """
        if arch not in (known := registry()):
            raise KeyError(f"unknown target {arch!r}; known: {', '.join(known)}")
        if not isinstance(found := known[arch], cls):
            raise KeyError(f"{arch!r} is a {found.vendor} target, not {cls.vendor}")
        return found

    @classmethod
    def table(cls) -> dict[str, Target]:
        """The architectures of this vendor by name."""
        return {}


def _fmt(*names: Dtype) -> frozenset[Dtype]:
    return frozenset(names)


_CORE = frozenset(
    {
        Feature.SHARED_MEMORY,
        Feature.WARP_SHUFFLE,
        Feature.WARP_VOTE,
        Feature.ATOMICS_64,
        Feature.FLOAT64,
    }
)
_F = Dtype
_AMPERE = _fmt(_F.F64, _F.F32, _F.TF32, _F.F16, _F.BF16, _F.I8, _F.I4)  # F32 means the FMA unit multiplies it
_ADA = _AMPERE | _fmt(_F.F8E4M3FN, _F.F8E5M2)
_BLACKWELL = _ADA | _fmt(_F.F6, _F.F4E2M1)


@dataclass(frozen=True, slots=True)
class Nvidia(Target):
    vendor: ClassVar[str] = "nvidia"

    @property
    def sm(self) -> int:
        """The compute capability as one number: 90 for sm_90a, 120 for sm_120a."""
        return int(self.arch.removeprefix("sm_").removesuffix("a"))

    @classmethod
    def table(cls) -> dict[str, Target]:
        ampere = _CORE | {Feature.TENSOR_CORE, Feature.ASYNC_COPY, Feature.GRID_SYNC}
        hopper = ampere | {Feature.BULK_COPY, Feature.ASYNC_MMA, Feature.CLUSTERS}
        datacenter = hopper | {Feature.TENSOR_MEMORY, Feature.BLOCK_SCALED_MMA}
        consumer = ampere | {Feature.BULK_COPY, Feature.BLOCK_SCALED_MMA}
        mx = frozenset({"nvfp4", "mxfp4", "mxfp8"})
        rows = {
            "sm_80": (163, ampere, _AMPERE, frozenset()),
            "sm_86": (99, ampere, _AMPERE, frozenset()),
            "sm_89": (99, ampere, _ADA, frozenset()),
            "sm_90a": (227, hopper, _ADA, frozenset()),
            "sm_100a": (227, datacenter, _BLACKWELL, mx),
            # sm_103a keeps fp4 and fp6: tcgen05 `kind::mxf4` and the cvt `.rs` form name it in PTX 9.4. sm_121a (GB10,
            # NVIDIA advertises FP4 for it) stays with consumer Blackwell: the sm_120f family target covers it.
            # Unverified on a card here: check with an NVRTC compile of the fp4 cvt on a GB10 host when reachable.
            "sm_103a": (227, datacenter, _BLACKWELL, mx),
            "sm_120a": (99, consumer, _BLACKWELL, mx),
            "sm_121a": (99, consumer, _BLACKWELL, mx),
        }
        return {
            # The strict flags keep --ftz=false, so denormals hold; the rows are not measured on a card here.
            arch: cls(arch, 32, kib * 1024, frozenset(features), dtypes, dtypes, scaled, frozenset())
            for arch, (kib, features, dtypes, scaled) in rows.items()
        }


@dataclass(frozen=True, slots=True)
class AMD(Target):
    """gfx942 is Sirius' MI300A: wave64, 64 KB of LDS, FNUZ fp8 and no block-scaled MFMA. gfx950 has OCP fp8 and MX.

    Float32 denormals are unmeasured here, so `denormal_flushes` stays None and no denormal claim is made.
    """

    vendor: ClassVar[str] = "amd"

    @classmethod
    def table(cls) -> dict[str, Target]:
        cdna3 = _CORE | {Feature.TENSOR_CORE, Feature.ASYNC_COPY, Feature.GRID_SYNC}
        cdna3_types = _fmt(_F.F64, _F.F32, _F.TF32, _F.F16, _F.BF16, _F.F8E4M3FNUZ, _F.F8E5M2FNUZ, _F.I8)
        cdna4_types = _fmt(_F.F64, _F.F32, _F.F16, _F.BF16, _F.F8E4M3FN, _F.F8E5M2, _F.F6, _F.F4E2M1, _F.I8)
        return {
            "gfx942": cls("gfx942", 64, 64 * 1024, frozenset(cdna3), cdna3_types, cdna3_types),
            "gfx950": cls(
                "gfx950",
                64,
                160 * 1024,
                frozenset(cdna3 | {Feature.BLOCK_SCALED_MMA}),
                cdna4_types,
                cdna4_types,
                frozenset({"mxfp4", "mxfp8"}),
            ),
        }


# Measured on M4 Pro (Apple9) by tests/compiler/test_denormals.py: these operations flush float32 subnormal operands or
# results to zero or to the sign, while copy and convert keep them (a float32 subnormal converted to f16 is zero on both
# sides, so only the copy keeps the bits). Every apple_m* row shares this set until another chip is measured.
_APPLE_FLUSHES = frozenset({"add", "sub", "mul", "div", "sqrt", "rsqrt", "fma", "minimum", "maximum", "compare"})


@dataclass(frozen=True, kw_only=True)
class Apple(Target):
    """All rows have 32 lanes, no fp64, and no full 64-bit atomic set.

    Measured on M4 Pro (Apple9), macOS 27, MSL 3.0 through 4.1: only unsigned 64-bit min/max
    in device memory compile, returning void. No 64-bit load, store, exchange, CAS, fetch operations,
    or threadgroup atomics compile. None of these rows advertises ATOMICS_64.

    Apple10 (M5) adds Metal 4 tensor operations and Neural Accelerators, not yet used by the renderer.
    Family 11 forward progress is not modeled by these M1 through M5 rows.
    """

    vendor: ClassVar[str] = "apple"
    family: int

    @classmethod
    def table(cls) -> dict[str, Target]:
        types = _fmt(_F.F32, _F.F16, _F.BF16, _F.I8)
        features = frozenset(_CORE - {Feature.FLOAT64, Feature.ATOMICS_64} | {Feature.TENSOR_CORE})
        return {
            f"apple_m{n}": cls(
                f"apple_m{n}",
                32,
                32 * 1024,
                features,
                types,
                types,
                denormal_flushes=_APPLE_FLUSHES,
                family=family,
            )
            for n, family in enumerate((7, 8, 9, 9, 10), 1)
        }


@dataclass(frozen=True, slots=True)
class CPU(Target):
    """A CPU: a block is a loop nest, a warp is a vector of float32 lanes, shared memory is a stack tile."""

    vendor: ClassVar[str] = "cpu"

    @classmethod
    def table(cls) -> dict[str, Target]:
        base = frozenset({Feature.SHARED_MEMORY, Feature.FLOAT64, Feature.ATOMICS_64})
        scalar = _fmt(_F.F64, _F.F32, _F.I8)
        spr = _fmt(_F.F64, _F.F32, _F.F16, _F.BF16, _F.I8)
        # The C backend compiles without fast math or flush-to-zero, so denormals hold. Measured with the arm64 clang
        # here; the x86 rows share its flags.
        return {
            "avx2": cls("avx2", 8, 32 * 1024, base, scalar, _fmt(_F.F32, _F.F64), denormal_flushes=frozenset()),
            "avx512": cls("avx512", 16, 32 * 1024, base, scalar, _fmt(_F.F32, _F.F64), denormal_flushes=frozenset()),
            "spr": cls(
                "spr",
                16,
                48 * 1024,
                base | {Feature.TENSOR_CORE},
                spr,
                _fmt(_F.F32, _F.F64, _F.F16, _F.BF16, _F.I8),
                denormal_flushes=frozenset(),
            ),
        }


@cache
def registry() -> dict[str, Target]:
    """Every architecture the tables know, by name. Built once."""
    return {arch: target for family in (Nvidia, AMD, Apple, CPU) for arch, target in family.table().items()}


def all_targets() -> tuple[Target, ...]:
    """Every architecture the tables know: the list to audit a family or a ladder against."""
    return tuple(registry().values())

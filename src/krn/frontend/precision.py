"""Precision ladders: use the narrowest format the machine handles well, and fall back wider on machines that do not.

A ladder lists formats narrowest first. Resolving it against a target walks the rungs and returns the first that fits
the policy, with the reason for every rung it skipped. Two questions are separate:

    storage   the format the weights are kept in, which sets the bytes moved
    compute   the format the matrix unit multiplies, which sets the instruction

When they differ the route is DECODE: the kernel loads narrow codes and widens them in registers before a native
product. That keeps the memory saving on a target without native FP4 (weight-only quantization, as in Marlin) and costs
decode work, so a compute-bound GEMM prefers the narrowest NATIVE rung and a memory-bound matvec the narrowest rung.
Settings pin a rung per ladder or set a floor. A pinned rung must also meet the floor;
a precision choice can never silently bypass it.
"""

from dataclasses import dataclass
from enum import StrEnum, auto

from krn.compiler.errors import TargetError
from krn.data.formats import BF16, FP16, FP32, BlockFormat, FloatFormat
from krn.facts.targets import Dtype, Target
from krn.runtime.settings import Policy, Settings, current_settings

from .memory import Matrix, QMatrix
from .numerics import f32, todo


class Route(StrEnum):
    NATIVE = auto()  # the matrix unit multiplies the stored format itself
    DECODE = auto()  # codes are stored narrow and widened to `compute` in registers


@dataclass(frozen=True, slots=True)
class Resolution[F: BlockFormat]:
    """The outcome for one ladder on one target. `reason` lists why the rungs above `storage` were skipped."""

    ladder: str
    storage: F
    compute: FloatFormat | BlockFormat
    route: Route
    reason: str

    def quantize(self, weights: Matrix[f32]) -> QMatrix[F]:
        """Convert float32 weights to the storage format, computing scales per block."""
        return todo()

    def __str__(self) -> str:
        return f"{self.ladder}: store {self.storage.name}, multiply {_name(self.compute)} ({self.route}); {self.reason}"


def _name(fmt: FloatFormat | BlockFormat) -> str:
    return fmt.name if isinstance(fmt, BlockFormat) else f"e{fmt.exp_bits}m{fmt.man_bits}"


def _decode_target(target: Target) -> FloatFormat:
    """The format a decoded product runs in: float16 if the unit multiplies it, else bfloat16, else float32."""
    for token, fmt in ((Dtype.F16, FP16), (Dtype.BF16, BF16)):
        if token in target.mma:
            return fmt
    return FP32


@dataclass(frozen=True, slots=True)
class Ladder[F: BlockFormat]:
    """Formats from narrow to wide by element bits.

    Among equal widths the earlier rung is preferred (NVFP4 before MXFP4, since its scales are finer). Construction
    checks that the widths do not decrease."""

    name: str
    rungs: tuple[F, ...]
    policy: Policy = Policy.COMPUTE_BOUND

    def __post_init__(self) -> None:
        widths = [rung.elem.bits for rung in self.rungs]
        if widths != sorted(widths):
            raise ValueError(f"ladder {self.name!r} must list rungs from narrow to wide, got {widths}")

    def resolve(self, target: Target, settings: Settings | None = None) -> Resolution[F]:
        """Pick a rung for this target.

        Under COMPUTE_BOUND the narrowest rung with a native product wins. Under MEMORY_BOUND the narrowest width wins,
        a native rung of that width first, and a wider native format decodes it.
        """
        settings = settings or current_settings()
        floor = settings.precision.floor_bits or 0
        if pin := settings.precision.pins.get(self.name):
            result = self._pinned(target, pin)
            if result.storage.elem.bits < floor:
                raise TargetError(
                    f"precision.pins.{self.name}",
                    target.arch,
                    f"{pin} is below the {floor}-bit precision floor",
                    "choose a wider rung or explicitly lower precision.floor_bits",
                )
            return result
        usable = [rung for rung in self.rungs if rung.elem.bits >= floor]
        dropped = [f"{rung.name} is under the {floor}-bit floor" for rung in self.rungs if rung not in usable]
        if (settings.precision.policy or self.policy) is Policy.MEMORY_BOUND and usable:
            tier = [rung for rung in usable if rung.elem.bits == usable[0].elem.bits]
            rung = next((r for r in tier if target.mma_native(r)), tier[0])
            return self._outcome(target, rung, dropped, "narrowest width, memory bound")
        skipped = [f"{rung.name} has no native product on {target.arch}" for rung in usable]
        for rung in usable:
            if target.mma_native(rung):
                cut = skipped[: usable.index(rung)]
                return self._outcome(target, rung, [*dropped, *cut], "narrowest native rung")
        raise TargetError(
            f"precision ladder {self.name}",
            target.arch,
            "; ".join(dropped + skipped) or "the ladder is empty",
            "provide a wider supported rung or explicitly lower precision.floor_bits",
        )

    def _outcome(self, target: Target, rung: F, notes: list[str], why: str) -> Resolution[F]:
        route = Route.NATIVE if target.mma_native(rung) else Route.DECODE
        compute = rung if route is Route.NATIVE else _decode_target(target)
        return Resolution(self.name, rung, compute, route, "; ".join([why, *notes]))

    def _pinned(self, target: Target, pin: str) -> Resolution[F]:
        match [rung for rung in self.rungs if rung.name == pin]:
            case [rung]:
                route = Route.NATIVE if target.mma_native(rung) else Route.DECODE
                compute = rung if route is Route.NATIVE else _decode_target(target)
                return Resolution(self.name, rung, compute, route, f"pinned to {pin} by precision.pins")
            case _:
                names = [rung.name for rung in self.rungs]
                raise LookupError(f"ladder {self.name!r} has no rung {pin!r}; rungs: {names}")

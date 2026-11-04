"""Backend choices and their reasons, resolved once per specialization.

Configuration defaults are handled by Settings.configuration. User callees have already been expanded at capture;
inlining here controls only the helpers emitted by a renderer.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum, auto
from itertools import combinations

from krn.facts.targets import CPU, Target
from krn.frontend.geometry import Blocks, Geometry
from krn.runtime.settings import Mode, Settings

from . import ir
from .debug import Explanation as Explanation
from .numerics import explain as explain_numerics

MAX_THREADS = 1024
WARP = 32  # the width of a reduction warp on every target: a wave of 64 reduces as two logical warps


@dataclass(frozen=True, slots=True)
class Defaults:
    """The values the rule of `experiments/heuristics_bench.py` chose (notes/round10/fe3-status.md section 4)."""

    inline_limit: int = 64
    element_threads: int = 256
    tile_threads: int = 128
    unroll: int = 2  # items per thread of an `items` loop: the C backend and the CUDA family both scored best at 2


DEFAULTS = Defaults()


class Inlining(StrEnum):
    """How a renderer spells a helper function. `INLINE` is each backend's own historical spelling for a small helper,
    `ALWAYS` is the hardest the backend has (`always_inline`), used internally by renderers."""

    ALWAYS = auto()
    INLINE = auto()
    PLAIN = auto()
    NEVER = auto()


@dataclass(frozen=True, slots=True)
class Choice[T]:
    """A decision and its reason. `explicit` says launch geometry chose it, not a rule."""

    value: T
    why: str
    explicit: bool = False


@dataclass(frozen=True, slots=True)
class Extent:
    """The bytes a call touches in one tensor, for dependencies and the restrict variant."""

    low: int
    high: int
    write: bool

    def conflicts(self, other: Extent) -> bool:
        """True when order matters: the byte ranges overlap and one side writes (RAW, WAR or WAW)."""
        return (self.write or other.write) and max(self.low, other.low) < min(self.high, other.high)


def aliased(extents: Sequence[Extent]) -> bool:
    return any(left.conflicts(right) for left, right in combinations(extents, 2))


@dataclass(frozen=True, slots=True, kw_only=True)
class Decisions:
    kernel: str
    target: str
    settings: Settings
    inline: Choice[Mode]
    inline_limit: Choice[int]
    restrict: Choice[bool]
    const: Choice[bool]
    block: Choice[int]
    launch_bounds: Choice[int | None]
    maxregs: Choice[int | None]
    unroll: Choice[int]
    lanes: Choice[int]
    numerics: Choice[tuple[str, ...]] = Choice((), "strict arithmetic; explicit calls only")

    def helper(self, size: int) -> Inlining:
        # Capture expands user callees; these qualifiers apply only to emitted helpers.
        if self.inline.value is Mode.ALWAYS:
            return Inlining.ALWAYS
        if self.inline.value is Mode.NEVER:
            return Inlining.NEVER
        limit = self.inline_limit.value
        return Inlining.INLINE if size <= limit else Inlining.PLAIN if size <= 8 * limit else Inlining.NEVER

    def rows(self) -> list[tuple[str, str, str]]:
        return [
            (field.name, str(choice.value), choice.why)
            for field in fields(self)
            if isinstance(choice := getattr(self, field.name), Choice)
        ]

    def floor(self, threads: int) -> Decisions:
        if threads <= self.block.value:
            return self
        if self.block.explicit:
            raise ValueError(f"{self.block.why}: tile product needs {threads} threads")
        return replace(
            self,
            block=Choice(threads, f"tile product needs {threads} threads"),
            launch_bounds=replace(self.launch_bounds, value=threads)
            if self.launch_bounds.value is not None
            else self.launch_bounds,
        )


def space_text(axis: ir.Axis) -> str:
    def expression(value: ir.Expr) -> str:
        match value:
            case ir.Dim(tensor=tensor, axis=dimension):
                return f"{'.'.join(tensor.path)}.shape[{dimension}]"
            case ir.Param(path=path):
                return ".".join(path)
            case ir.Const(value=constant):
                return str(constant)
            case ir.Cast(operand=inner):
                return expression(inner)
            case ir.Binary(op=op, lhs=left, rhs=right):
                return f"({expression(left)} {op.value} {expression(right)})"
        return str(value)

    extents = ", ".join(expression(value) for value in axis.extents)
    return f"{axis.grid}: items({extents})" if axis.strided else f"{axis.grid}: tile {axis.tile} of {extents}"


def check_lanes(width: int) -> int:
    """The lanes of a tile reduction: a power of two below the warp width or a multiple of it, so that no warp is
    part way populated."""
    if (width < WARP and width > 0 and width & (width - 1) == 0) or (width > 0 and width % WARP == 0):
        return width
    raise ValueError(
        f"launch threads={width}: a tile reduction needs 1, 2, 4, 8 or 16 threads, or a multiple of {WARP}"
    )


def lanes(launch: Geometry | None) -> Choice[int]:
    """The lanes that reduce a tile axis on every target, the emulator and the C backend included: the block of an
    explicit launch, else the tile default. The order of a float sum depends on it (notes/spec/numerics.md)."""
    if isinstance(launch, Blocks) and launch.threads is not None:
        return Choice(launch.threads, f"launch={launch}", True)
    return Choice(DEFAULTS.tile_threads, "rule: tile kernel default block")


def threads(kernel: ir.Kernel, settings: Settings) -> int:
    """The inferred block before the C backend's one-thread tile convention or an explicit launch.

    Only tile kernels take the smaller block: their loops stride by `blockDim`, while a kernel with `shared_array` is
    written against a block size (`shared_array(f32, 256)`), so it keeps the element default (`I`, owner rule: the
    heuristic never changes what a kernel's own text assumes)."""
    return DEFAULTS.tile_threads if kernel.tiled else DEFAULTS.element_threads


def decide(
    kernel: ir.Kernel,
    target: Target,
    settings: Settings,
    *,
    aliased: bool | None = None,
    launch: Geometry | None = None,
) -> Decisions:
    settings.validate(target)
    block = Choice(threads(kernel, settings), "rule: tile kernel" if kernel.tiled else "rule: element kernel")
    if isinstance(target, CPU) and kernel.tiled:
        block = Choice(1, "the C backend runs a tile as one thread")
    if isinstance(launch, Blocks) and launch.threads is not None:
        block = Choice(launch.threads, f"launch={launch}", True)
    if not isinstance(target, CPU) and not 0 < block.value <= MAX_THREADS:
        raise ValueError(f"launch={block.value}: expected positive threads, at most {MAX_THREADS}")
    return Decisions(
        kernel=kernel.name,
        target=target.arch,
        settings=settings,
        inline=Choice(Mode.AUTO, "rule: helper size"),
        inline_limit=Choice(DEFAULTS.inline_limit, "rule: helper size"),
        restrict=Choice(False, "rule: measured default never"),
        const=Choice(True, "rule: read-only parameters"),
        block=block,
        launch_bounds=Choice(
            block.value if launch is not None and not isinstance(target, CPU) else None,
            "launch=" + str(launch) if launch is not None else "rule: launch unknown",
        ),
        maxregs=Choice(None, "rule: compiler register allocation"),
        unroll=Choice(DEFAULTS.unroll, "rule: measured items per thread"),
        lanes=lanes(launch),
        numerics=Choice(explain_numerics(kernel, target), "explicit call-site numerical contracts"),
    )

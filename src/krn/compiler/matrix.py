"""Exact matrix instructions, ownership layouts, and staging budgets independent of source renderers."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from krn.facts.targets import Target

from .dtypes import BF16, F16, F32, Dtype, fp
from .errors import TargetError
from .ir import Accumulate, Cast, Expr


@dataclass(frozen=True, slots=True)
class Instruction:
    """One exact operand pair and its public instruction spelling and fragment dimensions."""

    operands: tuple[Dtype, Dtype]
    k: int
    minimum_sm: int
    spelling: str
    m: int = 16
    n: int = 8


FP8 = (fp(4, 3), fp(5, 2))
FP8_TARGETS = "sm_89, sm_90a, sm_100a, sm_120a"
FP8_AVAILABILITY = f"{FP8_TARGETS} have the NVIDIA fp8 forms with an f32 accumulator"


@dataclass(frozen=True, slots=True)
class InstructionSet:
    instructions: tuple[Instruction, ...]
    targets: str
    granule: tuple[int, int, int]
    arches: tuple[str, ...] = ()

    @property
    def availability(self) -> str:
        minimum_sm = min(i.minimum_sm for i in self.instructions)
        pairs = dict.fromkeys(
            f"{i.operands[0]} x {i.operands[1]}" for i in self.instructions if i.minimum_sm == minimum_sm
        )
        targets = " and ".join(self.arches) or self.targets
        return f"{targets} have {' and '.join(pairs)} with an f32 accumulator"

    def match(self, s: Accumulate, sm: int = 0) -> Instruction | str:
        a, b = operand(s.lhs).dtype, operand(s.rhs).dtype
        candidates = [i for i in self.instructions if i.operands == (a, b) and i.minimum_sm <= sm]
        if not candidates or s.acc.dtype != F32:
            exact = "; an f32 operand does not widen exactly to tf32" if F32 in (a, b) else ""
            return f"no tensor-core instruction for {a} x {b} with an {s.acc.dtype} accumulator{exact}"
        bm, bn = s.acc.shape
        bk = s.lhs.shape[1]
        gm, gn, gk = self.granule
        if a in FP8 or b in FP8:
            gk = 32
        if bm % gm or bn % gn or bk % gk:
            return f"tile {bm}x{bn}x{bk} needs BM divisible by {gm}, BN by {gn}, BK by {gk}"
        return next(i for i in candidates if bk % i.k == 0)

    def refusal(self, s: Accumulate, target: Target, reason: str) -> str:
        availability = FP8_AVAILABILITY if any(operand(e).dtype in FP8 for e in (s.lhs, s.rhs)) else self.availability
        return fallback(s, target, f"{reason}; {availability}")


@dataclass(frozen=True, slots=True)
class MatrixPlan:
    """An eligible instruction, tile dimensions, and a warp grid that owns its accumulator."""

    instruction: Instruction
    bm: int
    bn: int
    bk: int
    wm: int
    wn: int
    warp_size: int

    @property
    def tf32(self) -> bool:
        return self.instruction.operands in ((F16, BF16), (BF16, F16))

    @property
    def fp8(self) -> bool:
        return self.instruction.operands[0] in FP8

    @property
    def description(self) -> str:
        return self.instruction.spelling + (" (exact via tf32)" if self.tf32 else "")

    @property
    def threads(self) -> int:
        return self.warp_size * self.wm * self.wn

    @property
    def mt(self) -> int:
        return self.bm // self.wm // self.instruction.m

    @property
    def nt(self) -> int:
        return self.bn // self.wn // self.instruction.n


def operand(e: Expr) -> Expr:
    """Recover a losslessly widened storage operand without undoing explicit narrowing."""
    return e.operand if isinstance(e, Cast) and e.dtype == F32 and e.operand.dtype in (F16, BF16) else e


def warp_grid(bm: int, bn: int, granule: tuple[int, int] = (16, 8), warp_size: int = 32) -> tuple[int, int]:
    budget = (128 if bm * bn <= 128 * 128 else 256) // warp_size
    grids = [
        (m, n)
        for m in range(1, budget + 1)
        for n in range(1, budget + 1)
        if m * n <= budget and bm % (granule[0] * m) == 0 and bn % (granule[1] * n) == 0
    ]
    return min(
        grids,
        key=lambda g: (
            max(bm / g[0], bn / g[1]) > 64,
            -math.prod(g),
            abs(math.log2((bm / g[0]) / (bn / g[1]))),
            g[0],
        ),
    )


def fallback(s: Accumulate, target: Target, reason: str) -> str:
    """Refuse undecodable fp8 instead of silently selecting scalar arithmetic."""
    if any(operand(e).dtype in FP8 for e in (s.lhs, s.rhs)):
        raise TargetError(
            "fp8 matrix product",
            target.arch,
            f"{reason}; fp8 has no scalar fallback",
            f"use f16 or bf16 operands, or {FP8_TARGETS} with a supported tile",
            span=s.span,
        )
    return f"{reason}; widened to f32 and run on the scalar loop of fused multiply-adds, one rounding per product"


class MatrixUnit(ABC):
    """Cooperative fragments with optional element ownership and asynchronous operand copies."""

    owned_fragments: ClassVar[bool] = True
    async_copy: ClassVar[bool] = False

    def spill(self, name: str, plan: MatrixPlan, mirror: str) -> list[str]:
        raise NotImplementedError("owned fragments spill through their coordinates")

    def fill(self, name: str, plan: MatrixPlan, mirror: str) -> list[str]:
        raise NotImplementedError("owned fragments fill through their coordinates")

    @abstractmethod
    def plan(self, s: Accumulate, target: Target) -> MatrixPlan | str: ...

    @abstractmethod
    def declare(self, name: str, plan: MatrixPlan) -> str: ...

    @abstractmethod
    def coordinates(self, plan: MatrixPlan) -> tuple[str, str]: ...

    @abstractmethod
    def slot(self, name: str) -> str: ...

    @abstractmethod
    def active(self, plan: MatrixPlan) -> str: ...

    @abstractmethod
    def product(self, name: str, plan: MatrixPlan, a: str, b: str) -> list[str]: ...

    def copy_async(self, destination: str, source: str, size: str) -> str:
        raise NotImplementedError("this matrix unit has no asynchronous copy")

    def commit(self) -> str:
        raise NotImplementedError("this matrix unit has no asynchronous copy")

    def wait(self, pending: int) -> str:
        raise NotImplementedError("this matrix unit has no asynchronous copy")


def pipeline_stages(stage_bytes: int, requested: int, available: int, reserved: int = 0) -> int:
    """Prefer two resident 48 KiB blocks; larger stages may use the target's full shared allocation."""
    if requested:
        return requested if 2 <= requested <= 8 and requested * stage_bytes + reserved <= available else 0
    for limit in (min(48 * 1024, available), available):
        for stages in (4, 3, 2):
            if stages * stage_bytes + reserved <= limit:
                return stages
    return 0

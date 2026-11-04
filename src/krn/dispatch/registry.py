"""General functions resolved at capture: an overload is (name, operand pattern, target features, lowering).

A pattern is one dtype or a family. The most specific match wins; two overloads that could win the same call are an
error at registration. A lookup that matches nothing raises TargetError and names every candidate it tried.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto

from krn.compiler.dtypes import BF16, F16, F32, F64, I8X4, I16X2, U8, U8X4, U16, U16X2, U32, U64, Dtype, Kind, sint
from krn.compiler.errors import Span, TargetError
from krn.compiler.ir import Binary, Cast, Expr, Intrinsic, Math, Select, Unary
from krn.compiler.validity import children
from krn.facts import catalog
from krn.facts.targets import Feature, Target

GPU_VENDORS = {"nvidia": "NVIDIA GPUs", "amd": "AMD GPUs", "apple": "Apple GPUs"}


class Family(StrEnum):
    ANY = auto()
    INT = auto()
    SIGNED = auto()
    UNSIGNED = auto()
    FLOAT = auto()
    PACKED = auto()


type Pattern = Dtype | Family

_TESTS = {
    Family.ANY: lambda d: True,
    Family.INT: lambda d: d.is_int,
    Family.SIGNED: lambda d: d.kind is Kind.SINT,
    Family.UNSIGNED: lambda d: d.kind is Kind.UINT,
    Family.FLOAT: lambda d: d.is_float,
    Family.PACKED: lambda d: d.kind is Kind.PACKED,
}
SAMPLES = (
    U8, U16, U32, U64, sint(8), sint(16), sint(32), sint(64), F16, BF16, F32, F64, I8X4, U8X4, I16X2, U16X2,
)  # fmt: skip  # the dtypes that overlap tests use


def matches(pattern: Pattern, dtype: Dtype) -> bool:
    return pattern == dtype if isinstance(pattern, Dtype) else _TESTS[pattern](dtype)


def rank(pattern: Pattern) -> int:
    """An exact dtype outranks a family, and the ANY family outranks nothing."""
    return 3 if isinstance(pattern, Dtype) else 0 if pattern is Family.ANY else 1


def overlaps(a: Pattern, b: Pattern) -> bool:
    return any(matches(a, d) and matches(b, d) for d in SAMPLES)


@dataclass(frozen=True, slots=True)
class Instruction:
    """A catalog row. It is one native instruction on the vendors in `native`, and a software sequence elsewhere.
    `result` is the value's dtype, or None for the operand type."""

    row: str
    native: frozenset[str] = frozenset()
    result: Dtype | None = None


@dataclass(frozen=True, slots=True)
class Expansion:
    """Ordinary IR operations, correct on every target. It is always a software sequence."""

    build: Callable[[Sequence[Expr], Span], Expr]


@dataclass(frozen=True, slots=True)
class Refusal:
    """The overload exists and this target cannot run it; `instead` is the alternative the error names."""

    reason: str
    instead: str


type Lowering = Instruction | Expansion | Refusal


@dataclass(frozen=True, slots=True)
class Overload:
    name: str
    pattern: Pattern
    lowering: Lowering
    features: frozenset[Feature] = frozenset()

    @property
    def specificity(self) -> tuple[int, int]:
        return rank(self.pattern), len(self.features)


@dataclass(frozen=True, slots=True)
class Built:
    """A lowered call: its IR, whether one native instruction runs it, and the operations of the sequence."""

    expr: Expr
    native: bool
    instructions: int


_OVERLOADS: dict[str, list[Overload]] = {}


def register(name: str, pattern: Pattern, lowering: Lowering, features: frozenset[Feature] = frozenset()) -> None:
    """Add an overload. Two overloads of one name with the same specificity and overlapping patterns tie."""
    new = Overload(name, pattern, lowering, features)
    existing = _OVERLOADS.setdefault(name, [])
    for old in existing:
        if old.specificity == new.specificity and overlaps(old.pattern, new.pattern):
            raise ValueError(f"{name}: the overloads for {show(old.pattern)} and {show(new.pattern)} tie")
    existing.append(new)


def show(pattern: Pattern) -> str:
    return str(pattern) if isinstance(pattern, Dtype) else pattern.value


def resolve(name: str, dtype: Dtype, target: Target, span: Span | None = None) -> Overload:
    """The most specific overload of `name` for `dtype` whose features the target has."""
    candidates = _OVERLOADS.get(name, [])
    usable = [o for o in candidates if matches(o.pattern, dtype) and o.features <= target.features]
    if not usable:
        tried = ", ".join(show(o.pattern) for o in candidates) or "none"
        raise TargetError(
            name,
            target.arch,
            f"no overload of {name} for {dtype} on {target.arch}; candidates: {tried}",
            "convert the operands to one of the candidate types, or use a target with the feature the overload needs",
            span=span,
        )
    return max(usable, key=lambda o: o.specificity)


def operations(expr: Expr) -> int:
    """The operations an expression emits: arithmetic, comparisons, selects, conversions and calls."""
    own = isinstance(expr, Binary | Unary | Math | Select | Cast | Intrinsic)
    return own + sum(operations(child) for child in children(expr))


def statements(helper: str) -> int:
    """The statements of a software sequence written as C text."""
    return max(helper.count(";"), 1)


def lower(overload: Overload, operands: Sequence[Expr], span: Span, target: Target) -> Built:
    """Build the IR of one resolved overload on `target`: a native instruction, a composed sequence, or a refusal."""
    match overload.lowering:
        case Instruction(row=row, native=native, result=result):
            entry = catalog.BY_NAME[row]
            if not entry.lowering(target):
                raise TargetError(
                    overload.name,
                    target.arch,
                    f"`{row}` has no lowering on {target.arch}",
                    "use a target with the packed SIMD instructions, or unpack the word and compute lane by lane",
                    span=span,
                )
            is_native = target.vendor in native
            count = 1 if is_native else statements(entry.helper_for(target.vendor))
            expr = Intrinsic(result or operands[0].type, row, tuple(operands), span=span)
            return Built(expr, is_native, count)
        case Expansion(build=build):
            expr = build(operands, span)
            return Built(expr, False, operations(expr))
        case refusal:  # the only lowering left is a Refusal
            raise TargetError(overload.name, target.arch, refusal.reason, refusal.instead, span=span)


def fallback_message(overload: Overload, built: Built, dtype: Dtype, target: Target) -> str:
    """The text of the software-fallback warning: the function, the operand type, the size, and where it is native."""
    vendors = sorted(overload.lowering.native) if isinstance(overload.lowering, Instruction) else []
    natives = [GPU_VENDORS[v] for v in vendors if v in GPU_VENDORS]
    where = f"native on {' and '.join(natives)}" if natives else "not native on any GPU target"
    return (
        f"{overload.name} on {dtype} has no native instruction on {target.arch}; "
        f"it is emitted as a {built.instructions}-instruction sequence; {where}"
    )

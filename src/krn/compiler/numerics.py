"""Call-site numerical choices, explained independently of compiler-wide options."""

from collections.abc import Iterator

from krn.facts import catalog
from krn.facts.fast_math import accuracy, relaxation
from krn.facts.targets import Feature, Target
from krn.runtime.settings import current_settings

from .dtypes import F32, F64
from .errors import PrecisionWarning, Span, TargetError
from .ir import Binary, Expr, Fn, Intrinsic, Kernel, Math, Op, Reduce, Reduction, slots
from .performance import warn_once
from .validity import children, walk

# The strict float32 operations a target may flush, by the names of `facts.targets.denormal_flushes`.
_OPERATIONS = {
    Op.ADD: "add",
    Op.SUB: "sub",
    Op.MUL: "mul",
    Op.DIV: "div",
    Op.LT: "compare",
    Op.LE: "compare",
    Op.GT: "compare",
    Op.GE: "compare",
    Op.EQ: "compare",
    Op.NE: "compare",
}
_FUNCTIONS = {Fn.SQRT: "sqrt", Fn.RSQRT: "rsqrt", Fn.FMA: "fma", Fn.MINIMUM: "minimum", Fn.MAXIMUM: "maximum"}


def expressions(kernel: Kernel) -> Iterator[Expr]:
    def visit(expression: Expr) -> Iterator[Expr]:
        yield expression
        for child in children(expression):
            yield from visit(child)

    return (found for statement in walk(kernel.body) for child in children(statement) for found in visit(child))


def fast_calls(kernel: Kernel) -> tuple[Intrinsic, ...]:
    calls = (found for found in expressions(kernel) if isinstance(found, Intrinsic))
    return tuple(call for call in calls if call.name.startswith("fast."))


def strict_operations(kernel: Kernel) -> tuple[tuple[str, Span], ...]:
    """Each float32 operation the kernel names that a target may flush, with the span of its expression."""
    found: list[tuple[str, Span]] = []
    for expression in expressions(kernel):
        match expression:
            case Binary(op=op, lhs=lhs) if lhs.dtype == F32 and op in _OPERATIONS:
                found.append((_OPERATIONS[op], expression.span))
            case Math(fn=fn) if expression.dtype == F32 and fn in _FUNCTIONS:
                found.append((_FUNCTIONS[fn], expression.span))
            case Reduce(op=Reduction.MEAN) if expression.dtype == F32:  # the mean's division is a float32 division
                found.append(("div", expression.span))
    return tuple(found)


def describe(call: Intrinsic, target: Target) -> str:
    name = call.name.removeprefix("fast.")
    lowering = catalog.spell(call.name, target, *("abc"[i] for i in range(len(call.args))), bits=call.dtype.bits)
    if target.vendor == "nvidia":
        instruction = {
            "sqrt": "sqrt.approx.f32",
            "rsqrt": "rsqrt.approx.f32",
            "reciprocal": "rcp.approx.f32",
            "exp2": "ex2.approx.f32",
            "tanh": "tanh.approx.f32",
        }.get(name)
        if instruction is not None:
            lowering += f" ({instruction})"
    return (
        f"precision is reduced: fast.math.{name} lowers to {lowering}: {accuracy(name, target)}; "
        f"{relaxation(name, target)}; strict alternative: math.{name}; "
        "the emulator computes it precisely"
    )


def explain(kernel: Kernel, target: Target) -> tuple[str, ...]:
    calls = {call.name: call for call in reversed(fast_calls(kernel))}
    return tuple(describe(call, target) for call in calls.values())


def float64_refusal(target: Target, span: Span | None) -> TargetError:
    """The refusal of a float64 value on a target without 64-bit floats, which Apple GPUs are."""
    return TargetError(
        "f64",
        target.arch,
        "Apple GPUs have no 64-bit floating point",
        "use f32, or run on a CPU or NVIDIA target",
        span=span,
    )


def refuse_float64(kernel: Kernel, target: Target) -> None:
    """Refuse the first float64 parameter, local or value on a target without 64-bit floats; a constant is refused where
    it is converted, since folding can remove it from the IR."""
    if target.has(Feature.FLOAT64):
        return
    sites = [(found.dtype, found.span) for found in expressions(kernel)]
    sites += [(var.dtype, var.span) for var in (*kernel.locals, *kernel.tiles)]
    sites += [(slot.dtype, kernel.span) for name, kind in kernel.params for slot in slots(name, kind)]
    sites += [(ref.type.dtype, kernel.span) for ref in kernel.shared]
    if any(dtype == F64 for dtype, _ in sites):
        raise float64_refusal(target, next(span for dtype, span in sites if dtype == F64))


def check(kernel: Kernel, target: Target) -> None:
    for call in fast_calls(kernel):
        warn_once(kernel.name, target.arch, call.name, describe(call, target), PrecisionWarning, call.span, kernel.span)
    if target.denormal_flushes and current_settings().warn.warn_denormal_flush:
        for operation, span in strict_operations(kernel):
            if operation in target.denormal_flushes:
                warn_once(
                    kernel.name,
                    target.arch,
                    f"denormal-flush:{operation}",
                    _denormal_message(operation, target),
                    PrecisionWarning,
                    span,
                    kernel.span,
                )


def _denormal_message(operation: str, target: Target) -> str:
    return (
        f"{operation} flushes float32 subnormal operands and results on {target.arch}, so a device run differs from "
        "the emulator for subnormal inputs, which the emulator computes exactly; keep values normal, "
        "or disable warn.warn_denormal_flush"
    )

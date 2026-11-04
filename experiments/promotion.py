"""Type promotion for kernel arithmetic: torch, the C rule and strict same-dtype, on the dtypes a kernel uses.

Needs torch, so run it with the torch interpreter. Run: python promotion.py <out.json>
"""

import json
import logging
import sys
from pathlib import Path

import torch

from experiments.paths import RESULTS

log = logging.getLogger("promotion")

INTS = {"u8": (8, False), "i8": (8, True), "i16": (16, True), "u16": (16, False),
        "i32": (32, True), "u32": (32, False), "i64": (64, True), "u64": (64, False)}  # fmt: skip
FLOATS = ("f16", "bf16", "f32", "f64")
DTYPES = {
    "u8": torch.uint8, "i8": torch.int8, "i16": torch.int16, "u16": torch.uint16,
    "i32": torch.int32, "u32": torch.uint32, "i64": torch.int64, "u64": torch.uint64,
    "f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64,
}  # fmt: skip
NAMES = {dtype: name for name, dtype in DTYPES.items()}


EXAMPLES = ("u8+u8", "u16+u16", "u32+i32", "u64+i64", "i32+u8", "u32+u8", "i64+u64")


def c_rule(a: str, b: str) -> str:
    """Usual arithmetic conversions on an LP64 target: narrower than int promotes to int, then rank and signedness."""

    def promote(name: str) -> str:
        return "i32" if INTS[name][0] < 32 else name

    a, b = promote(a), promote(b)
    (bits_a, signed_a), (bits_b, signed_b) = INTS[a], INTS[b]
    if signed_a == signed_b:
        return a if bits_a >= bits_b else b
    unsigned, signed = (b, a) if signed_a else (a, b)
    return unsigned if INTS[unsigned][0] >= INTS[signed][0] else signed


def lossless_rule(a: str, b: str) -> str:
    """Implicit conversion only when every value survives: the smallest integer type whose range holds both, else error.

    Same dtype stays itself, so u8 + u8 wraps in u8. u8 + i8 is i16, u32 + i32 is i64, and u64 + i64 has no home.
    """

    def span(name: str) -> tuple[int, int]:
        bits, signed = INTS[name]
        return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)

    if a == b:
        return a
    low, high = min(span(a)[0], span(b)[0]), max(span(a)[1], span(b)[1])
    holders = [n for n in INTS if span(n)[0] <= low and high <= span(n)[1]]
    return min(holders, key=lambda n: (INTS[n][0], INTS[n][1])) if holders else "error: no integer type holds both"


def torch_rule(a: str, b: str) -> str:
    try:
        return NAMES[torch.promote_types(DTYPES[a], DTYPES[b])]
    except RuntimeError as error:
        return f"error: {str(error)[:60]}"


def scalar_rule(name: str, literal: int | float) -> str:
    return NAMES[torch.result_type(torch.zeros(1, dtype=DTYPES[name]), literal)]


def executed(name: str, expression: str) -> str:
    """Run a one-line tensor expression on CPU and report the dtype and value, or why it cannot run."""
    x = torch.tensor([65535 if name == "u16" else 200], dtype=DTYPES[name])
    try:
        out = eval(expression, {"x": x, "torch": torch})
    except (RuntimeError, TypeError, OverflowError) as error:
        return f"cannot run: {str(error)[:70]}"
    return f"{NAMES[out.dtype]} {out.tolist()}"


def main(out: Path) -> None:
    ints = list(INTS)
    c_vs_torch = {f"{a}+{b}": (c_rule(a, b), torch_rule(a, b)) for a in ints for b in ints}
    differs = {k: v for k, v in c_vs_torch.items() if v[0] != v[1]}
    errors = {k: v for k, v in c_vs_torch.items() if v[1].startswith("error")}
    same_type = {a: (c_rule(a, a), torch_rule(a, a)) for a in ints}
    literals = {a: (scalar_rule(a, 1), scalar_rule(a, 0.5)) for a in ints}
    mixed_float = {f"{a}+{f}": torch_rule(a, f) for a in ("u8", "i32", "u32", "i64", "u64") for f in FLOATS}
    ran = {
        "u8 200 + 100 (same type)": executed("u8", "x + 100"),
        "u16 (x & 0x7FFF) - 1, the cutok dead-state idiom": executed("u16", "(x & 0x7FFF) - 1"),
        "dtype of a u16 tensor minus the literal 1": scalar_rule("u16", 1),
        "u32 + i32 tensors": executed("u32", "x + torch.tensor([1], dtype=torch.int32)"),
    }  # fmt: skip
    supported = [n for n in ints if n not in ("u16", "u32", "u64")]
    lossless = {f"{a}+{b}": lossless_rule(a, b) for a in ints for b in ints}
    on_supported = [(a, b) for a in supported for b in supported]
    agree = sum(lossless_rule(a, b) == torch_rule(a, b) for a, b in on_supported)
    result = {
        "lossless_agrees_with_torch_on_supported_subset": f"{agree} of {len(on_supported)}",
        "lossless_differs_from_c": sum(lossless[f"{a}+{b}"] != c_rule(a, b) for a in ints for b in ints),
        "lossless_needs_explicit_conversion": sorted(k for k, v in lossless.items() if v.startswith("error")),
        "lossless_widens_to_64_bit": sorted(k for k, v in lossless.items() if v in ("i64", "u64") and "64" not in k),
        "torch": torch.__version__,
        "integer_pairs": len(c_vs_torch),
        "torch_differs_from_c": len(differs),
        "torch_errors": errors,
        "same_type_c_vs_torch": same_type,
        "literal_int_and_float_result": literals,
        "int_with_float_torch": mixed_float,
        "differing_examples": {k: differs[k] for k in EXAMPLES if k in differs},
        "executed_on_cpu": ran,
    }  # fmt: skip
    log.info(json.dumps(result, indent=2))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "promotion.json")

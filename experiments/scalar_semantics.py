"""Strict same-dtype arithmetic against C promotion, on the one cutok site where they differ.

cutoken/stages/split/device/machine.py:61 builds the next DFA state as `(u16(encoded) & 0x7FFF) - 1` and stores it in an
i32 field that "is negative once the walk is dead". Run: python scalar_semantics.py <cutoken/src/cutoken>.
"""

import json
import logging
import sys
from pathlib import Path
from typing import TypedDict

from experiments.paths import RESULTS
from experiments.text import words

log = logging.getLogger("scalar_semantics")

NARROW = ("u8", "u16", "i8", "i16", "u8x4", "u16x2", "i8x4", "i16x2")
WIDE = ("u32", "i32", "u64", "i64")


def wrap(value: int, bits: int, *, signed: bool) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if signed and value >> (bits - 1) else value


def next_state_c(encoded: int) -> int:
    """C rule: u16 & int is int, and int - 1 is int."""
    return (encoded & 0x7FFF) - 1


def next_state_strict(encoded: int) -> int:
    """Strict rule: u16 & literal is u16, and u16 - 1 wraps in 16 bits."""
    return wrap((encoded & 0x7FFF) - 1, 16, signed=False)


def next_state_strict_widened(encoded: int) -> int:
    """Strict rule with the widening a port has to write: i32(u16(encoded) & 0x7FFF) - 1."""
    return wrap((encoded & 0x7FFF) - 1, 32, signed=True)


def exposure(device_root: Path) -> dict[str, int]:
    """Count type-name tokens in the device code, which is where the arithmetic rule matters."""
    text = "\n".join(p.read_text() for p in sorted(device_root.rglob("*.py")) if "device" in p.parts)
    return {name: words(text).count(name) for name in (*WIDE, *NARROW)}


class ScalarResult(TypedDict):
    inputs: int
    strict_differs_from_c: int
    differing_inputs: list[str]
    c_values_there: list[int]
    strict_values_there: list[int]
    dead_walks_lost_by_strict: int
    widened_differs_from_c: int
    exposure: dict[str, int]


def main(root: Path) -> ScalarResult:
    codes = range(1 << 16)
    c = [next_state_c(e) for e in codes]
    strict = [next_state_strict(e) for e in codes]
    widened = [next_state_strict_widened(e) for e in codes]
    differing = [e for e in codes if c[e] != strict[e]]
    result: ScalarResult = {
        "inputs": len(c),
        "strict_differs_from_c": len(differing),
        "differing_inputs": [hex(e) for e in differing],
        "c_values_there": sorted({c[e] for e in differing}),
        "strict_values_there": sorted({strict[e] for e in differing}),
        "dead_walks_lost_by_strict": sum(1 for e in differing if c[e] < 0 <= strict[e]),
        "widened_differs_from_c": sum(1 for e in codes if c[e] != widened[e]),
        "exposure": exposure(root),
    }
    log.info(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = main(Path(sys.argv[1]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    RESULTS.joinpath("scalar_semantics.json").write_text(json.dumps(out, indent=2) + "\n")

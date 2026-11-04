"""es-toolkit's scalar arithmetic and row helpers, with caller-owned storage and precomputed integer keys."""

from krn import Vector, atomic, kernel
from krn.frontend.types import Idx, f32, u32


@kernel
def min_by(keys: Vector[f32], start: u32 | int, stop: u32 | int) -> Idx:
    """Position of the first minimum key; an empty row returns `start` without reading it."""
    best = start
    for i in range(start + 1, stop):
        if keys[i] < keys[best]:
            best = i
    return best


@kernel
def max_by(keys: Vector[f32], start: u32 | int, stop: u32 | int) -> Idx:
    """Position of the first maximum key; read a parallel array at that position to obtain its element."""
    best = start
    for i in range(start + 1, stop):
        if keys[i] > keys[best]:
            best = i
    return best


@kernel
def count_by(keys: Vector[u32], start: u32 | int, stop: u32 | int, counts: Vector[u32]) -> None:
    """Add to caller-initialized bins, sized to the key space; integer atomics keep the result deterministic."""
    for i in range(start, stop):
        atomic.add(counts, keys[i], 1)

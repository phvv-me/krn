"""Kernels over packed words and their lanes, for tests of the packed types and the general functions."""

from pathlib import Path

import numpy as np

LANE = {"u8x4": np.uint8, "i8x4": np.int8, "u16x2": np.uint16, "i16x2": np.int16}
NUMPY = {
    "u8": "uint8", "i8": "int8", "u16": "uint16", "i16": "int16", "u32": "uint32", "i32": "int32", "f32": "float32",
}  # fmt: skip


def kernel(directory: Path, body: str, *, ty: str, out: str, extra: str = "") -> Path:
    """A one-line body over `a`, `b` of type `ty` into `out` of type `out`, one thread per element."""
    file = directory / "kernel.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        "from krn.math import abs_diff, add_sat, maximum, pack, unpack\n"
        f"@kernel\ndef k(a: Vector[T.{ty}], b: Vector[T.{ty}], out: Vector[T.{out}]) -> None:\n"
        f"    i = grid.thread_rank()\n    if i < out.shape[0]:\n        {body}\n{extra}"
    )
    return file


def lanes(words: np.ndarray, ty: str) -> np.ndarray:
    """The lanes of packed words as int64, shaped (count, lanes), low lane first."""
    count = 4 if ty.endswith("x4") else 2
    return words.astype(np.uint32).view(LANE[ty]).reshape(-1, count).astype(np.int64)


def words(values: np.ndarray, ty: str) -> np.ndarray:
    """Packed words from lane values, each wrapped to its lane width."""
    bits = 8 if ty.endswith("x4") else 16
    lane = np.uint8 if bits == 8 else np.uint16
    wrapped = (np.asarray(values, dtype=np.int64) % (1 << bits)).astype(lane)
    return np.ascontiguousarray(wrapped).view(np.uint32).ravel()


def random_words(rng: np.random.Generator, count: int) -> np.ndarray:
    return rng.integers(0, 2**32, count, dtype=np.uint64).astype(np.uint32)


def random_values(rng: np.random.Generator, ty: str, count: int) -> np.ndarray:
    """Values of a plain type, with both ends of its range included so saturation edges are hit."""
    if ty == "f32":
        return rng.standard_normal(count).astype(np.float32)
    info = np.iinfo(np.dtype(NUMPY[ty]))
    values = rng.integers(info.min, info.max, count, endpoint=True, dtype=np.int64)
    values[: min(4, count)] = (info.min, info.max, 0, -1 if info.min < 0 else info.max)[: min(4, count)]
    return values.astype(NUMPY[ty])

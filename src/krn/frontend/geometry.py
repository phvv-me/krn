"""Explicit launch overrides: how many threads and blocks a kernel runs with `launch=`."""

from dataclasses import dataclass

from krn.arithmetic import ceildiv


@dataclass(frozen=True)
class Items:
    """Launch one thread per item and let the device profile choose block and grid size."""

    count: int


@dataclass(frozen=True)
class Blocks:
    """Launch an explicit grid of blocks."""

    x: int
    y: int = 1
    z: int = 1
    threads: int | None = None


@dataclass(frozen=True)
class Tiles:
    """Launch one block per tile of an output: `Tiles(c.shape, (bm, bn))` covers a matrix with ceiling division.

    The last axis runs along x, the one before it along y, so a block's `block.tile(bm, bn)` is the tile it owns.
    """

    shape: tuple[int, ...]
    tile: tuple[int, ...]

    @property
    def blocks(self) -> Blocks:
        counts = [ceildiv(extent, size) for extent, size in zip(self.shape, self.tile, strict=True)]
        return Blocks(*reversed(counts))


type Geometry = Items | Blocks | Tiles

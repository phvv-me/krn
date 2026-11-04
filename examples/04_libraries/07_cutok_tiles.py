"""cutok's Tiles rebuilt: spans cut into tiles, a warp to a span and a lane to a tile

Three steps on structs of arrays, the shape of a tokenizer's device code: `count` (a thread to a span), `split` (a
warp to a span, its lanes striding over the span's tiles) and `place` (a thread to a span again). The scan between
`count` and `split` is NumPy here. The rebuild spelled by hand what the language does not give yet: a warp to an
item (`grid.thread_rank() // warp.size()` and `warp.lane()`), ceiling division on unsigned integers because signed
`//` is refused, and `min(a, b)` as a conditional expression. `count` and `place` infer their launch with
`items(n)`, while `split` keeps `launch=Items(n * size)` because one warp per span is part of its algorithm.

Level: 4
Topics: structs of arrays, warp per item, real application
Needs: 03_memory_and_sync/05_warp_shuffles
"""

import asyncio
import logging

import numpy as np

from krn import Items, T, Tensor, grid, host, items, kernel, struct, warp
from krn.compiler.api import CPUDevice
from krn.emulate import emulate

log = logging.getLogger("cutok_tiles")


@struct
class Spans:
    starts: Tensor[T.u32, int]
    ends: Tensor[T.u32, int]

    def length(self, document: T.u32) -> T.u32:
        return self.ends[document] - self.starts[document]


@struct
class Tiles:
    starts: Tensor[T.u32, int]
    ends: Tensor[T.u32, int]
    documents: Tensor[T.u32, int]
    bases: Tensor[T.u32, int]
    tile_bytes: T.u32

    def needed(self, spans: Spans, document: T.u32) -> T.u32:
        """How many tiles `document` needs: the ceiling of its length over the tile."""
        return (spans.length(document) + self.tile_bytes - 1) // self.tile_bytes


@kernel
def count(spans: Spans, tiles: Tiles, counts: Tensor[T.u32, int]) -> None:
    """A thread to a span: how many tiles it needs."""
    for document in items(counts.shape[0]):
        counts[document] = tiles.needed(spans, T.u32(document))


@kernel
def split(spans: Spans, tiles: Tiles) -> None:
    """A warp to a span and a lane to a tile: a long span costs no serial chain."""
    document = T.u32(grid.thread_rank() // warp.size())
    if document + 1 < tiles.bases.shape[0]:
        base = tiles.bases[document]
        total = tiles.bases[document + 1] - base
        for index in range(warp.lane(), total, warp.size()):
            start = spans.starts[document] + index * tiles.tile_bytes
            end = start + tiles.tile_bytes
            tiles.starts[base + index] = start
            tiles.ends[base + index] = end if end < spans.ends[document] else spans.ends[document]
            tiles.documents[base + index] = document


@kernel
def place(tiles: Tiles, offsets: Tensor[T.u32, int], stride: T.u32, limit: T.u32, out: Tensor[T.u32, int]) -> None:
    """Where each span's first tile slot begins in the output, kept within `limit`."""
    for document in items(out.shape[0]):
        where = offsets[tiles.bases[document] * stride]
        out[document] = where if where < limit else limit


def expected(starts: np.ndarray, ends: np.ndarray, tile_bytes: int) -> tuple[np.ndarray, ...]:
    """The tiles by their definition, one span at a time."""
    counts = -(-(ends.astype(np.int64) - starts) // tile_bytes)  # signed, or the negation of an unsigned length wraps
    first = np.concatenate([[0], np.cumsum(counts)])
    documents = np.repeat(np.arange(len(starts)), counts)
    index = np.arange(first[-1]) - first[documents]
    tile_starts = starts[documents] + index * tile_bytes
    return counts, first, tile_starts, np.minimum(tile_starts + tile_bytes, ends[documents]), documents


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(3)
    n, tile_bytes = 37, 4096
    lengths = rng.integers(0, 20000, n).astype(np.uint32)
    lengths[::7] = 0  # empty spans own no tile
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.uint32)
    ends = starts + lengths
    counts_want, bases_want, starts_want, ends_want, documents_want = expected(starts, ends, tile_bytes)

    spans = Spans(device.empty(T.u32, n), device.empty(T.u32, n))
    np.asarray(spans.starts)[:], np.asarray(spans.ends)[:] = starts, ends
    counts = device.empty(T.u32, n)
    bases = device.empty(T.u32, n + 1)
    slots = int(bases_want[-1])
    tiles = Tiles(
        device.empty(T.u32, slots), device.empty(T.u32, slots), device.empty(T.u32, slots), bases, T.u32(tile_bytes)
    )

    count(spans, tiles, counts)
    await device.synchronize()
    np.testing.assert_array_equal(counts.array, counts_want)
    bases.array[:] = bases_want  # the scan, on the host

    size = device.target.warp_size
    split(spans, tiles, launch=Items(n * size))
    await device.synchronize()
    for name, want in (("starts", starts_want), ("ends", ends_want), ("documents", documents_want)):
        np.testing.assert_array_equal(np.asarray(getattr(tiles, name)), want, err_msg=name)

    offsets, out = device.empty(T.u32, slots * 3 + 1), device.empty(T.u32, n)
    offsets.array[:] = np.arange(offsets.shape[0], dtype=np.uint32) * 5
    place(tiles, offsets, T.u32(3), T.u32(300), out)
    await device.synchronize()
    np.testing.assert_array_equal(out.array, np.minimum(offsets.array[bases_want[:-1] * 3], 300))

    # the emulator is the oracle: a second run through it must agree with the backend
    again = Tiles(
        device.empty(T.u32, slots), device.empty(T.u32, slots), device.empty(T.u32, slots), bases, T.u32(tile_bytes)
    )
    emulate(split, Items(n * size), spans, again, target=device.target)
    np.testing.assert_array_equal(np.asarray(again.starts), np.asarray(tiles.starts))
    np.testing.assert_array_equal(np.asarray(again.ends), np.asarray(tiles.ends))
    log.info("%d spans cut into %d tiles of %d bytes; backend, emulator and numpy agree", n, slots, tile_bytes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())

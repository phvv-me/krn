"""cutok's `Tiles`, rebuilt in `examples/04_libraries/07_cutok_tiles`, run on the real GPUs and compared with the
emulator.

The three kernels take structs of arrays, a warp to a span and a lane to a tile, the shape of cutok's own device code.
They skip where there is no CUDA card or no Metal GPU."""

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Items
from krn.facts.targets import Target
from krn.runtime.cpu import Value
from tests.conftest import EXAMPLE_ROOT
from tests.helpers.devices import CUDARun, MetalRun, agree, compare

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

FILE = EXAMPLE_ROOT / "04_libraries" / "07_cutok_tiles.py"
if not FILE.is_file():
    pytest.skip("research examples are absent; examples have not migrated", allow_module_level=True)
spec = importlib.util.spec_from_file_location("cutok_tiles_example", FILE)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
TILE = 4096


def problem(warp_size: int) -> dict[str, tuple[list[Value], Items]]:
    """The arguments and launch of each kernel for a random set of spans (empty ones included)."""
    rng = np.random.default_rng(3)
    n = 37
    lengths = rng.integers(0, 20000, n).astype(np.uint32)
    lengths[::7] = 0
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.uint32)
    ends = starts + lengths
    _, bases, *_ = example.expected(starts, ends, TILE)
    slots = int(bases[-1])
    spans = {"starts": starts, "ends": ends}

    def tiles() -> dict[str, Value]:
        zeros = [np.zeros(slots, np.uint32) for _ in range(3)]
        return {
            "starts": zeros[0],
            "ends": zeros[1],
            "documents": zeros[2],
            "bases": bases.astype(np.uint32),
            "tile_bytes": TILE,
        }

    offsets = (np.arange(slots * 3 + 1) * 5).astype(np.uint32)
    return {
        "count": ([spans, tiles(), np.zeros(n, np.uint32)], Items(n)),
        "split": ([spans, tiles()], Items(n * warp_size)),
        "place": ([tiles(), offsets, 3, 300, np.zeros(n, np.uint32)], Items(n)),
    }


@pytest.mark.parametrize("name", ["count", "split", "place"])
def test_the_card_agrees_with_the_emulator(name: str, gpu: CUDADevice) -> None:
    card = gpu
    arguments, launch = problem(card.target.warp_size)[name]
    oracle, actual = compare(card.target, lambda kernel: CUDARun(card, kernel), name, launch, arguments, FILE)
    agree(oracle, actual)


@pytest.mark.parametrize("name", ["count", "split", "place"])
def test_the_metal_gpu_agrees_with_the_emulator(name: str) -> None:
    metal = pytest.importorskip("krn.runtime.metal")
    try:
        gpu = metal.MetalDevice(Target.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))
    arguments, launch = problem(gpu.target.warp_size)[name]
    oracle, actual = compare(gpu.target, lambda kernel: MetalRun(gpu, kernel), name, launch, arguments, FILE)
    agree(oracle, actual)

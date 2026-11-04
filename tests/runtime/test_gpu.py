"""Kernels that were only ever compiled, run on a CUDA card through the backend and compared with the emulator, the
oracle for the IR.

They skip where there is no card. An atomic whose result depends on the order the threads arrive in is compared by what
that order cannot change.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Items, Tiles
from krn.compiler.api import capture
from krn.frontend.geometry import Geometry
from krn.runtime.cpu import Value
from tests.compiler.test_validity import CASES as RAGGED
from tests.conftest import KERNELS, SKETCH
from tests.emulator.test_emulate import CASES
from tests.helpers.devices import FILE, CUDARun, agree, compare, cooperative

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

RNG = np.random.default_rng(13)


def twice(
    gpu: CUDADevice, name: str, launch: Geometry, arguments: list[Value], file=FILE
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """`compare` on the CUDA card."""
    return compare(gpu.nvidia, lambda kernel: CUDARun(gpu, kernel), name, launch, arguments, file)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_the_card_agrees_with_the_emulator(gpu: CUDADevice, case) -> None:
    arrays = list(case.make())
    count = len(arrays[0]) if case.name != "take_lock" else 64
    oracle, actual = twice(gpu, case.name, Items(count), arrays)
    agree(oracle, actual, case.outputs)


@pytest.mark.parametrize("limit", [0, 3, 7])
def test_a_wrapping_counter_ends_where_the_emulator_does(gpu: CUDADevice, limit: int) -> None:
    seen, counter = np.zeros(20, np.uint32), np.zeros(1, np.uint32)
    oracle, actual = twice(gpu, "wrapping_counter", Items(20), [limit, counter, seen])
    assert actual[0][0] == oracle[0][0]
    assert sorted(actual[1]) == sorted(oracle[1])  # the order of arrival differs, the multiset of old values cannot


def test_warp_shuffle_sums_each_warp(gpu: CUDADevice) -> None:
    x = RNG.standard_normal(256).astype(np.float32)
    oracle, actual = twice(gpu, "warp_sum", Items(256), [x, np.zeros(8, np.float32)])
    agree(oracle, actual, (1,))
    assert np.allclose(actual[1], x.reshape(8, 32).sum(axis=1), rtol=1e-5)


def test_ballot_and_vote_see_the_whole_warp(gpu: CUDADevice) -> None:
    x = np.where(np.arange(256) % 3 == 0, 1, -1).astype(np.int32)
    x[64:96] = 5
    oracle, actual = twice(gpu, "votes", Items(256), [x, np.zeros(8, np.uint64), np.zeros(8, np.int32)])
    agree(oracle, actual, (1, 2))
    assert actual[2].tolist() == [int(w == 2) for w in range(8)]


def test_block_barriers_count_and_vote(gpu: CUDADevice) -> None:
    x = np.where(np.arange(512) % 5 == 0, 1, -1).astype(np.int32)
    x[256:] = -1
    oracle, actual = twice(gpu, "block_votes", Items(512), [x, np.zeros(2, np.uint32), np.zeros(2, np.int32)])
    agree(oracle, actual, (1, 2))
    assert actual[1].tolist() == [52, 0]


def test_a_fence_and_a_barrier_order_the_writes(gpu: CUDADevice) -> None:
    oracle, actual = twice(gpu, "ordered", Items(256), [np.zeros(256, np.float32), np.zeros(1, np.int32)])
    agree(oracle, actual)
    assert actual[0].tolist() == [1.0] * 256
    assert actual[1].tolist() == [1]


def test_a_tile_kernel_covers_every_column_with_a_cooperative_block(gpu: CUDADevice) -> None:
    x = RNG.standard_normal((20, 40)).astype(np.float32)
    launch = cooperative(Tiles(x.shape, (8, 16)))
    oracle, actual = twice(gpu, "tiles_wide", launch, [x, np.full((20, 40), 99.0, np.float32)])
    agree(oracle, actual, (1,))
    assert np.allclose(actual[1], x + 1.0)


def test_the_fp16_gemm_runs_its_cooperative_loops_on_the_card(gpu: CUDADevice) -> None:
    a = RNG.standard_normal((20, 24)).astype(np.float16)
    b = RNG.standard_normal((24, 17)).astype(np.float16)
    c = np.zeros((20, 17), np.float32)
    launch = cooperative(Tiles(c.shape, (16, 16)))
    kernel = capture(
        SKETCH / "gemm.py",
        "gemm",
        consts={"t": {"bm": 16, "bn": 16, "bk": 8}},
        target=gpu.nvidia,
    )
    CUDARun(gpu, kernel).run(launch, [a, b, c, {}])
    assert np.allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("case", RAGGED, ids=lambda c: c.name)
def test_ragged_tiles_skip_their_invalid_lanes_on_the_card(gpu: CUDADevice, case) -> None:
    """The validity intervals of the CUDA lowering against the per-lane masks of the emulator, cooperative blocks."""
    arguments = case.arguments()
    oracle, actual = twice(gpu, case.name, cooperative(case.launch), arguments, KERNELS / "ragged.py")
    agree(oracle, actual)
    for index, want in case.expected(arguments):
        assert np.allclose(actual[index], want, rtol=2e-5, atol=1e-6), case.name

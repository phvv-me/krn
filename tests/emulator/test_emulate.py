"""The emulator against the C backend and its debugging promises: breakpoints, memory and assert checks, collectives."""

import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from krn import Items, Tiles, targets
from krn.compiler.api import CaptureError, capture
from krn.emulator.emulate import Deadlock, build
from krn.facts.targets import Target
from tests.conftest import KERNELS, SKETCH, build_kernel

FILE = KERNELS / "intrinsics.py"
RNG = np.random.default_rng(11)
GPU = Target.of("sm_90a")


def run(
    name: str,
    launch: Items | Tiles,
    *arrays: np.ndarray,
    file: Path = FILE,
    consts: dict | None = None,
) -> None:
    build(capture(file, name, consts=consts, target=GPU), GPU).run(launch, list(arrays))


def line_of(text: str, file: Path = FILE) -> int:
    return next(n for n, line in enumerate(file.read_text().splitlines(), 1) if text in line)


@dataclass(frozen=True)
class Case:
    name: str
    make: Callable[[], tuple[np.ndarray, ...]]
    outputs: tuple[int, ...]


CASES = [
    Case(
        "histogram",
        lambda: (RNG.integers(0, 16, 700, dtype=np.int32), np.zeros(16, np.int32)),
        (1,),
    ),
    Case(
        "largest",
        lambda: (RNG.standard_normal(300).astype(np.float32), np.full(1, -np.inf, np.float32)),
        (1,),
    ),
    Case(
        "smallest",
        lambda: (
            RNG.integers(-(1 << 40), 1 << 40, 300, dtype=np.int64),
            np.full(1, 1 << 50, np.int64),
        ),
        (1,),
    ),
    Case("take_lock", lambda: (np.zeros(1, np.int32), np.zeros(1, np.int32)), (0, 1)),
    Case(
        "bit_tricks",
        lambda: (
            RNG.integers(0, 1 << 32, 200, dtype=np.uint32),
            *(np.zeros(200, np.uint32) for _ in range(4)),
        ),
        (1, 2, 3, 4),
    ),
    Case(
        "math_helpers",
        lambda: (RNG.uniform(0.1, 50, 400).astype(np.float32), np.zeros(400, np.float32)),
        (1,),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_emulator_agrees_with_the_c_backend(case: Case) -> None:
    emulated, compiled = case.make(), None
    compiled = tuple(a.copy() for a in emulated)
    count = len(emulated[0]) if case.name != "take_lock" else 64
    run(case.name, Items(count), *emulated)
    build_kernel("intrinsics", case.name)(*compiled, launch=Items(count))
    for index in case.outputs:
        assert np.allclose(emulated[index], compiled[index], rtol=1e-5), (case.name, index)


def test_gemm_tile_kernel_agrees_with_numpy() -> None:
    a, b = (
        RNG.standard_normal((20, 24)).astype(np.float16),
        RNG.standard_normal((24, 17)).astype(np.float16),
    )
    c = np.zeros((20, 17), np.float32)
    launch = Tiles(c.shape, (16, 16))
    build(
        capture(SKETCH / "gemm.py", "gemm", consts={"t": {"bm": 16, "bn": 16, "bk": 8}}, target=GPU),
        GPU,
    ).run(launch, [a, b, c, {}])
    assert np.allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-3, atol=1e-3)


def test_a_tile_grid_wider_than_two_blocks_covers_every_column() -> None:
    """Block ids above 1 must survive the cast to a Python int; a `bool()` there passes any two-block test."""
    x = RNG.standard_normal((20, 40)).astype(np.float32)
    out = np.full((20, 40), 99.0, np.float32)
    run("tiles_wide", Tiles(out.shape, (8, 16)), x, out)
    assert np.allclose(out, x + 1.0)


def test_warp_shuffle_sums_each_warp() -> None:
    x = RNG.standard_normal(256).astype(np.float32)
    out = np.zeros(8, np.float32)
    run("warp_sum", Items(256), x, out)
    assert np.allclose(out, x.reshape(8, 32).sum(axis=1), rtol=1e-5)


def test_ballot_and_vote_see_the_whole_warp() -> None:
    x = np.where(np.arange(256) % 3 == 0, 1, -1).astype(np.int32)
    x[64:96] = 5
    ballots, everyone = np.zeros(8, np.uint64), np.zeros(8, np.int32)
    run("votes", Items(256), x, ballots, everyone)
    expected = [sum(1 << lane for lane in range(32) if x[w * 32 + lane] > 0) for w in range(8)]
    assert ballots.tolist() == expected
    assert everyone.tolist() == [int(w == 2) for w in range(8)]


def test_block_barriers_count_and_vote() -> None:
    x = np.where(np.arange(512) % 5 == 0, 1, -1).astype(np.int32)
    x[256:] = -1
    counts, anyone = np.zeros(2, np.uint32), np.zeros(2, np.int32)
    run("block_votes", Items(512), x, counts, anyone)
    assert counts.tolist() == [52, 0]
    assert anyone.tolist() == [1, 0]


def test_a_breakpoint_stops_in_the_kernel_source_with_the_threads_locals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stops: list[tuple[str, int, dict[str, object]]] = []

    def hook[A, K](*_args: A, **_kwargs: K) -> None:
        frame = sys._getframe(1)
        stops.append((frame.f_code.co_filename, frame.f_lineno, dict(frame.f_locals)))

    monkeypatch.setattr(sys, "breakpointhook", hook)
    x, out = np.array([3, 4, 5], np.int32), np.zeros(3, np.int32)
    run("debug_aids", Items(3), x, out)
    assert [(f, n) for f, n, _ in stops] == [(str(FILE), line_of("breakpoint()"))] * 3
    assert [locals_["i"] for _, _, locals_ in stops] == [0, 1, 2]
    assert [locals_["_thread_id"] for _, _, locals_ in stops] == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert out.tolist() == [3, 4, 5]


def test_a_failed_assert_names_the_line_the_block_and_the_thread() -> None:
    x, out = np.array([1, 2, -3, 4], np.int32), np.zeros(4, np.int32)
    with pytest.raises(AssertionError, match=r"intrinsics.py:\d+:\d+: assertion failed: .*(block 0, thread 2)") as stop:
        run("debug_aids", Items(4), x, out)
    assert f":{line_of('assert x[i]')}:" in str(stop.value)


def test_an_out_of_bounds_read_raises_at_the_line_that_made_it() -> None:
    x, out = np.arange(4, dtype=np.float32), np.zeros(4, np.float32)
    with pytest.raises(IndexError, match=r"read of `x`\[4\] is outside shape \(4,\)") as stop:
        run("reads_past_the_end", Items(4), x, out)
    frames = [f for f in traceback.extract_tb(stop.value.__traceback__) if f.filename == str(FILE)]
    assert frames[-1].lineno == line_of("out[i] = x[i + 1]")
    assert frames[-1].line == "out[i] = x[i + 1]"


def test_a_collective_that_half_the_block_skips_is_a_deadlock_naming_the_waiters() -> None:
    with pytest.raises(Deadlock, match="no thread can continue"):
        run("split_barrier", Items(32), np.zeros(32, np.int32))


def test_the_c_backend_refuses_what_it_cannot_run_sequentially() -> None:
    with pytest.raises(CaptureError, match="needs WARP_SHUFFLE"):
        build_kernel("intrinsics", "warp_sum")
    assert targets.CPU.of("avx512").vendor == "cpu"

"""The intrinsic vocabulary on both backends: the C backend runs what is sequential, NVRTC compiles everything, and the
reference semantics of `krn.frontend.atomic` and the catalog are the oracle."""

import numpy as np
import pytest

from krn import Items, catalog, targets
from krn.compiler.api import CaptureError, capture, emit
from krn.frontend.atomic import AtomicOp, Strategy, plan, reference
from krn.runtime.nvrtc import Nvidia as Nvrtc
from tests.conftest import KERNELS, build_kernel

FILE = KERNELS / "intrinsics.py"
GPU = ("sm_90a", "sm_120a")
UNITS = [
    "histogram", "largest", "smallest", "wrapping_counter", "take_lock", "bit_tricks", "math_helpers", "warp_sum",
    "votes", "ordered", "debug_aids", "group_ranks", "segmented_shuffles", "lane_masks", "masked_groups",
    "matching_lanes",
]  # fmt: skip
RNG = np.random.default_rng(7)


def test_histogram_matches_bincount() -> None:
    values = RNG.integers(0, 16, 1000, dtype=np.int32)
    bins = np.zeros(16, dtype=np.int32)
    build_kernel("intrinsics", "histogram")(values, bins, launch=Items(len(values)))
    assert (bins == np.bincount(values, minlength=16)).all()


def test_atomic_max_of_floats_and_min_of_int64() -> None:
    x = RNG.standard_normal(500).astype(np.float32)
    best = np.full(1, -np.inf, dtype=np.float32)
    build_kernel("intrinsics", "largest")(x, best, launch=Items(len(x)))
    assert best[0] == x.max()
    y = RNG.integers(-(1 << 40), 1 << 40, 300, dtype=np.int64)
    low = np.full(1, 1 << 50, dtype=np.int64)
    build_kernel("intrinsics", "smallest")(y, low, launch=Items(len(y)))
    assert low[0] == y.min()


@pytest.mark.parametrize("limit", [0, 3, 7])
def test_wrapping_counter_follows_the_reference(limit: int) -> None:
    seen, counter = np.zeros(20, dtype=np.uint32), np.zeros(1, dtype=np.uint32)
    build_kernel("intrinsics", "wrapping_counter")(limit, counter, seen, launch=Items(20))
    state, expected = 0, []
    for _ in range(20):
        expected.append(state)
        state = reference(AtomicOp.INC, state, limit)
    assert seen.tolist() == expected
    assert counter[0] == state


def test_compare_and_swap_gives_the_lock_to_one_thread() -> None:
    lock, winners = np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)
    build_kernel("intrinsics", "take_lock")(lock, winners, launch=Items(64))
    assert (lock[0], winners[0]) == (1, 1)


def test_bit_tricks_match_python_integers() -> None:
    x = np.concatenate(
        [
            RNG.integers(0, 1 << 32, 200, dtype=np.uint32),
            np.array([0, 1, 1 << 31], dtype=np.uint32),
        ]
    )
    ones, flipped, zeros, low = (np.zeros_like(x) for _ in range(4))
    build_kernel("intrinsics", "bit_tricks")(x, ones, flipped, zeros, low, launch=Items(len(x)))
    values = [int(v) for v in x]
    assert ones.tolist() == [v.bit_count() for v in values]
    assert flipped.tolist() == [int(f"{v:032b}"[::-1], 2) for v in values]
    assert zeros.tolist() == [32 - v.bit_length() for v in values]
    assert low.tolist() == [(v & -v).bit_length() for v in values]


def test_math_helpers_match_numpy() -> None:
    x = RNG.uniform(0.1, 50, 400).astype(np.float32)
    out = np.zeros_like(x)
    build_kernel("intrinsics", "math_helpers")(x, out, launch=Items(len(x)))
    expected = (
        x.astype(np.float64) * 2 + 1 + np.cbrt(x.astype(np.float64)) + np.log10(x.astype(np.float64)) + np.trunc(x)
    )
    assert np.allclose(out, expected, rtol=1e-5)


@pytest.mark.parametrize("arch", GPU)
@pytest.mark.parametrize("name", UNITS)
def test_unit_is_header_free_and_compiles(nvrtc: Nvrtc, name: str, arch: str) -> None:
    target = targets.Target.of(arch)
    source = emit(capture(FILE, name, target=target), target).source
    assert "#include" not in source
    assert nvrtc.compile(source, arch, "cubin").data[:4] == b"\x7fELF"


def test_float_max_is_a_compare_and_swap_loop_and_integer_add_is_native() -> None:
    target = targets.Target.of("sm_90a")
    loop = emit(capture(FILE, "largest", target=target), target).source
    native = emit(capture(FILE, "histogram", target=target), target).source
    assert "atomicCAS" in loop and "__float_as_uint" in loop
    assert "atomicAdd(p, v)" in native
    assert plan(AtomicOp.MAX, "f32", target).strategy is Strategy.CAS_LOOP
    assert plan(AtomicOp.ADD, "f32", target).strategy is Strategy.CAS_LOOP
    assert plan(AtomicOp.ADD, "i32", target).strategy is Strategy.NATIVE


def test_a_target_without_the_feature_refuses_at_the_users_line() -> None:
    with pytest.raises(CaptureError, match=r"intrinsics.py:\d+:\d+: error: .*needs WARP_SHUFFLE"):
        capture(FILE, "warp_sum", target=targets.Target.of("avx512"))
    with pytest.raises(CaptureError, match="apple_m2 has no 64-bit atomics"):
        capture(FILE, "smallest", target=targets.Target.of("apple_m2"))


def test_the_c_backend_refuses_a_barrier_it_cannot_honor() -> None:
    with pytest.raises(CaptureError, match="runs the threads of a block one after the other"):
        capture(FILE, "ordered", target=targets.Target.of("avx512"))


def test_a_dtype_outside_the_table_is_refused_with_the_list_of_accepted_ones() -> None:
    chosen = plan(AtomicOp.INC, "i32", targets.Target.of("sm_90a"))
    assert chosen.strategy is Strategy.REFUSED
    assert "takes ['u32', 'u64']" in chosen.reason


def test_lineinfo_names_the_python_file_in_the_source_and_in_the_ptx(nvrtc: Nvrtc) -> None:
    target = targets.Target.of("sm_90a")
    kernel = capture(FILE, "histogram", target=target)
    assert "#line" not in emit(kernel, target, lineinfo=False).source
    source = emit(kernel, target, lineinfo=True).source
    line = next(n for n, text in enumerate(FILE.read_text().splitlines(), 1) if "atomic.add(bins" in text)
    assert f'#line {line} "{FILE}"' in source
    ptx = nvrtc.compile(source, "sm_90a", "ptx", ("--generate-line-info",)).data.decode()
    assert FILE.name in ptx
    assert ".loc" in ptx


def test_the_catalog_answers_every_name_of_numba_cuda_mlir() -> None:
    assert catalog.missing() == []
    assert catalog.spell("warp.ballot", targets.Target.of("sm_90a"), "p", "m") == (
        "(unsigned long long)__ballot_sync((unsigned)(m), p)"
    )
    assert catalog.spell("popcount", targets.Target.of("gfx942"), "x", bits=64) == "__popcll(x)"
    with pytest.raises(LookupError, match=r"`warp\.ballot` has no lowering for avx512"):
        catalog.spell("warp.ballot", targets.Target.of("avx512"), "p", "m")

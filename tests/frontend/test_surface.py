"""The namespaced surface against numpy: block.tile, pipelined slices, AUTO, templates, selectors and target facts."""

import asyncio
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from krn import AUTO, Auto, Blocks, Items, T, Tiles, host, targets
from krn.compiler.api import CaptureError, CPUDevice, capture
from krn.compiler.dtypes import F16, F32, F64, Dtype
from krn.frontend.numerics import FloatingPoint, Scalar, Selector
from tests.conftest import build_kernel, kernel_path
from tests.kernels.surface import Tiling, gemm_generic, gemm_template

RNG = np.random.default_rng(1)
SURFACE = kernel_path("surface")
NUMPY: dict[type[Scalar], type[np.generic]] = {
    T.f16: np.float16,
    T.f32: np.float32,
    T.f64: np.float64,
}


@pytest.fixture(scope="module")
def device() -> CPUDevice:
    current = host.Device.current()
    assert isinstance(current, CPUDevice)
    return current


def settle(device: CPUDevice) -> None:
    device.default_stream.record().wait(60)


def test_auto_is_frozen_and_compares_by_identity() -> None:
    assert repr(AUTO) == "AUTO"
    assert AUTO is Tiling().stages
    assert Auto() != AUTO
    assert len({AUTO, AUTO, Auto()}) == 2
    with pytest.raises((FrozenInstanceError, TypeError)):
        AUTO.__setattr__("choice", 1)


def test_one_tile_size_walks_x_and_two_sizes_are_rows_then_columns() -> None:
    x = RNG.standard_normal((10, 16), dtype=np.float32)
    y = np.zeros_like(x)
    build_kernel("surface", "doubled_rows", {"t": {"rows": 4, "cols": 16}})(x, y, {}, launch=Blocks(3))
    np.testing.assert_array_equal(y, x * 2)
    x = RNG.standard_normal((9, 21), dtype=np.float32)
    y = np.full_like(x, np.nan)
    build_kernel("surface", "corner_tile")(x, y, launch=Tiles(x.shape, (4, 8)))
    np.testing.assert_array_equal(y, x + 1)


@pytest.mark.parametrize("k", [1, 4, 5, 33, 64])
def test_pipelined_yields_the_tiles_of_the_axis_and_the_last_may_be_ragged(k: int) -> None:
    a, b = (
        RNG.standard_normal((20, k)).astype(np.float16),
        RNG.standard_normal((k, 13)).astype(np.float16),
    )
    c = np.full((20, 13), np.nan, dtype=np.float32)
    build_kernel("surface", "gemm", {"t": {"bm": 8, "bn": 8, "bk": 4}})(a, b, c, {}, launch=Tiles(c.shape, (8, 8)))
    np.testing.assert_allclose(c, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-4, atol=1e-4)


def test_stages_is_a_request_the_cpu_rejects_and_auto_is_the_default() -> None:
    a, b = (
        RNG.standard_normal((24, 40)).astype(np.float16),
        RNG.standard_normal((40, 24)).astype(np.float16),
    )
    expected = a.astype(np.float32) @ b.astype(np.float32)
    for stages in (1, 3):
        with pytest.raises(CaptureError, match="pipelined\\(stages\\)"):
            build_kernel("surface", "gemm", {"t": {"bm": 8, "bn": 8, "bk": 16, "stages": stages}})
    c = np.full((24, 24), np.nan, dtype=np.float32)
    build_kernel("surface", "gemm", {"t": {"bm": 8, "bn": 8, "bk": 16}})(a, b, c, {}, launch=Tiles(c.shape, (8, 8)))
    np.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)


def test_a_schedule_the_target_cannot_run_is_an_error_not_a_hint() -> None:
    with pytest.raises(CaptureError, match="WarpSpecialized needs BULK_COPY, which target cpu lacks"):
        capture(SURFACE, "gemm_warp_specialized", target=targets.Target.of("avx512"))
    with pytest.raises(CaptureError, match="WarpSpecialized needs BULK_COPY, which target nvidia lacks"):
        capture(SURFACE, "gemm_warp_specialized", target=targets.Target.of("sm_80"))
    with pytest.raises(CaptureError, match="no implemented WarpSpecialized schedule lowering"):
        capture(SURFACE, "gemm_warp_specialized", target=targets.Target.of("sm_90a"))


def test_the_pipelined_loop_is_a_plain_loop_over_the_tile_starts() -> None:
    source = build_kernel("surface", "gemm", {"t": {"bm": 8, "bn": 8, "bk": 4}}).unit.source
    assert "for (long long ks_start = 0ll; ks_start < a_n1; ks_start += 4)" in source


def test_a_template_kernel_runs_through_the_runtime_with_its_configuration_in_the_brackets(
    device: CPUDevice,
) -> None:
    m, n, k = 37, 29, 45
    a, b = device.empty(T.f16, m, k), device.empty(T.f16, k, n)
    a.array[:], b.array[:] = RNG.standard_normal((m, k)), RNG.standard_normal((k, n))
    expected = a.array.astype(np.float32) @ b.array.astype(np.float32)
    for tiling in (Tiling(), Tiling(bm=8, bn=16, bk=8), Tiling(bm=32, bn=32, bk=64)):
        c = device.empty(T.f32, m, n)
        c.array[:] = np.nan
        gemm_template[tiling](a, b, c)
        settle(device)
        np.testing.assert_allclose(c.array, expected, rtol=1e-4, atol=1e-4)
    launcher = device.launcher(gemm_template, Tiles((m, n), (16, 16)).blocks)
    assert launcher.program(Tiling(), a, b, c) is launcher.program(Tiling(), a, b, c)
    assert launcher.program(Tiling(), a, b, c) is not launcher.program(Tiling(bk=16), a, b, c)


def test_a_template_launch_can_be_bound_rebound_and_awaited(device: CPUDevice) -> None:
    n = 24
    a, b = device.empty(T.f16, n, n), device.empty(T.f16, n, n)
    a.array[:], b.array[:] = RNG.standard_normal((n, n)), RNG.standard_normal((n, n))
    expected = a.array.astype(np.float32) @ b.array.astype(np.float32)
    first, second = device.empty(T.f32, n, n), device.empty(T.f32, n, n)
    first.array[:], second.array[:] = np.nan, np.nan
    bound = gemm_template[Tiling(bm=8, bn=8, bk=8)].bind(a, b, first)
    bound()
    settle(device)
    np.testing.assert_allclose(first.array, expected, rtol=1e-4, atol=1e-4)

    async def awaited() -> None:
        bound.rebind(a, b, second)
        await bound.submit()

    asyncio.run(awaited())
    np.testing.assert_allclose(second.array, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [T.f16, T.f32])
def test_a_type_parameter_is_bound_by_the_dtype_of_the_tensors(device: CPUDevice, dtype: type[FloatingPoint]) -> None:
    m, n, k = 20, 18, 33
    a, b = device.empty(dtype, m, k), device.empty(dtype, k, n)
    a.array[:], b.array[:] = RNG.standard_normal((m, k)), RNG.standard_normal((k, n))
    c = device.empty(T.f32, m, n)
    gemm_generic(Tiling(bm=8, bn=8, bk=8), a, b, c, launch=Tiles(c.shape, (8, 8)))
    settle(device)
    np.testing.assert_allclose(c.array, a.array.astype(np.float32) @ b.array.astype(np.float32), rtol=1e-4, atol=1e-4)
    assert a.array.dtype == NUMPY[dtype]


def test_a_type_parameter_that_no_tensor_binds_or_two_tensors_bind_differently_is_reported() -> None:
    with pytest.raises(CaptureError, match="type parameter `F` is unbound"):
        capture(SURFACE, "gemm_generic")
    with pytest.raises(CaptureError, match="type parameter `F` is f16 and `b` is f32"):
        capture(SURFACE, "gemm_generic", dtypes={"a": F16, "b": F32, "c": F32})


@pytest.mark.parametrize(("arch", "narrowest"), [("spr", F16), ("avx512", F32), ("avx2", F32)])
def test_selectors_resolve_for_the_capture_target(arch: str, narrowest: Dtype) -> None:
    low, high = capture(SURFACE, "selected", target=targets.Target.of(arch)).tiles
    assert (low.dtype, high.dtype) == (narrowest, F64)


@pytest.mark.parametrize("arch", ["spr", "avx512", "avx2"])
def test_a_selector_in_the_configuration_changes_the_rounding(arch: str) -> None:
    x = RNG.standard_normal((16, 16), dtype=np.float32) * 100
    y = np.full_like(x, np.nan)
    narrowest = T.fp.min.resolve(targets.Target.of(arch))
    build_kernel("surface", "round_trip", {"p": {"dtype": T.fp.min}}, arch=arch)({}, x, y, launch=Blocks(2, 2))
    np.testing.assert_array_equal(y, x.astype(NUMPY[narrowest]).astype(np.float32))
    assert (narrowest == T.f16) == (arch == "spr"), "float16 on Sapphire Rapids, float32 on the CPUs without it"


def test_the_accumulator_of_a_type_follows_the_rule_of_the_package() -> None:
    counts, doubles, halves = capture(SURFACE, "accumulators").tiles
    assert [str(v.dtype) for v in (counts, doubles, halves)] == ["i32", "f64", "f32"]


def test_selectors_that_the_target_cannot_satisfy_name_the_missing_width() -> None:
    with pytest.raises(CaptureError, match="avx512 has no 8-bit float"):
        capture(SURFACE, "eight_bits", target=targets.Target.of("avx512"))
    ranged, precise = capture(SURFACE, "eight_bits", target=targets.Target.of("sm_90a")).tiles
    assert [(v.dtype.bits, v.dtype.man) for v in (ranged, precise)] == [(8, 2), (8, 3)], "e5m2, then e4m3"


def test_device_empty_takes_a_selector_and_resolves_it_for_its_target(device: CPUDevice) -> None:
    for selector in (T.fp.min, T.fp.max, T.acc(T.fp.min)):
        assert isinstance(selector, Selector)
        assert device.empty(selector, 4, 4).array.dtype == NUMPY[selector.resolve(device.target)]


@pytest.mark.parametrize("negate", [True, False])
def test_a_bool_constant_folds_the_branch_it_decides(negate: bool) -> None:
    x = RNG.standard_normal(50, dtype=np.float32)
    y = np.zeros_like(x)
    build_kernel("surface", "flagged", {"flags": {"negate": negate}})(x, y, {}, launch=Items(50))
    np.testing.assert_array_equal(y, -x if negate else x)


def test_a_local_name_can_stand_for_a_type_and_for_auto() -> None:
    x = RNG.standard_normal((8, 24)).astype(np.float32)
    y = np.zeros((8, 8), dtype=np.float32)
    build_kernel("surface", "type_aliases")(x, y, launch=Tiles(y.shape, (8, 8)))
    np.testing.assert_allclose(y, x.astype(np.float16).astype(np.float32).reshape(8, 3, 8).sum(axis=1), rtol=1e-6)


@pytest.mark.parametrize(("arch", "expected"), [("spr", 111), ("avx512", 110), ("avx2", 110)])
def test_features_and_the_family_of_the_target_are_constants_of_the_capture(arch: str, expected: int) -> None:
    out = np.zeros(5, dtype=np.uint32)
    build_kernel("surface", "capability", arch=arch)(out, launch=Items(5))
    np.testing.assert_array_equal(out, [expected] * 5)
    assert capture(SURFACE, "capability", target=targets.Target.of("sm_90a")).body

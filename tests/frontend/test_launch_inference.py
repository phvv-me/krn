"""The body's domain and an explicit launch agree, including ragged items and tiles on every available backend."""

from pathlib import Path

import numpy as np
import pytest

from krn import (
    Blocks,
    Const,
    Feature,
    Items,
    Matrix,
    T,
    Tiles,
    Vector,
    block,
    configure,
    family,
    grid,
    items,
    kernel,
    pipelined,
    struct,
    targets,
    zeros,
)
from krn.compiler.api import CPUDevice, capture
from krn.compiler.space import LaunchError, evaluate, geometry
from krn.facts.targets import Target
from krn.runtime.cpu import Value
from tests.frontend.test_kernel_unification import Backend
from tests.frontend.test_kernel_unification import backend as backend

THIS = Path(__file__)


@kernel
def by_shape(x: Vector[T.u32], y: Vector[T.u32]) -> None:
    for i in items(x.shape[0]):
        y[i] = x[i] * 3 + 1


@kernel
def by_scalar(n: T.u32, x: Vector[T.u32], y: Vector[T.u32]) -> None:
    for i in items(n):
        y[i] = x[i] * 3 + 1


@kernel
def by_expression(n: T.u32, x: Vector[T.u32], y: Vector[T.u32]) -> None:
    for i in items(n * 2):
        y[i] = x[i] * 3 + 1


@kernel
def by_local(x: Vector[T.u32], y: Vector[T.u32]) -> None:
    n = x.shape[0]
    for i in items(n):
        y[i] = x[i] * 3 + 1


@kernel
def gelu_bias(x: Matrix[T.f32], bias: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    v = x[rows, cols] + bias[rows, cols]
    y[rows, cols] = v.div(v.mul(-1.702).exp().add(1.0))


@kernel
def row_softmax(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:
    rows = block.tile(4)
    v = x[rows, 0:64]
    v = (v - v.max(keepdims=True)).exp()
    y[rows, 0:64] = v / v.sum(keepdims=True)


@struct
class Tiling:
    bm: Const[int] = 8
    bn: Const[int] = 16
    bk: Const[int] = 8


@kernel
def gemm(t: Tiling, a: Matrix[T.f32], b: Matrix[T.f32], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def stored_extent(x: Vector[T.f32], y: Vector[T.f32]) -> None:
    rows = block.tile(4)
    y[rows] = x[rows] + 1.0


@kernel
def conflicting_stores(x: Vector[T.f32], y: Vector[T.f32]) -> None:
    rows = block.tile(4)
    x[rows] = x[rows] + 1.0
    y[rows] = y[rows] + 1.0


@kernel
def manual_index(x: Vector[T.u32], y: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = x[i]


@kernel
def mixed_space(x: Vector[T.f32], y: Vector[T.f32]) -> None:
    for i in items(x.shape[0]):
        y[i] = x[i]
    rows = block.tile(4)
    y[rows] = x[rows]


@struct
class Required[F: T.Scalar]:
    factor: Const[int]


@kernel
def needs_configuration(s: Required[T.u32], x: Vector[T.u32], y: Vector[T.u32]) -> None:
    for i in items(x.shape[0]):
        y[i] = x[i] * s.factor


@kernel
def synchronize_grid() -> None:
    grid.sync()


@kernel
def synchronize_through_callee() -> None:
    synchronize_grid()


@family
def family_copy(t: Tiling, x: Vector[T.u32], y: Vector[T.u32]) -> None:
    for i in items(x.shape[0]):
        y[i] = x[i] + t.bm


@family_copy.variant(targets.CPU)
def family_copy_cpu(t: Tiling, x: Vector[T.u32], y: Vector[T.u32]) -> None:
    for i in items(x.shape[0]):
        y[i] = x[i] + t.bm + 1


@pytest.mark.parametrize("n", [1, 37, 300, 1025])
@pytest.mark.parametrize("name", ["by_shape", "by_scalar", "by_expression", "by_local"])
def test_items_inference_matches_explicit_geometry_and_bits(backend: Backend, name: str, n: int) -> None:
    count = n * 2 if name == "by_expression" else n
    x = np.arange(count, dtype=np.uint32)
    inferred, explicit = np.zeros_like(x), np.zeros_like(x)
    prefix: list[Value] = [n] if name in {"by_scalar", "by_expression"} else []
    ir = capture(THIS, name, target=backend.target)
    launch = geometry(ir, [*prefix, x, inferred])
    assert launch == Items(count)
    backend.execute(ir, launch, [*prefix, x, inferred])
    backend.execute(ir, Items(count), [*prefix, x, explicit])
    np.testing.assert_array_equal(inferred, explicit)
    np.testing.assert_array_equal(inferred, x * 3 + 1)
    assert len(ir.space) == 1 and ir.space[0].strided
    assert ir.space[0].grid == "x" and ir.space[0].tile == 1


@pytest.mark.parametrize("shape", [(20, 40), (1, 1), (8, 16), (9, 17), (33, 5)])
@pytest.mark.parametrize("name,tile", [("gelu_bias", (8, 16)), ("row_softmax", (4,))])
def test_tile_inference_matches_explicit_geometry_and_bits(
    backend: Backend, name: str, tile: tuple[int, ...], shape: tuple[int, int]
) -> None:
    rng = np.random.default_rng(7)
    x, bias = (rng.standard_normal(shape).astype(np.float32) for _ in range(2))
    inferred, explicit = np.zeros_like(x), np.zeros_like(x)
    ir = capture(THIS, name, target=backend.target)
    prefix: list[Value] = [x, bias] if name == "gelu_bias" else [x]
    launch = geometry(ir, [*prefix, inferred])
    want = Tiles(shape if len(tile) == 2 else shape[:1], tile).blocks
    assert launch == want
    backend.execute(ir, launch, [*prefix, inferred])
    backend.execute(ir, want, [*prefix, explicit])
    np.testing.assert_array_equal(inferred.view(np.uint32), explicit.view(np.uint32))
    expected = x + bias if name == "gelu_bias" else x - x.max(axis=1, keepdims=True)
    expected = expected / (1.0 + np.exp(-1.702 * expected)) if name == "gelu_bias" else np.exp(expected)
    if name == "row_softmax":
        expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(inferred, expected, rtol=2e-6, atol=2e-6)
    axes = {axis.grid: axis for axis in ir.space}
    assert {key: axis.tile for key, axis in axes.items()} == ({"y": 8, "x": 16} if len(tile) == 2 else {"x": 4})
    arguments = {"y": inferred, "x": x, "bias": bias}
    assert {key: [evaluate(expr, arguments) for expr in axis.extents] for key, axis in axes.items()} == (
        {"y": [shape[0]], "x": [shape[1]]} if len(tile) == 2 else {"x": [shape[0]]}
    )


def test_real_launcher_configuration_forms_and_ragged_gemm(device: CPUDevice) -> None:
    rng = np.random.default_rng(5)
    a, b = (device.to_device(rng.standard_normal(shape).astype(np.float32)) for shape in ((50, 70), (70, 38)))
    outputs = [device.empty(T.f32, 50, 38) for _ in range(6)]
    t = Tiling(bm=16, bn=16, bk=8)
    launch = Tiles((50, 38), (16, 16)).blocks
    gemm(a, b, outputs[0])
    gemm[t](a, b, outputs[1])
    gemm(a, b, outputs[2], launch=Blocks(3, 7))
    gemm(t, a, b, outputs[3], launch=launch)
    gemm(t, a, b, outputs[4])
    gemm(t, a, b, outputs[5], launch=launch)
    device.default_stream.record().wait(60)
    for output in outputs:
        np.testing.assert_array_equal(output.array, outputs[0].array)
        np.testing.assert_allclose(output.array, a.array @ b.array, rtol=2e-5, atol=2e-5)


def test_bound_inference_recomputes_on_rebind_and_on_each_plain_call(device: CPUDevice) -> None:
    launcher = device.launcher(by_shape)
    bound = launcher.bind(device.empty(T.u32, 1), device.empty(T.u32, 1))
    for count in (1, 37, 300, 1025):
        x, y = device.to_device(np.arange(count, dtype=np.uint32)), device.empty(T.u32, count)
        bound.rebind(x, y)
        bound()
        launcher(x, y)
        device.default_stream.record().wait(60)
        np.testing.assert_array_equal(y.array, x.array * 3 + 1)


def test_stored_extent_wins_over_a_larger_load(backend: Backend) -> None:
    x, y = np.arange(37, dtype=np.float32), np.zeros(9, np.float32)
    ir = capture(THIS, "stored_extent", target=backend.target)
    assert geometry(ir, [x, y]) == Blocks(3)
    backend.execute(ir, geometry(ir, [x, y]), [x, y])
    np.testing.assert_array_equal(y, x[:9] + 1)


@pytest.mark.parametrize("name", ["conflicting_stores", "mixed_space", "manual_index"])
def test_unreadable_or_conflicting_geometry_fails_on_each_backend(backend: Backend, name: str) -> None:
    ir = capture(THIS, name, target=backend.target)
    for _ in range(2):
        with pytest.raises(LaunchError) as error:
            geometry(ir, [np.zeros(37), np.zeros(9)])
        assert name in str(error.value) and "launch=" in str(error.value)


def test_inference_can_recover_after_a_shape_conflict(device: CPUDevice) -> None:
    launcher = device.launcher(conflicting_stores)
    x, y = device.to_device(np.zeros(37, np.float32)), device.to_device(np.zeros(9, np.float32))
    with pytest.raises(LaunchError):
        launcher(x, y)
    launcher(x, device.to_device(np.zeros(37, np.float32)))
    device.default_stream.record().wait(60)
    np.testing.assert_array_equal(x.array, np.ones(37))


def test_numpy_scalar_extents_are_host_readable() -> None:
    ir = capture(THIS, "by_scalar")
    assert geometry(ir, [np.uint32(37), np.zeros(37), np.zeros(37)]) == Items(37)


def test_conflicting_stores_and_mixed_spaces_fail_on_every_call(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 37), device.empty(T.f32, 9)
    for fn in (conflicting_stores, mixed_space):
        for _ in range(2):
            with pytest.raises(LaunchError, match="launch="):
                fn(x, y)


def test_manual_index_fails_on_first_and_later_calls_with_both_fixes(device: CPUDevice) -> None:
    x, y = device.empty(T.u32, 37), device.empty(T.u32, 37)
    for _ in range(2):
        with pytest.raises(LaunchError) as error:
            manual_index(x, y)
        assert all(text in str(error.value) for text in ("manual_index", "items(", "launch="))


def test_required_configuration_names_the_kernel_and_explicit_fix(device: CPUDevice) -> None:
    x, y = device.empty(T.u32, 1), device.empty(T.u32, 1)
    with pytest.raises(TypeError) as error:
        needs_configuration(x, y)
    assert "needs_configuration" in str(error.value) and "[Config(...)]" in str(error.value)


@pytest.mark.parametrize("pinned", [False, True])
def test_family_short_and_configured_forms_use_the_selected_kernel(
    device: CPUDevice, monkeypatch: pytest.MonkeyPatch, pinned: bool
) -> None:
    if pinned:
        configure(**{"run.variants": {"family_copy": "portable"}})
    else:
        configure(**{"run.variants": {}})
    x = device.to_device(np.arange(37, dtype=np.uint32))
    outputs = [device.empty(T.u32, 37) for _ in range(3)]
    family_copy(x, outputs[0])
    family_copy[Tiling()](x, outputs[1])
    family_copy(x, outputs[2], launch=Items(37))
    device.default_stream.record().wait(60)
    for output in outputs:
        np.testing.assert_array_equal(output.array, x.array + (8 if pinned else 9))


@pytest.mark.parametrize(
    "name,cooperative",
    [("synchronize_grid", True), ("synchronize_through_callee", True), ("by_shape", False)],
)
def test_cooperative_flag_walks_callees(name: str, cooperative: bool) -> None:
    assert capture(THIS, name, target=Target.of("sm_90a")).cooperative is cooperative


@pytest.mark.parametrize("target", targets.all_targets(), ids=lambda target: target.arch)
def test_grid_sync_feature_matches_vendor(target: Target) -> None:
    assert (Feature.GRID_SYNC in target.features) == isinstance(target, targets.Nvidia | targets.AMD)

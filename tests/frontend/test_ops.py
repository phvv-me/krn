"""The fluent tile operations against numpy, on the CPU and through NVRTC, with kernels written out per test."""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from krn import Tiles, targets
from krn.compiler.api import CaptureError, capture, emit
from krn.compiler.capture import FLUENT, NORMALIZING, REDUCTIONS
from krn.frontend._ops import TileOps
from krn.runtime.nvrtc import Nvidia as Nvrtc
from tests.conftest import build_kernel
from tests.helpers.text import raises_literal

RNG = np.random.default_rng(2)
SHAPE = (5, 11)  # smaller than the 8 x 8 tile, so the ragged edge is part of every launch
HEADER = "from krn import Matrix, T, Vector, block, kernel, maximum, minimum\n\n\n@kernel\n"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def exact(function: Callable[[float], float], x: np.ndarray) -> np.ndarray:
    """`function` over float64 and rounded once to the type of `x`."""
    return np.vectorize(function, otypes=[np.float64])(x).astype(x.dtype)


UNARY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "exp": np.exp,
    "exp2": np.exp2,
    "expm1": np.expm1,
    "log": np.log,
    "log2": np.log2,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1 / np.sqrt(x),
    "reciprocal": lambda x: 1 / x,
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "sigmoid": sigmoid,
    "erf": lambda x: exact(math.erf, x),
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "relu": lambda x: np.maximum(x, 0),
    "gelu": lambda x: (0.5 * x * (1 + exact(math.erf, x / math.sqrt(2)))).astype(x.dtype),
    "silu": lambda x: x * sigmoid(x),
    "softplus": lambda x: np.logaddexp(0, x),
    "abs": np.abs,
}


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str, str], Path]:
    """Write a kernel file once per (name, text) and return its path."""
    folder = tmp_path_factory.mktemp("ops")

    def write(name: str, text: str) -> Path:
        path = folder / f"{name}.py"
        path.write_text(text)
        return path

    return write


def unary_text(*, inplace: bool, dtype: str = "f32") -> str:
    outputs = "".join(f", o_{name}: Matrix[T.{dtype}]" for name in UNARY)
    body = ""
    for name in UNARY:
        if inplace:
            body += f"    v_{name} = x[rows, cols]\n    v_{name}.{name}_()\n    o_{name}[rows, cols] = v_{name}\n"
        else:
            body += f"    o_{name}[rows, cols] = x[rows, cols].{name}()\n"
    return f"{HEADER}def unary(x: Matrix[T.{dtype}]{outputs}) -> None:\n    rows, cols = block.tile(8, 8)\n{body}"


def run_unary(path: Path, x: np.ndarray) -> dict[str, np.ndarray]:
    outputs = {name: np.full(SHAPE, np.nan, dtype=x.dtype) for name in UNARY}
    build_kernel(path, "unary")(x, *outputs.values(), launch=Tiles(SHAPE, (8, 8)))
    return outputs


def inputs(dtype: type[np.floating]) -> list[np.ndarray]:
    positive = RNG.uniform(0.1, 3.0, SHAPE).astype(dtype)
    signed = (RNG.standard_normal(SHAPE) * 2).astype(dtype)
    signed[0, :6] = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]  # the ties of round half to even
    return [positive, signed]


@pytest.mark.parametrize(("dtype", "name", "rtol"), [(np.float32, "f32", 3e-5), (np.float64, "f64", 1e-10)])
@pytest.mark.parametrize("domain", [0, 1], ids=["positive", "signed"])
def test_every_unary_operation_matches_numpy(
    sources: Callable[[str, str], Path],
    dtype: type[np.floating],
    name: str,
    rtol: float,
    domain: int,
) -> None:
    x = inputs(dtype)[domain]
    outputs = run_unary(sources(f"unary_{name}", unary_text(inplace=False, dtype=name)), x)
    with np.errstate(all="ignore"):
        for name, reference in UNARY.items():
            np.testing.assert_allclose(outputs[name], reference(x), rtol=rtol, atol=1e-6, equal_nan=True, err_msg=name)


def test_the_in_place_form_updates_the_variable_and_agrees_with_the_pure_form(
    sources: Callable[[str, str], Path],
) -> None:
    x = inputs(np.float32)[1]
    pure = run_unary(sources("unary", unary_text(inplace=False)), x)
    in_place = run_unary(sources("unary_", unary_text(inplace=True)), x)
    for name in UNARY:
        np.testing.assert_array_equal(in_place[name], pure[name], err_msg=name)


BINARY_TEXT = (
    HEADER
    + """def binary(a: Matrix[T.f32], b: Matrix[T.f32], s: T.f32, o: Matrix[T.f32], u: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    p, q = a[rows, cols], b[rows, cols]
    o[rows, cols] = p.add(q) + p.sub(q) * p.mul(q) - p.div(q) + p.maximum(q) - p.minimum(q) + p.pow(q)
    u[rows, cols] = maximum(p, q) + minimum(p, q) + p.mul(2.0).add(s) + p.pow(2) + p.sub(a[rows, slice(0, 1)])
"""
)


def test_binary_operations_take_tiles_literals_scalars_and_broadcast_columns(
    sources: Callable[[str, str], Path],
) -> None:
    a, b = (
        RNG.uniform(0.5, 2.0, SHAPE).astype(np.float32),
        RNG.uniform(0.5, 2.0, SHAPE).astype(np.float32),
    )
    o, u = np.full(SHAPE, np.nan, np.float32), np.full(SHAPE, np.nan, np.float32)
    build_kernel(sources("binary", BINARY_TEXT), "binary")(a, b, 0.25, o, u, launch=Tiles(SHAPE, (8, 8)))
    np.testing.assert_allclose(
        o,
        (a + b) + (a - b) * (a * b) - a / b + np.maximum(a, b) - np.minimum(a, b) + a**b,
        rtol=2e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        u,
        np.maximum(a, b) + np.minimum(a, b) + a * 2 + np.float32(0.25) + a**2 + (a - a[:, :1]),
        rtol=2e-5,
        atol=1e-5,
    )


CLIP_TEXT = (
    HEADER
    + """def limits(a: Matrix[T.f32], s: T.f32, o: Matrix[T.f32], u: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    o[rows, cols] = a[rows, cols].clip(-0.5, 0.75)
    v = a[rows, cols]
    v.clip_(s, 1.0)
    v.add_(1.0)
    v.maximum_(v)
    u[rows, cols] = v
"""
)


def test_clip_and_the_in_place_binary_forms(sources: Callable[[str, str], Path]) -> None:
    a = (RNG.standard_normal(SHAPE) * 2).astype(np.float32)
    o, u = np.full(SHAPE, np.nan, np.float32), np.full(SHAPE, np.nan, np.float32)
    build_kernel(sources("limits", CLIP_TEXT), "limits")(a, -1.5, o, u, launch=Tiles(SHAPE, (8, 8)))
    np.testing.assert_array_equal(o, np.clip(a, -0.5, 0.75))
    np.testing.assert_array_equal(u, np.clip(a, -1.5, 1.0) + 1)


REDUCTION_SHAPE = (8, 16)


def reference(name: str, x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    match name:
        case "mean" | "sum" | "prod" | "max" | "min":
            return getattr(x, name)(axis=axis, keepdims=keepdims)
        case "norm":
            return np.sqrt((x * x).sum(axis=axis, keepdims=keepdims))
        case "logsumexp":
            top = x.max(axis=axis, keepdims=True)
            return np.log(np.exp(x - top).sum(axis=axis, keepdims=keepdims)) + (top if keepdims else top.squeeze(axis))
        case "softmax":
            e = np.exp(x - x.max(axis=axis, keepdims=True))
            return e / e.sum(axis=axis, keepdims=True)
        case _:
            return x - reference("logsumexp", x, axis, keepdims=True)


def reduction_text(axis: int, keepdims: bool, names: list[str]) -> str:
    """One kernel that writes every reduction of an 8 x 16 tile to its own output."""
    shapes = {
        n: REDUCTION_SHAPE if n in NORMALIZING else np.empty(REDUCTION_SHAPE).sum(axis=axis, keepdims=keepdims).shape
        for n in names
    }
    outputs = "".join(f", o_{n}: {'Matrix' if len(shapes[n]) == 2 else 'Vector'}[T.f32]" for n in names)
    body = ""
    for n in names:
        index = ", ".join(f"slice(0, {extent})" for extent in shapes[n])
        arguments = f"axis={axis}" if n in NORMALIZING else f"axis={axis}, keepdims={keepdims}"
        body += f"    o_{n}[{index}] = x[slice(0, 8), slice(0, 16)].{n}({arguments})\n"
    return f"{HEADER}def reduce(x: Matrix[T.f32]{outputs}) -> None:\n{body}"


@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_every_reduction_matches_numpy_along_each_axis(
    sources: Callable[[str, str], Path], axis: int, keepdims: bool
) -> None:
    names = [*REDUCTIONS, *NORMALIZING]
    x = RNG.uniform(0.5, 1.5, REDUCTION_SHAPE).astype(np.float32)
    expected = {n: reference(n, x, axis, keepdims) for n in names}
    outputs = {n: np.full(expected[n].shape, np.nan, np.float32) for n in names}
    path = sources(f"reduce_{axis}_{keepdims}", reduction_text(axis, keepdims, names))
    build_kernel(path, "reduce")(x, *outputs.values(), launch=Tiles(REDUCTION_SHAPE, (8, 16)))
    for n in names:
        np.testing.assert_allclose(outputs[n], expected[n], rtol=2e-4, atol=1e-5, err_msg=f"{n} axis={axis}")


CHAIN_TEXT = (
    HEADER
    + """def chain(x: Matrix[T.f32], o: Matrix[T.f32], p: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    v = x[rows, cols]
    v.exp_().log_().add_(1.0)
    o[rows, cols] = v
    w = x[rows, cols]
    w.mul_(-1.0).exp_()
    p[rows, cols] = w
"""
)


def test_an_in_place_call_returns_its_variable_so_calls_chain(
    sources: Callable[[str, str], Path],
) -> None:
    x = RNG.uniform(-2.0, 2.0, SHAPE).astype(np.float32)
    o, p = np.full(SHAPE, np.nan, np.float32), np.full(SHAPE, np.nan, np.float32)
    build_kernel(sources("chain", CHAIN_TEXT), "chain")(x, o, p, launch=Tiles(SHAPE, (8, 8)))
    np.testing.assert_allclose(o, x + 1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p, np.exp(-x), rtol=1e-5, atol=1e-6)


RAGGED_TEXT = (
    HEADER
    + """def ragged(x: Matrix[T.f32], o: Matrix[T.f32], p: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 16)
    o[rows, slice(0, 1)] = x[rows, cols].max(axis=1, keepdims=True)
    p[rows, cols] = maximum(x[rows, cols], x[rows, cols])
"""
)


def test_a_reduction_over_a_ragged_tile_skips_the_lanes_past_the_edge(
    sources: Callable[[str, str], Path],
) -> None:
    """The maximum of a row of negative values in a tile wider than the tensor is a negative value, since the lanes past
    the edge are invalid and neutral; `maximum` of a tile with itself keeps every valid lane."""
    x = -RNG.uniform(1.0, 5.0, SHAPE).astype(np.float32)
    o, p = np.full((SHAPE[0], 1), np.nan, np.float32), np.full(SHAPE, np.nan, np.float32)
    build_kernel(sources("ragged", RAGGED_TEXT), "ragged")(x, o, p, launch=Tiles(SHAPE, (8, 16)))
    np.testing.assert_array_equal(o, x.max(axis=1, keepdims=True))
    np.testing.assert_array_equal(p, x)


NAN_TEXT = (
    HEADER
    + """def nans(a: Matrix[T.f32], b: Matrix[T.f32], hi: Matrix[T.f32], lo: Matrix[T.f32], top: Matrix[T.f32]) -> None:
    rows, cols = block.tile(4, 8)
    hi[rows, cols] = maximum(a[rows, cols], b[rows, cols])
    lo[rows, cols] = a[rows, cols].minimum(b[rows, cols])
    top[rows, slice(0, 1)] = a[rows, cols].max(axis=1, keepdims=True)
"""
)


def test_a_nan_propagates_through_maximum_minimum_and_max(
    sources: Callable[[str, str], Path],
) -> None:
    a, b = (
        RNG.standard_normal((4, 8)).astype(np.float32),
        RNG.standard_normal((4, 8)).astype(np.float32),
    )
    a[0, 1], b[1, 2], a[2, 5] = np.nan, np.nan, np.nan
    hi, lo, top = np.zeros_like(a), np.zeros_like(a), np.zeros((4, 1), np.float32)
    build_kernel(sources("nans", NAN_TEXT), "nans")(a, b, hi, lo, top, launch=Tiles(a.shape, (4, 8)))
    np.testing.assert_array_equal(hi, np.maximum(a, b))
    np.testing.assert_array_equal(lo, np.minimum(a, b))
    np.testing.assert_array_equal(top, np.max(a, axis=1, keepdims=True))
    assert np.isnan(hi[0, 1]) and np.isnan(hi[1, 2]) and np.isnan(top[0, 0]) and not np.isnan(top[3, 0])


def test_the_lowered_operations_are_exactly_the_methods_of_the_sketch_tile_ops() -> None:
    sketch = {name for name in vars(TileOps) if not name.startswith("_")}
    lowered = FLUENT | {f"{name}_" for name in FLUENT - {*REDUCTIONS, *NORMALIZING}}
    assert lowered == sketch


STATEMENTS = [
    ("v = x[rows, cols].sum_()", "`sum` has no in-place form"),
    ("v = x[rows, cols].softmax(axis=1, keepdims=True)", "unexpected argument `keepdims`"),
    ("v = x[rows, cols].log_softmax_()", "`log_softmax` has no in-place form"),
    (
        "v = x[rows, cols].to(T.u32).mean(axis=1, keepdims=True)",
        "mean needs a float tile, got u32",
    ),
    ("v = x[rows, cols].sum().sum()", "reducing a rank-1 tile to a scalar is not supported"),
    ("v = x[rows, cols].add(h[rows, cols])", "dtype mismatch in `+`: f32 and f16"),
    ("v = maximum(x[rows, cols], h[rows, cols])", "dtype mismatch in `maximum`: f32, f16"),
    ("v = x[rows, cols].clip(0.0)", "`x[rows, cols].clip` is missing high"),
    ("v = x[rows, cols].add()", "`x[rows, cols].add` is missing other"),
    ("v = x[rows, cols].add(other=1.0, extra=2.0)", "unexpected argument `extra`"),
    ("v = x[rows, 0:1]\n    v.add_(x[rows, cols])", "`add_` would make tile `v`"),
]


@pytest.mark.parametrize(("statement", "message"), STATEMENTS, ids=[m[:32] for _, m in STATEMENTS])
def test_misused_operations_are_capture_errors(
    sources: Callable[[str, str], Path], statement: str, message: str
) -> None:
    signature = "def bad(x: Matrix[T.f32], h: Matrix[T.f16]) -> None:"
    text = f"{HEADER}{signature}\n    rows, cols = block.tile(8, 8)\n    {statement}\n"
    with raises_literal(CaptureError, message):
        capture(
            sources(f"bad_{STATEMENTS.index((statement, message))}", text),
            "bad",
            target=targets.Target.of("avx512"),
        )


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a"])
def test_the_operations_compile_with_nvrtc(sources: Callable[[str, str], Path], nvrtc: Nvrtc, arch: str) -> None:
    target = targets.Target.of(arch)
    reductions = sources("reduce_1_True", reduction_text(1, True, [*REDUCTIONS, *NORMALIZING]))
    for path, name in [
        (sources("unary", unary_text(inplace=False)), "unary"),
        (sources("unary_", unary_text(inplace=True)), "unary"),
        (sources("binary", BINARY_TEXT), "binary"),
        (sources("limits", CLIP_TEXT), "limits"),
        (reductions, "reduce"),
    ]:
        source = emit(capture(path, name, target=target), target).source
        assert "#include" not in source
        assert nvrtc.compile(source, arch, "cubin").data[:4] == b"\x7fELF", name

"""Diagnostics: every mistake is a CaptureError that names the user's file, line and column."""

import re
from pathlib import Path

import pytest

from krn import targets
from krn.compiler.api import Arithmetic, CaptureError, capture
from krn.compiler.dtypes import F32
from tests.conftest import KERNELS, SKETCH

MISTAKES = KERNELS / "mistakes.py"
LINES = MISTAKES.read_text().splitlines()

# kernel, offending text, message fragment, arithmetic
CASES = [
    (
        "store_f16_into_f32",
        "x[i] = h[i]",
        "cannot store f16 into f32 tensor `x`",
        Arithmetic.LOSSLESS,
    ),
    ("u32_plus_u8", "n + b[i]", "dtype mismatch in `+`: u32 and u8", Arithmetic.STRICT),
    (
        "f32_plus_u32",
        "x[i] + n",
        "dtype mismatch in `+`: f32 and u32; int to float is never implicit",
        Arithmetic.LOSSLESS,
    ),
    ("u64_less_u32", "i < n", "dtype mismatch in `<`: u64 and u32", Arithmetic.STRICT),
    (
        "dynamic_extent",
        "0 : a.shape[1]",
        "slice extent must be a capture-time constant",
        Arithmetic.LOSSLESS,
    ),
    ("missing_const", "t: NoDefault", "missing Const `NoDefault.bm`", Arithmetic.LOSSLESS),
    (
        "too_many_indices",
        "x[i, i]",
        "tensor `x` has 1 axes but 2 indices were given",
        Arithmetic.LOSSLESS,
    ),
    (
        "tile_dtype_mix",
        "a[rows, cols] + b[rows, cols]",
        "dtype mismatch in `+`: f16 and f32",
        Arithmetic.STRICT,
    ),
    (
        "inner_dimension",
        "a[rows, 0:32] @ b[0:64, cols]",
        "inner dimensions of `@` differ: (8, 32) @ (64, 8)",
        Arithmetic.LOSSLESS,
    ),
    (
        "tile_into_f16",
        "out[rows, cols] = a[rows, cols]",
        "cannot store f16 tile (8, 8) into f32 slice (8, 8)",
        Arithmetic.LOSSLESS,
    ),
    (
        "scalar_into_tile",
        "c[rows, cols] = T.f32(1.0)",
        "cannot store a scalar f32 into a tile slice",
        Arithmetic.LOSSLESS,
    ),
    ("arbitrary_widths", "a + b", "dtype mismatch in `+`: u3 and u4", Arithmetic.LOSSLESS),
    ("literal_too_big", "x[i] + 300", "literal 300 does not fit u8", Arithmetic.LOSSLESS),
    (
        "bad_axis",
        "a[rows, cols].max(axis=3, keepdims=True)",
        "axis 3 is out of range for a rank-2 tile",
        Arithmetic.LOSSLESS,
    ),
    (
        "warp_op_on_cpu",
        "warp.shfl_xor(x[i], 1)",
        "`warp.shfl_xor` needs WARP_SHUFFLE, which target avx512 lacks; targets that have it: "
        "sm_80, sm_86, sm_89, sm_90a, sm_100a, sm_103a, sm_120a, sm_121a, gfx942, gfx950, "
        "apple_m1, apple_m2, apple_m3, apple_m4, apple_m5",
        Arithmetic.LOSSLESS,
    ),
    ("not_a_bool", "i:", "condition must be bool, got u64", Arithmetic.LOSSLESS),
    ("unsupported", "with x:", "unsupported construct `With`", Arithmetic.LOSSLESS),
    (
        "async_kernel",
        "async def async_kernel",
        "async kernels are not lowered in this slice",
        Arithmetic.LOSSLESS,
    ),
    (
        "host_in_kernel",
        "host.Device.current()",
        "`host.Device.current` is host-only and cannot be used in a kernel",
        Arithmetic.LOSSLESS,
    ),
    (
        "pipelined_over_range",
        "range(0, a.shape[1], 8)",
        "`pipelined(extent, tile)` takes the extent of the axis, not a `range`",
        Arithmetic.LOSSLESS,
    ),
    (
        "pipelined_stages_zero",
        "0):",
        "stages must be AUTO or a positive constant",
        Arithmetic.LOSSLESS,
    ),
    (
        "pipelined_dynamic_tile",
        "n):",
        "a tile size must be a capture-time constant",
        Arithmetic.LOSSLESS,
    ),
    ("auto_as_a_value", "AUTO", "expected a value, got AUTO", Arithmetic.LOSSLESS),
    (
        "in_place_on_expression",
        "(a[rows, cols] + 1.0).exp_()",
        "`exp_` updates a tile variable in place, but `a[rows, cols] + 1.0` is not one; use `exp`",
        Arithmetic.LOSSLESS,
    ),
    ("exp_of_half", "a[rows, cols].exp()", "exp needs f32 or f64, got f16", Arithmetic.LOSSLESS),
    (
        "tile_needs_one_or_two_sizes",
        "block.tile(8, 8, 8)",
        "`block.tile` takes one size",
        Arithmetic.LOSSLESS,
    ),
    ("selector_without_a_float", "T.fp.emax(4)", "avx512 has no 4-bit float", Arithmetic.LOSSLESS),
    (
        "format_as_a_value",
        "formats.NVFP4",
        "`formats.NVFP4` cannot be used as a value in a kernel",
        Arithmetic.LOSSLESS,
    ),
    (
        "several_values_in_one_name",
        "both = block.tile(8, 8)",
        "`both` cannot hold several values",
        Arithmetic.LOSSLESS,
    ),
]


STDLIB = [
    ("math", "math.sqrt(x[i])", "write `from krn import math`"),
    ("itertools", "itertools.chain(x, x)", "the standard library is not available in a kernel"),
]


@pytest.mark.parametrize(("module", "expression", "message"), STDLIB, ids=[case[0] for case in STDLIB])
def test_a_standard_library_module_in_a_kernel_is_refused_with_the_fix(
    tmp_path: Path, module: str, expression: str, message: str
) -> None:
    path = tmp_path / "kernel.py"
    path.write_text(
        f"import {module}\nfrom krn import Vector, items, kernel\nfrom krn.frontend.types import f32\n\n"
        f"@kernel\ndef k(x: Vector[f32], y: Vector[f32]) -> None:\n    for i in items(x.shape[0]):\n"
        f"        y[i] = {expression}\n"
    )
    with pytest.raises(CaptureError, match=message):
        capture(path, "k", dtypes={"x": F32, "y": F32})


@pytest.mark.parametrize("call", ["x[i].to(f32, 0)", "x[i].to(f32, copy=True)", "x[i].to()", "x[i].to(f32, device=0)"])
def test_to_takes_a_dtype_and_nothing_else(tmp_path: Path, call: str) -> None:
    path = tmp_path / "kernel.py"
    path.write_text(
        "from krn import Vector, items, kernel\nfrom krn.frontend.types import f32\n\n"
        f"@kernel\ndef k(x: Vector[f32], y: Vector[f32]) -> None:\n    for i in items(x.shape[0]):\n"
        f"        y[i] = {call}\n"
    )
    message = "krn's `.to` takes a dtype; devices are chosen with the launch"
    with pytest.raises(CaptureError, match=re.escape(message)):
        capture(path, "k", dtypes={"x": F32, "y": F32})


def kernel_lines(name: str) -> list[tuple[int, str]]:
    """The numbered source lines of one kernel of the mistakes file, from its `def` to the next top-level statement."""
    start = next(n for n, source in enumerate(LINES) if source.startswith((f"def {name}(", f"async def {name}(")))
    end = next((n for n in range(start + 1, len(LINES)) if LINES[n][:1] not in ("", " ")), len(LINES))
    return [(n + 1, source) for n, source in enumerate(LINES[start:end], start=start)]


@pytest.mark.parametrize(("name", "text", "message", "arithmetic"), CASES, ids=[c[0] for c in CASES])
def test_mistake_is_reported_at_the_users_position(name: str, text: str, message: str, arithmetic: Arithmetic) -> None:
    ((line, source),) = [(n, source) for n, source in kernel_lines(name) if text in source]
    column = source.index(text) + 1
    with pytest.raises(CaptureError) as raised:
        capture(MISTAKES, name, arithmetic=arithmetic)
    assert f"{MISTAKES}:{line}:{column}: error: " in str(raised.value)
    assert message in str(raised.value)


def test_error_shows_the_source_line_and_a_caret() -> None:
    with pytest.raises(CaptureError) as raised:
        capture(MISTAKES, "store_f16_into_f32")
    assert "\n        x[i] = h[i]\n        ^" in str(raised.value)


def test_lossless_arithmetic_widens_what_strict_rejects() -> None:
    capture(MISTAKES, "u32_plus_u8", arithmetic=Arithmetic.LOSSLESS)
    capture(MISTAKES, "u64_less_u32", arithmetic=Arithmetic.LOSSLESS)
    capture(MISTAKES, "tile_dtype_mix", arithmetic=Arithmetic.LOSSLESS)


@pytest.mark.parametrize(
    ("file", "name"),
    [
        ("elementwise", "saxpy"),
        ("elementwise", "add"),
        ("softmax", "softmax"),
        ("softmax", "softmax_steps"),
        ("gemm", "gemm"),
        ("automaton", "run_dfa"),
    ],
)
def test_examples_capture_with_default_consts(file: str, name: str) -> None:
    kernel = capture(SKETCH / f"{file}.py", name, target=targets.Target.of("avx512"))
    assert kernel.name == name
    assert kernel.body


def test_the_warp_specialized_example_needs_bulk_copies_which_hopper_has_and_the_cpu_does_not() -> None:
    gemm = SKETCH / "gemm.py"
    with pytest.raises(CaptureError, match="no implemented WarpSpecialized schedule lowering"):
        capture(gemm, "gemm_warp_specialized", target=targets.Target.of("sm_90a"))
    with pytest.raises(CaptureError, match="WarpSpecialized needs BULK_COPY, which target cpu lacks"):
        capture(gemm, "gemm_warp_specialized", target=targets.Target.of("avx512"))


def test_struct_consts_come_from_the_caller() -> None:
    kernel = capture(SKETCH / "gemm.py", "gemm", consts={"t": {"bm": 16, "bn": 8, "bk": 4}})
    assert kernel.tiles[0].shape == (16, 8)


@pytest.mark.parametrize("decorator", ["kernel", "device", "template", "comptime"])
def test_every_spelling_of_the_one_decorator_is_an_entry_point_and_a_plain_function_is_not(
    tmp_path: Path, decorator: str
) -> None:
    path = tmp_path / "kernel.py"
    path.write_text(
        "from krn import Vector, kernel\nfrom krn.frontend.types import f32\n\n"
        f"@{decorator}\ndef marked(x: Vector[f32]) -> None:\n    x[0] = 1.0\n\n"
        "def plain(x: Vector[f32]) -> None:\n    x[0] = 1.0\n"
    )
    if decorator == "kernel":
        assert capture(path, "marked", dtypes={"x": F32}).name == "marked"
    else:
        with pytest.raises(CaptureError) as error:
            capture(path, "marked", dtypes={"x": F32})
        assert "write @kernel" in str(error.value)
    with pytest.raises(CaptureError, match="no @kernel function named `plain`"):
        capture(path, "plain", dtypes={"x": F32})


@pytest.mark.parametrize("decorator", ["device", "template", "comptime"])
@pytest.mark.parametrize("scope", ["function", "method"])
def test_removed_decorators_are_rejected_on_callees_too(tmp_path: Path, decorator: str, scope: str) -> None:
    path = tmp_path / "callee.py"
    source = "from krn import Vector, kernel, struct\nfrom krn.frontend.types import f32\n"
    source += (
        f"@{decorator}\ndef helper(x: f32) -> f32:\n    return x + 1\n"
        "@kernel\ndef entry(x: Vector[f32]) -> None:\n    x[0] = helper(x[0])\n"
        if scope == "function"
        else "@struct\nclass Box:\n"
        f"    @{decorator}\n    def helper(self, x: f32) -> f32:\n        return x + 1\n"
        "@kernel\ndef entry(b: Box, x: Vector[f32]) -> None:\n    x[0] = b.helper(x[0])\n"
    )
    path.write_text(source)
    with pytest.raises(CaptureError) as error:
        capture(path, "entry", dtypes={"x": F32})
    assert "write @kernel" in str(error.value)


def test_block_sum_needs_warp_shuffles_that_the_cpu_target_lacks() -> None:
    with pytest.raises(
        CaptureError,
        match="target cpu lacks WARP_SHUFFLE; targets that have all: \\['nvidia', 'amd', 'apple'\\]",
    ):
        capture(SKETCH / "target_variables.py", "block_sum", target=targets.Target.of("avx512"))

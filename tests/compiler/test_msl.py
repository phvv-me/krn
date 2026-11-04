"""The MSL renderer: every unit the CUDA renderer compiles is emitted for an Apple GPU and must parse and type check
against `msl_shim/metal_stdlib` (a stand-in header, see there), contain nothing Metal lacks, and a kernel that needs
what the hardware lacks is refused at capture with the reason, at the user's line."""

import subprocess
from pathlib import Path

import pytest

from krn import targets
from krn.compiler.api import CaptureError, capture, emit
from krn.compiler.dtypes import I64, U64
from krn.compiler.errors import TargetError
from krn.compiler.msl import MSLRenderer
from krn.frontend.atomic import AtomicOp
from krn.runtime.cpu import CLANG
from tests.compiler.test_tensor_cores import source_of as matrix_source
from tests.conftest import KERNELS, SKETCH
from tests.helpers.text import forbidden_msl, raises_literal
from tests.helpers.units import COMMON

SHIM = Path(__file__).resolve().parents[1] / "msl_shim"
APPLE = targets.Target.of("apple_m4")
REFUSED = [
    (
        "intrinsics",
        "block_votes",
        "`block.count` has no lowering for target apple_m4: Metal needs a simd_sum plus threadgroup memory; "
        "the C backend is sequential; targets that have it: "
        "sm_80, sm_86, sm_89, sm_90a, sm_100a, sm_103a, sm_120a, sm_121a, gfx942, gfx950",
    ),
    (
        "gemm",
        "gemm",
        "need 98304 bytes of shared memory, apple has 32768",
    ),  # the default 128 x 128 tile
]


def source_of(file: str, name: str, consts: dict | None) -> str:
    path = KERNELS / f"{file}.py" if (KERNELS / f"{file}.py").exists() else SKETCH / f"{file}.py"
    return emit(capture(path, name, consts=consts, target=APPLE), APPLE, lineinfo=False).source


@pytest.mark.parametrize(
    ("file", "name", "consts"),
    [
        *COMMON,
        *(("intrinsics", name, None) for name in ("group_ranks", "segmented_shuffles", "lane_masks")),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_the_unit_has_nothing_metal_lacks_and_type_checks_against_the_shim(
    tmp_path: Path, file: str, name: str, consts: dict | None
) -> None:
    if name == "debug_aids":
        with pytest.raises(CaptureError, match="assert is not available"):
            source_of(file, name, consts)
        return
    source = source_of(file, name, consts)
    assert not forbidden_msl(source), forbidden_msl(source)
    assert source.count("#include") == 1 and "<metal_stdlib>" in source
    unit = tmp_path / f"{name}.metal"
    unit.write_text(source)
    command = [
        CLANG[0],
        "-fsyntax-only",
        "-x",
        "c++",
        "-std=c++17",
        "-Wno-unknown-attributes",
        "-I",
        str(SHIM),
        str(unit),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-3000:]


@pytest.mark.parametrize(("file", "name", "reason"), REFUSED, ids=lambda v: v[:24])
def test_what_the_hardware_lacks_is_refused_with_the_reason(file: str, name: str, reason: str) -> None:
    consts = None
    with raises_literal(CaptureError, reason):
        source_of(file, name, consts)


def test_print_assert_and_breakpoint_are_refused_because_metal_has_no_printf() -> None:
    with pytest.raises(CaptureError, match="Metal Shading Language has no device printf"):
        source_of("intrinsics", "debug_aids", None)
    assert not MSLRenderer.debuggable


def test_float16_is_stored_as_half_and_computed_in_float() -> None:
    # a tile too small for a simdgroup_matrix keeps the scalar code
    source = source_of("gemm", "gemm", {"t": {"bm": 4, "bn": 4, "bk": 4}})
    assert "const device half*" in source and "(float)a[" in source


def test_a_tile_gelu_defines_the_erf_it_calls_once(tmp_path: Path) -> None:
    path = tmp_path / "gelu.py"
    path.write_text(
        "from krn import Matrix, T, block, kernel\n"
        "@kernel\ndef gelu(x: Matrix[T.f32], y: Matrix[T.f32]) -> None:\n"
        "    rows, cols = block.tile(8, 16)\n"
        "    y[rows, cols] = x[rows, cols].gelu()\n"
    )
    source = emit(capture(path, "gelu", target=APPLE), APPLE, lineinfo=False).source
    assert source.count("float _erf(float x)") == 1
    assert "(1 + _erf(" in source


@pytest.mark.parametrize("dtype", ("f16", "bf16"))
@pytest.mark.parametrize("generic", (False, True))
@pytest.mark.parametrize("expression", (False, True))
def test_matrix_operands_type_check_against_the_shim(
    tmp_path: Path, dtype: str, generic: bool, expression: bool
) -> None:
    path = matrix_source(tmp_path / "gemm.py", 32, 32, 16, dtype, generic=generic)
    if expression:
        operand = "a[rows, slice(ks, ks + 16)]" if generic else "a[rows, ks]"
        path.write_text(path.read_text().replace(operand, f"({operand} + {dtype}(0.0))"))
    source = emit(capture(path, "gemm", target=APPLE), APPLE, lineinfo=False).source
    assert "simdgroup_multiply_accumulate" in source
    unit = tmp_path / "gemm.metal"
    unit.write_text(source)
    done = subprocess.run(
        [
            CLANG[0],
            "-fsyntax-only",
            "-x",
            "c++",
            "-std=c++17",
            "-Wno-unknown-attributes",
            "-I",
            str(SHIM),
            str(unit),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]


@pytest.mark.parametrize("generation,family", [(1, 7), (2, 8), (3, 9), (4, 9), (5, 10)])
def test_apple_families(generation: int, family: int) -> None:
    target = targets.Apple.of(f"apple_m{generation}")
    assert target.family == family
    assert not target.has(targets.Feature.ATOMICS_64)


@pytest.mark.parametrize("generation", range(1, 6))
@pytest.mark.parametrize("dtype", ["i64", "u64"])
@pytest.mark.parametrize("op", ["add", "sub", "min", "max", "and", "or", "xor", "exch", "cas", "inc", "dec"])
def test_64_bit_atomics_are_refused(tmp_path: Path, generation: int, dtype: str, op: str) -> None:
    target = targets.Apple.of(f"apple_m{generation}")
    if dtype == "i64" and op in {"inc", "dec"}:
        return  # These wrapping operations only accept unsigned elements.
    function = {"and": "and_", "or": "or_"}.get(op, op)
    operands = "x, 0, 1, 2" if op == "cas" else "x, 0, 1"
    path = tmp_path / "atomic.py"
    path.write_text(
        "from krn import T, Vector, atomic, kernel\n"
        f"@kernel\ndef run(x: Vector[T.{dtype}]):\n    atomic.{function}({operands})\n"
    )
    with raises_literal(CaptureError, f"{target.arch} has no 64-bit atomics"):
        capture(path, "run", target=target)
    renderer = MSLRenderer(target)
    for shared in (False, True):
        with raises_literal(TargetError, f"{target.arch} has no 64-bit atomics"):
            renderer.atomic(AtomicOp(op), U64 if dtype == "u64" else I64, "p", "v", None, shared=shared)
    assert not renderer.helpers

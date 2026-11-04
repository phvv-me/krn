"""Every CUDA unit is header-free and compiles with NVRTC for sm_90a and sm_120a. `test_gpu.py` launches them.

The compiler is the backend's: `cuda.bindings.nvrtc` with the flags `CompileOptions` resolves, so these tests need the
cuda-python wheels and no card.
"""

from collections.abc import Callable
from dataclasses import replace

import pytest

from krn import targets
from krn.compiler.api import CaptureError, capture, emit
from krn.runtime.compile_options import CompileOptions
from tests.conftest import KERNELS, SKETCH

type Compiler = Callable[[str, tuple[str, ...]], bytes]

UNITS = [
    ("elementwise", "saxpy", None),
    ("elementwise", "add", {"tiling": {"rows": 32, "cols": 32}}),
    ("elementwise", "add", None),
    ("softmax", "softmax", {"t": {"rows": 2, "cols": 4096}}),
    ("gemm", "gemm", None),
    ("gemm", "gemm", {"t": {"bm": 64, "bn": 64, "bk": 16}}),
    ("automaton", "run_dfa", None),
    ("extra", "wrap", None),
    ("extra", "collatz", None),
    ("extra", "shadow", None),
    ("extra", "scale", None),
    *(
        ("ragged", name, None)
        for name in (
            "row_max",
            "row_min",
            "row_sum",
            "row_prod",
            "softmax_rows",
            "centered",
            "larger_target",
            "scores_max",
        )
    ),
]
TARGETS = {arch: targets.Target.of(arch) for arch in ("sm_90a", "sm_120a")}


def source_of(file: str, name: str, consts: dict[str, dict[str, int]] | None, target: targets.Target) -> str:
    path = KERNELS / f"{file}.py" if (KERNELS / f"{file}.py").exists() else SKETCH / f"{file}.py"
    return emit(capture(path, name, consts=consts, target=target), target).source


def cubin_of(compile_cubin: Compiler, source: str, target: targets.Target) -> bytes:
    return compile_cubin(source, CompileOptions.resolve(target).flags())


@pytest.mark.parametrize("arch", TARGETS)
@pytest.mark.parametrize(("file", "name", "consts"), UNITS, ids=lambda v: v if isinstance(v, str) else "")
def test_unit_is_header_free_and_compiles(
    compile_cubin: Compiler,
    file: str,
    name: str,
    consts: dict[str, dict[str, int]] | None,
    arch: str,
) -> None:
    source = source_of(file, name, consts, TARGETS[arch])
    assert "#include" not in source
    assert cubin_of(compile_cubin, source, TARGETS[arch])[:4] == b"\x7fELF"


def test_shared_memory_over_the_target_limit_is_a_capture_error() -> None:
    # 256 x 256 exceeds the register budget of a warp, so the product stays on the FMA loop with its f32 buffers
    with pytest.raises(CaptureError, match="need 393216 bytes of shared memory, nvidia has 101376"):
        source_of("gemm", "gemm", {"t": {"bm": 256, "bn": 256, "bk": 64}}, TARGETS["sm_120a"])


def test_lane_sum_loop_bound_follows_the_warp_size() -> None:
    sources = {
        w: source_of(
            "target_variables",
            "block_sum",
            None,
            replace(targets.Target.of("sm_90a"), warp_size=w),
        )
        for w in (32, 64)
    }
    assert "step < 5" in sources[32]
    assert "step < 6" in sources[64]
    assert "__shfl_xor_sync" in sources[32]


def test_block_sum_compiles(compile_cubin: Compiler) -> None:
    source = source_of("target_variables", "block_sum", None, TARGETS["sm_90a"])
    assert "#include" not in source
    assert cubin_of(compile_cubin, source, TARGETS["sm_90a"])[:4] == b"\x7fELF"


def test_target_constants_choose_the_branch() -> None:
    assert "= 32u" in source_of("extra", "lanes", None, replace(targets.Target.of("sm_90a"), warp_size=32))
    assert "= 64u" in source_of("extra", "lanes", None, replace(targets.Target.of("sm_90a"), warp_size=64))

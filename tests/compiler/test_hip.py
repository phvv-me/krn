"""The HIP renderer: every common unit, and the ones only AMD runs (64-bit atomics, `block.count`), is emitted for
gfx942 and gfx950 and compiled to AMDGPU assembly by clang's backend against `hip_shim/`. Nothing here has run on
an AMD GPU; that waits for Sirius."""

import subprocess
from pathlib import Path

import pytest

from krn import targets
from krn.compiler.api import capture, emit
from krn.runtime.cpu import CLANG
from tests.conftest import KERNELS, SKETCH
from tests.helpers.text import descriptor
from tests.helpers.units import COMMON

SHIM = Path(__file__).resolve().parents[1] / "hip_shim"
ONLY_AMD = [("intrinsics", "smallest", None), ("intrinsics", "block_votes", None)]
CLANG_HAS_AMDGPU = (
    "amdgcn" in subprocess.run([CLANG[0], "--print-targets"], capture_output=True, text=True, check=False).stdout
)


def source_of(file: str, name: str, consts: dict | None, arch: str) -> str:
    path = KERNELS / f"{file}.py" if (KERNELS / f"{file}.py").exists() else SKETCH / f"{file}.py"
    target = targets.Target.of(arch)
    return emit(capture(path, name, consts=consts, target=target), target, lineinfo=False).source


@pytest.mark.skipif(not CLANG_HAS_AMDGPU, reason="this clang has no AMDGPU backend")
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    ("file", "name", "consts"), [*COMMON, *ONLY_AMD], ids=lambda v: v if isinstance(v, str) else ""
)
def test_the_unit_compiles_to_amdgpu_assembly(
    tmp_path: Path, file: str, name: str, consts: dict | None, arch: str
) -> None:
    source = source_of(file, name, consts, arch)
    assert source.startswith("#include <hip/hip_runtime.h>") and "sm_" not in source and "asm(" not in source
    unit = tmp_path / f"{name}.hip"
    unit.write_text(source)
    command = [
        CLANG[0],
        "-x",
        "hip",
        f"--offload-arch={arch}",
        "--offload-device-only",
        "-nogpulib",
        "-nogpuinc",
    ]
    done = subprocess.run(
        [*command, "-I", str(SHIM), "-S", "-o", str(tmp_path / "unit.s"), str(unit)],
        capture_output=True, text=True, timeout=240, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    assembly = (tmp_path / "unit.s").read_text()
    assert any(
        line.strip().startswith('.amdgcn_target "amdgcn-amd-amdhsa-') and line.strip().endswith(f'-{arch}"')
        for line in assembly.splitlines()
    ), assembly[:300]
    assert descriptor(assembly, name), "the kernel has no descriptor"


def test_the_wave_is_64_lanes_wide_and_float16_is_the_compilers_own() -> None:
    assert targets.Target.of("gfx942").warp_size == 64
    source = source_of("gemm", "gemm", {"t": {"bm": 16, "bn": 16, "bk": 8}}, "gfx942")
    assert "_Float16" in source and "h2f" not in source
    assert any(name in source_of("intrinsics", "votes", None, "gfx942") for name in ("__ballot", "__shfl"))

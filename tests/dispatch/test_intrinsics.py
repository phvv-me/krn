"""The raw NVIDIA SIMD spellings: one instruction on NVIDIA, a four-part TargetError everywhere else."""

from pathlib import Path

import pytest

from krn.compiler.api import capture, emit
from krn.compiler.errors import TargetError
from krn.facts.targets import Target
from krn.intrinsics import vabsdiffs4, vaddss4, vaddus4, vmaxs4

RAW = {"vaddss4": "__vaddss4(", "vaddus4": "__vaddus4(", "vmaxs4": "__vmaxs4(", "vabsdiffs4": "__vabsdiffs4("}


@pytest.mark.parametrize("name", sorted(RAW))
def test_raw_instruction_is_spelled_on_nvidia(tmp_path: Path, name: str) -> None:
    file = tmp_path / "raw.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\n"
        "from krn.intrinsics import " + name + "\n"
        "@kernel\ndef k(a: Vector[T.u32], b: Vector[T.u32], out: Vector[T.u32]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n"
        f"        out[i] = {name}(a[i], b[i])\n"
    )
    target = Target.of("sm_120a")
    assert RAW[name] in emit(capture(file, "k", target=target), target).source


@pytest.mark.parametrize("arch", ["avx512", "apple_m4", "gfx942"])
def test_raw_instruction_refuses_other_targets_with_four_parts(tmp_path: Path, arch: str) -> None:
    file = tmp_path / "raw.py"
    file.write_text(
        "from krn import T, Vector, grid, kernel\nfrom krn.intrinsics import vaddus4\n"
        "@kernel\ndef k(a: Vector[T.u32], b: Vector[T.u32], out: Vector[T.u32]) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n        out[i] = vaddus4(a[i], b[i])\n"
    )
    with pytest.raises(TargetError) as error:
        capture(file, "k", target=Target.of(arch))
    assert error.value.feature == "krn.intrinsics.vaddus4"
    assert error.value.target == arch
    assert "one NVIDIA SIMD instruction" in error.value.reason
    assert "krn.math" in error.value.instead
    assert error.value.span is not None and error.value.span.line == 7


@pytest.mark.parametrize("function", [vaddss4, vaddus4, vmaxs4, vabsdiffs4])
def test_host_stubs_only_describe_the_instruction(function) -> None:
    with pytest.raises(NotImplementedError):
        function(1, 2)

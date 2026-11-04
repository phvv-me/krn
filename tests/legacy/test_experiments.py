"""Offline contracts for migrated evidence tooling and conformance classification."""

from pathlib import Path

import pytest

from experiments.conformance import BEGIN, END, between
from experiments.conformance_model import Role, SkipClass, architecture, classify_skip, name_role, short_reason
from experiments.gen_unicode_tables import OUTPUT
from experiments.paths import RESULTS
from experiments.scalar_semantics import next_state_c, next_state_strict, next_state_strict_widened


@pytest.mark.parametrize(
    ("function", "stem", "expected"),
    [
        ("family_cuda", "test_intrinsic_families", Role.CUDA),
        ("foo_on_the_card", "test_groups", Role.CUDA),
        ("foo_compiles", "test_intrinsics", Role.NVRTC),
        ("metal_packed_algorithm_on_c", "test_numbers", Role.C),
        ("foo_on_c", "test_numbers", Role.C),
        ("foo_c_against_numpy", "test_numbers", None),
        ("cg_emulator_sum", "test_groups", Role.EMULATOR),
        ("foo_on_the_metal", "test_groups", Role.METAL),
    ],
)
def test_conformance_role(function: str, stem: str, expected: Role | None) -> None:
    assert name_role(function, stem) == expected


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("no CUDA card", SkipClass.NO_HARDWARE),
        ("macOS framework", SkipClass.OTHER_PLATFORM),
        ("signatures only", SkipClass.BY_DESIGN),
        ("NVRTC unavailable", SkipClass.NO_TOOLCHAIN),
        ("has no f16 semantics yet", SkipClass.NOT_IMPLEMENTED),
    ],
)
def test_conformance_skip(reason: str, expected: SkipClass) -> None:
    assert classify_skip(reason) == expected


def test_remote_markers_and_reasons() -> None:
    payload = '{"label":"mac"}\n'
    assert between(f"noise\n{BEGIN}suite\n{payload}\n{END}suite\nnoise", "suite") == payload
    with pytest.raises(ValueError, match="no suite document"):
        between(f"{BEGIN}suite\npartial", "suite")
    assert short_reason("tests/runtime/test_gpu.py:12: dlopen(/lib/path)") == "no libcuda.so.1 on this host"
    assert all(architecture(name) for name in ("sm_90a", "gfx942", "apple_m5", "avx512", "avx", "spr"))
    assert not any(architecture(name) for name in ("sm_no", "gfx942a", "apple_m5a"))


def test_result_and_generator_destinations() -> None:
    root = Path(__file__).resolve().parents[2]
    assert root / "evidence" / "results" == RESULTS
    assert root / "src" / "krn" / "data" / "unicode_tables.py" == OUTPUT


def test_strict_arithmetic_widening() -> None:
    assert next_state_c(0) == -1
    assert next_state_strict(0) == 65535
    assert all(next_state_strict_widened(value) == next_state_c(value) for value in range(1 << 16))

"""The documented requirements: every row names its source, a PTX version follows its target, and a target the documents
do not cover names nothing."""

from krn._internals.compat.requires import AMD_ARCH, APPLE, CPU_ISA, NVIDIA, requirement
from krn.facts.targets import Target, all_targets


def test_every_nvidia_row_gives_the_ptx_toolkit_and_driver_with_its_source() -> None:
    for arch, requires in NVIDIA.items():
        assert requires.ptx and requires.toolkit and requires.driver, arch
        assert requires.source.startswith("https://docs.nvidia.com/"), arch


def test_nvidia_rows_are_the_documented_pairs() -> None:
    assert (NVIDIA["sm_120a"].ptx, NVIDIA["sm_120a"].toolkit, NVIDIA["sm_120a"].driver) == ("8.7", "12.8", "570.26")
    assert (NVIDIA["sm_90a"].ptx, NVIDIA["sm_90a"].toolkit) == ("8.0", "12.0")
    assert NVIDIA["sm_100a"].driver == "R565 branch"


def test_each_vendor_gives_its_own_fact() -> None:
    assert requirement(Target.of("sm_120a")) is NVIDIA["sm_120a"]
    assert requirement(Target.of("apple_m3")) == APPLE[9]
    assert requirement(Target.of("gfx942")) == AMD_ARCH["gfx942"]
    assert requirement(Target.of("spr")) == CPU_ISA["spr"]
    assert requirement(Target.of("avx2")) == CPU_ISA["avx2"]


def test_a_bare_target_names_nothing() -> None:
    bare = Target(arch="bare", warp_size=32, shared_bytes=0, features=frozenset(), storage=frozenset(), mma=frozenset())
    assert requirement(bare) is None


def test_apple_m5_has_no_documented_family_row() -> None:
    assert requirement(Target.of("apple_m5")) is None


def test_exactly_the_documented_targets_have_a_requirement_and_each_has_a_summary() -> None:
    documented = {
        target.arch
        for target in all_targets()
        if target.arch in {**NVIDIA, **AMD_ARCH, **CPU_ISA} or getattr(target, "family", None) in APPLE
    }
    stated = {t.arch: requires for t in all_targets() if (requires := requirement(t)) is not None}
    assert set(stated) == documented
    assert all(requires.summary() for requires in stated.values())

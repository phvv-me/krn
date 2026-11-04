"""The NVRTC wrapper keeps the numerical contract: every compile carries it, an option that changes it is refused under
every spelling, each option is written once, and inspection hands the compiler the flags the runtime uses. The
recording tests need no NVRTC; the PTX test skips without it."""

import re
from collections.abc import Iterator

import pytest

from krn import targets
from krn.compiler.api import capture
from krn.compiler.artifacts import inspect_kernel
from krn.compiler.debug import Artifact, Debug
from krn.compiler.errors import TargetError
from krn.runtime.compile_options import STRICT_CUDA, CompileOptions
from krn.runtime.nvrtc import Kind
from krn.runtime.nvrtc import Nvidia as Nvrtc
from tests.conftest import KERNELS
from tests.helpers.fake_nvrtc import ADVERSARIAL, CONFLICTS, SOURCE, recording_nvidia

ARCHES = ("sm_90a", "sm_120a")
FILE = KERNELS / "intrinsics.py"


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Nvrtc, list[tuple[str, ...]]]]:
    yield recording_nvidia(monkeypatch)


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("kind", ["ptx", "cubin", "ltoir"])
def test_every_compile_carries_the_strict_set(
    recording: tuple[Nvrtc, list[tuple[str, ...]]], arch: str, kind: Kind
) -> None:
    nvidia, received = recording
    nvidia.compile(SOURCE, arch, kind)
    (flags,) = received
    arch_flag = f"-arch={arch}" if kind == "ltoir" else f"--gpu-architecture={arch}"
    assert arch_flag in flags and "-default-device" in flags
    assert all(flag in flags for flag in STRICT_CUDA)
    assert ("-dlto" in flags) == (kind == "ltoir")


def test_options_the_caller_gives_are_not_repeated(recording: tuple[Nvrtc, list[tuple[str, ...]]]) -> None:
    nvidia, received = recording
    options = (*STRICT_CUDA, "-default-device", "--gpu-architecture=sm_90a", "--std=c++20")
    nvidia.compile(SOURCE, "sm_90a", "cubin", options)
    assert received == [options]


def test_required_options_are_added_after_the_caller_options(recording: tuple[Nvrtc, list[tuple[str, ...]]]) -> None:
    nvidia, received = recording
    nvidia.compile(SOURCE, "sm_90a", "cubin", ("--device-debug", "--std=c++20", "-I/opt/include", "-DFOO=1"))
    assert received == [
        (
            "--device-debug",
            "--std=c++20",
            "-I/opt/include",
            "-DFOO=1",
            "--gpu-architecture=sm_90a",
            "-default-device",
            *STRICT_CUDA,
        )
    ]


@pytest.mark.parametrize("option", CONFLICTS)
def test_an_option_that_changes_the_strict_arithmetic_is_refused(
    recording: tuple[Nvrtc, list[tuple[str, ...]]], option: str
) -> None:
    nvidia, received = recording
    with pytest.raises(TargetError) as failure:
        nvidia.compile(SOURCE, "sm_90a", "cubin", (option,))
    error = failure.value
    assert error.feature == option and error.target == "sm_90a"
    assert "krn compiles strictly" in error.reason
    assert "krn.fast.math.<op>" in error.instead and "math.fma" in error.instead
    assert "is not available on sm_90a" in str(error)
    assert received == []


@pytest.mark.parametrize("option", ["--gpu-architecture=sm_80", "-arch=sm_80", "-arch sm_80"])
def test_an_architecture_for_another_target_is_refused(
    recording: tuple[Nvrtc, list[tuple[str, ...]]], option: str
) -> None:
    nvidia, received = recording
    with pytest.raises(TargetError) as failure:
        nvidia.compile(SOURCE, "sm_90a", "cubin", (option,))
    assert failure.value.feature == option and failure.value.target == "sm_90a"
    assert "drop the option" in failure.value.instead
    assert received == []


def test_a_required_option_with_another_value_is_refused(recording: tuple[Nvrtc, list[tuple[str, ...]]]) -> None:
    nvidia, received = recording
    with pytest.raises(TargetError) as failure:
        nvidia.compile(SOURCE, "sm_90a", "cubin", ("-default-device=false",))
    assert failure.value.feature == "-default-device=false"
    assert "krn sets default-device itself" in failure.value.reason
    assert received == []


def test_each_spelling_of_a_strict_option_is_written_once_in_the_canonical_form(
    recording: tuple[Nvrtc, list[tuple[str, ...]]],
) -> None:
    nvidia, received = recording
    nvidia.compile(SOURCE, "sm_90a", "cubin", ("--fmad false", "--ftz=false", "-ftz=false", "-prec-div=true"))
    assert received == [
        (
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--gpu-architecture=sm_90a",
            "-default-device",
            "--prec-sqrt=true",
        )
    ]


@pytest.mark.parametrize(("kind", "arch_flag"), [("ptx", "--gpu-architecture=sm_90a"), ("ltoir", "-arch=sm_90a")])
def test_an_architecture_is_written_once_in_the_spelling_of_its_kind(
    recording: tuple[Nvrtc, list[tuple[str, ...]]], kind: Kind, arch_flag: str
) -> None:
    nvidia, received = recording
    nvidia.compile(SOURCE, "sm_90a", kind, ("-arch=sm_90a", "--gpu-architecture sm_90a"))
    (flags,) = received
    assert [flag for flag in flags if "arch" in flag] == [arch_flag]
    assert flags.count("-dlto") == (kind == "ltoir")


def test_an_option_without_a_dash_passes_through(recording: tuple[Nvrtc, list[tuple[str, ...]]]) -> None:
    nvidia, received = recording
    nvidia.compile(SOURCE, "sm_90a", "cubin", ("plain",))
    assert received == [("plain", "--gpu-architecture=sm_90a", "-default-device", *STRICT_CUDA)]


def test_options_that_keep_the_strict_arithmetic_pass_through(recording: tuple[Nvrtc, list[tuple[str, ...]]]) -> None:
    nvidia, received = recording
    options = (
        "--device-debug",
        "--generate-line-info",
        "--std=c++20",
        "-I/opt/include",
        "-DFOO=1",
        "--ftz=false",
        "--fmad=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        "-default-device",
        "--gpu-architecture=sm_90a",
    )
    nvidia.compile(SOURCE, "sm_90a", "cubin", options)
    assert received == [options]


@pytest.mark.parametrize(("lineinfo", "debug"), [(False, False), (True, False), (False, True)])
def test_inspection_hands_the_compiler_the_runtime_flags(
    recording: tuple[Nvrtc, list[tuple[str, ...]]], lineinfo: bool, debug: bool
) -> None:
    nvidia, received = recording
    target = targets.Target.of("sm_90a")
    kernel = capture(FILE, "histogram", target=target)
    settings = Debug(lineinfo=lineinfo, debug=debug)
    inspect_kernel(kernel, target, nvrtc=nvidia, debug=settings, artifacts=frozenset({Artifact.PTX, Artifact.CUBIN}))
    runtime = CompileOptions.resolve(target, lineinfo=settings.lineinfo, debug=settings.debug).flags()
    assert received == [runtime, runtime]


@pytest.mark.parametrize("arch", ARCHES)
def test_nvrtc_keeps_the_multiply_and_the_add_separate(nvrtc: Nvrtc, arch: str) -> None:
    ptx = nvrtc.compile(ADVERSARIAL, arch, "ptx").data.decode()
    assert "fma" not in ptx
    assert re.search(r"\bmul(\.rn)?\.f32\b", ptx) and re.search(r"\badd(\.rn)?\.f32\b", ptx)

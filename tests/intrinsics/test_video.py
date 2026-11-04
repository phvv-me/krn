"""The raw NVIDIA video SIMD and SAD instructions of `krn.intrinsics`, named as CUDA documents them: lane by lane in
Python on the emulator, the predicates of the predicated ones, the four-part refusal off NVIDIA, the spelling on NVIDIA,
and the card and NVRTC where they exist."""

from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn import Blocks
from krn.compiler.api import capture, emit
from krn.compiler.errors import TargetError
from krn.compiler.ir import Kernel
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from tests.helpers.devices import CUDARun

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice
    from krn.runtime.nvrtc import Nvidia as Nvrtc

COUNT = 64
TERNARY = [
    f"{core}_{shape}{relu}"
    for core in ("viaddmax", "viaddmin", "vimax3", "vimin3")
    for shape, relu in (("s16x2", ""), ("s16x2", "_relu"), ("u16x2", ""), ("s32", ""), ("s32", "_relu"), ("u32", ""))
]
BINARY = ["vimax_s16x2_relu", "vimax_s32_relu", "vimin_s16x2_relu", "vimin_s32_relu"]
PREDICATE = [f"{core}_{shape}" for core in ("vibmax", "vibmin") for shape in ("s16x2", "s32", "u16x2", "u32")]
SAD = ["vsads2", "vsads4", "vsadu2", "vsadu4"]
VALUE = [*TERNARY, *BINARY, *SAD]
EVERY = [*VALUE, *PREDICATE]
SHAPE = {"s16x2": (True, 16, 2), "u16x2": (False, 16, 2), "s32": (True, 32, 1), "u32": (False, 32, 1)}
LANE_WORD = st.one_of(
    st.sampled_from([0, 1, 0x7FFF7FFF, 0x80008000, 0xFFFFFFFF, 0x00010001, 0x7FFFFFFF, 0x80000000]),
    st.integers(0, 2**32 - 1),
)


def shape_of(name: str) -> tuple[bool, int, int]:
    """Signedness, lane width and lane count of a name: `s16x2` is two signed halfwords, `u32` one unsigned word."""
    return next(SHAPE[part] for part in name.split("_") if part in SHAPE)


def lanes_of(word: int, width: int, count: int, signed: bool) -> list[int]:
    values = [(word >> (width * i)) & ((1 << width) - 1) for i in range(count)]
    return [v - (1 << width) if signed and v >> (width - 1) else v for v in values]


def word_of(values: list[int], width: int) -> int:
    return sum((v & ((1 << width) - 1)) << (width * i) for i, v in enumerate(values))


def value_oracle(name: str, *operands: int) -> int:
    """One lane at a time. A sum of `viadd` wraps to its lane; `vimax3` and `vimin3` take the extreme of three; the
    two-operand forms take the extreme of two; `_relu` clamps at zero. A SAD sums the absolute lane differences."""
    if name.startswith("vsad"):
        width, count = (16, 2) if name.endswith("2") else (8, 4)
        left, right = (lanes_of(word, width, count, name.startswith("vsads")) for word in operands[:2])
        return sum(abs(x - y) for x, y in zip(left, right, strict=True))
    signed, width, count = shape_of(name)
    padded = (*operands, 0, 0, 0)[:3]
    core = name.split("_")[0]
    values = []
    for x, y, z in zip(*(lanes_of(word, width, count, signed) for word in padded), strict=True):
        if core in ("viaddmax", "viaddmin"):
            total = lanes_of(word_of([x + y], width), width, 1, signed)[0]
            picked = max(total, z) if core == "viaddmax" else min(total, z)
        elif core == "vimax3":
            picked = max(x, y, z)
        elif core == "vimin3":
            picked = min(x, y, z)
        elif core in ("vimax", "vibmax"):
            picked = max(x, y)
        else:
            picked = min(x, y)
        values.append(max(picked, 0) if name.endswith("_relu") else picked)
    return word_of(values, width)


def predicate_oracle(name: str, part: str, a: int, b: int) -> int:
    """The predicate of one lane: `a >= b` for max and `a <= b` for min. `pred_hi` reads the high halfword."""
    signed, width, count = shape_of(name)
    lane = 1 if part == "pred_hi" else 0
    x, y = (lanes_of(word, width, count, signed)[lane] for word in (a, b))
    return int(x >= y if name.startswith("vibmax") else x <= y)


def operand_count(name: str) -> int:
    return 3 if name in TERNARY else 2


def parts_of(name: str) -> tuple[str, ...]:
    return ("pred_hi", "pred_lo") if name.endswith("16x2") else ("pred",)


def element(name: str) -> str:
    """The Vector element type of a name's operands and results: i32 for the signed scalars, u32 otherwise."""
    return "i32" if "_s32" in name else "u32"


def value_source(name: str) -> str:
    types = element(name)
    letters = "abc"[: operand_count(name)]
    params = ", ".join(f"{letter}: Vector[T.{types}]" for letter in letters)
    call = ", ".join(f"{letter}[i]" for letter in letters)
    return (
        "from krn import T, Vector, grid, kernel\n"
        f"from krn.intrinsics import {name}\n"
        f"@kernel\ndef k({params}, out: Vector[T.{types}]) -> None:\n"
        "    i = grid.thread_rank()\n"
        f"    if i < out.shape[0]:\n        out[i] = {name}({call})\n"
    )


def predicate_source(name: str) -> str:
    types, parts = element(name), parts_of(name)
    flags = [f"flag_{part}" for part in parts]
    params = ", ".join(
        [f"a: Vector[T.{types}]", f"b: Vector[T.{types}]", f"out: Vector[T.{types}]"]
        + [f"{flag}: Vector[T.u32]" for flag in flags]
    )
    stores = "".join(
        f"        if {part}:\n            {flag}[i] = 1\n        else:\n            {flag}[i] = 0\n"
        for part, flag in zip(parts, flags, strict=True)
    )
    results = ", ".join(["value", *parts])
    return (
        "from krn import T, Vector, grid, kernel\n"
        f"from krn.intrinsics import {name}\n"
        f"@kernel\ndef k({params}) -> None:\n"
        "    i = grid.thread_rank()\n    if i < out.shape[0]:\n"
        f"        {results} = {name}(a[i], b[i])\n        out[i] = value\n{stores}"
    )


@cache
def captured(name: str, arch: str) -> Kernel:
    """The IR of a name's kernel on a target, captured once: the hypothesis examples of a name reuse it."""
    source = value_source(name) if name in VALUE else predicate_source(name)
    with TemporaryDirectory() as directory:
        file = Path(directory) / "kernel.py"
        file.write_text(source)
        return capture(file, "k", target=Target.of(arch))


def words(data: list[int], name: str) -> np.ndarray:
    """The operand array of a name: 32-bit patterns, read as int32 for the signed scalars."""
    dtype = np.int32 if "_s32" in name else np.uint32
    return np.array(data, np.uint64).astype(np.uint32).view(dtype)


@pytest.mark.parametrize("name", VALUE)
@settings(max_examples=10, deadline=None)
@given(
    a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT),
    b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT),
    c=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT),
)
def test_value_matches_lane_oracle_on_the_emulator(name: str, a, b, c) -> None:
    operands = [a, b, c][: operand_count(name)]
    arrays = [words(values, name) for values in operands]
    out = np.zeros(COUNT, words([0], name).dtype)
    emulate(captured(name, "sm_120a"), Target.of("sm_120a")).run(Blocks(1, threads=COUNT), [*arrays, out])
    expected = [value_oracle(name, *values) for values in zip(*operands, strict=True)]
    np.testing.assert_array_equal(out, np.array(expected, np.uint64).astype(out.dtype))


@pytest.mark.parametrize("name", PREDICATE)
@settings(max_examples=10, deadline=None)
@given(a=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT), b=st.lists(LANE_WORD, min_size=COUNT, max_size=COUNT))
def test_predicated_value_and_predicates_match_lane_oracle(name: str, a, b) -> None:
    dtype = words([0], name).dtype
    flags = {part: np.zeros(COUNT, np.uint32) for part in parts_of(name)}
    out = np.zeros(COUNT, dtype)
    emulate(captured(name, "sm_120a"), Target.of("sm_120a")).run(
        Blocks(1, threads=COUNT), [words(a, name), words(b, name), out, *flags.values()]
    )
    expected = [value_oracle(name, x, y) for x, y in zip(a, b, strict=True)]
    np.testing.assert_array_equal(out, np.array(expected, np.uint64).astype(dtype))
    for part, flag in flags.items():
        wanted = [predicate_oracle(name, part, x, y) for x, y in zip(a, b, strict=True)]
        np.testing.assert_array_equal(flag, np.array(wanted, np.uint32))


@pytest.mark.parametrize("name", EVERY)
@pytest.mark.parametrize("arch", ["avx512", "apple_m4", "gfx942"])
def test_raw_names_refuse_every_other_target_with_four_parts(tmp_path: Path, name: str, arch: str) -> None:
    file = tmp_path / "raw.py"
    file.write_text(value_source(name) if name in VALUE else predicate_source(name))
    with pytest.raises(TargetError) as error:
        capture(file, "k", target=Target.of(arch))
    assert error.value.feature == f"krn.intrinsics.{name}"
    assert error.value.target == arch
    assert "one NVIDIA SIMD instruction" in error.value.reason
    assert "krn.math" in error.value.instead


@pytest.mark.parametrize("name", VALUE)
def test_value_names_are_one_nvidia_call(name: str) -> None:
    assert f"__{name}(" in emit(captured(name, "sm_120a"), Target.of("sm_120a")).source


@pytest.mark.parametrize("name", PREDICATE)
def test_predicated_names_call_the_instruction_in_their_helpers(name: str) -> None:
    source = emit(captured(name, "sm_120a"), Target.of("sm_120a")).source
    assert f"_family_{name}_value(" in source
    assert all(f"_family_{name}_{part}(" in source for part in parts_of(name))
    assert f"__{name}(a, b, &p" in source


@pytest.mark.parametrize("name", EVERY)
@pytest.mark.parametrize("arch", ["sm_80", "sm_90a", "sm_120a"])
def test_every_name_compiles_with_nvrtc(name: str, arch: str, nvrtc: Nvrtc) -> None:
    target = Target.of(arch)
    assert nvrtc.compile(emit(captured(name, arch), target).source, arch, "cubin").data[:4] == b"\x7fELF"


@pytest.mark.parametrize("name", EVERY)
def test_every_name_runs_on_the_card(name: str, gpu: CUDADevice) -> None:
    rng = np.random.default_rng(31)
    operands = [[int(x) for x in rng.integers(0, 2**32, COUNT, dtype=np.uint64)] for _ in range(operand_count(name))]
    if name in VALUE:
        arrays = [words(values, name) for values in operands]
        out = np.zeros(COUNT, arrays[0].dtype)
        CUDARun(gpu, captured(name, "sm_120a")).run(Blocks(1, threads=COUNT), [*arrays, out])
        expected = [value_oracle(name, *values) for values in zip(*operands, strict=True)]
        np.testing.assert_array_equal(out, np.array(expected, np.uint64).astype(out.dtype))
        return
    dtype = words([0], name).dtype
    flags = {part: np.zeros(COUNT, np.uint32) for part in parts_of(name)}
    out = np.zeros(COUNT, dtype)
    arrays = [words(operands[0], name), words(operands[1], name), out, *flags.values()]
    CUDARun(gpu, captured(name, "sm_120a")).run(Blocks(1, threads=COUNT), arrays)
    expected = [value_oracle(name, x, y) for x, y in zip(*operands, strict=True)]
    np.testing.assert_array_equal(out, np.array(expected, np.uint64).astype(dtype))
    for part, flag in flags.items():
        wanted = [predicate_oracle(name, part, x, y) for x, y in zip(*operands, strict=True)]
        np.testing.assert_array_equal(flag, np.array(wanted, np.uint32))

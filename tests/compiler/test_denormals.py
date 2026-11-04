"""Which float32 operations keep their subnormals on each target, measured against the emulator.

`measure` runs every kernel of `tests/kernels/denormals.py` on the cases below through one runner and reports, per
operation, the lanes where the run differs from the emulator. The flush data of `facts/targets.py` is that result.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from krn import Items, PrecisionWarning, configure
from krn.compiler import performance
from krn.compiler.api import capture
from krn.compiler.ir import Kernel
from krn.compiler.lower import emit
from krn.emulator.emulate import build as emulate
from krn.facts import targets
from krn.facts.targets import Target
from krn.runtime.cpu import CPUProgram
from tests.helpers.devices import MetalRun

KERNELS = Path(__file__).resolve().parents[1] / "kernels" / "denormals.py"
OPERATIONS = (
    "add",
    "sub",
    "mul",
    "div",
    "sqrt",
    "rsqrt",
    "fma",
    "minimum",
    "maximum",
    "copy",
    "compare",
    "convert",
)
SMALLEST = 2.0**-149  # the smallest positive float32 subnormal
LARGEST = 2.0**-126 - 2.0**-149  # the largest float32 subnormal
NORMAL = 2.0**-126  # the smallest positive normal float32
CASES = np.array(
    [
        (SMALLEST, 1.0, 1.0),  # a subnormal operand
        (LARGEST, 2.0, 0.5),
        (1.0, SMALLEST, 1.0),  # a subnormal operand of the second position
        (2.0, LARGEST, 1.0),
        (1.0, 1.0, SMALLEST),  # a subnormal third operand
        (NORMAL, 0.5, 0.0),  # normal operands whose product is the subnormal 2**-127
        (-NORMAL, 0.5, -0.0),
        (NORMAL, 2.0, 0.0),  # and whose quotient is 2**-127
        (-NORMAL, 2.0, 0.0),
        (NORMAL, 0.75 * NORMAL, 0.0),  # and whose difference is the subnormal 2**-128
        (SMALLEST, 0.0, 0.0),  # a comparison with zero, and the maximum of a subnormal and zero
        (-SMALLEST, 0.0, 0.0),  # the negative subnormal, the minimum of it and zero
        (SMALLEST, -1.0, 0.0),  # the minimum of a positive subnormal and a negative normal
    ],
    np.float32,
)


@dataclass(frozen=True, kw_only=True)
class Outcome:
    """How one operation's run differs from the emulator. `result_lanes` are the subnormal results the emulator
    computes; of those, `zeroed` the run returns as zero, and `sign_kept` of the zeros keep the result's sign.
    A flush of an operand alone (a normal result that changes) shows as lanes with no result lanes."""

    lanes: int
    result_lanes: int
    zeroed: int
    sign_kept: int

    @property
    def flushes(self) -> bool:
        return self.lanes > 0

    @property
    def mode(self) -> str:
        if not self.flushes:
            return "preserved"
        if not self.result_lanes:
            return "operand"
        if self.sign_kept == self.result_lanes:
            return "to the sign"
        return "to +0" if not self.sign_kept and self.zeroed == self.result_lanes else "mixed"


@dataclass(frozen=True, kw_only=True)
class CPURun:
    """A compiled C-backend program for `target`, run like the device runners: arrays are written in place."""

    kernel: Kernel
    target: Target

    def run(self, launch: Items, values: list[np.ndarray]) -> None:
        CPUProgram(self.kernel, emit(self.kernel, self.target))(*values, launch=launch)


def outcome(expected: np.ndarray, actual: np.ndarray) -> Outcome:
    differs = expected.view(np.uint32) != actual.view(np.uint32)
    subnormal = (expected != 0) & (np.abs(expected) < NORMAL)
    results = differs & subnormal
    zeroed = results & (actual == 0)
    sign_kept = zeroed & (np.signbit(actual) == np.signbit(expected))
    return Outcome(
        lanes=int(differs.sum()),
        result_lanes=int(results.sum()),
        zeroed=int(zeroed.sum()),
        sign_kept=int(sign_kept.sum()),
    )


def measure(target: Target, runner: Callable[[Kernel], MetalRun | CPURun]) -> dict[str, Outcome]:
    """The outcome of each operation of `OPERATIONS` on `target`, with `runner` as the device."""
    a, b, c = (np.ascontiguousarray(column) for column in CASES.T)
    launch = Items(len(a))
    found: dict[str, Outcome] = {}
    for name in OPERATIONS:
        kernel = capture(KERNELS, f"op_{name}", target=target)
        expected, actual = np.zeros_like(a), np.zeros_like(a)
        with np.errstate(all="ignore"):
            emulate(kernel, target).run(launch, [a.copy(), b.copy(), c.copy(), expected])
            runner(kernel).run(launch, [a.copy(), b.copy(), c.copy(), actual])
        found[name] = outcome(expected, actual)
    return found


def flushed(found: dict[str, Outcome]) -> frozenset[str]:
    """The operations of a measurement whose run differs from the emulator on a subnormal."""
    return frozenset(name for name, result in found.items() if result.flushes)


def write(tmp_path: Path, body: str) -> Path:
    """A kernel file `run` over float32 vectors `x` and `out`, with `body` as its statements."""
    file = tmp_path / "run.py"
    file.write_text(
        "from krn import T, Vector, kernel, math\n"
        "@kernel\ndef run(x: Vector[T.f32], out: Vector[T.f32]) -> None:\n" + body
    )
    return file


def warned(file: Path, target: Target) -> list[str]:
    """The denormal warnings `run` of `file` emits on `target`, as `operation` names in the order they are emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PrecisionWarning)
        emit(capture(file, "run", target=target), target)
    return [str(w.message).split(": ", 1)[1].split()[0] for w in caught if issubclass(w.category, PrecisionWarning)]


def test_the_flush_data_of_every_row_is_the_measured_one() -> None:
    apple = [row.denormal_flushes for row in targets.all_targets() if row.vendor == "apple"]
    assert len(set(apple)) == 1 and apple[0] is not None and apple[0] <= set(OPERATIONS)
    for row in targets.all_targets():
        if row.vendor in {"nvidia", "cpu"}:
            assert row.denormal_flushes == frozenset(), row.arch
        elif row.vendor == "amd":
            assert row.denormal_flushes is None, row.arch


def test_the_c_backend_keeps_every_float32_subnormal() -> None:
    target = targets.CPU.of("avx512")
    assert flushed(measure(target, lambda kernel: CPURun(kernel=kernel, target=target))) == frozenset()


def test_the_warning_fires_once_per_operation_on_a_flushing_target(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    body = "    for i in range(x.shape[0]):\n        out[i] = x[i] * x[i] + x[i]\n        out[i] = x[i] * 2.0\n"
    file = write(tmp_path, body)
    target = targets.Apple.of("apple_m4")
    assert sorted(warned(file, target)) == ["add", "mul"]
    assert warned(file, target) == []


def test_the_warning_text_says_the_emulator_differs_and_how_to_silence_it(tmp_path: Path) -> None:
    file = write(tmp_path, "    out[0] = x[0] / x[1]\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PrecisionWarning)
        emit(capture(file, "run", target=targets.Apple.of("apple_m4")), targets.Apple.of("apple_m4"))
    (message,) = [str(w.message) for w in caught]
    assert message.startswith("run on apple_m4: div flushes float32 subnormal operands and results on apple_m4")
    assert "differs from the emulator for subnormal inputs" in message
    assert "warn.warn_denormal_flush" in message


@pytest.mark.parametrize("arch", ["avx512", "sm_120a"])
def test_the_warning_is_silent_where_denormals_hold(tmp_path: Path, monkeypatch, arch: str) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    assert warned(write(tmp_path, "    out[0] = x[0] + x[1] * x[2]\n"), targets.Target.of(arch)) == []


def test_the_comparison_and_the_square_root_warn_too(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    file = write(tmp_path, "    out[0] = math.sqrt(x[0]) if x[0] > 0.0 else 0.0\n")
    assert sorted(warned(file, targets.Apple.of("apple_m5"))) == ["compare", "sqrt"]


def test_a_target_warns_only_for_the_operations_its_data_names(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    file = write(tmp_path, "    out[0] = x[0] / x[1] + x[2]\n")
    partial = replace(targets.Apple.of("apple_m4"), denormal_flushes=frozenset({"div"}))
    assert warned(file, partial) == ["div"]


def test_the_warning_is_silenced_by_its_setting(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    file = write(tmp_path, "    out[0] = x[0] / x[1]\n")
    with configure(**{"warn.warn_denormal_flush": False}):
        assert warned(file, targets.Apple.of("apple_m4")) == []
    assert warned(file, targets.Apple.of("apple_m4")) == ["div"]

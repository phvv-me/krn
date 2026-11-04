"""`lib.matmul` end to end on the CPU backend: the dispatcher picks a candidate, the candidate multiplies, numpy checks.

`KRN_LIB=ours` must work with nothing else installed, since it is the configuration every example runs under; the
vendor adapters (Accelerate here, cuBLASLt and the rest in `test_lib_vendors.py`) are checked against the same numpy."""

import contextlib
import math
import warnings
from collections.abc import Sequence
from dataclasses import fields
from itertools import permutations
from pathlib import Path
from time import sleep

import numpy as np
import pytest

from krn import PrecisionWarning, T, configure, lib
from krn.compiler import performance
from krn.compiler.api import CPUDevice
from krn.compiler.errors import Span, TargetError
from krn.data import sparse_build
from krn.data.sparse import Csr
from krn.facts.targets import Target
from krn.lib.lib import Epilogue, NoCandidate
from krn.runtime.api import Stream

RNG = np.random.default_rng(23)
M, N, K = 19, 23, 21  # none is a multiple of the tile, so the ragged edge is part of every product
TIMEOUT = 60


def erf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf, otypes=[np.float64])(x)


REFERENCE = {
    Epilogue.NONE: lambda x, bias: x,
    Epilogue.BIAS: lambda x, bias: x + bias,
    Epilogue.RELU: lambda x, bias: np.maximum(x, 0),
    Epilogue.GELU: lambda x, bias: 0.5 * x * (1 + erf(x / math.sqrt(2))),
    Epilogue.BIAS_RELU: lambda x, bias: np.maximum(x + bias, 0),
    Epilogue.BIAS_GELU: lambda x, bias: (lambda y: 0.5 * y * (1 + erf(y / math.sqrt(2))))(x + bias),
    Epilogue.SILU: lambda x, bias: x / (1 + np.exp(-x)),
}


def operands(dtype, device: CPUDevice):
    a, b, bias = device.empty(dtype, M, K), device.empty(dtype, K, N), device.empty(dtype, N)
    a.array[:], b.array[:], bias.array[:] = (RNG.standard_normal(size) for size in ((M, K), (K, N), N))
    return a, b, bias


@pytest.mark.parametrize("epilogue", Epilogue)
@pytest.mark.parametrize("dtype", [T.f32, T.f16])
def test_ours_multiplies_with_every_epilogue(library, device: CPUDevice, epilogue: Epilogue, dtype) -> None:
    library("ours")
    a, b, bias = operands(dtype, device)
    out = lib.matmul(a, b, epilogue=epilogue, bias=bias if "bias" in epilogue.value else None)
    device.default_stream.record().wait(TIMEOUT)
    x = a.array.astype(np.float32) @ b.array.astype(np.float32)
    want = REFERENCE[epilogue](x, bias.array.astype(np.float32))
    tolerance = 3e-3 if dtype is T.f16 else 2e-4
    np.testing.assert_allclose(np.asarray(out).astype(np.float32), want, rtol=tolerance, atol=tolerance)


def test_the_filter_keeps_only_candidates_of_the_origin_asked_for(library, device: CPUDevice) -> None:
    library("ours")
    assert (
        lib.dispatcher()
        .choose(
            lib.MatmulCandidate,
            lib.Problem(lib.Op.MATMUL, (T.f32,) * 3, (M, N, K), lib.LibOptions.of(device.target)),
            capturing=False,
        )
        .name
        == "ours"
    )
    library("vendor")
    for candidate in lib.dispatcher().candidates[lib.Op.MATMUL]:
        assert candidate.name != "ours" or candidate.origin is not lib.Origin.VENDOR
    silu = lib.Problem(lib.Op.MATMUL, (T.f32,) * 3, (M, N, K), lib.LibOptions.of(device.target), Epilogue.SILU)
    with pytest.raises(NoCandidate):  # the reason it lists depends on which vendor libraries this machine has
        lib.dispatcher().choose(lib.MatmulCandidate, silu, capturing=False)


def test_a_bad_value_raises_instead_of_being_ignored(library, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="bogus"):
        configure(lib="bogus")


class Fake:
    name = "fake"
    origin = lib.Origin.HOST
    version = "fallback"

    def __init__(self, *, refusal: str | None = None, unavailable: str | None = None, delay: float = 0) -> None:
        self.refused, self.missing, self.delay = refusal, unavailable, delay
        self.calls: list[lib.OperationOperands] = []
        self.release = "runtime-1"

    def refusal(self, problem: lib.Problem, target: Target) -> str | None:
        return self.refused

    def unavailable(self, target: Target) -> str | None:
        return self.missing

    def library_version(self) -> str:
        return self.release

    def run(self, problem: lib.Problem, operands: lib.OperationOperands, stream: Stream) -> None:
        if self.missing or self.refused:
            raise AssertionError("an unavailable or refused candidate must never run")
        self.calls.append(operands)
        np.asarray(operands.out)[...] = 42
        sleep(self.delay)


class FakeMatmul(Fake, lib.MatmulCandidate):
    op = lib.Op.MATMUL


class FakeSpmv(Fake, lib.SpmvCandidate):
    op = lib.Op.SPMV


class FakeSpmm(Fake, lib.SpmmCandidate):
    op = lib.Op.SPMM


@pytest.fixture(params=[FakeMatmul, FakeSpmv, FakeSpmm], ids=["matmul", "spmv", "spmm"])
def operation(request: pytest.FixtureRequest, device: CPUDevice):
    kind = request.param
    if kind.op is lib.Op.MATMUL:
        a, b, bias = operands(T.f32, device)
        args = lib.Operands(a, b, device.empty(T.f32, M, N), bias)
        shape, layout = (M, N, K), ""
    else:
        matrix = sparse_build.csr(device, np.array([[0, 2, 0], [3, 0, 4]], dtype=np.float32))
        assert isinstance(matrix, Csr)
        if kind.op is lib.Op.SPMV:
            args = lib.SpmvOperands(matrix, device.empty(T.f32, 3), device.empty(T.f64, 2))
            shape = (2, 3, 3)
        else:
            args = lib.SpmmOperands(matrix, device.empty(T.f32, 3, 4), device.empty(T.f64, 2, 4))
            shape = (2, 3, 3, 4)
        np.asarray(args.x)[...] = 7
        layout = "Csr"
    np.asarray(args.out)[...] = -17
    problem = lib.Problem(
        kind.op,
        (T.f32, T.f32, args.out.dtype),
        shape,
        lib.LibOptions.of(device.target),
        layout=layout,
    )
    return kind, problem, args


def test_refusals_and_unavailability_fall_through_in_every_order(
    operation, device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kind, problem, _ = operation
    refused, missing, working = (
        kind(refusal="unsupported problem"),
        kind(unavailable="backend missing"),
        kind(),
    )
    for candidate, name in zip((refused, missing, working), ("refused", "missing", "working"), strict=True):
        monkeypatch.setattr(candidate, "name", name, raising=False)
    dispatch = lib.Dispatcher(
        device.target,
        {},
        lib.TuningDb(tmp_path / "t.json"),
        lib.Mode.AUTO,
        {},
        "fingerprint",
        "off",
    )
    for order in permutations((refused, missing, working)):
        monkeypatch.setitem(lib.PREFERENCE, (problem.op, device.target.vendor), tuple(c.name for c in order))
        dispatch.candidates = {problem.op: tuple(reversed(order))}
        assert dispatch.choose(kind, problem, capturing=False) is working
    dispatch.candidates = {problem.op: (refused, missing)}
    with pytest.raises(NoCandidate) as error:
        dispatch.choose(kind, problem, capturing=False)
    assert "refused: unsupported problem" in str(error.value)
    assert "missing: backend missing" in str(error.value)
    assert not refused.calls and not missing.calls


def test_tuning_uses_caller_inputs_and_a_private_output_for_every_op(
    operation, device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kind, problem, args = operation
    slow, fast = kind(delay=0.01), kind(delay=0.001)
    for candidate, name in ((slow, "slow"), (fast, "fast")):
        monkeypatch.setattr(candidate, "name", name, raising=False)
    monkeypatch.setitem(lib.PREFERENCE, (problem.op, device.target.vendor), ("slow", "fast"))
    dispatch = lib.Dispatcher(
        device.target,
        {problem.op: (slow, fast)},
        lib.TuningDb(tmp_path / "t.json"),
        lib.Mode.AUTO,
        {},
        "fingerprint",
        "eager",
    )
    before = np.asarray(args.out).tobytes()
    assert dispatch.choose(kind, problem, capturing=False, operands=args) is slow
    assert len(dispatch.measuring) == 1
    dispatch.measuring[0].join(timeout=10)
    assert not dispatch.measuring[0].is_alive()
    assert list(dispatch.tuning.winners.values()) == ["fast"]
    assert lib.TuningDb(dispatch.tuning.path).winners == dispatch.tuning.winners
    for candidate in (slow, fast):
        assert len(candidate.calls) >= 11  # the warm-up, a probe and at least ten timed runs
        for measured in candidate.calls:
            assert measured.out is not args.out
            assert measured.out.dtype is args.out.dtype and measured.out.shape == args.out.shape
            assert all(getattr(measured, f.name) is getattr(args, f.name) for f in fields(args) if f.name != "out")
    assert np.asarray(args.out).tobytes() == before
    assert dispatch.choose(kind, problem, capturing=False, operands=args) is fast
    assert len(dispatch.measuring) == 1


def test_fingerprint_uses_the_runtime_version_and_preserves_the_literal_fallback() -> None:
    candidate = FakeMatmul()
    found = {lib.Op.MATMUL: [candidate]}
    assert lib.fingerprint(found) == "fake=runtime-1"
    candidate.release = "runtime-2"
    assert lib.fingerprint(found) == "fake=runtime-2"
    assert lib.Candidate.library_version(candidate) == "fallback"


def test_tuning_starts_only_with_operands_outside_capture_at_the_requested_call(
    operation, device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kind, problem, args = operation
    measured: list[lib.OperationOperands] = []

    def measure(
        self: lib.Dispatcher,
        key: str,
        problem: lib.Problem,
        pool: Sequence[lib.Candidate],
        operands: lib.OperationOperands,
    ) -> None:
        measured.append(operands)

    monkeypatch.setattr(lib.Dispatcher, "_measure", measure)
    for tune, capturing, supplied, calls, expected in (
        ("off", False, True, 8, 0),
        ("eager", True, True, 1, 0),
        ("eager", False, False, 1, 0),
        ("eager", False, True, 1, 1),
        ("lazy", False, True, 7, 0),
        ("lazy", False, True, 8, 1),
    ):
        dispatch = lib.Dispatcher(
            device.target,
            {problem.op: (kind(), kind())},
            lib.TuningDb(tmp_path / "t.json"),
            lib.Mode.AUTO,
            {},
            "f",
            tune,
        )
        measured.clear()
        for _ in range(calls):
            dispatch.choose(kind, problem, capturing=capturing, operands=args if supplied else None)
        for thread in dispatch.measuring:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert len(dispatch.measuring) == len(measured) == expected
        assert all(value is args for value in measured)


def test_tuning_keeps_static_preference_when_the_difference_is_noise(
    operation, device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kind, problem, args = operation
    first, second = kind(), kind()
    monkeypatch.setattr(first, "name", "first")
    monkeypatch.setattr(second, "name", "second")
    dispatch = lib.Dispatcher(
        device.target,
        {problem.op: (first, second)},
        lib.TuningDb(tmp_path / "t.json"),
        lib.Mode.AUTO,
        {},
        "f",
    )

    async def times(
        self: lib.Dispatcher,
        problem: lib.Problem,
        pool: Sequence[lib.Candidate],
        operands: lib.OperationOperands,
    ) -> dict[str, float]:
        return {"first": 1, "second": _duration}

    monkeypatch.setattr(lib.Dispatcher, "_time_all", times)
    for _duration, winner in ((1.1, "first"), (0.95, "first"), (0.9, "second"), (0.5, "second")):
        dispatch._measure("key", problem, (first, second), args)
        assert dispatch.tuning.winners["key"] == winner


class Contender(FakeMatmul):
    contender = True


class Declining(FakeMatmul):
    def run(self, problem: lib.Problem, operands: lib.OperationOperands, stream: Stream) -> None:
        self.calls.append(operands)
        raise lib.Declined("the library answered NOT_SUPPORTED")


class Uncapturable(FakeMatmul):
    def capturable(self, problem: lib.Problem, operands: lib.OperationOperands | None) -> bool:
        return False


def dispatcher_of(device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *candidates, tune="off"):
    """A dispatcher over fakes named after their position; the static order is the order given."""
    for position, candidate in enumerate(candidates):
        monkeypatch.setattr(candidate, "name", f"c{position}", raising=False)
    monkeypatch.setitem(lib.PREFERENCE, (lib.Op.MATMUL, device.target.vendor), tuple(f"c{i}" for i in range(9)))
    found = {lib.Op.MATMUL: candidates}
    return lib.Dispatcher(device.target, found, lib.TuningDb(tmp_path / "t.json"), lib.Mode.AUTO, {}, "f", tune)


def matmul_problem(device: CPUDevice, dtypes=(T.f32,) * 3) -> tuple[lib.Problem, lib.Operands]:
    a, b = device.empty(dtypes[0], M, K), device.empty(dtypes[1], K, N)
    args = lib.Operands(a, b, device.empty(dtypes[2], M, N))
    return lib.Problem(lib.Op.MATMUL, dtypes, (M, N, K), lib.LibOptions.of(device.target)), args


def test_a_contender_wins_only_the_tuning_never_the_static_order(
    device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, args = matmul_problem(device)
    main, variant = FakeMatmul(delay=0.01), Contender(delay=0.001)
    dispatch = dispatcher_of(device, tmp_path, monkeypatch, main, variant, tune="eager")
    assert dispatch.choose(lib.MatmulCandidate, problem, capturing=False) is main
    assert dispatch.choose(lib.MatmulCandidate, problem, capturing=False, operands=args) is main
    dispatch.measuring[0].join(timeout=10)
    assert list(dispatch.tuning.winners.values()) == ["c1"]
    assert dispatch.choose(lib.MatmulCandidate, problem, capturing=False) is variant
    only = dispatcher_of(device, tmp_path, monkeypatch, Contender())
    with pytest.raises(NoCandidate):
        only.choose(lib.MatmulCandidate, problem, capturing=False)


def test_a_candidate_that_declines_while_planning_falls_to_the_next_and_is_remembered(
    device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, args = matmul_problem(device)
    declining, working = Declining(), FakeMatmul()
    dispatch = dispatcher_of(device, tmp_path, monkeypatch, declining, working)
    dispatch.launch(lib.MatmulCandidate, problem, args, device.default_stream)
    dispatch.launch(lib.MatmulCandidate, problem, args, device.default_stream)
    assert len(declining.calls) == 1 and len(working.calls) == 2
    alone = dispatcher_of(device, tmp_path, monkeypatch, Declining())
    with pytest.raises(NoCandidate, match="c0: the library answered NOT_SUPPORTED"):
        alone.launch(lib.MatmulCandidate, problem, args, device.default_stream)


def test_a_capture_keeps_only_candidates_that_can_enqueue_inside_it(
    device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, _ = matmul_problem(device)
    first, second = Uncapturable(), FakeMatmul()
    dispatch = dispatcher_of(device, tmp_path, monkeypatch, first, second)
    assert dispatch.choose(lib.MatmulCandidate, problem, capturing=False) is first
    assert dispatch.choose(lib.MatmulCandidate, problem, capturing=True) is second
    with pytest.raises(NoCandidate, match="cannot enqueue while a graph is captured"):
        dispatcher_of(device, tmp_path, monkeypatch, Uncapturable()).choose(
            lib.MatmulCandidate, problem, capturing=True
        )


def test_what_a_problem_accepts_is_worked_out_once(
    device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, _ = matmul_problem(device)
    asked = []
    candidate = FakeMatmul()
    monkeypatch.setattr(candidate, "refusal", lambda problem, target: asked.append(problem))
    dispatch = dispatcher_of(device, tmp_path, monkeypatch, candidate)
    for _ in range(5):
        dispatch.choose(lib.MatmulCandidate, problem, capturing=False)
    assert len(asked) == 1
    other, _ = matmul_problem(device, (T.f16,) * 3)
    dispatch.choose(lib.MatmulCandidate, other, capturing=False)
    assert len(asked) == 2


def test_operands_carry_a_type_for_each_role(device: CPUDevice) -> None:
    problem, args = matmul_problem(device, (T.f16, T.f16, T.f32))
    private = args.private(device)
    assert (private.a, private.b) == (args.a, args.b) and private.out is not args.out
    assert private.out.dtype is T.f32 and problem.dtypes == (T.f16, T.f16, T.f32)


def test_the_tuner_gives_every_candidate_a_turn_that_is_not_the_first(
    device: CPUDevice, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem, args = matmul_problem(device)
    first, second, third = FakeMatmul(), FakeMatmul(), FakeMatmul()
    dispatch = dispatcher_of(device, tmp_path, monkeypatch, first, second, third)
    turns: list[str] = []
    slow_when_first = {"c0": (30.0, 12.0), "c1": (10.0, 10.0), "c2": (9.0, 14.0)}

    async def window(self: lib.Dispatcher, candidate, problem, operands, stream) -> float | None:
        turns.append(candidate.name)
        return None if candidate.name == "c2" and len(turns) > 3 else slow_when_first[candidate.name][len(turns) > 3]

    monkeypatch.setattr(lib.Dispatcher, "_window", window)
    times = lib.asyncio.run(dispatch._time_all(problem, [first, second, third], args))
    assert turns == ["c0", "c1", "c2", "c2", "c1", "c0"]
    assert times == {
        "c0": 12.0,
        "c1": 10.0,
        "c2": 9.0,
    }  # the best of the two turns, a declined turn does not count


def reduced_matmul(a, b) -> None:
    lib.matmul(a, b, precision=lib.Compute.TF32)


def test_reduced_accumulation_warns_once_per_call_site_operation_and_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    here, there = Span(__file__, 1, 1), Span(__file__, 2, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PrecisionWarning)
        lib._warn_reduced_accumulation(lib.Compute.TF32, "apple-m4", here)
        lib._warn_reduced_accumulation(lib.Compute.TF32, "apple-m4", here)
        assert len(caught) == 1
        lib._warn_reduced_accumulation(lib.Compute.TF32, "apple-m4", there)
        lib._warn_reduced_accumulation(lib.Compute.EMULATED, "apple-m4", here)
        lib._warn_reduced_accumulation(lib.Compute.TF32, "nvidia-sm90", here)
    assert len(caught) == 4
    assert all(issubclass(w.category, PrecisionWarning) for w in caught)
    assert str(caught[0].message).startswith(f"matmul precision={lib.Compute.TF32} on apple-m4: ")


def test_reduced_accumulation_raises_when_warnings_are_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        with pytest.raises(PrecisionWarning):
            lib._warn_reduced_accumulation(lib.Compute.TF32, "apple-m4", Span(__file__, 1, 1))


def test_matmul_warns_once_per_line_and_points_at_the_caller(
    monkeypatch: pytest.MonkeyPatch, device: CPUDevice
) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    a, b = device.empty(T.f32, M, K), device.empty(T.f32, K, N)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PrecisionWarning)
        for _ in range(2):
            with contextlib.suppress(NoCandidate, TargetError):  # dispatch refuses TF32 after the warning fires
                reduced_matmul(a, b)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert caught[0].lineno == reduced_matmul.__code__.co_firstlineno + 1

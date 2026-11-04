import dis
from dataclasses import FrozenInstanceError, replace

import pytest

from krn import Blocks, T, Vector, grid, kernel
from krn.compiler.api import CPUDevice
from krn.profile import DeviceEvidence, Feature, Profiler, launches
from krn.profile.launches import LaunchRecord
from krn.runtime.runtime import Call, CPUStream


@kernel
def twice(x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    y[i] = x[i] * 2.0


def test_runtime_capture_and_bound_reuse(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    x.array.fill(3)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    bound.submit().wait()
    with Profiler() as profiler:
        bound()
        bound()
        answer, window = Profiler.capture(lambda: (bound(), 7)[1])
        assert answer == 7
        assert len(window.launches) == 1
    result = profiler.result()
    assert len(result.launches) == 3
    assert result.device_evidence is DeviceEvidence.COLLECTED
    for record in result.launches:
        assert record.name == "twice"
        assert record.geometry == (2, 1, 1, 16)
        assert record.dtypes == ("f32", "f32")
        assert record.target == device.target.arch
        assert record.duration_ns is not None and record.duration_ns > 0
        assert record.start_ns is not None and record.host_enqueue_ns <= record.start_ns
        assert record.timing == "cpu-wall"
        assert (record.flops, record.bytes_moved) == (32, 256)
    assert (y.array == 6).all()
    assert launches.session is None


def test_cpu_direct_execution_retention_and_window_loss(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    bound = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)
    with Profiler(max_spans=1) as profiler:
        profiler.cpu_run(bound._call, device.default_stream)
        with pytest.raises(RuntimeError, match="overwritten"):
            Profiler.capture(lambda: (bound(), bound()))
    assert profiler.result().dropped_launches == 2


def test_cpu_graph_replay_records_each_kernel(device: CPUDevice) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    launch = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)

    def work() -> None:
        launch()
        launch()

    graph = device.graph(work)
    with Profiler() as profiler:
        graph()
    assert len(profiler.result().launches) == 2


def test_inactive_hook_has_no_call_or_allocation_before_none_branch() -> None:
    assert launches.session is None
    # Bytecode is the stable evidence for the hot path: a module load, attribute,
    # and a None branch precede all active-only calls and allocations.
    for method in (Call.run, CPUStream.enqueue):
        ops = list(dis.get_instructions(method))
        branch = next(index for index, op in enumerate(ops) if "IF" in op.opname and "NONE" in op.opname)
        assert all(op.opname not in ("CALL", "BUILD_TUPLE", "BUILD_LIST", "BUILD_MAP") for op in ops[:branch])


def test_launch_validation_and_frozen() -> None:
    record = LaunchRecord(name="x", geometry=(1, 1, 1, 32), dtypes=(), target="cpu", stream=1, host_enqueue_ns=0)
    assert record.duration_ns is None
    assert replace(record, elapsed_ns=5).duration_ns == 5
    with pytest.raises(FrozenInstanceError):
        record.__setattr__("name", "changed")
    for invalid in (
        lambda: replace(record, geometry=(-1, 1, 1, 1)),
        lambda: replace(record, geometry=(1, 1, 1, 0)),
        lambda: replace(record, host_enqueue_ns=-1),
        lambda: replace(record, flops=-1),
        lambda: replace(record, bytes_moved=-1),
        lambda: replace(record, start_ns=3),
        lambda: replace(record, end_ns=4),
        lambda: replace(record, start_ns=4, end_ns=3),
    ):
        with pytest.raises(ValueError):
            invalid()


def test_capture_contract_and_cleanup() -> None:
    with pytest.raises(ValueError):
        Profiler.capture(lambda: None, activities=Profiler.Activity.RUNTIME)
    with Profiler(features=Feature.SPANS) as profiler:
        with pytest.raises(RuntimeError, match="entered twice"):
            profiler.__enter__()
        with pytest.raises(RuntimeError, match="did not request"):
            Profiler.capture(lambda: None)
        with pytest.raises(RuntimeError, match="only one"), Profiler(features=Feature.SPANS):
            pass
    assert launches.session is None


def test_arithmetic_counts_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="work counts"):
        LaunchRecord(
            name="x",
            geometry=(1, 1, 1, 1),
            dtypes=(),
            target="cpu",
            stream=1,
            host_enqueue_ns=0,
            flops_by_dtype={"f32": -1},
        )

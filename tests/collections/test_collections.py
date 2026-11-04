"""Dense Python set/dict oracles over every available execution route."""

import ast
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from krn import Items, Tensor, host, targets
from krn.collections import CapacityError, Inserted, Map, Set, engine
from krn.compiler.api import CPUDevice, capture, emit
from krn.compiler.dtypes import U32
from krn.emulator.emulate import build as emulate
from krn.frontend.types import u32
from krn.runtime.cpu import CPUProgram, Value
from tests.conftest import build_kernel
from tests.helpers.devices import CUDARun, bindings

if TYPE_CHECKING:
    from krn.compiler.ir import Kernel
    from krn.runtime.cuda_backend import CUDADevice
    from tests.helpers.devices import MetalRun

FILE = Path(engine.__file__)
type Run = Callable[[str, list[Value]], None]
CASES = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


@pytest.mark.parametrize("field", ["inserted", "existing", "failed"])
def test_inserted_validation(field: str) -> None:
    values = {"inserted": 0, "existing": 0, "failed": 0, field: -1}
    with pytest.raises(ValueError, match="non-negative"):
        Inserted(**values)


@pytest.fixture(params=["source", "emulator", "c", "cuda", "metal"])
def run(request: pytest.FixtureRequest) -> Run:
    if request.param == "source":
        namespace = dict(vars(engine))
        namespace.update(u32=np.uint32, u64=np.uint64, items=range, atomic=SourceAtomics(), kernel=lambda fn: fn)
        for file in (FILE.parent.parent / "hashlib.py", FILE):
            tree = ast.parse(file.read_text(), filename=str(file))
            tree.body = [node for node in tree.body if not isinstance(node, ast.Import | ast.ImportFrom | ast.ClassDef)]
            # Keep decorator locations: coverage uses them as function exit arc destinations.
            exec(compile(tree, str(file), "exec"), namespace)
        functions = {name: fn for name, fn in namespace.items() if isinstance(fn, FunctionType)}

        def source(name: str, arguments: list[Value]) -> None:
            with np.errstate(over="ignore"):
                functions[name](*arguments)

        return source

    @cache
    def program(name: str, dtypes: tuple[tuple[str, str], ...]):
        dtype_bindings = bindings(FILE, name, arguments_for_types[dtypes])
        if request.param == "c":
            return build_kernel(FILE, name, dtypes=dtype_bindings)
        if request.param == "cuda":
            gpu: CUDADevice = request.getfixturevalue("gpu")
            return CUDARun(gpu, capture(FILE, name, dtypes=dtype_bindings, target=gpu.target))
        if request.param == "metal":
            launcher: Callable[[Kernel], MetalRun] = request.getfixturevalue("metal_program")
            return launcher(capture(FILE, name, dtypes=dtype_bindings, target=targets.Apple.of("apple_m4")))
        target = targets.CPU.of("avx512")
        return emulate(capture(FILE, name, dtypes=dtype_bindings, target=target), target)

    arguments_for_types: dict[tuple[tuple[str, str], ...], list[Value]] = {}

    def execute(name: str, arguments: list[Value]) -> None:
        if request.param == "metal" and any(
            isinstance(a, np.ndarray) and a.dtype in (np.dtype("u8"), np.dtype("i8")) for a in arguments
        ):
            pytest.skip("Metal has no 64-bit atomics beyond unsigned min/max")
        signature = tuple((str(i), str(a.dtype)) for i, a in enumerate(arguments) if isinstance(a, np.ndarray))
        arguments_for_types[signature] = arguments
        executable = program(name, signature)
        batch = arguments[2] if name in ("add", "update", "contains") else arguments[-1]
        assert isinstance(batch, np.ndarray)
        launch = Items(len(batch))
        if isinstance(executable, CPUProgram):
            executable(*arguments, launch=launch)
        else:
            executable.run(launch, arguments)

    return execute


class SourceAtomics:
    """Sequential atomic operations for coverage of the unmodified kernel bodies."""

    @staticmethod
    def cas(array: np.ndarray, index: int, expected: int, value: int) -> np.generic:
        old = array[index].copy()
        if old == expected:
            array[index] = value
        return old

    @staticmethod
    def max(array: np.ndarray, index: int, value: int) -> None:
        array[index] = max(array[index], value)


@pytest.fixture(params=["c", "cuda", "metal"])
def table_device(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> host.Device:
    selected: host.Device = request.getfixturevalue({"c": "device", "cuda": "gpu", "metal": "metal_gpu"}[request.param])
    monkeypatch.setattr(host.Device, "current", classmethod(lambda _cls: selected))
    return selected


@pytest.mark.parametrize("generation", range(1, 6))
@pytest.mark.parametrize("name", ["add", "contains", "update", "get"])
def test_apple_32_bit_publication(generation: int, name: str) -> None:
    target = targets.Apple.of(f"apple_m{generation}")
    dtypes = {"table": U32, "keys": U32}
    if name == "get":
        dtypes.update(values=U32, out=U32)
    source = emit(capture(FILE, name, dtypes=dtypes, target=target), target).source
    assert "(device atomic_uint*)p" in source
    assert "atomic_ulong" not in source
    if name == "update":
        assert "device uint*" in source and "atomic_fetch_max_explicit" in source


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
@CASES
@given(st.lists(st.integers(0, 2**64 - 1), max_size=20))
def test_engine(run: Run, dtype: type[np.uint32] | type[np.uint64], data: list[int]) -> None:
    maximum = int(np.iinfo(dtype).max)
    high_bit = 1 << (np.dtype(dtype).itemsize * 8 - 1)
    keys = np.array([0, maximum, 0, maximum, high_bit, high_bit - 1, *(x & maximum for x in data)], dtype)
    normalized = np.empty_like(keys)
    bits = np.dtype(dtype).itemsize * 8
    run(f"bits{bits}", [keys.view(f"i{bits // 8}"), normalized])
    np.testing.assert_array_equal(keys, normalized)
    table = np.full(65, maximum, dtype)
    side = np.zeros(1, np.uint32)
    results = np.zeros(len(keys), np.uint32)
    run("add", [table, side, keys, results])
    oracle = set(keys.tolist())
    assert sum(results == 0) == len(oracle)
    assert sum(results == 1) == len(keys) - len(oracle)
    assert not np.any(results == 2)
    queries = np.array([*keys, 1234567], dtype)
    present = np.zeros(len(queries), np.bool_)
    run("contains", [table, side, queries, present])
    assert present.tolist() == [int(k) in oracle for k in queries]
    values = np.arange(len(keys), dtype=np.uint32)
    stored = np.zeros(65, np.uint32)
    winners = np.zeros(65, np.uint32)
    run("update", [table, side, keys, results, winners])
    run("assign", [stored, values, winners])
    expected = dict(zip(keys.tolist(), values.tolist(), strict=True))
    out = np.zeros(len(queries), np.uint32)
    run("get", [table, side, stored, queries, 91, out])
    assert out.tolist() == [expected.get(int(k), 91) for k in queries]
    occupied = table[:-1] != maximum
    pairs = set(zip(table[:-1][occupied].tolist(), stored[:-1][occupied].tolist(), strict=True))
    pairs.add((maximum, int(stored[-1])))
    assert pairs == set(expected.items())
    empty_keys = keys[:0]
    empty_words = np.empty(0, np.uint32)
    run(f"bits{bits}", [empty_keys.view(f"i{bits // 8}"), empty_keys])
    run("add", [table, side, empty_keys, empty_words])
    run("contains", [table, side, empty_keys, present[:0]])
    run("update", [table, side, empty_keys, empty_words, winners])
    run("assign", [stored, empty_words, empty_words])
    run("get", [table, side, stored, empty_keys, 91, empty_words])


def test_full(run: Run) -> None:
    table = np.full(3, 2**32 - 1, np.uint32)
    side = np.zeros(1, np.uint32)
    queries = np.array([0, 1, 2, 3, 2**32 - 1], np.uint32)
    result = np.zeros(5, np.uint32)
    present = np.zeros(5, np.bool_)
    run("contains", [table, side, queries, present])
    assert not present.any()
    run("add", [table, side, queries, result])
    assert sorted(result.tolist()) == [0, 0, 0, 2, 2]
    run("contains", [table, side, queries, present])
    assert sum(present) == 3
    winners = np.zeros(3, np.uint32)
    run("update", [table, side, queries, result, winners])
    assert sorted(result.tolist()) == [1, 1, 1, 2, 2]


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64, np.int32, np.int64])
@CASES
@given(st.lists(st.integers(-100, 100), max_size=20))
def test_host(
    table_device: host.Device,
    dtype: type[np.uint32] | type[np.uint64] | type[np.int32] | type[np.int64],
    data: list[int],
) -> None:
    device = table_device
    if device.target.vendor == "apple" and np.dtype(dtype).itemsize == 8:
        pytest.skip("Metal has no 64-bit atomics beyond unsigned min/max")
    signed = np.dtype(dtype).kind == "i"
    keys = np.array([0, -1 if signed else np.iinfo(dtype).max, *(x if signed else abs(x) for x in data)], dtype)
    batch = device.to_device(keys)
    table = Set(32)
    assert len(table) == 0
    assert table.keys().shape == (0,)
    report = table.add(batch)
    report.check()
    assert report.inserted == len(set(keys.tolist()))
    assert report.existing == len(keys) - report.inserted
    assert len(table) == report.inserted
    assert set(table.keys().to_host().tolist()) == set(keys.tolist())
    contains = table.contains(batch)
    table._read(contains)
    assert contains.to_host().all()
    mapping = Map(32)
    assert mapping.items()[0].shape == (0,)
    values = device.to_device(np.arange(len(keys), dtype=dtype))
    assert mapping.update(batch, values) == report
    replacements = device.to_device(np.arange(len(keys), dtype=dtype) + 7)
    assert mapping.update(batch, replacements).existing == len(keys)
    expected = dict(zip(keys.tolist(), replacements.to_host().tolist(), strict=True))
    actual_keys, actual_values = mapping.items()
    assert set(zip(actual_keys.to_host().tolist(), actual_values.to_host().tolist(), strict=True)) == set(
        expected.items()
    )
    result = mapping.get(batch, u32(99))
    assert mapping._read(result).tolist() == [expected[int(k)] for k in keys]


def test_host_edges(device: CPUDevice) -> None:
    keys = device.to_device(np.arange(4, dtype=np.uint32))
    loaded = Set(4)
    assert loaded.add(keys).inserted == 4
    assert len(loaded) / loaded.slot_count == 0.5
    table = Set(1, load_factor=0.5)
    report = table.add(keys)
    assert report == Inserted(inserted=2, existing=0, failed=2)
    with pytest.raises(CapacityError):
        report.check()
    overflow = Map(1)
    with pytest.raises(CapacityError):
        overflow.update(keys, keys).check()
    assert not table._read(table.contains(device.to_device(np.array([99], np.uint32))))[0]
    mapping = Map(2)
    missing = mapping.get(keys, u32(7))
    assert mapping._read(missing).tolist() == [7] * 4
    empty = device.to_device(np.empty(0, np.uint32))
    assert Set(0).add(empty) == Inserted(inserted=0, existing=0, failed=0)
    with pytest.raises(ValueError):
        Set(-1)
    with pytest.raises(ValueError):
        Set(2**32)
    oversized = Tensor[u32, int]()
    oversized.dtype, oversized.shape = u32, (2**32,)
    with pytest.raises(ValueError):
        table.add(oversized)
    for load in (0.0, 1.1, float("nan")):
        with pytest.raises(ValueError):
            Set(1, load_factor=load)
    assert Set(3, load_factor=0.7).slot_count == 8
    with pytest.raises(TypeError):
        table.add(device.to_device(np.array([1], np.uint64)))
    with pytest.raises(TypeError):
        Set(1).add(device.to_device(np.array([1], np.uint16)))
    with pytest.raises(ValueError):
        mapping.update(keys, empty)
    mapping.update(keys, keys)
    with pytest.raises(TypeError):
        mapping.update(keys, device.to_device(np.arange(4, dtype=np.uint64)))
    with pytest.raises(TypeError):
        mapping.update(keys, device.to_device(np.arange(4, dtype=np.float32)))
    with pytest.raises(TypeError):
        Set(1).add(device.to_device(np.zeros((1, 1), np.uint32)))
    mask = device.empty(bool, 1)
    mask.array[:] = False
    assert mask.dtype is bool
    assert not mask.to_host()[0]

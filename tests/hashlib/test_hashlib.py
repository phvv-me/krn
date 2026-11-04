"""Bit-exact oracle checks on every available execution route."""

import ast
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING

import mmh3
import numpy as np
import pytest
import xxhash
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from krn import Items, hashlib, targets
from krn.compiler.api import capture
from krn.emulator.emulate import Emulated
from krn.emulator.emulate import build as emulate
from krn.runtime.cpu import CPUProgram, Value
from tests.conftest import build_kernel
from tests.helpers.devices import CUDARun

if TYPE_CHECKING:
    from krn.compiler.ir import Kernel
    from krn.runtime.cuda_backend import CUDADevice
    from tests.helpers.devices import MetalRun

FILE = Path(__file__).with_name("kernels.py")
MASK64 = (1 << 64) - 1
type Run = Callable[[str, list[Value]], None]
CASES = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


def host_source() -> dict[str, FunctionType]:
    """Execute the unmodified arithmetic with NumPy widths for source-level branch coverage.

    The IR emulator inlines callees under the caller's filename, so it cannot measure a separate scalar module.
    Only imports and decorators are removed; these are the production bodies, not a second hash implementation.
    """
    namespace = dict(vars(hashlib))
    namespace.update(u8=np.uint8, u32=np.uint32, u64=np.uint64, items=range)
    for path in (FILE.parents[1].parent / "src" / "krn" / "hashlib.py", FILE):
        if path == FILE:
            namespace = {**namespace, "__name__": "tests.hashlib.kernels"}
        tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [node for node in tree.body if not isinstance(node, ast.Import | ast.ImportFrom)]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
        exec(compile(tree, str(path), "exec"), namespace)
    return {name: fn for name, fn in namespace.items() if isinstance(fn, FunctionType)}


@pytest.fixture(params=["host", "emulator", "c", "cuda", "metal"])
def run(request: pytest.FixtureRequest) -> Run:
    if request.param == "host":
        functions = host_source()

        def execute_host(name: str, arguments: list[Value]) -> None:
            with np.errstate(over="ignore"):
                functions[name](*arguments)

        return execute_host

    @cache
    def program(name: str) -> CPUProgram | Emulated | CUDARun | MetalRun:
        if request.param == "c":
            return build_kernel(FILE, name)
        if request.param == "cuda":
            gpu: CUDADevice = request.getfixturevalue("gpu")
            return CUDARun(gpu, capture(FILE, name, target=gpu.target))
        if request.param == "metal":
            launcher: Callable[[Kernel], MetalRun] = request.getfixturevalue("metal_program")
            return launcher(capture(FILE, name, target=targets.Apple.of("apple_m4")))
        target = targets.CPU.of("avx512")
        return emulate(capture(FILE, name, target=target), target)

    def execute(name: str, arguments: list[Value]) -> None:
        first = arguments[1] if name == "ranges" else arguments[0]
        assert isinstance(first, np.ndarray)
        executable = program(name)
        if isinstance(executable, CPUProgram):
            executable(*arguments, launch=Items(len(first)))
        else:
            executable.run(Items(len(first)), arguments)

    return execute


def splitmix_reference(key: int) -> int:
    value = (key + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


@CASES
@given(st.lists(st.tuples(st.integers(0, 2**32 - 1), st.integers(0, 2**32 - 1)), min_size=1, max_size=24))
def test_words32(run: Run, values: list[tuple[int, int]]) -> None:
    values += [(key, seed) for key in (0, 1, 2**32 - 1) for seed in (0, 1, 2**32 - 1)]
    keys, seeds = (np.array(a, np.uint32) for a in zip(*values, strict=True))
    xx, murmur = np.zeros_like(keys), np.zeros_like(keys)
    run("words32", [keys, seeds, xx, murmur])
    assert xx.tolist() == [xxhash.xxh32_intdigest(x.to_bytes(4, "little"), s) for x, s in values]
    assert murmur.tolist() == [mmh3.hash(x.to_bytes(4, "little"), s, signed=False) for x, s in values]


@CASES
@given(st.lists(st.tuples(st.integers(0, MASK64), st.integers(0, MASK64)), min_size=1, max_size=24))
def test_words64(run: Run, values: list[tuple[int, int]]) -> None:
    values += [(key, seed) for key in (0, 1, 2**32 - 1, 2**63, MASK64) for seed in (0, 1, MASK64)]
    keys, seeds = (np.array(a, np.uint64) for a in zip(*values, strict=True))
    xx, mixed = np.zeros_like(keys), np.zeros_like(keys)
    run("words64", [keys, seeds, xx, mixed])
    assert xx.tolist() == [xxhash.xxh64_intdigest(x.to_bytes(8, "little"), s) for x, s in values]
    assert mixed.tolist() == [splitmix_reference(x) for x, _ in values]


@CASES
@given(data=st.binary(min_size=200, max_size=200), seed=st.integers(0, MASK64))
def test_byte_lengths(run: Run, data: bytes, seed: int) -> None:
    buffer = np.frombuffer(b"prefix!" + data + b"suffix", np.uint8).copy()
    starts = np.full(402, 7, np.uint32)
    starts[:201] = 0
    stops = starts + np.tile(np.arange(201, dtype=np.uint32), 2)
    seeds = np.full(402, seed, np.uint64)
    seeds[0] = 0
    out = np.zeros(402, np.uint64)
    run("ranges", [buffer, starts, stops, seeds, out])
    assert out[0] == 0xEF46DB3751D8E999
    assert out.tolist() == [
        xxhash.xxh64_intdigest(buffer[int(a) : int(b)].tobytes(), int(s))
        for a, b, s in zip(starts, stops, seeds, strict=True)
    ]


def test_properties(run: Run) -> None:
    keys = np.arange(100_000, dtype=np.uint64)
    mixed, xx = np.zeros_like(keys), np.zeros_like(keys)
    run("words64", [keys, np.zeros_like(keys), xx, mixed])
    assert len(np.unique(mixed)) == len(keys)
    left, right = keys // 317, keys % 317
    first, second, swapped = (np.zeros_like(keys) for _ in range(3))
    run("pairs", [left, right, first])
    run("pairs", [left, right, second])
    run("pairs", [right, left, swapped])
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 100_000
    assert np.all(first[left != right] != swapped[left != right])


def test_published_vectors() -> None:
    assert xxhash.xxh32_intdigest(b"", 0) == 0x02CC5D05
    assert xxhash.xxh64_intdigest(b"", 0) == 0xEF46DB3751D8E999
    hashes = b"".join(mmh3.hash(bytes(range(n)), 256 - n, signed=False).to_bytes(4, "little") for n in range(256))
    assert mmh3.hash(hashes, 0, signed=False) == 0xB0F57EE3


def test_murmur_verification_words(run: Run) -> None:
    keys = np.arange(256, dtype=np.uint32)
    seeds = np.array([mmh3.hash(bytes(range(n)), 256 - n, signed=False) for n in range(256)], np.uint32)
    xx, out = np.zeros_like(keys), np.zeros_like(keys)
    run("words32", [keys, seeds, xx, out])
    assert out.tolist() == [
        mmh3.hash(int(k).to_bytes(4, "little"), int(s), signed=False) for k, s in zip(keys, seeds, strict=True)
    ]

"""The sparse kernels, run on the real GPUs and compared with the emulator.

The layouts and the kernels written against `Rows` are in `sketch/krn/sparse.py`, the sparse vectors and the 2:4
kernels in `examples/04_libraries/06_sparse`. Every layout is a struct of dense arrays, so a card receives the same
records a dense kernel gets. `spmv`, `spmv_vector` and `spmm` are written once and run here on CSR, Ellpack, blocked
CSR and a dense matrix, which `structs` says. The row sums of the atomic kernels (`spmv_coo`, `spvv`) are added in an
order the card picks, which the float tolerance of `agree` absorbs. They skip where there is no CUDA card or no Metal
GPU."""

import importlib.util
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest

from krn import Items
from krn.compiler.api import capture, emit
from krn.compiler.dtypes import F32
from krn.compiler.ir import StructType
from krn.data.sparse_build import bsr_arrays, coo_arrays, csr_arrays, dense_arrays, ell_arrays
from krn.facts.targets import Target
from krn.runtime.cpu import CLANG, Value
from tests.compiler.test_hip import CLANG_HAS_AMDGPU
from tests.conftest import EXAMPLE_ROOT
from tests.helpers.devices import CUDARun, MetalRun, agree, bindings, compare
from tests.helpers.text import descriptor

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = EXAMPLE_ROOT / "04_libraries" / "06_sparse.py"
SPARSE = ROOT / "src" / "krn" / "data" / "sparse.py"
if not EXAMPLE.is_file():
    pytest.skip("research examples are absent; examples have not migrated", allow_module_level=True)
spec = importlib.util.spec_from_file_location("sparse_example", EXAMPLE)
assert spec is not None and spec.loader is not None
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

HIP_SHIM = Path(__file__).resolve().parents[1] / "hip_shim"
ROWS, COLUMNS, WIDTH, ENTRIES, BLOCK = 40, 36, 9, 11, 4


class Case(NamedTuple):
    kernel: str
    file: Path
    arguments: list[Value]
    launch: Items
    structs: Mapping[str, str] = {}


def problem(warp_size: int) -> dict[str, Case]:
    """Each kernel's arguments and launch, over one random matrix with empty rows and one full row."""
    rng = np.random.default_rng(5)
    dense = example.random_dense(rng, ROWS, COLUMNS, 0.15)
    x = rng.standard_normal(COLUMNS).astype(np.float32)
    b = rng.standard_normal((COLUMNS, WIDTH)).astype(np.float32)
    left = rng.standard_normal((ROWS, 5)).astype(np.float32)
    right = rng.standard_normal((5, COLUMNS)).astype(np.float32)
    positions = np.sort(rng.choice(COLUMNS, ENTRIES, replace=False)).astype(np.uint32)
    weights = rng.standard_normal((16, 32)).astype(np.float32)
    packed_values, packed_where, _ = example.prune_reference(weights)

    formats = {
        "Csr": csr_arrays(dense)._asdict(),
        "Ell": ell_arrays(dense)._asdict(),
        "Bsr": bsr_arrays(dense, BLOCK)._asdict(),
        "Dense": dense_arrays(dense)._asdict(),
    }
    entries = len(formats["Csr"]["values"])
    sparse = {"indices": positions, "values": x[positions].copy()}
    packed = {"values": packed_values, "where": packed_where}
    empty_packed = {"values": np.zeros_like(packed_values), "where": np.zeros_like(packed_where)}
    zeros = np.zeros(ROWS, np.float32)
    cases = {}
    for name, matrix in formats.items():
        structs = {"a": name}
        cases |= {
            f"spmv_{name}": Case("spmv", SPARSE, [matrix, x, zeros.copy()], Items(ROWS), structs),
            f"spmv_vector_{name}": Case(
                "spmv_vector", SPARSE, [matrix, x, zeros.copy()], Items(ROWS * warp_size), structs
            ),
            f"spmm_{name}": Case(
                "spmm",
                SPARSE,
                [matrix, b, np.zeros((ROWS, WIDTH), np.float32)],
                Items(ROWS * WIDTH),
                structs,
            ),
        }
    return cases | {
        "spmv_coo": Case("spmv_coo", SPARSE, [coo_arrays(dense)._asdict(), x, zeros.copy()], Items(entries)),
        "sddmm": Case(
            "sddmm",
            SPARSE,
            [formats["Csr"], left, right, np.zeros(entries, np.float32)],
            Items(ROWS),
            {"pattern": "Csr"},
        ),
        "gather": Case(
            "gather",
            EXAMPLE,
            [x, {"indices": positions, "values": np.zeros(ENTRIES, np.float32)}],
            Items(ENTRIES),
        ),
        "scatter": Case("scatter", EXAMPLE, [sparse, np.zeros(COLUMNS, np.float32)], Items(ENTRIES)),
        "scale": Case("scale", EXAMPLE, [0.5, x.copy()], Items(COLUMNS)),
        "axpy": Case("axpy", EXAMPLE, [2.0, sparse, x.copy()], Items(ENTRIES)),
        "spvv": Case("spvv", EXAMPLE, [sparse, x, np.zeros(1, np.float32)], Items(ENTRIES)),
        "prune_24": Case("prune_24", EXAMPLE, [weights, empty_packed], Items(16 * 8)),
        "spmv_24": Case(
            "spmv_24",
            EXAMPLE,
            [packed, rng.standard_normal(32).astype(np.float32), np.zeros(16, np.float32)],
            Items(16),
        ),
    }


NAMES = list(problem(32))


@pytest.mark.parametrize("name", NAMES)
def test_the_card_agrees_with_the_emulator(name: str, gpu: CUDADevice) -> None:
    card = gpu
    case = problem(card.target.warp_size)[name]
    oracle, actual = compare(
        card.target,
        lambda kernel: CUDARun(card, kernel),
        case.kernel,
        case.launch,
        case.arguments,
        case.file,
        case.structs,
    )
    agree(oracle, actual)


@pytest.mark.parametrize("name", NAMES)
def test_the_metal_gpu_agrees_with_the_emulator(name: str) -> None:
    metal = pytest.importorskip("krn.runtime.metal")
    try:
        gpu = metal.MetalDevice(Target.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))
    case = problem(gpu.target.warp_size)[name]
    oracle, actual = compare(
        gpu.target,
        lambda kernel: MetalRun(gpu, kernel),
        case.kernel,
        case.launch,
        case.arguments,
        case.file,
        case.structs,
    )
    agree(oracle, actual)


def test_a_kernel_written_for_a_base_class_is_captured_once_for_each_format() -> None:
    fields = {}
    for name in ("Csr", "Ell", "Bsr", "Dense"):
        bound = {"a.values": F32, "x": F32, "y": F32}
        parameter = capture(SPARSE, "spmv", structs={"a": name}, dtypes=bound).params[0][1]
        assert isinstance(parameter, StructType)
        fields[name] = [field for field, _ in parameter.fields]
    assert fields == {
        "Csr": ["values", "rows", "columns", "indptr", "indices"],
        "Ell": ["values", "rows", "columns", "indices", "width"],
        "Bsr": ["values", "rows", "columns", "indptr", "indices", "block"],
        "Dense": ["values", "rows", "columns"],
    }


@pytest.mark.skipif(not CLANG_HAS_AMDGPU, reason="this clang has no AMDGPU backend")
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("name", NAMES)
def test_the_hip_renderer_compiles_the_kernel_for_a_64_lane_wave(tmp_path: Path, name: str, arch: str) -> None:
    """Nothing here has run on an AMD GPU: the unit is compiled to AMDGPU assembly against the HIP shim."""
    case = problem(64)[name]
    target = Target.of(arch)
    kernel = capture(
        case.file,
        case.kernel,
        structs=case.structs,
        dtypes=bindings(case.file, case.kernel, case.arguments),
        target=target,
    )
    source = emit(kernel, target, lineinfo=False).source
    unit = tmp_path / f"{case.kernel}.hip"
    unit.write_text(source)
    command = [
        CLANG[0],
        "-x",
        "hip",
        f"--offload-arch={arch}",
        "--offload-device-only",
        "-nogpulib",
        "-nogpuinc",
    ]
    done = subprocess.run(
        [*command, "-I", str(HIP_SHIM), "-S", "-o", str(tmp_path / "unit.s"), str(unit)],
        capture_output=True, text=True, timeout=240, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-3000:]
    assert descriptor((tmp_path / "unit.s").read_text(), case.kernel)

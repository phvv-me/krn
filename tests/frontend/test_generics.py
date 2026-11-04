"""Struct specialization and the two explicit accumulation contexts; ordinary scalar arithmetic stays strict."""

import ast
import importlib.util
import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from krn import Blocks, Items, T, lib
from krn.compiler.api import CaptureError, CPUDevice, CPUTensor, build, capture, emit
from krn.compiler.dtypes import F16, F32, F64, I32, U8, U32, Dtype
from krn.compiler.ir import Accumulate, Cast, Expr, Kernel, Stmt, StructType, TensorType
from krn.compiler.source import Module
from krn.compiler.validity import walk
from krn.data import sparse, sparse_build
from krn.emulator.emulate import build as emulated
from krn.emulator.emulate import emulate
from krn.facts.targets import CPU, Target
from krn.frontend.geometry import Geometry
from krn.frontend.numerics import FloatingPoint, Scalar, name_of, product_type, widens_to
from krn.lib import lib_cusparse, lib_host, lib_ours
from krn.runtime.cpu import Value
from krn.runtime.runtime import DTYPES
from tests.helpers.devices import CUDARun, MetalRun, dtypes, fresh

PRELUDE = """
from abc import ABC, abstractmethod
from collections.abc import Iterator
from krn import Matrix, Vector, T, items, kernel, struct, sum, zeros
from krn.frontend.memory import Tensor3
from krn.frontend.types import FloatingPoint, Idx, f16, f32, f64, u8, u32

@struct
class Rows[F: FloatingPoint](ABC):
    values: Vector[F]
    rows: u32
    columns: u32
    @abstractmethod
    def row(self, i: u32, first: u32 | int = 0, step: u32 | int = 1) -> Iterator[tuple[Idx, F]]: ...

@struct
class Csr[F: FloatingPoint](Rows[F]):
    indptr: Vector[u32]
    indices: Vector[u32]
    def identity(self, v: F) -> F:
        return self.typed(v)
    def typed[G: FloatingPoint](self, v: G) -> G:
        return v
    def row(self, i: u32, first: u32 | int = 0, step: u32 | int = 1) -> Iterator[tuple[Idx, F]]:
        for k in range(self.indptr[i] + first, self.indptr[i + 1], step):
            v: F = self.identity(self.values[k])
            yield self.indices[k], v
"""
SPMV = """
@kernel
def spmv[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](a: Rows[A], x: Vector[X], y: Vector[Y]) -> None:
    for i in items(a.rows):
        y[i] = EXPRESSION
"""


def source(tmp_path: Path, body: str, prelude: str = PRELUDE) -> Path:
    path = tmp_path / "generic_kernels.py"
    path.write_text(prelude + "\n" + body)
    return path


def imported(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_kernels", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Runner:
    backend: str
    target: Target
    run: Callable[[Kernel, Geometry, list[Value]], None]

    def __call__(self, kernel: Kernel, launch: Geometry, args: list[Value]) -> None:
        self.run(kernel, launch, args)


@pytest.fixture(params=["cpu", "emulator", "metal", "cuda"])
def runner(request: pytest.FixtureRequest) -> Runner:
    if request.param == "cpu":
        cpu = CPU.of("avx512")
        return Runner("cpu", cpu, lambda kernel, launch, args: build(kernel, cpu)(*args, launch=launch))
    if request.param == "emulator":
        target = Target.of("sm_90a")
        return Runner(
            "emulator",
            target,
            lambda kernel, launch, args: emulated(kernel, target).run(launch, args),
        )
    if "left" in request.fixturenames and request.getfixturevalue("left") == "bf16":
        pytest.skip("bf16 tile casts need lowering in the CUDA and Metal renderers")
    if request.param == "metal":
        metal = pytest.importorskip("krn.runtime.metal")
        try:
            gpu = metal.MetalDevice(Target.of("apple_m4"))
        except metal.MetalError as error:
            pytest.skip(str(error))
        return Runner(
            "metal",
            gpu.target,
            lambda kernel, launch, args: MetalRun(gpu, kernel).run(launch, args),
        )
    card = request.getfixturevalue("gpu")
    return Runner("cuda", card.target, lambda kernel, launch, args: CUDARun(card, kernel).run(launch, args))


def test_struct_parameters_and_kernel_parameters_bind_positionally(tmp_path: Path) -> None:
    path = source(
        tmp_path,
        SPMV.replace("EXPRESSION", "sum(v.to(y.dtype) * x[c].to(y.dtype) for c, v in a.row(i))"),
    )
    module = Module.read(path)
    definition = module.defs["Csr"]
    assert isinstance(definition, ast.ClassDef)
    assert module.struct(definition).type_params == ("F",)
    ir = capture(path, "spmv", structs={"a": "Csr"}, dtypes={"a.values": F16, "x": F32, "y": F32})
    assert ir.params[0][1] == StructType(
        "Csr_f16",
        (
            ("values", TensorType(F16, 1)),
            ("rows", U32),
            ("columns", U32),
            ("indptr", TensorType(U32, 1)),
            ("indices", TensorType(U32, 1)),
        ),
    )
    assert ir.params[1:] == (("x", TensorType(F32, 1)), ("y", TensorType(F32, 1)))


@pytest.mark.parametrize("base", ["Rows", "Rows[G]", "Rows[F, G]"])
def test_inheritance_requires_the_same_parameters_in_order(tmp_path: Path, base: str) -> None:
    path = source(
        tmp_path,
        f"@struct\nclass Bad[F: FloatingPoint]({base}): pass\n@kernel\ndef f(a: Bad): pass",
    )
    with pytest.raises(CaptureError, match=r"same number of type parameters.*same order"):
        capture(path, "f", dtypes={"a.values": F16})


@pytest.mark.parametrize(
    ("body", "bindings", "message"),
    [
        ("@kernel\ndef f(a: Csr): pass", {}, "unbound; pass dtypes=\\{'a.values': "),
        (
            "@struct\nclass Pair[F: FloatingPoint]:\n a: Vector[F]\n b: Matrix[F]\n@kernel\ndef f(p: Pair): pass",
            {"p.a": F16, "p.b": F32},
            "p.a is f16 and p.b is f32",
        ),
        (
            "@kernel\ndef f(a: Csr[f32]): pass",
            {"a.values": F16},
            "parameter a is Csr\\[f16\\].*declares Csr\\[f32\\]",
        ),
        (
            "@kernel\ndef f(a: Rows[f32]): pass",
            {"a.values": F16},
            "parameter a is Csr\\[f16\\].*declares Csr\\[f32\\]",
        ),
        ("@kernel\ndef f():\n a = Csr()", {}, "construct generic struct Csr on the host"),
        ("@kernel\ndef f():\n a = Csr[f32]()", {}, "construct generic struct Csr on the host"),
        (
            "@kernel\ndef f[A: FloatingPoint](a: Rows[A], x: Vector[A]): pass",
            {"a.values": F16, "x": F32},
            "type parameter `A` is f16 and `x` is f32",
        ),
    ],
)
def test_struct_binding_errors(tmp_path: Path, body: str, bindings: dict[str, Dtype], message: str) -> None:
    path = source(tmp_path, body)
    with pytest.raises(CaptureError, match=message):
        capture(path, "f", structs={"a": "Csr"} if "a:" in body else {}, dtypes=bindings)


@pytest.mark.parametrize("annotation", ["F", "Vector[F]", "Matrix[F]", "Tensor3[F]"])
def test_each_field_rank_binds_a_struct_parameter(tmp_path: Path, annotation: str) -> None:
    path = source(
        tmp_path,
        f"@struct\nclass Box[F: FloatingPoint]:\n values: {annotation}\n@kernel\ndef f(a: Box): pass",
    )
    param = capture(path, "f", dtypes={"a.values": F16}).params[0][1]
    assert isinstance(param, StructType)
    assert param.fields[0][1] == (
        F16 if annotation == "F" else TensorType(F16, ["Vector[F]", "Matrix[F]", "Tensor3[F]"].index(annotation) + 1)
    )


def test_multiple_struct_parameters_bind_in_declaration_order(tmp_path: Path) -> None:
    path = source(
        tmp_path,
        """
@struct
class Pair[F: FloatingPoint, G: FloatingPoint]:
    right: Vector[G]
    left: Vector[F]
@kernel
def f[A: FloatingPoint, B: FloatingPoint](p: Pair[A, B], a: Vector[A], b: Vector[B]):
    a[0] = p.left[0]
    b[0] = p.right[0]
""",
    )
    ir = capture(path, "f", dtypes={"p.left": F16, "p.right": F32, "a": F16, "b": F32})
    definition = Module.read(path).defs["Pair"]
    assert isinstance(definition, ast.ClassDef)
    assert Module.read(path).struct(definition).type_params == ("F", "G")
    assert ir.params[1:] == (("a", TensorType(F16, 1)), ("b", TensorType(F32, 1)))


@pytest.mark.parametrize(
    "expression",
    [
        "sum(v.to(y.dtype) * x[c].to(y.dtype) for c, v in a.row(i))",
        "sum((v * x[c] for c, v in a.row(i)), acc=y.dtype)",
    ],
)
def test_generic_csr_product(
    tmp_path: Path, runner: Callable[[Kernel, Geometry, list[Value]], None], expression: str
) -> None:
    path = source(tmp_path, SPMV.replace("EXPRESSION", expression))
    matrix: dict[str, Value] = {
        "values": np.array([1.125, -2.25, 0.5, 3.5], np.float16),
        "rows": 3,
        "columns": 3,
        "indptr": np.array([0, 2, 2, 4], np.uint32),
        "indices": np.array([0, 2, 1, 2], np.uint32),
    }
    x, y = np.array([0.125, 2.5, -1.75], np.float32), np.zeros(3, np.float32)
    arguments: dict[str, Value] = {"a": matrix, "x": x, "y": y}
    ir = capture(path, "spmv", structs={"a": "Csr"}, dtypes=dtypes(arguments))
    runner(ir, Items(3), list(arguments.values()))
    expected = np.array([[1.125, 0, -2.25], [0, 0, 0], [0, 0.5, 3.5]], np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("wrapped", [False, True])
def test_runtime_and_emulate_infer_struct_dtypes_and_cache_programs(
    tmp_path: Path, device: CPUDevice, wrapped: bool
) -> None:
    path = source(
        tmp_path,
        SPMV.replace("EXPRESSION", "sum(v.to(y.dtype) * x[c].to(y.dtype) for c, v in a.row(i))"),
    )
    module = imported(path)
    launcher = device.launcher(module.spmv, Items(1))
    x, y = np.array([2.0], np.float32), np.zeros(1, np.float32)
    matrices = [
        module.Csr(
            CPUTensor(np.array([1.5], dtype)) if wrapped else np.array([1.5], dtype),
            1,
            1,
            np.array([0, 1], np.uint32),
            np.array([0], np.uint32),
        )
        for dtype in (np.float16, np.float32)
    ]
    programs = [launcher.program(a, x, y) for a in (*matrices, matrices[0])]
    assert programs[0] is programs[2] and programs[0] is not programs[1]
    assert len(launcher._plans) == 2
    for a in (*matrices, matrices[0]):
        launcher(a, x, y)
        device.default_stream.record().wait(60)
        np.testing.assert_array_equal(y, [3.0])
        y[:] = 0
        emulate(module.spmv, Items(1), a, x, y)
        np.testing.assert_array_equal(y, [3.0])


@pytest.mark.parametrize(
    "expression",
    [
        "sum(v.to(y.dtype) * w.to(y.dtype) for c, v in a.row(i, 0, 1) for d, w in b.row(i))",
        "sum((v * w for c, v in a.row(i, 0, 1) for d, w in b.row(i)), acc=Y)",
        "loops",
    ],
)
def test_two_instances_keep_independent_types(
    tmp_path: Path, runner: Callable[[Kernel, Geometry, list[Value]], None], expression: str
) -> None:
    body = (
        f"y[i] = {expression}"
        if expression != "loops"
        else """total: Y = 0.0
        for c, v in a.row(i):
            for d, w in b.row(i, 0, 1):
                total += v.to(Y) * w.to(Y)
        y[i] = total"""
    )
    path = source(
        tmp_path,
        f"""
@kernel
def f[A: FloatingPoint, B: FloatingPoint, Y: FloatingPoint](a: Rows[A], b: Rows[B], y: Vector[Y]):
    for i in items(a.rows):
        {body}
""",
    )
    ir = capture(
        path,
        "f",
        structs={"a": "Csr", "b": "Csr"},
        dtypes={"a.values": F16, "b.values": F32, "y": F32},
    )
    for (_, param), dtype in zip(ir.params[:2], (F16, F32), strict=True):
        assert isinstance(param, StructType)
        assert param.fields[0] == ("values", TensorType(dtype, 1))
    matrix: dict[str, Value] = {
        "values": np.array([2], np.float16),
        "rows": 1,
        "columns": 1,
        "indptr": np.array([0, 1], np.uint32),
        "indices": np.array([0], np.uint32),
    }
    other = matrix | {"values": np.array([3], np.float32)}
    y = np.zeros(1, np.float32)
    runner(ir, Items(1), [fresh(matrix), fresh(other), y])
    np.testing.assert_array_equal(y, [6])


@pytest.mark.parametrize(
    ("left", "right"),
    [("f16", "f32"), ("f32", "f16"), ("bf16", "f16"), ("bf16", "f32"), ("f16", "f16")],
)
def test_tile_products(
    tmp_path: Path, runner: Callable[[Kernel, Geometry, list[Value]], None], left: str, right: str
) -> None:
    path = source(
        tmp_path,
        f"""
@kernel
def f(a: Matrix[f32], b: Matrix[f32], out: Matrix[f32]):
    acc = zeros(f32, 2, 2)
    acc += a[0:2, 0:4].to(T.{left}) @ b[0:4, 0:2].to(T.{right})
    out[0:2, 0:2] = acc
""",
    )
    ir = capture(path, "f")
    product = next(s for s in ir.body if isinstance(s, Accumulate))
    assert product.lhs.dtype == product.rhs.dtype == (F16 if left == right == "f16" else F32)
    if left != right:
        assert isinstance(product.lhs, Cast) or isinstance(product.rhs, Cast)
    a = np.array([[1.00390625, 1.01171875, 0.5, 3.5], [0.25, 1.5, -1, 2]], np.float32)
    b = np.array([[0.5, 2], [1, -1.5], [2.5, 0.25], [-1, 1.125]], np.float32)
    out = np.zeros((2, 2), np.float32)
    runner(ir, Blocks(1), [a, b, out])
    left_reference = a.astype(np.float64)
    if left == "bf16":
        left_reference[0, :2] = [1.0, 1.015625]
    np.testing.assert_allclose(out, left_reference @ b.astype(np.float64), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(("left", "right"), [("f64", "f64"), ("f16", "f64"), ("u8", "f32"), ("f8e5m2", "f32")])
def test_tile_products_refuse_non_lossless_operands(tmp_path: Path, left: str, right: str) -> None:
    path = source(
        tmp_path,
        f"""
@kernel
def f(a: Matrix[f32], b: Matrix[f32]):
    acc = zeros(f32, 2, 2)
    acc += a[0:2, 0:2].to(T.{left}) @ b[0:2, 0:2].to(T.{right})
""",
    )
    with pytest.raises(CaptureError, match=r"f32 accumulator needs lossless operands.*fp8.*QTile / mma"):
        capture(path, "f")


@pytest.mark.parametrize(
    ("expression", "dtype", "expected"),
    [
        ("sum((x[i] for i in range(4)), acc=u32)", U8, 600),
        ("sum((x[i] for i in range(4)), acc=y.dtype)", F16, 600),
        ("sum((x[i] * 2.0 for i in range(4)), acc=T.acc(x.dtype))", F16, 1200),
        ("sum((x[i] * x[i] for i in range(4)), acc=f32)", F16, 90000),
        ("sum([x[0] * x[0], x[1] * x[1], x[2] * x[2], x[3] * x[3]], acc=f32)", F16, 90000),
        ("sum([x[0], x[1], x[2], x[3]], acc=u32)", U8, 600),
        ("sum(range(4), acc=u32)", U8, 6),
        ("sum([], acc=u32)", U8, 0),
    ],
)
def test_sum_accumulator_forms(
    tmp_path: Path,
    runner: Callable[[Kernel, Geometry, list[Value]], None],
    expression: str,
    dtype: Dtype,
    expected: int,
) -> None:
    output = "f32" if dtype == F16 else "u32"
    path = source(
        tmp_path,
        f"@kernel\ndef f[X: T.Scalar](x: Vector[X], y: Vector[{output}]):\n y[0] = {output}({expression})",
    )
    ir = capture(path, "f", dtypes={"x": dtype})
    x = np.full(4, 150, np.uint8 if dtype == U8 else np.float16)
    y = np.zeros(1, np.float32 if dtype == F16 else np.uint32)
    runner(ir, Items(1), [x, y])
    np.testing.assert_array_equal(y, [expected])


@pytest.mark.parametrize(
    "terms",
    [
        "[x[0] * z[0], x[1] * z[1]]",
        "(x[0] * z[0], x[1] * z[1])",
        "(x[i] * z[i] for i in range(2))",
    ],
)
def test_sum_widens_products_before_snapshotting_terms(
    tmp_path: Path, runner: Callable[[Kernel, Geometry, list[Value]], None], terms: str
) -> None:
    path = source(
        tmp_path,
        f"@kernel\ndef f(x: Vector[f16], z: Vector[f32], y: Vector[f32]):\n y[0] = sum({terms}, acc=f32)",
    )
    x, z, y = (
        np.array([150, 300], np.float16),
        np.array([300, 150], np.float32),
        np.zeros(1, np.float32),
    )
    runner(capture(path, "f"), Items(1), [x, z, y])
    np.testing.assert_array_equal(y, [90000])


@pytest.mark.parametrize(
    ("expression", "bindings", "message"),
    [
        ("x[0] * z[0]", {"x": F16, "z": F32}, "f16.*f32"),
        (
            "sum((x[i] for i in range(1)), acc=f32)",
            {"x": F64},
            "x\\[i\\].*f64.*f32.*convert explicitly",
        ),
        (
            "sum((x[i] for i in range(1)), acc=f32)",
            {"x": I32},
            "x\\[i\\].*i32.*f32.*convert explicitly",
        ),
        (
            "sum((x[i] for i in range(1)), acc=f16)",
            {"x": F32},
            "x\\[i\\].*f32.*f16.*convert explicitly",
        ),
        (
            "sum((x[i] * z[i] for i in range(1)), acc=f32)",
            {"x": F16, "z": F64},
            "z\\[i\\].*f64.*f32.*convert explicitly",
        ),
        *[
            (f"{name}((x[i] for i in range(1)), acc=f32)", {"x": F32}, "only sum takes acc=")
            for name in ("min", "max", "any", "all")
        ],
    ],
)
def test_accumulation_refusals(tmp_path: Path, expression: str, bindings: dict[str, Dtype], message: str) -> None:
    path = source(
        tmp_path,
        f"@kernel\ndef f[X: T.Scalar, Z: T.Scalar](x: Vector[X], z: Vector[Z], y: Vector[f32]):\n y[0] = {expression}",
    )
    with pytest.raises(CaptureError, match=message):
        capture(path, "f", dtypes={"z": F32} | bindings)


STANDARD: tuple[type[Scalar], ...] = (
    T.u8,
    T.u16,
    T.u32,
    T.u64,
    T.i8,
    T.i16,
    T.i32,
    T.i64,
    T.f16,
    T.bf16,
    T.f32,
    T.f64,
)
WIDENING = (
    "u8 u16 u32 u64 i16 i32 i64 f16 bf16 f32 f64",
    "u16 u32 u64 i32 i64 f32 f64",
    "u32 u64 i64 f64",
    "u64",
    "i8 i16 i32 i64 f16 bf16 f32 f64",
    "i16 i32 i64 f32 f64",
    "i32 i64 f64",
    "i64",
    "f16 f32 f64",
    "bf16 f32 f64",
    "f32 f64",
    "f64",
)


@pytest.mark.parametrize(("source_index", "target_index"), list(itertools.product(range(len(STANDARD)), repeat=2)))
def test_lossless_conversion_table(source_index: int, target_index: int) -> None:
    names = ("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64")
    assert widens_to(STANDARD[source_index], STANDARD[target_index]) == (
        names[target_index] in WIDENING[source_index].split()
    )


@pytest.mark.parametrize(
    ("source_type", "target_type", "expected"),
    [
        (T.f8e5m2, T.f16, True),
        (T.f8e4m3fn, T.f16, False),
        (T.f4e2m1, T.f32, False),
        (T.f8e4m3fn, T.f8e4m3fn, True),
        (T.c64, T.f64, False),
        (T.f32, T.c64, False),
    ],
)
def test_other_format_conversions(source_type: type[Scalar], target_type: type[Scalar], expected: bool) -> None:
    assert widens_to(source_type, target_type) is expected


SPARSE = Path(sparse.__file__)
DENSE = np.array(
    [[0, 0, 0, 0], [0.125, 0, -0.801, 0], [0.201, -0.401, 0.601, 0.801], [0, 0, 0, -0.375]],
    np.float64,
)
VECTOR = np.array([0.301, -0.701, 0.901, 0.125], np.float64)
RIGHT = np.array([[0.301, -0.1], [-0.701, 0.5], [0.901, -0.25], [0.125, 0.75]], np.float64)
ARRAY_BUILDERS = {
    "Csr": sparse_build.csr_arrays,
    "Ell": sparse_build.ell_arrays,
    "Bsr": lambda dense, dtype: sparse_build.bsr_arrays(dense, 2, dtype),
    "Dense": sparse_build.dense_arrays,
    "Coo": sparse_build.coo_arrays,
}
FLOAT_PAIRS = [(T.f16, T.f32, T.f32), (T.f32, T.f32, T.f32), (T.f16, T.f16, T.f16)]


def rounded(array: np.ndarray, dtype: type[FloatingPoint]) -> np.ndarray:
    return array.astype(sparse_build.NUMPY[dtype]).astype(np.float64)


def assert_product(actual: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
    tolerance = 2e-3 if actual.dtype == np.float16 else 1e-5
    largest = np.max(np.abs(left[:, :, None] * right.reshape(right.shape[0], -1)[None, :, :]))
    np.testing.assert_allclose(actual, left @ right, rtol=tolerance, atol=tolerance * largest)


@pytest.mark.parametrize("layout", ARRAY_BUILDERS)
@pytest.mark.parametrize(("a_type", "x_type", "out_type"), FLOAT_PAIRS)
@pytest.mark.parametrize("operation", ["spmv", "spmv_vector", "spmm"])
def test_sparse_kernels_use_the_output_dtype(
    runner: Runner,
    layout: str,
    a_type: type[FloatingPoint],
    x_type: type[FloatingPoint],
    out_type: type[FloatingPoint],
    operation: str,
) -> None:
    if layout == "Coo" and operation != "spmv":
        pytest.skip("COO has its own atomic SpMV and no row kernel")
    if layout == "Coo" and out_type == T.f16:
        pytest.skip("atomic.add currently has no f16 output overload")
    matrix = ARRAY_BUILDERS[layout](DENSE, a_type)._asdict()
    x = (RIGHT if operation == "spmm" else VECTOR).astype(sparse_build.NUMPY[x_type])
    out = np.zeros((4, 2) if operation == "spmm" else 4, sparse_build.NUMPY[out_type])
    name = "spmv_coo" if layout == "Coo" else operation
    arguments: dict[str, Value] = {
        "a": matrix,
        "b" if operation == "spmm" else "x": x,
        "c" if operation == "spmm" else "y": out,
    }
    bindings = dtypes(arguments)
    if operation == "spmv_vector" and runner.backend == "cpu":
        with pytest.raises(CaptureError, match=r"WARP_SHUFFLE|shfl_xor"):
            capture(SPARSE, name, structs={"a": layout}, dtypes=bindings, target=runner.target)
        return
    ir = capture(SPARSE, name, structs={"a": layout}, dtypes=bindings, target=runner.target)
    count = len(matrix["values"]) if layout == "Coo" else out.size
    if operation == "spmv_vector":
        count *= runner.target.warp_size
    runner(ir, Items(count), list(arguments.values()))
    assert_product(out, rounded(DENSE, a_type), x.astype(np.float64))


@pytest.mark.parametrize(
    ("a_layout", "b_layout", "a_type", "b_type"),
    [
        ("Csr", "Ell", T.f32, T.f16),
        ("Dense", "Csr", T.f16, T.f32),
        ("Bsr", "Csr", T.f16, T.f16),
        ("Csr", "Csr", T.f32, T.f32),
        ("Csr", "Csr", T.f16, T.f32),
    ],
)
def test_spgemm_two_layouts_and_independent_struct_types(
    runner: Runner,
    a_layout: str,
    b_layout: str,
    a_type: type[FloatingPoint],
    b_type: type[FloatingPoint],
) -> None:
    a = ARRAY_BUILDERS[a_layout](DENSE, a_type)._asdict()
    b = ARRAY_BUILDERS[b_layout](DENSE.T, b_type)._asdict()
    out = np.full((4, 4), np.nan, np.float32)
    arguments: dict[str, Value] = {"a": a, "b": b, "c": out}
    ir = capture(
        SPARSE,
        "spgemm",
        structs={"a": a_layout, "b": b_layout},
        dtypes=dtypes(arguments),
        target=runner.target,
    )
    runner(ir, Items(4), list(arguments.values()))
    assert_product(out, rounded(DENSE, a_type), rounded(DENSE.T, b_type))


@pytest.mark.parametrize(("a_type", "x_type", "out_type"), FLOAT_PAIRS)
def test_sddmm_and_clear(
    runner: Runner,
    a_type: type[FloatingPoint],
    x_type: type[FloatingPoint],
    out_type: type[FloatingPoint],
) -> None:
    pattern = sparse_build.csr_arrays(DENSE, a_type)
    a, b = (
        RIGHT.astype(sparse_build.NUMPY[a_type]),
        RIGHT.T.astype(sparse_build.NUMPY[x_type], order="C"),
    )
    out = np.full(len(pattern.values), np.nan, sparse_build.NUMPY[out_type])
    arguments: dict[str, Value] = {"pattern": pattern._asdict(), "a": a, "b": b, "out": out}
    ir = capture(SPARSE, "sddmm", dtypes=dtypes(arguments), target=runner.target)
    runner(ir, Items(4), list(arguments.values()))
    rows, columns = np.nonzero(DENSE)
    expected = (a.astype(np.float64) @ b.astype(np.float64))[rows, columns]
    tolerance = 2e-3 if out_type == T.f16 else 1e-5
    np.testing.assert_allclose(out, expected, rtol=tolerance, atol=tolerance * np.max(np.abs(a)) * np.max(np.abs(b)))
    runner(
        capture(SPARSE, "clear", dtypes={"y": DTYPES[out.dtype]}, target=runner.target),
        Items(out.size),
        [out],
    )
    np.testing.assert_array_equal(out, 0)


def expressions(node: Stmt | Expr) -> Iterator[Expr]:
    if isinstance(node, Expr):
        yield node
    for field in fields(node):
        value = getattr(node, field.name)
        if isinstance(value, Expr):
            yield from expressions(value)
        elif isinstance(value, tuple):
            for part in value:
                if isinstance(part, Expr):
                    yield from expressions(part)


@pytest.mark.parametrize("operation", ["spmv", "spmm", "spgemm", "spmv_coo"])
@pytest.mark.parametrize("mixed", [False, True])
def test_sparse_float_casts_are_only_the_explicit_conversions(tmp_path: Path, operation: str, mixed: bool) -> None:
    """Index and sum-initialization casts predate dtype conversion; the f32 emitted code stays identical."""
    dtype = F16 if mixed else F32
    structs = {"a": "Coo" if operation == "spmv_coo" else "Csr"}
    bindings = {"a.values": dtype, "x": F32, "y": F32, "b": F32, "c": F32}
    if operation == "spgemm":
        structs["b"] = "Csr"
        bindings["b.values"] = F32
    ir = capture(SPARSE, operation, structs=structs, dtypes=bindings)
    casts = [
        e
        for statement in walk(ir.body)
        for e in expressions(statement)
        if isinstance(e, Cast) and e.dtype.is_float and e.operand.dtype.is_float
    ]
    assert [(c.operand.dtype, c.dtype) for c in casts] == ([(F16, F32)] if mixed else [])
    if not mixed:
        before = source(
            tmp_path,
            SPARSE.read_text().replace(".to(y.dtype)", "").replace(".to(c.dtype)", ""),
            "",
        )
        baseline = capture(before, operation, structs=structs, dtypes=bindings)
        assert (
            emit(ir, CPU.of("avx512"), lineinfo=False).source == emit(baseline, CPU.of("avx512"), lineinfo=False).source
        )


@pytest.mark.parametrize("backend", ["cpu", "emulator"])
@pytest.mark.parametrize("operation", ["spmv", "spgemm"])
def test_sparse_launcher_and_emulator_infer_mixed_operands(device: CPUDevice, backend: str, operation: str) -> None:
    a = sparse_build.csr(device, DENSE, T.f16)
    b = sparse_build.csr(device, DENSE.T, T.f32) if operation == "spgemm" else device.empty(T.f32, 4)
    if operation == "spmv":
        np.asarray(b)[:] = VECTOR
    out = device.empty(T.f32, 4, 4) if operation == "spgemm" else device.empty(T.f32, 4)
    function = getattr(sparse, operation)
    if backend == "cpu":
        function(a, b, out, launch=Items(4))
        device.default_stream.record().wait(60)
    else:
        emulate(function, Items(4), a, b, out)
    assert_product(
        np.asarray(out),
        rounded(DENSE, T.f16),
        rounded(DENSE.T if operation == "spgemm" else VECTOR, T.f32),
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (T.f16, T.f32, T.f32),
        (T.f16, T.f16, T.f16),
        (T.f32, T.f64, T.f64),
        (T.f16, T.bf16, T.f32),
        (T.bf16, T.bf16, T.bf16),
        (T.f64, T.f64, T.f64),
    ],
)
def test_sparse_allocated_output_type(
    left: type[FloatingPoint], right: type[FloatingPoint], expected: type[FloatingPoint]
) -> None:
    assert product_type(left, right) == product_type(right, left) == expected


@pytest.mark.parametrize("operation", [lib.Op.SPMV, lib.Op.SPMM])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(("a_type", "x_type", "out_type"), [*FLOAT_PAIRS, (T.f16, T.f16, T.f32), (T.f32, T.f64, T.f64)])
def test_sparse_library_runs_mixed_dtypes(
    library,
    device: CPUDevice,
    operation: lib.Op,
    a_type: type[FloatingPoint],
    x_type: type[FloatingPoint],
    out_type: type[FloatingPoint],
    explicit: bool,
) -> None:
    library("ours")
    matrix = sparse_build.csr(device, DENSE, a_type)
    x = device.empty(x_type, 4, 2) if operation is lib.Op.SPMM else device.empty(x_type, 4)
    np.asarray(x)[:] = RIGHT if operation is lib.Op.SPMM else VECTOR
    function = getattr(lib, operation.value)
    provided = device.empty(out_type, 4, 2) if operation is lib.Op.SPMM else device.empty(out_type, 4)
    out = function(matrix, x, out=provided) if explicit else function(matrix, x)
    device.default_stream.record().wait(60)
    out_type = out_type if explicit else product_type(a_type, x_type)
    assert out.dtype == out_type and (not explicit or out is provided)
    assert_product(np.asarray(out), rounded(DENSE, a_type), np.asarray(x).astype(np.float64))
    problem = lib.Problem(
        operation,
        (a_type, x_type, out_type),
        (4, 4, len(np.asarray(matrix.values))),
        lib.LibOptions.of(device.target),
        layout="Csr",
    )
    kind = lib.SpmmCandidate if operation is lib.Op.SPMM else lib.SpmvCandidate
    assert lib.dispatcher().choose(kind, problem, capturing=False).name == "ours"
    host = device.empty(out_type, *out.shape)
    operands = getattr(lib, f"{operation.name.title()}Operands")(matrix, x, host)
    candidate = getattr(lib_host, f"Host{operation.name.title()}")()
    candidate.run(problem, operands, device.default_stream)
    np.testing.assert_allclose(np.asarray(out), np.asarray(host), rtol=2e-3 if out_type == T.f16 else 1e-5, atol=1e-5)


@pytest.mark.parametrize("operation", [lib.Op.SPMV, lib.Op.SPMM])
def test_accelerate_refuses_mixed_dtypes_and_auto_chooses_ours(
    accelerate, library, device: CPUDevice, operation: lib.Op
) -> None:
    library("vendor")
    a = sparse_build.csr(device, DENSE, T.f16)
    x = device.empty(T.f32, 4, 2) if operation is lib.Op.SPMM else device.empty(T.f32, 4)
    np.asarray(x)[:] = RIGHT if operation is lib.Op.SPMM else VECTOR
    function = getattr(lib, operation.value)
    reason = "Accelerate Sparse here takes f32 for the matrix, x and out, got f16, f32, f32"
    with pytest.raises(lib.NoCandidate, match=reason):
        function(a, x)
    library("auto")
    out = function(a, x)
    device.default_stream.record().wait(60)
    assert_product(np.asarray(out), rounded(DENSE, T.f16), np.asarray(x).astype(np.float64))
    problem = lib.Problem(operation, (T.f16, T.f32, T.f32), (4, 4, 7), lib.LibOptions.of(device.target), layout="Csr")
    kind = lib.SpmmCandidate if operation is lib.Op.SPMM else lib.SpmvCandidate
    assert lib.dispatcher().choose(kind, problem, capturing=False).name == "ours"
    baseline = lib.Problem(operation, (T.f32,) * 3, problem.shape, problem.options, layout="Csr")
    assert lib.dispatcher().choose(kind, baseline, capturing=False).name == "accelerate"


def test_tuning_keys_distinguish_float_dtypes(library, device: CPUDevice) -> None:
    library("ours")
    dispatcher = lib.dispatcher()
    for dtype in (T.f16, T.f32):
        problem = lib.Problem(lib.Op.SPMV, (dtype,) * 3, (4, 4, 7), lib.LibOptions.of(device.target), layout="Csr")
        dispatcher.choose(lib.SpmvCandidate, problem, capturing=False)
    assert len(dispatcher.seen) == 2
    assert {key.split("|")[3] for key in dispatcher.seen} == {"f16,f16,f16", "f32,f32,f32"}


@pytest.mark.parametrize(
    ("operation", "candidate"),
    [
        (lib.Op.SPMV, lib_cusparse.CusparseSpmv()),
        (lib.Op.SPMM, lib_cusparse.CusparseSpmm()),
    ],
)
@pytest.mark.parametrize(
    ("a_type", "x_type", "out_type"),
    list(itertools.product((T.f16, T.bf16, T.f32, T.f64), repeat=3)),
)
def test_cusparse_documented_dtype_pairs(
    operation: lib.Op,
    candidate: lib.SpmvCandidate | lib.SpmmCandidate,
    a_type: type[FloatingPoint],
    x_type: type[FloatingPoint],
    out_type: type[FloatingPoint],
) -> None:
    target = Target.of("sm_90a")
    problem = lib.Problem(operation, (a_type, x_type, out_type), (4, 4, 7), lib.LibOptions.of(target), layout="Csr")
    uniform = a_type == x_type == out_type and a_type in (T.f32, T.f64)
    narrow = a_type == x_type and a_type in (T.f16, T.bf16) and out_type in (a_type, T.f32)
    wide_vector = operation is lib.Op.SPMV and (a_type, x_type, out_type) == (T.f32, T.f64, T.f64)
    reason = candidate.refusal(problem, target)
    if uniform or narrow or wide_vector:
        assert reason is None
    else:
        assert reason is not None and "cuSPARSE documents (A, X, Y) =" in reason
        assert "pairs only; A and X must share a type except f32 with f64" in reason
        assert f"got ({', '.join(map(name_of, problem.dtypes))})" in reason


@pytest.mark.parametrize(
    ("pair", "arch", "reason"),
    [
        ((T.f16, T.f32, T.f32), "avx512", None),
        ((T.f16, T.f16, T.f16), "apple_m4", None),
        ((T.f32, T.f64, T.f64), "sm_90a", None),
        ((T.f16, T.f32, T.f16), "avx512", "losslessly"),
        ((T.f64, T.f32, T.f32), "avx512", "losslessly"),
        ((T.f64, T.f64, T.f64), "apple_m4", "FLOAT64"),
        ((T.bf16, T.f32, T.f32), "avx512", "no bf16 yet"),
        ((T.f8e5m2, T.f32, T.f32), "sm_90a", "take f16, f32 and f64"),
    ],
)
@pytest.mark.parametrize("candidate", [lib_ours.OursSpmv(), lib_ours.OursSpmm()])
def test_ours_sparse_dtype_refusals(
    pair: tuple[type[FloatingPoint], ...],
    arch: str,
    reason: str | None,
    candidate: lib.SpmvCandidate | lib.SpmmCandidate,
) -> None:
    target = Target.of(arch)
    problem = lib.Problem(candidate.op, pair, (4, 4, 7), lib.LibOptions.of(target), layout="Csr")
    actual = candidate.refusal(problem, target)
    if reason is None:
        assert actual is None
    else:
        assert actual is not None and reason in actual
        assert ", ".join(map(name_of, pair)) in actual


@pytest.mark.parametrize("layout", ARRAY_BUILDERS)
@pytest.mark.parametrize("dtype", [T.f16, T.f32, T.f64])
def test_sparse_builders_preserve_value_dtype(device: CPUDevice, layout: str, dtype: type[FloatingPoint]) -> None:
    arrays = ARRAY_BUILDERS[layout](DENSE, dtype)
    matrix = (
        sparse_build.bsr(device, DENSE, 2, dtype)
        if layout == "Bsr"
        else getattr(sparse_build, layout.lower())(device, DENSE, dtype)
    )
    assert matrix.values.dtype == dtype and arrays.values.dtype == sparse_build.NUMPY[dtype]
    np.testing.assert_array_equal(np.asarray(matrix.values), arrays.values)
    np.testing.assert_array_equal(lib_host.product(matrix, np.eye(4)), rounded(DENSE, dtype))
    if isinstance(arrays, sparse_build.CsrArrays):
        uploaded = sparse_build.csr_on(device, arrays, dtype)
        assert uploaded.values.dtype == dtype
        np.testing.assert_array_equal(np.asarray(uploaded.values), arrays.values)


@pytest.mark.parametrize("layout", ARRAY_BUILDERS)
def test_sparse_builders_default_to_identical_f32_bytes(layout: str) -> None:
    default = (
        sparse_build.bsr_arrays(DENSE, 2)
        if layout == "Bsr"
        else getattr(sparse_build, f"{layout.lower()}_arrays")(DENSE)
    )
    explicit = ARRAY_BUILDERS[layout](DENSE, T.f32)
    for before, after in zip(default, explicit, strict=True):
        if isinstance(before, np.ndarray):
            assert isinstance(after, np.ndarray)
            assert before.dtype == after.dtype and before.tobytes() == after.tobytes()
        else:
            assert before == after


@pytest.mark.parametrize("mode", ["ours", "auto"])
def test_half_coo_refuses_missing_atomics_or_uses_host(library, device: CPUDevice, mode: str) -> None:
    library(mode)
    matrix = sparse_build.coo(device, DENSE, T.f16)
    x = device.empty(T.f16, 4)
    np.asarray(x)[:] = VECTOR
    if mode == "ours":
        with pytest.raises(lib.NoCandidate, match=r"atomic.add, which has no f16 output support"):
            lib.spmv(matrix, x)
    else:
        out = lib.spmv(matrix, x)
        assert out.dtype == T.f16
        assert_product(np.asarray(out), rounded(DENSE, T.f16), np.asarray(x).astype(np.float64))

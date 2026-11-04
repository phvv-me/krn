"""C for the host CPU, compiled by clang and called through ctypes on numpy arrays.

A block is a loop nest: grid loops, then a loop over the block's threads. Buffer pointers are `restrict` locals and
struct fields are hoisted into locals before the loops, so the vectorizer sees no aliasing (round2/intel-cpu-tsukuba.md
section 3b). Tile kernels run one thread per block, so a cooperative loop is a plain loop and a barrier is empty.
"""

import ctypes
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from krn.cache.cpu import compile_c as cached_compile
from krn.compiler import space
from krn.compiler.dtypes import (
    BF16,
    BOOL,
    F16,
    F32,
    F64,
    I8X4,
    I16X2,
    PY_FLOAT,
    PY_INT,
    U8,
    U8X4,
    U16X2,
    U32,
    U64,
    Dtype,
    sint,
    uint,
)
from krn.compiler.emit import C_BYTES, LIBM, Renderer, Unit, check_shared, dimensions
from krn.compiler.heuristics import Inlining
from krn.compiler.ir import Fn, Kernel, ParamType, Slot, StructType, TensorType, slots
from krn.facts.targets import CPU
from krn.frontend.geometry import Geometry
from krn.hw.libraries import Libraries

from .compile_options import CompileOptions

CLANG: tuple[str, ...] = (
    shutil.which("clang-20") or shutil.which("clang") or "clang",
    *("-O2", "-march=native", "-mtune=generic", "-shared", "-fPIC", "-fwrapv", "-x", "c"),
)
CACHE = Path(tempfile.gettempdir()) / "krn.compiler.api"
STACK_BYTES = 6 << 20  # buffers are stack arrays; the main thread has 8 MiB

type Value = int | float | np.ndarray | Mapping[str, Value]
type Native = int | float

CTYPES = {
    BOOL: ctypes.c_bool, U8: ctypes.c_uint8, uint(16): ctypes.c_uint16, U32: ctypes.c_uint32, U64: ctypes.c_uint64,
    sint(8): ctypes.c_int8, sint(16): ctypes.c_int16, sint(32): ctypes.c_int32, sint(64): ctypes.c_int64,
    F32: ctypes.c_float, F64: ctypes.c_double, PY_INT: ctypes.c_longlong, I8X4: ctypes.c_uint32, U8X4: ctypes.c_uint32,
    I16X2: ctypes.c_uint32, U16X2: ctypes.c_uint32,
}  # fmt: skip
NUMPY = {
    BOOL: np.dtype("bool"),
    U8: np.dtype("uint8"), uint(16): np.dtype("uint16"), U32: np.dtype("uint32"), U64: np.dtype("uint64"),
    sint(8): np.dtype("int8"), sint(16): np.dtype("int16"), sint(32): np.dtype("int32"), sint(64): np.dtype("int64"),
    F16: np.dtype("float16"), F32: np.dtype("float32"), F64: np.dtype("float64"),
    I8X4: np.dtype("uint32"), U8X4: np.dtype("uint32"), I16X2: np.dtype("uint32"), U16X2: np.dtype("uint32"),
}  # fmt: skip


def checked(name: str, type: TensorType, value: np.ndarray) -> np.ndarray:
    """`value` when it is a C-contiguous array of the dtype and rank the parameter declares, else a TypeError."""
    if value.dtype != NUMPY[type.dtype] or value.ndim != type.rank or not value.flags.c_contiguous:
        raise TypeError(
            f"`{name}` must be a C-contiguous {type.dtype} array of rank {type.rank}, got {value.dtype} {value.shape}"
        )
    return value


def ctype(slot: Slot) -> type[ctypes._CData]:
    """The ctypes type of one C field of a parameter."""
    match slot.role:
        case "ptr":
            return ctypes.c_void_p
        case "dim":
            return ctypes.c_longlong
    return CTYPES[slot.dtype]


class CPURenderer(Renderer[CPU]):
    types: ClassVar[dict[Dtype, str]] = {
        BOOL: "_Bool", U8: "unsigned char", uint(16): "unsigned short", U32: "unsigned", U64: "unsigned long long",
        sint(8): "signed char", sint(16): "short", sint(32): "int", sint(64): "long long", F16: "_Float16",
        BF16: "__bf16", F32: "float", F64: "double", PY_INT: "long long", PY_FLOAT: "double",
        I8X4: "unsigned", U8X4: "unsigned", I16X2: "unsigned", U16X2: "unsigned",
    }  # fmt: skip
    functions: ClassVar[dict[Fn, str]] = LIBM
    libm = "__builtin_"
    qualifiers: ClassVar[Mapping[Inlining, str]] = {
        Inlining.ALWAYS: "static inline __attribute__((always_inline)) ",
        Inlining.INLINE: "static inline ",
        Inlining.PLAIN: "static ",
        Inlining.NEVER: "static __attribute__((noinline)) ",
    }
    unroll_pragma: ClassVar[str] = "#pragma clang loop unroll_count({count})"
    tid, bdim, sync, leaf_join = "tx", "bdim", "", "_"
    restrict, struct_param, depth = "restrict ", "const {type}* {name}", 2
    printf_decl = "int printf(const char *, ...);"

    def hoisted(self, kernel: Kernel) -> list[str]:
        """Struct fields copied into `restrict` locals, so the vectorizer sees plain pointers."""
        lines = []
        for name, type in kernel.params:
            if isinstance(type, StructType):
                for field, field_type in type.fields:
                    for slot in slots(field, field_type):
                        local = self.field(
                            Slot(f"{name}_{slot.name}", slot.dtype, slot.role),
                            (name, slot.name) in kernel.written,
                            self.restrict,
                        )
                        lines.append(f"    {local} = {name}->{slot.name};")
        return lines

    def unit(self, kernel: Kernel, body: list[str], buffers: dict[str, tuple[Dtype, int]]) -> Unit:
        total = sum(count * C_BYTES[self.vtype(dtype)] for dtype, count in buffers.values())
        if total > STACK_BYTES:
            check_shared(kernel, self.target, total)
        geometry = ["long long gx", "long long gy", "long long gz", "long long bdim"]
        header = f"void {kernel.name}({', '.join([*self.parameters(kernel), *geometry])}) {{"
        lines = [*self.struct_decls(kernel), *self.helper_text(), header]
        lines += self.hoisted(kernel)
        lines += [f"    {self.vtype(d)} {n}[{c}] __attribute__((aligned(64)));" for n, (d, c) in buffers.items()]
        lines += [f"    {self.vtype(r.type.dtype)} {r.path[0]}[{r.extent}];" for r in kernel.shared]
        lines += [f"    for (long long b{a} = 0; b{a} < g{a}; ++b{a})" for a in "zyx"]
        lines += ["    for (long long tx = 0; tx < bdim; ++tx) {"]
        lines += [f"        {self.vtype(v.dtype)} {v.name};" for v in kernel.locals]
        return Unit("\n".join([*lines, *body, "    }", "}"]) + "\n", kernel.name)


def compile_c(
    source: str, target: CPU | None = None, cache_dir: Path | None = None, options: CompileOptions | None = None
) -> tuple[Path, float]:
    """Compile or retrieve a CPU shared object."""
    options = options or CompileOptions.resolve(target or CPU.of("avx512"))
    omitted = {f"-std={options.std}", "-nostdlib++", "-fvisibility=hidden"}
    if sys.platform == "darwin":
        omitted |= {"-fveclib=libmvec", "-lmvec"}
    flags = tuple(flag for flag in options.flags() if flag not in omitted)
    return cached_compile(source, (CLANG[0], *flags, "-fwrapv", "-x", "c"), cache_dir)


class CPUProgram:
    """A compiled kernel with inferred geometry, an optional `launch=` override and a prepared `bind(launch)` call."""

    def __init__(
        self,
        kernel: Kernel,
        unit: Unit,
        target: CPU | None = None,
        cache_dir: Path | None = None,
        options: CompileOptions | None = None,
    ) -> None:
        self.kernel, self.unit = kernel, unit
        self.library, self.compile_seconds = compile_c(unit.source, target, cache_dir, options)
        self.function = getattr(Libraries.load(str(self.library)).handle, unit.entry)
        self.structs = {n: self.layout(t) for n, t in kernel.params if isinstance(t, StructType) and t.fields}
        flat = [
            ctypes.c_void_p if isinstance(t, StructType) else ctype(s)
            for n, t in kernel.params
            for s in self.flat(n, t)
        ]
        self.function.argtypes = [*flat, *[ctypes.c_longlong] * 4]

    @staticmethod
    def flat(name: str, type: ParamType) -> list[Slot]:
        """What the C function receives for one parameter: its slots, or one pointer for a struct."""
        if isinstance(type, StructType):
            return [Slot(name, PY_INT, "ptr")] if type.fields else []
        return slots(name, type)

    @staticmethod
    def layout(struct: StructType) -> type[ctypes.Structure]:
        fields = [(s.name, ctype(s)) for name, type in struct.fields for s in slots(name, type)]
        return type(struct.name, (ctypes.Structure,), {"_fields_": fields})

    def __call__(self, *args: Value, launch: Geometry | None = None) -> None:
        self.bind(launch if launch is not None else space.geometry(self.kernel, args))(*args)

    def bind(self, launch: Geometry) -> Callable[..., None]:
        geometry = dimensions(self.kernel, launch)  # the four integers the C function takes after its parameters

        def run(*args: Value) -> None:
            natives, live = self.marshal(args)
            self.function(*natives, *geometry)
            live.clear()  # the structs must outlive the call, which reads them through their addresses

        return run

    def marshal(self, args: Sequence[Value]) -> tuple[list[Native], list[ctypes.Structure]]:
        """The C arguments of a call and the struct instances whose addresses they hold, which must outlive the call.

        Nothing is shared between calls, so threads may marshal and call one program at the same time.
        """
        live: list[ctypes.Structure] = []
        natives = [
            v
            for (name, type), arg in zip(self.kernel.params, args, strict=True)
            for v in self.pack(name, type, arg, live)
        ]
        return natives, live

    def pack(self, name: str, type: ParamType, value: Value, live: list[ctypes.Structure]) -> list[Native]:
        """The C arguments of one Python argument: numbers, an array's address and shape, a struct's address."""
        match type, value:
            case Dtype(), int() | float():
                return [value]
            case TensorType(), np.ndarray():
                return [checked(name, type, value).ctypes.data, *value.shape]
            case StructType(fields=()), _:
                return []  # only Const fields: folded at capture, nothing to pass
            case StructType(), Mapping():
                flat = [
                    v for field, field_type in type.fields for v in self.pack(field, field_type, value[field], live)
                ]
                live.append(instance := self.structs[name](*flat))
                return [ctypes.addressof(instance)]
        raise TypeError(f"argument `{name}` is {value!r}, expected {type}")

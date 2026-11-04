"""MSL for Apple GPUs: one `kernel` function per unit, `metal_stdlib` the only header.

Each tensor is a `device` pointer in a buffer slot of its own. Every scalar and every dimension rides in one `constant`
struct in the last slot, which the host fills from a numpy structured array of the same layout; the kernel copies them
into locals named as the shared emitter expects. A struct parameter is flattened into its fields the way the CPU
renderer hoists them. Threadgroup memory is static in Metal, so the tile buffers are one aligned array cut into pieces.

Metal has no `long long`, no `double` and no `printf`: the C spellings the shared emitter writes are rewritten to
`long` and `ulong` after the fact, a kernel that needs float64 fails at emit time, and print, assert and breakpoint
raise TargetError. `erf`, `expm1`, `log1p` and `cbrt` are missing from the Metal library,
so they are defined from the others.
The library must be built as MSL `LANGUAGE_VERSION`: the system's default for run-time compilation is older.
"""

from collections.abc import Mapping
from typing import ClassVar

from krn.arithmetic import align_up
from krn.facts.targets import Apple, Target
from krn.frontend.atomic import AtomicOp, Strategy, plan

from .dtypes import BF16, BOOL, F16, F32, I8X4, I16X2, PY_FLOAT, PY_INT, U8, U8X4, U16X2, U32, U64, Dtype, sint, uint
from .emit import ATOMIC_UPDATE, HELPERS, LIBM, Renderer, Unit, check_shared
from .errors import TargetError
from .heuristics import Inlining
from .ir import Accumulate, Fn, Kernel, Slot, StructType, slots
from .matrix import Instruction, InstructionSet, MatrixPlan, MatrixUnit, warp_grid
from .text import integer_suffixes, replace_word

LANGUAGE_VERSION = (
    3,
    2,
)  # `atomic_thread_fence` with a memory order and a scope exists from MSL 3.2 (macOS 15)
SPELLINGS = (
    ("unsigned long long", "ulong"),
    ("long long", "long"),
    ("unsigned char", "uchar"),
    ("signed char", "char"),
    ("unsigned short", "ushort"),
    ("unsigned", "uint"),
)
BYTES = {
    "bool": 1,
    "uchar": 1,
    "char": 1,
    "ushort": 2,
    "short": 2,
    "half": 2,
    "uint": 4,
    "int": 4,
    "float": 4,
    "ulong": 8,
    "long": 8,
}
BUILTINS = (  # the thread, block, grid and simdgroup coordinates the catalog's MSL spellings read
    "uint3 _thread [[thread_position_in_threadgroup]]",
    "uint3 _block [[threadgroup_position_in_grid]]",
    "uint3 _blockdim [[threads_per_threadgroup]]",
    "uint3 _griddim [[threadgroups_per_grid]]",
    "uint3 _gid [[thread_position_in_grid]]",
    "uint _lane [[thread_index_in_simdgroup]]",
    "uint _simdgroup [[simdgroup_index_in_threadgroup]]",
)
MATH_HELPERS = {  # float32 functions the Metal library lacks; each is called by its own name once defined
    # Abramowitz and Stegun 7.1.26, an absolute error under 1.5e-7
    Fn.ERF: (
        "_erf",
        "float _erf(float x) { float t = 1.0f / (1.0f + 0.3275911f * fabs(x)); "
        "float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t "
        "+ 0.254829592f) * t * precise::exp(-x * x); return x < 0.0f ? -y : y; }",
    ),
    Fn.EXPM1: (
        "_expm1",
        "float _expm1(float x) { float u = precise::exp(x); return u == 1.0f ? x : (u - 1.0f == -1.0f ? -1.0f : "
        "(u - 1.0f) * x / precise::log(u)); }",  # Kahan's correction keeps the digits of a small x
    ),
    Fn.LOG1P: (
        "_log1p",
        "float _log1p(float x) { float u = 1.0f + x; return u == 1.0f ? x : precise::log(u) * x / (u - 1.0f); }",
    ),
    Fn.CBRT: ("_cbrt", "float _cbrt(float x) { return copysign(precise::pow(fabs(x), 1.0f / 3.0f), x); }"),
}
ATOMIC_NAMES = {
    AtomicOp.ADD: "add", AtomicOp.SUB: "sub", AtomicOp.AND: "and", AtomicOp.OR: "or", AtomicOp.XOR: "xor",
    AtomicOp.MAX: "max", AtomicOp.MIN: "min",
}  # fmt: skip


class AppleSimdgroup(MatrixUnit):
    """Opaque fragments cross into elementwise work through a threadgroup mirror."""

    table: ClassVar[InstructionSet] = InstructionSet(
        (
            Instruction((F16, F16), 8, 0, "simdgroup_multiply_accumulate", m=8, n=8),
            Instruction((BF16, BF16), 8, 0, "simdgroup_multiply_accumulate", m=8, n=8),
        ),
        "Apple GPUs (simdgroup_matrix)",
        (8, 8, 8),
    )
    owned_fragments = False

    def plan(self, s: Accumulate, target: Target) -> MatrixPlan | str:
        if not isinstance(target, Apple):
            return self.table.refusal(s, target, f"{target.arch} has no implemented matrix unit")
        instruction = self.table.match(s)
        if isinstance(instruction, str):
            return self.table.refusal(s, target, instruction)
        bm, bn = s.acc.shape
        bk = s.lhs.shape[1]
        wm, wn = warp_grid(bm, bn, self.table.granule[:2], target.warp_size)
        shared = 2 * (bm * (bk + 8) + bk * (bn + 8)) + 4 * bm * bn
        if shared > target.shared_bytes:
            return self.table.refusal(
                s,
                target,
                f"tile {bm}x{bn}x{bk} needs {shared} bytes of operand stages and accumulator mirror, "
                f"exceeding {target.shared_bytes} bytes of threadgroup memory",
            )
        return MatrixPlan(instruction, bm, bn, bk, wm, wn, target.warp_size)

    def declare(self, name: str, plan: MatrixPlan) -> str:
        return f"simdgroup_float8x8 _f_{name}[{plan.mt}][{plan.nt}];"

    def coordinates(self, plan: MatrixPlan) -> tuple[str, str]:
        raise NotImplementedError("simdgroup_matrix fragment coordinates are opaque")

    def slot(self, name: str) -> str:
        raise NotImplementedError("simdgroup_matrix fragment elements are opaque")

    def active(self, plan: MatrixPlan) -> str:
        return f"_thread.x < {plan.threads}"

    def _transfer(self, name: str, plan: MatrixPlan, mirror: str, operation: str) -> list[str]:
        offset = (
            f"(_simdgroup / {plan.wn} * {plan.bm // plan.wm} + _mt * 8) * {plan.bn}"
            f" + _simdgroup % {plan.wn} * {plan.bn // plan.wn} + _nt * 8"
        )
        return [
            f"if ({self.active(plan)}) {{",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
            f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{",
            f"simdgroup_{operation}(_f_{name}[_mt][_nt], {mirror} + {offset}, {plan.bn});",
            "}",
            "}",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]

    def spill(self, name: str, plan: MatrixPlan, mirror: str) -> list[str]:
        return self._transfer(name, plan, mirror, "store")

    def fill(self, name: str, plan: MatrixPlan, mirror: str) -> list[str]:
        return self._transfer(name, plan, mirror, "load")

    def product(self, name: str, plan: MatrixPlan, a: str, b: str) -> list[str]:
        fragment = "simdgroup_matrix<bfloat, 8, 8>" if plan.instruction.operands[0] == BF16 else "simdgroup_half8x8"
        return [
            f"if ({self.active(plan)}) {{",
            f"uint _wr = _simdgroup / {plan.wn} * {plan.bm // plan.wm};",
            f"uint _wc = _simdgroup % {plan.wn} * {plan.bn // plan.wn};",
            f"for (int _kk = 0; _kk < {plan.bk}; _kk += 8) {{",
            f"{fragment} _a[{plan.mt}], _b[{plan.nt}];",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
            f"simdgroup_load(_a[_mt], {a} + (_wr + _mt * 8) * {plan.bk + 8} + _kk, {plan.bk + 8});",
            "}",
            f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{",
            f"simdgroup_load(_b[_nt], {b} + _kk * {plan.bn + 8} + _wc + _nt * 8, {plan.bn + 8});",
            "}",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
            f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{",
            f"simdgroup_multiply_accumulate(_f_{name}[_mt][_nt], _a[_mt], _b[_nt], _f_{name}[_mt][_nt]);",
            "}",
            "}",
            "}",
            "}",
        ]


class MSLRenderer(Renderer[Apple]):
    matrix_unit: ClassVar[MatrixUnit | None] = AppleSimdgroup()
    bitcast: ClassVar[tuple[str, str]] = ("as_type<uint>({0})", "as_type<float>({0})")
    types: ClassVar[dict[Dtype, str]] = {
        BOOL: "bool", U8: "uchar", uint(16): "ushort", U32: "uint", U64: "ulong", sint(8): "char", sint(16): "short",
        sint(32): "int", sint(64): "long", F16: "half", BF16: "ushort", F32: "float", PY_INT: "long", PY_FLOAT: "float",
        I8X4: "uint", U8X4: "uint", I16X2: "uint", U16X2: "uint",
    }  # fmt: skip
    functions: ClassVar[dict[Fn, str]] = LIBM
    libm = ""
    qualifiers: ClassVar[Mapping[Inlining, str]] = {
        Inlining.ALWAYS: "inline __attribute__((always_inline)) ",
        Inlining.INLINE: "inline ",
        Inlining.PLAIN: "",
        Inlining.NEVER: "__attribute__((noinline)) ",
    }
    tid, bdim, sync, leaf_join = (
        "_thread.x",
        "_blockdim.x",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "_",
    )
    restrict, struct_param, depth = (
        "__restrict ",
        "{type} {name}",
        1,
    )  # M: the Metal compiler accepts `__restrict`
    debuggable, libm_suffix = False, False
    renamed: ClassVar[dict[str, str]] = {
        name: f"_{name}" if fn in MATH_HELPERS else f"precise::{name}" for fn, name in LIBM.items()
    }

    def vtype(self, dtype: Dtype) -> str:
        """Half values compute in float; bf16 tensor storage is raw words."""
        return "float" if dtype in (F16, BF16) else super().vtype(dtype)

    def load(self, dtype: Dtype, pointer: str, index: str) -> str:
        if dtype == BF16:
            return f"(float)as_type<bfloat>({pointer}[{index}])"
        return f"(float){pointer}[{index}]" if dtype == F16 else super().load(dtype, pointer, index)

    def store(self, dtype: Dtype, pointer: str, index: str, value: str) -> str:
        if dtype == BF16:
            return f"{pointer}[{index}] = as_type<ushort>((bfloat)({value}));"
        return f"{pointer}[{index}] = (half)({value});" if dtype == F16 else super().store(dtype, pointer, index, value)

    def value_cast(self, src: Dtype, dst: Dtype, text: str) -> str:
        if dst == BF16:
            return f"(float)((bfloat)({text}))"
        return f"(float)((half)({text}))" if dst == F16 else super().value_cast(src, dst, text)

    def matrix_bits(self, dtype: Dtype, value: str) -> str:
        return f"as_type<bfloat>({value})" if dtype == BF16 else super().matrix_bits(dtype, value)

    def matrix_store(self, dtype: Dtype, pointer: str, index: str, value: str) -> str:
        return (
            f"{pointer}[{index}] = (bfloat)({value});"
            if dtype == BF16
            else super().matrix_store(dtype, pointer, index, value)
        )

    def math(self, fn: Fn, dtype: Dtype, args: list[str]) -> str:
        self._define(fn)
        if fn in HELPERS:  # a formula spells the libm names it uses, so each helper it names must be defined too
            for helper in MATH_HELPERS:
                if f"{{{LIBM[helper]}}}" in HELPERS[fn][1]:
                    self._define(helper)
        return super().math(fn, dtype, args)

    def _define(self, fn: Fn) -> None:
        """Register the helper of a function the Metal library lacks, once."""
        if fn in MATH_HELPERS:
            name, definition = MATH_HELPERS[fn]
            self.helpers.setdefault(name, f"{self.inline}{definition}")

    def atomic(
        self,
        op: AtomicOp,
        dtype: Dtype,
        pointer: str,
        value: str,
        compare: str | None,
        *,
        shared: bool = False,
        scope: str = "device",
    ) -> str:
        """A helper for the address space of the element: a supported integer instruction, else a compare-and-swap
        loop over the bits."""
        choice = plan(op, str(dtype), self.target)
        if choice.strategy is Strategy.REFUSED:
            raise TargetError(
                f"atomic.{op}", self.target.arch, choice.reason, "use 32-bit atomic operands or another target"
            )
        space = "threadgroup" if shared else "device"
        name, ctype = f"_atomic_{op}_{dtype}_{space}", self.vtype(dtype)
        carrier = "atomic_int" if dtype.kind.value == "sint" else "atomic_uint"
        native = choice.strategy is Strategy.NATIVE
        cell = f"{space} {ctype}* p"
        params = f"{cell}, {ctype} c, {ctype} v" if op is AtomicOp.CAS else f"{cell}, {ctype} v"
        body = self.native_body(op, space, carrier, ctype) if native else self.cas_loop(op, dtype, space, ctype)
        self.helpers.setdefault(name, f"{self.inline}{ctype} {name}({params}) {{ {body} }}")
        operands = [pointer, *([compare] if compare is not None else []), value]
        return f"{name}({', '.join(operands)})"

    @staticmethod
    def native_body(op: AtomicOp, space: str, carrier: str, ctype: str) -> str:
        cell = f"({space} {carrier}*)p"
        value = f"({ctype})v"
        if op is AtomicOp.EXCH:
            return f"return ({ctype})atomic_exchange_explicit({cell}, {value}, memory_order_relaxed);"
        if op is AtomicOp.CAS:  # the weak form may fail without a conflict, so retry unless another value is there
            return (
                f"{ctype} e = ({ctype})c; for (;;) {{ e = ({ctype})c; "
                f"if (atomic_compare_exchange_weak_explicit({cell}, &e, {value}, "
                f"memory_order_relaxed, memory_order_relaxed)) return c; if (e != ({ctype})c) return ({ctype})e; }}"
            )
        return f"return ({ctype})atomic_fetch_{ATOMIC_NAMES[op]}_explicit({cell}, {value}, memory_order_relaxed);"

    @staticmethod
    def cas_loop(op: AtomicOp, dtype: Dtype, space: str, ctype: str) -> str:
        """Compare-and-swap on the integer holding the element's bits, retried after contention."""
        if dtype.is_float:
            read, write = "as_type<float>(seen)", "as_type<uint>(n)"
        else:
            read, write = f"({ctype})seen", "(uint)n"
        return (
            f"{space} atomic_uint* q = ({space} atomic_uint*)p; "
            "uint seen = atomic_load_explicit(q, memory_order_relaxed); "
            f"for (;;) {{ {ctype} o = {read}; {ctype} n = ({ctype})({ATOMIC_UPDATE[op]}); "
            "if (atomic_compare_exchange_weak_explicit(q, &seen, " + write + ", memory_order_relaxed, "
            "memory_order_relaxed)) return o; }"
        )

    def unit(self, kernel: Kernel, body: list[str], buffers: dict[str, tuple[Dtype, int]]) -> Unit:
        offsets, dynamic = {}, 0
        for name, (dtype, count) in buffers.items():
            offsets[name] = dynamic
            dynamic += align_up(count * BYTES[self.vtype(dtype)], 16)
        static = sum(ref.extent * BYTES[self.vtype(ref.type.dtype)] for ref in kernel.shared)
        check_shared(kernel, self.target, dynamic + static)
        pointers, scalars = self.slots(kernel)
        assert self.decisions is not None
        qualifier = self.restrict if self.decisions.restrict.value else ""
        arguments = [
            f"{'const ' if self.decisions.const.value and not written else ''}device "
            f"{self.ctype(slot.dtype)}* {qualifier}{slot.name} [[buffer({index})]]"
            for index, (slot, written) in enumerate(pointers)
        ]
        lines = []
        if scalars:
            fields = " ".join(f"{self.ctype(slot.dtype)} {slot.name};" for slot in scalars)
            lines.append(f"struct _Args {{ {fields} }};")
            arguments.append(f"constant _Args& _a [[buffer({len(pointers)})]]")
        chosen = self.decisions.launch_bounds.value
        bound = (
            f"[[max_total_threads_per_threadgroup({max(chosen or self.decisions.block.value, self.threads)})]] "
            if chosen is not None or self.threads
            else ""
        )  # M: on the M4 (MSL 3.2) the attribute compiles only before `kernel`, never after the parameter list
        signature = f"{bound}kernel void {kernel.name}({', '.join([*arguments, *BUILTINS])}) {{"
        lines = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "#pragma clang fp contract(off)",
            *self.helper_text(),
            *lines,
            signature,
        ]
        used = "\n".join(body)
        lines += [
            f"    {self.ctype(slot.dtype)} {slot.name} = _a.{slot.name};"
            for slot in scalars
            if replace_word(used, slot.name, "") != used
        ]
        if buffers:
            lines.append(f"    threadgroup float4 _smem4[{max(dynamic, 16) // 16}];")
            lines.append("    threadgroup uchar* _smem = (threadgroup uchar*)_smem4;")
        lines += [
            f"    threadgroup {ctype}* {n} = (threadgroup {ctype}*)(_smem + {offsets[n]});"
            for n, (d, _) in buffers.items()
            for ctype in [
                "bfloat"
                if self.matrix_buffers.get(n) == BF16
                else self.ctype(self.matrix_buffers[n])
                if n in self.matrix_buffers
                else self.vtype(d)
            ]
        ]
        lines += [f"    threadgroup {self.vtype(r.type.dtype)} {r.path[0]}[{r.extent}];" for r in kernel.shared]
        lines += [f"    {self.vtype(v.dtype)} {v.name};" for v in kernel.locals]
        if self.threads:
            lines.append(f"    if (_blockdim.x < {self.threads} || _blockdim.y != 1 || _blockdim.z != 1) return;")
        source = "\n".join([*lines, *body, "}"]) + "\n"
        source = integer_suffixes(source)
        for pattern, metal in SPELLINGS:
            source = replace_word(source, pattern, metal)
        return Unit(source, kernel.name, dynamic, self.threads)

    @staticmethod
    def slots(kernel: Kernel) -> tuple[list[tuple[Slot, bool]], list[Slot]]:
        """The pointers (with whether the kernel writes through them) and the scalars of the flattened parameters."""
        pointers: list[tuple[Slot, bool]] = []
        scalars: list[Slot] = []

        def add(slot: Slot, written: bool) -> None:
            (pointers.append((slot, written)) if slot.role == "ptr" else scalars.append(slot))

        for name, kind in kernel.params:
            if isinstance(kind, StructType):
                for field, field_kind in kind.fields:
                    for slot in slots(field, field_kind):
                        add(
                            Slot(f"{name}_{slot.name}", slot.dtype, slot.role),
                            (name, slot.name) in kernel.written,
                        )
            else:
                for slot in slots(name, kind):
                    add(slot, (name,) in kernel.written)
        return pointers, scalars

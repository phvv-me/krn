"""Header-free CUDA C++: scalar arithmetic and exact matrix products with register fragments.

Warp fragments live in registers; padded operands and optional fragment mirrors use dynamic shared memory.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from krn.arithmetic import align_up
from krn.data.formats import FloatFormat
from krn.facts.targets import Nvidia, Target
from krn.frontend.atomic import AtomicOp, Strategy, plan

from . import narrow
from .dtypes import (
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
    Kind,
    fp,
    sint,
    uint,
)
from .emit import ATOMIC_UPDATE, C_BYTES, LIBM, Renderer, Unit, check_shared
from .heuristics import Inlining
from .ir import Accumulate, Fn, Kernel
from .matrix import FP8, Instruction, InstructionSet, MatrixPlan, MatrixUnit, warp_grid
from .performance import PerformanceWarning, warn_once

F16_HELPERS = "\n".join(
    [
        '@inline float h2f(unsigned short h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }',
        "@inline unsigned short f2h(float f) {"
        ' unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }',
        "@inline float rd16(float f) { return h2f(f2h(f)); }",
    ]
)
BF16_HELPERS = "\n".join(
    [
        "@inline float bf2f(unsigned short h) { return __uint_as_float((unsigned)h << 16); }",
        "@inline unsigned short f2bf(float f) { unsigned short h; "
        'asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }',
        "@inline float rdbf(float f) { return bf2f(f2bf(f)); }",
    ]
)


class NvidiaMma(MatrixUnit):
    table: ClassVar[InstructionSet] = InstructionSet(
        (
            *(
                Instruction(
                    (dtype, dtype),
                    k,
                    80,
                    f"mma.sync.aligned.m16n8k{k}.row.col.f32.{dtype}.{dtype}.f32",
                )
                for dtype in (F16, BF16)
                for k in (16, 8)
            ),
            # Mixed halves widen exactly to tf32; an f32 operand would lose precision there.
            *(
                Instruction(pair, 8, 80, "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
                for pair in ((F16, BF16), (BF16, F16))
            ),
            # NVIDIA e4m3 means OCP E4M3FN; the FNUZ dtypes are not in FP8, so they never reach this table.
            *(
                Instruction((a, b), 32, 89, f"mma.sync.aligned.m16n8k32.row.col.f32.{an}.{bn}.f32")
                for a, an in zip(FP8, ("e4m3", "e5m2"), strict=True)
                for b, bn in zip(FP8, ("e4m3", "e5m2"), strict=True)
            ),
        ),
        "sm_80 and later",
        (16, 8, 8),
    )
    async_copy = True

    def copy_async(self, destination: str, source: str, size: str) -> str:
        return (
            'asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: '
            f'"r"((unsigned)__cvta_generic_to_shared({destination})), "l"({source}), "r"({size}) : "memory");'
        )

    def commit(self) -> str:
        return 'asm volatile("cp.async.commit_group;" ::: "memory");'

    def wait(self, pending: int) -> str:
        return f'asm volatile("cp.async.wait_group {pending};" ::: "memory");'

    def plan(self, s: Accumulate, target: Target) -> MatrixPlan | str:
        if not isinstance(target, Nvidia) or target.sm < min(i.minimum_sm for i in self.table.instructions):
            return self.table.refusal(s, target, f"{target.arch} has no implemented matrix unit")
        instruction = self.table.match(s, target.sm)
        if isinstance(instruction, str):
            return self.table.refusal(s, target, instruction)
        bm, bn = s.acc.shape
        bk = s.lhs.shape[1]
        wm, wn = warp_grid(bm, bn, self.table.granule[:2], target.warp_size)
        if bm // wm > 64 or bn // wn > 64:
            return self.table.refusal(s, target, f"tile {bm}x{bn} exceeds the 64x64 per-warp register budget")
        return MatrixPlan(instruction, bm, bn, bk, wm, wn, target.warp_size)

    def declare(self, name: str, plan: MatrixPlan) -> str:
        return f"float _f_{name}[{plan.mt}][{plan.nt}][4];"

    def coordinates(self, plan: MatrixPlan) -> tuple[str, str]:
        width = plan.warp_size
        return (
            f"(threadIdx.x / {width} / {plan.wn} * {plan.bm // plan.wm} + _mt * 16 "
            f"+ (threadIdx.x % {width}) / 4 + 8 * (_fr / 2))",
            f"(threadIdx.x / {width} % {plan.wn} * {plan.bn // plan.wn} + _nt * 8 + 2 * (threadIdx.x % 4) + _fr % 2)",
        )

    def slot(self, name: str) -> str:
        return f"_f_{name}[_mt][_nt][_fr]"

    def active(self, plan: MatrixPlan) -> str:
        return f"threadIdx.x < {plan.threads}"

    def product(self, name: str, plan: MatrixPlan, a: str, b: str) -> list[str]:
        """Load warp fragments from padded row-major stages; B is transposed by ldmatrix."""
        if plan.fp8 or plan.tf32:
            return self._wide_product(name, plan, a, b)
        k = plan.instruction.k
        width = plan.warp_size
        lines = [
            f"if ({self.active(plan)}) {{",
            f"unsigned _lane = threadIdx.x % {width};",
            f"unsigned _wr = threadIdx.x / {width} / {plan.wn} * {plan.bm // plan.wm};",
            f"unsigned _wc = threadIdx.x / {width} % {plan.wn} * {plan.bn // plan.wn};",
            "#pragma unroll",
            f"for (int _kk = 0; _kk < {plan.bk}; _kk += {k}) {{",
            f"unsigned _a[{plan.mt}][4], _b[{plan.nt}][2];",
            "#pragma unroll",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
        ]
        # k8 A uses the first two registers (the two row blocks), k16 all four.
        acount = 4 if k == 16 else 2
        aindex = f"(_wr + _mt * 16 + _lane % 16) * {plan.bk + 8} + _kk"
        if k == 16:
            aindex += " + (_lane / 16) * 8"
        lines += self._load_matrix(f"&{a}[{aindex}]", [f"_a[_mt][{i}]" for i in range(acount)])
        lines += ["}", "#pragma unroll"]
        bcount = 2 if k == 16 else 1
        if plan.nt % 2 == 0:
            lines += [f"for (int _np = 0; _np < {plan.nt // 2}; ++_np) {{"]
            row = "_lane % 8 + (_lane / 8 % 2) * 8" if k == 16 else "_lane % 8"
            col = "_lane / 16 * 8" if k == 16 else "(_lane / 8 % 2) * 8"
            bindex = f"(_kk + {row}) * {plan.bn + 8} + _wc + _np * 16 + {col}"
            registers = [f"_b[_np * 2 + {i // bcount}][{i % bcount}]" for i in range(bcount * 2)]
        else:  # an odd tail cannot read the next n block
            lines += [f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{"]
            bindex = f"(_kk + _lane % {k}) * {plan.bn + 8} + _wc + _nt * 8"
            registers = [f"_b[_nt][{i}]" for i in range(bcount)]
        lines += self._load_matrix(f"&{b}[{bindex}]", registers, transpose=True)
        lines += [
            "}",
            "#pragma unroll",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
            "#pragma unroll",
            f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{",
        ]
        outputs = ", ".join(f'"+f"(_f_{name}[_mt][_nt][{i}])' for i in range(4))
        inputs = ", ".join(
            [
                *(f'"r"(_a[_mt][{i}])' for i in range(acount)),
                *(f'"r"(_b[_nt][{i}])' for i in range(bcount)),
            ]
        )
        ar = ", ".join(f"%{4 + i}" for i in range(acount))
        br = ", ".join(f"%{4 + acount + i}" for i in range(bcount))
        lines += [
            f'asm volatile("{plan.instruction.spelling} {{%0, %1, %2, %3}}, '
            f'{{{ar}}}, {{{br}}}, {{%0, %1, %2, %3}};" : {outputs} : {inputs});',
            "}",
            "}",
            "}",
            "}",
        ]
        return lines

    def _wide_product(self, name: str, plan: MatrixPlan, a: str, b: str) -> list[str]:
        lines = [
            f"if ({self.active(plan)}) {{",
            "unsigned _lane = threadIdx.x % 32;",
            f"unsigned _wr = threadIdx.x / 32 / {plan.wn} * {plan.bm // plan.wm};",
            f"unsigned _wc = threadIdx.x / 32 % {plan.wn} * {plan.bn // plan.wn};",
            "#pragma unroll",
            f"for (int _kk = 0; _kk < {plan.bk}; _kk += {plan.instruction.k}) {{",
            f"unsigned _a[{plan.mt}][4], _b[{plan.nt}][2];",
            "#pragma unroll",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
        ]
        if plan.fp8:
            # ldmatrix moves b16 units: four bytes per register, two adjacent units per lane.
            # x4 A visits (rows 0:8, 8:16) at byte columns 0:16, then 16:32.
            # A 16-byte pad keeps the eight 16-byte rows in distinct shared-bank groups.
            ai = f"(_wr + _mt * 16 + _lane % 16) * {plan.bk + 16} + _kk + (_lane / 16) * 16"
            lines += self._load_matrix(f"&{a}[{ai}]", [f"_a[_mt][{i}]" for i in range(4)])
        else:
            # BK+4 is 4 mod 8 floats: the eight g rows occupy every four-bank group once.
            for i in range(4):
                ai = f"(_wr + _mt * 16 + _lane / 4 + {8 * (i % 2)}) * {plan.bk + 4} + _kk + _lane % 4 + {4 * (i // 2)}"
                lines.append(f"_a[_mt][{i}] = __float_as_uint({a}[{ai}]);")
        lines += ["}", "#pragma unroll"]
        if plan.fp8:
            # B is staged [N][K+16]: x4 visits the two k halves of n block j, then j+1.
            # Non-transposed ldmatrix yields four consecutive k bytes for each n and register.
            paired = plan.nt % 2 == 0
            lines += [f"for (int _nt = 0; _nt < {plan.nt}; _nt += {2 if paired else 1}) {{"]
            bi = "(_wc + _nt * 8 + _lane % 8" + (" + (_lane / 16) * 8" if paired else "")
            bi += f") * {plan.bk + 16} + _kk + (_lane / 8 % 2) * 16"
            registers = [f"_b[_nt + {i // 2}][{i % 2}]" for i in range(4 if paired else 2)]
            lines += self._load_matrix(f"&{b}[{bi}]", registers)
        else:
            # B[t][g] spans eight consecutive banks per t. Stride == 8 mod 32 makes
            # t=0..3 cover all 32 banks once; padding is (8-BN)%32 floats.
            lines += [f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{"]
            stride = plan.bn + (8 - plan.bn) % 32
            for i in range(2):
                bi = f"(_kk + _lane % 4 + {4 * i}) * {stride} + _wc + _nt * 8 + _lane / 4"
                lines.append(f"_b[_nt][{i}] = __float_as_uint({b}[{bi}]);")
        outputs = ", ".join(f'"+f"(_f_{name}[_mt][_nt][{i}])' for i in range(4))
        inputs = ", ".join([*(f'"r"(_a[_mt][{i}])' for i in range(4)), *(f'"r"(_b[_nt][{i}])' for i in range(2))])
        lines += [
            "}",
            "#pragma unroll",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
            "#pragma unroll",
            f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{",
            f'asm volatile("{plan.instruction.spelling} {{%0, %1, %2, %3}}, '
            f'{{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};" : {outputs} : {inputs});',
            "}",
            "}",
            "}",
            "}",
        ]
        return lines

    @staticmethod
    def _load_matrix(address: str, registers: list[str], *, transpose: bool = False) -> list[str]:
        count = len(registers)
        slots = ", ".join(f"%{i}" for i in range(count))
        outputs = ", ".join(f'"=r"({r})' for r in registers)
        trans = ".trans" if transpose else ""
        return [
            f"unsigned _p = (unsigned)__cvta_generic_to_shared({address});",
            f'asm volatile("ldmatrix.sync.aligned.m8n8.x{count}{trans}.shared.b16 '
            f'{{{slots}}}, [%{count}];" : {outputs} : "r"(_p));',
        ]


class SimtRenderer[T: Target](Renderer[T]):
    """What CUDA C++ and HIP C++ share: the kernel, dynamic shared memory, the atomics and the intrinsic spellings of
    the catalog. A subclass names its architecture and says how a half-precision number is converted."""

    types: ClassVar[dict[Dtype, str]] = {
        BOOL: "bool", U8: "unsigned char", uint(16): "unsigned short", U32: "unsigned", U64: "unsigned long long",
        sint(8): "signed char", sint(16): "short", sint(32): "int", sint(64): "long long", F16: "unsigned short",
        F32: "float", F64: "double", PY_INT: "long long", PY_FLOAT: "double",
        I8X4: "unsigned", U8X4: "unsigned", I16X2: "unsigned", U16X2: "unsigned",
    }  # fmt: skip
    functions: ClassVar[dict[Fn, str]] = LIBM
    libm = ""
    qualifiers: ClassVar[Mapping[Inlining, str]] = {
        Inlining.ALWAYS: "__device__ __forceinline__ ",
        Inlining.INLINE: "__device__ __forceinline__ ",
        Inlining.PLAIN: "__device__ inline ",
        Inlining.NEVER: "__device__ __noinline__ ",
    }
    tid, bdim, sync, leaf_join = "threadIdx.x", "blockDim.x", "__syncthreads();", "."
    restrict, struct_param, depth = "__restrict__ ", "{type} {name}", 1
    prologue: ClassVar[tuple[str, ...]] = ()  # lines before everything else: an include where the runtime wants one
    trap: ClassVar[str] = "__trap();"
    launch_bounds: ClassVar[str] = "__launch_bounds__({threads}) "
    flushes: bool = False  # set per renderer, one kernel each: an atomic runs a loop because its native form flushes

    @property
    @abstractmethod
    def arch(self) -> str: ...

    def vtype(self, dtype: Dtype) -> str:
        """An f16 value is a float rounded to f16 precision."""
        return "float" if dtype == F16 else super().vtype(dtype)

    def unit(self, kernel: Kernel, body: list[str], buffers: dict[str, tuple[Dtype, int]]) -> Unit:
        offsets, dynamic = {}, 0
        for name, (dtype, count) in buffers.items():
            offsets[name] = dynamic
            dynamic += align_up(count * C_BYTES[self.vtype(dtype)], 16)
        static = sum(ref.extent * C_BYTES[self.vtype(ref.type.dtype)] for ref in kernel.shared)
        check_shared(kernel, self.target, dynamic + static)
        assert self.decisions is not None
        if self.flushes and self.decisions.settings.warn.warn_strict_atomic_cas:
            warn_once(
                kernel.name, self.target.arch, "strict-atomic-cas", STRICT_ATOMIC_CAS, PerformanceWarning, kernel.span
            )
        chosen = self.decisions.launch_bounds.value
        bound = (
            self.launch_bounds.format(threads=max(chosen or self.decisions.block.value, self.threads))
            if chosen is not None or self.threads
            else ""
        )
        if self.decisions.maxregs.value is not None:
            bound = f"__maxnreg__({self.decisions.maxregs.value}) "
        signature = f'extern "C" __global__ {bound}void {kernel.name}({", ".join(self.parameters(kernel))}) {{'
        lines = [*self.prologue, *self.struct_decls(kernel), *self.helper_text(), signature]
        if buffers:
            lines.append("    extern __shared__ char _smem[];")
        lines += [
            f"    {self.vtype(d)}* {n} = ({self.vtype(d)}*)(_smem + {offsets[n]});" for n, (d, _) in buffers.items()
        ]
        lines += [
            f"    alignas(16) __shared__ {self.vtype(r.type.dtype)} {r.path[0]}[{r.extent}];" for r in kernel.shared
        ]
        lines += [f"    {self.vtype(v.dtype)} {v.name};" for v in kernel.locals]
        if self.threads:
            lines += [
                f"    if (blockDim.x < {self.threads}) {{ "
                f'printf("kernel needs {self.threads} threads\\n"); {self.trap} }}'
            ]
        return Unit("\n".join([*lines, *body, "}"]) + "\n", kernel.name, dynamic, self.threads)

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
        """The CUDA instruction where `atomic.plan` says there is one, else a compare-and-swap loop."""
        suffix = "" if scope == "device" else f"_{scope}"
        name, ctype = f"_atomic_{op}_{dtype}{suffix}", self.vtype(dtype)
        params = f"{ctype}* p, {ctype} c, {ctype} v" if op is AtomicOp.CAS else f"{ctype}* p, {ctype} v"
        choice = plan(op, str(dtype), self.target)
        self.flushes = self.flushes or choice.flushes
        native = choice.strategy is Strategy.NATIVE
        body = self.native_body(op, dtype, ctype, scope) if native else self.cas_body(op, dtype, ctype, scope)
        self.helpers.setdefault(name, f"{self.inline}{ctype} {name}({params}) {{ {body} }}")
        operands = [pointer, *([compare] if compare is not None else []), value]
        return f"{name}({', '.join(operands)})"

    def native_body(self, op: AtomicOp, dtype: Dtype, ctype: str, scope: str) -> str:
        return native_body(op, dtype, ctype, scope)

    def cas_body(self, op: AtomicOp, dtype: Dtype, ctype: str, scope: str) -> str:
        return cas_loop(op, dtype, ctype, cas="atomicCAS" + ("" if scope == "device" else f"_{scope}"))


PTX_NARROW = {  # the packed PTX form of each narrow float (PTX ISA cvt section); the fp4 form packs into a byte
    narrow.FORMATS[fp(4, 3)]: "e4m3x2", narrow.FORMATS[fp(5, 2)]: "e5m2x2", narrow.FORMATS[fp(2, 1)]: "e2m1x2",
    narrow.FORMATS[fp(2, 3)]: "e2m3x2", narrow.FORMATS[fp(3, 2)]: "e3m2x2",
}  # fmt: skip


def native_narrow(fmt: FloatFormat) -> str:
    """The cvt instructions of one narrow float, with one value packed as both halves and the other half discarded.
    The encoder keeps the saturation of `.satfinite` and maps a NaN to the canonical code, as the composed text does."""
    ptx, name = PTX_NARROW[fmt], narrow.NAMES[fmt]
    if fmt.bits == 4:
        encode = (
            f'unsigned r; asm("{{ .reg .b8 t; cvt.rn.satfinite.{ptx}.f32 t, %1, %1; cvt.u32.u8 %0, t; }}" '
            ': "=r"(r) : "f"(x));'
        )
        decode = f'unsigned d; asm("{{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.{ptx} %0, t; }}" : "=r"(d) : "r"(c));'
    else:
        encode = f'unsigned short r; asm("cvt.rn.satfinite.{ptx}.f32 %0, %1, %1;" : "=h"(r) : "f"(x));'
        decode = f'unsigned d; asm("cvt.rn.f16x2.{ptx} %0, %1;" : "=r"(d) : "h"((unsigned short)c));'
    mask, nan = (1 << fmt.bits) - 1, narrow.nan_code(fmt)
    return "\n".join(
        [
            f"@inline unsigned _native_encode_{name}(float x) {{ {encode} return x != x ? {nan}u : (r & {mask}u); }}",
            f"@inline float _native_decode_{name}(unsigned c) {{ {decode} float f; "
            'asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short)d)); return f; }',
        ]
    )


# Where `atom.global.add.noftz.f32` kept every subnormal bit on a card (the noftz probe: GH200 and RTX 5080).
NOFTZ_ADD = frozenset({"sm_90a", "sm_120a"})
NOFTZ_ADD_NAME = "_atomic_add_f32_noftz"
NOFTZ_ADD_BODY = (
    'float r; asm volatile("atom.global.add.noftz.f32 %0, [%1], %2;" : "=f"(r) : "l"(p), "f"(v) : "memory"); return r;'
)


class CUDARenderer(SimtRenderer[Nvidia]):
    """Header-free: the half-precision conversions are inline PTX, not a header."""

    types: ClassVar[dict[Dtype, str]] = SimtRenderer.types | {
        BF16: "unsigned short",
        **dict.fromkeys(FP8, "unsigned char"),
    }
    matrix_unit: ClassVar[MatrixUnit | None] = NvidiaMma()
    bitcast: ClassVar[tuple[str, str]] = ("__float_as_uint({0})", "__uint_as_float({0})")

    def vtype(self, dtype: Dtype) -> str:
        return "float" if dtype == BF16 else super().vtype(dtype)

    @property
    def arch(self) -> str:
        return f"sm_{self.target.sm}a"

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
        """A float32 add into global memory on sm_120a is one `atom.global.add.noftz.f32`, which keeps subnormals where
        the plain instruction flushes them. Every other float32 add keeps the compare-and-swap loop."""
        if op is AtomicOp.ADD and dtype == F32 and not shared and scope == "device" and self.target.arch in NOFTZ_ADD:
            self.helpers.setdefault(
                NOFTZ_ADD_NAME, f"{self.inline}float {NOFTZ_ADD_NAME}(float* p, float v) {{ {NOFTZ_ADD_BODY} }}"
            )
            return f"{NOFTZ_ADD_NAME}({pointer}, {value})"
        return super().atomic(op, dtype, pointer, value, compare, shared=shared, scope=scope)

    def load(self, dtype: Dtype, pointer: str, index: str) -> str:
        value = (
            f"__ldg(&{pointer}[{index}])" if pointer in self.noncoherent and dtype != BOOL else f"{pointer}[{index}]"
        )
        if dtype == BF16:
            self.helpers.setdefault("bf16", BF16_HELPERS)
            return f"bf2f({value})"
        if dtype == F16:
            self.helpers.setdefault("f16", F16_HELPERS)
            return f"h2f({value})"
        return value

    def store(self, dtype: Dtype, pointer: str, index: str, value: str) -> str:
        if dtype == BF16:
            self.helpers.setdefault("bf16", BF16_HELPERS)
            return f"{pointer}[{index}] = f2bf({value});"
        if dtype == F16:
            self.helpers.setdefault("f16", F16_HELPERS)
            return f"{pointer}[{index}] = {self.stored_bits(value) or f'f2h({value})'};"
        return super().store(dtype, pointer, index, value)

    @staticmethod
    def stored_bits(value: str) -> str | None:
        """A direct half read can copy its bits, including NaN payloads."""
        if not (value.startswith("h2f(") and value.endswith(")")):
            return None
        inner, depth = value[4:-1], 0
        for char in inner:
            depth += (char == "(") - (char == ")")
            if depth < 0:
                return None
        return inner if depth == 0 else None

    def value_cast(self, src: Dtype, dst: Dtype, text: str) -> str:
        if dst == BF16:
            self.helpers.setdefault("bf16", BF16_HELPERS)
            return f"rdbf((float)({text}))"
        if dst == F16:
            self.helpers.setdefault("f16", F16_HELPERS)
            return f"rd16((float)({text}))"
        return super().value_cast(src, dst, text)

    def decode(self, dtype: Dtype, text: str) -> str:
        if not narrow.native(self.target, dtype):
            return super().decode(dtype, text)
        name = narrow.NAMES[narrow.FORMATS[dtype]]
        self.helpers.setdefault(f"native {name}", native_narrow(narrow.FORMATS[dtype]))
        return f"_native_decode_{name}({text})"

    def encode(self, dtype: Dtype, text: str) -> str:
        if not narrow.native(self.target, dtype):
            return super().encode(dtype, text)
        name = narrow.NAMES[narrow.FORMATS[dtype]]
        self.helpers.setdefault(f"native {name}", native_narrow(narrow.FORMATS[dtype]))
        return f"_native_encode_{name}({text})"


NATIVE_CALL = {
    AtomicOp.ADD: "atomicAdd", AtomicOp.AND: "atomicAnd", AtomicOp.OR: "atomicOr", AtomicOp.XOR: "atomicXor",
    AtomicOp.EXCH: "atomicExch", AtomicOp.INC: "atomicInc", AtomicOp.DEC: "atomicDec", AtomicOp.MAX: "atomicMax",
    AtomicOp.MIN: "atomicMin", AtomicOp.CAS: "atomicCAS",
}  # fmt: skip
UNSIGNED_64 = {
    AtomicOp.ADD,
    AtomicOp.AND,
    AtomicOp.OR,
    AtomicOp.XOR,
    AtomicOp.EXCH,
    AtomicOp.CAS,
}  # no signed overload


def native_body(op: AtomicOp, dtype: Dtype, ctype: str, scope: str = "device") -> str:
    """`return atomicXxx(p, ...)`; a signed 64-bit operand goes through the unsigned overload CUDA defines."""
    wide = dtype.bits == 64
    suffix = "" if scope == "device" else f"_{scope}"
    if op is AtomicOp.SUB:
        if dtype.is_float:
            return f"return atomicAdd{suffix}(p, -v);"
        if not wide:
            return f"return atomicSub{suffix}(p, v);"
        return f"return ({ctype})atomicAdd{suffix}((unsigned long long*)p, 0ull - (unsigned long long)v);"
    operands = ("c", "v") if op is AtomicOp.CAS else ("v",)
    if wide and dtype.kind is Kind.SINT and op in UNSIGNED_64:
        cast = ", ".join(f"(unsigned long long){o}" for o in operands)
        return f"return ({ctype}){NATIVE_CALL[op]}{suffix}((unsigned long long*)p, {cast});"
    return f"return {NATIVE_CALL[op]}{suffix}(p, {', '.join(operands)});"


STRICT_ATOMIC_CAS = (
    "a float32 atomic runs a compare-and-swap loop, because the native instruction flushes subnormal inputs and "
    "results to zero on global memory; the loop keeps them as the emulator does, and contended elements retry. "
    "Accumulate fixed-point values in a 32-bit integer, whose native atomic keeps every bit, or reduce within each "
    "block so that each block makes one atomic add. Set warn.warn_strict_atomic_cas to False to silence this."
)


def cas_loop(op: AtomicOp, dtype: Dtype, ctype: str, *, cas: str = "atomicCAS") -> str:
    """`atomicCAS` on the integer that holds the element's bits, retried until nobody got in between."""
    wide = dtype.bits == 64
    carrier = "unsigned long long" if wide else "unsigned"
    if dtype.is_float:
        to_bits = "__double_as_longlong" if wide else "__float_as_uint"
        from_bits = "__longlong_as_double" if wide else "__uint_as_float"
        signed = "(long long)" if wide else ""
        read, result, write = (
            f"{from_bits}({signed}seen)",
            f"{from_bits}({signed}old)",
            f"({carrier}){to_bits}(n)",
        )
    else:
        read, result, write = f"({ctype})seen", f"({ctype})old", f"({carrier})n"
    return (
        f"{carrier}* q = ({carrier}*)p; {carrier} seen, old = *q; "
        f"do {{ seen = old; {ctype} o = {read}; {ctype} n = ({ctype})({ATOMIC_UPDATE[op]}); "
        f"old = {cas}(q, seen, {write}); }} while (old != seen); return {result};"
    )

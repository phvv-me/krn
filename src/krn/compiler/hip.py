"""HIP C++ for AMD GPUs (gfx942, the MI300A of Sirius, and gfx950). It is the CUDA renderer with AMD's names, which the
catalog already holds: the same `__global__` kernel, dynamic shared memory (LDS), `threadIdx` and `atomicAdd`, and a
wave of 64 lanes whose masks are 64 bits wide.

The one include is `hip/hip_runtime.h`, which hipRTC supplies itself and hipcc finds in ROCm; a machine without ROCm
checks the units against `tests/hip_shim/`. Half precision is the compiler's own `_Float16`, not inline PTX.
"""

from typing import ClassVar

from krn.facts.targets import AMD, Target
from krn.frontend.atomic import AtomicOp

from .cuda import SimtRenderer, cas_loop
from .dtypes import BF16, F16, Dtype, uint
from .ir import Accumulate
from .matrix import Instruction, InstructionSet, MatrixPlan, MatrixUnit, warp_grid


class AMDMFMA(MatrixUnit):
    """The CDNA k16 instruction with a 64-lane wave and four accumulator values per lane."""

    table: ClassVar[InstructionSet] = InstructionSet(
        (
            Instruction((F16, F16), 16, 0, "__builtin_amdgcn_mfma_f32_16x16x16f16", n=16),
            Instruction((BF16, BF16), 16, 0, "__builtin_amdgcn_mfma_f32_16x16x16bf16_1k", n=16),
        ),
        "",
        (16, 16, 16),
        ("gfx942", "gfx950"),
    )

    def plan(self, s: Accumulate, target: Target) -> MatrixPlan | str:
        if not isinstance(target, AMD) or target.arch not in self.table.arches:
            return self.table.refusal(s, target, f"{target.arch} has no implemented matrix unit")
        instruction = self.table.match(s)
        if isinstance(instruction, str):
            return self.table.refusal(s, target, instruction)
        bm, bn = s.acc.shape
        bk = s.lhs.shape[1]
        wm, wn = warp_grid(bm, bn, self.table.granule[:2], target.warp_size)
        return MatrixPlan(instruction, bm, bn, bk, wm, wn, target.warp_size)

    def declare(self, name: str, plan: MatrixPlan) -> str:
        return (
            "typedef float _mfma_float4 __attribute__((ext_vector_type(4))); "
            f"_mfma_float4 _f_{name}[{plan.mt}][{plan.nt}];"
        )

    def coordinates(self, plan: MatrixPlan) -> tuple[str, str]:
        width = plan.warp_size
        return (
            f"(threadIdx.x / {width} / {plan.wn} * {plan.bm // plan.wm} + _mt * 16 "
            f"+ (threadIdx.x % {width}) / 16 * 4 + _fr)",
            f"(threadIdx.x / {width} % {plan.wn} * {plan.bn // plan.wn} + _nt * 16 + threadIdx.x % 16)",
        )

    def slot(self, name: str) -> str:
        return f"_f_{name}[_mt][_nt][_fr]"

    def active(self, plan: MatrixPlan) -> str:
        return f"threadIdx.x < {plan.threads}"

    def operand_indices(self, plan: MatrixPlan) -> tuple[str, str]:
        return (
            f"(_wr + _mt * 16 + _lane % 16) * {plan.bk + 8} + _kk + _lane / 16 * 4",
            f"(_kk + _lane / 16 * 4 + _j) * {plan.bn + 8} + _wc + _nt * 16 + _lane % 16",
        )

    def product(self, name: str, plan: MatrixPlan, a: str, b: str) -> list[str]:
        """A uses one aligned eight-byte LDS load; B keeps the common K-major stage and gathers four halves."""
        scalar = "_Float16" if plan.instruction.operands[0] == F16 else "short"
        ai, bi = self.operand_indices(plan)
        return [
            f"if ({self.active(plan)}) {{",
            f"unsigned _lane = threadIdx.x % {plan.warp_size};",
            f"unsigned _wr = threadIdx.x / {plan.warp_size} / {plan.wn} * {plan.bm // plan.wm};",
            f"unsigned _wc = threadIdx.x / {plan.warp_size} % {plan.wn} * {plan.bn // plan.wn};",
            f"typedef {scalar} _mfma_operand4 __attribute__((ext_vector_type(4)));",
            f"for (int _kk = 0; _kk < {plan.bk}; _kk += 16) {{",
            f"_mfma_operand4 _a[{plan.mt}], _b[{plan.nt}];",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
            f"_a[_mt] = *reinterpret_cast<const _mfma_operand4*>({a} + {ai});",
            "}",
            f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{",
            "for (int _j = 0; _j < 4; ++_j) {",
            f"_b[_nt][_j] = reinterpret_cast<const {scalar}*>({b})[{bi}];",
            "}",
            "}",
            f"for (int _mt = 0; _mt < {plan.mt}; ++_mt) {{",
            f"for (int _nt = 0; _nt < {plan.nt}; ++_nt) {{",
            f"_f_{name}[_mt][_nt] = {plan.instruction.spelling}(_a[_mt], _b[_nt], _f_{name}[_mt][_nt], 0, 0, 0);",
            "}",
            "}",
            "}",
            "}",
        ]


class HIPRenderer(SimtRenderer[AMD]):
    matrix_unit: ClassVar[MatrixUnit | None] = AMDMFMA()
    trap: ClassVar[str] = "__builtin_trap();"
    launch_bounds: ClassVar[str] = "__attribute__((amdgpu_flat_work_group_size(1, {threads}))) "
    prologue: ClassVar[tuple[str, ...]] = ("#include <hip/hip_runtime.h>",)
    types: ClassVar[dict[Dtype, str]] = SimtRenderer.types | {
        F16: "_Float16",
        BF16: "unsigned short",
        uint(16): "unsigned short",
    }

    def vtype(self, dtype: Dtype) -> str:
        return "float" if dtype == BF16 else super().vtype(dtype)

    def matrix_bits(self, dtype: Dtype, value: str) -> str:
        return f"__builtin_bit_cast(unsigned short, {value})" if dtype == F16 else value

    def matrix_store(self, dtype: Dtype, pointer: str, index: str, value: str) -> str:
        bits = f"__builtin_bit_cast(unsigned short, (_Float16)({value}))" if dtype == F16 else f"_f2bf({value})"
        if dtype == BF16:
            self.bf16_helpers()
        return f"{pointer}[{index}] = {bits};"

    def bf16_helpers(self) -> None:
        self.helpers.setdefault(
            "bf16",
            "@inline float _bf2f(unsigned short h) { return __uint_as_float((unsigned)h << 16); }\n"
            "@inline unsigned short _f2bf(float f) { unsigned u = __float_as_uint(f); "
            "if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40u); "
            "return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }",
        )

    @property
    def arch(self) -> str:
        return self.target.arch

    def native_body(self, op: AtomicOp, dtype: Dtype, ctype: str, scope: str) -> str:
        if scope == "device":
            return super().native_body(op, dtype, ctype, scope)
        memory = "__HIP_MEMORY_SCOPE_WORKGROUP" if scope == "block" else "__HIP_MEMORY_SCOPE_SYSTEM"
        if op in (AtomicOp.INC, AtomicOp.DEC):
            domain = "workgroup" if scope == "block" else ""
            return f'return __builtin_amdgcn_atomic_{op}32(p, v, __ATOMIC_RELAXED, "{domain}");'
        if op is AtomicOp.CAS:
            return (
                f"{ctype} old = c; __hip_atomic_compare_exchange_strong(p, &old, v, "
                f"__ATOMIC_RELAXED, __ATOMIC_RELAXED, {memory}); return old;"
            )
        name = "exchange" if op is AtomicOp.EXCH else f"fetch_{op}"
        return f"return __hip_atomic_{name}(p, v, __ATOMIC_RELAXED, {memory});"

    def cas_body(self, op: AtomicOp, dtype: Dtype, ctype: str, scope: str) -> str:
        if scope == "device":
            return super().cas_body(op, dtype, ctype, scope)
        carrier = "unsigned long long" if dtype.bits == 64 else "unsigned"
        name = f"_hip_cas_{dtype.bits}_{scope}"
        memory = "__HIP_MEMORY_SCOPE_WORKGROUP" if scope == "block" else "__HIP_MEMORY_SCOPE_SYSTEM"
        self.helpers.setdefault(
            name,
            f"{self.inline}{carrier} {name}({carrier}* p, {carrier} c, {carrier} v) {{ "
            f"__hip_atomic_compare_exchange_strong(p, &c, v, __ATOMIC_RELAXED, __ATOMIC_RELAXED, {memory}); "
            "return c; }",
        )
        return cas_loop(op, dtype, ctype, cas=name)

    def load(self, dtype: Dtype, pointer: str, index: str) -> str:
        if dtype == BF16:
            self.bf16_helpers()
            return f"_bf2f({pointer}[{index}])"
        return f"(float){pointer}[{index}]" if dtype == F16 else super().load(dtype, pointer, index)

    def store(self, dtype: Dtype, pointer: str, index: str, value: str) -> str:
        if dtype == BF16:
            self.bf16_helpers()
            return f"{pointer}[{index}] = _f2bf({value});"
        if dtype == F16:
            return f"{pointer}[{index}] = (_Float16)({value});"
        return super().store(dtype, pointer, index, value)

    def value_cast(self, src: Dtype, dst: Dtype, text: str) -> str:
        if dst == BF16:
            self.bf16_helpers()
            return f"_bf2f(_f2bf({text}))"
        return f"(float)((_Float16)({text}))" if dst == F16 else super().value_cast(src, dst, text)

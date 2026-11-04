"""Raw NVIDIA SIMD instructions, one function per instruction, for a program that must have exactly that instruction.

Each function is the instruction of the same name on NVIDIA GPUs and raises TargetError elsewhere, where the lanes are
computed in software. The general functions in `krn.math` pick the instruction for a target themselves. A predicated
instruction returns its predicates with its value, as a tuple, in the order of the CUDA pointer arguments.
"""

from krn.frontend.numerics import i32, todo, u32


def vaddss4(a: u32, b: u32) -> u32:
    """Four signed 8-bit lanes added with saturation.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vaddss4` is one NVIDIA SIMD instruction and this target has no such...)
        AMD: gfx942+ refused (`vaddss4` is one NVIDIA SIMD instruction and this target has no such...)
        CPU: avx2+ refused (`vaddss4` is one NVIDIA SIMD instruction and this target has no such...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vaddus4(a: u32, b: u32) -> u32:
    """Four unsigned 8-bit lanes added with saturation.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vaddus4` is one NVIDIA SIMD instruction and this target has no such...)
        AMD: gfx942+ refused (`vaddus4` is one NVIDIA SIMD instruction and this target has no such...)
        CPU: avx2+ refused (`vaddus4` is one NVIDIA SIMD instruction and this target has no such...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vmaxs4(a: u32, b: u32) -> u32:
    """The larger of each pair of four signed 8-bit lanes.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vmaxs4` is one NVIDIA SIMD instruction and this target has no such...)
        AMD: gfx942+ refused (`vmaxs4` is one NVIDIA SIMD instruction and this target has no such...)
        CPU: avx2+ refused (`vmaxs4` is one NVIDIA SIMD instruction and this target has no such...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vabsdiffs4(a: u32, b: u32) -> u32:
    """The magnitude of the difference of each pair of four signed 8-bit lanes, in the unsigned lane bits.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vabsdiffs4` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vabsdiffs4` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vabsdiffs4` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def viaddmax_s16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, signed: max(a + b, c). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`viaddmax_s16x2` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`viaddmax_s16x2` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`viaddmax_s16x2` is one NVIDIA SIMD instruction and this target has...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def viaddmax_s16x2_relu(a: u32, b: u32, c: u32) -> u32:
    """Per word, signed: max(max(a + b, c), 0). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`viaddmax_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
        AMD: gfx942+ refused (`viaddmax_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
        CPU: avx2+ refused (`viaddmax_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def viaddmax_u16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, unsigned: max(a + b, c). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`viaddmax_u16x2` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`viaddmax_u16x2` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`viaddmax_u16x2` is one NVIDIA SIMD instruction and this target has...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def viaddmax_s32(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(a + b, c). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.viaddmax_s32` accepts i32, got u32)
        Apple: apple_m1+ refused (`viaddmax_s32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`viaddmax_s32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`viaddmax_s32` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def viaddmax_s32_relu(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(max(a + b, c), 0). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.viaddmax_s32_relu` accepts i32, got u32)
        Apple: apple_m1+ refused (`viaddmax_s32_relu` is one NVIDIA SIMD instruction and this target...)
        AMD: gfx942+ refused (`viaddmax_s32_relu` is one NVIDIA SIMD instruction and this target...)
        CPU: avx2+ refused (`viaddmax_s32_relu` is one NVIDIA SIMD instruction and this target...)

    """
    return todo()


def viaddmax_u32(a: u32, b: u32, c: u32) -> u32:
    """Per word, unsigned: max(a + b, c). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`viaddmax_u32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`viaddmax_u32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`viaddmax_u32` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def viaddmin_s16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, signed: min(a + b, c). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`viaddmin_s16x2` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`viaddmin_s16x2` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`viaddmin_s16x2` is one NVIDIA SIMD instruction and this target has...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def viaddmin_s16x2_relu(a: u32, b: u32, c: u32) -> u32:
    """Per word, signed: max(min(a + b, c), 0). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`viaddmin_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
        AMD: gfx942+ refused (`viaddmin_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
        CPU: avx2+ refused (`viaddmin_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def viaddmin_u16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, unsigned: min(a + b, c). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`viaddmin_u16x2` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`viaddmin_u16x2` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`viaddmin_u16x2` is one NVIDIA SIMD instruction and this target has...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def viaddmin_s32(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: min(a + b, c). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.viaddmin_s32` accepts i32, got u32)
        Apple: apple_m1+ refused (`viaddmin_s32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`viaddmin_s32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`viaddmin_s32` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def viaddmin_s32_relu(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(min(a + b, c), 0). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.viaddmin_s32_relu` accepts i32, got u32)
        Apple: apple_m1+ refused (`viaddmin_s32_relu` is one NVIDIA SIMD instruction and this target...)
        AMD: gfx942+ refused (`viaddmin_s32_relu` is one NVIDIA SIMD instruction and this target...)
        CPU: avx2+ refused (`viaddmin_s32_relu` is one NVIDIA SIMD instruction and this target...)

    """
    return todo()


def viaddmin_u32(a: u32, b: u32, c: u32) -> u32:
    """Per word, unsigned: min(a + b, c). The sum wraps to the lane width.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`viaddmin_u32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`viaddmin_u32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`viaddmin_u32` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimax3_s16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, signed: max(a, b, c).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimax3_s16x2` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vimax3_s16x2` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vimax3_s16x2` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimax3_s16x2_relu(a: u32, b: u32, c: u32) -> u32:
    """Per word, signed: max(max(a, b, c), 0).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimax3_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
        AMD: gfx942+ refused (`vimax3_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
        CPU: avx2+ refused (`vimax3_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimax3_u16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, unsigned: max(a, b, c).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimax3_u16x2` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vimax3_u16x2` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vimax3_u16x2` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimax3_s32(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(a, b, c).

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.vimax3_s32` accepts i32, got u32)
        Apple: apple_m1+ refused (`vimax3_s32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vimax3_s32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vimax3_s32` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vimax3_s32_relu(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(max(a, b, c), 0).

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.vimax3_s32_relu` accepts i32, got u32)
        Apple: apple_m1+ refused (`vimax3_s32_relu` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`vimax3_s32_relu` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`vimax3_s32_relu` is one NVIDIA SIMD instruction and this target has...)

    """
    return todo()


def vimax3_u32(a: u32, b: u32, c: u32) -> u32:
    """Per word, unsigned: max(a, b, c).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimax3_u32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vimax3_u32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vimax3_u32` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimin3_s16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, signed: min(a, b, c).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimin3_s16x2` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vimin3_s16x2` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vimin3_s16x2` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimin3_s16x2_relu(a: u32, b: u32, c: u32) -> u32:
    """Per word, signed: max(min(a, b, c), 0).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimin3_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
        AMD: gfx942+ refused (`vimin3_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
        CPU: avx2+ refused (`vimin3_s16x2_relu` is one NVIDIA SIMD instruction and this target...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimin3_u16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, unsigned: min(a, b, c).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimin3_u16x2` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vimin3_u16x2` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vimin3_u16x2` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimin3_s32(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: min(a, b, c).

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.vimin3_s32` accepts i32, got u32)
        Apple: apple_m1+ refused (`vimin3_s32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vimin3_s32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vimin3_s32` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vimin3_s32_relu(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(min(a, b, c), 0).

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.vimin3_s32_relu` accepts i32, got u32)
        Apple: apple_m1+ refused (`vimin3_s32_relu` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`vimin3_s32_relu` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`vimin3_s32_relu` is one NVIDIA SIMD instruction and this target has...)

    """
    return todo()


def vimin3_u32(a: u32, b: u32, c: u32) -> u32:
    """Per word, unsigned: min(a, b, c).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimin3_u32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vimin3_u32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vimin3_u32` is one NVIDIA SIMD instruction and this target has no...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimax_s16x2_relu(a: u32, b: u32) -> u32:
    """Per word, signed: max(max(a, b), 0).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimax_s16x2_relu` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`vimax_s16x2_relu` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`vimax_s16x2_relu` is one NVIDIA SIMD instruction and this target has...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimax_s32_relu(a: i32, b: i32) -> i32:
    """Per word, signed: max(max(a, b), 0).

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.vimax_s32_relu` accepts i32, got u32)
        Apple: apple_m1+ refused (`vimax_s32_relu` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`vimax_s32_relu` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`vimax_s32_relu` is one NVIDIA SIMD instruction and this target has...)

    """
    return todo()


def vimin_s16x2_relu(a: u32, b: u32) -> u32:
    """Per word, signed: max(min(a, b), 0).

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vimin_s16x2_relu` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`vimin_s16x2_relu` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`vimin_s16x2_relu` is one NVIDIA SIMD instruction and this target has...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vimin_s32_relu(a: i32, b: i32) -> i32:
    """Per word, signed: max(min(a, b), 0).

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.vimin_s32_relu` accepts i32, got u32)
        Apple: apple_m1+ refused (`vimin_s32_relu` is one NVIDIA SIMD instruction and this target has...)
        AMD: gfx942+ refused (`vimin_s32_relu` is one NVIDIA SIMD instruction and this target has...)
        CPU: avx2+ refused (`vimin_s32_relu` is one NVIDIA SIMD instruction and this target has...)

    """
    return todo()


def vibmax_s16x2(a: u32, b: u32) -> tuple[u32, bool, bool]:
    """Per halfword max of a and b; the predicates are a >= b for each half.

    Supported on (generated):
        NVIDIA: sm_80+ refused (expected a value, got several values: `intrinsics.vibmax_s16x2(a0[i],...)
        Apple: apple_m1+ refused (`vibmax_s16x2` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vibmax_s16x2` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vibmax_s16x2` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vibmax_s32(a: i32, b: i32) -> tuple[i32, bool]:
    """The max of a and b; the predicate is a >= b.

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.vibmax_s32` accepts i32, got u32)
        Apple: apple_m1+ refused (`vibmax_s32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vibmax_s32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vibmax_s32` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vibmax_u16x2(a: u32, b: u32) -> tuple[u32, bool, bool]:
    """Per halfword max of a and b; the predicates are a >= b for each half.

    Supported on (generated):
        NVIDIA: sm_80+ refused (expected a value, got several values: `intrinsics.vibmax_u16x2(a0[i],...)
        Apple: apple_m1+ refused (`vibmax_u16x2` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vibmax_u16x2` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vibmax_u16x2` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vibmax_u32(a: u32, b: u32) -> tuple[u32, bool]:
    """The max of a and b; the predicate is a >= b.

    Supported on (generated):
        NVIDIA: sm_80+ refused (expected a value, got several values: `intrinsics.vibmax_u32(a0[i],...)
        Apple: apple_m1+ refused (`vibmax_u32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vibmax_u32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vibmax_u32` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vibmin_s16x2(a: u32, b: u32) -> tuple[u32, bool, bool]:
    """Per halfword min of a and b; the predicates are a <= b for each half.

    Supported on (generated):
        NVIDIA: sm_80+ refused (expected a value, got several values: `intrinsics.vibmin_s16x2(a0[i],...)
        Apple: apple_m1+ refused (`vibmin_s16x2` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vibmin_s16x2` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vibmin_s16x2` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vibmin_s32(a: i32, b: i32) -> tuple[i32, bool]:
    """The min of a and b; the predicate is a <= b.

    Supported on (generated):
        NVIDIA: sm_80+ refused (`simd.vibmin_s32` accepts i32, got u32)
        Apple: apple_m1+ refused (`vibmin_s32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vibmin_s32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vibmin_s32` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vibmin_u16x2(a: u32, b: u32) -> tuple[u32, bool, bool]:
    """Per halfword min of a and b; the predicates are a <= b for each half.

    Supported on (generated):
        NVIDIA: sm_80+ refused (expected a value, got several values: `intrinsics.vibmin_u16x2(a0[i],...)
        Apple: apple_m1+ refused (`vibmin_u16x2` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vibmin_u16x2` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vibmin_u16x2` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vibmin_u32(a: u32, b: u32) -> tuple[u32, bool]:
    """The min of a and b; the predicate is a <= b.

    Supported on (generated):
        NVIDIA: sm_80+ refused (expected a value, got several values: `intrinsics.vibmin_u32(a0[i],...)
        Apple: apple_m1+ refused (`vibmin_u32` is one NVIDIA SIMD instruction and this target has no...)
        AMD: gfx942+ refused (`vibmin_u32` is one NVIDIA SIMD instruction and this target has no...)
        CPU: avx2+ refused (`vibmin_u32` is one NVIDIA SIMD instruction and this target has no...)

    """
    return todo()


def vsads2(a: u32, b: u32) -> u32:
    """Sum of the absolute differences of the two signed halfwords.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vsads2` is one NVIDIA SIMD instruction and this target has no such...)
        AMD: gfx942+ refused (`vsads2` is one NVIDIA SIMD instruction and this target has no such...)
        CPU: avx2+ refused (`vsads2` is one NVIDIA SIMD instruction and this target has no such...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vsads4(a: u32, b: u32) -> u32:
    """Sum of the absolute differences of the four signed bytes.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vsads4` is one NVIDIA SIMD instruction and this target has no such...)
        AMD: gfx942+ refused (`vsads4` is one NVIDIA SIMD instruction and this target has no such...)
        CPU: avx2+ refused (`vsads4` is one NVIDIA SIMD instruction and this target has no such...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vsadu2(a: u32, b: u32) -> u32:
    """Sum of the absolute differences of the two unsigned halfwords.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vsadu2` is one NVIDIA SIMD instruction and this target has no such...)
        AMD: gfx942+ refused (`vsadu2` is one NVIDIA SIMD instruction and this target has no such...)
        CPU: avx2+ refused (`vsadu2` is one NVIDIA SIMD instruction and this target has no such...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


def vsadu4(a: u32, b: u32) -> u32:
    """Sum of the absolute differences of the four unsigned bytes.

    Supported on (generated):
        NVIDIA: sm_80+ native
        Apple: apple_m1+ refused (`vsadu4` is one NVIDIA SIMD instruction and this target has no such...)
        AMD: gfx942+ refused (`vsadu4` is one NVIDIA SIMD instruction and this target has no such...)
        CPU: avx2+ refused (`vsadu4` is one NVIDIA SIMD instruction and this target has no such...)
    Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0.

    """
    return todo()


__all__ = [
    "vabsdiffs4",
    "vaddss4",
    "vaddus4",
    "viaddmax_s16x2",
    "viaddmax_s16x2_relu",
    "viaddmax_s32",
    "viaddmax_s32_relu",
    "viaddmax_u16x2",
    "viaddmax_u32",
    "viaddmin_s16x2",
    "viaddmin_s16x2_relu",
    "viaddmin_s32",
    "viaddmin_s32_relu",
    "viaddmin_u16x2",
    "viaddmin_u32",
    "vibmax_s16x2",
    "vibmax_s32",
    "vibmax_u16x2",
    "vibmax_u32",
    "vibmin_s16x2",
    "vibmin_s32",
    "vibmin_u16x2",
    "vibmin_u32",
    "vimax3_s16x2",
    "vimax3_s16x2_relu",
    "vimax3_s32",
    "vimax3_s32_relu",
    "vimax3_u16x2",
    "vimax3_u32",
    "vimax_s16x2_relu",
    "vimax_s32_relu",
    "vimin3_s16x2",
    "vimin3_s16x2_relu",
    "vimin3_s32",
    "vimin3_s32_relu",
    "vimin3_u16x2",
    "vimin3_u32",
    "vimin_s16x2_relu",
    "vimin_s32_relu",
    "vmaxs4",
    "vsads2",
    "vsads4",
    "vsadu2",
    "vsadu4",
]

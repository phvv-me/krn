"""Integer dot products and independent packed byte or halfword lanes, named as CUDA names them.

A u32 holds two 16-bit lanes (suffix 2) or four 8-bit lanes (suffix 4). `ss` and `us` saturate as signed or unsigned,
a trailing `s` or `u` in `vmax`, `vabsdiff`, `vavg` and the comparisons picks the lane signedness, `vcmp*` makes a true
lane all ones and `vset*` makes it 1. Signed averages round ties away from zero, unsigned averages round up (`vhaddu`
truncates).
"""

from krn.frontend.numerics import Integer, i32, todo, u32


def dp4a[T: Integer](a: T, b: T, c: T) -> T:
    """`c` plus the sum of the four byte products of `a` and `b`; the bytes are signed for i32 and unsigned for u32."""
    return todo()


def dp2a_lo[T: Integer](a: T, b: T, c: T) -> T:
    """`c` plus the two products of the 16-bit halves of `a` with bytes 0 and 1 of `b`."""
    return todo()


def dp2a_hi[T: Integer](a: T, b: T, c: T) -> T:
    """`c` plus the two products of the 16-bit halves of `a` with bytes 2 and 3 of `b`."""
    return todo()


def vadd2(a: u32, b: u32) -> u32:
    return todo()


def vadd4(a: u32, b: u32) -> u32:
    return todo()


def vsub2(a: u32, b: u32) -> u32:
    return todo()


def vsub4(a: u32, b: u32) -> u32:
    return todo()


def vaddss2(a: u32, b: u32) -> u32:
    return todo()


def vaddss4(a: u32, b: u32) -> u32:
    return todo()


def vaddus2(a: u32, b: u32) -> u32:
    return todo()


def vaddus4(a: u32, b: u32) -> u32:
    return todo()


def vsubss2(a: u32, b: u32) -> u32:
    return todo()


def vsubss4(a: u32, b: u32) -> u32:
    return todo()


def vsubus2(a: u32, b: u32) -> u32:
    return todo()


def vsubus4(a: u32, b: u32) -> u32:
    return todo()


def vabsdiffs2(a: u32, b: u32) -> u32:
    return todo()


def vabsdiffs4(a: u32, b: u32) -> u32:
    return todo()


def vabsdiffu2(a: u32, b: u32) -> u32:
    return todo()


def vabsdiffu4(a: u32, b: u32) -> u32:
    return todo()


def vmaxs2(a: u32, b: u32) -> u32:
    return todo()


def vmaxs4(a: u32, b: u32) -> u32:
    return todo()


def vmaxu2(a: u32, b: u32) -> u32:
    return todo()


def vmaxu4(a: u32, b: u32) -> u32:
    return todo()


def vmins2(a: u32, b: u32) -> u32:
    return todo()


def vmins4(a: u32, b: u32) -> u32:
    return todo()


def vminu2(a: u32, b: u32) -> u32:
    return todo()


def vminu4(a: u32, b: u32) -> u32:
    return todo()


def vavgs2(a: u32, b: u32) -> u32:
    return todo()


def vavgs4(a: u32, b: u32) -> u32:
    return todo()


def vavgu2(a: u32, b: u32) -> u32:
    return todo()


def vavgu4(a: u32, b: u32) -> u32:
    return todo()


def vhaddu2(a: u32, b: u32) -> u32:
    return todo()


def vhaddu4(a: u32, b: u32) -> u32:
    return todo()


def vneg2(a: u32) -> u32:
    return todo()


def vneg4(a: u32) -> u32:
    return todo()


def vnegss2(a: u32) -> u32:
    return todo()


def vnegss4(a: u32) -> u32:
    return todo()


def vabs2(a: u32) -> u32:
    return todo()


def vabs4(a: u32) -> u32:
    return todo()


def vabsss2(a: u32) -> u32:
    return todo()


def vabsss4(a: u32) -> u32:
    return todo()


def vcmpeq2(a: u32, b: u32) -> u32:
    return todo()


def vcmpeq4(a: u32, b: u32) -> u32:
    return todo()


def vcmpne2(a: u32, b: u32) -> u32:
    return todo()


def vcmpne4(a: u32, b: u32) -> u32:
    return todo()


def vcmpges2(a: u32, b: u32) -> u32:
    return todo()


def vcmpges4(a: u32, b: u32) -> u32:
    return todo()


def vcmpgeu2(a: u32, b: u32) -> u32:
    return todo()


def vcmpgeu4(a: u32, b: u32) -> u32:
    return todo()


def vcmpgts2(a: u32, b: u32) -> u32:
    return todo()


def vcmpgts4(a: u32, b: u32) -> u32:
    return todo()


def vcmpgtu2(a: u32, b: u32) -> u32:
    return todo()


def vcmpgtu4(a: u32, b: u32) -> u32:
    return todo()


def vcmples2(a: u32, b: u32) -> u32:
    return todo()


def vcmples4(a: u32, b: u32) -> u32:
    return todo()


def vcmpleu2(a: u32, b: u32) -> u32:
    return todo()


def vcmpleu4(a: u32, b: u32) -> u32:
    return todo()


def vcmplts2(a: u32, b: u32) -> u32:
    return todo()


def vcmplts4(a: u32, b: u32) -> u32:
    return todo()


def vcmpltu2(a: u32, b: u32) -> u32:
    return todo()


def vcmpltu4(a: u32, b: u32) -> u32:
    return todo()


def vseteq2(a: u32, b: u32) -> u32:
    return todo()


def vseteq4(a: u32, b: u32) -> u32:
    return todo()


def vsetne2(a: u32, b: u32) -> u32:
    return todo()


def vsetne4(a: u32, b: u32) -> u32:
    return todo()


def vsetges2(a: u32, b: u32) -> u32:
    return todo()


def vsetges4(a: u32, b: u32) -> u32:
    return todo()


def vsetgeu2(a: u32, b: u32) -> u32:
    return todo()


def vsetgeu4(a: u32, b: u32) -> u32:
    return todo()


def vsetgts2(a: u32, b: u32) -> u32:
    return todo()


def vsetgts4(a: u32, b: u32) -> u32:
    return todo()


def vsetgtu2(a: u32, b: u32) -> u32:
    return todo()


def vsetgtu4(a: u32, b: u32) -> u32:
    return todo()


def vsetles2(a: u32, b: u32) -> u32:
    return todo()


def vsetles4(a: u32, b: u32) -> u32:
    return todo()


def vsetleu2(a: u32, b: u32) -> u32:
    return todo()


def vsetleu4(a: u32, b: u32) -> u32:
    return todo()


def vsetlts2(a: u32, b: u32) -> u32:
    return todo()


def vsetlts4(a: u32, b: u32) -> u32:
    return todo()


def vsetltu2(a: u32, b: u32) -> u32:
    return todo()


def vsetltu4(a: u32, b: u32) -> u32:
    return todo()


def viaddmax_s16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, signed: max(a + b, c). The sum wraps to the lane width."""
    return todo()


def viaddmax_s16x2_relu(a: u32, b: u32, c: u32) -> u32:
    """Per word, signed: max(max(a + b, c), 0). The sum wraps to the lane width."""
    return todo()


def viaddmax_u16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, unsigned: max(a + b, c). The sum wraps to the lane width."""
    return todo()


def viaddmax_s32(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(a + b, c). The sum wraps to the lane width."""
    return todo()


def viaddmax_s32_relu(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(max(a + b, c), 0). The sum wraps to the lane width."""
    return todo()


def viaddmax_u32(a: u32, b: u32, c: u32) -> u32:
    """Per word, unsigned: max(a + b, c). The sum wraps to the lane width."""
    return todo()


def viaddmin_s16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, signed: min(a + b, c). The sum wraps to the lane width."""
    return todo()


def viaddmin_s16x2_relu(a: u32, b: u32, c: u32) -> u32:
    """Per word, signed: max(min(a + b, c), 0). The sum wraps to the lane width."""
    return todo()


def viaddmin_u16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, unsigned: min(a + b, c). The sum wraps to the lane width."""
    return todo()


def viaddmin_s32(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: min(a + b, c). The sum wraps to the lane width."""
    return todo()


def viaddmin_s32_relu(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(min(a + b, c), 0). The sum wraps to the lane width."""
    return todo()


def viaddmin_u32(a: u32, b: u32, c: u32) -> u32:
    """Per word, unsigned: min(a + b, c). The sum wraps to the lane width."""
    return todo()


def vimax3_s16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, signed: max(a, b, c)."""
    return todo()


def vimax3_s16x2_relu(a: u32, b: u32, c: u32) -> u32:
    """Per word, signed: max(max(a, b, c), 0)."""
    return todo()


def vimax3_u16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, unsigned: max(a, b, c)."""
    return todo()


def vimax3_s32(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(a, b, c)."""
    return todo()


def vimax3_s32_relu(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(max(a, b, c), 0)."""
    return todo()


def vimax3_u32(a: u32, b: u32, c: u32) -> u32:
    """Per word, unsigned: max(a, b, c)."""
    return todo()


def vimin3_s16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, signed: min(a, b, c)."""
    return todo()


def vimin3_s16x2_relu(a: u32, b: u32, c: u32) -> u32:
    """Per word, signed: max(min(a, b, c), 0)."""
    return todo()


def vimin3_u16x2(a: u32, b: u32, c: u32) -> u32:
    """Per halfword, unsigned: min(a, b, c)."""
    return todo()


def vimin3_s32(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: min(a, b, c)."""
    return todo()


def vimin3_s32_relu(a: i32, b: i32, c: i32) -> i32:
    """Per word, signed: max(min(a, b, c), 0)."""
    return todo()


def vimin3_u32(a: u32, b: u32, c: u32) -> u32:
    """Per word, unsigned: min(a, b, c)."""
    return todo()


def vimax_s16x2_relu(a: u32, b: u32) -> u32:
    """Per word, signed: max(max(a, b), 0)."""
    return todo()


def vimax_s32_relu(a: i32, b: i32) -> i32:
    """Per word, signed: max(max(a, b), 0)."""
    return todo()


def vimin_s16x2_relu(a: u32, b: u32) -> u32:
    """Per word, signed: max(min(a, b), 0)."""
    return todo()


def vimin_s32_relu(a: i32, b: i32) -> i32:
    """Per word, signed: max(min(a, b), 0)."""
    return todo()


def vibmax_s16x2(a: u32, b: u32) -> tuple[u32, bool, bool]:
    """Per halfword max of a and b; the predicates are a >= b for each half."""
    return todo()


def vibmax_s32(a: i32, b: i32) -> tuple[i32, bool]:
    """The max of a and b; the predicate is a >= b."""
    return todo()


def vibmax_u16x2(a: u32, b: u32) -> tuple[u32, bool, bool]:
    """Per halfword max of a and b; the predicates are a >= b for each half."""
    return todo()


def vibmax_u32(a: u32, b: u32) -> tuple[u32, bool]:
    """The max of a and b; the predicate is a >= b."""
    return todo()


def vibmin_s16x2(a: u32, b: u32) -> tuple[u32, bool, bool]:
    """Per halfword min of a and b; the predicates are a <= b for each half."""
    return todo()


def vibmin_s32(a: i32, b: i32) -> tuple[i32, bool]:
    """The min of a and b; the predicate is a <= b."""
    return todo()


def vibmin_u16x2(a: u32, b: u32) -> tuple[u32, bool, bool]:
    """Per halfword min of a and b; the predicates are a <= b for each half."""
    return todo()


def vibmin_u32(a: u32, b: u32) -> tuple[u32, bool]:
    """The min of a and b; the predicate is a <= b."""
    return todo()


def vsads2(a: u32, b: u32) -> u32:
    """Sum of the absolute differences of the two signed halfwords."""
    return todo()


def vsads4(a: u32, b: u32) -> u32:
    """Sum of the absolute differences of the four signed bytes."""
    return todo()


def vsadu2(a: u32, b: u32) -> u32:
    """Sum of the absolute differences of the two unsigned halfwords."""
    return todo()


def vsadu4(a: u32, b: u32) -> u32:
    """Sum of the absolute differences of the four unsigned bytes."""
    return todo()

"""PTX bit selection, extraction, insertion, and Boolean truth tables."""

from krn.frontend.numerics import Integer, todo, u32


def bfe[T: Integer](x: T, pos: u32, length: u32) -> T:
    """The `length` bits of `x` from bit `pos`, zero extended (unsigned T) or sign extended (signed T).

    Both counts use their low 8 bits; bits at or above the top of the type read as 0, or as the sign bit for signed T.
    """
    return todo()


def bfi[T: Integer](insert: T, base: T, pos: u32, length: u32) -> T:
    """`base` with its bits `pos` to `pos + length - 1` replaced by the low bits of `insert`; the type's top clips."""
    return todo()


def byte_perm(x: u32, y: u32, selector: u32) -> u32:
    """Byte n of the result is byte `selector >> 4n & 7` of y:x, or that byte's sign when bit 3 is set (PRMT)."""
    return todo()


def funnelshift_l(lo: u32, hi: u32, shift: u32) -> u32:
    """The upper word of hi:lo shifted left by `shift & 31`."""
    return todo()


def funnelshift_lc(lo: u32, hi: u32, shift: u32) -> u32:
    """The upper word of hi:lo shifted left by `shift` clamped to 32."""
    return todo()


def funnelshift_r(lo: u32, hi: u32, shift: u32) -> u32:
    """The lower word of hi:lo shifted right by `shift & 31`."""
    return todo()


def funnelshift_rc(lo: u32, hi: u32, shift: u32) -> u32:
    """The lower word of hi:lo shifted right by `shift` clamped to 32."""
    return todo()


def lop3(a: u32, b: u32, c: u32, table: int) -> u32:
    """Each result bit is bit `a << 2 | b << 1 | c` of `table`, an eight-bit capture-time truth table."""
    return todo()

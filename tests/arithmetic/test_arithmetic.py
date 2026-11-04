import math
from fractions import Fraction

from hypothesis import given
from hypothesis import strategies as st

from krn.arithmetic import align_up, ceildiv

nonzero = st.integers(-(10**12), 10**12).filter(bool)


@given(st.integers(-(10**12), 10**12), nonzero)
def test_ceildiv_is_the_exact_ceiling(numerator: int, denominator: int) -> None:
    assert ceildiv(numerator, denominator) == math.ceil(Fraction(numerator, denominator))


@given(st.integers(0, 10**12), st.integers(1, 4096))
def test_align_up_is_the_next_multiple(value: int, multiple: int) -> None:
    aligned = align_up(value, multiple)
    assert aligned % multiple == 0
    assert 0 <= aligned - value < multiple

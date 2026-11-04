"""Dtype names: a float prints as `f<bits>e<exponent>m<mantissa>`, with the fn suffix on the named OCP formats, and the
FNUZ fp8 floats are dtypes of their own."""

import pytest

from krn.compiler.dtypes import F16, F32, Dtype, fp, from_sketch
from krn.frontend import numerics


@pytest.mark.parametrize(
    ("dtype", "name"),
    [
        (F16, "f16"),
        (F32, "f32"),
        (fp(4, 3), "f8e4m3fn"),
        (fp(5, 2), "f8e5m2"),
        (fp(2, 3), "f6e2m3"),
        (fp(3, 2), "f6e3m2"),
        (fp(2, 1), "f4e2m1"),
        (fp(3, 4), "f8e3m4"),
        (fp(2, 5), "f8e2m5"),
    ],
)
def test_a_float_prints_as_its_exponent_and_mantissa_widths(dtype: Dtype, name: str) -> None:
    assert str(dtype) == name


@pytest.mark.parametrize(
    ("sketch", "dtype", "name"),
    [
        (numerics.f8e4m3fnuz, fp(4, 3, fnuz=True), "f8e4m3fnuz"),
        (numerics.f8e5m2fnuz, fp(5, 2, fnuz=True), "f8e5m2fnuz"),
    ],
)
def test_an_fnuz_float_is_its_own_dtype_beside_the_ocp_format(
    sketch: type[numerics.Scalar], dtype: Dtype, name: str
) -> None:
    assert from_sketch(sketch) == dtype and str(dtype) == name
    assert dtype != fp(4, 3) and dtype != fp(5, 2)


@pytest.mark.parametrize(("sketch", "dtype"), [(numerics.f8e4m3fn, fp(4, 3)), (numerics.f8e5m2, fp(5, 2))])
def test_an_ocp_float_keeps_its_plain_dtype(sketch: type[numerics.Scalar], dtype: Dtype) -> None:
    assert from_sketch(sketch) == dtype and not dtype.fnuz

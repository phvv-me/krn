"""Number protocols parsed by capture, including deliberate refusals."""

from krn import Matrix, T, Vector, exp, grid, items, kernel, math
from krn.frontend.memory import Tensor3


@kernel
def split(x: Vector[T.u32], divisor: T.u32, out: Matrix[T.u32]) -> None:
    for i in items(x.shape[0]):
        q, r = divmod(x[i], divisor)
        a, b = divmod(x[i], 7)
        c, d = divmod(255, x[i] + 1)
        out[i, 0] = q
        out[i, 1] = r
        out[i, 2] = a
        out[i, 3] = b
        out[i, 4] = c
        out[i, 5] = d


@kernel
def powers(x: Vector[T.f32], out: Matrix[T.f32]) -> None:
    for i in items(x.shape[0]):
        v = x[i]
        out[i, 0] = v**0
        out[i, 1] = v**1
        out[i, 2] = v**2
        out[i, 3] = v**3
        out[i, 4] = v**5
        out[i, 5] = v**9
        out[i, 6] = pow(v, 3)
        out[i, 7] = v**0.5
        out[i, 8] = 2.0**v
        out[i, 9] = v**-2
        out[i, 10] = v**v
        v **= 2
        out[i, 11] = v


@kernel
def int_powers(x: Vector[T.u8], y: Vector[T.i32], out8: Vector[T.u8], out32: Vector[T.i32]) -> None:
    for i in items(x.shape[0]):
        out8[i] = x[i] ** 3
        v = y[i]
        v **= 3
        out32[i] = v


@kernel
def tile_powers(x: Vector[T.f32], zero: Vector[T.f32], odd: Vector[T.f32], large: Vector[T.f32]) -> None:
    v = x[0:4]
    zero[0:4] = v**0
    odd[0:4] = v**5
    large[0:4] = v**9


@kernel
def magnitudes(
    x: Vector[T.i32],
    y: Vector[T.f32],
    z: Vector[T.u32],
    outi: Vector[T.i32],
    outf: Vector[T.f32],
    outu: Vector[T.u32],
) -> None:
    for i in items(x.shape[0]):
        outi[i] = abs(x[i])
        outf[i] = abs(y[i])
        outu[i] = abs(z[i])


@kernel
def extrema(
    x: Vector[T.u8],
    y: Vector[T.i32],
    a: Vector[T.f32],
    b: Vector[T.f32],
    out8: Matrix[T.u8],
    out32: Matrix[T.i32],
    outf: Matrix[T.f32],
) -> None:
    for i in items(x.shape[0]):
        out8[i, 0] = min(x[i], 7)
        out8[i, 1] = max(7, x[i])
        out32[i, 0] = min(-3, y[i])
        out32[i, 1] = max(y[i], -3)
        outf[i, 0] = min(a[i], b[i])
        outf[i, 1] = max(a[i], b[i])
        outf[i, 2] = min(b[i], a[i])
        outf[i, 3] = max(b[i], a[i])


@kernel
def bit_widths(x: Vector[T.u8], y: Vector[T.u32], z: Vector[T.u64], out: Matrix[T.u32]) -> None:
    for i in items(x.shape[0]):
        out[i, 0] = x[i].bit_count()
        out[i, 1] = x[i].bit_length()
        out[i, 2] = y[i].bit_count()
        out[i, 3] = y[i].bit_length()
        out[i, 4] = z[i].bit_count()
        out[i, 5] = z[i].bit_length()


@kernel
def unary_math(x: Vector[T.f32], out: Matrix[T.f32]) -> None:
    for i in items(x.shape[0]):
        v = x[i]
        out[i, 0] = math.sqrt(v)
        out[i, 1] = math.exp(v)
        out[i, 2] = math.exp2(v)
        out[i, 3] = math.log(v)
        out[i, 4] = math.log2(v)
        out[i, 5] = math.log10(v)
        out[i, 6] = math.sin(v)
        out[i, 7] = math.cos(v)
        out[i, 8] = math.tanh(v)
        out[i, 9] = math.floor(v)
        out[i, 10] = math.ceil(v)
        out[i, 11] = math.trunc(v)
        out[i, 12] = math.cbrt(v)
        out[i, 13] = v + math.pi + math.e + math.tau
        out[i, 14] = exp(v)


@kernel
def binary_math(x: Vector[T.f32], out: Matrix[T.f32]) -> None:
    for i in items(x.shape[0]):
        v = x[i]
        out[i, 0] = 1.0 if math.isnan(v) else 0.0
        out[i, 1] = 1.0 if math.isfinite(v) else 0.0
        out[i, 2] = math.fma(v, 2.0, 1.0)


@kernel
def copy_sign(x: Vector[T.f32], signs: Vector[T.f32], out: Matrix[T.f32]) -> None:
    for i in items(x.shape[0]):
        out[i, 0] = math.copysign(x[i], signs[i])
        out[i, 1] = math.copysign(2.0, x[i])
        out[i, 2] = math.copysign(x[i], -0.0)


@kernel
def scale_exponent(x: Vector[T.f32], n: Vector[T.i64], out: Vector[T.f32]) -> None:
    for i in items(x.shape[0]):
        out[i] = math.ldexp(x[i], n[i])


@kernel
def double_math(x: Vector[T.f64], out: Matrix[T.f64]) -> None:
    for i in items(x.shape[0]):
        out[i, 0] = math.sqrt(x[i])
        out[i, 1] = x[i] ** 9
        out[i, 2] = math.fma(x[i], 2.0, 1.0)


@kernel
def generic_numbers(x: Vector[T.u8], y: Vector[T.f32], n: int, f: float, a: Vector[T.u8], b: Vector[T.f32]) -> None:
    for i in items(x.shape[0]):
        a[i] = x[i] + n
        b[i] = f + y[i]


@kernel
def generic_integer(x: Vector[T.u8], n: int, out: Vector[T.u8]) -> None:
    for i in items(x.shape[0]):
        out[i] = x[i] + n


@kernel
def cube(x: Tensor3[T.f32], out: Tensor3[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        out[i, 1, 2] = x[i, 1, 2] + 1.0


@kernel
def constants(out: Vector[T.f32]) -> None:
    if grid.thread_rank() == 0:
        out[0] = 2**3
        out[1] = 2**-2
        out[2] = pow(3.0, 2)
        out[3] = abs(-4)
        out[4] = min(8, 2)
        out[5] = max(8, 2)
        out[6] = (255).bit_count()
        out[7] = (256).bit_length()
        a, b = divmod(17, 5)
        out[8] = a
        out[9] = b

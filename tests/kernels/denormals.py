"""One kernel per float32 operation of the denormal measurement: `out[i]` from `a[i]`, `b[i]` and `c[i]`."""

from krn import T, Vector, f16, f32, items, kernel, math


@kernel
def op_add(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = a[i] + b[i]


@kernel
def op_sub(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = a[i] - b[i]


@kernel
def op_mul(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = a[i] * b[i]


@kernel
def op_div(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = math.div(a[i], b[i])


@kernel
def op_sqrt(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = math.sqrt(a[i])


@kernel
def op_rsqrt(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = math.rsqrt(a[i])


@kernel
def op_fma(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = math.fma(a[i], b[i], c[i])


@kernel
def op_minimum(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = min(a[i], b[i])


@kernel
def op_maximum(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = max(a[i], b[i])


@kernel
def op_copy(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = a[i]


@kernel
def op_compare(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = 1.0 if a[i] > b[i] else 0.0


@kernel
def op_convert(a: Vector[T.f32], b: Vector[T.f32], c: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(a.shape[0]):
        out[i] = f32(f16(a[i]))

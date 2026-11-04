"""Generic number types in kernels: sub-byte integers, odd widths, fp8 flavors per vendor, complex.

`Matrix[T.u3]` is stored bit-packed along its last axis and read as an unpacked tile: the kernel never mentions packing.
Any width is a type: `T.uint[T.L[10]]` is a 10-bit sensor value and `T.fp[T.L[6], T.L[5]]` a 12-bit float with no new
class. Sirius' MI300A takes `T.f8e4m3fnuz`, NVIDIA and gfx950 take `T.f8e4m3fn`: the host picks the alias from the
target, or says `T.fp.mmax(8)` and lets the target choose.
"""

from krn import Const, Matrix, T, Vector, block, grid, kernel, struct, target
from krn.facts import targets as targets


@struct
class Tiling:
    rows: Const[int] = 64
    cols: Const[int] = 128


@kernel
def dequantize_u3(w: Matrix[T.u3], scale: T.f16, out: Matrix[T.f16], t: Tiling) -> None:
    rows, cols = block.tile(t.rows, t.cols)
    out[rows, cols] = w[rows, cols].to(T.f16) * scale


def dequantize[Q: (T.u2, T.u3, T.u4, T.i4)](w: Matrix[Q], scale: T.f16, out: Matrix[T.f16], t: Tiling) -> None:
    """The same body for every listed width. The checker verifies it once per member of the constraint."""
    rows, cols = block.tile(t.rows, t.cols)
    out[rows, cols] = w[rows, cols].to(T.f16) * scale


@kernel
def quantize_fp8(x: Matrix[T.f32], scale: T.f32, out: Matrix[T.f8e4m3fn], t: Tiling) -> None:
    """OCP e4m3fn (max 448). On gfx942 the format is fnuz (max 240), chosen by the host from the target."""
    rows, cols = block.tile(t.rows, t.cols)
    out[rows, cols] = (x[rows, cols] * scale).to(T.f8e4m3fn)


@kernel
def quantize_fp8_gfx942(x: Matrix[T.f32], scale: T.f32, out: Matrix[T.f8e4m3fnuz], t: Tiling) -> None:
    rows, cols = block.tile(t.rows, t.cols)
    out[rows, cols] = (x[rows, cols] * scale).to(T.f8e4m3fnuz)


@kernel
def caxpy(a: T.c64, x: Vector[T.c64], y: Vector[T.c64]) -> None:
    """cfp arithmetic is a type, not two arrays: c64 is cfp[f32]."""
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


def on_cpu() -> bool:
    return isinstance(target, targets.CPU)


@kernel
def sensor_to_f16(raw: Matrix[T.uint[T.L[10]]], gain: T.f16, out: Matrix[T.f16], t: Tiling) -> None:
    """A 10-bit ADC image to f16. The 10-bit values are packed in memory and unpacked in registers."""
    rows, cols = block.tile(t.rows, t.cols)
    out[rows, cols] = raw[rows, cols].to(T.f16) * gain


@kernel
def to_float12(x: Matrix[T.f32], out: Matrix[T.fp[T.L[6], T.L[5]]], t: Tiling) -> None:
    """A float with 6 exponent bits and 5 mantissa bits (plus sign): an ieee-style 12-bit format, written as a type."""
    rows, cols = block.tile(t.rows, t.cols)
    out[rows, cols] = x[rows, cols].to(T.fp[T.L[6], T.L[5]])

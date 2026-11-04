"""Deliberate mistakes in the sketch syntax, one per kernel, with the marker of the offending text on its line."""

from krn import (
    AUTO,
    Const,
    Matrix,
    T,
    Vector,
    block,
    copy,
    formats,
    grid,
    host,
    kernel,
    pipelined,
    struct,
    warp,
    zeros,
)


@struct
class NoDefault:
    bm: Const[int]


@kernel
def store_f16_into_f32(x: Vector[T.f32], h: Vector[T.f16]) -> None:
    i = grid.thread_rank()
    x[i] = h[i]


@kernel
def u32_plus_u8(b: Vector[T.u8], n: T.u32) -> None:
    i = grid.thread_rank()
    j = n + b[i]


@kernel
def f32_plus_u32(x: Vector[T.f32], n: T.u32) -> None:
    i = grid.thread_rank()
    x[i] = x[i] + n


@kernel
def u64_less_u32(x: Vector[T.f32], n: T.u32) -> None:
    i = grid.thread_rank()
    if i < n:
        x[i] = T.f32(0.0)


@kernel
def dynamic_extent(a: Matrix[T.f16], n: T.u32) -> None:
    rows = block.tile(8)
    t = a[rows, 0 : a.shape[1]]


@kernel
def missing_const(a: Matrix[T.f32], t: NoDefault) -> None:
    rows = block.tile(t.bm)


@kernel
def too_many_indices(x: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    x[i, i] = T.f32(0.0)


@kernel
def tile_dtype_mix(a: Matrix[T.f16], b: Matrix[T.f32], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    c[rows, cols] = a[rows, cols] + b[rows, cols]


@kernel
def inner_dimension(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    acc = zeros(T.f32, 8, 8)
    acc += a[rows, 0:32] @ b[0:64, cols]
    c[rows, cols] = acc


@kernel
def tile_into_f16(a: Matrix[T.f16], out: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    out[rows, cols] = a[rows, cols]


@kernel
def scalar_into_tile(c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    c[rows, cols] = T.f32(1.0)


@kernel
def arbitrary_widths(a: T.u3, b: T.u4, x: Vector[T.u8]) -> None:
    x[0] = T.u8(a + b)


@kernel
def literal_too_big(x: Vector[T.u8]) -> None:
    i = grid.thread_rank()
    x[i] = x[i] + 300


@kernel
def bad_axis(a: Matrix[T.f32], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    c[rows, cols] = a[rows, cols] - a[rows, cols].max(axis=3, keepdims=True)


@kernel
def warp_op_on_cpu(x: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    x[i] = warp.shfl_xor(x[i], 1)


@kernel
def not_a_bool(x: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i:
        x[i] = T.f32(0.0)


@kernel
def unsupported(x: Vector[T.f32]) -> None:
    with x:
        pass


@kernel
async def async_kernel(a: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    first = copy(a[rows, cols])
    c[rows, cols] = (await first).to(T.f32)


@kernel
def host_in_kernel(x: Vector[T.f32]) -> None:
    device = host.Device.current()


@kernel
def pipelined_over_range(a: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    acc = zeros(T.f32, 8, 8)
    for k in pipelined(range(0, a.shape[1], 8), stages=2):
        acc += a[rows, k : k + 8] @ a[k : k + 8, cols]
    c[rows, cols] = acc


@kernel
def pipelined_stages_zero(a: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    acc = zeros(T.f32, 8, 8)
    for ks in pipelined(a.shape[1], 8, stages=0):
        acc += a[rows, ks] @ a[ks, cols]
    c[rows, cols] = acc


@kernel
def pipelined_dynamic_tile(a: Matrix[T.f16], n: T.u32) -> None:
    for ks in pipelined(a.shape[1], n):
        pass


@kernel
def auto_as_a_value(x: Vector[T.f32]) -> None:
    x[0] = AUTO


@kernel
def in_place_on_expression(a: Matrix[T.f32], c: Matrix[T.f32]) -> None:
    rows, cols = block.tile(8, 8)
    c[rows, cols] = (a[rows, cols] + 1.0).exp_()


@kernel
def exp_of_half(a: Matrix[T.f16], c: Matrix[T.f16]) -> None:
    rows, cols = block.tile(8, 8)
    c[rows, cols] = a[rows, cols].exp()


@kernel
def tile_needs_one_or_two_sizes(a: Matrix[T.f32]) -> None:
    rows, cols, planes = block.tile(8, 8, 8)


@kernel
def format_as_a_value(x: Vector[T.f32]) -> None:
    packed = formats.NVFP4


@kernel
def several_values_in_one_name(a: Matrix[T.f32]) -> None:
    both = block.tile(8, 8)


@kernel
def selector_without_a_float(a: Matrix[T.f32]) -> None:
    acc = zeros(T.fp.emax(4), 8, 8)

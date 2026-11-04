"""Kernels in the sketch syntax that the examples do not cover: integer wrap-around, the lossless widening, a while
loop, a compile-time branch on the target, and forward references to a struct defined further down."""

from krn import Const, T, Vector, grid, kernel, struct, target


@kernel
def wrap(x: Vector[T.u8], y: Vector[T.u8], z: Vector[T.u16], w: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = x[i] + x[i] + 100
        z[i] = T.u16(x[i]) * 300 - 1
        w[i] = T.u32(z[i]) * 70000 + T.u32(x[i])


@kernel
def widen(x: Vector[T.u8], z: Vector[T.u16], w: Vector[T.u32], f: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        w[i] = T.u32(x[i] + z[i]) + w[i]
        f[i] = T.f32(w[i]) * 0.5


@kernel
def collatz(x: Vector[T.u32], steps: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        n = x[i]
        count = T.u32(0)
        while n > 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = n * 3 + 1
            count += 1
        steps[i] = count


@kernel
def shadow(x: Vector[T.f32]) -> None:
    tx = grid.thread_rank()
    float = T.f32(1.0)
    if tx < x.shape[0]:
        x[tx] = float


@kernel
def lanes(out: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    if i < out.shape[0]:
        if target.warp_size == 32:
            out[i] = T.u32(32)
        else:
            out[i] = T.u32(target.warp_size)


@kernel
def scale(x: Vector[T.f32], y: Vector[T.f32], cfg: Cfg) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = x[i] * cfg.factor + cfg.shift


@struct
class Cfg:
    factor: Const[float] = 2.0
    shift: T.f32

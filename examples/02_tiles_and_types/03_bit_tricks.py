"""Integer bit tricks and scalar math helpers: popcount, reversed bits, fma, cbrt

The bit functions (`popcount`, `reverse_bits`, `leading_zeros`, `first_set`) take any fixed-width integer, and the
right instruction follows from its width (`__popc` or `__popcll`, `popcount`, a clang builtin). The scalar math
helpers `fma` (which rounds once), `cbrt`, `log10`, `exp10` and `trunc` complete what the fluent methods of a float
tile already do, and an `f16` value needs no `h`-prefixed functions because it has the operators and the methods.
Everything runs on the CPU backend and is compared with Python integers and numpy.

Level: 2
Topics: integer types, bit intrinsics, math helpers
Needs: 01_first_kernels/01_saxpy
"""

import asyncio
import logging

import numpy as np

from krn import (
    Items,
    T,
    Vector,
    catalog,
    cbrt,
    first_set,
    fma,
    grid,
    host,
    kernel,
    leading_zeros,
    log10,
    popcount,
    reverse_bits,
    trunc,
)
from krn.compiler.api import CPUDevice

log = logging.getLogger("bits")


@kernel
def bit_tricks(
    x: Vector[T.u32], ones: Vector[T.u32], flipped: Vector[T.u32], zeros: Vector[T.u32], low: Vector[T.u32]
) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        ones[i] = popcount(x[i])
        flipped[i] = reverse_bits(x[i])
        zeros[i] = leading_zeros(x[i])
        low[i] = first_set(x[i])


@kernel
def math_helpers(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        v = x[i]
        out[i] = fma(v, 2.0, 1.0) + cbrt(v) + log10(v) + trunc(v)


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(9)

    x = device.empty(T.u32, 203)
    x.array[:] = np.concatenate([rng.integers(0, 1 << 32, 200, dtype=np.uint32), [0, 1, 1 << 31]])
    outputs = [device.empty(T.u32, 203) for _ in range(4)]
    bit_tricks(x, *outputs, launch=Items(203))
    await device.synchronize()
    values = [int(v) for v in x.array]
    oracle = [
        [v.bit_count() for v in values],
        [int(f"{v:032b}"[::-1], 2) for v in values],
        [32 - v.bit_length() for v in values],
        [(v & -v).bit_length() for v in values],
    ]
    for name, got, want in zip(
        ("popcount", "reverse_bits", "leading_zeros", "first_set"), outputs, oracle, strict=True
    ):
        log.info("%-14s equals Python: %s", name, got.array.tolist() == want)

    y = device.empty(T.f32, 400)
    y.array[:] = rng.uniform(0.1, 50, 400)
    z = device.empty(T.f32, 400)
    math_helpers(y, z, launch=Items(400))
    await device.synchronize()
    f = y.array.astype(np.float64)
    want = f * 2 + 1 + np.cbrt(f) + np.log10(f) + np.trunc(f)
    log.info("fma + cbrt + log10 + trunc match numpy: %s", bool(np.allclose(z.array, want, rtol=1e-5)))

    log.info("the 16-bit functions of numba-cuda-mlir, as Python:")
    for name in ("fp16.hadd", "fp16.hsin", "fp16.hrint", "fp16.hmax", "fp16.heq", "selp"):
        log.info("  %-11s -> %s", name, catalog.OPERATORS[name])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())

"""Run with an isolated installed package and KRN_TARGET=avx2, KRN_WORKERS=1.

This checks scalar C execution on the host, not x86 instruction support.
"""

import asyncio
from importlib.metadata import version

import numpy as np

import krn
from krn import T, Vector, host, items, kernel


@kernel
def add_two(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    for i in items(x.shape[0]):
        out[i] = x[i] + 2


async def main() -> None:
    assert version("krn") == "0.0.1"
    assert "site-packages" in krn.__file__
    device = host.Device.current()
    for count in (1, 17):
        source = np.arange(count, dtype=np.float32)
        x = device.to_device(source)
        out = device.empty(T.f32, count)
        device.launcher(add_two)(x, out)
        await device.synchronize()
        np.testing.assert_array_equal(out.to_host(), source + 2)


asyncio.run(main())

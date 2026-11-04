"""Keeping the Python in the loop out of the way: the slow, fast and bound launch paths

The host side of a GPU program is Python and its cost is the per-launch overhead, so three paths stay apart. The
slow path (first call for a specialization) captures, type checks, compiles and caches by content hash. The fast
path, `kernel(..., launch=geometry)`, looks up the specialization, marshals the arguments and makes one driver call.
The bound path, `launcher.bind(...)` once and then `bound()` or `bound.rebind(...)`, only moves pointers. Measured
on the CPU backend (saxpy, n = 64): an eager call costs 5 to 9 microseconds of Python, a `Bound` call 0.2 to 0.5 and
a graph replay 0.1 to 0.15 per launch.

Level: 5
Topics: launch overhead, bind, rebind, specialization cache
Needs: 05_systems/07_cpu_backend
"""

import asyncio
import importlib.util
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from krn import Items, T, Vector, grid, host, kernel
from krn.compiler.api import CPUDevice

try:
    json = sys.modules["json"]
except KeyError:
    spec = importlib.util.find_spec("json")
    assert spec is not None and spec.loader is not None
    spec.loader = importlib.util.LazyLoader(spec.loader)
    json = importlib.util.module_from_spec(spec)
    sys.modules["json"] = json
    spec.loader.exec_module(json)

log = logging.getLogger("serve")


@kernel
def saxpy(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + y[i]


@dataclass(slots=True, frozen=True)
class Batch:
    x: Vector[T.f32]
    y: Vector[T.f32]


def serve(batches: Iterable[Batch], a: T.f32, size: int) -> None:
    """The loop: one `bind` before it, one pointer rewrite and one driver call per iteration."""
    first, *rest = batches
    bound = (
        host.Device.current().launcher(saxpy, Items(size)).bind(a, first.x, first.y)
    )  # captures, compiles and encodes the arguments
    bound()
    for batch in rest:
        bound.rebind(a, batch.x, batch.y)
        bound()
    if log.isEnabledFor(logging.DEBUG):
        log.debug(json.dumps({"batches": 1 + len(rest), "size": size}))


async def main() -> None:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    rng = np.random.default_rng(0)
    size = 64
    xs, ys = [device.empty(T.f32, size) for _ in range(5)], [device.empty(T.f32, size) for _ in range(5)]
    for x, y in zip(xs, ys, strict=True):
        x.array[:], y.array[:] = rng.standard_normal(size), rng.standard_normal(size)
    before = [y.array.copy() for y in ys]
    serve([Batch(x, y) for x, y in zip(xs, ys, strict=True)], T.f32(0.5), size)
    await device.synchronize()
    for x, y, y0 in zip(xs, ys, before, strict=True):
        np.testing.assert_allclose(y.array, 0.5 * x.array + y0, rtol=1e-6)
    log.info("one bind and %d rebinds served %d batches", len(xs) - 1, len(xs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())

"""Time one million u32 keys at load 0.5 on the C backend; emit kl-bench/1 JSON."""

import asyncio
import json
import logging
import platform
import time
from datetime import date
from pathlib import Path

import cyclopts
import numpy as np

from krn import host
from krn.collections import Set
from krn.compiler.api import CPUDevice
from krn.frontend.types import u32

from .results_schema import BenchFile, BenchRow

LOG = logging.getLogger(__name__)
APP = cyclopts.App(help=__doc__)


async def measure() -> list[BenchRow]:
    device = host.Device.current()
    if not isinstance(device, CPUDevice):
        raise RuntimeError("collections_bench requires KRN_TARGET=avx512 (the C backend)")
    keys = device.empty(u32, 1_000_000)
    keys.array[:] = np.arange(1_000_000, dtype=np.uint32)
    warm = Set(1)
    warm.add(device.to_device(np.array([0], np.uint32))).check()
    warm.contains(device.to_device(np.array([0], np.uint32)))
    await device.synchronize()
    table = Set(1_000_000, load_factor=0.5)
    start = time.perf_counter()
    table.add(keys).check()
    inserted = time.perf_counter() - start
    start = time.perf_counter()
    table.contains(keys)
    await device.synchronize()
    found = time.perf_counter() - start
    return [
        BenchRow(group="1e6 u32, load 0.5", series=operation, value=seconds, entries=1_000_000)
        for operation, seconds in (("insert", inserted), ("find", found))
    ]


@APP.default
def main(output: Path) -> None:
    """Write the measurement to an explicitly chosen evidence path."""
    record: BenchFile = {
        "schema": "kl-bench/1",
        "name": "collections_c",
        "title": "Fixed Set bulk insert and find on C",
        "host": platform.node(),
        "date": date.today().isoformat(),
        "unit": "seconds",
        "lower_is_better": True,
        "kind": "bar",
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "rows": asyncio.run(measure()),
        "note": "Warm kernels; insert includes allocation, initialization, and count readback. Slots round to 2^21.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    LOG.info("Wrote %s: %s", output, record["rows"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    APP()

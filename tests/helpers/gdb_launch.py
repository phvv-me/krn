"""The program the cuda-gdb test debugs: launch one kernel of `tests/kernels/intrinsics.py` on the first card.

Usage: gdb_launch.py FILE KERNEL. The histogram kernel takes 64 values and 8 bins.
"""

import logging
import sys
from pathlib import Path

import numpy as np

from krn import Items
from krn.compiler.api import capture
from krn.hw import system
from krn.runtime.cuda_backend import BACKEND

target = next((t for t in system.System.detect().targets if t.vendor == "nvidia"), None)
assert target is not None
card = BACKEND.device(target)
kernel = capture(Path(sys.argv[1]), sys.argv[2], target=card.target)
values, bins = np.arange(64, dtype=np.int32) % 8, np.zeros(8, dtype=np.int32)
device_values, device_bins = card.to_device(values), card.to_device(bins)
program = card.program(kernel)
program.call(program.geometry(Items(64)), [device_values, device_bins]).enqueue(card.default_stream)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger(__name__).info("bins %s", device_bins.to_host().tolist())

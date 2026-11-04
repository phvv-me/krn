"""Run example subprocesses with the same optional native-probe seam as the parent suite."""

import os
import runpy
import sys

from krn import configure
from krn.hw.devices import intel

from . import mirror


def main() -> None:
    configure(**{"lib.mode": "ours", "lib.tune": "off"})
    if os.environ.get("KRN_HARDWARE") != "1":
        intel.openvino = None
    if os.environ.get("KRN_MIRROR"):
        mirror.main()
    else:
        runpy.run_path(sys.argv[1], run_name="__main__")


if __name__ == "__main__":
    main()

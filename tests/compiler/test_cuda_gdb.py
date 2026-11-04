"""`cuda-gdb` stops on a Python line of a kernel running on a card, the promise of `KRN_DEBUG_LINEINFO=1`.

The launch is a real process under the real debugger, so the tests skip without a card, an NVRTC library or cuda-gdb.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from krn import debug
from krn.compiler.debug import Location
from tests.conftest import KERNELS, NVRTC_LIB

FILE = KERNELS / "intrinsics.py"
LAUNCH = Path(__file__).parents[1] / "helpers" / "gdb_launch.py"
CUDA_GDB = shutil.which("cuda-gdb") or "/usr/local/cuda/bin/cuda-gdb"
ATOMIC_LINE = next(n for n, text in enumerate(FILE.read_text().splitlines(), 1) if "atomic.add(bins" in text)


@pytest.fixture(scope="module", autouse=True)
def card_and_debugger(gpu) -> None:
    if not Path(CUDA_GDB).exists():
        pytest.skip("cuda-gdb is not installed")


def debug_session(lineinfo: str) -> str:
    """The transcript of a batch cuda-gdb session that breaks at the histogram's `atomic.add` line and traces."""
    gdb = debug.CUDAGdb(
        breakpoints=(Location(FILE.name, ATOMIC_LINE),),
        commands=("bt",),
        break_on_launch=False,
        batch=True,
    )
    command = [CUDA_GDB, *gdb.argv(LAUNCH, str(FILE), "histogram")[1:]]
    env = os.environ | {
        "KRN_DEBUG_LINEINFO": lineinfo,
        "PYTHONPATH": os.pathsep.join(sys.path),
        "KRN_NVRTC_LIB": str(NVRTC_LIB),
    }
    done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=240, check=False)
    return done.stdout + done.stderr


def test_the_debugger_stops_at_the_python_line_of_a_kernel() -> None:
    transcript = debug_session("1")
    stop = next(
        (
            line
            for line in transcript.splitlines()
            if "CUDA thread hit Breakpoint 1, histogram<<<" in line and f" () at {FILE}:{ATOMIC_LINE}" in line
        ),
        None,
    )
    assert stop, transcript[-3000:]
    assert Path(stop.partition(" () at ")[2].rsplit(":", 1)[0]) == FILE

    assert (
        f"{ATOMIC_LINE}\t        atomic.add(bins, values[i], 1)" in transcript
    )  # the debugger lists the Python source
    assert any(
        line.split()[:1] == ["#0"] and "histogram<<<" in line and f" () at {FILE}:{ATOMIC_LINE}" in line
        for line in transcript.splitlines()
    )


def test_without_line_information_the_binary_does_not_know_the_python_file() -> None:
    transcript = debug_session("0")
    assert "hit Breakpoint" not in transcript
    assert "bins [8, 8, 8, 8, 8, 8, 8, 8]" in transcript  # the program ran to its end, so the kernel did run

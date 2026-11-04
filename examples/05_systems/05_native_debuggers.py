"""Command lines for the native tools: cuda-gdb, compute-sanitizer, rocgdb and the Metal validation layers

The emulator shows the logic of a kernel, but a problem only the hardware shows (a race, a misaligned access, a
wrong answer on `sm_120a` that is right in the emulator) needs the vendor's tools, and each wraps the Python
interpreter. A `Debugger` object builds that command line: `CUDAGdb` (break on launch, break at a Python line given
line information, `brkpt` from `breakpoint()`), `ComputeSanitizer` (memcheck, racecheck, synccheck, initcheck),
`RocGdb`, and `MetalValidation` (environment variables, since Apple has no command-line shader debugger).
`batch=True` makes cuda-gdb run the commands and exit, so an agent gets a transcript instead of a prompt. The
commands are printed, not run.

Level: 5
Topics: native debuggers, sanitizers, cuda-gdb, rocgdb
Needs: 05_systems/04_line_info_artifacts
"""

import logging
import shlex
import shutil
from pathlib import Path

from krn import debug

log = logging.getLogger("debugger")
SCRIPT = Path("run_my_kernel.py")


def show(title: str, tool: debug.Debugger, executable: str) -> None:
    found = shutil.which(executable)
    log.info("%s  [%s]", title, f"found at {found}" if found else f"{executable} is not installed here")
    log.info("  $ %s", shlex.join(tool.argv(SCRIPT, "--size", "4096")))
    if environment := tool.environment():
        log.info("  with the environment %s", environment)


def main() -> None:
    where = debug.Location("saxpy.py", 12)
    show(
        "stop at the first line of every kernel, then at a Python line", debug.CUDAGdb(breakpoints=(where,)), "cuda-gdb"
    )
    show(
        "the same, in batch mode for an agent",
        debug.CUDAGdb(breakpoints=(where,), commands=("print i", "continue"), batch=True),
        "cuda-gdb",
    )
    for tool in debug.Sanitizer:
        show(f"compute-sanitizer {tool.value}", debug.ComputeSanitizer(tool), "compute-sanitizer")
    show("AMD", debug.RocGdb(breakpoints=(where,), batch=True), "rocgdb")
    show("Apple", debug.MetalValidation(), "xcrun")
    log.info("build the program with KRN_DEBUG_LINEINFO=1 first, so every tool reports lines of your Python file")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

"""Line information and saved artifacts: generated code that points back at the Python line that wrote it

Debuggers, profilers and sanitizers report lines, and for a kernel written in Python the line should be the Python
one. `lineinfo=True` (or `KRN_DEBUG_LINEINFO=1`) puts a `#line N "file.py"` directive before every statement of the
generated source and passes the line-table option to the compiler, so `cuda-gdb`, Nsight Compute and
`compute-sanitizer` show your file. `debug=True` adds `--device-debug`; both change the binary, so both are part of
the cache key. `inspect_kernel` returns what a compilation made (IR and source, and PTX and cubin when given NVRTC)
as attributes, and saves the artifacts named in `Debug.keep` to `dump_dir`.

Level: 5
Topics: line info, artifacts, compile flags, inspection
Needs: 01_first_kernels/01_saxpy
"""

import logging
import tempfile
from pathlib import Path

from krn import Debug, T, Vector, debug, grid, host, kernel, targets
from krn.compiler.api import capture, emit, inspect_kernel
from krn.errors import TargetError

log = logging.getLogger("lineinfo")


@kernel
def scale_and_shift(a: T.f32, x: Vector[T.f32], y: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y[i] = a * x[i] + 1.0


def main() -> None:
    target = targets.Target.of("sm_90a")
    ir = capture(__file__, "scale_and_shift", target=target)

    log.info("1. the generated source, with and without line information")
    for lineinfo in (False, True):
        source = emit(ir, target, lineinfo=lineinfo).source
        found = [line for line in source.splitlines() if line.startswith("#line")]
        log.info("   lineinfo=%-5s %d directives%s", lineinfo, len(found), f", e.g. {found[0]}" if found else "")

    log.info("2. what each backend's compiler is asked for")
    for arch in ("sm_90a", "gfx942", "apple_m5", "avx512"):
        chosen = targets.Target.of(arch)
        for lineinfo, device_debug in ((True, False), (True, True)):
            try:
                flags = host.CompileOptions.resolve(chosen, {}, lineinfo=lineinfo, debug=device_debug).flags()
            except TargetError as error:
                log.info("   %s", error)
                continue
            log.info("   %-8s lineinfo=%d debug=%d  %s", arch, lineinfo, device_debug, " ".join(flags[:8]))

    log.info("3. the settings the environment would give")
    log.info("   %s", Debug.from_environ({"KRN_DEBUG": "1", "KRN_DEBUG_KEEP": "source,ir"}))

    log.info("4. the IR and the source, kept in a folder and returned as attributes")
    with tempfile.TemporaryDirectory(prefix="krn-keep-") as folder:
        settings = Debug(
            lineinfo=True, keep=frozenset({debug.Artifact.IR, debug.Artifact.SOURCE}), dump_dir=Path(folder)
        )
        inspection = inspect_kernel(ir, target, debug=settings)
        log.info(
            "   saved: %s", ", ".join(f"{p.name} ({p.stat().st_size} bytes)" for p in sorted(Path(folder).rglob("*")))
        )
        log.info("   the source names this file: %s", Path(__file__).name in str(inspection.source))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

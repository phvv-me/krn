"""Does a prebuilt device library plus a tiny generated kernel beat recompiling everything? NVRTC and nvJitLink.

Monolithic: one unit holding the kernel and the K functions it uses. Split: the library of N functions is built once
(`--relocatable-device-code`), then each kernel compiles against prototypes and links against it. The library is built
as a relocatable cubin and as LTO-IR. Run: python split_link.py <nvidia-cu13 dir> <out.json>
"""

import json
import logging
import statistics
import sys
from pathlib import Path

from experiments.paths import RESULTS
from experiments.units import caller, library, synthetic
from krn.runtime.nvrtc import Nvidia, NvJitLinkInput

log = logging.getLogger("split_link")

ARCH = "sm_90a"
USED = 5
LIBRARY_SIZES = (14, 70, 320)
REPS = 5


def median_ms(samples: list[float]) -> float:
    return 1000 * statistics.median(samples)


def main(lib_dir: Path, out: Path) -> None:
    nv = Nvidia(lib_dir)
    rdc = ("--relocatable-device-code=true",)
    rows: list[dict[str, float | int | str]] = []
    mono_source, mono_lines = synthetic(USED, dialect="cuda")
    for _ in range(2):
        nv.compile(mono_source, ARCH, "cubin")
    mono = [nv.compile(mono_source, ARCH, "cubin").seconds for _ in range(REPS)]
    log.info("monolithic kernel with %d functions inlined (%d lines): %.1f ms", USED, mono_lines, median_ms(mono))
    rows.append(
        {"variant": "monolithic", "library_functions": USED, "lines": mono_lines, "per_kernel_ms": median_ms(mono)}
    )
    kernel_src = caller(USED)
    for size in LIBRARY_SIZES:
        lib_src = library(size)
        lines = lib_src.count("\n") + 1
        lib_cubin = nv.compile(lib_src, ARCH, "cubin", rdc)
        lib_ltoir = nv.compile(lib_src, ARCH, "ltoir", rdc)
        log.info(
            "library of %d functions (%d lines) built once: cubin %.0f ms, LTO-IR %.0f ms",
            size,
            lines,
            1000 * lib_cubin.seconds,
            1000 * lib_ltoir.seconds,
        )
        for variant, kernel_kind, lib_input, lto in (
            ("split_cubin", "cubin", (NvJitLinkInput.CUBIN, lib_cubin.data), False),
            ("split_ltoir", "ltoir", (NvJitLinkInput.LTOIR, lib_ltoir.data), True),
        ):
            samples = []
            for _ in range(REPS + 1):
                kernel = nv.compile(kernel_src, ARCH, kernel_kind, rdc)
                kind = NvJitLinkInput.CUBIN if kernel_kind == "cubin" else NvJitLinkInput.LTOIR
                linked = nv.link(ARCH, [(kind, kernel.data), lib_input], lto=lto)
                samples.append((kernel.seconds, linked.seconds))
            samples = samples[1:]
            row = {
                "variant": variant,
                "library_functions": size,
                "lines": lines,
                "compile_kernel_ms": median_ms([s[0] for s in samples]),
                "link_ms": median_ms([s[1] for s in samples]),
                "per_kernel_ms": median_ms([s[0] + s[1] for s in samples]),
            }
            rows.append(row)
            log.info(
                "  %-12s kernel %6.1f ms + link %7.1f ms = %7.1f ms per kernel",
                variant,
                row["compile_kernel_ms"],
                row["link_ms"],
                row["per_kernel_ms"],
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arch": ARCH, "used": USED, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(Path(sys.argv[1]), (Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "split_link.json"))

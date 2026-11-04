"""Render one kernel for every vendor, so the website shows what the compiler really emits.

Run `mb run python -m experiments.site_code` from the repository root; it writes `../krn.dev/data/code.json`.
"""

import json
import logging
from pathlib import Path

from krn.compiler.api import capture, emit
from krn.compiler.dtypes import F32
from krn.facts.targets import Target

from .site_data import DATA

SOURCE = Path(__file__).resolve().parents[1] / "examples" / "01_first_kernels" / "01_saxpy.py"
GEMM_SOURCE = Path(__file__).resolve().parents[1] / "examples" / "05_systems" / "07_cpu_backend.py"
KERNEL = "saxpy"
TABS = (("CUDA", "sm_120a"), ("Metal", "apple_m4"), ("HIP", "gfx942"), ("C", "avx512"))
LOG = logging.getLogger(__name__)


def python_source() -> str:
    """The decorated function exactly as it is written in the example file."""
    lines = SOURCE.read_text().splitlines()
    start = lines.index("@kernel")
    end = next(i for i in range(start + 2, len(lines)) if not lines[i].strip())
    return "\n".join(lines[start:end])


def gemm_source() -> str:
    """The tiling record and the GEMM kernel of the CPU-backend example, up to the SAXPY that follows them."""
    lines = GEMM_SOURCE.read_text().splitlines()
    start = lines.index("@struct")
    stop = next(i for i, line in enumerate(lines) if line.startswith("def saxpy")) - 1
    return "\n".join(lines[start:stop]).rstrip()


def main() -> None:
    renders = []
    for label, arch in TABS:
        target = Target.of(arch)
        kernel = capture(SOURCE, KERNEL, dtypes={"a": F32, "x": F32, "y": F32}, target=target)
        renders.append({"label": label, "arch": arch, "source": emit(kernel, target, lineinfo=False).source.strip()})
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "code.json").write_text(
        json.dumps({"python": python_source(), "gemm": gemm_source(), "renders": renders}, indent=2) + "\n"
    )
    LOG.info("wrote %s", DATA / "code.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

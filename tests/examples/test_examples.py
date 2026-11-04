"""Every example runs, with `KRN_LIB=ours` (no vendor library), on the CPU backend, on the Mac's GPU and on a card.

The ids are `<backend>-<example>`: `c`, `metal` or `cuda`. A GPU run is the script itself on the host device of
`mirror.py`, whose launches go to the GPU while the tensors stay numpy arrays, so the example's own assertions check the
GPU's results. An example that launches no kernel there (a file that only defines kernels, one that runs them in the
emulator, one that only prints host facts) has nothing to run on the GPU and is skipped, never counted as a pass.

Attaching to another process needs the OS's permission to inspect it, which macOS grants to root or to an entitled
process only. `experiments/conformance.py examples` reads the set below without importing this file.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

from tests.conftest import EXAMPLE_ROOT, ROOT

pytestmark = pytest.mark.skipif(not EXAMPLE_ROOT.is_dir(), reason="local examples are absent")
NEEDS_INSPECTION_RIGHTS = {"11_attach_to_process.py"}
BACKENDS = ("c", "metal", "cuda")
RUNNABLE = [
    *sorted(EXAMPLE_ROOT.glob("[0-9][0-9]_*/[0-9][0-9]_*.py")),
    *sorted((ROOT / "experiments" / "library_checks").glob("*.py")),
]
CASES = [(backend, example) for backend in BACKENDS for example in RUNNABLE]
GPU = {
    "metal": "metal_gpu",
    "cuda": "gpu",
}  # the fixture that provides each GPU, or skips where there is none


class Needs(NamedTuple):
    """A GPU run that fails for a reason of the example's design, not of the language."""

    backends: tuple[str, ...]
    example: str
    failure: str  # a pattern of the output of the failing run; a failure that does not match it still fails the test
    reason: str


NEEDS = (
    Needs(
        ("metal",),
        "04_python_in_kernels.py",
        "assert is not available on apple",
        "needs a device assert: Metal Shading Language has no device printf for it to report through",
    ),
    Needs(
        ("metal",),
        "04_python_in_kernels.py",
        "f64 is not lowered for target apple",
        "needs f64: Metal has no double precision and the example ends with a width check on f64",
    ),
    Needs(
        ("cuda",),
        "04_python_in_kernels.py",
        r"assert here\.array\.tolist\(\)",
        "needs a CPU target: `lanes` picks its arm by target and the example asserts the C backend took arm 1",
    ),
    Needs(
        ("metal", "cuda"),
        "11_attach_to_process.py",
        r"assert child\.stdout\.readline\(\)",
        "needs a CPU process: the example attaches a debugger to a child it starts on the CPU backend",
    ),
    Needs(
        ("metal", "cuda"),
        "columns.py",
        r"assert wide == np\.sum\(values",
        "needs a sequential f16 sum: the example expects the C backend's order (2048), a GPU adds in parallel (2112)",
    ),
)


def needed(backend: str, example: Path, output: str) -> str | None:
    """The reason in `NEEDS` for a failing run of `example` on `backend`, when its failure is the one listed."""
    return next(
        (
            row.reason
            for row in NEEDS
            if backend in row.backends and example.name == row.example and row.failure.replace("\\", "") in output
        ),
        None,
    )


def unlaunched(example: Path) -> str:
    """Why a run that made no launch has nothing to run on a GPU, from what the file is."""
    source = example.read_text()
    if not any(line.startswith("if __name__ == ") for line in source.splitlines()):
        return "needs a main: the file only defines kernels and launches none"
    if "emulate(" in source:
        return "needs a device launch: the example runs its kernels in the emulator and launches none"
    return (
        "needs a launch: the example only captures, renders or prints (facts, tables, diagnostics) and runs no kernel"
    )


@pytest.mark.parametrize(("backend", "example"), CASES, ids=[f"{b}-{e.stem}" for b, e in CASES])
def test_the_example_runs_without_a_vendor_library(request: pytest.FixtureRequest, backend: str, example: Path) -> None:
    if example.name in NEEDS_INSPECTION_RIGHTS and sys.platform == "darwin" and os.geteuid() != 0:
        pytest.skip("macOS lets only root or an entitled process attach to another process")
    environment = os.environ | {
        "KRN_TARGET": "avx512",
        "PYTHONPATH": os.pathsep.join([*sys.path, str(Path(__file__).parents[1] / "helpers")]),
    }
    command = [sys.executable, "-m", "tests.helpers.example_launch", str(example)]
    if backend != "c":
        arch = request.getfixturevalue(GPU[backend]).target.arch
        environment |= {"KRN_TARGET": arch, "KRN_MIRROR": backend}
    done = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=180, check=False)
    output = done.stdout + done.stderr
    if done.returncode and (reason := needed(backend, example, output)):
        pytest.skip(reason)
    assert done.returncode == 0, output[-2500:]
    if backend != "c" and not int(
        next(
            line.split()[1]
            for line in reversed(done.stdout.splitlines())
            if line.startswith("mirror: ") and line.split()[1].isdigit()
        )
    ):
        pytest.skip(unlaunched(example))

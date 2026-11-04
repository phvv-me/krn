"""The debugging tools around the kernel: artifacts and line information, native debugger commands, the settings, and
attaching to a live process."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from krn import configure, debug, targets
from krn.compiler.api import capture
from krn.compiler.artifacts import format_ir, inspect_kernel
from krn.compiler.debug import Artifact, Debug, Location
from krn.runtime.nvrtc import Nvidia as Nvrtc
from tests.conftest import KERNELS

FILE = KERNELS / "intrinsics.py"
SM90 = targets.Target.of("sm_90a")


def test_settings_read_the_environment_once_and_debug_implies_lineinfo() -> None:
    quiet = Debug.from_environ({})
    assert (quiet.debug, quiet.lineinfo, quiet.keep, quiet.emulate) == (
        False,
        False,
        frozenset(),
        False,
    )
    loud = Debug.from_environ(
        {
            "KRN_DEBUG": "1",
            "KRN_DEBUG_KEEP": "source,ptx",
            "KRN_DEBUG_DUMP_DIR": "/tmp/x",
            "KRN_LOG_LEVEL": "info",
        }
    )
    assert loud.debug and loud.lineinfo
    assert loud.keep == {Artifact.SOURCE, Artifact.PTX}
    assert loud.dump_dir == Path("/tmp/x")
    assert not hasattr(loud, "log_level")
    assert Debug.from_environ({"KRN_DEBUG_KEEP": "all"}).keep == set(Artifact)
    with pytest.raises(ValueError, match="'sass2'"):
        Debug.from_environ({"KRN_DEBUG_KEEP": "sass2"})


def test_inspection_hands_artifacts_over_as_attributes_and_saves_them(tmp_path: Path, nvrtc: Nvrtc) -> None:
    kernel = capture(FILE, "histogram", target=SM90)
    inspection = inspect_kernel(
        kernel,
        SM90,
        nvrtc=nvrtc,
        debug=Debug(
            lineinfo=True,
            keep=frozenset({Artifact.SOURCE, Artifact.PTX, Artifact.CUBIN}),
            dump_dir=tmp_path,
        ),
    )
    assert "#line" in str(inspection.source)
    assert f"{FILE.name}" in str(inspection.ptx)
    assert inspection.cubin[:4] == b"\x7fELF"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "histogram.cubin",
        "histogram.ptx",
        "histogram.src",
    ]
    with pytest.raises(AttributeError, match="has no 'sass'"):
        _ = inspection.sass


def test_the_ir_text_gives_the_source_line_of_every_statement() -> None:
    text = format_ir(capture(FILE, "histogram", target=SM90).body)
    assert "Assign" in text and "Effect" in text
    assert f"(line {FILE.read_text().splitlines().index('        atomic.add(bins, values[i], 1)') + 1})" in text


def test_native_debuggers_build_command_lines_an_agent_can_run_in_batch_mode() -> None:
    where = Location("saxpy.py", 12)
    gdb = debug.CUDAGdb(breakpoints=(where,), commands=("print i", "continue"), batch=True)
    assert gdb.argv(Path("run.py"), "--n", "4") == (
        "cuda-gdb", "-batch", "-ex", "set cuda break_on_launch application", "-ex", "set breakpoint pending on",
        "-ex", "break saxpy.py:12", "-ex", "run", "-ex", "print i", "-ex", "continue",
        "--args", sys.executable, "run.py", "--n", "4",
    )  # fmt: skip
    assert debug.ComputeSanitizer(debug.Sanitizer.RACECHECK).argv(Path("run.py"))[:3] == (
        "compute-sanitizer",
        "--tool",
        "racecheck",
    )
    assert debug.MetalValidation().environment()["MTL_SHADER_VALIDATION"] == "1"
    assert debug.RocGdb().argv(Path("run.py"))[0] == "rocgdb"


def test_pdb_with_scripted_commands_shows_the_kernel_source_and_its_locals(tmp_path: Path) -> None:
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(f"""
        import numpy as np
        from krn import Items
        from krn.emulator.emulate import build
        from krn.compiler.api import capture
        from krn.facts.targets import Target
        t = Target.of("sm_90a")
        ir = capture({str(FILE)!r}, "debug_aids", target=t)
        build(ir, t).run(Items(2), [np.array([5, 6], np.int32), np.zeros(2, np.int32)])
        """)
    )
    line = next(n for n, text in enumerate(FILE.read_text().splitlines(), 1) if "breakpoint()" in text)
    argv = debug.Pdb(breakpoints=(Location(str(FILE), line),), commands=("p i", "p _thread_id", "ll")).argv(script)
    env = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)}  # PYTHONBREAKPOINT=0: only the line breakpoint stops
    done = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60, check=False)
    assert "\n0\n(0, 0, 0)\n" in done.stdout  # `p i` and `p _thread_id` at the first thread
    assert f"{line} B->\t        breakpoint()" in done.stdout  # `ll` lists the kernel's own source at the stop


def test_record_writes_the_locals_of_the_calling_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(**{"debug.dump_dir": tmp_path})
    answer = 42  # noqa: F841  (read through the frame's locals)
    debug.record()
    written = json.loads((tmp_path / "breakpoint-1.json").read_text())
    assert written["function"] == "test_record_writes_the_locals_of_the_calling_frame"
    assert written["locals"]["answer"] == "42"


CHILD = """
import sys, time
from krn import debug
from krn.compiler.api import CPUBackend


class Worker(debug.Described):
    def describe(self):
        return {"type": "Worker", "step": 7}


keep = debug.watch(Worker())
debug.install()
print("ready", flush=True)
for _ in range(600):
    time.sleep(0.05)
"""


def spawn(tmp_path: Path) -> subprocess.Popen[str]:
    env = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)}
    child = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE, text=True, env=env, cwd=tmp_path)
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "ready"
    return child


@pytest.mark.skipif(
    sys.platform == "darwin" and os.geteuid() != 0,
    reason="macOS lets only root or an entitled process inject code",
)
def test_attach_fetches_a_snapshot_of_a_live_process(tmp_path: Path) -> None:
    child = spawn(tmp_path)
    try:
        found = debug.attach(child.pid, timeout=20)
    finally:
        child.kill()
    assert found.pid == child.pid
    assert found.free_threaded == (not sys._is_gil_enabled())
    assert {"type": "Worker", "step": 7} in found.objects
    assert any("sleep" in line for stacks in found.threads.values() for line in stacks)


def test_sigusr1_dumps_every_threads_stack_from_the_c_level(tmp_path: Path) -> None:
    child = spawn(tmp_path)
    try:
        os.kill(child.pid, signal.SIGUSR1)
        time.sleep(0.3)
    finally:
        child.kill()
        _, _ = child.communicate()

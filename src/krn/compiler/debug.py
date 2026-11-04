"""Looking inside a kernel and a running process, for people and for agents.

Four layers, each usable without the others.

1. The kernel body. `breakpoint()`, `print(...)` and `assert` are the Python statements, and they mean what they say
   on the device: a stop, a `printf` that names the thread, a failure that names the thread and the line. Where the
   code runs decides the tool. In the emulator (`prototype/krn.compiler.api/emulate.py`) `breakpoint()` is Python's
own, so
   `pdb` shows the kernel source with its real file and line, the locals, `_block_id` and `_thread_id`. On NVIDIA it
   is a `brkpt` instruction that `cuda-gdb` stops on, on AMD and the CPU a `debugtrap`, and Apple has none.
2. The generated code. `KRN_DEBUG_LINEINFO=1` makes the generated source carry `#line` directives that name the Python
file
   and line, and passes the line-table options to the compiler, so `cuda-gdb`, Nsight and `compute-sanitizer` report
   Python lines. `KRN_DEBUG=1` adds `--device-debug` and the strictest checks. `KRN_DEBUG_KEEP=source,ptx,cubin,sass`
   saves the artifacts and `Inspection` hands them over as attributes, as CuTe DSL does with `__ptx__` and
`__sass__`.
3. The native tools. `Debugger` objects build the command line of the tool that wraps the Python interpreter:
   `CUDAGdb`, `ComputeSanitizer`, `RocGdb`, `MetalValidation`, and `Pdb` for the emulator, with scripted commands so
   an agent can drive them in batch mode and read the transcript.
4. The process. Python 3.14 added a way to attach to a running interpreter and run a script in it
   (`sys.remote_exec`, `python -m pdb -p PID`; docs.python.org/3/howto/remote_debugging.html). `attach(pid)` uses it
to
   fetch a `snapshot()` of the streams, queues and thread stacks of a live process as JSON. The script runs only
when
   the interpreter reaches a safe point, so a process blocked inside a driver call never answers: `install()`
   registers a signal that dumps every thread's stack from the C level, which works there too.

The environment variables, all optional, are read once into `Debug`:

    KRN_DEBUG        1: device debug info, no optimization, the strict checks
    KRN_DEBUG_LINEINFO     1: Python file and line in the generated code and in the binary
    KRN_DEBUG_KEEP         ir,source,ptx,cubin,sass,asm or all: artifacts to save
    KRN_DEBUG_DUMP_DIR     where saved artifacts go (default: the current directory)
    KRN_DEBUG_DIAG         warnings,remarks,remarks{ptxas},1 or decisions: diagnostics and backend decisions
    KRN_DEBUG_EMULATE      1: run kernels in the Python emulator instead of the backend"""

import faulthandler
import json
import logging
import os
import signal
import sys
import tempfile
import time
import traceback
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Self

from .errors import SettingError

log = logging.getLogger("krn.debug")

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


class Artifact(StrEnum):
    """What a compilation can leave behind for inspection."""

    IR = auto()  # the typed IR of the captured kernel, as text
    SOURCE = auto()  # CUDA C++, HIP C++, MSL or C
    PTX = auto()
    CUBIN = auto()
    SASS = auto()  # the disassembly of the cubin
    ASM = auto()  # AMDGPU or x86 assembly


CALLEE_NOTE = "Capture expands every user callee at its call site; inline decisions apply to emitted helpers only."


@dataclass(frozen=True, slots=True)
class Explanation:
    """The capture roles, launch plan and backend decisions, without executing a kernel."""

    kernel: str
    target: str
    space: tuple[str, ...]
    geometry: dict[str, Json]
    decisions: tuple[tuple[str, str, str], ...]
    variant: str
    helpers: tuple[tuple[str, int, str, str], ...]
    roles: tuple[tuple[str, tuple[str, ...]], ...]
    cap: str | None = None
    callee_note: str = CALLEE_NOTE

    def to_json(self) -> dict[str, Json]:
        # Convert tuples too, so a JSON round trip preserves the returned value.
        return {
            "kernel": self.kernel,
            "target": self.target,
            "space": list(self.space),
            "geometry": self.geometry,
            "decisions": [list(row) for row in self.decisions],
            "variant": self.variant,
            "helpers": [list(row) for row in self.helpers],
            "roles": [[name, list(roles)] for name, roles in self.roles],
            "cap": self.cap,
            "callee_note": self.callee_note,
        }

    def __str__(self) -> str:
        rows = [
            ("kernel", self.kernel, self.target),
            ("space", "; ".join(self.space), ""),
            ("geometry", str(self.geometry), ""),
            *self.decisions,
            ("variant", self.variant, ""),
        ]
        rows.extend((f"helper {name} ({size})", qualifier, why) for name, size, qualifier, why in self.helpers)
        rows.extend(
            ("roles " + name, ", ".join(roles), self.callee_note if "callee" in roles else "")
            for name, roles in self.roles
        )
        if self.cap:
            rows.append(("cooperative cap", self.cap, ""))
        width = max(len(name) for name, _, _ in rows)
        return "\n".join(f"{name:<{width}}  {value}  {why}".rstrip() for name, value, why in rows)


def _flag(name: str, text: str | None) -> bool:
    if text is None:
        return False
    if text.strip().lower() not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise SettingError(name, text, "a boolean (0|1|false|true|no|yes|off|on)")
    return text.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, kw_only=True)
class Debug:
    debug: bool = field(default=False, metadata={"backends": ("cuda", "amd", "metal", "cpu")})
    lineinfo: bool = field(default=False, metadata={"backends": ("cuda", "amd", "metal", "cpu")})
    keep: frozenset[Artifact] = field(default=frozenset(), metadata={"backends": ("cuda", "amd", "metal", "cpu")})
    dump_dir: Path = field(default=Path(), metadata={"backends": ("cuda", "amd", "metal", "cpu")})
    diagnostics: frozenset[str] = field(default=frozenset(), metadata={"backends": ("cuda", "amd", "metal", "cpu")})
    decisions: bool = field(default=False, metadata={"backends": ("cuda", "amd", "metal", "cpu")})
    emulate: bool = field(default=False, metadata={"backends": ("cuda", "amd", "metal", "cpu")})

    def __post_init__(self) -> None:
        for name in ("debug", "lineinfo", "decisions", "emulate"):
            if not isinstance(getattr(self, name), bool):
                raise SettingError(f"debug.{name}", str(getattr(self, name)), "a boolean")
        try:
            object.__setattr__(self, "keep", frozenset(Artifact(value) for value in self.keep))
        except ValueError:
            raise SettingError("debug.keep", str(self.keep), "known Artifact values") from None
        diagnostics = frozenset(self.diagnostics)
        if diagnostics - {"warnings", "remarks", "remarks{ptxas}", "1", "decisions"}:
            raise SettingError("debug.diagnostics", str(diagnostics), "warnings,remarks,remarks{ptxas},1 or decisions")
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "dump_dir", Path(self.dump_dir))
        object.__setattr__(self, "lineinfo", self.lineinfo or self.debug)
        object.__setattr__(self, "decisions", self.decisions or bool({"1", "decisions"} & diagnostics))

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> Self:
        env = os.environ if environ is None else environ
        keep = env.get("KRN_DEBUG_KEEP", "")
        try:
            artifacts = (
                frozenset(Artifact) if keep == "all" else frozenset(Artifact(name) for name in keep.split(",") if name)
            )
        except ValueError:
            raise SettingError("KRN_DEBUG_KEEP", keep, "ir,source,ptx,cubin,sass,asm or all") from None
        diagnostics = frozenset(part for part in env.get("KRN_DEBUG_DIAG", "").split(",") if part)
        if diagnostics - {"warnings", "remarks", "remarks{ptxas}", "1", "decisions"}:
            raise SettingError(
                "KRN_DEBUG_DIAG", env["KRN_DEBUG_DIAG"], "warnings,remarks,remarks{ptxas},1 or decisions"
            )
        dump = env.get("KRN_DEBUG_DUMP_DIR", ".")
        if not dump or "\0" in dump:
            raise SettingError("KRN_DEBUG_DUMP_DIR", dump, "a nonempty filesystem path")
        return cls(
            debug=_flag("KRN_DEBUG", env.get("KRN_DEBUG")),
            lineinfo=_flag("KRN_DEBUG_LINEINFO", env.get("KRN_DEBUG_LINEINFO"))
            or _flag("KRN_DEBUG", env.get("KRN_DEBUG")),
            keep=artifacts,
            dump_dir=Path(dump),
            diagnostics=diagnostics,
            decisions=bool({"1", "decisions"} & diagnostics),
            emulate=_flag("KRN_DEBUG_EMULATE", env.get("KRN_DEBUG_EMULATE")),
        )


@dataclass(frozen=True, slots=True)
class Inspection:
    """The artifacts of one compiled kernel. A missing one is absent, not empty: `KRN_DEBUG_KEEP` says what to build."""

    kernel: str
    artifacts: Mapping[Artifact, str | bytes]

    def __getattr__(self, name: str) -> str | bytes:
        try:
            return self.artifacts[Artifact(name)]
        except KeyError, ValueError:
            raise AttributeError(f"{self.kernel} has no {name!r}; the artifacts are {sorted(self.artifacts)}") from None

    def save(self, folder: Path) -> list[Path]:
        """Write every artifact to `folder/<kernel>.<extension>` and return the paths."""
        suffix = {Artifact.IR: "ir", Artifact.SOURCE: "src", Artifact.ASM: "s"}
        folder.mkdir(parents=True, exist_ok=True)
        paths = [folder / f"{self.kernel}.{suffix.get(kind, kind.value)}" for kind in self.artifacts]
        for path, data in zip(paths, self.artifacts.values(), strict=True):
            path.write_bytes(data) if isinstance(data, bytes) else path.write_text(data)
        return paths


class Described:
    """An object a snapshot can describe. Streams, devices and queues override `describe`."""

    def describe(self) -> dict[str, Json]:
        return {"type": type(self).__name__}


_watched: weakref.WeakSet[Described] = weakref.WeakSet()


def watch[D: Described](obj: D) -> D:
    """Make `obj` appear in snapshots for as long as it lives. Runtimes call this in their constructors."""
    _watched.add(obj)
    return obj


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The state of a process that explains a hang or a wrong answer, and the JSON it travels as."""

    time: float
    pid: int
    python: str
    free_threaded: bool
    argv: list[str]
    environment: dict[str, str]  # the KRN_* variables
    objects: list[dict[str, Json]]  # what the watched objects describe: streams, devices, queues
    threads: dict[str, list[str]]  # the stack of every thread, innermost frame last

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls(**json.loads(text))


def snapshot() -> Snapshot:
    """The state of this process: interpreter, watched objects, thread stacks."""
    return Snapshot(
        time=time.time(),
        pid=os.getpid(),
        python=sys.version.split()[0],
        free_threaded=not sys._is_gil_enabled(),
        argv=[*sys.argv],
        environment={key: value for key, value in os.environ.items() if key.startswith("KRN_")},
        objects=[obj.describe() for obj in list(_watched)],
        threads={str(ident): [*traceback.format_stack(frame)] for ident, frame in sys._current_frames().items()},
    )


_SCRIPT = """
import os, sys, traceback
try:
    from krn.compiler.debug import Snapshot, snapshot
    text = snapshot().to_json()
except ImportError as error:
    import json, time
    text = json.dumps({{
        "time": time.time(), "pid": os.getpid(), "python": sys.version.split()[0],
        "free_threaded": not sys._is_gil_enabled(), "argv": list(sys.argv), "environment": {{}},
        "objects": [{{"type": "krn is not importable in the target", "error": str(error)}}],
        "threads": {{str(i): traceback.format_stack(f) for i, f in sys._current_frames().items()}},
    }})
with open({part!r}, "w") as out:
    out.write(text)
os.replace({part!r}, {result!r})
"""


def attach(pid: int, *, timeout: float = 10.0) -> Snapshot:
    """Run `snapshot()` inside the Python process `pid` and return what it found.

    Needs the OS permission that a native debugger needs (ptrace), an interpreter of the same version and the same
    build kind (free-threaded or not) as this one, and a target that reaches a safe evaluation point in `timeout`
    seconds; a process stuck in a native call does not, and the `TimeoutError` says to use `install()`'s signal.
    """
    with tempfile.TemporaryDirectory(prefix="krn-attach-") as folder:
        script, result = Path(folder, "snapshot.py"), Path(folder, "snapshot.json")
        script.write_text(_SCRIPT.format(part=str(result) + ".part", result=str(result)))
        sys.remote_exec(pid, str(script))
        deadline = time.monotonic() + timeout
        while not result.exists():  # the script file must outlive the injection, so the wait is inside the `with`
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"process {pid} did not reach a safe point in {timeout}s; it may be blocked in a native call: "
                    f"send SIGUSR1 to a process that called debug.install() to dump its thread stacks"
                )
            time.sleep(0.005)
        return Snapshot.from_json(result.read_text())


def install(stacks: Path | None = None) -> None:
    """Dump every thread's stack on SIGUSR1, from the C level, so it works while the interpreter is blocked.

    Also lets `PYTHONBREAKPOINT=krn.compiler.debug.record` work: see `record`.
    """
    faulthandler.register(signal.SIGUSR1, file=stacks.open("a") if stacks else sys.stderr, all_threads=True)


_recorded = 0
_debug_settings: Callable[[], Debug] = Debug


def record[A, K](*_args: A, **_kwargs: K) -> None:
    """A breakpoint for agents: `PYTHONBREAKPOINT=krn.compiler.debug.record` makes every `breakpoint()` write the locals
    of the frame that called it to `breakpoint-<n>.json` in the dump directory and carry on, instead of waiting.
    """
    global _recorded
    frame = sys._getframe(1)
    _recorded += 1
    out = _debug_settings().dump_dir / f"breakpoint-{_recorded}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
                "locals": {name: repr(value)[:200] for name, value in frame.f_locals.items()},
            },
            indent=1,
        )
    )
    log.info("breakpoint %d recorded at %s:%d", _recorded, frame.f_code.co_filename, frame.f_lineno)


@dataclass(frozen=True, slots=True)
class Location:
    """A place in the Python source of a kernel: the file and line a breakpoint or a report names."""

    file: str
    line: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"


def pending(breakpoints: Sequence[Location]) -> list[str]:
    """The kernel's module loads after the debugger starts, so a Python-line breakpoint must be allowed to wait for it.

    Without this a batch session answers the debugger's question "make breakpoint pending?" with no and sets nothing.
    """
    return ["set breakpoint pending on"] if breakpoints else []


class Debugger(ABC):
    """A tool that wraps the interpreter. `argv` builds the command; running it is the caller's choice."""

    def environment(self) -> Mapping[str, str]:
        """Variables the tool needs set in the child process."""
        return {}

    @abstractmethod
    def argv(self, script: Path, *args: str) -> tuple[str, ...]:
        """The command line that runs `python script args` under the tool."""


@dataclass(frozen=True, slots=True)
class CUDAGdb(Debugger):
    """`cuda-gdb` on the interpreter. `batch` runs the commands and exits, so an agent can read the transcript."""

    breakpoints: Sequence[Location] = ()
    commands: Sequence[str] = ()
    break_on_launch: bool = True
    batch: bool = False

    def argv(self, script: Path, *args: str) -> tuple[str, ...]:
        prelude = ["set cuda break_on_launch application"] if self.break_on_launch else []
        steps = [
            *prelude,
            *pending(self.breakpoints),
            *(f"break {where}" for where in self.breakpoints),
            "run",
            *self.commands,
        ]
        flags = [item for step in steps for item in ("-ex", step)]
        return (
            "cuda-gdb",
            *(("-batch",) if self.batch else ()),
            *flags,
            "--args",
            sys.executable,
            str(script),
            *args,
        )


@dataclass(frozen=True, slots=True)
class RocGdb(Debugger):
    """`rocgdb`, the AMD counterpart; the same shape as `CUDAGdb` without the CUDA-only launch option."""

    breakpoints: Sequence[Location] = ()
    commands: Sequence[str] = ()
    batch: bool = False

    def argv(self, script: Path, *args: str) -> tuple[str, ...]:
        steps = [
            *pending(self.breakpoints),
            *(f"break {where}" for where in self.breakpoints),
            "run",
            *self.commands,
        ]
        flags = [item for step in steps for item in ("-ex", step)]
        return (
            "rocgdb",
            *(("-batch",) if self.batch else ()),
            *flags,
            "--args",
            sys.executable,
            str(script),
            *args,
        )


class Sanitizer(StrEnum):
    MEMCHECK = auto()  # out-of-bounds and misaligned accesses
    RACECHECK = auto()  # shared-memory data races
    SYNCCHECK = auto()  # invalid barriers and warp collectives
    INITCHECK = auto()  # reads of uninitialized global memory


@dataclass(frozen=True, slots=True)
class ComputeSanitizer(Debugger):
    tool: Sanitizer = Sanitizer.MEMCHECK

    def argv(self, script: Path, *args: str) -> tuple[str, ...]:
        return ("compute-sanitizer", "--tool", self.tool.value, sys.executable, str(script), *args)


@dataclass(frozen=True, slots=True)
class MetalValidation(Debugger):
    """Apple has no command-line debugger for shaders; the Metal layers report through the environment instead."""

    def environment(self) -> Mapping[str, str]:
        return {"MTL_SHADER_VALIDATION": "1", "MTL_DEBUG_LAYER": "1"}

    def argv(self, script: Path, *args: str) -> tuple[str, ...]:
        return (sys.executable, str(script), *args)


@dataclass(frozen=True, slots=True)
class Pdb(Debugger):
    """`python -m pdb` with scripted commands: break at the lines, continue, run the commands, quit."""

    breakpoints: Sequence[Location] = ()
    commands: Sequence[str] = ()

    def argv(self, script: Path, *args: str) -> tuple[str, ...]:
        steps = [
            *(f"break {where}" for where in self.breakpoints),
            "continue",
            *self.commands,
            "quit",
        ]
        flags = [item for step in steps for item in ("-c", step)]
        return (sys.executable, "-m", "pdb", *flags, str(script), *args)

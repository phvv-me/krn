"""Frozen settings snapshots, seeded at import and changed through krn.configure.

Only KRN_TARGET, KRN_CACHE_DIR, KRN_WORKERS, KRN_DEBUG and the KRN_DEBUG_ group seed settings.
Code arguments take precedence over scoped settings, process settings, seeds and defaults.
"""

import math
import os
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum, auto
from functools import wraps
from pathlib import Path
from threading import RLock
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Literal, Self, get_args, get_origin, get_type_hints

from krn.compiler import debug as debug_module
from krn.compiler.debug import Debug
from krn.compiler.errors import SettingError as SettingError
from krn.compiler.errors import TargetError
from krn.facts.targets import Target
from krn.frontend.options import Mode as Mode
from krn.frontend.records import Const

from .compile_options import Opt

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

BACKENDS = ("cuda", "amd", "metal", "cpu")
PRINTF_BACKENDS = ("cuda", "amd", "cpu")
SUPPORTED = {"backends": BACKENDS}


class Policy(StrEnum):
    """How a precision ladder trades decode work for bytes moved."""

    COMPUTE_BOUND = auto()
    MEMORY_BOUND = auto()


def _positive(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise SettingError(name, str(value), "a positive integer")


@dataclass(frozen=True, kw_only=True)
class Run:
    target: str | None = field(default=None, metadata=SUPPORTED)
    workers: int | None = field(
        default=None,
        metadata={
            "backends": ("cpu",),
            "reasons": dict.fromkeys(("nvidia", "amd", "apple"), "GPU launches do not use a CPU worker pool"),
            "instead": "set run.workers=None or use a CPU target",
        },
    )
    graphs: bool = field(default=True, metadata=SUPPORTED)
    drain_seconds: float = field(default=60.0, metadata=SUPPORTED)  # the longest `krn.asyncio.drain` waits by default
    variants: Mapping[str, str] = field(default_factory=dict, metadata=SUPPORTED)
    config: Mapping[str, str] = field(default_factory=dict, metadata=SUPPORTED)

    def __post_init__(self) -> None:
        _positive("run.workers", self.workers)
        if (
            isinstance(self.drain_seconds, bool)
            or not isinstance(self.drain_seconds, int | float)
            or self.drain_seconds <= 0
        ):
            raise SettingError("run.drain_seconds", str(self.drain_seconds), "a positive number of seconds")
        if self.target is not None:
            try:
                Target.of(self.target)
            except ValueError, KeyError:
                raise SettingError("run.target", self.target, "a known target architecture") from None
        if not isinstance(self.graphs, bool):
            raise SettingError("run.graphs", str(self.graphs), "a boolean")
        for key, value in self.config.items():
            kernel, dot, name = key.partition(".")
            if not kernel or not dot or not name or "." in name:
                raise SettingError("run.config", f"{key}={value}", "kernel.field=value pairs")
        object.__setattr__(self, "variants", MappingProxyType(dict(self.variants)))
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True, kw_only=True)
class Cache:
    dir: Path | None = field(
        default=None,
        metadata={
            "backends": ("cuda", "amd", "cpu"),
            "reasons": {"apple": "MTLLibrary cannot serialize a source-compiled library for persistent caching"},
            "instead": "set cache.dir=None to use Metal in-memory program caching",
        },
    )

    def __post_init__(self) -> None:
        if self.dir is not None:
            object.__setattr__(self, "dir", Path(self.dir))


@dataclass(frozen=True, kw_only=True)
class Precision:
    floor_bits: int | None = field(default=None, metadata=SUPPORTED)
    policy: Policy | None = field(default=None, metadata=SUPPORTED)
    pins: Mapping[str, str] = field(default_factory=dict, metadata=SUPPORTED)

    def __post_init__(self) -> None:
        _positive("precision.floor_bits", self.floor_bits)
        if self.policy is not None:
            object.__setattr__(self, "policy", Policy(self.policy))
        object.__setattr__(self, "pins", MappingProxyType(dict(self.pins)))


LIB_MODES = frozenset(
    (
        "auto",
        "ours",
        "vendor",
        "cublas",
        "cublaslt",
        "cusparse",
        "hipblaslt",
        "accelerate",
        "mps",
        "mkl",
        "onednn",
        "host",
    )
)


@dataclass(frozen=True, kw_only=True)
class Lib:
    mode: str = field(default="auto", metadata=SUPPORTED)
    matmul: str | None = field(default=None, metadata=SUPPORTED)
    cholesky: str | None = field(default=None, metadata=SUPPORTED)
    solve_triangular: str | None = field(default=None, metadata=SUPPORTED)
    fft: str | None = field(default=None, metadata=SUPPORTED)
    spmv: str | None = field(default=None, metadata=SUPPORTED)
    spmm: str | None = field(default=None, metadata=SUPPORTED)
    tune: str = field(default="lazy", metadata=SUPPORTED)

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            allowed = {"off", "lazy", "eager"} if item.name == "tune" else LIB_MODES
            if value is not None and value not in allowed:
                raise SettingError(f"lib.{item.name}", value, "|".join(sorted(allowed)))


@dataclass(frozen=True, kw_only=True)
class BackendCompile:
    std: str | None = field(default=None, metadata={"backends": ("cuda", "amd")})
    opt: Opt | None = field(default=None, metadata=SUPPORTED)
    march: str | None = field(default=None, metadata={"backends": ("cuda", "amd", "cpu")})

    def __post_init__(self) -> None:
        if self.opt is not None:
            object.__setattr__(self, "opt", Opt(self.opt))


@dataclass(frozen=True, kw_only=True)
class Compile:
    cuda: BackendCompile = field(default_factory=BackendCompile, metadata=SUPPORTED)
    amd: BackendCompile = field(default_factory=BackendCompile, metadata=SUPPORTED)
    metal: BackendCompile = field(default_factory=BackendCompile, metadata=SUPPORTED)
    cpu: BackendCompile = field(default_factory=BackendCompile, metadata=SUPPORTED)


@dataclass(frozen=True, kw_only=True)
class Warn:
    """Conservative warning budgets: half typical GPU resources, any spill, and whole warps.

    Registers above 128, occupancy below 25%, any spilled bytes, fewer than 64 threads,
    or more than 256 expanded instructions merit inspection. Checks require backend facts.

    `warn_software_fallback` reports a general function (`krn.math.add_sat` and its kin) that a GPU runs as a software
    sequence rather than one native instruction, and a conversion of a narrow float (fp8, fp6, fp4) that a GPU composes
    in software. A conversion's kind is spelled `software-fallback:<from>-to-<to>`, such as
    `software-fallback:f8e4m3fn-to-f32`. `software_fallback_min_instructions` is the shortest such sequence that warns:
    the default of 3 lets a one- or two-operation sequence pass silently. The CPU backend never warns.

    `hint_fusable_multiply_add` reports a multiply whose only use is an add or subtract: strict arithmetic rounds the
    product first, so `math.fma` is one fused instruction and more accurate. Half types need a native half FMA and f64
    needs float64 arithmetic; no target table records a fused f32, so f32 is hinted on every target. A hint is not a
    fault: `min_fusable_multiply_adds` is the fewest such patterns in a kernel that warn, the default 1 warns on any.

    `warn_strict_atomic_cas` reports a float32 atomic add or subtract on a GPU that runs as a compare-and-swap loop,
    because the native instruction would flush subnormals to zero.

    `warn_denormal_flush` reports a strict float32 operation that a target flushes (`denormal_flushes` in
    `facts.targets`): the device run then differs from the emulator for subnormal inputs. It warns once per kernel,
    operation and target, and the CPU and NVIDIA targets never warn.
    """

    max_registers_per_thread: int = field(default=128, metadata=SUPPORTED)
    min_occupancy: float = field(default=0.25, metadata=SUPPORTED)
    max_spilled_bytes: int = field(default=0, metadata=SUPPORTED)
    min_threads_per_block: int = field(default=64, metadata=SUPPORTED)
    max_unrolled_instructions: int = field(default=256, metadata=SUPPORTED)
    warn_on_unproven_aliasing: bool = field(default=True, metadata=SUPPORTED)
    warn_on_unfilled_warps: bool = field(default=True, metadata=SUPPORTED)
    warn_software_fallback: bool = field(default=True, metadata=SUPPORTED)
    software_fallback_min_instructions: int = field(default=3, metadata=SUPPORTED)
    hint_fusable_multiply_add: bool = field(default=True, metadata=SUPPORTED)
    min_fusable_multiply_adds: int = field(default=1, metadata=SUPPORTED)
    warn_strict_atomic_cas: bool = field(default=True, metadata=SUPPORTED)
    warn_denormal_flush: bool = field(default=True, metadata=SUPPORTED)

    def __post_init__(self) -> None:
        for name in (
            "max_registers_per_thread",
            "max_spilled_bytes",
            "min_threads_per_block",
            "max_unrolled_instructions",
            "software_fallback_min_instructions",
            "min_fusable_multiply_adds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SettingError(f"warn.{name}", str(value), "a nonnegative integer")
        if not math.isfinite(self.min_occupancy) or not 0 <= self.min_occupancy <= 1:
            raise SettingError("warn.min_occupancy", str(self.min_occupancy), "a fraction from 0 to 1")
        for name in (
            "warn_on_unproven_aliasing",
            "warn_on_unfilled_warps",
            "warn_software_fallback",
            "hint_fusable_multiply_add",
            "warn_strict_atomic_cas",
            "warn_denormal_flush",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SettingError(f"warn.{name}", str(getattr(self, name)), "a boolean")


@dataclass(frozen=True, kw_only=True)
class Errors:
    """What a kernel's unhandled error does.

    capacity: device error records per launch; errors past it are counted and reported, never kept.
    policy: flag, where the failing thread records its error and leaves the kernel and the host raises once the
    launch settles; abort, where the thread traps at once.
    """

    capacity: int = field(default=256, metadata=SUPPORTED)
    policy: Literal["flag", "abort"] = field(default="flag", metadata=SUPPORTED)

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or self.capacity <= 0:
            raise SettingError("errors.capacity", str(self.capacity), "a positive integer")
        if self.policy not in ("flag", "abort"):
            raise SettingError("errors.policy", str(self.policy), "flag or abort")


@dataclass(frozen=True, kw_only=True)
class Log:
    """Device logging from kernels: records per launch, argument slots per record, and the transport.

    transport: `ring` works on every backend; `printf` only where the device has a printf (CUDA, HIP and the C
    backend), so a Metal target that asks for it gets a TargetError at validation, never a silent switch to the ring.
    A call's level comes from the client's logger when the kernel is compiled, so this group has no level.
    """

    capacity: int = field(default=4096, metadata=SUPPORTED)
    args: int = field(default=8, metadata=SUPPORTED)
    transport: Literal["ring", "printf"] = field(default="ring", metadata=SUPPORTED)

    def __post_init__(self) -> None:
        _positive("log.capacity", self.capacity)
        _positive("log.args", self.args)
        if self.transport not in ("ring", "printf"):
            raise SettingError("log.transport", str(self.transport), "ring or printf")


@dataclass(frozen=True, kw_only=True)
class Settings:
    warn: Warn = field(default_factory=Warn, metadata=SUPPORTED)
    debug: Debug = field(default_factory=Debug, metadata=SUPPORTED)
    lib: Lib = field(default_factory=Lib, metadata=SUPPORTED)
    cache: Cache = field(default_factory=Cache, metadata=SUPPORTED)
    precision: Precision = field(default_factory=Precision, metadata=SUPPORTED)
    compile: Compile = field(default_factory=Compile, metadata=SUPPORTED)
    run: Run = field(default_factory=Run, metadata=SUPPORTED)
    errors: Errors = field(default_factory=Errors, metadata=SUPPORTED)
    log: Log = field(default_factory=Log, metadata=SUPPORTED)

    @property
    def key(self) -> str:
        """Stable value key, including frozen mappings; never an identity or mutable hash."""
        return repr(self)

    def validate(self, target: Target) -> None:
        """Check target capabilities before compilation or device allocation."""
        backend = {"nvidia": "cuda", "amd": "amd", "apple": "metal", "cpu": "cpu"}[target.vendor]
        for name in ("run", "cache"):
            group = getattr(self, name)
            for item in fields(group):
                if backend not in item.metadata["backends"] and getattr(group, item.name) is not None:
                    raise TargetError(
                        f"{name}.{item.name}",
                        target.arch,
                        item.metadata["reasons"][target.vendor],
                        item.metadata["instead"],
                    )
        if self.log.transport == "printf" and backend not in PRINTF_BACKENDS:
            raise TargetError(
                "log.transport='printf'", target.arch, "Metal has no device printf", "set log.transport='ring'"
            )

    def configuration[S: DataclassInstance](self, kernel: str, struct: type[S]) -> S:
        """Construct the defaults, then coerce this kernel's overrides by each field's Const argument."""
        defaults = struct()
        annotations = get_type_hints(struct)
        overrides: dict[str, int | float | bool] = {}
        for key, text in self.run.config.items():
            owner, _, name = key.partition(".")
            if owner != kernel:
                continue
            annotation = annotations.get(name)
            if get_origin(annotation) is not Const:
                raise SettingError("run.config", f"{key}={text}", f"a Const field of {struct.__name__}")
            kind = get_args(annotation)[0]
            try:
                if kind is bool:
                    overrides[name] = {"0": False, "1": True, "false": False, "true": True}[text.lower()]
                elif kind in (int, float):
                    overrides[name] = kind(text)
                else:
                    raise ValueError
            except ValueError, KeyError:
                raise SettingError("run.config", f"{key}={text}", f"{kind.__name__} for {key}") from None
        return replace(defaults, **overrides)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> Self:
        """Parse seeds explicitly; normal work uses current_settings(), never reparses the environment."""
        env = os.environ if environ is None else environ
        workers = env.get("KRN_WORKERS")
        try:
            count = int(workers) if workers is not None else None
            _positive("KRN_WORKERS", count)
        except ValueError:
            raise SettingError("KRN_WORKERS", workers or "", "a positive integer") from None
        target = env.get("KRN_TARGET")
        try:
            run = Run(target=target, workers=count)
        except ValueError:
            raise SettingError("KRN_TARGET", target or "", "a known target architecture") from None
        cache = env.get("KRN_CACHE_DIR")
        if cache == "" or (cache is not None and "\0" in cache):
            raise SettingError("KRN_CACHE_DIR", cache, "a nonempty filesystem path")
        return cls(run=run, cache=Cache(dir=Path(cache) if cache is not None else None), debug=Debug.from_environ(env))


_lock = RLock()
_scoped: ContextVar[Settings | None] = ContextVar("krn_settings", default=None)


def current_settings() -> Settings:
    """The snapshot for work beginning in this thread/task."""
    scoped = _scoped.get()
    if scoped is not None:
        return scoped
    with _lock:
        return _current


@contextmanager
def snapshot_scope(snapshot: Settings) -> Generator[None]:
    """Keep nested compiler selectors on the request's snapshot."""
    token = _scoped.set(snapshot)
    try:
        yield
    finally:
        _scoped.reset(token)


def retained[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Pin one snapshot throughout a compiler or launch entry, including nested selectors."""

    @wraps(fn)
    def run(*args: P.args, **kwargs: P.kwargs) -> R:
        token = _scoped.set(current_settings())
        try:
            return fn(*args, **kwargs)
        finally:
            _scoped.reset(token)

    return run


type Value = (
    str
    | int
    | float
    | bool
    | Path
    | Mapping[str, str]
    | Debug
    | Lib
    | Cache
    | Precision
    | Compile
    | BackendCompile
    | Run
    | Warn
    | Errors
    | Log
    | None
)


def _change[T: DataclassInstance](snapshot: T, path: list[str], value: Value) -> T:
    name, *tail = path
    if name not in {item.name for item in fields(snapshot)}:
        raise SettingError(".".join(path), str(value), "a known setting")
    previous = getattr(snapshot, name)
    if tail:
        value = _change(previous, tail, value)
    elif isinstance(previous, Lib) and isinstance(value, str):
        value = replace(previous, mode=value)
    elif isinstance(previous, Cache) and isinstance(value, str | Path):
        value = Cache(dir=Path(value))
    elif isinstance(
        previous, (Debug, Lib, Cache, Precision, Compile, BackendCompile, Run, Warn, Errors, Log)
    ) and not isinstance(value, type(previous)):
        raise SettingError(name, str(value), type(previous).__name__)
    return replace(snapshot, **{name: value})


class Configuration:
    """A plain call installs globally; its context limits the override to the current thread/task."""

    def __init__(self, previous: Settings, chosen: Settings, token: Token[Settings | None] | None = None) -> None:
        self.previous, self.chosen = previous, chosen
        self.token = token

    def __enter__(self) -> Settings:
        global _current
        with _lock:
            if _current is self.chosen:
                _current = self.previous
        if self.token is None:
            self.token = _scoped.set(self.chosen)
        return self.chosen

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if self.token is not None:
            _scoped.reset(self.token)


def configure(**changes: Value) -> Configuration:
    """Replace full values atomically, accepting dotted keys or group dataclasses."""
    global _current
    with _lock:
        previous = _current
        chosen = current_settings()
        for key, value in changes.items():
            path = {"cache_dir": "cache.dir"}.get(key, key).split(".")
            chosen = _change(chosen, path, value)
        token = None
        if _scoped.get() is not None:
            token = _scoped.set(chosen)
        else:
            _current = chosen
        return Configuration(previous, chosen, token)


class GroupView:
    _path: str
    """Attribute writes publish a new snapshot, never mutate a retained group."""

    def __init__(self, path: str) -> None:
        self.__dict__["_path"] = path

    def __getattr__(self, name: str) -> Value:
        group = current_settings()
        for part in self._path.split("."):
            group = getattr(group, part)
        return getattr(group, name)

    def __setattr__(self, name: str, value: Value) -> None:
        configure(**{f"{self._path}.{name}": value})


class CompileView:
    cuda = GroupView("compile.cuda")
    amd = GroupView("compile.amd")
    metal = GroupView("compile.metal")
    cpu = GroupView("compile.cpu")


class Config:
    warn = GroupView("warn")
    debug = GroupView("debug")
    lib = GroupView("lib")
    cache = GroupView("cache")
    precision = GroupView("precision")
    compile = CompileView()
    run = GroupView("run")
    errors = GroupView("errors")
    log = GroupView("log")

    def __setattr__(self, name: str, value: Value) -> None:
        configure(**{name: value})


_current = Settings.from_environ()
config = Config()

debug_module._debug_settings = lambda: current_settings().debug

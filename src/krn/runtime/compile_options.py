"""Backend compiler choices from one settings snapshot and explicit code arguments.

Qualified settings for other backends stay inert. Unsupported explicit choices raise TargetError.
Resolved flags participate in program and persistent binary cache keys.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Self

from . import settings as settings_state

if TYPE_CHECKING:
    from .settings import Settings

from krn.compiler.errors import TargetError
from krn.facts.targets import AMD, CPU, Apple, Nvidia, Target


class Dialect(StrEnum):
    """The source backend: CUDA C++, C for a CPU, HIP C++ or Metal Shading Language."""

    CUDA = auto()
    CPU = auto()
    AMD = auto()
    METAL = auto()

    @classmethod
    def of(cls, target: Target) -> Dialect:
        match target:
            case Nvidia():
                return cls.CUDA
            case AMD():
                return cls.AMD
            case Apple():
                return cls.METAL
            case CPU():
                return cls.CPU
        raise ValueError(f"no backend for {target.arch}")


class Opt(StrEnum):
    O0 = "0"
    O1 = "1"
    O2 = "2"
    O3 = "3"
    SIZE = "s"
    FAST_COMPILE = "fast-compile"


CXX_STANDARDS = ("c++14", "c++17", "c++20", "c++23", "c++2c")
CAP = {Dialect.CUDA: "c++20", Dialect.CPU: "c++2c", Dialect.AMD: "c++2c", Dialect.METAL: "c++17"}
DEFAULT_STD = {
    Dialect.CUDA: "c++20",
    Dialect.CPU: "c++20",
    Dialect.AMD: "c++20",
    Dialect.METAL: "c++17",
}


@dataclass(frozen=True, kw_only=True)
class CompileOptions:
    """What one compile uses. `std` is the newest C++ the emitted text may rely on; `march` is the CPU `-march`, or
    the GPU architecture (`sm_90a`, `gfx942`) taken from the device."""

    dialect: Dialect
    std: str
    opt: Opt = Opt.O2
    march: str = "native"
    lineinfo: bool = False
    debug: bool = False

    @classmethod
    def resolve(
        cls,
        target: Target,
        environ: Mapping[str, str] | None = None,
        *,
        settings: Settings | None = None,
        march: str | None = None,
        std: str | None = None,
        opt: Opt | None = None,
        lineinfo: bool | None = None,
        debug: bool | None = None,
    ) -> Self:
        dialect = Dialect.of(target)

        snapshot = settings if settings is not None else settings_state.current_settings()
        group = getattr(snapshot.compile, dialect.value)

        def lookup(name: str) -> str | None:
            return {"CXX_STD": group.std, "OPT": group.opt, "MARCH": group.march}.get(name)

        chosen = cls(
            dialect=dialect,
            std=std or lookup("CXX_STD") or DEFAULT_STD[dialect],
            opt=opt or Opt(lookup("OPT") or Opt.O2),
            march=march or lookup("MARCH") or ("native" if dialect is Dialect.CPU else target.arch),
            lineinfo=lineinfo if lineinfo is not None else snapshot.debug.lineinfo,
            debug=debug if debug is not None else snapshot.debug.debug,
        )

        def refuse(name: str, reason: str, instead: str) -> None:
            variable = (
                f"debug.{name.lower()}"
                if name in {"DEBUG", "LINEINFO"}
                else f"compile.{dialect}.{dict(CXX_STD='std').get(name, name.lower())}"
            )
            raise TargetError(variable, target.arch, reason, instead)

        if (
            dialect is not Dialect.METAL
            and chosen.std in CXX_STANDARDS
            and CXX_STANDARDS.index(chosen.std) > CXX_STANDARDS.index(CAP[dialect])
        ):
            refuse("CXX_STD", f"the {dialect} compiler supports at most {CAP[dialect]}", f"use {CAP[dialect]} or max")
        if dialect is Dialect.CPU and (std is not None or lookup("CXX_STD") is not None):
            refuse(
                "CXX_STD",
                "krn's CPU backend emits C and invokes a C compiler, not C++",
                "set compile.std=None or use a GPU C++ backend",
            )
        if (
            dialect is Dialect.CUDA
            and (opt is not None or lookup("OPT") is not None)
            and chosen.opt not in (Opt.O0, Opt.FAST_COMPILE)
        ):
            refuse(
                "OPT",
                "NVRTC exposes no numeric optimization levels or size optimization",
                "set compile.opt=None for full optimization, or use 0 or fast-compile",
            )
        if dialect is Dialect.METAL:
            if std is not None or lookup("CXX_STD") is not None:
                refuse(
                    "CXX_STD",
                    "the runtime compiles fixed MSL 3.2 source, not a selectable C++ dialect",
                    "set compile.std=None or use CUDA or HIP",
                )
            if chosen.opt not in (Opt.O2, Opt.SIZE):
                refuse(
                    "OPT",
                    "krn's Metal source compiler API has no lowering for this optimization level",
                    "use compile.opt=2 or set compile.opt=None",
                )
            if march is not None or lookup("MARCH") is not None:
                refuse(
                    "MARCH",
                    "Metal selects the GPU architecture from the MTLDevice, not a march flag",
                    "set compile.march=None and select the device with KRN_TARGET",
                )
            if chosen.lineinfo or chosen.debug:
                refuse(
                    "DEBUG" if chosen.debug else "LINEINFO",
                    "krn's Metal source compiler API has no lowering for source debug or line-table flags",
                    "disable this option or use the offline Metal compiler",
                )
            if chosen.opt is Opt.SIZE:
                refuse(
                    "OPT",
                    "krn's Metal source API exposes no size optimization flag",
                    "use compile.opt=2",
                )
        return replace(chosen, std=chosen._standard())

    def _standard(self) -> str:
        """`max` selects the backend's own default cap; explicit standards never clamp."""
        if self.std != "max" and self.std not in CXX_STANDARDS:
            raise ValueError(f"unknown C++ standard {self.std!r}; known: {', '.join(CXX_STANDARDS)}, max")
        if self.std == "max":
            return CAP[self.dialect]
        return self.std

    def flags(self) -> tuple[str, ...]:
        """The compiler options for this backend. Debug is `-G` or `-O0 -g`; line info keeps optimization."""
        return {Dialect.CUDA: _cuda, Dialect.CPU: _cpu, Dialect.AMD: _amd, Dialect.METAL: _metal}[self.dialect](self)


def _level(options: CompileOptions) -> str:
    """The `-O` digit of clang: debug is 0 and fast-compile is 1, the others are what they say."""
    return "0" if options.debug else "1" if options.opt is Opt.FAST_COMPILE else options.opt.value


STRICT_CUDA = ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
"""The NVRTC numerical options of every CUDA compile. `numerical_options` gives them to every compile and refuses a
caller option that changes them."""

_SPELLING = re.compile(r"-+([^=\s]+)(?:[=\s]+(.*))?")
_ALIASES = {"arch": "gpu_architecture"}
_REQUIRED = ("-default-device", *STRICT_CUDA)


@dataclass(frozen=True, kw_only=True)
class _Flag:
    """An NVRTC option with its name (dashes read as underscores, so `-use-fast-math` and `--use_fast_math` agree) and
    its value (a bare switch is `true`, as NVRTC reads it). Text that is not a dashed option is its own name."""

    text: str
    name: str
    value: str

    @classmethod
    def of(cls, text: str) -> Self:
        match = _SPELLING.fullmatch(text)
        if match is None:
            return cls(text=text, name=text, value="")
        name, value = match.groups()
        name = name.replace("-", "_")
        return cls(text=text, name=_ALIASES.get(name, name), value="true" if value is None else value)


_STRICT_NAMES = frozenset({"use_fast_math", *(_Flag.of(flag).name for flag in STRICT_CUDA)})


def numerical_options(
    options: Sequence[str], arch_flag: str | None = None, extra: Sequence[str] = ()
) -> tuple[str, ...]:
    """The options of one CUDA compile under the numerical contract: each required option once, in the spelling given
    here, and the missing ones after the caller's. `arch_flag` is the target's architecture option, or the caller's own
    when None; `extra` are required options of one kind of compile. A caller option with another value for a required
    one, or fast math in any spelling, raises TargetError."""
    flags = [_Flag.of(option) for option in options]
    given = next((flag for flag in flags if flag.name == "gpu_architecture"), None)
    architecture = arch_flag or (f"--gpu-architecture={given.value}" if given else None)
    texts = (*([architecture] if architecture else []), *_REQUIRED, *extra)
    required = {flag.name: flag for flag in map(_Flag.of, texts)}
    target = required["gpu_architecture"].value if "gpu_architecture" in required else "CUDA"
    for flag in flags:
        pinned = required.get(flag.name)
        if flag.name == "use_fast_math" or (pinned is not None and pinned.value != flag.value):
            raise _refused(flag, target)
    kept: list[str] = []
    placed: set[str] = set()
    for flag in flags:
        if flag.name not in required:
            kept.append(flag.text)
        elif flag.name not in placed:
            placed.add(flag.name)
            kept.append(required[flag.name].text)
    return (*kept, *(flag.text for name, flag in required.items() if name not in placed))


def _refused(flag: _Flag, target: str) -> TargetError:
    """Names what was given, the target, why the contract forbids it and what to do instead."""
    if flag.name == "gpu_architecture":
        reason = "one compile builds for one architecture, the target's"
        instead = "pass the target's architecture, or drop the option"
    elif flag.name in _STRICT_NAMES:
        reason = (
            "krn compiles strictly: fused multiply-add, flush-to-zero, fast math and approximate division or square "
            "root would round the same source differently from the runtime"
        )
        instead = "use krn.fast.math.<op> for an approximate function, or the explicit math.fma"
    else:
        reason, instead = f"krn sets {flag.name.replace('_', '-')} itself on every compile", "drop the option"
    return TargetError(flag.text, target, reason, instead)


def _cuda(o: CompileOptions) -> tuple[str, ...]:
    """NVRTC has no `-O` option: level 0 and debug are `--device-debug`, fast-compile is `--Ofast-compile=min`, and
    everything else is full optimization. `-default-device` lets unannotated helpers and lambdas run on the device."""

    debug = ("--device-debug",) if o.debug or o.opt is Opt.O0 else ()
    compile_time = ("--Ofast-compile=min",) if o.opt is Opt.FAST_COMPILE and not debug else ()
    lines = ("--generate-line-info",) if o.lineinfo else ()
    return (
        f"--gpu-architecture={o.march}",
        f"--std={o.std}",
        "-default-device",
        *STRICT_CUDA,
        *debug,
        *compile_time,
        *lines,
    )


def _cpu(o: CompileOptions) -> tuple[str, ...]:
    """Export kernels with default visibility and keep each floating-point operation separate."""

    base = (
        f"-std={o.std}",
        f"-O{_level(o)}",
        f"-march={o.march}",
        "-nostdlib++",
        "-fPIC",
        "-shared",
    )
    base += ("-fvisibility=hidden", "-fno-semantic-interposition")
    return (
        *base,
        "-ffp-contract=off",
        *(("-g",) if o.debug else ("-gline-tables-only",) if o.lineinfo else ()),
    )


def _amd(o: CompileOptions) -> tuple[str, ...]:

    base = (
        "-x",
        "hip",
        f"--offload-arch={o.march}",
        "--offload-device-only",
        "-nogpulib",
        "-nogpuinc",
    )
    base += (f"-std={o.std}", f"-O{_level(o)}")
    return (
        *base,
        "-ffp-contract=off",
        "-fhip-fp32-correctly-rounded-divide-sqrt",
        "-fno-gpu-flush-denormals-to-zero",
        *(("-g",) if o.debug else ("-gline-tables-only",) if o.lineinfo else ()),
    )


def _metal(o: CompileOptions) -> tuple[str, ...]:
    """Metal's own default is fast math, the opposite of the other backends, so the mode is always passed."""

    recording = ("-gline-tables-only", "-frecord-sources") if o.lineinfo or o.debug else ()
    return (
        "-x",
        "metal",
        f"-std=metal{'4.0' if o.std == 'c++17' else '3.2'}",
        "-Os" if o.opt is Opt.SIZE else "-O2",
        "-ffp-contract=off",
        "-fmetal-math-mode=safe",
        "-fmetal-math-fp32-functions=precise",
        *recording,
    )

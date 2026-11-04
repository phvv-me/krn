"""What a compilation leaves behind for inspection: the IR as text, the generated source, and PTX and cubin from NVRTC.

`inspect_kernel` builds only the artifacts `KRN_DEBUG_KEEP` names (or all of them when asked), saves them to
`KRN_DEBUG_DUMP_DIR`, and returns an `Inspection` whose attributes are the artifacts, `inspection.ptx`, the way CuTe DSL
exposes `__ptx__`. Line information and device debug follow `KRN_DEBUG_LINEINFO` and `KRN_DEBUG`.
"""

import dataclasses
import logging
from dataclasses import is_dataclass

from krn.facts.targets import Nvidia as NvidiaTarget
from krn.facts.targets import Target
from krn.runtime.compile_options import CompileOptions
from krn.runtime.nvrtc import Nvidia
from krn.runtime.settings import current_settings

from .debug import Artifact, Debug, Inspection
from .dtypes import Dtype
from .errors import TargetError
from .ir import Kernel, Node
from .lower import emit

log = logging.getLogger(__name__)


def format_ir[V](value: V, depth: int = 0) -> str:
    """The IR as indented text, one node per line, with the source line of every statement."""
    pad = "  " * depth
    if isinstance(value, tuple):
        return "".join(format_ir(item, depth) for item in value)
    if not (is_dataclass(value) and not isinstance(value, type)):
        return f"{pad}{value}\n"
    fields = [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value) if f.name != "span"]
    scalars = ", ".join(f"{n}={v}" for n, v in fields if isinstance(v, str | int | float | bool | Dtype))
    line = f" (line {value.span.line})" if isinstance(value, Node) else ""
    nested = "".join(format_ir(v, depth + 1) for _, v in fields if not isinstance(v, str | int | float | bool | Dtype))
    return f"{pad}{type(value).__name__} {scalars}{line}\n{nested}"


def inspect_kernel(
    kernel: Kernel,
    target: Target,
    *,
    nvrtc: Nvidia | None = None,
    debug: Debug | None = None,
    artifacts: frozenset[Artifact] | None = None,
) -> Inspection:
    """Build the artifacts of `kernel` for `target`; save the ones `debug.keep` names when `artifacts` is not given."""
    settings = debug or current_settings().debug
    asked = set(settings.keep if artifacts is None else artifacts) or {
        Artifact.IR,
        Artifact.SOURCE,
    }
    if unsupported := asked & {Artifact.SASS, Artifact.ASM}:
        raise TargetError(
            f"KRN_DEBUG_KEEP={','.join(sorted(unsupported))}",
            target.arch,
            "krn's artifact builder has no disassembler or assembly-output lowering for this target",
            "request ir or source, or disassemble a compiled binary with the target's native tools",
        )
    unit = emit(kernel, target, lineinfo=settings.lineinfo)
    made: dict[Artifact, str | bytes] = {}
    if Artifact.IR in asked:
        made[Artifact.IR] = format_ir(kernel.body)
    if Artifact.SOURCE in asked:
        made[Artifact.SOURCE] = unit.source
    if asked & {Artifact.PTX, Artifact.CUBIN}:
        if nvrtc is None or not isinstance(target, NvidiaTarget):
            raise TargetError(
                "KRN_DEBUG_KEEP (PTX/cubin)",
                target.arch,
                "PTX and cubin need NVRTC and an NVIDIA target",
                "request ir or source artifacts, or use an NVIDIA target with NVRTC",
            )
        else:
            options = CompileOptions.resolve(target, lineinfo=settings.lineinfo, debug=settings.debug).flags()
            for kind, artifact in (("ptx", Artifact.PTX), ("cubin", Artifact.CUBIN)):
                if artifact in asked:
                    data = nvrtc.compile(unit.source, f"sm_{target.sm}a", kind, options).data
                    made[artifact] = data.decode() if kind == "ptx" else data
    inspection = Inspection(kernel.name, made)
    if settings.keep:
        log.info("saved %s", ", ".join(map(str, inspection.save(settings.dump_dir))))
    return inspection

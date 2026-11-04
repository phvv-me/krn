"""Warnings about computed compiler facts, independent of compiler choices."""

import warnings
from dataclasses import dataclass
from threading import RLock

from krn.runtime.settings import Warn

from .errors import Span
from .ir import Kernel


class PerformanceWarning(UserWarning):
    """A computed kernel resource or launch fact exceeds a configurable budget."""


@dataclass(frozen=True, kw_only=True)
class Facts:
    registers: int | None = None
    spilled_bytes: int | None = None
    occupancy: float | None = None
    threads: int | None = None
    warp_size: int | None = None
    thread_limit: int | None = None
    unrolled_instructions: int | None = None
    unproven_aliasing: bool = False


_seen: set[tuple[str, int, str, str, str]] = set()
_lock = RLock()


def warn_once(
    source: str,
    target: str,
    kind: str,
    message: str,
    category: type[UserWarning],
    span: Span,
    scope: Span | None = None,
) -> None:
    """Source-located warnings share identity and retain normal Python warning filters.

    scope: the line that keys the dedupe when it is not the warning's own (a kernel's line for its calls).
    """
    anchor = span if scope is None else scope
    key = (anchor.file, anchor.line, source, kind, target)
    with _lock:
        if key in _seen:
            return
        # Mark after emission so warnings promoted to errors remain repeatable.
        warnings.warn_explicit(f"{source} on {target}: {message}", category, span.file, span.line)
        _seen.add(key)


def check(kernel: Kernel, target: str, limits: Warn, facts: Facts) -> None:
    """Emit each exceeded kind once per source kernel and target; missing facts have no check."""
    messages: dict[str, str] = {}
    if facts.registers is not None and facts.registers > limits.max_registers_per_thread:
        messages["registers"] = (
            f"uses {facts.registers} registers per thread, limit {limits.max_registers_per_thread}; "
            "use a smaller tile or fewer live values"
        )
    if facts.spilled_bytes is not None and facts.spilled_bytes > limits.max_spilled_bytes:
        messages["spills"] = (
            f"spills {facts.spilled_bytes} bytes, limit {limits.max_spilled_bytes}; "
            "use fewer live values or split the kernel"
        )
    if facts.occupancy is not None and facts.occupancy < limits.min_occupancy:
        messages["occupancy"] = (
            f"occupancy {100 * facts.occupancy:g} percent, minimum {100 * limits.min_occupancy:g} percent; "
            "use a smaller tile, fewer live values, or change Blocks(threads=...)"
        )
    if facts.threads is not None:
        if facts.threads < limits.min_threads_per_block:
            messages["threads"] = (
                f"uses {facts.threads} threads per block, minimum {limits.min_threads_per_block}; "
                "increase Blocks(threads=...) or the tile size"
            )
        if limits.warn_on_unfilled_warps and facts.warp_size and facts.threads % facts.warp_size:
            messages["warps"] = (
                f"uses {facts.threads} threads per block with warp width {facts.warp_size}; "
                "choose Blocks(threads=...) or a tile size that fills whole warps"
            )
    if facts.thread_limit is not None and facts.thread_limit < limits.min_threads_per_block:
        messages["pipeline"] = (
            f"pipeline allows {facts.thread_limit} threads per block, minimum {limits.min_threads_per_block}; "
            "use a smaller tile or fewer live values"
        )
    if facts.unrolled_instructions is not None and facts.unrolled_instructions > limits.max_unrolled_instructions:
        messages["unroll"] = (
            f"expanded IR has {facts.unrolled_instructions} statements (instruction estimate), "
            f"limit {limits.max_unrolled_instructions}; "
            "reduce constant loop bounds or split the kernel"
        )
    if limits.warn_on_unproven_aliasing and facts.unproven_aliasing:
        messages["aliasing"] = "cannot prove writable arguments disjoint; use separate buffers or split the kernel"
    for kind, message in messages.items():
        warn_once(kernel.name, target, kind, message, PerformanceWarning, kernel.span)


@dataclass(frozen=True, kw_only=True)
class Compilation:
    cubin: bytes
    registers: int | None = None
    spilled_bytes: int | None = None


def compilation(cubin: bytes, compiler_log: str) -> Compilation:
    """Read ptxas resource lines, never mistake local arrays or stack bytes for spills."""
    registers = None
    spilled = None
    for line in compiler_log.splitlines():
        words = line.replace(",", "").split()
        if "Used" in words and "registers" in words:
            registers = int(words[words.index("registers") - 1])
        if "spill" in words:
            spilled = sum(
                int(words[i - 3])
                for i, word in enumerate(words)
                if word in {"stores", "loads"} and words[i - 1] == "spill"
            )
    return Compilation(cubin=cubin, registers=registers, spilled_bytes=spilled)

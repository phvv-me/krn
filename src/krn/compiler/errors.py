"""Source spans and compiler-style diagnostics."""

import ast
import linecache
from dataclasses import dataclass


class PrecisionWarning(UserWarning):
    """krn reduces numeric precision because the user asked for it at a call site.

    Emitted for approximate `fast.math.*` calls, for reduced-precision matmul accumulation, for float32
    operations a target flushes denormals in (kind `denormal-flush:<operation>`), and later for truncating casts.
    Separate from PerformanceWarning, which reports computed facts.
    Kernel-level warnings go through `performance.warn_once`: once per (kernel, kind, target), keyed by the
    kernel's line and located at the user's line. A fast.math call's kind is its operation (`fast.sqrt`), so
    each approximate operation warns once per kernel and target. Host-side library warnings (matmul reduced
    accumulation, kind `reduced-accumulation`) are once per (operation, kind, target, call site), so each
    line that asks warns once. Silence or escalate with
    `warnings.filterwarnings("ignore" | "error", category=PrecisionWarning)`.
    """


@dataclass(frozen=True, slots=True)
class Span:
    """A position in the user's file; `col` is 1-based like a compiler's."""

    file: str
    line: int
    col: int

    @classmethod
    def of(cls, file: str, node: ast.AST) -> Span:
        line: int = getattr(node, "lineno", 0)
        col: int = getattr(node, "col_offset", 0)
        return cls(file, line, col + 1)

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


class CaptureError(Exception):
    """A compiler-style diagnostic: `file:line:col: error: message`, then the source line and a caret."""

    def __init__(self, span: Span | None, message: str) -> None:
        super().__init__(f"{span}: error: {message}" if span else f"error: {message}")
        self.span: Span | None = span
        self.message = message

    def __str__(self) -> str:
        if self.span is None:
            return self.args[0]
        source = linecache.getline(self.span.file, self.span.line).rstrip()
        return f"{self.args[0]}\n    {source}\n    {' ' * (self.span.col - 1)}^" if source else self.args[0]


class TargetError(CaptureError):
    """An unavailable target feature, with an actionable alternative and an optional source location."""

    def __init__(self, feature: str, target: str, reason: str, instead: str, *, span: Span | None = None) -> None:
        self.feature, self.target, self.reason, self.instead = feature, target, reason, instead
        message = f"{feature} is not available on {target}: {reason.rstrip('.')}. Instead: {instead.rstrip('.')}."
        super().__init__(span, message)


class SettingError(ValueError):
    """An override cannot be interpreted as the setting it names."""

    def __init__(self, variable: str, value: str, expected: str) -> None:
        super().__init__(f"{variable}={value!r}: expected {expected}")

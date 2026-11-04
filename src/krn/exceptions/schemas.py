"""Construction schemas of the builtin exception classes a kernel may raise.

A device passes the recorded arguments to the constructor positionally and unchanged, which is what Python does for the
same call, so `args`, `code`, `errno`, `value` and the repr of a rebuilt error are Python's own. A schema only bounds
the positional count, where the constructor rejects more or documents a bound. A builtin missing here cannot be raised.
"""

import builtins
from dataclasses import dataclass


def is_builtin(cls: type) -> bool:
    """Whether cls is the interpreter's own class of that name, whose constructor is C code."""
    return getattr(builtins, cls.__name__, None) is cls


@dataclass(frozen=True, kw_only=True)
class Schema:
    """The positional arguments a builtin constructor takes from a device: at most `most`, any count when None."""

    most: int | None = None

    def accepts(self, count: int) -> bool:
        return self.most is None or count <= self.most


_ANY = Schema()
_ONE = Schema(most=1)
_OS = Schema(most=5)
# BlockingIOError's third argument, characters_written, must be an int, which a device's float or bool would violate.
_BLOCKING = Schema(most=2)

SCHEMAS: dict[type[BaseException], Schema] = {
    BaseException: _ANY,
    Exception: _ANY,
    GeneratorExit: _ANY,
    KeyboardInterrupt: _ANY,
    SystemExit: _ANY,
    ArithmeticError: _ANY,
    FloatingPointError: _ANY,
    OverflowError: _ANY,
    ZeroDivisionError: _ANY,
    AssertionError: _ANY,
    AttributeError: _ANY,
    BufferError: _ANY,
    EOFError: _ANY,
    ImportError: _ANY,
    ModuleNotFoundError: _ANY,
    LookupError: _ANY,
    IndexError: _ANY,
    KeyError: _ANY,
    MemoryError: _ANY,
    NameError: _ANY,
    UnboundLocalError: _ANY,
    OSError: _OS,
    BlockingIOError: _BLOCKING,
    ChildProcessError: _OS,
    ConnectionError: _OS,
    BrokenPipeError: _OS,
    ConnectionAbortedError: _OS,
    ConnectionRefusedError: _OS,
    ConnectionResetError: _OS,
    FileExistsError: _OS,
    FileNotFoundError: _OS,
    InterruptedError: _OS,
    IsADirectoryError: _OS,
    NotADirectoryError: _OS,
    PermissionError: _OS,
    ProcessLookupError: _OS,
    TimeoutError: _OS,
    ReferenceError: _ANY,
    RuntimeError: _ANY,
    NotImplementedError: _ANY,
    RecursionError: _ANY,
    StopAsyncIteration: _ANY,
    StopIteration: _ANY,
    SyntaxError: _ONE,
    IndentationError: _ONE,
    TabError: _ONE,
    SystemError: _ANY,
    TypeError: _ANY,
    ValueError: _ANY,
    UnicodeError: _ANY,
    Warning: _ANY,
    UserWarning: _ANY,
    DeprecationWarning: _ANY,
    PendingDeprecationWarning: _ANY,
    SyntaxWarning: _ANY,
    RuntimeWarning: _ANY,
    FutureWarning: _ANY,
    ImportWarning: _ANY,
    UnicodeWarning: _ANY,
    BytesWarning: _ANY,
    ResourceWarning: _ANY,
    EncodingWarning: _ANY,
}

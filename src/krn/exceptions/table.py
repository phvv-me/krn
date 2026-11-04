"""Exception classes with the small integer ids a kernel's `raise` and `except` compare."""

import json
import sys
from dataclasses import dataclass
from types import FunctionType
from typing import Self

from .device_safe import Member, Reason, device_safe
from .schemas import is_builtin

# The builtins every program starts with, in this fixed order, so their ids match across programs.
BUILTINS: tuple[type[BaseException], ...] = (
    BaseException,
    BaseExceptionGroup,
    GeneratorExit,
    KeyboardInterrupt,
    SystemExit,
    Exception,
    ArithmeticError,
    FloatingPointError,
    OverflowError,
    ZeroDivisionError,
    AssertionError,
    AttributeError,
    BufferError,
    EOFError,
    ExceptionGroup,
    ImportError,
    ModuleNotFoundError,
    LookupError,
    IndexError,
    KeyError,
    MemoryError,
    NameError,
    UnboundLocalError,
    OSError,
    ReferenceError,
    RuntimeError,
    NotImplementedError,
    RecursionError,
    StopAsyncIteration,
    StopIteration,
    SyntaxError,
    IndentationError,
    TabError,
    SystemError,
    TypeError,
    ValueError,
    UnicodeError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    UnicodeTranslateError,
    Warning,
    UserWarning,
    DeprecationWarning,
    RuntimeWarning,
)
# A class id is stored in the device record as a u16.
MAX_ID = 0xFFFF
# Namespace entries that Python fills in on first access without changing what a class does.
_VOLATILE = frozenset({"__annotations_cache__"})


class ClassChangedError(RuntimeError):
    """A class changed after it was registered, so the table's subclass sets and device verdict for it are stale."""


@dataclass(frozen=True, kw_only=True)
class _Snapshot:
    """The identities of a class's name, bases, MRO, namespaces and function code.

    The parts are held so that their ids cannot be reused by a replacement while the snapshot lives. Annotations are
    tracked through __annotate__, because reading __annotations__ can raise for a forward reference; an in-place edit
    of an annotations dict is therefore not seen.
    """

    parts: tuple[Member, ...]
    ids: tuple[int, ...]

    @classmethod
    def of(cls, klass: type) -> Self:
        parts = tuple(_parts(klass))
        return cls(parts=parts, ids=tuple(map(id, parts)))


def _parts(klass: type) -> list[Member]:
    parts: list[Member] = [klass.__bases__, klass.__mro__]
    for ancestor in klass.__mro__:
        if is_builtin(ancestor):
            continue
        parts.append(ancestor.__qualname__)
        for name, value in vars(ancestor).items():
            if name in _VOLATILE:
                continue
            parts += [name, value]
            if isinstance(value, FunctionType):
                parts += [value.__code__, value.__defaults__, value.__kwdefaults__, value.__annotate__]
    return parts


def resolve(module: str, qualname: str) -> type | None:
    """The class that module.qualname names in this process, or None when nothing or something else does."""
    found = sys.modules.get(module)
    for part in qualname.split("."):
        found = getattr(found, part, None)
    return found if isinstance(found, type) else None


def importable(cls: type) -> bool:
    """Whether cls is found again under its module and qualified name, so a rebuild or another process can find it."""
    return resolve(cls.__module__, cls.__qualname__) is cls


class ErrorClassTable:
    """Maps exception classes to ids, and records per class its registered subclass ids and its device verdict.

    Ids are stable within one process, in registration order. A host in another process exchanges dumps(), the
    qualified names in id order, and loads it, which gives the same ids. A class whose name, bases, MRO, namespace or
    function code changes after registration raises ClassChangedError on the next query that depends on it.
    Bases register before their subclasses, so every ancestor already knows a new class when it arrives.
    """

    def __init__(self) -> None:
        self._ids: dict[type[BaseException], int] = {}
        self._classes: list[type[BaseException]] = []
        self._handled: list[set[int]] = []
        self._snapshots: list[_Snapshot] = []
        self._verdicts: list[Reason | None] = []
        for cls in BUILTINS:
            self.register(cls)

    def register(self, cls: type[BaseException]) -> int:
        """The id of cls, registering it and its bases first when it is new. The device verdict is kept with it."""
        if cls in self._ids:
            identifier = self._ids[cls]
            self._check(identifier)
            return identifier
        if not issubclass(cls, BaseException):
            raise TypeError(f"{cls.__qualname__} is not an exception class")
        for base in cls.__bases__:
            if issubclass(base, BaseException):
                self.register(base)
        identifier = len(self._classes)
        if identifier > MAX_ID:
            raise OverflowError(
                f"{cls.__qualname__} would be exception class {identifier}; device records hold {MAX_ID}"
            )
        self._ids[cls] = identifier
        self._classes.append(cls)
        self._handled.append({identifier})
        self._snapshots.append(_Snapshot.of(cls))
        self._verdicts.append(device_safe(cls))
        for ancestor in cls.__mro__[1:]:
            if issubclass(ancestor, BaseException) and ancestor in self._ids:
                self._handled[self._ids[ancestor]].add(identifier)
        return identifier

    def cls_of(self, identifier: int) -> type[BaseException]:
        """The class registered under identifier, as it is now: a rebuild runs its current constructor."""
        return self._classes[identifier]

    def device_reason(self, cls: type[BaseException]) -> Reason | None:
        """Why cls cannot be raised on a device, registering it first; None when it can."""
        return self._verdicts[self.register(cls)]

    def handler_ids(self, handler: type[BaseException]) -> tuple[int, ...]:
        """Sorted ids of the handler and every registered subclass, the set an `except handler` tests."""
        identifier = self.register(handler)
        for other in range(len(self._classes)):
            self._check(other)
        return tuple(sorted(self._handled[identifier]))

    def matches(self, raised_id: int, handler: type[BaseException]) -> bool:
        """Whether an error of class raised_id is caught by `except handler`, as issubclass decides."""
        self._check(raised_id)
        return raised_id in self._handled[self.register(handler)]

    def dumps(self) -> str:
        """The qualified names in id order, as JSON. Every class must be importable by its name."""
        for cls in self._classes:
            if not importable(cls):
                raise ValueError(f"{cls.__qualname__} is not importable by name, so another process cannot find it")
        return json.dumps([[cls.__module__, cls.__qualname__] for cls in self._classes])

    @classmethod
    def loads(cls, text: str) -> Self:
        """The table dumps wrote, with the same ids: each name is resolved here, and every id must come out the same."""
        table = cls()
        for identifier, (module, qualname) in enumerate(json.loads(text)):
            found = resolve(module, qualname)
            if not (isinstance(found, type) and issubclass(found, BaseException)):
                raise ValueError(f"{module}.{qualname} cannot be found in this process")
            if table.register(found) != identifier:
                raise ValueError(f"{module}.{qualname} gets a different id here")
        return table

    def _check(self, identifier: int) -> None:
        cls = self._classes[identifier]
        if _Snapshot.of(cls).ids != self._snapshots[identifier].ids:
            raise ClassChangedError(
                f"{cls.__qualname__} changed after it was registered; this table's subclass sets and device verdict "
                "for it are stale, so register it in a new ErrorClassTable"
            )

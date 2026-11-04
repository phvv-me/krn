"""Host raising: the kernel's errors as the classes it raised, with where they came from as notes."""

from operator import attrgetter

import numpy as np

from .._internals.records import Scalar
from .records import ErrorEvent, decode_error
from .sites import ErrorSiteTable
from .table import ErrorClassTable, importable


class KernelError(RuntimeError):
    """An error the host could not rebuild as the class the kernel raised.

    The message names the class and its arguments. kernel, site (file and line), class_name, block, thread and count
    say where it was raised, count being the other threads that raised the same site. __cause__ is the failure that
    prevented the rebuild, when there was one.
    """

    def __init__(
        self,
        message: str,
        *,
        kernel: str,
        site: tuple[str, int],
        class_name: str,
        block: int,
        thread: int,
        count: int,
    ) -> None:
        super().__init__(message)
        self.kernel = kernel
        self.site = site
        self.class_name = class_name
        self.block = block
        self.thread = thread
        self.count = count


def rebuild(table: ErrorClassTable, event: ErrorEvent, others: int = 0) -> BaseException:
    """The exception the kernel raised, built as its host class, or a KernelError when that class cannot be rebuilt.

    A constructor that raises anything, not only TypeError, is the same case: the failure becomes the KernelError's
    cause. A BaseException that is not an Exception, such as KeyboardInterrupt, is not the kernel's and propagates.
    others: how many other threads raised the same site.
    """
    cls = table.cls_of(event.class_id)
    arguments = event.site.arguments(event.values)
    if not importable(cls):
        return _kernel_error(event, cls, arguments, others, cause=None)
    try:
        error = cls(*arguments)
    except BaseException as failure:
        if not isinstance(failure, Exception):
            raise
        return _kernel_error(event, cls, arguments, others, cause=failure)
    for note in _notes(event, others):
        error.add_note(note)
    return error


def raise_errors(table: ErrorClassTable, sites: ErrorSiteTable, records: np.ndarray, dropped: int, kernel: str) -> None:
    """Raise nothing, the one rebuilt error, or an ExceptionGroup named after the kernel.

    Errors are ordered by their first (block, thread). dropped counts the errors past the buffer's capacity.
    """
    events = sorted((decode_error(record, sites) for record in records), key=attrgetter("block", "thread"))
    if not events:
        return
    groups: dict[tuple[int, int], list[ErrorEvent]] = {}
    for event in events:
        groups.setdefault((event.site.id, event.class_id), []).append(event)
    errors = [rebuild(table, members[0], others=len(members) - 1) for members in groups.values()]
    raised = errors[0] if len(errors) == 1 else BaseExceptionGroup(kernel, errors)
    if len(errors) > 1:
        raised.add_note(f"{len(events)} threads raised {len(errors)} distinct errors")
    if dropped:
        raised.add_note(f"{dropped} more errors were not recorded (buffer capacity {len(records)})")
    raise raised


def _kernel_error(
    event: ErrorEvent,
    cls: type[BaseException],
    arguments: tuple[Scalar | str, ...],
    others: int,
    cause: Exception | None,
) -> KernelError:
    site = event.site
    class_name = f"{cls.__module__}.{cls.__qualname__}"
    error = KernelError(
        f"{class_name}: {', '.join(str(argument) for argument in arguments)}",
        kernel=site.kernel,
        site=(site.file, site.line),
        class_name=class_name,
        block=event.block,
        thread=event.thread,
        count=others,
    )
    error.__cause__ = cause
    for note in _notes(event, others):
        error.add_note(note)
    return error


def _notes(event: ErrorEvent, others: int) -> list[str]:
    site = event.site
    notes = [
        f"raised in kernel {site.kernel} at {site.file}:{site.line}",
        f"block {event.block}, thread {event.thread}",
    ]
    if others:
        notes.append(f"{others} other {'thread' if others == 1 else 'threads'} raised the same site")
    return notes

import functools
import inspect
import time
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
    from types import TracebackType

    from .profiler import Profiler


_active: Profiler | None = None
type SpanToken = tuple[Profiler, int, int] | None


def active() -> Profiler | None:
    """Return the profiler currently receiving spans, or None if none is active."""
    return _active


def activate(session: Profiler) -> None:
    """Route dormant spans to `session` until it is deactivated."""
    global _active
    if _active is not None:
        raise RuntimeError("only one Profiler may be active in a process")
    _active = session


def deactivate(session: Profiler) -> None:
    """Stop routing spans when `session` owns the active slot."""
    global _active
    if _active is session:
        _active = None


def start(name: str) -> SpanToken:
    """Start `name` when a profiler is active, otherwise return no token."""
    session = _active
    if session is None:
        return None
    token = session.enter(name)
    return session, token, time.perf_counter_ns()


def finish(token: SpanToken) -> None:
    """Close a token returned by `start`."""
    if token is None:
        return
    session, key, start_ns = token
    session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)


class Span:
    """A named context that is dormant until a `Profiler` is active."""

    __slots__ = ("name", "session", "start_ns", "token")

    name: str
    session: Profiler | None
    start_ns: int
    token: int

    @staticmethod
    def _decorate_async[**P, R](func: Callable[P, Awaitable[R]], label: str) -> Callable[P, Coroutine[None, None, R]]:

        @functools.wraps(func)
        async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            session = _active
            if session is None:
                return await func(*args, **kwargs)
            key = session.enter(label)
            start_ns = time.perf_counter_ns()
            try:
                return await func(*args, **kwargs)
            finally:
                session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)

        return inner

    @staticmethod
    def _decorate_async_generator[**P, Y](
        func: Callable[P, AsyncIterator[Y]], label: str
    ) -> Callable[P, AsyncIterator[Y]]:
        """Plain async-for delegation, not forwarding `asend`/`athrow`, as CPython's fix does."""

        @functools.wraps(func)
        async def inner(*args: P.args, **kwargs: P.kwargs) -> AsyncIterator[Y]:
            session = _active
            if session is None:
                async for item in func(*args, **kwargs):
                    yield item
                return
            key = session.enter(label)
            start_ns = time.perf_counter_ns()
            try:
                async for item in func(*args, **kwargs):
                    yield item
            finally:
                session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)

        return inner

    @staticmethod
    def _decorate_generator[**P, Y](func: Callable[P, Iterator[Y]], label: str) -> Callable[P, Iterator[Y]]:
        """Branch on the session at first iteration; `yield from` keeps send/throw/close intact."""

        @functools.wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> Iterator[Y]:
            session = _active
            if session is None:
                yield from func(*args, **kwargs)
                return
            key = session.enter(label)
            start_ns = time.perf_counter_ns()
            try:
                yield from func(*args, **kwargs)
            finally:
                session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)

        return inner

    @staticmethod
    def _decorate_sync[**P, R](func: Callable[P, R], label: str) -> Callable[P, R]:

        @functools.wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            session = _active
            if session is None:
                return func(*args, **kwargs)
            key = session.enter(label)
            start_ns = time.perf_counter_ns()
            try:
                return func(*args, **kwargs)
            finally:
                session.exit(key, wall_ns=time.perf_counter_ns() - start_ns)

        return inner

    def __call__[**P, R](self, func: Callable[P, R]) -> Callable[P, R]:
        """Decorate `func` with this span name, adding no collection policy.

        A generator's span covers its consumption: timing only its creation is the silent
        zero-length-span bug CPython fixed for its own decorators.
        """
        if inspect.iscoroutinefunction(func):
            # cast: the check proves `func` returns an Awaitable and the wrapper takes its place,
            # a ParamSpec decorator's return no checker can follow.
            return cast(
                "Callable[P, R]",
                Span._decorate_async(cast("Callable[P, Awaitable[R]]", func), self.name),
            )
        if inspect.isasyncgenfunction(func):
            # cast: the wrapper takes the async generator function's place under its signature.
            return cast("Callable[P, R]", Span._decorate_async_generator(func, self.name))
        if inspect.isgeneratorfunction(func):
            # cast: the wrapper takes the generator function's place under its signature.
            return cast("Callable[P, R]", Span._decorate_generator(func, self.name))
        return Span._decorate_sync(func, self.name)

    def __enter__(self) -> Span:
        if self.session is not None:
            raise RuntimeError(
                f"span {self.name!r} is already open on this instance; open a new span({self.name!r}) to nest "
                "or to enter it concurrently"
            )
        token = start(self.name)
        if token is not None:
            self.session, self.token, self.start_ns = token
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        session = self.session
        if session is None:
            return
        self.session = None
        finish((session, self.token, self.start_ns))

    def __init__(self, name: str) -> None:
        self.name = name
        self.session = None
        self.start_ns = 0
        self.token = 0


def span[**P, R](name: str | Callable[P, R]) -> Span | Callable[P, R]:
    """Mark a named block or function for an active `Profiler`, carrying no collection policy.

    Without an active profiler it does no clock, memory, marker, device or context-variable work.
    """
    if isinstance(name, str):
        return Span(name)
    return Span(getattr(name, "__qualname__", name.__class__.__qualname__))(name)

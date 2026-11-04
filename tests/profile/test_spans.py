import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn.profile import Feature, Profiler, RegionWindow, span
from krn.profile.bottleneck import annotate
from krn.profile.spans import active, deactivate, finish, start
from krn.profile.trace import BottleneckReport


@span("sync")
def sync(value: int) -> int:
    if value < 0:
        raise ValueError("negative")
    return value


@span("async")
async def asynchronous() -> int:
    await asyncio.sleep(0)
    return 1


@span("generator")
def generator() -> Iterator[int]:
    yield 1
    yield 2


@span("async_generator")
async def asynchronous_generator() -> AsyncIterator[int]:
    yield 3
    yield 4


async def consume() -> None:
    assert await asynchronous() == 1
    assert [value async for value in asynchronous_generator()] == [3, 4]


def test_all_decorator_shapes_inactive_active_and_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    assert active() is None and start("inert") is None
    finish(None)
    with span("inactive"):
        assert sync(2) == 2 and list(generator()) == [1, 2]
        asyncio.run(consume())
    with Profiler(features=Feature.SPANS) as profiler:
        assert sync(3) == 3 and list(generator()) == [1, 2]
        asyncio.run(consume())
        with pytest.raises(ValueError):
            sync(-1)
        with span("parent"):
            sync(4)
        key = profiler.enter("manual")
        profiler.exit(key, wall_ns=1)
        profiler.exit(key, wall_ns=1)
        deactivate(Profiler(features=Feature.SPANS))
        assert active() is profiler
    assert {row.name for row in profiler.result().summaries} >= {
        "sync",
        "generator",
        "async",
        "async_generator",
        "parent.sync",
    }
    assert not profiler.spans.frames
    # Feed the callbacks directly for empty and unmatched unwind paths.
    annotate._frames.set(None)
    annotate._close()
    annotate._on_unwind(sync.__code__, 0, ValueError())
    annotate._on_start(sync.__code__, 0)
    annotate._close()
    monkeypatch.setattr(annotate, "_codes", (sync.__code__,))
    with Profiler(features=Feature.SPANS) as profiler:
        annotate._on_start(sync.__code__, 0)
        annotate._on_unwind(sync.__code__, 0, ValueError())
    assert profiler.result().summaries
    monkeypatch.setattr(annotate, "_codes", (sync.__code__,))
    annotate._on_unwind(sync.__code__, 0, ValueError())


@given(st.integers(1, 12), st.integers(1, 8))
def test_span_nesting_paths_and_retention(depth: int, capacity: int) -> None:
    with Profiler(features=Feature.SPANS, max_spans=capacity) as profiler, ExitStack() as stack:
        for index in range(depth):
            stack.enter_context(span(str(index)))
    summaries = profiler.result().summaries
    assert len(summaries) == min(depth, capacity)
    assert profiler.result().dropped_spans == max(depth - capacity, 0)
    assert all(row.name == ".".join(str(i) for i in range(len(row.name.split(".")))) for row in summaries)
    assert not profiler.spans.frames


@given(st.integers(0, 100), st.integers(1, 100))
def test_region_binning_uses_narrowest_half_open_interval(start_ns: int, width: int) -> None:
    outer = RegionWindow(name="outer", start_ns=0, end_ns=500, wall_ns=500)
    inner = RegionWindow(name="inner", start_ns=start_ns, end_ns=start_ns + width, wall_ns=width)
    assert BottleneckReport._region_of(start_ns, (outer, inner)) == "inner"
    assert BottleneckReport._region_of(start_ns + width, (outer, inner)) == "outer"
    assert BottleneckReport._region_of(500, (outer, inner)) == "(outside regions)"


def test_reentering_one_span_instance_is_refused_and_the_outer_span_still_closes() -> None:
    with Profiler() as profiler:
        outer = span("reused")
        with outer, pytest.raises(RuntimeError, match="already open"), outer:
            pass
        with outer:
            pass
    assert [item.name for item in profiler.result().summaries] == ["reused", "reused"]

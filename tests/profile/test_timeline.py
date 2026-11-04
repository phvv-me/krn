"""The device timeline and `busy_ns` share one union of activity spans."""

from collections.abc import Sequence

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from krn.profile import DeviceTimeline, KernelTrace, RegionWindow, busy_ns
from krn.profile.trace import UNPOSITIONED, BottleneckReport, region_walls

_SPANS = st.lists(st.tuples(st.integers(0, 40), st.integers(0, 40), st.sampled_from("abc")), max_size=8)


def _kernels(spans: Sequence[tuple[int, int, str]]) -> list[KernelTrace]:
    return [KernelTrace(name=name, start_ns=start, end_ns=end) for start, end, name in spans]


@given(spans=_SPANS)
@example(spans=[(0, 5, "a"), (5, 9, "b"), (3, 4, "c"), (20, 20, "a"), (12, 8, "b")])
def test_the_busy_time_counts_each_instant_once_however_the_spans_overlap(
    spans: list[tuple[int, int, str]],
) -> None:
    covered = {tick for start, end, _ in spans for tick in range(start, end)}
    pairs = [(start, end) for start, end, _ in spans]

    busy = (busy_ns(iter(pairs)), DeviceTimeline.from_traces(_kernels(spans)).busy_ns)

    assert busy == (len(covered), len(covered))


@given(spans=_SPANS)
@example(spans=[(0, 5, "a"), (8, 9, "b"), (20, 30, "c"), (22, 24, "d")])
def test_the_listed_gaps_are_the_idle_stretches_named_by_the_activities_beside_them(
    spans: list[tuple[int, int, str]],
) -> None:
    real = [span for span in spans if span[1] > span[0]]
    covered = {tick for start, end, _ in real for tick in range(start, end)}
    first = min((start for start, _, _ in real), default=0)
    last = max((end for _, end, _ in real), default=0)

    timeline = DeviceTimeline.from_traces(_kernels(spans), top_gaps=99, min_gap_ns=0)

    assert (
        {tick for gap in timeline.gaps for tick in range(gap.start_ns, gap.end_ns)},
        {(gap.start_ns, gap.after) for gap in timeline.gaps} <= {(end, name) for _, end, name in real},
        {(gap.end_ns, gap.before) for gap in timeline.gaps} <= {(start, name) for start, _, name in real},
        (timeline.activities, timeline.span_ns),
    ) == (set(range(first, last)) - covered, True, True, (len(real), last - first))


@pytest.mark.parametrize(
    ("limits", "listed"),
    [
        pytest.param(
            {"min_gap_ns": 3},
            [(10, 15, "a", "b"), (30, 33, "c", "a")],
            id="a-window-under-the-floor-is-idle-but-unlisted",
        ),
        pytest.param(
            {"min_gap_ns": 1},
            [(10, 15, "a", "b"), (30, 33, "c", "a"), (20, 22, "b", "c")],
            id="the-longest-window-comes-first",
        ),
        pytest.param(
            {"min_gap_ns": 1, "top_gaps": 1},
            [(10, 15, "a", "b")],
            id="only-the-top-windows-are-kept",
        ),
    ],
)
def test_the_floor_and_the_top_choose_which_gaps_are_listed_longest_first(
    limits: dict[str, int], listed: list[tuple[int, int, str, str]]
) -> None:
    spans = [(0, 10, "a"), (15, 20, "b"), (22, 30, "c"), (33, 40, "a")]

    timeline = DeviceTimeline.from_traces(
        _kernels(spans), min_gap_ns=limits["min_gap_ns"], top_gaps=limits.get("top_gaps", 10)
    )

    assert [(g.start_ns, g.end_ns, g.after, g.before) for g in timeline.gaps] == listed


def test_wider_window_does_not_replace_narrower_window() -> None:
    windows = [
        RegionWindow(name="narrow", start_ns=0, end_ns=2, wall_ns=2),
        RegionWindow(name="wide", start_ns=0, end_ns=10, wall_ns=10),
    ]
    assert BottleneckReport._region_of(1, windows) == "narrow"


def test_repeated_region_wall_counts_nested_reentry_once_and_siblings_add() -> None:
    nested = [
        RegionWindow(name="r", start_ns=0, end_ns=10, wall_ns=10),
        RegionWindow(name="r", start_ns=2, end_ns=5, wall_ns=3),
    ]
    assert region_walls(nested) == {"r": 10}
    siblings = [
        RegionWindow(name="r", start_ns=0, end_ns=4, wall_ns=4),
        RegionWindow(name="r", start_ns=4, end_ns=9, wall_ns=5),
    ]
    assert region_walls(siblings) == {"r": 9}


@given(gaps=st.lists(st.integers(0, 5), max_size=6), widths=st.lists(st.integers(1, 9), min_size=1, max_size=6))
def test_disjoint_occurrences_of_one_name_add_exactly(gaps: list[int], widths: list[int]) -> None:
    windows, cursor = [], 0
    for gap, width in zip(gaps, widths, strict=False):
        cursor += gap
        windows.append(RegionWindow(name="r", start_ns=cursor, end_ns=cursor + width, wall_ns=width))
        cursor += width
    assert region_walls(windows).get("r", 0) == sum(window.wall_ns for window in windows)


def test_unpositioned_kernels_add_to_totals_but_not_to_busy_time() -> None:
    timed = KernelTrace(name="a", start_ns=0, end_ns=10)
    untimed = KernelTrace(name="b", end_ns=7, clock=UNPOSITIONED)
    timeline = DeviceTimeline.from_traces([timed, untimed])
    assert timeline.busy_ns == 10 and timeline.activities == 1 and timeline.unpositioned == 1
    assert "without timestamps" in str(timeline)
    assert timeline.__rich__().row_count == 6
    report = BottleneckReport.from_traces((), [timed, untimed], [])
    assert report.total_kernel_ns == 17 and report.device_busy_ns == 10
    assert DeviceTimeline.from_traces([untimed]).unpositioned == 1

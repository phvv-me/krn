"""Log sites intern on every field, refuse ids a record cannot name, and reload from JSON with the same ids."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn._internals.records import Message, Tag
from krn._internals.records import sites as record_sites
from krn.log.sites import LogSite, SiteTable

MESSAGES = [
    Message.of("i={} a={:.3f}", [Tag.INT, Tag.FLOAT64]),
    Message.of("{:>8}|{:d}", ["saxpy", Tag.BOOL]),
    Message.of("{{x}} {:c}", [Tag.UINT]),
    Message.of("plain"),
]


def add(
    table: SiteTable,
    *,
    logger: str = "saxpy",
    level: int = 10,
    file: str = "saxpy.py",
    line: int = 7,
    kernel: str = "saxpy",
    message: Message = MESSAGES[0],
    column: int = 0,
) -> LogSite:
    """Add a call; the defaults are one call at saxpy.py:7, and each test changes one field of it."""
    return table.add(logger=logger, level=level, file=file, line=line, kernel=kernel, message=message, column=column)


def test_the_same_call_keeps_its_id_and_new_calls_take_the_next() -> None:
    table = SiteTable()
    first = add(table)
    assert add(table) is first and first.id == 0
    assert add(table, message=MESSAGES[1]).id == 1
    assert add(table, line=8).id == 2
    assert add(table, file="other.py").id == 3
    assert len(table) == 4 and table[0] is first


@pytest.mark.parametrize(
    "changes",
    [
        {"level": 30},
        {"logger": "lane"},
        {"kernel": "other"},
        {"column": 4},
        {"line": 8},
        {"file": "other.py"},
        {"message": MESSAGES[2]},
    ],
)
def test_a_call_that_differs_in_one_field_is_another_site(changes: dict) -> None:
    table = SiteTable()
    original = add(table)
    other = add(table, **changes)
    assert other.id != original.id


def test_any_integer_level_is_a_site_level() -> None:
    table = SiteTable()
    assert add(table, level=-1000).level == -1000
    assert add(table, level=2**40).level == 2**40


def test_ids_past_the_u16_limit_are_refused_at_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record_sites, "MAX_SITE_ID", 0)
    table = SiteTable()
    add(table)
    with pytest.raises(OverflowError, match="exceeds the 1 sites a record can name"):
        add(table, line=8)


@given(
    st.lists(
        st.tuples(st.integers(1, 20), st.sampled_from(MESSAGES), st.sampled_from(["a.py", "b.py"]), st.integers(0, 3)),
        max_size=12,
    )
)
def test_table_reloads_with_the_same_ids(calls: list[tuple[int, Message, str, int]]) -> None:
    table = SiteTable()
    for line, message, file, column in calls:
        add(table, line=line, message=message, file=file, column=column)
    reloaded = SiteTable.loads(table.dumps(), LogSite.from_dict)
    assert len(reloaded) == len(table)
    assert [reloaded[identifier] for identifier in range(len(table))] == [table[i] for i in range(len(table))]
    assert reloaded.dumps() == table.dumps()

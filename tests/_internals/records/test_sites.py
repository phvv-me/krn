"""Sites intern on every field, refuse ids a record cannot name, and reload from JSON with the same ids."""

from dataclasses import dataclass

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn._internals.records import JSON, UNASSIGNED, Message, Site, SiteTable, Tag
from krn._internals.records import sites as sites_module
from krn._internals.records.jsonio import json_int, json_object, json_text


@dataclass(frozen=True, kw_only=True)
class Probe(Site):
    """A site with a tag and a message, so each field can differ between two calls."""

    tag: str
    message: Message

    def to_dict(self) -> dict[str, JSON]:
        return {**super().to_dict(), "tag": self.tag, "message": self.message.to_dict()}

    @classmethod
    def from_dict(cls, item: dict[str, JSON]) -> Probe:
        return cls(
            kernel=json_text(item, "kernel"),
            file=json_text(item, "file"),
            line=json_int(item, "line"),
            column=json_int(item, "column"),
            tag=json_text(item, "tag"),
            message=Message.from_dict(json_object(item["message"])),
        )


MESSAGES = [
    Message.of("i={} a={:.3f}", [Tag.INT, Tag.FLOAT64]),
    Message.of("{:>8}|{}", ["saxpy", Tag.BOOL]),
    Message.of("{{x}} {:c}", [Tag.UINT]),
    Message.of("plain"),
]


def probe(
    *,
    kernel: str = "k",
    file: str = "k.py",
    line: int = 7,
    column: int = 3,
    tag: str = "info",
    message: Message = MESSAGES[0],
    id: int = UNASSIGNED,
) -> Probe:
    return Probe(kernel=kernel, file=file, line=line, column=column, tag=tag, message=message, id=id)


def test_identical_sites_keep_one_id_and_new_ones_take_the_next() -> None:
    table: SiteTable[Probe] = SiteTable()
    first = table.intern(probe())
    assert table.intern(probe()) == first and first.id == 0
    assert table.intern(probe(line=8)).id == 1
    assert table[0] == first and len(table) == 2


@pytest.mark.parametrize(
    "other",
    [
        probe(tag="error"),
        probe(column=4),
        probe(kernel="other"),
        probe(file="other.py"),
        probe(line=8),
        probe(message=MESSAGES[1]),
    ],
)
def test_any_field_that_differs_makes_a_new_site(other: Probe) -> None:
    table: SiteTable[Probe] = SiteTable()
    original = table.intern(probe())
    interned = table.intern(other)
    assert interned.id != original.id
    assert table[original.id] == original


def test_the_id_is_not_part_of_identity() -> None:
    table: SiteTable[Probe] = SiteTable()
    table.intern(probe(line=1))
    assert table.intern(probe(line=2, id=99)).id == 1


def test_ids_past_the_u16_limit_are_refused_at_interning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sites_module, "MAX_SITE_ID", 1)
    table: SiteTable[Probe] = SiteTable()
    table.intern(probe(line=1))
    table.intern(probe(line=2))
    with pytest.raises(OverflowError, match="exceeds the 2 sites a record can name"):
        table.intern(probe(line=3))
    assert len(table) == 2


@given(
    st.lists(
        st.tuples(st.integers(1, 20), st.sampled_from(MESSAGES), st.sampled_from(["a.py", "b.py"]), st.integers(0, 3)),
        max_size=12,
    )
)
def test_table_reloads_with_the_same_ids(calls: list[tuple[int, Message, str, int]]) -> None:
    table: SiteTable[Probe] = SiteTable()
    for line, message, file, column in calls:
        table.intern(probe(line=line, message=message, file=file, column=column))
    reloaded = SiteTable.loads(table.dumps(), Probe.from_dict)
    assert len(reloaded) == len(table)
    assert [reloaded[identifier] for identifier in range(len(table))] == [table[i] for i in range(len(table))]
    assert reloaded.dumps() == table.dumps()

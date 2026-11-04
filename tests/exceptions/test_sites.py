"""Raise sites: argument tags as krn._internals.records tags and messages, argument mapping, interning on every field,
and JSON."""

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn._internals.records import JSON, Message, Tag
from krn.exceptions import ArgTag, ErrorSiteTable, RaiseSite

SCALARS = [Tag.INT, Tag.UINT, Tag.BOOL, Tag.FLOAT32, Tag.FLOAT64]
TEXT = st.text(alphabet="abc _={}", max_size=4)
MESSAGES = [
    Message.of("n={}", [Tag.INT]),
    Message.of("{:.2f}|{:>4}", [Tag.FLOAT64, Tag.UINT]),
    Message.of("{{{}}} {}", [Tag.BOOL, "text"]),
]
ARG_TAGS = st.one_of(st.sampled_from(SCALARS), TEXT.map(Message.constant), st.sampled_from(MESSAGES))


def site(*tags: ArgTag) -> RaiseSite:
    return RaiseSite(id=0, kernel="k", file="k.py", line=3, class_id=7, arg_tags=tags)


def test_a_message_argument_renders_its_scalars_in_the_printf_subset() -> None:
    raised = site(Tag.INT, Message.constant("tag"), Message.of("n={}", [Tag.BOOL]), Message.of("{:.2f}", [Tag.FLOAT64]))
    assert raised.kinds == (Tag.INT, Tag.BOOL, Tag.FLOAT64)
    assert raised.arguments([5, False, 2.5]) == (5, "tag", "n=0", "2.50")


def test_a_constant_message_keeps_its_braces_as_given() -> None:
    assert site(Message.constant("{x}")).arguments([]) == ("{x}",)
    assert site(Message.constant("{x}")).kinds == ()


def test_a_str_constant_tag_takes_no_slot_so_it_is_refused() -> None:
    with pytest.raises(ValueError, match=r"Message\.constant"):
        site(Tag.STR_CONSTANT)


def test_identical_sites_share_an_id_and_new_ones_take_the_next() -> None:
    table = ErrorSiteTable()
    first = table.add(kernel="k", file="k.py", line=3, class_id=1, arg_tags=(Tag.INT,))
    again = table.add(kernel="k", file="k.py", line=3, class_id=1, arg_tags=(Tag.INT,))
    other = table.add(kernel="k", file="k.py", line=4, class_id=1, arg_tags=(Tag.INT,))
    assert first == again and first.id == 0 and other.id == 1
    assert table[1] is other


@pytest.mark.parametrize(
    "other",
    [
        RaiseSite(kernel="k", file="k.py", line=3, column=2, class_id=7, arg_tags=(Tag.INT,)),
        RaiseSite(kernel="k", file="k.py", line=3, class_id=2, arg_tags=(Tag.INT,)),
        RaiseSite(kernel="k", file="k.py", line=3, class_id=7, arg_tags=(Tag.UINT,)),
        RaiseSite(kernel="k", file="k.py", line=3, class_id=7, arg_tags=(Message.constant("x"),)),
        RaiseSite(kernel="other", file="k.py", line=3, class_id=7, arg_tags=(Tag.INT,)),
    ],
)
def test_a_site_that_differs_in_any_field_is_a_new_site(other: RaiseSite) -> None:
    table = ErrorSiteTable()
    original = table.intern(site(Tag.INT))
    interned = table.intern(other)
    assert interned.id != original.id and table[original.id] == original


@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(TEXT, st.integers(1, 9999), st.lists(ARG_TAGS, max_size=4)), max_size=6))
def test_table_round_trips_through_json_with_the_same_ids(entries: list[tuple[str, int, list[ArgTag]]]) -> None:
    table = ErrorSiteTable()
    added = [
        table.add(kernel=kernel, file="f.py", line=line, class_id=index, arg_tags=tuple(tags))
        for index, (kernel, line, tags) in enumerate(entries)
    ]
    loaded = ErrorSiteTable.loads(table.dumps(), RaiseSite.from_dict)
    assert loaded.dumps() == table.dumps()
    assert [loaded[site.id] for site in added] == added


@pytest.mark.parametrize(
    "tag,message",
    [
        ({"other": 1}, "not an argument tag"),
        ([1], "not an argument tag"),
        ("str-constant", "no slot"),
    ],
)
def test_a_tag_that_is_not_a_known_shape_is_refused(tag: JSON, message: str) -> None:
    item: dict[str, JSON] = {"kernel": "k", "file": "f", "line": 1, "column": 0, "class_id": 1, "arg_tags": [tag]}
    with pytest.raises(ValueError, match=message):
        RaiseSite.from_dict(item)


def test_a_site_with_a_missing_key_names_the_key() -> None:
    with pytest.raises(KeyError, match="class_id"):
        RaiseSite.from_dict({"kernel": "k", "file": "f", "line": 1, "column": 0, "arg_tags": []})


def test_the_json_of_a_site_names_each_tag_by_its_kind() -> None:
    raised = site(Tag.INT, Message.constant("t"))
    encoded = json.loads(json.dumps(raised.to_dict()))
    assert encoded["arg_tags"] == ["int", {"format": "t", "fields": []}]

"""The JSON readers refuse a value of the wrong shape or type, and the error names the key or the value."""

import pytest

from krn._internals.records.jsonio import (
    JSON,
    json_bool,
    json_int,
    json_list,
    json_object,
    json_optional_int,
    json_text,
)


@pytest.mark.parametrize("value", [[1], "x", 3, None, True])
def test_a_table_entry_that_is_not_an_object_is_refused(value: JSON) -> None:
    with pytest.raises(ValueError, match="expected a JSON object"):
        json_object(value)


def test_an_object_passes_through_unchanged() -> None:
    item: dict[str, JSON] = {"key": 1}
    assert json_object(item) is item


@pytest.mark.parametrize("value", ["text", 1, None, {"nested": 1}])
def test_a_key_that_is_not_an_array_is_refused_by_name(value: JSON) -> None:
    item: dict[str, JSON] = {"fields": value}
    with pytest.raises(ValueError, match="'fields' must be a JSON array"):
        json_list(item, "fields")


def test_a_key_that_is_an_array_is_returned() -> None:
    item: dict[str, JSON] = {"fields": [1, "a"]}
    assert json_list(item, "fields") == [1, "a"]


@pytest.mark.parametrize("value", [1, None, ["a"]])
def test_a_key_that_is_not_text_is_refused_by_name(value: JSON) -> None:
    item: dict[str, JSON] = {"format": value}
    with pytest.raises(ValueError, match="'format' must be a string"):
        json_text(item, "format")


def test_text_passes_through() -> None:
    assert json_text({"format": "x={}"}, "format") == "x={}"


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_a_key_that_is_not_an_integer_is_refused_by_name(value: JSON) -> None:
    item: dict[str, JSON] = {"width": value}
    with pytest.raises(ValueError, match="'width' must be an integer"):
        json_int(item, "width")


def test_an_integer_passes_through_including_negative_and_zero() -> None:
    assert json_int({"line": -3}, "line") == -3 and json_int({"line": 0}, "line") == 0


def test_an_optional_integer_is_none_or_an_integer() -> None:
    assert json_optional_int({"width": None}, "width") is None
    assert json_optional_int({"width": 4}, "width") == 4
    with pytest.raises(ValueError, match="'width' must be an integer"):
        json_optional_int({"width": "4"}, "width")


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_a_key_that_is_not_a_boolean_is_refused_by_name(value: JSON) -> None:
    item: dict[str, JSON] = {"zero": value}
    with pytest.raises(ValueError, match="'zero' must be true or false"):
        json_bool(item, "zero")


def test_a_boolean_passes_through() -> None:
    assert json_bool({"zero": True}, "zero") is True and json_bool({"zero": False}, "zero") is False

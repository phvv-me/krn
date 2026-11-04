"""The Errors settings group: defaults, scoped configuration, validation, and the metadata every group carries."""

from dataclasses import fields

import pytest

import krn
from krn.compiler.errors import SettingError
from krn.runtime.settings import BACKENDS, Errors, Settings


def test_defaults_are_256_records_flag_policy() -> None:
    assert Errors() == Errors(capacity=256, policy="flag")
    assert Settings().errors == Errors()
    assert krn.config.errors.capacity == 256 and krn.config.errors.policy == "flag"


def test_configure_replaces_the_group_and_its_fields_within_a_scope() -> None:
    with krn.configure(errors=krn.Errors(capacity=8, policy="abort")):
        assert krn.config.errors.capacity == 8 and krn.config.errors.policy == "abort"
    with krn.configure(**{"errors.capacity": 4}):
        assert krn.config.errors.capacity == 4 and krn.config.errors.policy == "flag"
    assert krn.config.errors.capacity == 256


def test_a_group_cannot_be_replaced_by_a_plain_value() -> None:
    with pytest.raises(SettingError, match="Errors"):
        krn.configure(errors="flag")


@pytest.mark.parametrize(
    "changes",
    [{"capacity": 0}, {"capacity": -1}, {"capacity": True}, {"capacity": 1.5}, {"policy": "ignore"}, {"policy": None}],
)
def test_invalid_values_name_the_setting(changes: dict[str, int | float | bool | str | None]) -> None:
    with pytest.raises(SettingError, match="errors\\."):
        krn.configure(**{f"errors.{key}": value for key, value in changes.items()})


def test_every_field_names_its_backends_like_the_other_groups() -> None:
    assert {item.name for item in fields(Errors)} == {"capacity", "policy"}
    for item in fields(Errors):
        assert set(item.metadata["backends"]) == set(BACKENDS)
    assert "errors" in {item.name for item in fields(Settings)}

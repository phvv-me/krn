"""The Log settings group: validated, reachable from krn.config and krn.configure, and honored per backend."""

from collections.abc import Callable
from dataclasses import fields

import pytest

from krn.compiler.errors import SettingError, TargetError
from krn.facts.targets import Target
from krn.runtime import settings as state
from krn.runtime.settings import BACKENDS, PRINTF_BACKENDS, Log, Settings, config, configure


@pytest.fixture
def restored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state, "_current", state.current_settings())


def test_defaults_are_the_ring() -> None:
    assert (Log().capacity, Log().args, Log().transport) == (4096, 8, "ring")
    assert Settings().log == Log()


@pytest.mark.parametrize(
    "build,name",
    [
        (lambda: Log(capacity=0), "log.capacity"),
        (lambda: Log(capacity=True), "log.capacity"),
        (lambda: Log(args=-1), "log.args"),
    ],
)
def test_invalid_values_name_the_setting(build: Callable[[], Log], name: str) -> None:
    with pytest.raises(SettingError, match=name):
        build()


def test_group_and_fields_change_copy_on_write(restored: None) -> None:
    before = state.current_settings()
    configure(log=Log(capacity=16))
    assert config.log.capacity == 16 and before.log.capacity == 4096
    configure(**{"log.args": 4})
    assert state.current_settings().log == Log(capacity=16, args=4)
    with pytest.raises(SettingError, match=r"log\.transport"):
        configure(**{"log.transport": "bogus"})
    with pytest.raises(SettingError, match="Log"):
        configure(log=4)


def test_printf_is_refused_on_metal_and_honored_elsewhere() -> None:
    printf = Settings(log=Log(transport="printf"))
    with pytest.raises(TargetError, match="Metal has no device printf") as caught:
        printf.validate(Target.of("apple_m4"))
    assert "set log.transport='ring'" in str(caught.value)
    for arch in ("sm_90a", "avx512"):
        printf.validate(Target.of(arch))
    Settings(log=Log(transport="ring")).validate(Target.of("apple_m4"))


def test_log_metadata_is_complete() -> None:
    assert set(PRINTF_BACKENDS) <= set(BACKENDS)
    for item in fields(Log):
        assert set(item.metadata["backends"]) <= set(BACKENDS)

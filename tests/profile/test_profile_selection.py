"""Automatic spans follow owned function bodies without importing shared wrappers."""

import importlib
import sys
from collections import Counter
from pathlib import Path
from types import CodeType, ModuleType

import pytest

from krn.profile import Profiler
from krn.profile.bottleneck import annotate
from krn.profile.bottleneck.owned import module_codes, owned_codes

_FOREIGN = """
from krn.profile.spans import span

@span("manual_foreign")
def unrelated():
    return 7

class Foreign:
    def inherited(self):
        return 8
"""

_OWNED = """
import _profile_foreign as foreign
from krn.profile.spans import span

foreign_alias = foreign.unrelated
foreign_class = foreign.Foreign
reexport = span("manual_reexport")(foreign.unrelated)
reexport.__module__ = __name__

def plain():
    return 1

alias = plain

@span("manual_first")
def first():
    return 2

@span("manual_second")
def second():
    return 3

@span
def unlabelled():
    return 9

class Worker(foreign.Foreign):
    borrowed = foreign.unrelated
    borrowed_static = staticmethod(foreign.unrelated)
    borrowed_class = classmethod(foreign.unrelated)

    def regular(self):
        return 4

    @staticmethod
    @span("manual_static")
    def static():
        return 5

    @classmethod
    @span("manual_classed")
    def classed(cls):
        return 6

def nested():
    def values():
        yield 1
        yield 2
    return sum((lambda value: value)(value) for value in values())
"""


@pytest.fixture
def modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    (tmp_path / "_profile_foreign.py").write_text(_FOREIGN, encoding="utf-8")
    (tmp_path / "_profile_owned.py").write_text(_OWNED, encoding="utf-8")
    monkeypatch.syspath_prepend(tmp_path)
    for name in ("_profile_foreign", "_profile_owned"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    owned = importlib.import_module("_profile_owned")
    return owned, owned.foreign


def test_owned_bodies_descriptors_and_foreign_wrappers(modules: tuple[ModuleType, ModuleType]):
    owned, foreign = modules
    assert owned.first.__code__ is owned.second.__code__ is foreign.unrelated.__code__
    selected = owned_codes(owned)
    assert len(selected) == len(set(selected))
    assert {code.co_qualname for code in selected} == {
        "plain",
        "first",
        "second",
        "unlabelled",
        "Worker.regular",
        "Worker.static",
        "Worker.classed",
        "nested",
    }


def test_automatic_spans_preserve_explicit_labels(modules: tuple[ModuleType, ModuleType]):
    owned, foreign = modules
    with Profiler(features=Profiler.Feature.SPANS, auto=(owned.__name__,)) as profiler:
        assert [
            owned.plain(),
            owned.first(),
            owned.second(),
            owned.Worker().regular(),
            owned.Worker.static(),
            owned.Worker.classed(),
            foreign.unrelated(),
            owned.Worker().inherited(),
            owned.unlabelled(),
        ] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert Counter(row.name for row in profiler.result().summaries) == Counter(
        {
            "plain": 1,
            "manual_first.first": 1,
            "manual_first": 1,
            "manual_second.second": 1,
            "manual_second": 1,
            "Worker.regular": 1,
            "manual_static.Worker.static": 1,
            "manual_static": 1,
            "manual_classed.Worker.classed": 1,
            "manual_classed": 1,
            "manual_foreign": 1,
            "unlabelled": 1,
            "unlabelled.unlabelled": 1,
        }
    )
    assert not annotate.enabled_codes()
    assert not profiler.spans.frames
    assert profiler.result().dropped_spans == 0


@pytest.mark.parametrize("generated", ["<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>"])
def test_nested_functions_survive_generated_parent(modules: tuple[ModuleType, ModuleType], generated: str):
    owned, _ = modules
    code = owned.nested.__code__
    # Python 3.12+ inlines list/set/dict comprehensions; cover their earlier code form too.
    owned.nested.__code__ = code.replace(
        co_consts=tuple(
            item.replace(co_name=generated) if isinstance(item, CodeType) and item.co_name == "<genexpr>" else item
            for item in code.co_consts
        )
    )
    selected = module_codes((owned.__name__,))
    assert {item.co_name for item in selected} >= {"nested", "values", "<lambda>"}
    assert generated not in {item.co_name for item in selected}
    with Profiler(features=Profiler.Feature.SPANS, auto=(owned.__name__,)) as profiler:
        assert owned.nested() == 3
    # Named generators still emit an active segment per resume, including final return.
    assert Counter(row.name for row in profiler.result().summaries) == Counter(
        {
            "nested": 1,
            "nested.nested.<locals>.values": 3,
            "nested.nested.<locals>.<genexpr>.<lambda>": 2,
        }
    )


def test_failed_automatic_enable_leaves_no_tool_or_event_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = module_codes(["krn.profile.bottleneck.owned"])
    real = sys.monitoring.set_local_events
    calls: list[CodeType] = []

    def fail_second(tool: int, code: CodeType, events: int) -> None:
        calls.append(code)
        if len(calls) == 2:
            raise RuntimeError("injected registration failure")
        real(tool, code, events)

    monkeypatch.setattr(sys.monitoring, "set_local_events", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        annotate.enable_auto(codes)
    monkeypatch.undo()
    assert sys.monitoring.get_tool(annotate._tool_id) is None and annotate.enabled_codes() == ()
    annotate.enable_auto(codes)
    assert annotate.enabled_codes() == tuple(dict.fromkeys(codes))
    annotate.disable_auto()
    assert sys.monitoring.get_tool(annotate._tool_id) is None

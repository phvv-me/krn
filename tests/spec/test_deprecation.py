"""The deprecation cross-check: the `warnings.deprecated` decorator and the `.. deprecated::` directive must agree."""

from pathlib import Path

import griffe
import pytest

from scripts.build_spec import Callout, SpecError, _deprecation, read_docstring


def _object(tmp_path: Path, source: str) -> tuple[griffe.Object, tuple[Callout, ...]]:
    (tmp_path / "shim.py").write_text(source)
    module = griffe.load("shim", search_paths=[tmp_path], allow_inspection=False)
    obj = module["old"]
    _, callouts, _ = read_docstring(obj.path, obj.docstring.value)
    return obj, callouts


DECORATED = '''
import warnings


@warnings.deprecated("Use `new`.")
def old():
    """Old.

    .. deprecated:: 0.4.0
       Use `new`. Removed in 0.6.0.
    """
'''

DIRECTIVE_ONLY = '''
def old():
    """Old.

    .. deprecated:: 0.4.0
       Use `new`.
    """
'''

DECORATOR_ONLY = '''
import warnings


@warnings.deprecated("Use `new`.")
def old():
    """Old."""
'''

NEITHER = '''
def old():
    """Old."""
'''

TWO_DIRECTIVES = '''
import warnings


@warnings.deprecated("Use `new`.")
def old():
    """Old.

    .. deprecated:: 0.4.0
       Use `new`.

    .. deprecated:: 0.5.0
       Still `new`.
    """
'''


def test_a_decorator_and_a_directive_agree_and_the_message_is_the_decorators(tmp_path: Path) -> None:
    obj, callouts = _object(tmp_path, DECORATED)
    assert _deprecation(obj, callouts) == "Use `new`."


def test_neither_marks_nothing_deprecated(tmp_path: Path) -> None:
    obj, callouts = _object(tmp_path, NEITHER)
    assert _deprecation(obj, callouts) is None


@pytest.mark.parametrize("source", [DIRECTIVE_ONLY, DECORATOR_ONLY, TWO_DIRECTIVES])
def test_a_mismatch_raises_and_names_the_object(tmp_path: Path, source: str) -> None:
    obj, callouts = _object(tmp_path, source)
    with pytest.raises(SpecError, match=r"shim\.old"):
        _deprecation(obj, callouts)

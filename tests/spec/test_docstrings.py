"""The docstring reader: callout directives, the Google sections the spec keeps, and the Supported on block."""

import re
import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.build_spec import Requirement, Run, SpecError, read_docstring

VERSION = st.lists(st.integers(0, 99), min_size=2, max_size=4).map(lambda parts: ".".join(map(str, parts)))
WORDS = st.from_regex(r"[a-z]+( [a-z]+){0,4}", fullmatch=True)
NOT_A_VERSION = st.text(alphabet=string.ascii_letters + " .", min_size=1).filter(
    lambda text: not re.fullmatch(r"\d+(\.\d+)+", text)
)
SUPPORTED = """Summary.

Supported on (generated):
    NVIDIA: sm_80+ composed (13 instructions), 8 of 16 operand types native or refused
    Apple: apple_m1+ composed (13 instructions), 8 of 16 operand types native or refused
    AMD: gfx942+ composed (136 instructions), 4 of 16 operand types refused
    CPU: avx2+ composed (136 instructions), 4 of 16 operand types refused
Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; Apple from apple_m1: Apple7.

"""


@pytest.mark.parametrize("kind", ["versionadded", "versionchanged", "deprecated"])
def test_each_directive_kind_is_a_callout(kind: str) -> None:
    _, callouts, _ = read_docstring("f", f"Summary.\n\n.. {kind}:: 0.2.0\n   Text of the callout.\n")
    assert [(c.kind, c.version, c.text) for c in callouts] == [(kind, "0.2.0", "Text of the callout.")]


def test_directive_text_runs_over_its_indented_lines_and_stops_at_a_blank_line() -> None:
    text = "Summary.\n\n.. versionchanged:: 0.3.0\n   `x` now raises\n   TargetError on Apple.\n\nDescription after."
    doc, callouts, _ = read_docstring("f", text)
    assert callouts[0].text == "`x` now raises TargetError on Apple."
    assert doc.description == "Description after."


def test_directives_leave_the_prose_and_its_sections_alone() -> None:
    text = "Summary.\n\n.. versionadded:: 0.0.1\n\nArgs:\n    a: first.\n\nReturns:\n    The result.\n"
    doc, callouts, _ = read_docstring("f", text)
    assert doc.summary == "Summary."
    assert [(p.name, p.description) for p in doc.parameters] == [("a", "first.")]
    assert doc.returns == "The result."
    assert len(callouts) == 1


@given(version=VERSION)
def test_any_dotted_version_is_kept_as_written(version: str) -> None:
    _, callouts, _ = read_docstring("f", f"Summary.\n\n.. versionadded:: {version}\n")
    assert callouts[0].version == version


@given(version=NOT_A_VERSION)
def test_a_version_that_is_not_dotted_digits_raises(version: str) -> None:
    with pytest.raises(SpecError, match="dotted version"):
        read_docstring("f", f"Summary.\n\n.. versionadded:: {version}\n")


@given(lines=st.lists(WORDS, min_size=1, max_size=5))
def test_a_multi_line_directive_joins_its_lines_with_spaces(lines: list[str]) -> None:
    body = "\n".join(f"   {line}" for line in lines)
    _, callouts, _ = read_docstring("f", f"Summary.\n\n.. deprecated:: 0.4.0\n{body}\n")
    assert callouts[0].text == " ".join(lines)


def test_the_supported_block_is_a_vendor_table_and_its_requirements() -> None:
    _, _, supported = read_docstring("f", SUPPORTED)
    assert supported is not None
    assert [vendor.vendor for vendor in supported.vendors] == ["NVIDIA", "Apple", "AMD", "CPU"]
    assert supported.vendors[0].runs == (
        Run(targets="sm_80+", reading="composed (13 instructions), 8 of 16 operand types native or refused"),
    )
    assert supported.requires == (
        Requirement(vendor="NVIDIA", target="sm_80", text="CUDA 11.0, driver 450.80.02, PTX ISA 7.0"),
        Requirement(vendor="Apple", target="apple_m1", text="Apple7"),
    )


def test_a_semicolon_inside_brackets_does_not_split_a_vendor_row() -> None:
    row = "NVIDIA: sm_80+ composed (2 instructions), 1 of 2 operand types native (a native; b refused)"
    text = f"Summary.\n\nSupported on (generated):\n    {row}\n"
    _, _, supported = read_docstring("f", text)
    assert supported is not None
    assert len(supported.vendors[0].runs) == 1


def test_a_block_that_says_one_reading_for_every_target_is_everywhere() -> None:
    _, _, supported = read_docstring("f", "Summary.\n\nSupported on (generated):\n    Native on every target\n")
    assert supported is not None
    assert supported.everywhere == "Native on every target"
    assert supported.vendors == ()


def test_an_unknown_line_in_the_supported_block_raises() -> None:
    with pytest.raises(SpecError, match="unknown line"):
        read_docstring("f", "Summary.\n\nSupported on (generated):\n    Somewhere: sm_80 native\n")


def test_a_section_the_spec_does_not_keep_raises() -> None:
    with pytest.raises(SpecError, match="does not keep the `yields` section"):
        read_docstring("f", "Summary.\n\nYields:\n    int: a number.\n")

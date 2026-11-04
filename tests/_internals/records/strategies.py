"""Hypothesis strategies for the printf subset: a spec the contract accepts for a tag, and values of that tag."""

import string

from hypothesis import strategies as st

from krn._internals.records import Argument, Scalar, Tag

FLOATS = (Tag.FLOAT32, Tag.FLOAT64)
INTEGERS = (Tag.INT, Tag.UINT, Tag.BOOL)
LETTERS = {
    Tag.INT: "diuxXoc",
    Tag.UINT: "diuxXoc",
    Tag.BOOL: "diuxXo",
    Tag.FLOAT32: "fFeEgG",
    Tag.FLOAT64: "fFeEgG",
    Tag.STR_CONSTANT: "s",
}
SIGNED_LETTERS = "dieEfFgG"
TEXT = st.text(string.ascii_letters + " ", max_size=6)


@st.composite
def spec_cases(draw: st.DrawFn) -> tuple[Argument, str]:
    """An argument (a tag, or a constant text) and a spec text the contract accepts for it."""
    argument: Argument = draw(st.sampled_from([*INTEGERS, *FLOATS]) | TEXT)
    tag = argument if isinstance(argument, Tag) else Tag.STR_CONSTANT
    letters = [*LETTERS[tag], *([""] if tag not in FLOATS else [])]
    letter = draw(st.sampled_from(letters))
    numeric = tag is not Tag.STR_CONSTANT
    align = draw(st.sampled_from(["", "<", ">"]))
    signed = tag in (Tag.INT, *FLOATS) and (letter in SIGNED_LETTERS or (letter == "" and tag is Tag.INT))
    sign = draw(st.sampled_from(["", "+", " "])) if signed else ""
    zero = numeric and not align and letter != "c" and draw(st.booleans())
    width = draw(st.none() | st.integers(1, 12))
    precision = draw(st.none() | st.integers(0, 8)) if tag in FLOATS or not numeric else None
    text = f"{align}{sign}{'0' if zero else ''}{'' if width is None else width}"
    text += "" if precision is None else f".{precision}"
    return argument, text + letter


def values_for(tag: Tag, letter: str) -> st.SearchStrategy[Scalar]:
    """Values a field of this tag and letter takes: `c` takes codes 0 to 127, and floats are never NaN."""
    if tag is Tag.BOOL:
        return st.booleans()
    if tag in FLOATS:
        return st.floats(allow_nan=False)
    if letter == "c":
        return st.integers(0, 127)
    if tag is Tag.INT:
        return st.integers(-(2**63), 2**63 - 1)
    return st.integers(0, 2**64 - 1)

"""Messages: constant text and typed fields, rendered in the printf subset every backend can print.

The contract, identical on the host and on the device. Integers take `d i u x X o c`: a signed value is read as its
64-bit two's complement for `u x X o`, a bool is 0 or 1, and `c` takes codes 0 to 127 only. Floats take `f F e E g G`
and have no default, so every float states its precision. Constant text takes `s`. Flags are `<` (left, printf `-`),
`+` or space, and `0`. Conversions (`!r`, `!a`, `!s`) and `b` have no printf form and are refused.
"""

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import count
from math import isfinite
from string import Formatter
from string.templatelib import Template
from typing import Self

from .jsonio import JSON, json_bool, json_int, json_list, json_object, json_optional_int, json_text
from .tags import WORD_MASK, Scalar, Tag

type Argument = Tag | str


class MessageFormatError(Exception):
    """A message or spec uses a form outside the printf subset, and the message names the rule and the form to use."""


_LETTERS = {
    Tag.INT: "diuxXoc",
    Tag.UINT: "diuxXoc",
    Tag.BOOL: "diuxXo",
    Tag.FLOAT32: "fFeEgG",
    Tag.FLOAT64: "fFeEgG",
    Tag.STR_CONSTANT: "s",
}
_DEFAULT_LETTER = {Tag.INT: "d", Tag.UINT: "u", Tag.BOOL: "u", Tag.FLOAT32: "", Tag.FLOAT64: "", Tag.STR_CONSTANT: "s"}
_SIGNED_TAGS = frozenset((Tag.INT, Tag.FLOAT32, Tag.FLOAT64))
_SIGNED_LETTERS = "dieEfFgG"
_NUMERIC_TAGS = frozenset((Tag.INT, Tag.UINT, Tag.BOOL, Tag.FLOAT32, Tag.FLOAT64))
_FLOAT_TAGS = frozenset((Tag.FLOAT32, Tag.FLOAT64))
_PYTHON_LETTER = {"i": "d", "u": "d"}
_SPEC = re.compile(
    r"(?P<align>[<>])?(?P<sign>[+ ])?(?P<zero>0)?(?P<width>\d+)?(?:\.(?P<precision>\d+))?(?P<type>[A-Za-z])?"
)


@dataclass(frozen=True, kw_only=True)
class Spec:
    """A format spec: alignment `<` or `>`, a sign `+` or space, `0` padding, width, precision and a printf type.

    The spec is checked against its tag by Field, since the same letter means different things per tag.
    """

    align: str = ""
    sign: str = ""
    zero: bool = False
    width: int | None = None
    precision: int | None = None
    type: str = ""

    def __post_init__(self) -> None:
        if self.align not in ("", "<", ">") or self.sign not in ("", "+", " "):
            raise MessageFormatError(f"alignment {self.align!r} or sign {self.sign!r} is not in the printf subset")
        if (self.width is not None and self.width < 0) or (self.precision is not None and self.precision < 0):
            raise MessageFormatError("a width or precision cannot be negative")

    @classmethod
    def parse(cls, source: str) -> Self:
        """Read a spec's syntax; whether its parts suit the field's tag is checked when the field is built."""
        if (found := _SPEC.fullmatch(source)) is None:
            raise MessageFormatError(
                f"format spec {source!r} is not supported: write [<|>] alignment, a sign (+ or space), 0 padding, a "
                "width, .precision and a printf type; fill characters, ^, #, grouping, - and z are not"
            )
        return cls(
            align=found["align"] or "",
            sign=found["sign"] or "",
            zero=found["zero"] is not None,
            width=None if found["width"] is None else int(found["width"]),
            precision=None if found["precision"] is None else int(found["precision"]),
            type=found["type"] or "",
        )

    @classmethod
    def from_dict(cls, item: dict[str, JSON]) -> Self:
        return cls(
            align=json_text(item, "align"),
            sign=json_text(item, "sign"),
            zero=json_bool(item, "zero"),
            width=json_optional_int(item, "width"),
            precision=json_optional_int(item, "precision"),
            type=json_text(item, "type"),
        )

    def to_dict(self) -> dict[str, JSON]:
        return {
            "align": self.align,
            "sign": self.sign,
            "zero": self.zero,
            "width": self.width,
            "precision": self.precision,
            "type": self.type,
        }


@dataclass(frozen=True, kw_only=True)
class Field:
    """One placeholder: its position among the fields, its tag, its spec, and a str constant's text."""

    position: int
    tag: Tag
    spec: Spec = Spec()
    text: str = ""

    def __post_init__(self) -> None:
        spec, allowed = self.spec, _LETTERS[self.tag]
        if self.tag in _FLOAT_TAGS and not spec.type:
            raise MessageFormatError("a float needs a printf type: write {:f}, {:e} or {:g}, or F, E or G")
        if spec.type == "b":
            raise MessageFormatError("binary b has no printf form; write x or o for hex or octal, or d for decimal")
        if spec.type and spec.type not in allowed:
            raise MessageFormatError(f"type {spec.type!r} is not allowed for {self.tag}; allowed: {allowed}")
        if spec.sign and (self.tag not in _SIGNED_TAGS or self.letter not in _SIGNED_LETTERS):
            raise MessageFormatError(f"a sign is only for signed numbers, not {self.tag} with {self.letter!r}")
        if spec.zero and (self.tag not in _NUMERIC_TAGS or spec.align or self.letter == "c"):
            raise MessageFormatError("0 padding is only for numbers without an alignment or c")
        if spec.precision is not None and self.tag not in _FLOAT_TAGS | {Tag.STR_CONSTANT}:
            raise MessageFormatError(f"precision is only for floats and str-constant, not {self.tag}")

    @property
    def letter(self) -> str:
        """The printf type the field renders with: its spec's, or the tag's default."""
        return self.spec.type or _DEFAULT_LETTER[self.tag]

    def _format(self, *, zero: bool) -> str:
        flags = ("-" if self.spec.align == "<" else "") + self.spec.sign + ("0" if zero else "")
        width = "" if self.spec.width is None else str(self.spec.width)
        precision = "" if self.spec.precision is None else f".{self.spec.precision}"
        return f"%{flags}{width}{precision}{_PYTHON_LETTER.get(self.letter, self.letter)}"

    @cached_property
    def _pattern(self) -> str:
        return self._format(zero=self.spec.zero)

    @cached_property
    def _spaced(self) -> str:
        """The pattern with the zero flag dropped: C pads inf and nan with spaces, Python would pad them with zeros."""
        return self._format(zero=False)

    def render(self, value: Scalar | str) -> str:
        """The field's text: a str constant passes its text, a scalar its value, rendered as the printf type says."""
        if self.tag is Tag.STR_CONSTANT:
            return self._pattern % str(value)
        if self.tag in _FLOAT_TAGS:
            number = float(value)
            return (self._pattern if isfinite(number) else self._spaced) % number
        integer = int(value) & WORD_MASK if self.letter in "xXou" else int(value)
        if self.letter == "c" and not 0 <= integer <= 127:
            raise MessageFormatError(f"%c takes a code from 0 to 127, not {integer}; write {{:d}} for the number")
        return self._pattern % integer

    def to_dict(self) -> dict[str, JSON]:
        return {"position": self.position, "tag": self.tag.value, "spec": self.spec.to_dict(), "text": self.text}

    @classmethod
    def from_dict(cls, item: dict[str, JSON]) -> Self:
        return cls(
            position=json_int(item, "position"),
            tag=Tag(json_text(item, "tag")),
            spec=Spec.from_dict(json_object(item["spec"])),
            text=json_text(item, "text"),
        )


@dataclass(frozen=True, kw_only=True)
class Message:
    """A parsed message: `format` is the constant text with one `{}` per field, escaped for str.format."""

    format: str
    fields: tuple[Field, ...]

    @classmethod
    def of(cls, source: ast.JoinedStr | Template | str, args: Sequence[Argument] = ()) -> Message:
        """Parse an f-string, a t-string, or a `{}` text whose placeholders take `args` in order.

        For an f-string or t-string, `args` describes its replacement fields in order; for a text, `args` are its
        positional arguments, numbered by `{}` or `{0}`, `{1}`, ... A str argument is a constant whose text is kept.
        An argument that no placeholder uses is an error.
        """
        match source:
            case ast.JoinedStr():
                pieces = _from_fstring(source)
            case Template():
                pieces = _from_template(source)
            case _:
                pieces = _from_braces(source)
        return _build(pieces, args)

    @classmethod
    def constant(cls, text: str) -> Self:
        """A message that is only `text`, with no fields, so it renders as given whatever braces it holds."""
        return cls(format=_escape(text), fields=())

    @cached_property
    def scalars(self) -> tuple[Field, ...]:
        """The fields that take a slot in a record, in order; constants take none."""
        return tuple(field for field in self.fields if field.tag.is_scalar)

    def render(self, values: Sequence[Scalar]) -> str:
        """The text of one record: `values` are the scalar fields' values in order."""
        if len(values) != len(self.scalars):
            raise ValueError(f"the message takes {len(self.scalars)} values, not {len(values)}")
        scalars = iter(values)
        return self.format.format(
            *[field.render(field.text if field.tag is Tag.STR_CONSTANT else next(scalars)) for field in self.fields]
        )

    def to_dict(self) -> dict[str, JSON]:
        return {"format": self.format, "fields": [field.to_dict() for field in self.fields]}

    @classmethod
    def from_dict(cls, item: dict[str, JSON]) -> Self:
        return cls(
            format=json_text(item, "format"),
            fields=tuple(Field.from_dict(json_object(field)) for field in json_list(item, "fields")),
        )


@dataclass(frozen=True, kw_only=True)
class _Hole:
    """A placeholder before its argument is known: the argument index, conversion and raw spec text."""

    index: int
    conversion: str
    spec: str


def _from_fstring(node: ast.JoinedStr) -> list[str | _Hole]:
    holes = count()
    pieces: list[str | _Hole] = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            pieces.append(str(part.value))
        else:
            assert isinstance(part, ast.FormattedValue)
            conversion = "" if part.conversion == -1 else chr(part.conversion)
            pieces.append(_Hole(index=next(holes), conversion=conversion, spec=_fstring_spec(part.format_spec)))
    return pieces


def _fstring_spec(node: ast.expr | None) -> str:
    if node is None:
        return ""
    assert isinstance(node, ast.JoinedStr)
    parts = [part.value for part in node.values if isinstance(part, ast.Constant)]
    if len(parts) != len(node.values):
        raise MessageFormatError(
            "a format spec cannot hold replacement fields; write the width or precision as a number"
        )
    return "".join(str(part) for part in parts)


def _from_template(template: Template) -> list[str | _Hole]:
    holes = count()
    pieces: list[str | _Hole] = []
    for item in template:
        if isinstance(item, str):
            pieces.append(item)
        else:
            pieces.append(_Hole(index=next(holes), conversion=item.conversion or "", spec=item.format_spec))
    return pieces


def _from_braces(source: str) -> list[str | _Hole]:
    try:
        parsed = list(Formatter().parse(source))
    except ValueError as error:
        raise MessageFormatError(f"malformed message {source!r}: {error}") from error
    names = [name for _, name, _, _ in parsed if name is not None]
    if "" in names and any(names):
        raise MessageFormatError("use either {} or numbered {0}, {1} placeholders in one message, not both")
    automatic = count()
    pieces: list[str | _Hole] = []
    for literal, name, spec, conversion in parsed:
        pieces.append(literal)
        if name is None:
            continue
        if name == "":
            index = next(automatic)
        elif name.isdecimal():
            index = int(name)
        else:
            raise MessageFormatError(f"placeholder {{{name}}} is not supported; use {{}} or {{0}}, {{1}}, ...")
        pieces.append(_Hole(index=index, conversion=conversion or "", spec=spec or ""))
    return pieces


def _build(pieces: Sequence[str | _Hole], args: Sequence[Argument]) -> Message:
    template: list[str] = []
    fields: list[Field] = []
    used: set[int] = set()
    for piece in pieces:
        if isinstance(piece, str):
            template.append(_escape(piece))
            continue
        fields.append(_field(len(fields), piece, args))
        used.add(piece.index)
        template.append("{}")
    if unused := set(range(len(args))) - used:
        raise MessageFormatError(f"argument {min(unused)} has no placeholder in the message")
    return Message(format="".join(template), fields=tuple(fields))


def _escape(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _field(position: int, hole: _Hole, args: Sequence[Argument]) -> Field:
    if hole.index >= len(args):
        raise MessageFormatError(f"placeholder {hole.index} has no argument; the call passes {len(args)}")
    if hole.conversion:
        raise MessageFormatError(
            f"conversion !{hole.conversion} has no printf form; write a printf type (d, x, f, s and so on), or pass "
            "the text as a str argument"
        )
    argument = args[hole.index]
    if argument is Tag.STR_CONSTANT:
        raise MessageFormatError("a constant passes its text as a str argument, not as the str-constant tag")
    if isinstance(argument, Tag):
        return Field(position=position, tag=argument, spec=Spec.parse(hole.spec))
    return Field(position=position, tag=Tag.STR_CONSTANT, spec=Spec.parse(hole.spec), text=argument)

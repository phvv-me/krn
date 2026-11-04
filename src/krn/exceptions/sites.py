"""Raise sites: the class a kernel raised, where, and the constructor arguments it passed,
as krn._internals.records tags."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Self

from .._internals.records import JSON, Message, Scalar, Site, SiteTable, Tag
from .._internals.records.jsonio import json_int, json_list, json_text

# One argument: a runtime scalar takes its Tag, a str argument is a Message (constant text, or text built from scalars).
type ArgTag = Tag | Message


@dataclass(frozen=True, kw_only=True)
class RaiseSite(Site):
    """One `raise` in a kernel: its class id and its constructor arguments, one tag per argument, in order.

    The host passes the arguments to the class's constructor as Python would, so the rebuilt error's `args`, attributes
    and repr are the ones the same call gives. The display message is the rebuilt error's own `str`.
    """

    class_id: int
    arg_tags: tuple[ArgTag, ...]

    def __post_init__(self) -> None:
        if any(isinstance(tag, Tag) and not tag.is_scalar for tag in self.arg_tags):
            raise ValueError("a str constant takes no slot; write its text as Message.constant")

    @cached_property
    def kinds(self) -> tuple[Tag, ...]:
        """The scalar tags, in order: the ones the record stores, one slot each."""
        return tuple(tag for argument in self.arg_tags for tag in _scalars(argument))

    def arguments(self, values: Sequence[Scalar]) -> tuple[Scalar | str, ...]:
        """The constructor arguments: runtime values in place, and each message rendered from its scalars."""
        runtime = iter(values)
        return tuple(_argument(argument, runtime) for argument in self.arg_tags)

    def to_dict(self) -> dict[str, JSON]:
        return {**super().to_dict(), "class_id": self.class_id, "arg_tags": [_tag_json(tag) for tag in self.arg_tags]}

    @classmethod
    def from_dict(cls, item: dict[str, JSON]) -> Self:
        return cls(
            kernel=json_text(item, "kernel"),
            file=json_text(item, "file"),
            line=json_int(item, "line"),
            column=json_int(item, "column"),
            class_id=json_int(item, "class_id"),
            arg_tags=tuple(_tag_from_json(tag) for tag in json_list(item, "arg_tags")),
        )


class ErrorSiteTable(SiteTable[RaiseSite]):
    """The raise sites of a program: a raise with the same kernel, position, class and arguments keeps its id."""

    def add(
        self,
        *,
        kernel: str,
        file: str,
        line: int,
        class_id: int,
        arg_tags: tuple[ArgTag, ...],
        column: int = 0,
    ) -> RaiseSite:
        return self.intern(
            RaiseSite(kernel=kernel, file=file, line=line, column=column, class_id=class_id, arg_tags=arg_tags)
        )


def _scalars(argument: ArgTag) -> tuple[Tag, ...]:
    if isinstance(argument, Tag):
        return (argument,)
    return tuple(field.tag for field in argument.scalars)


def _argument(argument: ArgTag, runtime: Iterator[Scalar]) -> Scalar | str:
    if isinstance(argument, Tag):
        return next(runtime)
    return argument.render([next(runtime) for _ in argument.scalars])


def _tag_json(tag: ArgTag) -> JSON:
    return tag.value if isinstance(tag, Tag) else tag.to_dict()


def _tag_from_json(tag: JSON) -> ArgTag:
    if isinstance(tag, dict) and "format" in tag and "fields" in tag:
        return Message.from_dict(tag)
    if isinstance(tag, str):
        return Tag(tag)
    raise ValueError(f"{tag!r} is not an argument tag")

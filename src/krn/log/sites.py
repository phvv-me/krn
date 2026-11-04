"""Log sites: one per log call in a kernel, interned on every field and numbered in a table a cached program reloads."""

from dataclasses import dataclass
from typing import Self

from .._internals.records import JSON, Message, Site
from .._internals.records import SiteTable as RecordSiteTable
from .._internals.records.jsonio import json_int, json_object, json_text


@dataclass(frozen=True, kw_only=True)
class LogSite(Site):
    """A log call: the logger it names, the level it logs at, and its message."""

    logger: str
    level: int
    message: Message

    def to_dict(self) -> dict[str, JSON]:
        return {**super().to_dict(), "logger": self.logger, "level": self.level, "message": self.message.to_dict()}

    @classmethod
    def from_dict(cls, item: dict[str, JSON]) -> Self:
        return cls(
            kernel=json_text(item, "kernel"),
            file=json_text(item, "file"),
            line=json_int(item, "line"),
            column=json_int(item, "column"),
            logger=json_text(item, "logger"),
            level=json_int(item, "level"),
            message=Message.from_dict(json_object(item["message"])),
        )


class SiteTable(RecordSiteTable[LogSite]):
    """The log sites of a program: a call with the same logger, level, kernel, position and message keeps its id."""

    def add(
        self, *, logger: str, level: int, file: str, line: int, kernel: str, message: Message, column: int = 0
    ) -> LogSite:
        return self.intern(
            LogSite(kernel=kernel, file=file, line=line, column=column, logger=logger, level=level, message=message)
        )

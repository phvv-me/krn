"""Sites: one per call or raise in a kernel, numbered by a table that interns identical sites and reloads from JSON."""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Self

from .jsonio import JSON, json_object

UNASSIGNED = -1
# A site's id is a u16 in every record envelope, so a table refuses the 65,537th site when it is interned.
MAX_SITE_ID = 0xFFFF


@dataclass(frozen=True, kw_only=True)
class Site:
    """Where a call or raise is: its kernel, file, line and column (0 when unknown), and its id once interned.

    Two sites that differ in any field are different sites, so the kernel, the column and every subclass field decide
    identity, and a call captured twice on one line is one site only when everything else matches.
    """

    id: int = UNASSIGNED
    kernel: str
    file: str
    line: int
    column: int = 0

    def to_dict(self) -> dict[str, JSON]:
        return {"kernel": self.kernel, "file": self.file, "line": self.line, "column": self.column}


class SiteTable[S: Site]:
    """The sites of a program, numbered 0, 1, 2 in the order they were interned.

    `intern` returns the stored site equal to the one given in every field but the id, so interning a site twice keeps
    its id. The id is what a record carries, so the table refuses a site past MAX_SITE_ID.
    """

    def __init__(self) -> None:
        self._sites: list[S] = []
        self._ids: dict[S, int] = {}

    def __getitem__(self, identifier: int) -> S:
        return self._sites[identifier]

    def __len__(self) -> int:
        return len(self._sites)

    def intern(self, site: S) -> S:
        """The stored site equal to `site` in every field but the id, stored under the next id when it is new."""
        key = replace(site, id=UNASSIGNED)
        if (identifier := self._ids.get(key)) is None:
            identifier = len(self._sites)
            if identifier > MAX_SITE_ID:
                raise OverflowError(
                    f"site {identifier} exceeds the {MAX_SITE_ID + 1} sites a record can name (its site id is a u16); "
                    "split the kernel or share the messages"
                )
            self._sites.append(replace(key, id=identifier))
            self._ids[key] = identifier
        return self._sites[identifier]

    def dumps(self) -> str:
        """The table as JSON, listed in id order."""
        return json.dumps([site.to_dict() for site in self._sites])

    @classmethod
    def loads(cls, text: str, load: Callable[[dict[str, JSON]], S]) -> Self:
        """The table written by dumps, with the same ids; `load` reads one site from its JSON object."""
        table = cls()
        for item in json.loads(text):
            table.intern(load(json_object(item)))
        return table

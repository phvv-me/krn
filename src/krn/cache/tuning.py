"""Persistent library tuning winners."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TuningDb:
    """Measured winners on disk. Keys carry library versions, so an upgrade measures again."""

    path: Path
    winners: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.winners = json.loads(self.path.read_text())
        except FileNotFoundError, json.JSONDecodeError:
            self.winners = {}

    def put(self, key: str, name: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.winners[key] = name
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.winners, indent=1))
        temporary.replace(self.path)  # atomic, so concurrent processes never read half a file

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from krn.hw.devices import Device

type Vendor = Literal["nvidia", "amd", "apple", "intel"]


@dataclass(frozen=True, kw_only=True)
class Diagnostic:
    id: str
    severity: Literal["note", "warning", "error"]
    message: str
    suggested_fix: str
    targets: tuple[Vendor, ...]


@dataclass(frozen=True, kw_only=True)
class Rule(ABC):
    target: Device

    @abstractmethod
    def evaluate(self) -> tuple[Diagnostic, ...]:
        """Evaluate against the target facts supplied at construction."""

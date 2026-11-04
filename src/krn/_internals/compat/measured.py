"""Measured results: what the hardware-gated tests recorded on each machine, and how they become a cell's evidence.

A machine file holds one record per (feature, target) its tests ran. `attach` turns the records into the MEASURED
evidence of the cells they name. A record that did not match, or that names a cell the analysis does not call native or
composed, is an error: a measured mismatch never passes silently.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum, auto
from pathlib import Path

from .model import Evidence, Matrix, Measurement, Status

SCHEMA = "krn.compat.measured"
VERSION = 1


class Outcome(StrEnum):
    MATCH = auto()  # the kernel ran on the machine and agreed with the emulator within the tolerance
    MISMATCH = auto()  # it ran and disagreed
    ERROR = auto()  # it did not run: a compile or launch error


@dataclass(frozen=True, kw_only=True)
class Record:
    """One run of one feature on one target, on one machine: the OS and driver it ran on, and its tolerance."""

    feature: str
    target: str
    machine: str
    os: str
    driver: str
    status: Outcome
    tolerance: str


def slug(machine: str) -> str:
    """The file name of a machine: `Apple M4 Pro` is `apple-m4-pro`."""
    return re.sub(r"[^a-z0-9]+", "-", machine.lower()).strip("-")


def dumps(records: Iterable[Record]) -> str:
    """The records of one machine as JSON, one record per line, in the order of feature and target."""
    ordered = sorted(records, key=lambda r: (r.feature, r.target))
    lines = [json.dumps(_record_json(record), sort_keys=True) for record in ordered]
    body = ",\n".join(lines)
    return f'{{\n"schema": "{SCHEMA}",\n"version": {VERSION},\n"records": [\n{body}\n]\n}}\n'


def loads(text: str) -> list[Record]:
    """The records of a machine file. Another schema or version is an error."""
    document = json.loads(text)
    if document.get("schema") != SCHEMA or document.get("version") != VERSION:
        raise ValueError(f"expected {SCHEMA} version {VERSION}, got {document.get('schema')} {document.get('version')}")
    return [Record(**{**row, "status": Outcome(row["status"])}) for row in document["records"]]


def _record_json(record: Record) -> dict[str, str]:
    return {**asdict(record), "status": record.status.value}


def write(directory: Path, machine: str, records: Iterable[Record]) -> Path:
    """Write the records of one machine to `directory/<slug>.json`. They replace what the file already holds for the
    same feature and target, and keep the rest, so a partial run adds to the machine's file."""
    path = directory / f"{slug(machine)}.json"
    held = {(r.feature, r.target): r for r in loads(path.read_text())} if path.exists() else {}
    held.update({(r.feature, r.target): r for r in records})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(held.values()))
    return path


def read(directory: Path) -> list[Record]:
    """Every record of every machine file in `directory`, in file-name order. A missing directory has no records."""
    if not directory.is_dir():
        return []
    return [record for path in sorted(directory.glob("*.json")) for record in loads(path.read_text())]


def attach(matrix: Matrix, records: Iterable[Record]) -> Matrix:
    """The matrix with each matched record added to its cell as MEASURED evidence. Any other outcome, or a record of a
    cell that is not native or composed, or that the matrix does not have, is an error."""
    records = list(records)
    failed = [r for r in records if r.status is not Outcome.MATCH]
    if failed:
        names = ", ".join(f"{r.feature} on {r.target} ({r.machine}): {r.status}" for r in failed)
        raise ValueError(f"measured runs did not match: {names}")
    cells = {feature: dict(row) for feature, row in matrix.cells.items()}
    for record in records:
        support = cells.get(record.feature, {}).get(record.target)
        if support is None:
            raise ValueError(f"measured {record.feature} on {record.target} has no cell in the matrix")
        if support.status not in (Status.NATIVE, Status.COMPOSED):
            raise ValueError(f"{record.feature} on {record.target} is {support.status}, yet a run matched it")
        run = Measurement(machine=record.machine, os=record.os, driver=record.driver, tolerance=record.tolerance)
        measured = tuple(dict.fromkeys((*support.measured, run)))
        cells[record.feature][record.target] = replace(support, evidence=Evidence.MEASURED, measured=measured)
    return replace(matrix, cells=cells)

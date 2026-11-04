"""The compatibility matrix as data: features, targets, and the support of each feature on each target.

A cell's status says what krn does on that target, and its evidence says how that was learned. The JSON form is sorted
and versioned, so a regenerated file differs only where a cell changed.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto

from .requires import Requires

SCHEMA = "krn.compat.matrix"
VERSION = 2


class Status(StrEnum):
    NATIVE = auto()  # one hardware instruction or library call
    COMPOSED = auto()  # a software sequence: correct everywhere, with a cost
    REFUSED = auto()  # the target cannot compile it; the note is the reason and `instead` the alternative
    EMULATOR = auto()  # only the Python emulator runs it
    UNKNOWN = auto()  # not analyzed, or analyzed to an error krn does not classify as a refusal


class Evidence(StrEnum):
    ANALYZED = auto()  # derived from the compiled kernel and the lowering tables
    MEASURED = auto()  # a hardware-gated test ran it and matched the emulator
    DOCUMENTED = auto()  # vendor documentation


@dataclass(frozen=True, kw_only=True)
class Measurement:
    """A hardware run that matched the emulator: the machine, its OS and driver, and the tolerance it checked."""

    machine: str
    os: str
    driver: str
    tolerance: str


@dataclass(frozen=True, kw_only=True)
class Support:
    """What one feature does on one target. `length` is the estimated instruction count of a COMPOSED sequence.

    A MEASURED cell lists the runs that matched, and only those; an ANALYZED cell has none.
    """

    status: Status
    note: str = ""
    instead: str = ""
    requires: Requires | None = None
    evidence: Evidence = Evidence.ANALYZED
    length: int | None = None
    measured: tuple[Measurement, ...] = ()

    def __post_init__(self) -> None:
        if self.status is Status.REFUSED and not (self.note and self.instead):
            raise ValueError("a REFUSED cell names its reason and an alternative")
        if self.length is not None and self.status is not Status.COMPOSED:
            raise ValueError(f"only a COMPOSED cell has a length, got {self.status}")
        if bool(self.measured) != (self.evidence is Evidence.MEASURED):
            raise ValueError("a cell has measured runs exactly when its evidence is MEASURED")


@dataclass(frozen=True, kw_only=True)
class Feature:
    """A row: a catalog intrinsic, a general function on one operand type, or a number type's operation.

    `id` is unique: the name, with the operand type in brackets when one row per type exists. `doc` is the first line of
    the docstring, or the note that says what the row stands for.
    """

    id: str
    module: str
    group: str
    name: str
    signature: str = ""
    doc: str = ""


@dataclass(frozen=True, kw_only=True)
class Hardware:
    """A column: a target with its vendor and the family that decides what it has."""

    arch: str
    vendor: str
    family: str


@dataclass(frozen=True, kw_only=True)
class Matrix:
    """Features and targets in the order the JSON keeps them, and the cells by feature id, then target arch."""

    features: tuple[Feature, ...]
    targets: tuple[Hardware, ...]
    cells: Mapping[str, Mapping[str, Support]]
    generated_from: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [feature.id for feature in self.features]
        if len(set(ids)) != len(ids):
            raise ValueError("feature ids are not unique")
        arches = {target.arch for target in self.targets}
        for feature_id, row in self.cells.items():
            if feature_id not in ids or not set(row) <= arches:
                raise ValueError(f"cell row {feature_id} names an unknown feature or target")

    def counts(self) -> dict[str, dict[Status, int]]:
        """How many cells of each target have each status: `counts()[arch][status]`."""
        out: dict[str, dict[Status, int]] = {target.arch: {} for target in self.targets}
        for row in self.cells.values():
            for arch, support in row.items():
                bucket = out[arch]
                bucket[support.status] = bucket.get(support.status, 0) + 1
        return out


def _support_json(
    support: Support, alternatives: Mapping[str, int], requirements: Mapping[Requires, int]
) -> dict[str, str | int | list[dict[str, str]]]:
    fields = {
        "status": support.status.value,
        "note": support.note,
        "instead": alternatives.get(support.instead),
        "requires": requirements.get(support.requires) if support.requires else None,
        "evidence": None if support.evidence is Evidence.ANALYZED else support.evidence.value,
        "length": support.length,
        "measured": [asdict(run) for run in support.measured] or None,
    }
    return {key: value for key, value in fields.items() if value not in ("", None)}


def _requires_json(requires: Requires) -> dict[str, str]:
    """The facts that are set, and their one-line summary for the page, which `loads` does not read back."""
    facts = {key: value for key, value in asdict(requires).items() if value}
    return {**facts, "summary": requires.summary()}


def _requires_from(item: Mapping[str, str]) -> Requires:
    return Requires(**{key: value for key, value in item.items() if key != "summary"})


def dumps(matrix: Matrix) -> str:
    """The matrix as JSON, one feature per line. Each distinct alternative and each requirement is stored once, in the
    head, and a cell refers to it by index. The same matrix always gives the same bytes."""
    supports = [s for row in matrix.cells.values() for s in row.values()]
    alternatives = sorted({s.instead for s in supports if s.instead})
    requirements = sorted({s.requires for s in supports if s.requires}, key=lambda r: json.dumps(_requires_json(r)))
    index = {text: number for number, text in enumerate(alternatives)}
    requirement_index = {requires: number for number, requires in enumerate(requirements)}
    head = {
        "alternatives": alternatives,
        "generated_from": dict(sorted(matrix.generated_from.items())),
        "requirements": [_requires_json(r) for r in requirements],
        "schema": SCHEMA,
        "targets": [{"arch": t.arch, "family": t.family, "vendor": t.vendor} for t in matrix.targets],
        "version": VERSION,
    }
    features = [
        {"doc": f.doc, "group": f.group, "id": f.id, "module": f.module, "name": f.name, "signature": f.signature}
        for f in matrix.features
    ]
    lines = [
        f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}" for key, value in head.items()
    ]
    lines.append(
        '"features": [\n' + ",\n".join(json.dumps(f, ensure_ascii=False, sort_keys=True) for f in features) + "\n]"
    )
    rows = []
    for feature_id in sorted(matrix.cells):
        row = {
            arch: _support_json(matrix.cells[feature_id][arch], index, requirement_index)
            for arch in sorted(matrix.cells[feature_id])
        }
        rows.append(
            f"{json.dumps(feature_id)}: {json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        )
    lines.append('"cells": {\n' + ",\n".join(rows) + "\n}")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def loads(text: str) -> Matrix:
    """The matrix of a JSON document written by `dumps`. A document of another schema or version is an error."""
    document = json.loads(text)
    if document.get("schema") != SCHEMA or document.get("version") != VERSION:
        raise ValueError(f"expected {SCHEMA} version {VERSION}, got {document.get('schema')} {document.get('version')}")
    alternatives = document["alternatives"]
    requirements = [_requires_from(item) for item in document["requirements"]]
    cells = {
        feature_id: {
            arch: Support(
                status=Status(item["status"]),
                note=item.get("note", ""),
                instead=alternatives[item["instead"]] if "instead" in item else "",
                requires=requirements[item["requires"]] if "requires" in item else None,
                evidence=Evidence(item.get("evidence", Evidence.ANALYZED.value)),
                length=item.get("length"),
                measured=tuple(Measurement(**run) for run in item.get("measured", ())),
            )
            for arch, item in row.items()
        }
        for feature_id, row in document["cells"].items()
    }
    return Matrix(
        features=tuple(Feature(**row) for row in document["features"]),
        targets=tuple(Hardware(**row) for row in document["targets"]),
        cells=cells,
        generated_from=document["generated_from"],
    )

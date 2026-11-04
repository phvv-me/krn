"""The shape of the files in `results/` that scripts write and the docs site reads back.

`conformance_*.json` is written by `conformance.py parse` and read by `conformance_report.py` and `docs/`; the
`kl-bench/1` files are written by `backend_latency.py` (and by any track that measures) and read by `docs/`. A key that
Python reserves (`pass`, `class`) makes a type use the functional form.
"""

from collections.abc import Mapping, Sequence
from typing import Literal, NotRequired, TypedDict

# JSON as covariant containers, so a dict of lists of floats is a Json without a copy
type Json = str | int | float | bool | Sequence[Json] | Mapping[str, Json] | None

# the pass rate of a set of subjects on one target: `percent` is over the applicable ones, `percent_of_all` over
# the `universe` of subjects the suite has, tested or not; either is None when its denominator is empty
FullStats = TypedDict(
    "FullStats",
    {
        "subjects": int,
        "applicable": int,
        "pass": int,
        "partial": int,
        "fail": int,
        "skip": int,
        "by_design": int,
        "percent": float | None,
        "universe": int,
        "percent_of_all": float | None,
    },
)
Cell = TypedDict(
    "Cell",
    {
        "state": str,
        "pass": int,
        "fail": int,
        "skip": dict[str, int],
        "class": str,
        "reason": str,
        "failure": str,
        "seconds": float,
    },
)
Group = TypedDict(
    "Group", {"target": str, "class": str, "reason": str, "subjects": int, "kernel": int, "sample": list[str]}
)
HostSkip = TypedDict("HostSkip", {"class": str, "reason": str, "nodes": int})


class Subject(TypedDict):
    id: str
    family: str
    named: bool  # the id of a kernel or case; False for a subject named by the sentence of a test function
    cells: dict[str, Cell]


class TargetSummary(TypedDict):
    overall: FullStats
    kernels: FullStats
    by_family: dict[str, FullStats]


class Facts(TypedDict):
    hostname: str
    platform: str
    machine: str
    cpu: str
    gpus: list[str]
    python: str
    clang: str
    packages: dict[str, str]
    date: str


class HostFacts(Facts):
    seconds: NotRequired[float]
    target: NotRequired[str]
    examples_target: NotRequired[str]


class ExampleRow(TypedDict):
    example: str
    host: str
    status: str
    seconds: NotRequired[float]
    error: NotRequired[str]
    reason: NotRequired[str]


class Document(TypedDict):
    date: str
    hosts: dict[str, HostFacts]
    targets: list[str]
    families: list[str]
    summary: dict[str, TargetSummary]
    node_outcomes: dict[str, dict[str, int]]
    subjects: list[Subject]
    groups: list[Group]
    host_skips: dict[str, list[HostSkip]]
    stems: dict[str, dict[str, dict[str, int]]]
    examples: list[ExampleRow]


class NodeRow(TypedDict):
    stem: str
    function: str
    parameter: str
    outcome: str
    seconds: float
    message: str


class SuiteDocument(TypedDict):
    label: str
    seconds: float
    target: str
    facts: Facts
    nodes: list[NodeRow]


class ExamplesDocument(TypedDict):
    label: str
    target: str
    rows: list[ExampleRow]


class BenchRow(TypedDict):
    """One measurement; the optional keys are details a chart shows on hover."""

    group: str
    series: str
    value: float
    note: NotRequired[str]
    seconds: NotRequired[float]
    error: NotRequired[float | None]
    cold: NotRequired[float]
    link: NotRequired[float | None]
    entries: NotRequired[int]
    resident_mb: NotRequired[float | None]
    launch_bound: NotRequired[bool]


class BenchFile(TypedDict):
    """A `kl-bench/1` file: any track may write one into `results/` and the site picks it up."""

    schema: Literal["kl-bench/1"]
    name: str
    title: str
    host: str
    date: str
    unit: str
    lower_is_better: bool
    kind: str
    versions: dict[str, str]
    rows: list[BenchRow]
    note: NotRequired[str]

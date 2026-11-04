"""Build the compatibility matrix and write it to `../krn.dev/data/compat/matrix.json`, the site's compatibility data.

Run from the repository root with `mb run python -m experiments.compat_matrix`. Every cell is compiled for its target by
`krn._internals.compat.analyze`. The catalog rows reuse the case table of
tests/compiler/test_intrinsic_families.py, so a family the tests compile is a family the page shows. `--check` fails
when the committed file is not what the generator writes.
"""

import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Annotated

import cyclopts

from krn._internals.compat import Matrix, Status, dumps, measured
from krn._internals.compat.analyze import Probe, analyze, catalog_feature, catalog_judge, function_probes, type_probes
from krn.facts import catalog
from krn.facts.targets import all_targets
from tests.compiler.test_intrinsic_families import CASES, Case

COMPAT = (Path(__file__).resolve().parents[1] / ".." / "krn.dev" / "data" / "compat").resolve()
OUTPUT = COMPAT / "matrix.json"
MEASURED = COMPAT / "measured"
GENERATED_FROM = {
    "generator": "experiments/compat_matrix.py",
    "cases": "tests/compiler/test_intrinsic_families.py",
    "catalog_entries": str(len(catalog.ENTRIES)),
    "measured": "krn/krn.dev/data/compat/measured",
}
LOG = logging.getLogger(__name__)
APP = cyclopts.App(name="compat_matrix", help=__doc__.splitlines()[0])


def case_source(case: Case) -> str:
    """The kernel file of one case, as the family tests write it."""
    with tempfile.TemporaryDirectory() as directory:
        return case.file(Path(directory)).read_text()


def catalog_probes() -> list[Probe]:
    """One probe per catalog entry: a compile per case of the family tests, or the spelling alone for an entry no case
    covers. An entry with several cases (`lop3` has six tables) has one row per operand type."""
    by_name: dict[str, dict[tuple[str, ...], Case]] = {}
    for case in CASES:
        by_name.setdefault(case.name, {}).setdefault(case.dtypes, case)
    probes = []
    for entry in catalog.ENTRIES:
        cases = by_name.get(entry.name, {})
        judge = catalog_judge(entry)
        if not cases:
            probes.append(Probe(feature=catalog_feature(entry), judge=judge))
        for dtypes, case in cases.items():
            feature = catalog_feature(entry, dtypes if len(cases) > 1 else ())
            probes.append(Probe(feature=feature, source=case_source(case), entry="family_case", judge=judge))
    return probes


def analyzed() -> Matrix:
    """The matrix of every catalog row, every krn.math and krn.intrinsics function, and every number and packed type,
    as the compiler decides it, before any hardware run is added."""
    probes = [*catalog_probes(), *function_probes(), *type_probes()]
    return analyze(probes, all_targets(), GENERATED_FROM)


def build(measured_dir: Path = MEASURED) -> Matrix:
    """`analyzed` with the runs the machines in `measured_dir` recorded as the evidence of their cells."""
    return measured.attach(analyzed(), measured.read(measured_dir))


def summarize(matrix: Matrix) -> None:
    """Log how many cells of each target have each status, and how many rows the matrix has."""
    LOG.info("%d features on %d targets", len(matrix.features), len(matrix.targets))
    for target in matrix.targets:
        counts = matrix.counts()[target.arch]
        cells = " ".join(f"{status.value}={counts.get(status, 0)}" for status in Status)
        LOG.info("%-9s %s", target.arch, cells)
    totals = Counter(s.status.value for row in matrix.cells.values() for s in row.values())
    LOG.info("total %s", dict(sorted(totals.items())))


@APP.default
def main(
    *,
    output: Annotated[Path, cyclopts.Parameter(help="where the JSON goes")] = OUTPUT,
    measured_dir: Annotated[Path, cyclopts.Parameter(help="where the machine files are read from")] = MEASURED,
    check: bool = False,
) -> None:
    """Write the matrix, or with `--check` exit non-zero when the file at `output` differs from it."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stderr)
    LOG.setLevel(logging.INFO)  # the compiler logs every fallback at INFO; only this script's lines are the report
    matrix = build(measured_dir)
    summarize(matrix)
    text = dumps(matrix)
    if check:
        if not output.exists() or output.read_text() != text:
            LOG.error("%s is stale: run `mb run python -m experiments.compat_matrix`", output)
            raise SystemExit(1)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    LOG.info("wrote %s (%d bytes)", output, len(text))


if __name__ == "__main__":
    APP()

"""Measured runs: machine files round trip, a file keeps the runs it does not replace, and a run becomes the
MEASURED evidence of its cell only when it matched and the cell is native or composed."""

from dataclasses import replace
from pathlib import Path

import pytest

from krn._internals.compat import Evidence, Feature, Hardware, Matrix, Measurement, Status, Support
from krn._internals.compat.measured import Outcome, Record, attach, dumps, loads, read, slug, write


def record(feature: str = "f.op", target: str = "sm_120a", status: Outcome = Outcome.MATCH) -> Record:
    return Record(
        feature=feature,
        target=target,
        machine="silo RTX 5080",
        os="Linux 6.8",
        driver="570.26",
        status=status,
        tolerance="exact",
    )


def matrix(status: Status = Status.NATIVE) -> Matrix:
    support = Support(status=status, note="why", instead="how") if status is Status.REFUSED else Support(status=status)
    return Matrix(
        features=(Feature(id="f.op", module="types", group="f", name="op"),),
        targets=(Hardware(arch="sm_120a", vendor="nvidia", family="sm_120"),),
        cells={"f.op": {"sm_120a": support}},
    )


def test_a_machine_file_round_trips_and_is_sorted() -> None:
    text = dumps([record("b.op"), record("a.op")])
    assert [r.feature for r in loads(text)] == ["a.op", "b.op"]
    assert dumps(loads(text)) == text


def test_loads_rejects_another_schema_or_version() -> None:
    with pytest.raises(ValueError, match="expected"):
        loads('{"schema": "other", "version": 1, "records": []}')


def test_slug_names_the_machine_file() -> None:
    assert slug("Apple M4 Pro") == "apple-m4-pro"
    assert slug("silo NVIDIA GeForce RTX 5080") == "silo-nvidia-geforce-rtx-5080"


def test_write_replaces_the_runs_it_repeats_and_keeps_the_others(tmp_path: Path) -> None:
    write(tmp_path, "silo RTX 5080", [record("a.op"), record("b.op")])
    path = write(tmp_path, "silo RTX 5080", [record("b.op", status=Outcome.ERROR), record("c.op")])
    assert path == tmp_path / "silo-rtx-5080.json"
    assert {(r.feature, r.status) for r in read(tmp_path)} == {
        ("a.op", Outcome.MATCH),
        ("b.op", Outcome.ERROR),
        ("c.op", Outcome.MATCH),
    }


def test_read_gives_nothing_without_a_directory(tmp_path: Path) -> None:
    assert read(tmp_path / "absent") == []


def test_a_matched_run_is_the_measured_evidence_of_its_cell() -> None:
    measured = attach(matrix(), [record()])
    support = measured.cells["f.op"]["sm_120a"]
    assert support.evidence is Evidence.MEASURED
    assert support.measured == (
        Measurement(machine="silo RTX 5080", os="Linux 6.8", driver="570.26", tolerance="exact"),
    )


def test_two_machines_and_a_repeated_run_keep_one_measurement_each() -> None:
    other = replace(record(), machine="crimson RTX 3090")
    measured = attach(matrix(Status.COMPOSED), [record(), record(), other])
    assert [run.machine for run in measured.cells["f.op"]["sm_120a"].measured] == ["silo RTX 5080", "crimson RTX 3090"]


def test_a_mismatch_or_an_error_is_never_attached() -> None:
    with pytest.raises(ValueError, match="did not match"):
        attach(matrix(), [record(status=Outcome.MISMATCH)])
    with pytest.raises(ValueError, match="did not match"):
        attach(matrix(), [record(status=Outcome.ERROR)])


def test_a_run_of_a_refused_cell_or_of_a_missing_one_is_an_error() -> None:
    with pytest.raises(ValueError, match="yet a run matched it"):
        attach(matrix(Status.REFUSED), [record()])
    with pytest.raises(ValueError, match="has no cell"):
        attach(matrix(), [record(feature="g.op")])
    with pytest.raises(ValueError, match="has no cell"):
        attach(matrix(), [record(target="gfx942")])


def test_attach_with_no_runs_is_the_same_matrix() -> None:
    assert attach(matrix(), []) == matrix()

"""The matrix as data: the status rules of a cell, the shape rules of a matrix, and the JSON round trip."""

import json

import pytest

from krn._internals.compat import (
    Evidence,
    Feature,
    Hardware,
    Matrix,
    Measurement,
    Requires,
    Status,
    Support,
    dumps,
    loads,
)
from krn._internals.compat.model import SCHEMA, VERSION


def sample() -> Matrix:
    features = (
        Feature(id="a.f", module="catalog", group="a", name="a.f", signature="(x: f32) -> f32", doc="sine"),
        Feature(id="b.op", module="types", group="b", name="b.op", doc="add"),
    )
    targets = (
        Hardware(arch="sm_120a", vendor="nvidia", family="sm_120"),
        Hardware(arch="apple_m4", vendor="apple", family="Apple family 9"),
        Hardware(arch="avx2", vendor="cpu", family="x86-64"),
    )
    cells = {
        "a.f": {
            "sm_120a": Support(status=Status.NATIVE, requires=Requires(ptx="8.7", toolkit="12.8", driver="570.26")),
            "apple_m4": Support(status=Status.COMPOSED, note="a helper", length=3),
            "avx2": Support(
                status=Status.EMULATOR,
                evidence=Evidence.MEASURED,
                measured=(Measurement(machine="silo", os="Linux 6.8", driver="570.26", tolerance="exact"),),
            ),
        },
        "b.op": {
            "sm_120a": Support(status=Status.REFUSED, note="no fp6 path", instead="use f16"),
            "apple_m4": Support(status=Status.UNKNOWN, note="not analyzed"),
            "avx2": Support(status=Status.REFUSED, note="no fp6 path", instead="use f16"),
        },
    }
    return Matrix(features=features, targets=targets, cells=cells, generated_from={"generator": "test"})


def test_round_trip_keeps_every_field() -> None:
    matrix = sample()
    assert loads(dumps(matrix)) == matrix


def test_dumps_is_byte_stable_and_stores_each_alternative_once() -> None:
    assert dumps(sample()) == dumps(sample())
    assert dumps(sample()).count("use f16") == 1


def test_loads_rejects_another_schema_or_version() -> None:
    document = json.loads(dumps(sample()))
    with pytest.raises(ValueError, match="expected"):
        loads(json.dumps({**document, "version": VERSION + 1}))
    with pytest.raises(ValueError, match="expected"):
        loads(json.dumps({**document, "schema": "other"}))
    assert document["schema"] == SCHEMA and document["version"] == VERSION


def test_refused_cell_needs_reason_and_alternative() -> None:
    with pytest.raises(ValueError, match="reason and an alternative"):
        Support(status=Status.REFUSED, note="why")
    with pytest.raises(ValueError, match="reason and an alternative"):
        Support(status=Status.REFUSED, instead="use f16")


def test_measured_runs_exist_exactly_when_the_evidence_is_measured() -> None:
    run = Measurement(machine="silo", os="Linux 6.8", driver="570.26", tolerance="exact")
    with pytest.raises(ValueError, match="exactly when"):
        Support(status=Status.NATIVE, measured=(run,))
    with pytest.raises(ValueError, match="exactly when"):
        Support(status=Status.NATIVE, evidence=Evidence.MEASURED)


def test_requires_names_at_least_one_fact_and_summarizes_them_in_order() -> None:
    with pytest.raises(ValueError, match="at least one fact"):
        Requires(source="https://example.test")
    requires = Requires(toolkit="12.8", driver="570.26", ptx="8.7", gpu="sm_120a", cpu="", source="x")
    assert requires.summary() == "CUDA 12.8, driver 570.26, PTX ISA 8.7, sm_120a"
    assert Requires(os="27.0", metal="4.1", rocm="7.0", cpu="AVX2").summary() == "macOS 27.0, Metal 4.1, ROCm 7.0, AVX2"


def test_length_is_only_for_composed_cells() -> None:
    with pytest.raises(ValueError, match="only a COMPOSED"):
        Support(status=Status.NATIVE, length=2)


def test_matrix_rejects_repeated_features_and_unknown_cells() -> None:
    matrix = sample()
    feature = matrix.features[0]
    with pytest.raises(ValueError, match="not unique"):
        Matrix(features=(feature, feature), targets=matrix.targets, cells={})
    with pytest.raises(ValueError, match="unknown feature or target"):
        Matrix(features=matrix.features, targets=matrix.targets, cells={"missing": {}})
    with pytest.raises(ValueError, match="unknown feature or target"):
        Matrix(
            features=matrix.features, targets=matrix.targets, cells={"a.f": {"gfx942": Support(status=Status.NATIVE)}}
        )


def test_counts_by_target_and_status() -> None:
    counts = sample().counts()
    assert counts["sm_120a"] == {Status.NATIVE: 1, Status.REFUSED: 1}
    assert counts["apple_m4"] == {Status.COMPOSED: 1, Status.UNKNOWN: 1}

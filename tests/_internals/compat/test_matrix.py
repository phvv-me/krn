"""The committed matrix is what the generator writes now, every catalog entry has a row, and every refusal says why."""

import pytest

from experiments.compat_matrix import OUTPUT, build
from krn._internals.compat import Matrix, Status, dumps, loads
from krn.facts import catalog
from krn.facts.targets import registry


@pytest.fixture(scope="module")
def built() -> Matrix:
    return build()


committed = pytest.mark.skipif(not OUTPUT.exists(), reason="the site data is not checked out here")


@committed
def test_committed_matrix_equals_what_the_generator_writes(built: Matrix) -> None:
    assert OUTPUT.read_text() == dumps(built)


def test_two_builds_give_identical_bytes(built: Matrix) -> None:
    assert dumps(built) == dumps(build())


def test_every_catalog_entry_has_a_row(built: Matrix) -> None:
    assert {entry.name for entry in catalog.ENTRIES} <= {feature.name for feature in built.features}


def test_every_target_is_a_column(built: Matrix) -> None:
    assert {target.arch for target in built.targets} == set(registry())


def test_every_refused_cell_names_a_reason_and_an_alternative(built: Matrix) -> None:
    refused = [s for row in built.cells.values() for s in row.values() if s.status is Status.REFUSED]
    assert refused
    assert all(s.note and s.instead for s in refused)


@committed
def test_committed_file_reads_back_to_the_same_matrix(built: Matrix) -> None:
    assert loads(OUTPUT.read_text()) == built

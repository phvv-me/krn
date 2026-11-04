"""The mean of a ragged tile divides the sum of its valid lanes by their number, so a lane past the edge of its tensor
does not dilute it. An element with no valid lane is NaN, as numpy's mean of nothing is, and that NaN is stored."""

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn import Blocks, PrecisionWarning, Tiles
from krn.compiler import performance
from krn.compiler.api import capture
from krn.compiler.lower import emit
from krn.emulator.emulate import build as emulate
from krn.facts.targets import Target
from tests.conftest import KERNELS, build_kernel

FILE = KERNELS / "ragged.py"
GPU = Target.of("sm_90a")
APPLE = Target.of("apple_m4")


def on_emulator(name: str, arrays: list[np.ndarray], launch: Blocks | Tiles, cols: int | None = None) -> None:
    """Run the emulator; `cols` is the tile width of a kernel that takes a `Width` struct."""
    consts, values = (None, arrays) if cols is None else ({"t": {"cols": cols}}, [{}, *arrays])
    with np.errstate(all="ignore"):
        emulate(capture(FILE, name, consts=consts, target=GPU), GPU).run(launch, values)


def on_c_backend(name: str, arrays: list[np.ndarray], launch: Blocks | Tiles, cols: int | None = None) -> None:
    consts, values = (None, arrays) if cols is None else ({"t": {"cols": cols}}, [{}, *arrays])
    build_kernel(FILE, name, consts)(*values, launch=launch)


def assert_bits_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(actual.view(np.uint32)[finite], expected.view(np.uint32)[finite])


def test_a_thousand_ones_in_a_1024_wide_tile_mean_exactly_one_on_the_emulator_and_the_c_backend() -> None:
    ones = np.ones((1, 1000), np.float32)
    emulated, compiled = np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float32)
    on_emulator("mean_row", [ones, emulated], Blocks(1), cols=1024)
    on_c_backend("mean_row", [ones, compiled], Blocks(1), cols=1024)
    assert emulated[0, 0] == 1.0
    assert compiled[0, 0] == 1.0


@pytest.mark.hardware
def test_a_thousand_ones_in_a_1024_wide_tile_mean_exactly_one_on_the_apple_gpu(metal_program) -> None:
    ones, out = np.ones((1, 1000), np.float32), np.zeros((1, 1), np.float32)
    metal_program(capture(FILE, "mean_row", consts={"t": {"cols": 1024}}, target=APPLE)).run(Blocks(1), [{}, ones, out])
    assert out[0, 0] == 1.0


@settings(max_examples=12, deadline=None)
@given(cols=st.integers(1, 2048), data=st.data())
def test_the_mean_of_the_valid_lanes_equals_numpy_bit_for_bit_on_both_backends(cols: int, data: st.DataObject) -> None:
    valid = data.draw(st.integers(0, cols))
    rng = np.random.default_rng(data.draw(st.integers(0, 2**32 - 1)))
    values = rng.integers(-64, 65, size=(1, valid)).astype(np.float32)  # integers keep every sum exact
    expected = np.full((1, 1), np.nan, np.float32)
    if valid:
        expected = np.mean(values, axis=1, keepdims=True, dtype=np.float32)
    emulated, compiled = np.full((1, 1), -7.0, np.float32), np.full((1, 1), -7.0, np.float32)
    on_emulator("mean_row", [values, emulated], Blocks(1), cols=cols)
    on_c_backend("mean_row", [values, compiled], Blocks(1), cols=cols)
    assert_bits_equal(emulated, expected)
    assert_bits_equal(compiled, expected)


def test_an_element_whose_lanes_are_all_past_the_edge_is_nan_and_stored_on_both_backends() -> None:
    x = np.ones((5, 11), np.float32)
    emulated, compiled = np.full((5, 1), -7.0, np.float32), np.full((5, 1), -7.0, np.float32)
    on_emulator("mean_past_edge", [x, emulated], Tiles((5, 1), (4, 16)))
    on_c_backend("mean_past_edge", [x, compiled], Tiles((5, 1), (4, 16)))
    assert np.isnan(emulated).all()
    assert np.isnan(compiled).all()


def test_a_row_past_the_tensor_is_nan_while_the_rows_inside_keep_their_mean() -> None:
    x = np.ones((5, 11), np.float32)
    emulated, compiled = np.full((9, 1), -7.0, np.float32), np.full((9, 1), -7.0, np.float32)
    on_emulator("mean_row", [x, emulated], Tiles((9, 1), (1, 16)), cols=16)
    on_c_backend("mean_row", [x, compiled], Tiles((9, 1), (1, 16)), cols=16)
    for out in (emulated, compiled):
        np.testing.assert_array_equal(out[:5], np.ones((5, 1), np.float32))
        assert np.isnan(out[5:]).all()


def test_a_mean_of_lanes_that_cannot_be_invalid_is_the_plain_mean_on_both_backends() -> None:
    emulated, compiled = np.zeros((4, 1), np.float32), np.zeros((4, 1), np.float32)
    on_emulator("mean_of_full_lanes", [emulated], Tiles((4, 1), (4, 16)))
    on_c_backend("mean_of_full_lanes", [compiled], Tiles((4, 1), (4, 16)))
    np.testing.assert_array_equal(emulated, np.full((4, 1), 2.0, np.float32))
    np.testing.assert_array_equal(compiled, np.full((4, 1), 2.0, np.float32))


def test_the_mean_warns_about_its_division_on_a_target_that_flushes_denormals(monkeypatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PrecisionWarning)
        emit(capture(FILE, "mean_row", consts={"t": {"cols": 1024}}, target=APPLE), APPLE)
    messages = [str(w.message) for w in caught]
    assert any(message.startswith("mean_row on apple_m4: div flushes float32") for message in messages)

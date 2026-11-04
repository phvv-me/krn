"""Nullable columns, mixed precision and UTF-8 lists using collective-free CPU kernels."""

import logging
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import numpy as np

from krn import Items, targets
from krn.compiler.api import CPUProgram, build, capture
from krn.compiler.dtypes import Dtype
from krn.compiler.source import Module
from krn.data import columns as col
from krn.data.columns import Argument
from krn.data.strings import as_lists, as_strings, strings_arrays, strings_values
from krn.runtime.runtime import DTYPES

log = logging.getLogger("columns")
FILE = Path(col.__file__)
type Binding = tuple[tuple[str, Dtype], ...]


def _dtypes(arguments: Mapping[str, Argument]) -> dict[str, Dtype]:
    found: dict[str, Dtype] = {}
    for name, value in arguments.items():
        match value:
            case np.ndarray():
                found[name] = DTYPES[value.dtype]
            case Mapping():
                found.update(_dtypes({f"{name}.{field}": part for field, part in value.items()}))
    return found


@cache
def program(name: str, binding: Binding) -> CPUProgram:
    target = targets.CPU.of("avx512")
    return build(capture(FILE, name, target=target, dtypes=dict(binding)), target)


def launch(name: str, rows: int, arguments: list[Argument]) -> None:
    names = [parameter.arg for parameter in Module.read(FILE).kernel(name).args.args]
    binding = tuple(_dtypes(dict(zip(names, arguments, strict=True))).items())
    program(name, binding)(*arguments, launch=Items(rows))


def main() -> None:
    c = col.column_arrays([1, None, 3, None, 5], "i4")
    assert col.column_values(col.is_null(launch, c)) == [0, 1, 0, 1, 0]
    assert col.null_count(launch, c) == 2
    filled = col.fill_null(launch, c, 9)
    np.testing.assert_array_equal(filled.values[: filled.rows], np.array([1, 9, 3, 9, 5], "i4"))
    log.info("nulls filled: %s", col.column_values(filled))

    cond = col.column_arrays([True, False, None, True, False])
    left = col.column_arrays([1.5, 2.5, 3.5, None, 5.5], "f2")
    right = col.column_arrays([10, 20, 30, 40, 50], "f4")
    chosen = col.copy_if_else(launch, cond, left, right)
    assert chosen.values.dtype == np.dtype("f4")
    assert col.column_values(chosen) == [1.5, 20, None, None, 50]
    np.testing.assert_array_equal(
        chosen.values[:2], np.where(np.array([True, False]), left.values[:2].astype("f4"), right.values[:2])
    )
    log.info("mixed f16/f32 selection: %s", col.column_values(chosen))

    selected = col.apply_boolean_mask(launch, c, cond)
    assert col.column_values(selected) == [1, None]
    log.info("stable boolean selection: %s", col.column_values(selected))

    values = np.concatenate([np.array([2048], "f2"), np.ones(64, "f2")])
    halves = col.column_arrays(values)
    narrow = col.reduce(launch, halves, col.Reduction.SUM, acc="f2")
    wide = col.reduce(launch, halves, col.Reduction.SUM, acc="f4")
    assert wide == np.sum(values, dtype=np.float64) and narrow != wide
    log.info(
        "f16 stored values: f16 accumulation=%s; f32 accumulation=%s (reference=%s)",
        narrow,
        wide,
        np.sum(values, dtype=np.float64),
    )

    cumulative = col.scan(launch, filled, col.Scan.SUM, chunk=4)
    np.testing.assert_array_equal(
        cumulative.values[: cumulative.rows], np.cumsum(filled.values[: filled.rows], dtype=np.int64)
    )
    log.info("cumulative sum: %s", col.column_values(cumulative))

    texts = ["é🙂", None, "", "日本語"]
    lists = as_lists(strings_arrays(texts))
    taken = col.gather(launch, lists, [3, 0, 1, -1])
    assert strings_values(as_strings(taken)) == ["日本語".encode(), "é🙂".encode(), None, None]
    combined = col.concatenate(launch, [taken, col.head(launch, lists, 2)])
    expected = ["日本語", "é🙂", None, None, "é🙂", None]
    assert strings_values(as_strings(combined)) == [None if s is None else s.encode() for s in expected]
    log.info("UTF-8 gather and concatenate: %s", strings_values(as_strings(combined)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

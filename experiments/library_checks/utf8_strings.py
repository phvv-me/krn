"""UTF-8 text stays a struct of vectors inside a kernel, including variable-width characters.

The byte producers count, scan on the host, then fill a fresh buffer. The same layout supports a custom
code-point loop, full Unicode casing, literal search and replacement, and zero-copy split ranges.
"""

import logging
from collections.abc import Sequence
from functools import cache
from inspect import getsource
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from krn import Items, Vector, items, kernel, targets, u32
from krn.compiler.api import CPUProgram, build, capture
from krn.data.strings import (
    NOT_FOUND,
    Argument,
    Side,
    Strings,
    column,
    documents_arrays,
    sized,
    strings_arrays,
    strings_values,
)
from krn.data.unicode_tables import unicode_arrays

log = logging.getLogger("strings")
FILE = Path(__file__).resolve()
LIBRARY = FILE.parents[2] / "src" / "krn" / "data" / "strings.py"


@kernel
def non_ascii_bytes(s: Strings, out: Vector[u32]) -> None:
    """The bytes of each row that belong to characters outside ASCII."""
    for i in items(s.rows):
        count = u32(0)
        for cp, at in s.codepoints(i):
            if cp >= 128:
                count += s.width(at)
        out[i] = count


@cache
def program(name: str) -> CPUProgram:
    target = targets.CPU.of("avx512")
    if name == "non_ascii_bytes":
        # Capture follows relative imports only; keep the library and this kernel together in its source view.
        with TemporaryDirectory(prefix="krn-strings-") as folder:
            source = Path(folder) / "strings_example.py"
            source.write_text(LIBRARY.read_text() + "\n\n" + getsource(non_ascii_bytes.fn))
            return build(capture(source, name, target=target), target)
    return build(capture(LIBRARY, name, target=target), target)


def launch(name: str, rows: int, arguments: list[Argument]) -> None:
    program(name)(*arguments, launch=Items(rows))


def main() -> None:
    texts = ["  Straße éé 🙂  ", "ΑΣ", "İstanbul", "", "-42"]
    s = strings_arrays(texts)
    tables = unicode_arrays()
    for name, extra, expected in (
        ("to_upper", [tables], [v.upper() for v in texts]),
        (
            "replace",
            [np.frombuffer("é".encode(), np.uint8), np.frombuffer(b"E", np.uint8), NOT_FOUND],
            [v.replace("é", "E") for v in texts],
        ),
        ("pad", [20, ord("界"), Side.LEFT], [v.rjust(20, "界") for v in texts]),
        ("strip", [tables, Side.BOTH], [v.strip() for v in texts]),
    ):
        result = sized(launch, name, s.rows, s, *extra)
        assert strings_values(result) == [v.encode() for v in expected]
        log.info("%s: %s", name, expected)

    np.testing.assert_array_equal(
        column(launch, "non_ascii_bytes", s.rows, np.uint32, s),
        [sum(len(ch.encode()) for ch in v if ord(ch) >= 128) for v in texts],
    )
    pattern = np.frombuffer("é".encode(), np.uint8)
    np.testing.assert_array_equal(column(launch, "find", s.rows, np.int32, s, pattern), [v.find("é") for v in texts])
    log.info("custom code-point loop and find agree with Python")

    pieces = column(launch, "split_count", s.rows, np.uint32, s, pattern)
    documents = np.concatenate([np.zeros(1, np.uint32), np.cumsum(pieces, dtype=np.uint32)])
    starts, ends = (np.zeros(int(documents[-1]), np.uint32) for _ in range(2))
    launch("split_record", s.rows, [s._asdict(), pattern, documents, starts, ends])
    result = [bytes(s.chars[a:b]) for a, b in zip(starts, ends, strict=True)]
    assert result == [part.encode() for v in texts for part in v.split("é")]
    log.info("split: %s", result)

    groups: list[Sequence[str | None]] = [v.split("é") for v in texts] + [[], [None, "🙂"]]
    d = documents_arrays(groups)
    result = sized(launch, "join_strings", d.count, d, np.frombuffer(b"|", np.uint8))
    expected = ["|".join(v for v in group if v is not None).encode() for group in groups]
    assert strings_values(result) == expected
    log.info("join: %s", expected)

    numbers = ["-42", "+17", "9223372036854775807", "-9223372036854775808"]
    values = strings_arrays(numbers)
    np.testing.assert_array_equal(
        column(launch, "to_integers", values.rows, np.int64, values), [int(v) for v in numbers]
    )
    log.info("to_integers agrees with Python, including int64 boundaries")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

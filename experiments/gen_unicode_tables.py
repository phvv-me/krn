"""Generate `src/krn/data/unicode_tables.py`: character classes and full case mappings as paged tables.

Everything comes from the running CPython (`unicodedata` and the `str` methods), so the tables are the oracle the string
tests compare with, by construction, for `unicodedata.unidata_version`. A page is 256 code points and equal pages are
stored once, the paging of cutok's `PagedTable` (cutoken/tables.py in research/cuda-tokenization), which reads a value
in two loads. Case mappings are the unconditional full mappings of one code point (`str.upper`, `str.lower`,
`str.casefold`); the context rule of final sigma is code in the kernel, and the classes below carry what it needs.

    mb run --no-install python -m experiments.gen_unicode_tables          write the module
    mb run --no-install python -m experiments.gen_unicode_tables --check  exit 1 when the module is stale
"""

import base64
import inspect
import logging
import sys
import unicodedata
import zlib
from enum import IntFlag
from pathlib import Path
from textwrap import dedent

import cyclopts
import numpy as np

from krn.arithmetic import align_up

log = logging.getLogger("unicode_tables")
app = cyclopts.App()

OUTPUT = Path(__file__).resolve().parents[1] / "src" / "krn" / "data" / "unicode_tables.py"
SCALARS = 0x110000
PAGE_SHIFT = 8
MULTIPLE = 0x80000000
CAPITAL_SIGMA, FINAL_SIGMA = "Σ", "ς"
LINE = 96


class CharacterType(IntFlag):
    """Bits of `class_pages`: the types of libcudf's `string_character_types` plus what `str.lower` needs.

    CASED is cased and not case-ignorable, which is what CPython's final sigma rule counts as cased.
    """

    ALPHA = 1
    DECIMAL = 2
    DIGIT = 4
    NUMERIC = 8
    SPACE = 16
    CASED = 32
    CASE_IGNORABLE = 64
    ALNUM = ALPHA | DECIMAL | DIGIT | NUMERIC


def classes(c: str) -> int:
    """The `CharacterType` bits of one character as CPython defines them.

    CPython lowers a capital sigma to the final form when the nearest code point before it that is not case-ignorable is
    cased. After "A" that holds for a cased or a case-ignorable `c`, and after "1" only for a cased one that is not
    case-ignorable, which tells the two apart.
    """
    skipped_or_cased = (("A" + c + CAPITAL_SIGMA).lower()[-1]) == FINAL_SIGMA
    cased = (("1" + c + CAPITAL_SIGMA).lower()[-1]) == FINAL_SIGMA
    return (
        CharacterType.ALPHA * c.isalpha()
        | CharacterType.DECIMAL * c.isdecimal()
        | CharacterType.DIGIT * c.isdigit()
        | CharacterType.NUMERIC * c.isnumeric()
        | CharacterType.SPACE * c.isspace()
        | CharacterType.CASED * cased
        | CharacterType.CASE_IGNORABLE * (skipped_or_cased and not cased)
    )


def paged(values: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """(index, pages): `values` split into pages of 256, equal pages kept once, the all-zero page first."""
    size = 1 << PAGE_SHIFT
    padded = np.zeros(align_up(len(values), size), np.dtype(dtype))
    padded[: len(values)] = values
    unique, index = np.unique(padded.reshape(-1, size), axis=0, return_inverse=True)
    assert len(unique) <= 1 << 16 and not unique[0].any()
    return index.ravel().astype(np.uint16), unique.ravel()


class CasePool:
    """The code points of every multi code point mapping, each distinct result stored once."""

    def __init__(self) -> None:
        self.points: list[int] = []
        self.starts: dict[tuple[int, ...], int] = {}

    def value(self, c: str, mapped: str) -> int:
        """0 when `mapped` is `c`, the replacement when it is one code point, else a pointer into the pool."""
        if mapped == c:
            return 0
        points = tuple(map(ord, mapped))
        if len(points) == 1:
            return points[0]
        if points not in self.starts:
            self.starts[points] = len(self.points)
            self.points.extend(points)
        return MULTIPLE | len(points) << 24 | self.starts[points]


def dense_tables() -> dict[str, np.ndarray]:
    """One value per code point for the classes and the three case mappings, and the pool the long ones share."""
    methods = {"upper": str.upper, "lower": str.lower, "fold": str.casefold}
    flags, mappings, pool = (
        np.zeros(SCALARS, np.uint8),
        {n: np.zeros(SCALARS, np.uint32) for n in methods},
        CasePool(),
    )
    for cp in range(SCALARS):
        if 0xD800 <= cp < 0xE000:
            continue  # surrogates are not scalar values and UTF-8 cannot hold them
        c = chr(cp)
        flags[cp] = classes(c)
        for name, method in methods.items():
            mappings[name][cp] = pool.value(c, method(c))
    assert len(pool.points) < 1 << 24
    return {"class": flags, **mappings, "case_pool": np.array(pool.points, np.uint32)}


def literal(array: np.ndarray) -> str:
    """The text of one array: compressed, base 85, one string literal per line."""
    text = base64.b85encode(zlib.compress(array.tobytes(), 9)).decode()
    return "\n".join(f'        "{text[at : at + LINE]}"' for at in range(0, len(text), LINE))


HEAD = '''\
"""Unicode {version} character classes and full case mappings as paged tables, generated by
experiments/gen_unicode_tables.py from CPython's unicodedata and str methods. Do not edit.

A value is read in two loads, `pages[(index[cp >> 8] << 8) | (cp & 255)]`, the paging of cutok's PagedTable. A case page
holds 0 (unchanged), the one replacement code point, or MULTIPLE | n << 24 | start for the n (2 or 3) code points
`case_pool[start : start + n]`. `class_pages` holds `CharacterType` bits.
"""

import base64
import zlib
from enum import IntFlag

import numpy as np

UNICODE_VERSION = "{version}"
PAGE_SHIFT = 8
MULTIPLE = 0x80000000


{character_type}


def _array(dtype: str, text: str) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b85decode(text)), dtype).astype(dtype)


'''

TAIL = '''

def unicode_arrays() -> dict[str, np.ndarray]:
    """The fields of `strings.Unicode`, by name."""
    return {{
{fields}
    }}


def _read(table: str, codepoint: int) -> int:
    tables = unicode_arrays()
    index, pages = tables[f"{{table}}_index"], tables[f"{{table}}_pages"]
    return int(pages[(int(index[codepoint >> PAGE_SHIFT]) << PAGE_SHIFT) | (codepoint & 255)])


def character_type(codepoint: int) -> CharacterType:
    """The class bits of one code point."""
    return CharacterType(_read("class", codepoint))


def case_mapping(table: str, codepoint: int) -> str:
    """The full mapping of one code point in `upper`, `lower` or `fold`."""
    value = _read(table, codepoint)
    if value == 0:
        return chr(codepoint)
    if value < MULTIPLE:
        return chr(value)
    start, count = value & 0xFFFFFF, (value >> 24) & 0x7F
    return "".join(map(chr, CASE_POOL[start : start + count]))
'''


def render() -> str:
    """The text of the generated module."""
    dense = dense_tables()
    arrays: list[tuple[str, np.ndarray]] = []
    for name, dtype in (
        ("class", "uint8"),
        ("upper", "uint32"),
        ("lower", "uint32"),
        ("fold", "uint32"),
    ):
        index, pages = paged(dense[name], dtype)
        arrays += [(f"{name}_index", index), (f"{name}_pages", pages)]
    arrays.append(("case_pool", dense["case_pool"]))
    head = HEAD.format(
        version=unicodedata.unidata_version,
        character_type=dedent(inspect.getsource(CharacterType)).rstrip(),
    )
    body = "\n".join(
        f'{name.upper()} = _array(\n    "{array.dtype}",\n    (\n{literal(array)}\n    ),\n)\n'
        for name, array in arrays
    )
    fields = "\n".join(f'        "{name}": {name.upper()},' for name, _ in arrays)
    return head + body + TAIL.format(fields=fields)


@app.default
def main(*, check: bool = False) -> None:
    """Write the module, or with `check` only compare it with what would be written."""
    text = render()
    if check:
        if OUTPUT.read_text() != text:
            log.error("%s is stale for Unicode %s", OUTPUT, unicodedata.unidata_version)
            sys.exit(1)
        return
    OUTPUT.write_text(text)
    log.info("wrote %s (%d bytes) for Unicode %s", OUTPUT, len(text), unicodedata.unidata_version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()

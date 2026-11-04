"""One literal-operation table, with CPython oracles, runs on C, the emulator, and an optional CUDA card."""

import ast
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xxhash

from experiments.gen_unicode_tables import OUTPUT, render
from krn import Items, targets
from krn.compiler.api import capture
from krn.data.columns import ALIGNMENT
from krn.data.strings import (
    NOT_FOUND,
    Argument,
    Launch,
    Side,
    StringsArrays,
    as_lists,
    as_strings,
    column,
    documents_arrays,
    head,
    sized,
    strings_arrays,
    strings_values,
    tail,
)
from krn.data.unicode_tables import (
    UNICODE_VERSION,
    CharacterType,
    case_mapping,
    character_type,
    unicode_arrays,
)
from krn.emulator.emulate import Emulated
from krn.emulator.emulate import build as emulate
from krn.runtime.cpu import CPUProgram
from tests.conftest import build_kernel
from tests.helpers.devices import CUDARun

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice

FILE = Path(__file__).resolve().parents[2] / "src" / "krn" / "data" / "strings.py"
SPACE = "\t\n\v\f\r\x1c\x1d\x1e\x1f \x85\xa0\u1680" + "".join(map(chr, range(0x2000, 0x200B)))
SPACE += "\u2028\u2029\u202f\u205f\u3000"
CLASSES = (
    "abcXYZ019 ",
    "éÀøΩΑΣςЯя中文",
    "\U00010400\U0001e900",
    "🙂🚀\U0001f3fd\u200d",
    "".join(map(chr, range(0x0300, 0x0370))) + "\u20d0",
    "ßİŉǰΐǅﬁ",
    SPACE,
    "١٢²½Ⅻ",
)
INTEGER_ROWS = (
    "",
    "+",
    "-",
    "00",
    "-0",
    "+0",
    "9223372036854775807",
    "9223372036854775808",
    "-9223372036854775808",
    "-9223372036854775809",
    " 1",
    "1_0",
    "١٢",
    "+123",
    "12a",
    "0" * 80,
)
TABLES = unicode_arrays()
type Oracle = Callable[[str], str | bytes | int]


def vector(text: str | bytes) -> np.ndarray:
    return np.frombuffer(text.encode() if isinstance(text, str) else text, np.uint8).copy()


@cache
def corpus(small: bool, long: bool) -> tuple[tuple[str | None, ...], ...]:
    corpora: list[tuple[str | None, ...]] = []
    for seed in (9, 27):
        rng = np.random.default_rng(seed)
        rows = [
            "",
            "ΑΣ",
            "ΑΣΑ",
            "ΑΣ.",
            "Σ",
            "ΑΣ A",
            "\u0391\u0301Σ\u0301",
            "ΑΣ\u0301\u0391",
            "A'Σ",
            "AΣ'B",
        ]
        rows += [
            SPACE,
            SPACE + "word" + SPACE,
            "aéé🙂é",
            "éé",
            "-é",
            "+é",
            "ßİŉǰΐǅﬁ",
            "\U00010400\U0001e900",
            "١٢²½Ⅻ",
            "\u0301Σ",
            "AΣ\u0301.",
            "AΣ\u0345B",
            *INTEGER_ROWS,
        ]
        for n in (0, 1, *range(2, 13)):
            rows.append("".join(str(rng.choice(list(CLASSES[k % len(CLASSES)]))) for k in range(n)))
        if not small:
            rows += ["x" * 63, "x" * 64, "x" * 65, "é🙂" * 682 + "abcd"]
            rows += ["".join(str(rng.choice(list(group))) for _ in range(11)) for group in CLASSES]
        if long:
            rows.append("a" * 65537)
        corpora.extend((tuple(rows), (*rows[:10], None, *rows[10:], None)))
    return tuple(corpora)


@dataclass(frozen=True)
class Operation:
    name: str
    kernels: tuple[str, ...]
    oracle: Oracle | None = None
    extra: tuple[Argument, ...] = ()
    dtype: str | None = None


def producer(name: str, oracle: Oracle, *extra: Argument) -> Operation:
    return Operation(name, (f"{name}_count", f"{name}_fill"), oracle, extra)


def scalar(name: str, oracle: Oracle, dtype: str, *extra: Argument) -> Operation:
    return Operation(name, (name,), oracle, extra, dtype)


def integer(value: str) -> int:
    digits = value[1:] if value.startswith(("+", "-")) else value
    good = bool(digits) and all("0" <= ch <= "9" for ch in digits)
    return int(good and -(1 << 63) <= int(value) < (1 << 63))


def polynomial(value: str) -> int:
    result = 0
    for byte in value.encode():
        result = (result * 0x100000001B3 + byte + 1) & ((1 << 64) - 1)
    return result


OPERATIONS = [
    *(
        producer("slice_strings", lambda s, a=a, b=b: s[a:b], a, b)
        for a, b in (
            (0, 0x7FFFFFFF),
            (1, 4),
            (-4, -1),
            (-200, 200),
            (4, 1),
            (-2147483648, 2147483647),
        )
    ),
    *(
        producer("slice_bytes", lambda s, a=a, b=b: s.encode()[a:b], a, b)
        for a, b in ((1, 4), (-4, -1), (0, 0), (-200, 200), (4, 1))
    ),
    producer("repeat_strings", lambda s: s * 3, 3),
    producer("repeat_strings", lambda s: s * 0, 0),
    producer("pad", lambda s: s.rjust(17, "🙂"), 17, ord("🙂"), Side.LEFT),
    producer("pad", lambda s: s.ljust(9, "é"), 9, ord("é"), Side.RIGHT),
    producer("zfill", lambda s: s.zfill(22), 22),
    *(
        producer("strip", lambda s, method=method: getattr(s, method)(), TABLES, side)
        for method, side in (("lstrip", Side.LEFT), ("rstrip", Side.RIGHT), ("strip", Side.BOTH))
    ),
    *(
        producer(name, lambda s, method=method: getattr(s, method)(), TABLES)
        for name, method in (
            ("to_upper", "upper"),
            ("to_lower", "lower"),
            ("casefold", "casefold"),
        )
    ),
    *(
        producer(
            "replace",
            lambda s, n=n, t=t, r=r: s.replace(t, r, n),
            vector(t),
            vector(r),
            NOT_FOUND if n < 0 else n,
        )
        for t, r, n in (("é", "🙂X", -1), ("é", "", 1), ("a", "xy", 0), ("aa", "z", -1))
    ),
    scalar("count_bytes", lambda s: len(s.encode()), "uint32"),
    scalar("count_characters", len, "uint32"),
    scalar("hash_strings", polynomial, "uint64"),
    scalar("hash_strings_xxh64", lambda s: xxhash.xxh64_intdigest(s.encode(), 0), "uint64"),
    scalar("is_integer", integer, "uint8"),
    scalar("to_integers", lambda s: int(s) if integer(s) else 0, "int64"),
    *(
        scalar(
            "all_characters_of_type",
            lambda s, method=method: int(getattr(s, method)()),
            "uint8",
            TABLES,
            int(bit),
        )
        for method, bit in (
            ("isalpha", CharacterType.ALPHA),
            ("isdecimal", CharacterType.DECIMAL),
            ("isdigit", CharacterType.DIGIT),
            ("isnumeric", CharacterType.NUMERIC),
            ("isspace", CharacterType.SPACE),
            ("isalnum", CharacterType.ALNUM),
        )
    ),
    *(
        scalar(
            name,
            lambda s, pattern=pattern, method=method: int(getattr(s, method)(pattern)),
            dtype,
            vector(pattern),
        )
        for name, method, dtype in (
            ("starts_with", "startswith", "uint8"),
            ("ends_with", "endswith", "uint8"),
            ("contains", "__contains__", "uint8"),
            ("find", "find", "int32"),
            ("rfind", "rfind", "int32"),
        )
        for pattern in ("", "é", "🙂", "absent")
    ),
    Operation("concatenate", ("concatenate_count", "concatenate_fill")),
    Operation("join_strings", ("join_strings_count", "join_strings_fill")),
    Operation("split", ("split_count", "split_record")),
    Operation("reverse", ("reverse",)),
    Operation("ordering", ("equal", "compare")),
    Operation("validation", ("first_invalid_byte", "is_valid_utf8")),
    Operation("from_integers", ("from_integers_count", "from_integers_fill")),
    Operation("head_tail", ("slice_strings_count", "slice_strings_fill")),
]


def check_standard(op: Operation, launch: Launch, rows: Sequence[str | None]) -> None:
    assert op.oracle is not None
    s = strings_arrays(rows)
    if op.dtype is None:
        expected = [None if value is None else op.oracle(value) for value in rows]
        encoded = [value.encode() if isinstance(value, str) else value for value in expected]
        result = sized(launch, op.name, s.rows, s, *op.extra)
        assert strings_values(result) == encoded
        for i, value in enumerate(rows):
            if value is None:
                assert result.offsets[i] == result.offsets[i + 1]
    else:
        expected = [op.oracle(value or "") for value in rows]
        np.testing.assert_array_equal(column(launch, op.name, s.rows, op.dtype, s, *op.extra), expected)


def invalid_corpus() -> list[bytes]:
    bad = [
        b"\xc0\x80",
        b"\xe0\x80\x80",
        b"\xed\xa0\x80",
        b"\xf4\x90\x80\x80",
        b"\x80",
        b"\xc2",
        b"\xe1\x80",
        b"\xf0\x90\x80",
        b"ok\xc2a",
        b"\xe1a\x80",
        b"\xf1\x80a\x80",
        b"\xf1\x80\x80a",
        b"",
        b"ok",
    ]
    bad += [bytes([lead]) for lead in range(0xF5, 0x100)]
    bad += [
        bytes([lead, second, *([0x80] * (width - 2))])
        for lead, width in (
            (0xC2, 2),
            (0xDF, 2),
            (0xE0, 3),
            (0xE1, 3),
            (0xEC, 3),
            (0xED, 3),
            (0xEE, 3),
            (0xEF, 3),
            (0xF0, 4),
            (0xF1, 4),
            (0xF3, 4),
            (0xF4, 4),
        )
        for second in (0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0)
    ]
    for seed in (3, 17):
        rng = np.random.default_rng(seed)
        bad += [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes() for n in range(20)]
        for value in ("é中🙂", "ΑΣﬁ", "\U00010400\u200d"):
            encoded = value.encode()
            for i in range(len(encoded)):
                mutated = bytearray(encoded)
                mutated[i] = int(rng.integers(256))
                bad.append(bytes(mutated))
    return bad


def first_bad(value: bytes) -> int:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError as error:
        return error.start
    return NOT_FOUND


def check_special(op: Operation, launch: Launch, small: bool, long: bool) -> None:
    if op.name == "validation":
        values = invalid_corpus()
        expected = [first_bad(value) for value in values]
        s = strings_arrays(values)
        np.testing.assert_array_equal(column(launch, "first_invalid_byte", s.rows, np.uint32, s), expected)
        np.testing.assert_array_equal(
            column(launch, "is_valid_utf8", s.rows, np.uint8, s),
            [value == NOT_FOUND for value in expected],
        )
    elif op.name == "join_strings":
        groups: list[list[str | None]] = [
            [],
            [None],
            ["", None, "é", "", "🙂"],
            ["a"],
            [],
            [None, "b", None],
        ]
        d = documents_arrays(groups)
        for sep in ("", "界"):
            result = sized(launch, "join_strings", d.count, d, vector(sep))
            assert strings_values(result) == [sep.join(v for v in group if v is not None).encode() for group in groups]
    elif op.name == "from_integers":
        values = np.array([0, 1, -1, 9, 10, -10, 1234567890, -(1 << 63), (1 << 63) - 1], np.int64)
        assert strings_values(sized(launch, "from_integers", len(values), values)) == [str(v).encode() for v in values]
    else:
        for rows in corpus(small, long):
            s = strings_arrays(rows)
            if op.name == "reverse":
                chars = np.full_like(s.chars, 0xFF)
                launch("reverse", s.rows, [s._asdict(), chars])
                result = StringsArrays(chars, s.offsets, s.valid, s.rows)
                assert strings_values(result) == [None if v is None else v[::-1].encode() for v in rows]
            elif op.name in ("concatenate", "ordering"):
                other = list(reversed(rows))
                b = strings_arrays(other)
                if op.name == "concatenate":
                    for sep in ("", "界"):
                        result = sized(launch, "concatenate", s.rows, s, b, vector(sep))
                        assert strings_values(result) == [
                            None if a is None or b is None else (a + sep + b).encode()
                            for a, b in zip(rows, other, strict=True)
                        ]
                else:
                    avalues, bvalues = [v or "" for v in rows], [v or "" for v in other]
                    for name, dtype, expected in (
                        (
                            "equal",
                            np.uint8,
                            [a == b for a, b in zip(avalues, bvalues, strict=True)],
                        ),
                        (
                            "compare",
                            np.int32,
                            [(a > b) - (a < b) for a, b in zip(avalues, bvalues, strict=True)],
                        ),
                    ):
                        np.testing.assert_array_equal(column(launch, name, s.rows, dtype, s, b), expected)
            elif op.name == "head_tail":
                for n in (0, 1, 4):
                    assert strings_values(head(launch, s, n)) == [None if v is None else v[:n].encode() for v in rows]
                    assert strings_values(tail(launch, s, n)) == [
                        None if v is None else (v[-n:] if n else "").encode() for v in rows
                    ]
            elif op.name == "split":
                check_split(launch, [*rows, "éé", "🙂éé🙂é", "é", "", None])


def check_split(launch: Launch, values: Sequence[str | None]) -> None:
    s = strings_arrays(values)
    for sep in ("é", "🙂", "zz"):
        pieces = [None if v is None else v.split(sep) for v in values]
        counts = column(launch, "split_count", s.rows, np.uint32, s, vector(sep))
        np.testing.assert_array_equal(counts, [0 if p is None else len(p) for p in pieces])
        documents = np.concatenate([np.zeros(1, np.uint32), np.cumsum(counts, dtype=np.uint32)])
        starts, ends = (np.full(int(documents[-1]), NOT_FOUND, np.uint32) for _ in range(2))
        launch("split_record", s.rows, [s._asdict(), vector(sep), documents, starts, ends])
        expected = [part.encode() for row in pieces if row is not None for part in row]
        assert [bytes(s.chars[a:b]) for a, b in zip(starts, ends, strict=True)] == expected
        for row in range(s.rows):
            for piece in range(int(documents[row]), int(documents[row + 1])):
                assert s.offsets[row] <= starts[piece] <= ends[piece] <= s.offsets[row + 1]


def check(op: Operation, launch: Launch, *, small: bool = False, long: bool = False) -> None:
    if op.oracle is not None:
        for rows in corpus(small, long):
            check_standard(op, launch, rows)
    else:
        check_special(op, launch, small, long)


@cache
def c_program(name: str) -> CPUProgram:
    return build_kernel(FILE, name)


def c_launch(name: str, rows: int, arguments: list[Argument]) -> None:
    c_program(name)(*arguments, launch=Items(rows))


@cache
def emulator_program(name: str) -> Emulated:
    target = targets.Target.of("sm_90a")
    return emulate(capture(FILE, name, target=target), target)


def emulator_launch(name: str, rows: int, arguments: list[Argument]) -> None:
    emulator_program(name).run(Items(rows), arguments)


@pytest.mark.parametrize("op", OPERATIONS, ids=lambda op: op.name)
def test_c(op: Operation) -> None:
    check(op, c_launch, long=True)


@pytest.mark.parametrize("op", OPERATIONS, ids=lambda op: op.name)
def test_emulator(op: Operation) -> None:
    check(op, emulator_launch, small=True)


@pytest.fixture(scope="module")
def card(gpu: CUDADevice) -> CUDADevice:
    return gpu


@pytest.fixture(scope="module")
def card_launch(card: CUDADevice) -> Launch:
    @cache
    def program(name: str) -> CUDARun:
        return CUDARun(card, capture(FILE, name, target=card.target))

    def launch(name: str, rows: int, arguments: list[Argument]) -> None:
        program(name).run(Items(rows), arguments)

    return launch


@pytest.mark.parametrize("op", OPERATIONS, ids=lambda op: op.name)
def test_card(op: Operation, card_launch: Launch) -> None:
    check(op, card_launch)


def test_kernel_table_is_complete() -> None:
    functions = ast.parse(FILE.read_text()).body
    kernels = {
        fn.name
        for fn in functions
        if isinstance(fn, ast.FunctionDef)
        and any(isinstance(d, ast.Name) and d.id == "kernel" for d in fn.decorator_list)
    }
    entries = {name for op in OPERATIONS for name in op.kernels}
    reachable = set(entries)
    functions = {fn.name: fn for fn in ast.parse(FILE.read_text()).body if isinstance(fn, ast.FunctionDef)}
    pending = list(entries)
    while pending:
        for call in ast.walk(functions[pending.pop()]):
            if isinstance(call, ast.Name):
                name = call.id
                if name in functions and name not in reachable:
                    reachable.add(name)
                    pending.append(name)
    assert kernels <= reachable
    assert entries <= kernels


def test_generated_tables_are_current_and_match_cpython() -> None:
    assert render() == OUTPUT.read_text()
    sample = [*range(0, 0x110000, 13), *map(ord, "ßİŉǰΐǅﬁΣςᵃ́Ⅻ²½٣")]
    for cp in sample:
        if 0xD800 <= cp < 0xE000:
            continue
        c = chr(cp)
        assert [case_mapping(table, cp) for table in ("upper", "lower", "fold")] == [
            c.upper(),
            c.lower(),
            c.casefold(),
        ]
        kind = character_type(cp)
        assert bool(kind & CharacterType.ALPHA) == c.isalpha()
        assert bool(kind & CharacterType.DECIMAL) == c.isdecimal()
        assert bool(kind & CharacterType.DIGIT) == c.isdigit()
        assert bool(kind & CharacterType.NUMERIC) == c.isnumeric()
        assert bool(kind & CharacterType.SPACE) == c.isspace()


def test_host_contract() -> None:
    assert unicodedata.unidata_version == UNICODE_VERSION
    assert len(SPACE) == 29 and all(ch.isspace() for ch in SPACE)
    s = strings_arrays(["", "a"])
    assert s.valid[0] == 0b11 and not s.valid[1:].any()
    for values in ([], ["", "a"], [None]):
        s = strings_arrays(values)
        for arrays in (s, sized(c_launch, "repeat_strings", s.rows, s, 1)):
            assert arrays.offsets.dtype == np.int32
            for buffer in (arrays.chars, arrays.offsets, arrays.valid):
                assert buffer.nbytes > 0 and buffer.nbytes % ALIGNMENT == 0
            lists = as_lists(arrays)
            assert lists.values is arrays.chars and lists.offsets is arrays.offsets and lists.valid is arrays.valid
            roundtrip = as_strings(lists)
            assert roundtrip.rows == arrays.rows
            assert all(a is b for a, b in zip(roundtrip[:3], arrays[:3], strict=True))
    values = [None if i % 5 == 0 else "é" for i in range(70)]
    assert strings_values(strings_arrays(values)) == [None if v is None else v.encode() for v in values]
    s = strings_arrays(["abc"])
    with pytest.raises(ValueError):
        sized(c_launch, "replace", s.rows, s, vector(""), vector("x"), NOT_FOUND)
    with pytest.raises(ValueError):
        column(c_launch, "split_count", s.rows, np.uint32, s, vector(""))

    def overflow(name: str, rows: int, args: list[Argument]) -> None:
        assert name.endswith("_count")
        lengths = args[-1]
        assert isinstance(lengths, np.ndarray)
        lengths[:] = NOT_FOUND

    with pytest.raises(OverflowError):
        sized(overflow, "repeat_strings", 2, strings_arrays(["a", "b"]), 2)

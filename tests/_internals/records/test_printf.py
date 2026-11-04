"""The reference renderer prints what C's printf prints, checked against this machine's libc for every accepted spec.

Variadic calls cannot be made through ctypes on every ABI (Apple arm64 passes variadic arguments on the stack), so the
test calls vsnprintf with a va_list it builds for the platform: every argument word on the stack, which all ABIs read.
"""

import ctypes
import platform
import struct

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn._internals.records import Field, Message, Spec, Tag
from krn._internals.records.tags import WORD_MASK
from tests._internals.records.strategies import spec_cases, values_for


def _abi() -> str:
    machine = platform.machine()
    if platform.system() == "Darwin" and machine == "arm64":
        return "darwin-arm64"
    if machine == "x86_64":
        return "sysv"
    if machine == "aarch64":
        return "aapcs64"
    return "unknown"


class _SysVVaList(ctypes.Structure):
    _fields_ = [
        ("gp_offset", ctypes.c_uint32),
        ("fp_offset", ctypes.c_uint32),
        ("overflow_arg_area", ctypes.c_void_p),
        ("reg_save_area", ctypes.c_void_p),
    ]


class _AAPCS64VaList(ctypes.Structure):
    _fields_ = [
        ("stack", ctypes.c_void_p),
        ("gr_top", ctypes.c_void_p),
        ("vr_top", ctypes.c_void_p),
        ("gr_offs", ctypes.c_int32),
        ("vr_offs", ctypes.c_int32),
    ]


_ABI = _abi()
pytestmark = pytest.mark.skipif(_ABI == "unknown", reason=f"no va_list layout for {platform.machine()}")
_libc = ctypes.CDLL(None)
_libc.vsnprintf.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_void_p]
_libc.vsnprintf.restype = ctypes.c_int


def _c_render(pattern: bytes, words: list[int]) -> bytes:
    """What libc's vsnprintf writes for `pattern` with these argument words, passed as the platform's va_list."""
    area = (ctypes.c_uint64 * len(words))(*words)
    address = ctypes.addressof(area)
    if _ABI == "darwin-arm64":
        va_list = address
    elif _ABI == "sysv":
        frame = _SysVVaList(gp_offset=48, fp_offset=176, overflow_arg_area=address)
        va_list = ctypes.addressof(frame)
    else:
        frame = _AAPCS64VaList(stack=address, gr_offs=0, vr_offs=0)
        va_list = ctypes.addressof(frame)
    out = ctypes.create_string_buffer(512)
    count = _libc.vsnprintf(out, len(out), pattern, va_list)
    return out.raw[:count]


def _c_pattern(spec: Spec, tag: Tag, letter: str) -> bytes:
    """The C conversion for a spec: 64-bit integers take `ll`, and an integer printed as unsigned takes `u`."""
    flags = ("-" if spec.align == "<" else "") + spec.sign + ("0" if spec.zero else "")
    width = "" if spec.width is None else str(spec.width)
    precision = "" if spec.precision is None else f".{spec.precision}"
    if letter in "xXo":
        conversion = f"ll{letter}"
    elif tag is Tag.INT and letter in "di":
        conversion = "lld"
    elif letter in "diu":
        conversion = "llu"
    else:
        conversion = letter
    return f"%{flags}{width}{precision}{conversion}".encode("ascii")


def _word(tag: Tag, value: float | int | bool) -> int:
    if tag in (Tag.FLOAT32, Tag.FLOAT64):
        return struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    return int(value) & WORD_MASK


@settings(max_examples=500, deadline=None)
@given(spec_cases(), st.data())
def test_reference_rendering_matches_libc_printf(case: tuple, data: st.DataObject) -> None:
    argument, text = case
    message = Message.of("{:" + text + "}", [argument])
    field: Field = message.fields[0]
    if field.tag is Tag.STR_CONSTANT:
        constant = ctypes.create_string_buffer(field.text.encode("ascii"))
        expected = message.render([])
        actual = _c_render(_c_pattern(field.spec, field.tag, field.letter), [ctypes.addressof(constant)])
    else:
        value = data.draw(values_for(field.tag, field.letter))
        expected = message.render([value])
        actual = _c_render(_c_pattern(field.spec, field.tag, field.letter), [_word(field.tag, value)])
    assert expected.encode("ascii") == actual


@pytest.mark.parametrize(
    "text,tag,value",
    [
        ("d", Tag.INT, -(2**63)),
        ("x", Tag.INT, -1),
        ("u", Tag.INT, -1),
        ("d", Tag.UINT, 2**64 - 1),
        ("+.3e", Tag.FLOAT32, 0.0),
        ("08.2f", Tag.FLOAT64, -3.14159),
        ("<6c", Tag.UINT, 65),
        ("c", Tag.UINT, 0),
    ],
)
def test_known_hard_cases_match_libc(text: str, tag: Tag, value: int | float) -> None:
    message = Message.of("{:" + text + "}", [tag])
    field = message.fields[0]
    assert message.render([value]).encode("ascii") == _c_render(
        _c_pattern(field.spec, tag, field.letter), [_word(tag, value)]
    )


def test_constant_text_with_a_precision_matches_libc() -> None:
    message = Message.of("{:5.2s}", ["abcdefg"])
    constant = ctypes.create_string_buffer(b"abcdefg")
    actual = _c_render(_c_pattern(message.fields[0].spec, Tag.STR_CONSTANT, "s"), [ctypes.addressof(constant)])
    assert message.render([]).encode("ascii") == actual

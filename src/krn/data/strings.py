"""UTF-8 rows as flat vectors, with literal operations and two-pass byte producers.

Except for the two validators, character operations require well-formed UTF-8. Positions are code points unless
the name says bytes. Null rows have equal adjacent offsets; byte producers preserve that invariant. Buffers use
i32 offsets and the host refuses more than 2**31 - 1 bytes. Validity is always allocated, and every buffer has a
positive multiple of 64 bytes of capacity. No operation allocates in a kernel. Case mappings are the full mappings
of Python's `str.upper`, `str.lower` (with final sigma) and `str.casefold` (conditions C and F), from
the tables of `unicode_tables.py`; there is no simple-mapping variant.
"""

from collections.abc import Iterator, Mapping, Sequence
from enum import IntEnum, auto
from typing import NamedTuple

import numpy as np

from krn import Vector, items, kernel, struct
from krn.frontend.types import i32, i64, u8, u16, u32, u64
from krn.hashlib import xxh64_bytes

from .columns import Argument as Argument
from .columns import Launch as Launch
from .columns import ListsArrays, empty, offsets_array, padded, validity_bits, validity_words


@struct
class Strings:
    chars: Vector[u8]
    offsets: Vector[i32]
    valid: Vector[u32]
    rows: u32

    def is_valid(self, row: u32) -> bool:
        """Allocated validity gives one specialization and kernel signature; no empty buffer reaches a launch."""
        word = self.valid[row // 32]
        bit = row % 32
        for step in (16, 8, 4, 2, 1):
            if bit >= step:
                word >>= step
                bit -= step
        return (word & 1) != 0

    def begin(self, row: u32) -> u32:
        return u32(self.offsets[row])

    def end(self, row: u32) -> u32:
        return u32(self.offsets[row + 1])

    def width(self, at: u32) -> u32:
        lead = self.chars[at]
        result = u32(1)
        if lead >= 0xF0:
            result = 4
        elif lead >= 0xE0:
            result = 3
        elif lead >= 0xC0:
            result = 2
        return u32(result)

    def codepoint(self, at: u32) -> u32:
        lead = u32(self.chars[at])
        result = lead
        if lead >= 0xF0:
            result = ((lead & 7) << 18) | ((u32(self.chars[at + 1]) & 63) << 12)
            result |= ((u32(self.chars[at + 2]) & 63) << 6) | (u32(self.chars[at + 3]) & 63)
        elif lead >= 0xE0:
            result = ((lead & 15) << 12) | ((u32(self.chars[at + 1]) & 63) << 6)
            result |= u32(self.chars[at + 2]) & 63
        elif lead >= 0xC0:
            result = ((lead & 31) << 6) | (u32(self.chars[at + 1]) & 63)
        return result

    def codepoints(self, row: u32) -> Iterator[tuple[u32, u32]]:
        """Yield the scalar and its absolute byte position."""
        at = self.begin(row)
        while at < self.end(row):
            yield self.codepoint(at), at
            at += self.width(at)

    def previous(self, at: u32, floor: u32) -> u32:
        result = at
        if result > floor:
            result -= 1
            while result > floor and (self.chars[result] & 0xC0) == 0x80:
                result -= 1
        return result


@struct
class Documents(Strings):
    documents: Vector[i32]
    count: u32


@struct
class OtherStrings(Strings):
    """The second column has its own C typedef until the renderer deduplicates struct declarations."""


@struct
class Unicode:
    class_index: Vector[u16]
    class_pages: Vector[u8]
    upper_index: Vector[u16]
    upper_pages: Vector[u32]
    lower_index: Vector[u16]
    lower_pages: Vector[u32]
    fold_index: Vector[u16]
    fold_pages: Vector[u32]
    case_pool: Vector[u32]

    def character_type(self, cp: u32) -> u32:
        return u32(self.class_pages[(u32(self.class_index[cp >> 8]) << 8) | (cp & 255)])


@kernel
def utf8(cp: u32) -> Iterator[u8]:
    if cp < 0x80:
        yield u8(cp)
    else:
        if cp < 0x800:
            yield u8(u32(0xC0) | (cp >> 6))
        else:
            if cp < 0x10000:
                yield u8(u32(0xE0) | (cp >> 12))
            else:
                yield u8(u32(0xF0) | (cp >> 18))
                yield u8(u32(0x80) | ((cp >> 12) & 63))
            yield u8(u32(0x80) | ((cp >> 6) & 63))
        yield u8(u32(0x80) | (cp & 63))


@kernel
def slice_bound(index: i32, length: u32) -> u32:
    """Widen before adding so negative bounds also work on buffers larger than INT32_MAX."""
    bound = i64(index)
    if bound < 0:
        bound += i64(length)
    return u32(max(i64(0), min(bound, i64(length))))


@kernel
def slice_strings_bytes(s: Strings, row: u32, start: i32, stop: i32) -> Iterator[u8]:
    if s.is_valid(row):
        length = u32(sum(1 for _ in s.codepoints(row)))
        first = slice_bound(start, length)
        last = slice_bound(stop, length)
        if stop == 0x7FFFFFFF:
            last = length
        for k, (_, at) in enumerate(s.codepoints(row)):
            if k >= first and k < last:
                for j in range(s.width(at)):
                    yield s.chars[at + j]


@kernel
def slice_bytes_bytes(s: Strings, row: u32, start: i32, stop: i32) -> Iterator[u8]:
    if s.is_valid(row):
        base = s.begin(row)
        length = s.end(row) - base
        last = slice_bound(stop, length)
        if stop == 0x7FFFFFFF:
            last = length
        for at in range(base + slice_bound(start, length), base + last):
            yield s.chars[at]


@kernel
def concatenate_bytes(a: Strings, b: OtherStrings, row: u32, separator: Vector[u8]) -> Iterator[u8]:
    if a.is_valid(row) and b.is_valid(row):
        for at in range(a.begin(row), a.end(row)):
            yield a.chars[at]
        for at in range(separator.shape[0]):
            yield separator[at]
        for at in range(b.begin(row), b.end(row)):
            yield b.chars[at]


@kernel
def join_strings_bytes(d: Documents, row: u32, separator: Vector[u8]) -> Iterator[u8]:
    seen = False
    for i in range(u32(d.documents[row]), u32(d.documents[row + 1])):
        piece = u32(i)
        if d.is_valid(piece):
            if seen:
                for at in range(separator.shape[0]):
                    yield separator[at]
            for at in range(d.begin(piece), d.end(piece)):
                yield d.chars[at]
            seen = True


@kernel
def repeat_strings_bytes(s: Strings, row: u32, count: u32) -> Iterator[u8]:
    if s.is_valid(row):
        for _ in range(count):
            for at in range(s.begin(row), s.end(row)):
                yield s.chars[at]


@kernel
def pad_bytes(s: Strings, row: u32, width: u32, fill: u32, side: u32) -> Iterator[u8]:
    if s.is_valid(row):
        length = u32(sum(1 for _ in s.codepoints(row)))
        extra = u32(0)
        if width > length:
            extra = width - length
        if side == 0:
            for _ in range(extra):
                for byte in utf8(fill):
                    yield byte
        for at in range(s.begin(row), s.end(row)):
            yield s.chars[at]
        if side == 1:
            for _ in range(extra):
                for byte in utf8(fill):
                    yield byte


@kernel
def zfill_bytes(s: Strings, row: u32, width: u32) -> Iterator[u8]:
    if s.is_valid(row):
        length = u32(sum(1 for _ in s.codepoints(row)))
        first = s.begin(row)
        end = s.end(row)
        if first < end and (s.chars[first] == 43 or s.chars[first] == 45):
            yield s.chars[first]
            first += 1
        if width > length:
            for _ in range(width - length):
                yield u8(48)
        for at in range(first, end):
            yield s.chars[at]


@kernel
def matches(s: Strings, at: u32, end: u32, pattern: Vector[u8]) -> bool:
    """Bounds belong here so callers never speculate a read past a row."""
    length = u32(pattern.shape[0])
    same = length <= end - at
    j = u32(0)
    while j < length and same:
        same = s.chars[at + j] == pattern[j]
        j += 1
    return same


@kernel
def replace_bytes(s: Strings, row: u32, target: Vector[u8], replacement: Vector[u8], maxrepl: u32) -> Iterator[u8]:
    if s.is_valid(row):
        at = s.begin(row)
        end = s.end(row)
        replaced = u32(0)
        while at < end:
            if replaced < maxrepl and matches(s, at, end, target):
                for j in range(replacement.shape[0]):
                    yield replacement[j]
                at += u32(target.shape[0])
                replaced += 1
            else:
                yield s.chars[at]
                at += 1


@kernel
def from_integers_bytes(values: Vector[i64], row: u32) -> Iterator[u8]:
    value = values[row]
    magnitude = u64(value)
    if value < 0:
        yield u8(45)
        magnitude = u64(0) - magnitude
    power = u64(1)
    while magnitude // power >= 10:
        power *= 10
    while power > 0:
        yield u8(48 + magnitude // power % 10)
        power //= 10


@kernel
def count_bytes(s: Strings, out: Vector[u32]) -> None:
    for i in items(s.rows):
        out[i] = s.end(i) - s.begin(i)


@kernel
def count_characters(s: Strings, out: Vector[u32]) -> None:
    for i in items(s.rows):
        out[i] = u32(sum(1 for _ in s.codepoints(i)))


@kernel
def invalid_byte(s: Strings, row: u32) -> u32:
    """Unicode Table 3-7; a bad sequence is reported at its opening byte, including truncation."""
    at = s.begin(row)
    end = s.end(row)
    bad = u32(0xFFFFFFFF)
    while at < end and bad == 0xFFFFFFFF:
        lead = u32(s.chars[at])
        width = u32(1)
        lo = u32(0x80)
        hi = u32(0xBF)
        good = True
        if lead < 0x80:
            width = 1
        elif lead >= 0xC2 and lead <= 0xDF:
            width = 2
        elif lead >= 0xE0 and lead <= 0xEF:
            width = 3
            if lead == 0xE0:
                lo = 0xA0
            if lead == 0xED:
                hi = 0x9F
        elif lead >= 0xF0 and lead <= 0xF4:
            width = 4
            if lead == 0xF0:
                lo = 0x90
            if lead == 0xF4:
                hi = 0x8F
        else:
            good = False
        if width > end - at:
            good = False
        if good and width > 1:
            second = u32(s.chars[at + 1])
            good = second >= lo and second <= hi
            j = u32(2)
            while j < width and good:
                byte = s.chars[at + j]
                good = byte >= 0x80 and byte <= 0xBF
                j += 1
        if not good:
            bad = at - s.begin(row)
        at += width
    return bad


@kernel
def first_invalid_byte(s: Strings, out: Vector[u32]) -> None:
    for i in items(s.rows):
        out[i] = invalid_byte(s, i)


@kernel
def is_valid_utf8(s: Strings, out: Vector[u8]) -> None:
    for i in items(s.rows):
        out[i] = u8(invalid_byte(s, i) == 0xFFFFFFFF)


@kernel
def reverse(s: Strings, chars: Vector[u8]) -> None:
    for i in items(s.rows):
        if s.is_valid(i):
            dest = s.end(i)
            for _, at in s.codepoints(i):
                width = s.width(at)
                dest -= width
                for j in range(width):
                    chars[dest + j] = s.chars[at + j]


@kernel
def search(s: Strings, row: u32, pattern: Vector[u8], last: bool) -> i32:
    at = s.begin(row)
    end = s.end(row)
    position = i32(0)
    found = i32(-1)
    done = False
    while at <= end and not done and (last or found < 0):
        if matches(s, at, end, pattern):
            found = position
        if at < end and (s.chars[at] & 0xC0) != 0x80:
            position += 1
        if at == end:
            done = True
        else:
            at += 1
    return found


@kernel
def starts_with(s: Strings, pattern: Vector[u8], out: Vector[u8]) -> None:
    for i in items(s.rows):
        out[i] = u8(matches(s, s.begin(i), s.end(i), pattern))


@kernel
def ends_with(s: Strings, pattern: Vector[u8], out: Vector[u8]) -> None:
    for i in items(s.rows):
        end = s.end(i)
        fits = u32(pattern.shape[0]) <= end - s.begin(i)
        result = False
        if fits:
            result = matches(s, end - u32(pattern.shape[0]), end, pattern)
        out[i] = u8(result)


@kernel
def contains(s: Strings, pattern: Vector[u8], out: Vector[u8]) -> None:
    for i in items(s.rows):
        out[i] = u8(search(s, i, pattern, False) >= 0)


@kernel
def find(s: Strings, pattern: Vector[u8], out: Vector[i32]) -> None:
    for i in items(s.rows):
        out[i] = search(s, i, pattern, False)


@kernel
def rfind(s: Strings, pattern: Vector[u8], out: Vector[i32]) -> None:
    for i in items(s.rows):
        out[i] = search(s, i, pattern, True)


@kernel
def split_ranges(s: Strings, row: u32, delimiter: Vector[u8]) -> Iterator[tuple[u32, u32]]:
    if s.is_valid(row):
        first = s.begin(row)
        at = first
        end = s.end(row)
        while at < end:
            if matches(s, at, end, delimiter):
                yield first, at
                at += u32(delimiter.shape[0])
                first = at
            else:
                at += 1
        yield first, end


@kernel
def split_count(s: Strings, delimiter: Vector[u8], pieces: Vector[u32]) -> None:
    for i in items(s.rows):
        pieces[i] = u32(sum(1 for _ in split_ranges(s, i, delimiter)))


@kernel
def split_record(
    s: Strings,
    delimiter: Vector[u8],
    documents: Vector[u32],
    starts: Vector[u32],
    ends: Vector[u32],
) -> None:
    """`documents` is the exclusive scan of `split_count`: a position among the pieces, so it is u32 like `starts`."""
    for i in items(s.rows):
        k = documents[i]
        for first, last in split_ranges(s, i, delimiter):
            starts[k] = first
            ends[k] = last
            k += 1


@kernel
def ordering(a: Strings, b: OtherStrings, row: u32) -> i32:
    first = a.begin(row)
    second = b.begin(row)
    an = a.end(row) - first
    bn = b.end(row) - second
    k = u32(0)
    result = i32(0)
    while k < min(an, bn) and result == 0:
        if a.chars[first + k] < b.chars[second + k]:
            result = -1
        elif a.chars[first + k] > b.chars[second + k]:
            result = 1
        k += 1
    if result == 0:
        if an < bn:
            result = -1
        elif an > bn:
            result = 1
    return i32(result)


@kernel
def equal(a: Strings, b: OtherStrings, out: Vector[u8]) -> None:
    for i in items(a.rows):
        out[i] = u8(ordering(a, b, i) == 0)


@kernel
def compare(a: Strings, b: OtherStrings, out: Vector[i32]) -> None:
    for i in items(a.rows):
        out[i] = ordering(a, b, i)


@kernel
def hash_strings(s: Strings, out: Vector[u64]) -> None:
    """Polynomial base 0x100000001B3, with byte + 1 terms and zero seed, modulo 2**64.

    Superseded by hash_strings_xxh64.
    """
    for i in items(s.rows):
        value = u64(0)
        for at in range(s.begin(i), s.end(i)):
            value = value * 0x100000001B3 + (u64(s.chars[at]) + 1)
        out[i] = value


@kernel
def hash_strings_xxh64(s: Strings, out: Vector[u64]) -> None:
    """Hash each row's bytes with seed zero; null rows have the empty-input hash."""
    for i in items(s.rows):
        out[i] = xxh64_bytes(s.chars, s.begin(i), s.end(i), u64(0))


@kernel
def parsed_integer(s: Strings, row: u32) -> Iterator[i64]:
    """Yield once on success; unsigned magnitude admits INT64_MIN without signed division or overflow."""
    at = s.begin(row)
    end = s.end(row)
    negative = False
    if at < end and (s.chars[at] == 43 or s.chars[at] == 45):
        negative = s.chars[at] == 45
        at += 1
    good = at < end
    magnitude = u64(0)
    limit = u64(9223372036854775807)
    if negative:
        limit += 1
    while at < end and good:
        byte = u64(s.chars[at])
        if byte < 48 or byte > 57:
            good = False
        else:
            digit = byte - 48
            if magnitude > limit // 10 or (magnitude == limit // 10 and digit > limit % 10):
                good = False
            else:
                magnitude = magnitude * 10 + digit
        at += 1
    if good:
        if negative:
            magnitude = u64(0) - magnitude
        yield i64(magnitude)


@kernel
def is_integer(s: Strings, out: Vector[u8]) -> None:
    for i in items(s.rows):
        out[i] = u8(sum(1 for _ in parsed_integer(s, i)))


@kernel
def to_integers(s: Strings, out: Vector[i64]) -> None:
    for i in items(s.rows):
        out[i] = 0
        for value in parsed_integer(s, i):
            out[i] = value


@kernel
def slice_strings_count(s: Strings, start: i32, stop: i32, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in slice_strings_bytes(s, i, start, stop)))


@kernel
def slice_strings_fill(s: Strings, start: i32, stop: i32, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in slice_strings_bytes(s, i, start, stop):
            chars[at] = byte
            at += 1


@kernel
def slice_bytes_count(s: Strings, start: i32, stop: i32, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in slice_bytes_bytes(s, i, start, stop)))


@kernel
def slice_bytes_fill(s: Strings, start: i32, stop: i32, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in slice_bytes_bytes(s, i, start, stop):
            chars[at] = byte
            at += 1


@kernel
def concatenate_count(a: Strings, b: OtherStrings, separator: Vector[u8], lengths: Vector[u32]) -> None:
    for i in items(a.rows):
        lengths[i] = u32(sum(1 for _ in concatenate_bytes(a, b, i, separator)))


@kernel
def concatenate_fill(
    a: Strings, b: OtherStrings, separator: Vector[u8], offsets: Vector[i32], chars: Vector[u8]
) -> None:
    for i in items(a.rows):
        at = u32(offsets[i])
        for byte in concatenate_bytes(a, b, i, separator):
            chars[at] = byte
            at += 1


@kernel
def join_strings_count(d: Documents, separator: Vector[u8], lengths: Vector[u32]) -> None:
    for i in items(d.count):
        lengths[i] = u32(sum(1 for _ in join_strings_bytes(d, i, separator)))


@kernel
def join_strings_fill(d: Documents, separator: Vector[u8], offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(d.count):
        at = u32(offsets[i])
        for byte in join_strings_bytes(d, i, separator):
            chars[at] = byte
            at += 1


@kernel
def repeat_strings_count(s: Strings, count: u32, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in repeat_strings_bytes(s, i, count)))


@kernel
def repeat_strings_fill(s: Strings, count: u32, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in repeat_strings_bytes(s, i, count):
            chars[at] = byte
            at += 1


@kernel
def pad_count(s: Strings, width: u32, fill: u32, side: u32, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in pad_bytes(s, i, width, fill, side)))


@kernel
def pad_fill(s: Strings, width: u32, fill: u32, side: u32, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in pad_bytes(s, i, width, fill, side):
            chars[at] = byte
            at += 1


@kernel
def zfill_count(s: Strings, width: u32, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in zfill_bytes(s, i, width)))


@kernel
def zfill_fill(s: Strings, width: u32, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in zfill_bytes(s, i, width):
            chars[at] = byte
            at += 1


@kernel
def replace_count(s: Strings, target: Vector[u8], replacement: Vector[u8], maxrepl: u32, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in replace_bytes(s, i, target, replacement, maxrepl)))


@kernel
def replace_fill(
    s: Strings,
    target: Vector[u8],
    replacement: Vector[u8],
    maxrepl: u32,
    offsets: Vector[i32],
    chars: Vector[u8],
) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in replace_bytes(s, i, target, replacement, maxrepl):
            chars[at] = byte
            at += 1


@kernel
def from_integers_count(values: Vector[i64], lengths: Vector[u32]) -> None:
    for i in items(values.shape[0]):
        lengths[i] = u32(sum(1 for _ in from_integers_bytes(values, u32(i))))


@kernel
def from_integers_fill(values: Vector[i64], offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(values.shape[0]):
        at = u32(offsets[i])
        for byte in from_integers_bytes(values, u32(i)):
            chars[at] = byte
            at += 1


@kernel
def strip_bytes(s: Strings, row: u32, u: Unicode, side: u32) -> Iterator[u8]:
    """SPACE is bit 16 in the generated Python character classes."""
    if s.is_valid(row):
        first = s.begin(row)
        last = s.end(row)
        done = side == 1
        while first < last and not done:
            if (u.character_type(s.codepoint(first)) & 16) != 0:
                first += s.width(first)
            else:
                done = True
        done = side == 0
        while last > first and not done:
            previous = s.previous(last, first)
            if (u.character_type(s.codepoint(previous)) & 16) != 0:
                last = previous
            else:
                done = True
        for at in range(first, last):
            yield s.chars[at]


@kernel
def final_sigma(s: Strings, row: u32, at: u32, u: Unicode) -> bool:
    """Ignore bit 64 while finding the nearest neighbors; bit 32 means cased and not ignorable."""
    before = False
    after = False
    cursor = at
    done = False
    while cursor > s.begin(row) and not done:
        cursor = s.previous(cursor, s.begin(row))
        kind = u.character_type(s.codepoint(cursor))
        if (kind & 64) == 0:
            before = (kind & 32) != 0
            done = True
    cursor = at + s.width(at)
    done = False
    while cursor < s.end(row) and not done:
        kind = u.character_type(s.codepoint(cursor))
        if (kind & 64) == 0:
            after = (kind & 32) != 0
            done = True
        cursor += s.width(cursor)
    return before and not after


@kernel
def mapped_bytes(s: Strings, row: u32, u: Unicode, index: Vector[u16], pages: Vector[u32], lower: bool) -> Iterator[u8]:
    """Bit 31 marks a full mapping: bits 24..30 hold its length, bits 0..23 its pool offset."""
    for cp, at in s.codepoints(row):
        mapping = pages[(u32(index[cp >> 8]) << 8) | (cp & 255)]
        is_final = False
        if lower and cp == 0x03A3:
            is_final = final_sigma(s, row, at, u)
        if is_final:
            mapping = u32(0x03C2)
        length = u32(1)
        if (mapping & 0x80000000) != 0:
            length = (mapping >> 24) & 127
        for j in range(length):
            replacement = cp
            if (mapping & 0x80000000) != 0:
                replacement = u.case_pool[(mapping & 0xFFFFFF) + j]
            elif mapping != 0:
                replacement = mapping
            for byte in utf8(replacement):
                yield u8(byte)


@kernel
def to_upper_bytes(s: Strings, row: u32, u: Unicode) -> Iterator[u8]:
    if s.is_valid(row):
        for byte in mapped_bytes(s, row, u, u.upper_index, u.upper_pages, False):
            yield u8(byte)


@kernel
def to_lower_bytes(s: Strings, row: u32, u: Unicode) -> Iterator[u8]:
    if s.is_valid(row):
        for byte in mapped_bytes(s, row, u, u.lower_index, u.lower_pages, True):
            yield u8(byte)


@kernel
def casefold_bytes(s: Strings, row: u32, u: Unicode) -> Iterator[u8]:
    if s.is_valid(row):
        for byte in mapped_bytes(s, row, u, u.fold_index, u.fold_pages, False):
            yield u8(byte)


@kernel
def all_characters_of_type(s: Strings, u: Unicode, types: u32, out: Vector[u8]) -> None:
    for i in items(s.rows):
        good = s.begin(i) < s.end(i)
        for cp, _ in s.codepoints(i):
            if (u.character_type(cp) & types) == 0:
                good = False
        out[i] = u8(good)


@kernel
def strip_count(s: Strings, u: Unicode, side: u32, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in strip_bytes(s, i, u, side)))


@kernel
def strip_fill(s: Strings, u: Unicode, side: u32, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in strip_bytes(s, i, u, side):
            chars[at] = byte
            at += 1


@kernel
def to_upper_count(s: Strings, u: Unicode, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in to_upper_bytes(s, i, u)))


@kernel
def to_upper_fill(s: Strings, u: Unicode, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in to_upper_bytes(s, i, u):
            chars[at] = byte
            at += 1


@kernel
def to_lower_count(s: Strings, u: Unicode, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in to_lower_bytes(s, i, u)))


@kernel
def to_lower_fill(s: Strings, u: Unicode, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in to_lower_bytes(s, i, u):
            chars[at] = byte
            at += 1


@kernel
def casefold_count(s: Strings, u: Unicode, lengths: Vector[u32]) -> None:
    for i in items(s.rows):
        lengths[i] = u32(sum(1 for _ in casefold_bytes(s, i, u)))


@kernel
def casefold_fill(s: Strings, u: Unicode, offsets: Vector[i32], chars: Vector[u8]) -> None:
    for i in items(s.rows):
        at = u32(offsets[i])
        for byte in casefold_bytes(s, i, u):
            chars[at] = byte
            at += 1


class Side(IntEnum):
    LEFT = 0
    RIGHT = auto()
    BOTH = auto()


NOT_FOUND = 0xFFFFFFFF


class StringsArrays(NamedTuple):
    chars: np.ndarray
    offsets: np.ndarray
    valid: np.ndarray
    rows: int


class DocumentsArrays(NamedTuple):
    chars: np.ndarray
    offsets: np.ndarray
    valid: np.ndarray
    rows: int
    documents: np.ndarray
    count: int


type Input = Argument | StringsArrays | DocumentsArrays


def as_lists(s: StringsArrays) -> ListsArrays:
    """The bytes and offsets are identical, only the field names differ."""
    return ListsArrays(s.chars, s.valid, s.rows, s.offsets)


def as_strings(lists: ListsArrays) -> StringsArrays:
    """The bytes and offsets are identical, only the field names differ."""
    return StringsArrays(lists.values, lists.offsets, lists.valid, lists.rows)


def strings_arrays(values: Sequence[str | bytes | None]) -> StringsArrays:
    encoded = [value.encode("utf-8") if isinstance(value, str) else value or b"" for value in values]
    offsets = offsets_array(np.asarray([len(value) for value in encoded], np.uint64))
    return StringsArrays(
        padded(np.frombuffer(b"".join(encoded), np.uint8)),
        offsets,
        validity_words(np.array([v is not None for v in values], bool)),
        len(values),
    )


def strings_values(arrays: StringsArrays) -> list[bytes | None]:
    valid = validity_bits(arrays.valid, arrays.rows)
    return [
        bytes(arrays.chars[arrays.offsets[i] : arrays.offsets[i + 1]]) if valid[i] else None for i in range(arrays.rows)
    ]


def documents_arrays(groups: Sequence[Sequence[str | bytes | None]]) -> DocumentsArrays:
    arrays = strings_arrays([value for group in groups for value in group])
    return DocumentsArrays(
        *arrays,
        offsets_array(np.asarray([len(group) for group in groups], np.uint64)),
        len(groups),
    )


def _arguments(name: str, args: Sequence[Input]) -> list[Argument]:
    arguments = [arg._asdict() if isinstance(arg, StringsArrays | DocumentsArrays) else arg for arg in args]
    if name in ("replace", "split_count", "split_record"):
        pattern = arguments[1]
        if not isinstance(pattern, np.ndarray) or pattern.size == 0:
            raise ValueError("the literal target or delimiter must not be empty")
    return arguments


def sized(launch: Launch, op: str, rows: int, *args: Input) -> StringsArrays:
    """Count, exclusive scan, allocate, fill; validity propagates except for the null-skipping document join."""
    arguments = _arguments(op, args)
    lengths = empty(rows, np.uint32)
    launch(f"{op}_count", rows, [*arguments, lengths])
    offsets = offsets_array(lengths[:rows])
    chars = empty(int(offsets[rows]), np.uint8)
    launch(f"{op}_fill", rows, [*arguments, offsets, chars])
    valid = np.ones(rows, bool)
    if op != "join_strings":
        for arg in arguments:
            bitmap = arg.get("valid") if isinstance(arg, Mapping) and arg.get("rows") == rows else None
            if isinstance(bitmap, np.ndarray):
                valid &= validity_bits(bitmap, rows)
    return StringsArrays(chars, offsets, validity_words(valid), rows)


def column(launch: Launch, name: str, rows: int, dtype: np.typing.DTypeLike, *args: Input) -> np.ndarray:
    out = empty(rows, dtype)
    launch(name, rows, [*_arguments(name, args), out])
    return out[:rows]


def head(launch: Launch, s: StringsArrays, n: int) -> StringsArrays:
    return sized(launch, "slice_strings", s.rows, s, 0, n)


def tail(launch: Launch, s: StringsArrays, n: int) -> StringsArrays:
    return sized(launch, "slice_strings", s.rows, s, -n, 0 if n == 0 else 0x7FFFFFFF)

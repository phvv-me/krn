"""Non-cryptographic hashes for kernels; callers canonicalize floats before hashing their bits.

Integer keys use little-endian byte order. Byte ranges require 0 <= start <= stop <= data.shape[0].
"""

from krn import Vector, kernel
from krn.frontend.types import u8, u32, u64


@kernel
def splitmix64(x: u64) -> u64:
    """The splitmix64 step, including its increment; a bijection over 64-bit keys."""
    z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@kernel
def combine(h: u64, field_hash: u64) -> u64:
    """Chain field hashes in order. This is not an external standard and has no oracle.

    Tested by determinism, order sensitivity, and collision checks on 100,000 distinct small-integer pairs.
    """
    return splitmix64(h ^ (field_hash + 0x9E3779B97F4A7C15 + (h << 6) + (h >> 2)))


@kernel
def _rot32(x: u32, n: int) -> u32:
    return (x << n) | (x >> (32 - n))


@kernel
def _rot64(x: u64, n: int) -> u64:
    return (x << n) | (x >> (64 - n))


@kernel
def xxh32(x: u32, seed: u32) -> u32:
    """XXH32 of one four-byte key."""
    h = seed + 0x165667B1 + 4
    h = _rot32(h + x * 0xC2B2AE3D, 17) * 0x27D4EB2F
    h = (h ^ (h >> 15)) * 0x85EBCA77
    h = (h ^ (h >> 13)) * 0xC2B2AE3D
    return h ^ (h >> 16)


@kernel
def _round64(acc: u64, lane: u64) -> u64:
    return _rot64(acc + lane * 0xC2B2AE3D27D4EB4F, 31) * 0x9E3779B185EBCA87


@kernel
def _merge64(h: u64, lane: u64) -> u64:
    return (h ^ _round64(u64(0), lane)) * 0x9E3779B185EBCA87 + 0x85EBCA77C2B2AE63


@kernel
def _avalanche64(h: u64) -> u64:
    h = (h ^ (h >> 33)) * 0xC2B2AE3D27D4EB4F
    h = (h ^ (h >> 29)) * 0x165667B19E3779F9
    return h ^ (h >> 32)


@kernel
def _tail64(h: u64, lane: u64) -> u64:
    return _rot64(h ^ _round64(u64(0), lane), 27) * 0x9E3779B185EBCA87 + 0x85EBCA77C2B2AE63


@kernel
def xxh64(x: u64, seed: u64) -> u64:
    """XXH64 of one eight-byte key."""
    return _avalanche64(_tail64(seed + 0x27D4EB2F165667C5 + 8, x))


@kernel
def _load32(data: Vector[u8], at: u32 | int) -> u32:
    return u32(data[at]) | (u32(data[at + 1]) << 8) | (u32(data[at + 2]) << 16) | (u32(data[at + 3]) << 24)


@kernel
def _load64(data: Vector[u8], at: u32 | int) -> u64:
    return u64(_load32(data, at)) | (u64(_load32(data, at + 4)) << 32)


@kernel
def xxh64_bytes(data: Vector[u8], start: u32 | int, stop: u32 | int, seed: u64) -> u64:
    """XXH64 of data[start:stop], without alignment requirements or allocation."""
    at = start
    h = seed + 0x27D4EB2F165667C5
    if stop - start >= 32:
        v1 = seed + 0x9E3779B185EBCA87 + 0xC2B2AE3D27D4EB4F
        v2 = seed + 0xC2B2AE3D27D4EB4F
        v3 = seed
        v4 = seed - 0x9E3779B185EBCA87
        while stop - at >= 32:
            v1 = _round64(v1, _load64(data, at))
            v2 = _round64(v2, _load64(data, at + 8))
            v3 = _round64(v3, _load64(data, at + 16))
            v4 = _round64(v4, _load64(data, at + 24))
            at += 32
        h = _rot64(v1, 1) + _rot64(v2, 7) + _rot64(v3, 12) + _rot64(v4, 18)
        h = _merge64(_merge64(_merge64(_merge64(h, v1), v2), v3), v4)
    h += u64(stop - start)
    while stop - at >= 8:
        h = _tail64(h, _load64(data, at))
        at += 8
    if stop - at >= 4:
        h = _rot64(h ^ (u64(_load32(data, at)) * 0x9E3779B185EBCA87), 23)
        h = h * 0xC2B2AE3D27D4EB4F + 0x165667B19E3779F9
        at += 4
    while at < stop:
        h = _rot64(h ^ (u64(data[at]) * 0x27D4EB2F165667C5), 11) * 0x9E3779B185EBCA87
        at += 1
    return _avalanche64(h)


@kernel
def murmur3_32(x: u32, seed: u32) -> u32:
    """MurmurHash3 x86_32 of one four-byte key."""
    k = _rot32(x * 0xCC9E2D51, 15) * 0x1B873593
    h = _rot32(seed ^ k, 13) * 5 + 0xE6546B64
    h ^= 4
    h = (h ^ (h >> 16)) * 0x85EBCA6B
    h = (h ^ (h >> 13)) * 0xC2B2AE35
    return h ^ (h >> 16)

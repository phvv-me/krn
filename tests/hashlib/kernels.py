"""Batched callers exercise the scalar hashes as kernel callees."""

from krn import Vector, items, kernel
from krn.frontend.types import u8, u32, u64
from krn.hashlib import combine, murmur3_32, splitmix64, xxh32, xxh64, xxh64_bytes


@kernel
def words32(keys: Vector[u32], seeds: Vector[u32], xx: Vector[u32], murmur: Vector[u32]) -> None:
    for i in items(keys.shape[0]):
        xx[i] = xxh32(keys[i], seeds[i])
        murmur[i] = murmur3_32(keys[i], seeds[i])


@kernel
def words64(keys: Vector[u64], seeds: Vector[u64], xx: Vector[u64], mixed: Vector[u64]) -> None:
    for i in items(keys.shape[0]):
        xx[i] = xxh64(keys[i], seeds[i])
        mixed[i] = splitmix64(keys[i])


@kernel
def pairs(keys: Vector[u64], fields: Vector[u64], out: Vector[u64]) -> None:
    for i in items(keys.shape[0]):
        out[i] = combine(combine(u64(0), splitmix64(keys[i])), splitmix64(fields[i]))


@kernel
def ranges(data: Vector[u8], starts: Vector[u32], stops: Vector[u32], seeds: Vector[u64], out: Vector[u64]) -> None:
    for i in items(starts.shape[0]):
        out[i] = xxh64_bytes(data, starts[i], stops[i], seeds[i])

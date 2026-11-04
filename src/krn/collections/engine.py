"""Bounded linear probing with splitmix64 and an all-ones side slot.

The immutable last key word holds the empty pattern; side[0] publishes its presence.
Set insert and lookup may share a launch. Map payloads are published in a later
launch; readers must follow that launch. There are no inter-workgroup waits.
"""

from enum import IntEnum

from krn import Vector, atomic, items, kernel
from krn.frontend.types import Scalar, i32, i64, u32, u64
from krn.hashlib import splitmix64


class InsertResult(IntEnum):
    INSERTED = 0
    EXISTING = 1
    FULL = 2


@kernel
def bits32(keys: Vector[i32], out: Vector[u32]) -> None:
    for i in items(keys.shape[0]):
        out[i] = u32(keys[i])


@kernel
def bits64(keys: Vector[i64], out: Vector[u64]) -> None:
    for i in items(keys.shape[0]):
        out[i] = u64(keys[i])


@kernel
def insert[K: u32 | u64](table: Vector[K], side: Vector[u32], key: K) -> u32:
    """Return INSERTED=0, EXISTING=1, or FULL=2 in at most capacity probes."""
    capacity = table.shape[0] - 1
    empty = table[capacity]
    result = u32(2)
    if key == empty:
        was_present = atomic.cas(side, 0, 0, 1)
        result = u32(0) if was_present == 0 else u32(1)
    else:
        slot = u32(splitmix64(u64(key)) & u64(capacity - 1))
        probes = 0
        while probes < capacity and result == 2:
            old = atomic.cas(table, slot, empty, key)
            if old == empty:
                result = u32(0)
            elif old == key:
                result = u32(1)
            slot = (slot + 1) & (capacity - 1)
            probes += 1
    return result


@kernel
def find[K: u32 | u64](table: Vector[K], side: Vector[u32], key: K) -> int:
    """Return the slot, including the side slot at capacity, or -1."""
    capacity = table.shape[0] - 1
    empty = table[capacity]
    result = -1
    if key == empty:
        present = atomic.cas(side, 0, 0, 0)
        result = capacity if present != 0 else -1
    else:
        slot = u32(splitmix64(u64(key)) & u64(capacity - 1))
        probes = 0
        searching = True
        while probes < capacity and searching:
            old = atomic.cas(table, slot, empty, empty)
            if old == key:
                result = int(slot)
                searching = False
            elif old == empty:
                searching = False
            slot = (slot + 1) & (capacity - 1)
            probes += 1
    return result


@kernel
def add[K: u32 | u64](table: Vector[K], side: Vector[u32], keys: Vector[K], results: Vector[u32]) -> None:
    for i in items(keys.shape[0]):
        results[i] = insert(table, side, keys[i])


@kernel
def contains[K: u32 | u64](table: Vector[K], side: Vector[u32], keys: Vector[K], out: Vector[bool]) -> None:
    for i in items(keys.shape[0]):
        out[i] = find(table, side, keys[i]) >= 0


@kernel
def update[K: u32 | u64](
    table: Vector[K], side: Vector[u32], keys: Vector[K], results: Vector[u32], winners: Vector[u32]
) -> None:
    for i in items(keys.shape[0]):
        result = insert(table, side, keys[i])
        results[i] = result
        if result != 2:
            slot = find(table, side, keys[i])
            atomic.max(winners, slot, u32(i + 1))


@kernel
def assign[V: Scalar](values: Vector[V], incoming: Vector[V], winners: Vector[u32]) -> None:
    for slot in items(winners.shape[0]):
        winner = winners[slot]
        if winner != 0:
            values[slot] = incoming[winner - 1]


@kernel
def get[K: u32 | u64, V: Scalar](
    table: Vector[K], side: Vector[u32], values: Vector[V], keys: Vector[K], fallback: V, out: Vector[V]
) -> None:
    for i in items(keys.shape[0]):
        slot = find(table, side, keys[i])
        out[i] = values[slot] if slot >= 0 else fallback

"""Shared host storage, validation, and synchronous snapshots."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from krn import Items, Tensor, host
from krn.arithmetic import ceildiv
from krn.frontend.types import i32, i64, u32, u64
from krn.runtime.runtime import numpy_dtype

from . import engine
from .exceptions import CapacityError

type Key = u32 | u64 | i32 | i64


@dataclass(frozen=True, kw_only=True)
class Inserted:
    """Counts per input occurrence, including duplicate and failed occurrences."""

    inserted: int
    existing: int
    failed: int

    def __post_init__(self) -> None:
        if min(self.inserted, self.existing, self.failed) < 0:
            raise ValueError("counts must be non-negative")

    def check(self) -> None:
        if self.failed:
            raise CapacityError(f"{self.failed} keys did not fit in the fixed table")


@dataclass(frozen=True, kw_only=True)
class _Sizing:
    capacity: int
    load_factor: float

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")
        if not isfinite(self.load_factor) or not 0 < self.load_factor <= 1:
            raise ValueError("load_factor must be finite and in (0, 1]")

    def slots(self) -> int:
        numerator, denominator = self.load_factor.as_integer_ratio()
        required = ceildiv(self.capacity * denominator, numerator)
        if required > 2**32:
            raise ValueError("tables support at most 2**32 ordinary slots")
        return 1 << max(0, required - 1).bit_length()


class Table[K: Key]:
    def __init__(self, capacity: int, *, load_factor: float = 0.5) -> None:
        self.slot_count = _Sizing(capacity=capacity, load_factor=load_factor).slots()
        self.device = host.Device.current()
        self._table: Tensor | None = None
        self._dtype: np.dtype | None = None
        self._side = self.device.to_device(np.zeros(1, np.uint32))

    def _read(self, tensor: Tensor) -> np.ndarray:
        stream = host.current_stream.get() or self.device.default_stream
        stream.record().wait()
        return tensor.to_host()

    def _keys(self, keys: Tensor[K, int]) -> Tensor:
        dtype = numpy_dtype(keys.dtype)
        if len(keys.shape) != 1 or dtype not in (np.dtype("u4"), np.dtype("u8"), np.dtype("i4"), np.dtype("i8")):
            raise TypeError("keys must be a vector of u32, u64, i32, or i64")
        if keys.shape[0] >= 2**32:
            raise ValueError("batches support fewer than 2**32 input occurrences")
        if self._dtype is None:
            self._dtype = dtype
            unsigned = np.dtype(f"u{dtype.itemsize}")
            self._table = self.device.to_device(np.full(self.slot_count + 1, np.iinfo(unsigned).max, unsigned))
        if dtype != self._dtype:
            raise TypeError("key dtype is fixed by the first batch")
        if dtype.kind == "i":
            if dtype.itemsize == 4:
                normalized = self.device.empty(u32, keys.shape[0])
                engine.bits32(keys, normalized, launch=Items(keys.shape[0]))
            else:
                normalized = self.device.empty(u64, keys.shape[0])
                engine.bits64(keys, normalized, launch=Items(keys.shape[0]))
            return normalized
        return keys

    def _storage(self) -> Tensor:
        assert self._table is not None
        return self._table

    def _counts(self, results: Tensor) -> Inserted:
        counts = np.bincount(self._read(results), minlength=3)
        return Inserted(inserted=int(counts[0]), existing=int(counts[1]), failed=int(counts[2]))

    def contains(self, keys: Tensor[K, int]) -> Tensor[bool, int]:
        batch = self._keys(keys)
        out = self.device.to_device(np.empty(keys.shape[0], np.bool_))
        engine.contains(self._storage(), self._side, batch, out, launch=Items(keys.shape[0]))
        return out

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if self._table is None:
            return np.empty(0, np.uint32), np.empty(0, np.int64)
        table = self._read(self._table)
        occupied = table != table[-1]
        occupied[-1] = self._read(self._side)[0] != 0
        assert self._dtype is not None
        return table[occupied].view(self._dtype), np.flatnonzero(occupied)

    def __len__(self) -> int:
        return len(self._snapshot()[0])

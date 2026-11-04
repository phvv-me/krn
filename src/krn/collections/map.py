"""Fixed integer maps: batched operations, constant defaults, no order or popitem.

len and items synchronize. Updates publish values in a separate launch before
queries. Repeated keys in a batch take the last input value, as in dict.update.
"""

from typing import cast

import numpy as np

from krn import Items, Tensor
from krn.frontend.types import u32
from krn.runtime.runtime import numpy_dtype

from . import engine
from ._table import Inserted, Key, Table


class Map[K: Key, V: Key](Table[K]):
    _values: Tensor[V, int] | None = None

    def update(self, keys: Tensor[K, int], values: Tensor[V, int]) -> Inserted:
        if values.shape != keys.shape:
            raise ValueError("keys and values must have the same vector shape")
        if numpy_dtype(values.dtype) not in (np.dtype("u4"), np.dtype("u8"), np.dtype("i4"), np.dtype("i8")):
            raise TypeError("values must be u32, u64, i32, or i64")
        batch = self._keys(keys)
        if self._values is None:
            self._values = self.device.empty(values.dtype, self.slot_count + 1)
        if values.dtype != self._values.dtype:
            raise TypeError("value dtype is fixed by the first update")
        results = self.device.empty(u32, keys.shape[0])
        winners = self.device.to_device(np.zeros(self.slot_count + 1, np.uint32))
        engine.update(self._storage(), self._side, batch, results, winners, launch=Items(keys.shape[0]))
        engine.assign(self._values, values, winners, launch=Items(self.slot_count + 1))
        return self._counts(results)

    def get(self, keys: Tensor[K, int], default: V) -> Tensor[V, int]:
        batch = self._keys(keys)
        if self._values is None:
            # Python preserves the boxed generic alias; checkers lose V through Scalar's Self attribute.
            dtype = cast(type[V], default.__orig_class__)
            self._values = self.device.empty(dtype, self.slot_count + 1)
        out = self.device.empty(self._values.dtype, keys.shape[0])
        engine.get(self._storage(), self._side, self._values, batch, default, out, launch=Items(keys.shape[0]))
        return out

    def items(self) -> tuple[Tensor[K, int], Tensor[V, int]]:
        """Independent key and value snapshots, paired in unspecified order."""
        keys, slots = self._snapshot()
        values = np.empty(0, np.uint32) if self._values is None else self._read(self._values)[slots]
        return self.device.to_device(keys), self.device.to_device(values)

"""Fixed integer sets: batched operations, unspecified order, no growth or popitem.

len and keys synchronize. All key bit patterns are valid, including all ones.
"""

from krn import Items, Tensor
from krn.frontend.types import u32

from . import engine
from ._table import Inserted, Key, Table


class Set[K: Key](Table[K]):
    """The first batch fixes the key dtype; capacity is the expected unique count."""

    def add(self, keys: Tensor[K, int]) -> Inserted:
        batch = self._keys(keys)
        results = self.device.empty(u32, keys.shape[0])
        engine.add(self._storage(), self._side, batch, results, launch=Items(keys.shape[0]))
        return self._counts(results)

    def keys(self) -> Tensor[K, int]:
        """An independent snapshot in unspecified order."""
        return self.device.to_device(self._snapshot()[0])

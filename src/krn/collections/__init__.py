"""Fixed integer hash tables with batched operations and unordered snapshots.

No automatic growth, scalar indexing, erase, or popitem. len and snapshots
synchronize; add and update also synchronize to report insertion counts.
"""

from ._table import Inserted
from .engine import InsertResult
from .exceptions import CapacityError
from .map import Map
from .set import Set

__all__ = ["CapacityError", "InsertResult", "Inserted", "Map", "Set"]

from collections.abc import Iterator
from types import TracebackType
from typing import Self

class Cache:
    cache_depth: int
    cache_type: int
    size: int
class Node:
    os_index: int
    local_memory: int
class PCIAttributes:
    domain: int
    bus: int
    dev: int
    func: int
class PCIDevice:
    attr: PCIAttributes
class Topology:
    def __enter__(self) -> Self: ...
    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None) -> None: ...
    def set_io_types_filter(self, type_filter: int) -> Self: ...
    def load(self) -> Self: ...
    def iter_numa_nodes(self) -> Iterator[Node]: ...
    def iter_all_breadth_first(self) -> Iterator[Cache | Node]: ...
    def iter_pci_devices(self) -> Iterator[PCIDevice]: ...

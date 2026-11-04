"""Frozen hwloc topology snapshots; no native handles escape detection."""

from dataclasses import dataclass

try:
    import pyhwloc
    from pyhwloc.hwobject import Cache
    from pyhwloc.topology import TypeFilter
except ImportError:
    pyhwloc = None
    Cache = None
    TypeFilter = None

from patos import value_dispatch

from .platforms import otherwise


@dataclass(frozen=True, kw_only=True)
class Topology:
    numa_nodes: tuple[tuple[int, int], ...] = ()
    caches: tuple[tuple[str, int], ...] = ()
    pci_devices: tuple[str, ...] = ()
    unavailable_reason: str | None = None


@value_dispatch(kind="system")
def detect_topology() -> Topology:
    return Topology(unavailable_reason="pyhwloc: topology is enabled on Linux only")


@detect_topology.register("Linux")
def _linux_topology() -> Topology:
    if pyhwloc is None or Cache is None or TypeFilter is None:
        return Topology(unavailable_reason="pyhwloc: import unavailable")
    try:
        with pyhwloc.Topology() as topology:
            topology.set_io_types_filter(TypeFilter.KEEP_ALL)
            topology.load()
            return Topology(
                numa_nodes=tuple((node.os_index, node.local_memory) for node in topology.iter_numa_nodes()),
                caches=tuple(
                    (f"l{node.cache_depth}-{node.cache_type}", node.size)
                    for node in topology.iter_all_breadth_first()
                    if isinstance(node, Cache)
                ),
                pci_devices=tuple(
                    f"{node.attr.domain:04x}:{node.attr.bus:02x}:{node.attr.dev:02x}.{node.attr.func}"
                    for node in topology.iter_pci_devices()
                ),
            )
    except (OSError, RuntimeError, ValueError, AttributeError) as error:
        return Topology(unavailable_reason=f"pyhwloc: {error}")


otherwise(detect_topology, "Darwin")

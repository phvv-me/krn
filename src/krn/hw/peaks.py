"""Immutable roofline ceilings and per-number provenance."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType

from krn.facts.peaks import DEVICE_RATES, MEMORY_BANDWIDTH_GBS, RATES


class PeakSource(StrEnum):
    LIBRARY = "library"
    TABLE = "table"
    MEASURED = "measured"


@dataclass(frozen=True, kw_only=True)
class PeakRates:
    """Decimal GB/s and GFLOP/s (integer dtypes: GOP/s), with no sparse multiplier."""

    memory_bandwidth_gbs: float | None = None
    flops: Mapping[str, float] = field(default_factory=dict, hash=False)
    memory_bandwidth_source: PeakSource | None = None
    flops_sources: Mapping[str, PeakSource] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if (self.memory_bandwidth_gbs is None) != (self.memory_bandwidth_source is None):
            raise ValueError("bandwidth and its source must be supplied together")
        if self.flops.keys() != self.flops_sources.keys():
            raise ValueError("each FLOP rate must have exactly one source")
        values = tuple(self.flops.values())
        if self.memory_bandwidth_gbs is not None:
            values += (self.memory_bandwidth_gbs,)
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("peak rates must be positive and finite")
        self.__dict__["flops"] = MappingProxyType(dict(self.flops))
        self.__dict__["flops_sources"] = MappingProxyType(dict(self.flops_sources))


def positive(read: Callable[[], float | None]) -> float | None:
    """Unavailable properties and unsupported/sentinel values are unknown."""
    try:
        value = read()
        return float(value) if value is not None and isfinite(value) and 0 < value < 2**64 - 1 else None
    except AttributeError, KeyError, RuntimeError, OSError, ValueError, TypeError:
        return None


def bandwidth(clock_khz: float | None, bus_bits: float | None) -> PeakRates:
    """DDR clock * 2 transfers/cycle * bus bits / 8, in decimal GB/s.

    This is raw pin bandwidth. ECC overhead and access patterns reduce useful
    payload bandwidth; no universal ECC discount exists (HBM has dedicated ECC).
    CUDA's GDDR clocks are normalized by the driver, not physical oscillator clocks.
    https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#bandwidth
    """
    if clock_khz is None or bus_bits is None:
        return PeakRates()
    return PeakRates(
        memory_bandwidth_gbs=clock_khz * 2 * bus_bits / 8 / 1e6,
        memory_bandwidth_source=PeakSource.LIBRARY,
    )


def resolve(arch: str, name: str, units: int | None, clock_mhz: float | None, library: PeakRates) -> PeakRates:
    """Prefer library values individually; fill only known table quantities."""
    product = name.removeprefix("NVIDIA ").removeprefix("GeForce ")
    row = DEVICE_RATES.get((arch, product), RATES.get(arch))
    rates: dict[str, float] = {}
    if row is not None and units is not None and clock_mhz is not None and units > 0 and clock_mhz > 0:
        rates = {dtype: ops * units * clock_mhz / 1000 for dtype, ops in row.operations_per_cycle.items()}
    sources = dict.fromkeys(rates, PeakSource.TABLE)
    rates.update(library.flops)
    sources.update(library.flops_sources)
    memory = library.memory_bandwidth_gbs
    memory_source = library.memory_bandwidth_source
    if memory is None:
        memory = MEMORY_BANDWIDTH_GBS.get(name)
        memory_source = PeakSource.TABLE if memory is not None else None
    return PeakRates(
        memory_bandwidth_gbs=memory, memory_bandwidth_source=memory_source, flops=rates, flops_sources=sources
    )

class Partition:
    mountpoint: str

def disk_partitions() -> list[Partition]: ...

class Error(Exception): ...

class Process:
    def cpu_affinity(self) -> list[int]: ...

class Frequency:
    current: float
    min: float
    max: float

class VirtualMemory:
    total: int
    available: int

class SwapMemory:
    total: int
    used: int

class DiskUsage:
    total: int
    free: int
    used: int

class Temperature:
    current: float
    high: float | None
    critical: float | None

class Fan:
    current: int

def cpu_count(logical: bool = True) -> int | None: ...
def cpu_freq() -> Frequency | None: ...
def virtual_memory() -> VirtualMemory: ...
def swap_memory() -> SwapMemory: ...
def disk_usage(path: str) -> DiskUsage: ...
def sensors_temperatures() -> dict[str, list[Temperature]]: ...
def sensors_fans() -> dict[str, list[Fan]]: ...

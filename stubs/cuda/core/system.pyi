"""Survey-confirmed cuda-core system API subset."""

from enum import StrEnum

class ClockType(StrEnum):
    SM = "sm"

class MemoryInfo:
    total: int
    used: int

class Temperature:
    def get_sensor(self) -> int: ...
    def get_threshold(self, kind: str) -> int: ...

class NvlinkInfo:
    state: bool
    version: tuple[int, int]

class Clock:
    def get_current_mhz(self) -> int: ...
    def get_max_mhz(self) -> int: ...

class Fan:
    speed: int

class Utilization:
    gpu: int

class PciInfo:
    max_link_generation: int
    max_link_width: int

class Reason:
    name: str

class ProcessInfo:
    pid: int
    used_gpu_memory: int

class Device:
    index: int
    def __init__(self, *, index: int) -> None: ...
    name: str
    uuid: str
    pci_bus_id: str
    numa_node_id: int
    memory_info: MemoryInfo
    pci_info: PciInfo
    num_fans: int
    temperature: Temperature
    utilization: Utilization
    current_clock_event_reasons: list[Reason]
    performance_state: int | None
    compute_running_processes: list[ProcessInfo]
    def get_clock(self, kind: str) -> Clock: ...
    def get_fan(self, fan: int) -> Fan: ...
    def get_cpu_affinity(self) -> list[int]: ...
    def get_nvlinks(self) -> list[NvlinkInfo]: ...

def get_num_devices() -> int: ...

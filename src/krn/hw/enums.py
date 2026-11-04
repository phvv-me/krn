from enum import IntEnum, StrEnum, auto


class Vendor(IntEnum):
    NVIDIA = 0x10DE
    AMD = 0x1002
    INTEL = 0x8086
    APPLE = 0x106B
    UNKNOWN = 0


class DeviceKind(StrEnum):
    CPU = auto()
    GPU = auto()
    NPU = auto()
    SOFTWARE = auto()


class Severity(StrEnum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()

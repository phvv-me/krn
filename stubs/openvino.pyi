"""OpenVINO device property subset read from the survey wheel's official stubs and headers."""

from typing import Literal, overload

class PCIInfo:
    domain: int
    bus: int
    device: int
    function: int

class DeviceType:
    name: str

class Core:
    available_devices: list[str]
    @overload
    def get_property(self, device_name: str, name: Literal["DEVICE_PCI_INFO"]) -> PCIInfo: ...
    @overload
    def get_property(self, device_name: str, name: Literal["DEVICE_TYPE"]) -> DeviceType: ...
    @overload
    def get_property(self, device_name: str, name: Literal["FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE"]) -> str: ...
    @overload
    def get_property(self, device_name: str, name: Literal["OPTIMIZATION_CAPABILITIES"]) -> list[str]: ...
    @overload
    def get_property(self, device_name: str, name: Literal["GPU_DEVICE_TOTAL_MEM_SIZE", "NPU_DEVICE_TOTAL_MEM_SIZE", "GPU_EXECUTION_UNITS_COUNT"]) -> int: ...

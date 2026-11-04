"""Survey API subset; runtime return shapes still need host verification."""

class Properties:
    name: str | bytes
    gcnArchName: str | bytes
    pciDomainID: int
    pciBusID: int
    pciDeviceID: int
    totalGlobalMem: int
    integrated: int
    multiProcessorCount: int
    warpSize: int
    maxThreadsPerBlock: int
    sharedMemPerBlock: int
    clockRate: int

def hipGetDeviceCount() -> int | tuple[int, int]: ...
def hipGetDeviceProperties(index: int) -> Properties | tuple[int, Properties]: ...

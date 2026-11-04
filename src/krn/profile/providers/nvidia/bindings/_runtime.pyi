from enum import IntEnum

class cudaError_t(IntEnum):
    cudaSuccess = 0

def cudaDeviceSynchronize() -> tuple[cudaError_t]: ...
def cudaGetDevice() -> tuple[cudaError_t, int]: ...

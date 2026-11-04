"""CUDA storage types passed to the nvmath library bindings."""

from enum import IntEnum

class cudaDataType(IntEnum):
    CUDA_R_16F = 2
    CUDA_R_16BF = 14
    CUDA_R_32F = 0
    CUDA_R_64F = 1

"""The part of PyObjC's MetalPerformanceShaders that `patos_gpu.lib_mps` calls."""

import Metal

MPSDataTypeFloat32: int
MPSDataTypeFloat16: int

class MPSMatrixDescriptor:
    @classmethod
    def matrixDescriptorWithRows_columns_rowBytes_dataType_(
        cls, rows: int, columns: int, row_bytes: int, data_type: int
    ) -> MPSMatrixDescriptor: ...

class MPSMatrix:
    @classmethod
    def alloc(cls) -> MPSMatrix: ...
    def initWithBuffer_descriptor_(self, buffer: Metal.MTLBuffer, descriptor: MPSMatrixDescriptor) -> MPSMatrix: ...

class MPSMatrixMultiplication:
    @classmethod
    def alloc(cls) -> MPSMatrixMultiplication: ...
    def initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
        self,
        device: Metal.MTLDevice,
        left: bool,
        right: bool,
        rows: int,
        columns: int,
        interior: int,
        alpha: float,
        beta: float,
    ) -> MPSMatrixMultiplication: ...
    def encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        self, command: Metal.MTLCommandBuffer, left: MPSMatrix, right: MPSMatrix, result: MPSMatrix
    ) -> None: ...

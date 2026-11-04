"""Metal Performance Shaders: Apple's own matrix multiply on the GPU, for float32 and float16 buffers.

`multiply` is the work: it encodes an `MPSMatrixMultiplication` on a command buffer. The candidate hands it the Metal
buffers of the tensors, which the Metal backend of the runtime provides as `tensor.buffer` and `stream.queue`; until
that backend exists, `multiply` is driven directly (see `tests/test_lib_vendors.py`). MPS has no epilogue.
"""

import platform

import Metal
import MetalPerformanceShaders as mps

from krn.facts.targets import Apple, Target
from krn.frontend.numerics import FloatingPoint, f16, f32
from krn.runtime.api import Stream

from .lib import Epilogue, MatmulCandidate, Op, Operands, Origin, Problem

TYPES: dict[type[FloatingPoint], tuple[int, int]] = {
    f32: (mps.MPSDataTypeFloat32, 4),
    f16: (mps.MPSDataTypeFloat16, 2),
}


def multiply(
    device: Metal.MTLDevice, command: Metal.MTLCommandBuffer, buffers: tuple[Metal.MTLBuffer, ...],
    shape: tuple[int, ...], dtype: type[FloatingPoint],
) -> None:  # fmt: skip
    """Encode `out = a @ b` for row-major matrices of `dtype`; `shape` is (m, n, k) and `buffers` are a, b and out."""
    (m, n, k), (code, size) = shape, TYPES[dtype]

    def matrix(buffer: Metal.MTLBuffer, rows: int, columns: int) -> mps.MPSMatrix:
        descriptor = mps.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
            rows, columns, columns * size, code
        )
        return mps.MPSMatrix.alloc().initWithBuffer_descriptor_(buffer, descriptor)

    product = mps.MPSMatrixMultiplication.alloc()
    kernel = product.initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
        device, False, False, m, n, k, 1.0, 0.0
    )
    a, b, out = buffers
    kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        command, matrix(a, m, k), matrix(b, k, n), matrix(out, m, n)
    )


class Mps(MatmulCandidate):
    name, op, origin, version = "mps", Op.MATMUL, Origin.VENDOR, platform.mac_ver()[0]

    def refusal(self, problem: Problem, target: Target) -> str | None:
        if not isinstance(target, Apple):
            return f"MPS multiplies Metal buffers, and {target.arch} tensors are not"
        if not {*problem.dtypes} <= {f16, f32} or len(set(problem.dtypes)) != 1:
            return f"MPS takes f16 or f32 in all three matrices, not {', '.join(d.__name__ for d in problem.dtypes)}"
        if problem.epilogue is not Epilogue.NONE:
            return f"MPS has no {problem.epilogue} epilogue"
        return None

    def run[S: FloatingPoint](self, problem: Problem, operands: Operands[S], stream: Stream) -> None:
        raise NotImplementedError(
            "tensors get a `buffer` and streams a `queue` from the Metal backend of the runtime; "
            "until it exists, call `multiply`"
        )


CANDIDATES = [Mps()]

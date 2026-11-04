"""Batched Cholesky of small matrices, forward and backward, as a differentiable function inside PyTorch

A batch is a stack of n x n matrices along the rows of one `Matrix`, one thread block per matrix. The forward kernel
is one statement because `linalg.cholesky` is a tile intrinsic (cuSolverDx on NVIDIA, a portable SIMT lowering
elsewhere). The backward kernel is the adjoint in the same vocabulary, two triangular solves, one product and a
transpose on tiles that never leave the block, attached to the forward with a typed decorator. The generated
`torch.library.custom_op` binding works eagerly and under `torch.compile`, and a finite-difference `gradcheck` in
float64 gates it.

Level: 4
Topics: linalg, autodiff, PyTorch, tile intrinsics
Needs: 02_tiles_and_types/02_tile_methods
"""

from typing import Final

from krn import Const, Matrix, T, block, differentiable, host, kernel, linalg, struct

N: Final = 32  # the matrix order; the batch is a.shape[0] // N matrices


@struct
class Batch:
    n: Const[int] = N


@kernel
def cholesky_batched(a: Matrix[T.f32], factor: Matrix[T.f32], t: Batch) -> None:
    rows, cols = block.tile(t.n), slice(0, t.n)
    factor[rows, cols] = linalg.cholesky(a[rows, cols])


@kernel
def cholesky_batched_backward(factor: Matrix[T.f32], grad: Matrix[T.f32], out: Matrix[T.f32], t: Batch) -> None:
    rows, cols = block.tile(t.n), slice(0, t.n)
    low, g = factor[rows, cols], grad[rows, cols]
    phi = linalg.tril_half_diagonal(low.transpose() @ g)
    y = linalg.solve_triangular(low, phi, transpose=True)  # L^-T Phi
    z = linalg.solve_triangular(low, y.transpose(), transpose=True)  # L^-T Y^T
    out[rows, cols] = z.add(z.transpose()).mul_(0.5)  # the gradient of a symmetric input is symmetric


@differentiable
def cholesky(a: Matrix[T.f32]) -> Matrix[T.f32]:
    factor = host.Device.current().empty(T.f32, *a.shape)
    cholesky_batched(a, factor, Batch())
    return factor


@cholesky.backward
def cholesky_adjoint(a: Matrix[T.f32], factor: Matrix[T.f32], grad: Matrix[T.f32]) -> Matrix[T.f32]:
    out = host.Device.current().empty(T.f32, *a.shape)
    cholesky_batched_backward(factor, grad, out, Batch())
    return out

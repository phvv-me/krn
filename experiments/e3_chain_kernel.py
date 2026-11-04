"""The chain of experiment E3 as a kernel source: parsed by capture, never run, so the checkers skip it.

Scalar math methods such as `.tanh()` are accepted by capture but absent from the typed surface of a scalar; until the
surface has them, a kernel that uses them lives in a file like this one.
"""

from krn import T, Vector, grid, kernel


@kernel
def chain(x: Vector[T.f32], b: Vector[T.f32], out: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < x.shape[0]:
        y = x[i] + b[i]
        y = 0.5 * y * (1.0 + (0.7978845608 * (y + 0.044715 * y * y * y)).tanh())
        out[i] = (-y * y).exp() * y + 1.0

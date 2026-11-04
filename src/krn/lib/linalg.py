"""Dense linear algebra on tiles that fit one thread block, the building blocks of blocked factorizations.

Each function is an `Intrinsic`: one typed contract and one lowering per target (cuSolverDx where it exists, the
portable SIMT lowering of this package everywhere else), so a factorization is written once. Results are lower
triangular unless stated, and the strict upper triangle of a result is zero.
"""

from krn.frontend.memory import Tile
from krn.frontend.numerics import f32, todo


def cholesky[N](a: Tile[f32, N, N]) -> Tile[f32, N, N]:
    """The lower Cholesky factor L with A = L L^T, read from the lower triangle of A."""
    return todo()


def solve_triangular[N, K](factor: Tile[f32, N, N], b: Tile[f32, N, K], *, transpose: bool = False) -> Tile[f32, N, K]:
    """Solve L X = B, or L^T X = B with `transpose=True`, where `factor` is the lower triangular L."""
    return todo()


def tril_half_diagonal[N](a: Tile[f32, N, N]) -> Tile[f32, N, N]:
    """The lower triangle of A with the diagonal halved: the Phi operator of the Cholesky adjoint."""
    return todo()

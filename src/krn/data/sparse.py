"""Matrices whose layout parameter is the stored value dtype. Kernels accumulate in `T.acc` of the output dtype.

Row layouts share `row(i)`, yielding (column, value); COO uses atomic updates instead.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from krn import Matrix, T, Vector, atomic, items, kernel, math, static_assert, struct, target, warp
from krn.frontend.types import FloatingPoint, Idx, u32


@struct
class Rows[F: FloatingPoint](ABC):
    """A matrix stored by rows."""

    values: Vector[F]
    rows: u32
    columns: u32

    @abstractmethod
    def row(self, i: u32, first: u32 | int = 0, step: u32 | int = 1) -> Iterator[tuple[Idx, F]]:
        """The (column, value) of the entries stored for row `i`, from the `first`-th on, every `step`-th."""


@struct
class Csr[F: FloatingPoint](Rows[F]):
    """Compressed sparse row: the entries of a row are next to each other, from `indptr[row]` on."""

    indptr: Vector[u32]
    indices: Vector[u32]

    def row(self, i: u32, first: u32 | int = 0, step: u32 | int = 1) -> Iterator[tuple[Idx, F]]:
        for k in range(self.indptr[i] + first, self.indptr[i + 1], step):
            yield self.indices[k], self.values[k]


@struct
class Ell[F: FloatingPoint](Rows[F]):
    """Ellpack: every row padded to `width` entries (value 0), stored column-major so neighboring rows are neighbors."""

    indices: Vector[u32]
    width: u32

    def row(self, i: u32, first: u32 | int = 0, step: u32 | int = 1) -> Iterator[tuple[Idx, F]]:
        for k in range(first, self.width, step):
            slot = k * self.rows + i
            yield self.indices[slot], self.values[slot]


@struct
class Bsr[F: FloatingPoint](Rows[F]):
    """Blocked CSR: dense `block` by `block` tiles (row-major inside), and one column index per tile, not per entry."""

    indptr: Vector[u32]
    indices: Vector[u32]
    block: u32

    def row(self, i: u32, first: u32 | int = 0, step: u32 | int = 1) -> Iterator[tuple[Idx, F]]:
        start = self.indptr[i // self.block]
        for e in range(first, (self.indptr[i // self.block + 1] - start) * self.block, step):
            tile = start + e // self.block
            column = self.indices[tile] * self.block + e % self.block
            yield (
                column,
                self.values[(tile * self.block + i % self.block) * self.block + e % self.block],
            )


@struct
class Dense[F: FloatingPoint](Rows[F]):
    """A dense matrix, row-major: every column of every row is stored. The same protocol, so `spmv` is a `gemv`."""

    def row(self, i: u32, first: u32 | int = 0, step: u32 | int = 1) -> Iterator[tuple[Idx, F]]:
        for k in range(first, self.columns, step):
            yield k, self.values[i * self.columns + k]


@struct
class Coo[F: FloatingPoint]:
    """Coordinate: one (row, column, value) triple per entry, in any order. No row owns entries, so not `Rows`."""

    values: Vector[F]
    row_indices: Vector[u32]
    column_indices: Vector[u32]
    rows: u32
    columns: u32


@kernel
def butterfly_steps(width: int) -> int:
    return width.bit_length() - 1


@kernel
def lane_sum[F: FloatingPoint](x: F) -> F:
    """The sum over the lanes of a warp, whatever their number."""
    static_assert(target.warp_size & (target.warp_size - 1) == 0, "the lane count must be a power of two")
    for step in range(butterfly_steps(target.warp_size)):
        x += warp.shfl_xor(x, 1 << step)
    return x


@kernel
def spmv[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](a: Rows[A], x: Vector[X], y: Vector[Y]) -> None:
    """y = A x, a thread to a row. Each product is fused into the running sum, kept in `T.acc` of the output type."""
    for i in items(a.rows):
        y[i] = 0.0
        total = y[i].to(T.acc(y.dtype))
        for c, v in a.row(i):
            total = math.fma(v.to(T.acc(y.dtype)), x[c].to(T.acc(y.dtype)), total)
        y[i] = total.to(y.dtype)


@kernel
def spmv_vector[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](a: Rows[A], x: Vector[X], y: Vector[Y]) -> None:
    """y = A x, a warp to a row: the lanes stride over its entries, then add up. Long rows cost no serial chain."""
    for t in items(a.rows * warp.size()):
        i = t // warp.size()
        y[i] = 0.0  # every lane writes the same zero before the lanes' collective, which seeds each partial
        partial = y[i].to(T.acc(y.dtype))
        for c, v in a.row(i, warp.lane(), warp.size()):
            partial = math.fma(v.to(T.acc(y.dtype)), x[c].to(T.acc(y.dtype)), partial)
        total = lane_sum(partial)
        if warp.lane() == 0:
            y[i] = total.to(y.dtype)


@kernel
def clear[F: FloatingPoint](y: Vector[F]) -> None:
    for i in items(y.shape[0]):
        y[i] = 0.0


@kernel
def spmv_coo[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](a: Coo[A], x: Vector[X], y: Vector[Y]) -> None:
    """y += A x, a thread to an entry and an atomic add into the row. `y` starts at zero (see `clear`)."""
    for k in items(a.values.shape[0]):
        atomic.add(
            y,
            a.row_indices[k],
            a.values[k].to(y.dtype) * x[a.column_indices[k]].to(y.dtype),
        )


@kernel
def spmm[A: FloatingPoint, X: FloatingPoint, Y: FloatingPoint](a: Rows[A], b: Matrix[X], c: Matrix[Y]) -> None:
    """C = A B, a thread to an element of C. Each product is fused into the running sum, kept in `T.acc` of C's type."""
    n = b.shape[1]
    for t in items(a.rows * n):
        i, j = divmod(t, n)
        c[i, j] = 0.0
        total = c[i, j].to(T.acc(c.dtype))
        for k, v in a.row(i):
            total = math.fma(v.to(T.acc(c.dtype)), b[k, j].to(T.acc(c.dtype)), total)
        c[i, j] = total.to(c.dtype)


@kernel
def sddmm[P: FloatingPoint, A: FloatingPoint, B: FloatingPoint, Y: FloatingPoint](
    pattern: Csr[P], a: Matrix[A], b: Matrix[B], out: Vector[Y]
) -> None:
    """The entries of A B that the pattern keeps, each the dot product of a row of A and a column of B, with each
    product fused into the running sum."""
    for i in items(pattern.rows):
        for k in range(pattern.indptr[i], pattern.indptr[i + 1]):
            out[k] = 0.0
            total = out[k].to(T.acc(out.dtype))
            for d in range(a.shape[1]):
                total = math.fma(a[i, d].to(T.acc(out.dtype)), b[d, pattern.indices[k]].to(T.acc(out.dtype)), total)
            out[k] = total.to(out.dtype)


@kernel
def spgemm[A: FloatingPoint, B: FloatingPoint, C: FloatingPoint](a: Rows[A], b: Rows[B], c: Matrix[C]) -> None:
    """C = A B for two row layouts, a thread to a row of dense C."""
    for i in items(a.rows):
        for column in range(c.shape[1]):
            c[i, column] = 0.0
        for k, v in a.row(i):
            for j, w in b.row(u32(k)):
                c[i, j] = math.fma(v.to(c.dtype), w.to(c.dtype), c[i, j])

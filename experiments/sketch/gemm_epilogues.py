"""Kernels as objects: a GEMM with an abstract epilogue, resolved at capture with no virtual call on the device

`Epilogue` is an abstract base class, the contract an epilogue meets. Each concrete epilogue is a `@struct` (a
frozen dataclass on the host, a by-value record on the device) that carries its own data (a bias vector, a residual
matrix, a scale) and overrides `apply`. The GEMM is a `@kernel` whose configuration holds the tiling and one
epilogue: capture specializes on the concrete class, so `apply` is inlined with no vtable or indirect branch, and a
new epilogue never touches the GEMM. The launch is the same for every epilogue, since the epilogue is part of the
configuration in the brackets: `gemm[Gemm(BiasGelu(bias))](a, b, out)`.

Level: 4
Topics: structs, abstract classes, epilogues, specialization
Needs: 04_libraries/02_gemm_selectors
"""

from abc import ABC, abstractmethod
from typing import final, override

from krn import Const, Matrix, T, Tile, Vector, block, kernel, pipelined, struct, zeros


class Epilogue(ABC):
    """Turns the float32 accumulator of one output tile into the float16 tile that is stored."""

    @abstractmethod
    def apply(self, acc: Tile[T.f32, int, int], rows: slice, cols: slice) -> Tile[T.f16, int, int]: ...


@final
@struct
class Plain(Epilogue):
    @override
    def apply(self, acc: Tile[T.f32, int, int], rows: slice, cols: slice) -> Tile[T.f16, int, int]:
        return acc.to(T.f16)


@final
@struct
class BiasGelu(Epilogue):
    bias: Vector[T.f16]

    @override
    def apply(self, acc: Tile[T.f32, int, int], rows: slice, cols: slice) -> Tile[T.f16, int, int]:
        v = acc.add(self.bias[cols].to(T.f32))
        return v.div(v.mul(-1.702).exp().add(1.0)).to(T.f16)


@final
@struct
class ScaledResidual(Epilogue):
    residual: Matrix[T.f16]
    alpha: T.f32

    @override
    def apply(self, acc: Tile[T.f32, int, int], rows: slice, cols: slice) -> Tile[T.f16, int, int]:
        return acc.add(self.residual[rows, cols].to(T.f32).mul(self.alpha)).to(T.f16)


@struct
class Gemm[E: Epilogue]:
    """The configuration: the epilogue first, then tiling constants that join the cache key."""

    epilogue: E
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 32


@kernel
def gemm[E: Epilogue](cfg: Gemm[E], a: Matrix[T.f16], b: Matrix[T.f16], out: Matrix[T.f16]) -> None:
    rows, cols = block.tile(cfg.bm, cfg.bn)
    acc = zeros(T.f32, cfg.bm, cfg.bn)
    for ks in pipelined(a.shape[1], cfg.bk):
        acc += a[rows, ks] @ b[ks, cols]
    out[rows, cols] = cfg.epilogue.apply(acc, rows, cols)

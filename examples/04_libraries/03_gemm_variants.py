"""One GEMM, five implementations, picked by what the machine can do

A family starts from a portable kernel that every target can run. Variants are registered next to it with a rule for
where they apply: by type of target (`@gemm.variant(targets.CPU)`), by value (`@gemm.variant(arch={"sm_90a"})`) or
by capability (`@gemm.variant(targets.Nvidia, requires={F.TENSOR_MEMORY})`, the one that survives new hardware). The
most specific rule wins, a tie is an error, and `gemm.audit(...)` runs the selection over every known target so a
missing or ambiguous case fails in CI and not on the cluster. Every variant has the signature of the portable
kernel, which the type checker verifies, and states intent with `schedule=` and `space=` since the algorithm does
not change between Hopper and Blackwell. The prototype shows the selection and the typed contract, not a
tensor-memory lowering.

Level: 4
Topics: families, dispatch by target, capabilities, audit
Needs: 04_libraries/02_gemm_selectors
"""

import logging

from krn import (
    Const,
    Feature,
    Matrix,
    PingPong,
    Space,
    T,
    WarpSpecialized,
    block,
    family,
    pipelined,
    struct,
    targets,
    zeros,
)


@struct
class Tiling:
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 64


@family
def gemm(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """Portable: C = A @ B on whatever matrix unit the target has."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@gemm.variant(targets.Nvidia, requires={Feature.TENSOR_MEMORY})
def gemm_blackwell_tensor_memory(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """The accumulator lives in tensor memory, so registers stay free for the epilogue of the previous tile."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn, space=Space.TENSOR_MEMORY)
    for ks in pipelined(a.shape[1], t.bk, schedule=WarpSpecialized(producers=1, consumers=1)):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@gemm.variant(arch={"sm_90a"})
def gemm_hopper_warp_specialized(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """TMA producer warp and two consumer warpgroups issuing wgmma; accumulators stay in registers."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, schedule=WarpSpecialized(producers=1, consumers=2)):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@gemm.variant(targets.AMD, requires={Feature.ASYNC_COPY})
def gemm_amd_ping_pong(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """Without TMA the waves alternate between loading into LDS and multiplying with MFMA."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, stages=2, schedule=PingPong()):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@gemm.variant(targets.CPU, requires={Feature.TENSOR_CORE})
def gemm_cpu_amx(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """On a CPU a block is a loop nest and a stage is a software prefetch; AMX multiplies 16 x 32 bf16 tiles."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, stages=1):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


def launch(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """Selects the variant for the device of the calling thread or task, with no `if` in the caller."""
    gemm(a, b, c)


if __name__ == "__main__":
    for arch, variant in gemm.audit(targets.all_targets()).items():
        logging.info(f"{arch:9} -> {variant}")

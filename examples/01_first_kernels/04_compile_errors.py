"""Compile-time errors that name the missing feature, the line and the targets that would work

The target is a constant of the capture, so a kernel that needs something the target lacks fails before any vendor
compiler runs, with the line of your own code. Nothing is launched. `block_sum` needs warp shuffles, which a CPU
lacks: the error lists the targets that have them, and its lane loop unrolls to 5 steps for 32 lanes and 6 for 64
from one source line. A GEMM asking for 4 stages of 128 x 128 x 64 needs 128 KB of shared memory and is refused on
an RTX 50 (99 KB) with the arithmetic. `schedule=WarpSpecialized()` is a request, not a hint: without bulk copies
and a specialized lowering it raises `TargetError`, where `AUTO` lets the compiler pick what the target has.

Level: 1
Topics: diagnostics, TargetError, capture, targets
Needs: 01_first_kernels/01_saxpy
"""

import logging
from dataclasses import replace
from pathlib import Path

from krn import (
    AUTO,
    Auto,
    Const,
    Matrix,
    T,
    Vector,
    WarpSpecialized,
    block,
    grid,
    kernel,
    pipelined,
    shared_array,
    static_assert,
    struct,
    target,
    targets,
    warp,
    zeros,
)
from krn.compiler.api import capture, emit
from krn.errors import CaptureError

log = logging.getLogger("targets")
THIS = Path(__file__).resolve()


@kernel
def butterfly_steps(width: int) -> int:
    """Rounds of an xor butterfly over `width` lanes."""
    return width.bit_length() - 1


@kernel
def block_sum(x: Vector[T.f32], out: Vector[T.f32]) -> None:
    """Warp sums, then one slot per warp in shared memory, then the first warp adds those."""
    static_assert(target.warp_size & (target.warp_size - 1) == 0, "the lane count must be a power of two")
    slots = shared_array(T.f32, 1024 // target.warp_size)
    i = grid.thread_rank()
    total = x[i] if i < x.shape[0] else T.f32(0.0)
    for step in range(butterfly_steps(target.warp_size)):  # unrolled: 5 rounds for 32 lanes, 6 for 64
        total += warp.shfl_xor(total, 1 << step)
    if warp.lane() == 0:
        slots[warp.index()] = total
    block.sync()
    if warp.index() == 0:
        out[block.index().x] = slots[warp.lane()] if warp.lane() < slots.shape[0] else T.f32(0.0)


@struct
class GemmTiling:
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 32
    stages: Const[int | Auto] = AUTO


@kernel
def gemm(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32], t: GemmTiling) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, stages=t.stages):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


@kernel
def gemm_warp_specialized(a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32], t: GemmTiling) -> None:
    """The same body asking for producer and consumer warps: a request the target must be able to run (TMA)."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk, schedule=WarpSpecialized()):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc


def try_capture(name: str, arch: targets.Target, consts: dict[str, dict[str, int]] | None = None) -> str:
    """Capture for one target and report the outcome in one line."""
    try:
        kernel = capture(THIS, name, consts=consts, target=arch)
        source = emit(kernel, arch).source
    except CaptureError as error:
        return f"refused: {error}".splitlines()[0].replace(f"{THIS.parent}/", "")
    except NotImplementedError:
        return "captured (no renderer for this vendor yet)"
    return f"ok, {len(source.splitlines())} lines of generated source"


def report(
    title: str, name: str, every_target: list[targets.Target], consts: dict[str, dict[str, int]] | None = None
) -> None:
    log.info("\n%s", title)
    for arch in every_target:
        log.info("  %-9s %s", arch.arch, try_capture(name, arch, consts))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    every = [targets.Target.of(arch) for arch in ("sm_90a", "sm_120a", "gfx942", "apple_m5", "avx512")]
    report("block_sum needs warp shuffles", "block_sum", every)
    report(
        "gemm with a 128 x 128 x 64 tile and 4 stages needs 128 KB of shared memory",
        "gemm",
        [targets.Target.of("sm_90a"), targets.Target.of("sm_120a")],
        {"t": {"bm": 128, "bn": 128, "bk": 64, "stages": 4}},
    )
    report("gemm that asks for a warp-specialized pipeline needs bulk copies", "gemm_warp_specialized", every)
    log.info("\nthe lane loop follows the warp size")
    for width in (32, 64):
        wide = replace(targets.Target.of("sm_90a"), warp_size=width)
        source = emit(capture(THIS, "block_sum", target=wide), wide).source
        steps = next(line for line in source.splitlines() if "step <" in line).strip()
        log.info("  warp_size %d: %s", width, steps)

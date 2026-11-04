"""One linear layer that stores its weights as FP4 where the machine has FP4 and widens gracefully elsewhere

A ladder lists formats narrowest first (`nvfp4 > mxfp4 > mxfp8 > fp8 > fp16 > bf16 > fp32`), and resolving it
against a target picks a rung and keeps the reason: nvfp4 on B200 and RTX 50, mxfp4 on MI355, fp8 on H100, FNUZ fp8
on MI300A, fp16 on A100 and Apple, fp32 where the vector unit has no half precision. Two policies decide how hard it
pushes: `COMPUTE_BOUND` takes the narrowest rung the matrix unit multiplies natively, `MEMORY_BOUND` the narrowest
width at all, decoded in registers. The kernel is generic over the rung and `target.mma_native(w.fmt)` is a
capture-time constant, so the dead branch is never lowered. Narrower is not automatically faster, so a ladder ships
with a benchmark gate and pins, for example `krn.configure(**{"precision.pins": {"weights": "fp8-fn"}})`.

Level: 4
Topics: precision ladders, FP4, FP8, policies, formats
Needs: 04_libraries/02_gemm_selectors
"""

import logging

from krn import Const, Matrix, QMatrix, T, block, formats, host, kernel, pipelined, struct, target, targets, zeros

type Weights = (
    formats.NVFP4
    | formats.MXFP4
    | formats.MXFP8
    | formats.FP8FN
    | formats.FP8FNUZ
    | formats.Plain16
    | formats.PlainBF16
    | formats.Plain32
)

WEIGHTS = host.Ladder[Weights](
    "weights",
    (
        formats.NVFP4(),
        formats.MXFP4(),
        formats.MXFP8(),
        formats.FP8FN(),
        formats.FP8FNUZ(),
        formats.Plain16(),
        formats.PlainBF16(),
        formats.Plain32(),
    ),
)


@struct
class Tiling:
    bm: Const[int] = 64
    bn: Const[int] = 128
    bk: Const[int] = 128


@kernel
def linear[F: Weights](t: Tiling, x: Matrix[T.f16], w: QMatrix[F], y: Matrix[T.f16]) -> None:
    """y = x @ w with weights in format F and float32 accumulation."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    for ks in pipelined(x.shape[1], t.bk):
        if target.mma_native(w.fmt):
            acc += x[rows, ks].quantize(w.fmt) @ w[ks, cols]  # both operands in F: a native (block-scaled) product
        else:
            acc += x[rows, ks] @ w[ks, cols].dequantize(T.f16)  # codes widened to float16 in registers, then a product
    y[rows, cols] = acc.to(T.f16)


if __name__ == "__main__":
    for policy in (host.Policy.COMPUTE_BOUND, host.Policy.MEMORY_BOUND):
        ladder = host.Ladder("weights", WEIGHTS.rungs, policy)
        logging.info(f"\n{policy}")
        for each in targets.all_targets():
            r = ladder.resolve(each, host.Settings())
            logging.info(f"  {each.arch:9} store {r.storage.name:9} {r.route:7} {r.reason.split(';')[0]}")

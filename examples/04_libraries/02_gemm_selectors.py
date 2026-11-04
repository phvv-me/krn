"""A GEMM written once as a template: configuration in brackets, data in parentheses, float type by the target

`gemm(a, b, c)` uses the configuration defaults and `gemm[Tiling(bm=64, bn=128)](a, b, c)` chooses one; the body
supplies the geometry. The first parameter is the configuration because it is a `@struct` of `Const` fields, and
each distinct configuration is its own specialization in the cache. The float type is the other half of the
template: `a` and `b` carry the type `F`, named (`T.f16`) or chosen by the machine (`T.fp.min` is the narrowest
float the target has, `T.fp.max` the widest, `T.fp.emax(8)` and `mmax(8)` the 8-bit float with the most exponent or
mantissa bits). So `device.empty(T.fp.min, m, k)` allocates FP4 on Blackwell, FNUZ FP8 on Sirius and FP16 on an A100
or an Apple GPU, and `gemm` follows. Run the file to see what each selector resolves to on every target.

Level: 4
Topics: templates, configuration, float type selectors, targets
Needs: 04_libraries/01_gemm_tiles
"""

import logging

from krn import Const, Matrix, T, block, host, kernel, pipelined, struct, targets, zeros


@struct
class Tiling:
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 32


@kernel
def gemm[F: T.FloatingPoint, A: T.FloatingPoint](t: Tiling, a: Matrix[F], b: Matrix[F], c: Matrix[A]) -> None:
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.acc(a.dtype), t.bm, t.bn)
    for ks in pipelined(a.shape[1], t.bk):
        acc += a[rows, ks] @ b[ks, cols]
    c[rows, cols] = acc.to(c.dtype)


def launch(a: Matrix[T.FloatingPoint], b: Matrix[T.FloatingPoint], c: Matrix[T.FloatingPoint]) -> None:
    """Use the struct defaults or choose a configuration; infer geometry from the body in both cases."""
    gemm(a, b, c)
    gemm[Tiling(bm=64, bn=128)](a, b, c)


if __name__ == "__main__":
    selectors = {
        "fp.min": T.fp.min,
        "fp.max": T.fp.max,
        "fp.emax(8)": T.fp.emax(8),
        "fp.mmax(8)": T.fp.mmax(8),
        "acc(fp.min)": T.acc(T.fp.min),
    }
    logging.info(f"{'target':9}" + "".join(f"{name:>13}" for name in selectors))
    for target in targets.all_targets():
        row = []
        for selector in selectors.values():
            try:
                row.append(T.name(selector.resolve(target)))
            except LookupError:
                row.append("-")
        logging.info(f"{target.arch:9}" + "".join(f"{name:>13}" for name in row))
    logging.info(f"\nthis machine: {host.System.detect().target.arch}")

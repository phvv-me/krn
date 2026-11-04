"""The reduction order the language defines, in numpy, bit for bit what every backend computes.

One output element of a tile reduction is folded by `lanes` lanes. Lane t folds the valid elements t, t + lanes,
t + 2 * lanes, ... of the reduced axis in index order, starting from the identity of the operation. The lanes are cut
into warps of 32 (the last warp is padded with identities), the lanes of a warp combine through an xor butterfly
(offsets 16, 8, 4, 2, 1, each lane combining its own value first), and the value of lane 0 of each warp is the warp's.
The warps combine in warp order. A mean divides the total by the number of valid elements.

Sums depend on `lanes`, the block of the launch, and on nothing else. Maximum and minimum do not, except for the sign
of a zero and the payload of a NaN.
"""

import numpy as np

from krn.compiler.heuristics import WARP, check_lanes
from krn.compiler.ir import Reduction


def identity(op: Reduction, dtype: np.dtype) -> np.generic:
    """The value that leaves any other unchanged: a float sum starts from -0.0, so that it cannot turn a -0.0 into 0.0."""
    if dtype.kind == "f":
        value = {Reduction.MAX: -np.inf, Reduction.MIN: np.inf, Reduction.PROD: 1.0}.get(op, -0.0)
    else:
        info = np.iinfo(dtype)
        value = {Reduction.MAX: info.min, Reduction.MIN: info.max, Reduction.PROD: 1}.get(op, 0)
    return dtype.type(value)


def combine(op: Reduction, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`a` then `b`: a NaN in either is the result, and on a tie the maximum and the minimum keep `b`."""
    match op:
        case Reduction.MAX:
            return np.where((a > b) | (a != a), a, b)
        case Reduction.MIN:
            return np.where((a < b) | (a != a), a, b)
        case Reduction.PROD:
            return a * b
    return a + b


def lane_reduce(
    op: Reduction, x: np.ndarray, valid: np.ndarray | None, axis: int, keepdims: bool, lanes: int
) -> np.ndarray:
    """`op` over the valid elements of `x` along `axis` in the order of the module docstring; a float16 tile sums in
    float32, as the backends do, and the caller rounds the result."""
    check_lanes(lanes)
    work = np.dtype(np.float32) if x.dtype == np.float16 else x.dtype
    zero = identity(op, work)
    x = np.moveaxis(x.astype(work, copy=False), axis, -1)
    where = None if valid is None else np.moveaxis(np.broadcast_to(valid, np.moveaxis(x, -1, axis).shape), axis, -1)
    if where is not None:
        x = np.where(where, x, zero)
    count = x.shape[-1]
    chunks = -(-count // lanes)
    padded = np.full((*x.shape[:-1], chunks * lanes), zero, work)
    padded[..., :count] = x
    padded = padded.reshape(*x.shape[:-1], chunks, lanes)
    partial = np.full((*x.shape[:-1], -(-lanes // WARP) * WARP), zero, work)
    for chunk in range(chunks):
        partial[..., :lanes] = combine(op, partial[..., :lanes], padded[..., chunk, :])
    warps = partial.reshape(*x.shape[:-1], -1, WARP)
    partner = np.arange(WARP)
    for offset in (16, 8, 4, 2, 1):
        warps = combine(op, warps, warps[..., partner ^ offset])
    total = warps[..., 0, 0]
    for warp in range(1, warps.shape[-2]):
        total = combine(op, total, warps[..., warp, 0])
    if op is Reduction.MEAN:
        with np.errstate(divide="ignore", invalid="ignore"):
            total = total / (count if where is None else where.sum(axis=-1)).astype(work)
    return np.expand_dims(total, axis) if keepdims else total

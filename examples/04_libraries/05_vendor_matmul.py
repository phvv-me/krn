"""One call that reaches cuBLASLt, hipBLASLt, Accelerate or oneDNN when they are better, and our kernels when not

`lib.matmul(a, b)` is a plain function. On first use `host.System.detect()` identifies the machine, each candidate
says whether it can run the problem, and the survivors are ordered: the user's filter (`krn.configure(lib="ours")`),
a measured winner on disk, the static preference per vendor, and a lazy measurement on the eighth call of a key. A
refusal carries its reason and never falls back silently, and `lib.report()` prints which candidate ran and why.
Plan-time options live in one frozen dataclass whose result-changing fields are part of every key. Only `ours`,
`host` and `accelerate` run through `lib.matmul` today.

Level: 4
Topics: vendor libraries, dispatch, lib.matmul, measured selection
Needs: 04_libraries/01_gemm_tiles
"""

import asyncio
import logging
import math

import numpy as np

from krn import Matrix, T, Vector, host, lib

log = logging.getLogger("feed_forward")


def feed_forward(x: Matrix[T.f16], w_up: Matrix[T.f16], bias: Vector[T.f16], w_down: Matrix[T.f16]) -> Matrix[T.f16]:
    """Two matmuls with one fused epilogue each: the vendor library where it fuses it, our kernel where it does not."""
    hidden = lib.matmul(x, w_up, epilogue=lib.Epilogue.BIAS_GELU, bias=bias)
    return lib.matmul(hidden, w_down, options=lib.LibOptions(determinism=lib.Determinism.STABLE))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.vectorize(math.erf, otypes=[np.float64])(x / np.sqrt(2)))


def main() -> None:
    """Run the block on this machine's device and check it against numpy, whichever candidate the call chose."""
    device = host.Device.current()
    rng = np.random.default_rng(0)
    x, w_up, bias, w_down = (
        device.empty(T.f16, 8, 32),
        device.empty(T.f16, 32, 64),
        device.empty(T.f16, 64),
        device.empty(T.f16, 64, 32),
    )
    for tensor in (x, w_up, bias, w_down):
        np.asarray(tensor)[:] = rng.standard_normal(tensor.shape) / 4
    y = feed_forward(x, w_up, bias, w_down)
    asyncio.run(device.synchronize())
    f32 = {
        name: np.asarray(t).astype(np.float32) for name, t in (("x", x), ("up", w_up), ("b", bias), ("down", w_down))
    }
    hidden = gelu(f32["x"] @ f32["up"] + f32["b"]).astype(np.float16).astype(np.float32)
    error = float(np.abs(np.asarray(y).astype(np.float32) - hidden @ f32["down"]).max())
    log.info("feed_forward on %s: largest error against numpy %.2e", device.target.arch, error)
    assert error < 5e-3, error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

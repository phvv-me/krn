import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from statistics import fmean

from ._logging import logger

_MICROSECONDS_PER_SECOND = 1e6


@dataclass(frozen=True, kw_only=True)
class BenchSample:
    """Timing of one callable.

    samples: microseconds of each timed iteration in run order; every aggregate reads them.
    """

    label: str
    samples: tuple[float, ...]

    @property
    def mean_us(self) -> float:
        return fmean(self.samples)

    @property
    def min_us(self) -> float:
        return min(self.samples)

    @property
    def runs(self) -> int:
        return len(self.samples)


def _noop() -> None:
    pass


def benchmark[T, S](
    fn: Callable[[], T],
    *,
    label: str = "fn",
    iters: int = 20,
    warmup: int = 3,
    sync: Callable[[], S] | None = None,
) -> BenchSample:
    """Time zero-arg `fn` over `iters` runs after `warmup` untimed calls.

    sync: a barrier called after the warmup and after each run (e.g.
        `torch.cuda.synchronize`) so async GPU work is timed, not just its launch.
    """
    if iters < 1 or warmup < 0:
        raise ValueError("benchmark requires iters >= 1 and warmup >= 0")
    barrier: Callable[[], S | None] = sync or _noop
    for _ in range(warmup):
        fn()
    barrier()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        barrier()
        samples.append((time.perf_counter() - start) * _MICROSECONDS_PER_SECOND)
    return BenchSample(label=label, samples=tuple(samples))


def compare[T, S](
    cases: Mapping[str, Callable[[], T]],
    *,
    iters: int = 20,
    warmup: int = 3,
    sync: Callable[[], S] | None = None,
) -> list[BenchSample]:
    """Benchmark each named callable and print a mean/min-time table, fastest first."""
    samples = sorted(
        (benchmark(fn, label=name, iters=iters, warmup=warmup, sync=sync) for name, fn in cases.items()),
        key=lambda s: s.mean_us,
    )
    width = max((len(s.label) for s in samples), default=4)
    for sample in samples:
        logger.info(f"{sample.label:{width}s}  {sample.mean_us:10.3f} us  (min {sample.min_us:.3f})")
    return samples

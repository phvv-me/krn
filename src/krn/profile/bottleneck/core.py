# One-call bottleneck profiling and GPU-contention gating.

from typing import TYPE_CHECKING

from ..enums import Activity
from ..profiler import Profiler
from ..spans import span
from .report import ProfileReport

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...hw.devices.base import Device


def profile[T, S](
    fn: Callable[[], T],
    *,
    gpu: Device | None = None,
    iters: int = 1,
    warmup: int = 0,
    sync: Callable[[], S] | None = None,
    kinds: Activity = Activity.KERNEL,
) -> ProfileReport:
    """Run zero-arg `fn` `iters` times in one traced pass, after `warmup` untimed runs.

    gpu: overrides the session's own device discovery. Bandwidth is scored against whichever
        device the session settled on, so a report never loses its peak for want of this.
    sync: a device barrier after each run (e.g. `torch.cuda.synchronize`) so async GPU work
        is captured rather than just the launch.
    kinds: unsupported native activity kinds raise TargetError. Vendor kernel collection is always requested,
        so foreign library kernels (Torch, CuPy) are timed; a missing vendor extra raises rather than degrading.
    """
    session = _run(fn, iters=iters, warmup=warmup, sync=sync, kinds=kinds, gpu=gpu)
    scored = session.sampler.gpu
    return ProfileReport.from_profile(
        session.result(),
        iterations=iters,
        peak_bandwidth_gbps=scored.peak.memory_bandwidth_gbs if scored is not None else None,
    )


def _run[T, S](
    fn: Callable[[], T],
    *,
    iters: int,
    warmup: int,
    sync: Callable[[], S] | None,
    kinds: Activity,
    gpu: Device | None,
) -> Profiler:
    """The finished session rather than its result, since the caller also reads its device."""
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    # ACTIVITY only when a backend granted kinds, since the profiler refuses a deep trace
    # nobody can serve rather than returning an empty pass.
    deep = Profiler.Feature.ACTIVITY if kinds else Profiler.Feature(0)
    with Profiler(
        gpus=(gpu,) if gpu is not None else (),
        features=(Profiler.Feature.SPANS | Profiler.Feature.DEVICE | deep),
        activities=kinds,
        native=True,
    ) as profiler:
        for _ in range(iters):
            with span("fn"):
                fn()
            if sync is not None:
                sync()
    return profiler


def gpu_busy(
    gpu: Device | None,
    *,
    util_threshold: int = 10,
    memory_threshold_pct: float = 90.0,
) -> bool:
    """Whether `gpu` is under load, its compute or memory percent over the threshold.

    A CPU-only host (None) always reads as idle.
    """
    if gpu is None:
        return False
    reading = gpu.snapshot()
    used, total = reading.memory_used_bytes, reading.memory_total_bytes
    memory_busy = used is not None and total is not None and total > 0 and 100 * used / total > memory_threshold_pct
    return (reading.utilization_percent or 0) > util_threshold or memory_busy

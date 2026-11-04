from dataclasses import dataclass

from ...hw.devices.base import Device, Telemetry
from .meter import Meter

_BYTES_PER_GIB = 2**30


@dataclass(frozen=True, kw_only=True)
class Diagnosis:
    """One-line hardware verdict for a metered trial.

    near_oom: peak GPU memory came within `headroom_pct` of device capacity.
    gpu_underutilized: compute utilization at or below `util_floor` percent.
    host_offload: host memory grew by `host_growth_gb` or more, or other processes shared the GPU.
    throttled: a non-benign thermal/power throttle was active.
    unknown: the checks whose evidence was unavailable; they neither fired nor cleared.
    reason: the most severe flag rendered, else the unavailable evidence, else `healthy`.
    """

    near_oom: bool = False
    gpu_underutilized: bool = False
    host_offload: bool = False
    throttled: bool = False
    unknown: tuple[str, ...] = ()
    reason: str = "healthy"

    @classmethod
    def diagnose(
        cls,
        meter: Meter,
        gpu: Device | None,
        *,
        headroom_pct: float = 5.0,
        util_floor: int = 25,
        host_growth_gb: float = 4.0,
    ) -> Diagnosis:
        """Diagnose a closed `meter` against a snapshot of the live `gpu` and its capacity.

        gpu: None on a CPU-only host, where there is no GPU to judge, so the verdict says so rather than healthy.
        """
        if gpu is None:
            return cls(unknown=("GPU",), reason="unknown: no GPU to diagnose")
        processes = gpu.processes()
        total = gpu.memory.total
        return cls.of(
            meter,
            gpu.snapshot(),
            capacity_gb=None if total is None else total / _BYTES_PER_GIB,
            headroom_pct=headroom_pct,
            util_floor=util_floor,
            host_growth_gb=host_growth_gb,
            process_count=None if processes is None else len(processes),
        )

    @classmethod
    def of(
        cls,
        meter: Meter,
        gpu: Telemetry,
        *,
        capacity_gb: float | None,
        headroom_pct: float = 5.0,
        util_floor: int = 25,
        host_growth_gb: float = 4.0,
        process_count: int | None = None,
    ) -> Diagnosis:
        """Diagnose a closed `meter` from a device snapshot taken near the trial.

        capacity_gb: total device memory, the near-OOM denominator; None when the device does not report it.
        """
        peak = meter.peak_gpu_gb
        used_pct = None if peak is None or capacity_gb is None or capacity_gb <= 0 else 100.0 * peak / capacity_gb
        near_oom = used_pct is not None and used_pct >= 100.0 - headroom_pct
        compute_pct = gpu.utilization_percent
        gpu_underutilized = compute_pct is not None and compute_pct <= util_floor
        contended = None if process_count is None else process_count > 1
        host_offload = meter.host_delta_gb >= host_growth_gb or contended is True
        throttled = bool(gpu.throttle_reasons)
        unknown = tuple(
            name
            for name, missing in (
                ("GPU memory", used_pct is None),
                ("GPU utilization", compute_pct is None),
                ("throttle state", gpu.throttle_reasons is None),
                ("process count", process_count is None),
            )
            if missing
        )
        # Severity order: an imminent OOM kill outranks a throttle, which outranks contention
        # or offload thrash, which outranks an idle GPU.
        if used_pct is not None and near_oom:
            reason = f"near OOM: {used_pct:.0f}% of device memory in use"
        elif throttled:
            reason = f"throttled: {', '.join(gpu.throttle_reasons or ())}"
        elif host_offload and contended:
            reason = f"host offload: {process_count} processes share the GPU"
        elif host_offload:
            reason = f"host offload: host memory grew {meter.host_delta_gb:.1f} GB"
        elif gpu_underutilized:
            reason = f"GPU underutilized: {compute_pct}% compute"
        elif unknown:
            reason = f"unknown: {', '.join(unknown)} unavailable"
        else:
            reason = "healthy"
        return cls(
            near_oom=near_oom,
            gpu_underutilized=gpu_underutilized,
            host_offload=host_offload,
            throttled=throttled,
            unknown=unknown,
            reason=reason,
        )

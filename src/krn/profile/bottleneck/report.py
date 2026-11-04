# One-call bottleneck reporting: run a callable, say where the GPU time went.

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .._constants import BYTES_PER_GB, NANOSECONDS_PER_MILLISECOND
from ..enums import Activity, Bound
from ..launches import LaunchRecord
from ..result import DeviceEvidence, Profile
from ..result.models import known_max, known_mean
from ..trace import BottleneckReport, HotKernel, KernelTrace


def _fully_counted(record: LaunchRecord) -> bool:
    """Whether every arithmetic operation and payload byte of `record` is known and dtype-attributed."""
    return (
        record.flops is not None
        and record.bytes_moved is not None
        and sum(record.flops_by_dtype.values()) == record.flops
    )


@dataclass(frozen=True, kw_only=True)
class KernelStat:
    """One kernel name's aggregate over the run: its share and representative shape.

    The shape fields come from the last-seen launch of the name, since launches of one name
    share a config. occupancy_pct: threads per block over the hardware max of 1024, a proxy,
    since the CUPTI activity record carries the launch config but not achieved occupancy.
    """

    name: str
    calls: int
    total_ns: int
    avg_ns: float
    share_pct: float
    grid: str
    block: str
    threads_per_block: int
    occupancy_pct: float
    registers: int
    static_shared_mem: int
    dynamic_shared_mem: int


@dataclass(frozen=True, kw_only=True)
class ProfileReport:
    """Structured bottleneck verdict for one profiled callable.

    dominant_kernel/dominant_share_pct: the hottest kernel and its slice of kernel time.
    total_kernel_ns/total_memcpy_ns/device_busy_ns: summed WORK time per class and the CLOCK
        time busy with either, as on `BottleneckReport`.
    achieved_bandwidth_gbps/peak_bandwidth_gbps: copy bandwidth against the device peak, the
        memory-bound signal.
    peak_memory_bytes/avg_memory_bytes: the sampled device-memory high-water mark and mean,
        how much HBM the work needed.
    kernels: the per-kernel breakdown, hottest first.
    unavailable: activity-kind labels the device could not trace.
    device_evidence: whether device evidence was asked for and came back, so an empty verdict
        never reads like a real one.
    """

    device: str = ""
    device_evidence: DeviceEvidence = DeviceEvidence.UNSOUGHT
    iterations: int = 0
    dominant_kernel: str = ""
    dominant_share_pct: float = 0.0
    bound: Bound = Bound.UNKNOWN
    total_kernel_ns: int = 0
    total_memcpy_ns: int = 0
    total_memcpy_bytes: int = 0
    device_busy_ns: int = 0
    compute_pct: float = 0.0
    memcpy_pct: float = 0.0
    achieved_bandwidth_gbps: float | None = None
    peak_bandwidth_gbps: float | None = None
    peak_memory_bytes: int | None = None
    avg_memory_bytes: int | None = None
    avg_memory_util_pct: float | None = None
    avg_compute_util_pct: float | None = None
    kernels: tuple[KernelStat, ...] = ()
    unavailable: tuple[str, ...] = ()

    @classmethod
    def from_profile(
        cls,
        profile: Profile,
        *,
        iterations: int,
        peak_bandwidth_gbps: float | None = None,
        supported: int | None = None,
        requested: int | None = None,
    ) -> ProfileReport:
        """Distill a `Profile` into a bottleneck verdict.

        peak_bandwidth_gbps: device peak to score copy bandwidth against, 0 disables the score.
        supported/requested: the `Activity` values the device offered and the run asked for;
            their difference becomes `unavailable`, so a partial trace is visible.
        """
        split = BottleneckReport.from_traces((), profile.kernels, profile.memcpys, top=len(profile.kernels))
        kernels = cls._kernels(profile.kernels, split.hot_kernels)
        mem_util, gpu_util = cls._utilization(profile)
        peak_memory, avg_memory = cls._memory(profile)
        copy_ns = split.total_memcpy_ns
        return cls(
            device=profile.device,
            device_evidence=profile.device_evidence,
            iterations=iterations,
            dominant_kernel=kernels[0].name if kernels else "",
            dominant_share_pct=kernels[0].share_pct if kernels else 0.0,
            bound=cls._roofline(profile)
            if profile.launches
            else cls._classify(
                kernel_ns=split.total_kernel_ns,
                memcpy_ns=copy_ns,
                mem_util=mem_util,
                gpu_util=gpu_util,
            ),
            total_kernel_ns=split.total_kernel_ns,
            total_memcpy_ns=copy_ns,
            total_memcpy_bytes=split.total_memcpy_bytes,
            device_busy_ns=split.device_busy_ns,
            compute_pct=split.compute_pct,
            memcpy_pct=split.memcpy_pct,
            # Bytes per nanosecond equals GB/s.
            achieved_bandwidth_gbps=profile.efficiency().achieved_gbs
            if profile.launches
            else (split.total_memcpy_bytes / copy_ns if copy_ns > 0 else None),
            peak_bandwidth_gbps=peak_bandwidth_gbps
            if peak_bandwidth_gbps is not None
            else (profile.peak.memory_bandwidth_gbs if profile.peak is not None else None),
            peak_memory_bytes=peak_memory,
            avg_memory_bytes=avg_memory,
            avg_memory_util_pct=mem_util,
            avg_compute_util_pct=gpu_util,
            kernels=kernels,
            unavailable=cls._unavailable(supported=supported, requested=requested),
        )

    @staticmethod
    def _roofline(profile: Profile) -> Bound:
        """Memory- or compute-bound by which ceiling takes longer over every launch.

        Each dtype's arithmetic runs at that dtype's own peak, so mixed precision adds its compute times.
        Unknown work, arithmetic without a known dtype, or a dtype without a ceiling gives UNKNOWN.
        """
        peak = profile.peak
        if peak is None or peak.memory_bandwidth_gbs is None or not profile.launches:
            return Bound.UNKNOWN
        if any(not _fully_counted(r) for r in profile.launches):
            return Bound.UNKNOWN
        ops: Counter[str] = Counter()
        for record in profile.launches:
            ops.update(record.flops_by_dtype)
        if any(count and dtype not in peak.flops for dtype, count in ops.items()):
            return Bound.UNKNOWN
        payload = sum(r.bytes_moved for r in profile.launches if r.bytes_moved is not None)
        # GFLOP/s is flops per nanosecond and GB/s is bytes per nanosecond, so both quotients are times in ns.
        compute_ns = sum(count / peak.flops[dtype] for dtype, count in ops.items() if count)
        memory_ns = payload / peak.memory_bandwidth_gbs
        if compute_ns == 0 and memory_ns == 0:
            return Bound.UNKNOWN
        return Bound.COMPUTE if compute_ns >= memory_ns else Bound.MEMORY

    @staticmethod
    def _classify(*, kernel_ns: int, memcpy_ns: int, mem_util: float | None, gpu_util: float | None) -> Bound:
        """Memory- or compute-bound by the copy/kernel time split, else by utilization."""
        if kernel_ns or memcpy_ns:
            return Bound.MEMORY if memcpy_ns >= kernel_ns else Bound.COMPUTE
        if mem_util is not None and gpu_util is not None and (mem_util or gpu_util):
            return Bound.MEMORY if mem_util >= gpu_util else Bound.COMPUTE
        return Bound.UNKNOWN

    @staticmethod
    def _kernels(kernels: Sequence[KernelTrace], hottest: Sequence[HotKernel]) -> tuple[KernelStat, ...]:
        """Attach each hot kernel name's last-seen launch shape."""
        shape = {kernel.name: kernel for kernel in kernels}
        return tuple(
            KernelStat(
                name=hot.name,
                calls=hot.calls,
                total_ns=hot.total_ns,
                avg_ns=hot.avg_ns,
                share_pct=hot.share_pct,
                grid=shape[hot.name].grid,
                block=shape[hot.name].block,
                threads_per_block=shape[hot.name].threads_per_block,
                occupancy_pct=shape[hot.name].occupancy_pct,
                registers=shape[hot.name].registers,
                static_shared_mem=shape[hot.name].static_shared_mem,
                dynamic_shared_mem=shape[hot.name].dynamic_shared_mem,
            )
            for hot in hottest
        )

    @staticmethod
    def _memory(profile: Profile) -> tuple[int | None, int | None]:
        """Device-memory (peak single sample, mean across regions) in bytes.

        Zero when nothing was sampled, e.g. a kernel that finished between two sampler ticks,
        where the deep kernel trace remains the reliable signal.
        """
        summaries = profile.summaries
        if not summaries:
            return None, None
        peak = known_max(s.peak_memory_bytes for s in summaries)
        avg = known_mean(s.avg_memory_bytes for s in summaries)
        return peak, None if avg is None else int(avg)

    @staticmethod
    def _unavailable(*, supported: int | None, requested: int | None) -> tuple[str, ...]:
        if supported is None or requested is None:
            return ()
        missing = Activity(requested) & ~Activity(supported)
        return tuple(flag.label for flag in Activity if flag in missing and flag.label != "all")

    @staticmethod
    def _utilization(profile: Profile) -> tuple[float | None, float | None]:
        """Mean sampled (memory-controller, compute) utilization across all regions."""
        summaries = profile.summaries
        if not summaries:
            return None, None
        compute = known_mean(s.avg_util_pct for s in summaries)
        memory = known_mean(s.avg_memory_util_pct for s in summaries)
        return memory, compute

    def __str__(self) -> str:
        return self.report()

    def _notes(self) -> str:
        """Trailing notes: missing device evidence, peak memory, bandwidth, untraced kinds."""
        notes = []
        if self.device_evidence is DeviceEvidence.ABSENT:
            notes.append("no device evidence collected: asked for it and observed none")
        if self.peak_memory_bytes:
            notes.append(f"peak memory {self.peak_memory_bytes / BYTES_PER_GB:.3f} GB")
        if self.peak_bandwidth_gbps and self.achieved_bandwidth_gbps is not None:
            pct = 100.0 * self.achieved_bandwidth_gbps / self.peak_bandwidth_gbps
            notes.append(
                f"copy bandwidth {self.achieved_bandwidth_gbps:.1f}/"
                f"{self.peak_bandwidth_gbps:.1f} GB/s ({pct:.0f}% of peak)"
            )
        if self.unavailable:
            notes.append(f"unavailable on this device: {', '.join(self.unavailable)}")
        return "\n" + "\n".join(notes) if notes else ""

    def report(self) -> str:
        """A compact plain-text verdict and per-kernel table."""
        head = (
            f"device {self.device or 'cpu'} | {self.bound.value}-bound | "
            f"dominant {self.dominant_kernel or '(none)'} {self.dominant_share_pct:.1f}%"
        )
        if not self.kernels:
            return f"{head}\nNo kernels traced." + self._notes()
        rows = [f"{'kernel':<40}{'calls':>6}{'total ms':>10}{'share%':>8}{'regs':>6}"]
        rows += [
            f"{k.name[:40]:<40}{k.calls:>6d}{k.total_ns / NANOSECONDS_PER_MILLISECOND:>10.3f}"
            f"{k.share_pct:>8.1f}{k.registers:>6d}"
            for k in self.kernels
        ]
        return "\n".join([head, *rows]) + self._notes()

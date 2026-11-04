import time
from collections import defaultdict, deque
from collections.abc import (
    Callable,
    Iterator,
    Sequence,
)
from contextlib import ExitStack
from dataclasses import replace
from itertools import islice
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, TypeAlias

from ..compiler.errors import TargetError
from ..facts.targets import Target
from ..hw.devices.base import Device
from ..hw.enums import DeviceKind, Vendor
from ..hw.system import System
from . import launches
from .bottleneck import annotate
from .bottleneck.owned import module_codes
from .enums import Activity as NativeActivity
from .launches import LaunchRecord
from .recording import DeviceSampler, SpanRecorder
from .result import DeviceEvidence, Profile
from .result.models import RegionSummary
from .session import Collection, Feature
from .spans import activate, active, deactivate
from .trace import UNPOSITIONED, KernelTrace, MemcpyTrace, RegionWindow, TraceCollector
from .tracer import Tracer

if TYPE_CHECKING:
    import Metal

    from ..compiler.ir import Kernel
    from ..runtime.cuda_backend import CUDACall, CUDAEvent, CUDAStream
    from ..runtime.metal import MetalCall, MetalStream
    from ..runtime.runtime import Call, CPUStream

_MAX_SPANS = 100_000
_CLOCKS = {"cpu-wall": "host", "metal-gpu": "metal"}


class Profiler:
    """Collect selected evidence through one bounded profiling session.

    `span` annotations stay dormant until this context is active, then open and close in
    `spans`, while `sampler` reads the selected device. `features` controls what may be
    collected; the resulting `Profile` holds only evidence actually observed.

    Launch hooks never fail a launch: once the session stops admitting new work, a hook only
    executes the launch. Work admitted earlier is still recorded, because exit settles it first.
    """

    # Not a PEP 695 `type`, whose `TypeAliasType` would not forward `Profiler.Feature.SPANS`.
    Feature: TypeAlias = Feature  # noqa: UP040  reason=type statement would not forward attribute access since=2026-08-16
    Activity = NativeActivity

    active: bool
    auto_on: bool
    collection: Collection
    collector: TraceCollector
    dropped_launches: int
    gpus: Sequence[Device]
    launches: deque[LaunchRecord]
    sampler: DeviceSampler
    spans: SpanRecorder

    @classmethod
    def capture[Answer](
        cls,
        work: Callable[[], Answer],
        *,
        activities: NativeActivity = NativeActivity.KERNEL,
        device_index: int = 0,
    ) -> tuple[Answer, Profile]:
        """Run once inside a synchronized, single-context native activity window.

        Reuses the active activity owner or opens one; nested windows are views of its records,
        never new subscribers or additions to its physical totals. All CUDA streams in the
        current context are synchronized, and CUDA must be issued serially from one host
        thread: unrelated concurrent CUDA work is not attributed. The active owner's `native`
        setting decides what the window collects; a window opened here is always native.
        """
        if not activities or activities & ~NativeActivity.DEFAULT:
            raise ValueError("synchronized windows currently support kernels and memory copies")
        owner = active()
        if owner is not None:
            if not isinstance(owner, cls):
                raise RuntimeError("the active span owner cannot provide native activity windows")
            if owner.collection.device_index != device_index:
                raise RuntimeError("activity window requested a different CUDA-visible device")
            return owner.capture_window(work, activities)
        with cls(
            features=Feature.SPANS | Feature.ACTIVITY, activities=activities, device_index=device_index, native=True
        ) as owner:
            return owner.capture_window(work, activities)

    @classmethod
    def under(cls, collection: Collection, *, gpus: Sequence[Device] = ()) -> Profiler:
        """Build a profiler from one collection policy, for a caller that holds one already.

        The constructor takes the choices flat for a one-line caller; a study hands the value
        over rather than risk unpacking it differently next time.
        """
        return cls(
            gpus=gpus,
            features=collection.features,
            activities=collection.activities,
            device_index=collection.device_index,
            sample_interval_ms=collection.sample_interval_ms,
            max_spans=collection.max_spans,
            auto=collection.auto,
            native=collection.native,
        )

    def __enter__(self) -> Profiler:
        if self.active:
            raise RuntimeError("a Profiler instance cannot be entered twice")
        wanted = self.collection.features
        if wanted & (self.Feature.DEVICE | self.Feature.ACTIVITY | self.Feature.MARKERS):
            # Handed no probe, discover the host's own cards rather than collecting nothing.
            self.gpus = self.gpus or tuple(
                device for device in System.detect().visible_devices if device.kind is DeviceKind.GPU
            )
            self.sampler.gpu = self._selected()
        if wanted & self.Feature.MARKERS or (wanted & self.Feature.ACTIVITY and self._needs_native()):
            selected = self.sampler.gpu
            present = (
                frozenset({selected.vendor})
                if selected is not None
                else (frozenset(gpu.vendor for gpu in self.gpus) or frozenset({Vendor.UNKNOWN}))
            )
            self.spans.tracer = Tracer.detect(present=present)
        with ExitStack() as rollback:
            activate(self)
            rollback.callback(deactivate, self)
            self._armed(rollback, wanted)
            self._admitting = True
            launches.session = self if wanted & Feature.ACTIVITY else None
            rollback.callback(setattr, launches, "session", None)
            self.active = True
            rollback.pop_all()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        with ExitStack() as cleanup:
            cleanup.callback(setattr, self, "active", False)
            cleanup.callback(self._stop_collector)
            cleanup.callback(self.sampler.stop)
            cleanup.callback(deactivate, self)
            cleanup.callback(setattr, launches, "session", None)
            if self.auto_on:
                cleanup.callback(annotate.disable_auto)
                self.auto_on = False
            cleanup.callback(self._release)
            # Stop admitting first, so no launch can join after the drain, then settle what was admitted
            # while the session still owns the hooks that record it.
            with self._launch_lock:
                self._admitting = False
            self._drain_launches()

    def __init__(
        self,
        *,
        gpus: Sequence[Device] = (),
        features: Feature = Feature.SPANS | Feature.ACTIVITY,
        activities: NativeActivity = NativeActivity.KERNEL,
        device_index: int = 0,
        sample_interval_ms: int = 50,
        max_spans: int = _MAX_SPANS,
        auto: Sequence[str] = (),
        native: bool = False,
    ) -> None:
        self.collection = Collection(
            features=features,
            activities=activities,
            device_index=device_index,
            sample_interval_ms=sample_interval_ms,
            max_spans=max_spans,
            auto=tuple(auto),
            native=native,
        )
        self.gpus = tuple(gpus)
        self.auto_on = False
        self.active = False
        self.collector = TraceCollector()
        self.sampler = DeviceSampler(interval_ms=self.collection.sample_interval_ms)
        self.spans = SpanRecorder(self.collection, tracer=Tracer(), snapshot=self.sampler.snapshot)
        self.launches = deque()
        self.dropped_launches = 0
        self._admitting = False
        # Queued CPU calls by (call, stream) identity; a None entry is a call past the storage bound.
        self._cpu_pending: defaultdict[tuple[int, int], deque[LaunchRecord | None]] = defaultdict(deque)
        self._cpu_queued = 0
        self._cpu_streams: dict[int, CPUStream] = {}
        self._metal_pending: deque[tuple[LaunchRecord, Metal.MTLCommandBuffer]] = deque()
        self._cuda_pending: deque[tuple[LaunchRecord, CUDAEvent, CUDAEvent]] = deque()
        self._launch_lock = RLock()

    def _armed(self, rollback: ExitStack, wanted: Feature) -> None:
        """Start the collector, auto-instrumentation and sampler `wanted` asks.

        Each is undone by `rollback` should a later one fail.
        """
        gpu = self.sampler.gpu
        if wanted & self.Feature.ACTIVITY and self._needs_native():
            kinds = self.collection.activities
            self.collector = rollback.enter_context(self.spans.tracer.collect(kinds))
        if self.collection.auto:
            self._auto(self.collection.auto)
            rollback.callback(annotate.disable_auto)
        if wanted & self.Feature.DEVICE and gpu is not None:
            self.sampler.start(self.spans)
            rollback.callback(self.sampler.stop)

    def _needs_native(self) -> bool:
        return self.collection.native or self.collection.activities != NativeActivity.KERNEL

    def _auto(self, modules: Sequence[str]) -> None:
        """Enable local events for owned function bodies, leaving explicit spans active.

        Explicit annotations and automatic spans can nest, even with the same label. Named
        generators and coroutines emit active resume segments, not invocation counts.
        """
        annotate.enable_auto(module_codes(modules))
        self.auto_on = True

    def _capturable(self, activities: NativeActivity) -> None:
        """Refuse a capture of `activities` this session does not collect on this device."""
        if not self.active or not self.collection.features & Feature.ACTIVITY:
            raise RuntimeError("the active Profiler did not request native activity collection")
        if activities & ~self.collection.activities:
            raise RuntimeError("the active Profiler did not enable the requested activity kinds")
        if type(self.collector) is not TraceCollector and self.collector.device_index != self.collection.device_index:
            raise RuntimeError("selected device differs from the actual current CUDA device")

    def _evidence(self, *, observed: bool) -> DeviceEvidence:
        """Say whether device evidence came back, or was never asked for in the first place."""
        if observed:
            return DeviceEvidence.COLLECTED
        wanted = self.collection.features & (self.Feature.DEVICE | self.Feature.ACTIVITY)
        return DeviceEvidence.ABSENT if wanted else DeviceEvidence.UNSOUGHT

    def _retrieve(
        self, *, since: int, until: int, activities: NativeActivity
    ) -> tuple[tuple[KernelTrace, ...], tuple[MemcpyTrace, ...]]:
        """Return the interval's kernels and copies, keeping only the kinds `activities` names."""
        kernels = tuple(self.collector.kernels(since=since, until=until))
        memcpys = tuple(self.collector.memcpys(since=since, until=until))
        return (
            kernels if activities & NativeActivity.KERNEL else (),
            memcpys if activities & NativeActivity.MEMCPY else (),
        )

    def _run_checkpointed[Answer](
        self, work: Callable[[], Answer], activities: NativeActivity
    ) -> tuple[Answer, int, int]:
        """Run `work`, then checkpoint the shared stream even when it raises."""
        try:
            answer = work()
        finally:
            until, loss_after = self.collector.checkpoint(activities)
        return answer, until, loss_after

    def _selected(self) -> Device | None:
        """Select exactly the requested visible device; never substitute another card."""
        if not self.gpus:
            return None
        index = self.collection.device_index
        try:
            return self.gpus[index]
        except IndexError:
            raise ValueError(f"device_index {index} is outside the {len(self.gpus)} visible devices") from None

    def _stop_collector(self) -> None:
        """Stop the native collector, when this session collected device activity."""
        self.collector.stop()

    def _cursor(self) -> int:
        """How many launches were ever recorded; retention moves the oldest index, never this one."""
        return self.dropped_launches + len(self.launches)

    def _window(self, first: int, end: int) -> tuple[LaunchRecord, ...]:
        """The launches numbered in [first, end), refused when retention or a reset already passed one."""
        if not self.dropped_launches <= first <= end <= self._cursor():
            raise RuntimeError(
                f"launch window [{first}, {end}) is outside the retained launches: it was reset, "
                "overwritten, or not yet recorded; increase max_spans"
            )
        with self._launch_lock:
            return tuple(islice(self.launches, first - self.dropped_launches, end - self.dropped_launches))

    def capture_window[Answer](self, work: Callable[[], Answer], activities: NativeActivity) -> tuple[Answer, Profile]:
        """Checkpoint the shared stream, run `work` once, then retrieve exactly that interval.

        The profiler must be entered; `Profiler.capture` opens one when none is active.
        """
        self._capturable(activities)
        if type(self.collector) is TraceCollector:
            self._drain_launches()
            first = self._cursor()
            try:
                answer = work()
            finally:
                self._drain_launches()
            return answer, self._with_launches(Profile(), self._window(first, self._cursor()))
        since, loss_before = self.collector.checkpoint(activities)
        self._drain_launches()
        first_launch = self._cursor()
        start_ns = self.spans.tracer.timestamp()
        began = time.perf_counter_ns()
        answer, until, loss_after = self._run_checkpointed(work, activities)
        wall_ns = time.perf_counter_ns() - began
        end_ns = self.spans.tracer.timestamp()
        if loss_after != loss_before:
            raise RuntimeError(f"native activity window lost {loss_after - loss_before} records")
        kernels, memcpys = self._retrieve(since=since, until=until, activities=activities)
        self._drain_launches()
        return answer, self._with_launches(
            Profile(
                device=self.sampler.gpu.name if self.sampler.gpu is not None else "",
                device_evidence=DeviceEvidence.COLLECTED if kernels or memcpys else DeviceEvidence.ABSENT,
                kernels=kernels,
                memcpys=memcpys,
                windows=(
                    RegionWindow(
                        name=getattr(work, "__qualname__", type(work).__qualname__),
                        start_ns=start_ns,
                        end_ns=end_ns,
                        wall_ns=wall_ns,
                        clock=self.spans.tracer.clock(),
                    ),
                ),
            ),
            self._window(first_launch, self._cursor()),
        )

    def enter(self, name: str) -> int:
        """Open one span in `spans`, as every active `span` asks through this session."""
        return self.spans.enter(name)

    def exit(self, token: int, *, wall_ns: int) -> None:
        """Close the span `token` names in `spans`."""
        self.spans.exit(token, wall_ns=wall_ns)

    def result(self) -> Profile:
        """Freeze the evidence collected so far into one `Profile`."""
        if self.active:
            self.collector.flush()
            self._drain_launches()
        kernels = tuple(self.collector.kernels())
        memcpys = tuple(self.collector.memcpys())
        activities = tuple(self.collector.activities())
        used_gpu = self.sampler.evidence or bool(kernels or memcpys or activities)
        with self._launch_lock:
            recorded = tuple(self.launches)
        return self._with_launches(
            Profile(
                device=self.sampler.label
                if self.sampler.evidence
                else (self.sampler.gpu.name if used_gpu and self.sampler.gpu is not None else ""),
                device_evidence=self._evidence(observed=used_gpu),
                summaries=tuple(
                    RegionSummary.from_snaps(item.name, item.wall_ms, item.samples) for item in self.spans.measurements
                ),
                windows=tuple(self.spans.windows),
                kernels=kernels,
                memcpys=memcpys,
                activities=activities,
                dropped_spans=self.spans.dropped,
                dropped_activities=self.collector.dropped(),
            ),
            recorded,
        )

    def cpu_enqueue(self, calls: Sequence[Call], stream: CPUStream) -> None:
        """Note each call queued while admitting, keeping its enqueue record while under `max_spans`.

        Calls past the bound are still tracked for draining, and their records are made at execution.
        """
        with self._launch_lock:
            if not self._admitting:
                return
            self._cpu_streams[id(stream)] = stream
            for task in calls:
                if self._cpu_queued < self.collection.max_spans:
                    self._cpu_queued += 1
                    record = self._record(task.program.kernel, _geometry(task), stream)
                else:
                    record = None
                self._cpu_pending[id(task), id(stream)].append(record)

    def cuda_graph(self, target: str) -> None:
        raise TargetError(
            "profile launch records",
            target,
            "CUDA graph replay hides individual node dispatches from the runtime hook",
            "configure run.graphs=False for per-launch profiling",
        )

    def cpu_run(self, call: Call, stream: CPUStream) -> None:
        """Execute `call`, recording it when it was queued under this session or starts while admitting."""
        key = id(call), id(stream)
        with self._launch_lock:
            queue = self._cpu_pending.get(key)
            tracked = queue is not None
            record = queue.popleft() if queue else None
            if queue is not None and not queue:
                del self._cpu_pending[key]
            if record is not None:
                self._cpu_queued -= 1
        if not tracked and not self._admitting:
            call.program.function(*call.arguments)
            return
        if record is None:
            record = self._record(call.program.kernel, _geometry(call), stream)
        start = time.perf_counter_ns()
        call.program.function(*call.arguments)
        end = time.perf_counter_ns()
        self._append_launch(replace(record, start_ns=start, end_ns=end, timing="cpu-wall"))

    def cuda_run(self, call: CUDACall, stream: CUDAStream) -> None:
        if not self._admitting:
            call.program.launch(stream, call.grid, call.threads, call.packed)
            return
        self._admit_native()
        record = self._record(call.program.ir, (*call.grid, call.threads), stream)
        before = stream.record()
        call.program.launch(stream, call.grid, call.threads, call.packed)
        with self._launch_lock:
            self._cuda_pending.append((record, before, stream.record()))

    def metal_enqueue(self, call: MetalCall, stream: MetalStream, command: Metal.MTLCommandBuffer) -> None:
        if not self._admitting:
            return
        self._admit_native()
        with self._launch_lock:
            self._metal_pending.append((self._record(call.program.ir, call.geometry, stream), command))

    def _record(
        self, ir: Kernel, geometry: tuple[int, int, int, int], stream: CPUStream | CUDAStream | MetalStream
    ) -> LaunchRecord:
        """Make one record stamped with this stream's device, and the span path open on this thread."""
        return launches.record(
            ir,
            geometry,
            stream.device.target.arch,
            id(stream),
            device_key=id(stream.device),
            region=self.spans.path(),
        )

    def _admit_native(self) -> None:
        """Settle the outstanding CUDA and Metal records once they reach `max_spans`, keeping memory bounded.

        Settling waits for the device, so the bound costs a synchronization; no record is discarded.
        """
        with self._launch_lock:
            full = len(self._cuda_pending) + len(self._metal_pending) >= self.collection.max_spans
        if full:
            self._settle_native()

    def _next[Pending](self, pending: deque[Pending]) -> Pending | None:
        """Remove the oldest pending entry, None when none is left."""
        with self._launch_lock:
            return pending.popleft() if pending else None

    def _settle_native(self) -> None:
        """Wait for and record each outstanding CUDA event pair and Metal command buffer, oldest first.

        An entry leaves its queue before it is waited on, so a failure keeps the rest queued for `_release`.
        """
        while (pending := self._next(self._cuda_pending)) is not None:
            record, before, after = pending
            after.wait()
            self._append_launch(
                replace(record, elapsed_ns=int(after.elapsed_since(before) * 1e6), timing="cuda-events")
            )
        while (pending := self._next(self._metal_pending)) is not None:
            record, command = pending
            command.waitUntilCompleted()
            if command.error() is not None:
                raise RuntimeError(str(command.error()))
            # PyObjC exposes this selector dynamically; the bundled Metal stub omits it.
            start_selector = "GPUStartTime"
            self._append_launch(
                replace(
                    record,
                    start_ns=int(getattr(command, start_selector)() * 1e9),
                    end_ns=int(command.GPUEndTime() * 1e9),
                    timing="metal-gpu",
                )
            )

    def _drain_launches(self) -> None:
        """Settle every admitted launch: wait for each CPU stream, then for the native records.

        A stream leaves the drain set only once none of its calls remain queued.
        """
        with self._launch_lock:
            streams = tuple(self._cpu_streams.values())
        for stream in streams:
            stream.record().wait()
        self._settle_native()
        with self._launch_lock:
            busy = {stream_id for _, stream_id in self._cpu_pending}
            for stream_id in [key for key in self._cpu_streams if key not in busy]:
                del self._cpu_streams[stream_id]

    def _release(self) -> None:
        """Drop every unsettled native reference, waiting on each one so no GPU work outlives the session."""
        with ExitStack() as cleanup:
            cleanup.callback(self._forget)
            for _, command in tuple(self._metal_pending):
                cleanup.callback(command.waitUntilCompleted)
            for _, _, after in tuple(self._cuda_pending):
                cleanup.callback(after.wait)

    def _forget(self) -> None:
        """Clear every queue and stream reference, under the lock every hook takes."""
        with self._launch_lock:
            self._cpu_pending.clear()
            self._cpu_streams.clear()
            self._cpu_queued = 0
            self._metal_pending.clear()
            self._cuda_pending.clear()

    def _append_launch(self, record: LaunchRecord) -> None:
        """Retain one completed launch, the oldest going when `max_spans` is full and counted as dropped."""
        with self._launch_lock:
            if len(self.launches) == self.collection.max_spans:
                self.launches.popleft()
                self.dropped_launches += 1
            self.launches.append(record)

    def _with_launches(self, profile: Profile, records: Sequence[LaunchRecord]) -> Profile:
        kernels = profile.kernels or (
            tuple(_kernel_traces(records)) if self.collection.activities & NativeActivity.KERNEL else ()
        )
        gpu = self._attributed(records)
        return replace(
            profile,
            launches=tuple(records),
            kernels=kernels,
            device=profile.device or _device_label(records, gpu),
            device_evidence=self._evidence(
                observed=bool(records or kernels or profile.memcpys) or self.sampler.evidence
            ),
            peak=gpu.peak if gpu is not None else None,
            coherent=gpu.coherent if gpu is not None else None,
            arch_key=gpu.arch_key if gpu is not None else None,
            compute_units=gpu.limits.compute_units if gpu is not None else None,
            dropped_launches=self.dropped_launches,
        )

    def _attributed(self, records: Sequence[LaunchRecord]) -> Device | None:
        """The device the records ran on, or None when they span devices or no visible device can be named.

        Runtime records carry the identity of the runtime device object, not a visible ordinal, so a
        second card of the same architecture stays unnamed rather than being guessed.
        """
        if not records:
            return self.sampler.gpu
        targets = {record.target for record in records}
        if len(targets) != 1 or None in {record.device_key for record in records}:
            return None
        matching = [gpu for gpu in self.gpus if gpu.arch_key in targets]
        return matching[0] if len(matching) == 1 else None


def _geometry(call: Call) -> tuple[int, int, int, int]:
    x, y, z, threads = call.arguments[-4:]
    return int(x), int(y), int(z), int(threads)


def _kernel_traces(records: Sequence[LaunchRecord]) -> Iterator[KernelTrace]:
    """One kernel per launch timed at all: positioned when both timestamps exist, duration-only otherwise."""
    for record in records:
        grid = "x".join(map(str, record.geometry[:3]))
        block = f"{record.geometry[3]}x1x1"
        if record.start_ns is not None and record.end_ns is not None:
            yield KernelTrace(
                name=record.name,
                start_ns=record.start_ns,
                end_ns=record.end_ns,
                grid=grid,
                block=block,
                stream_id=record.stream,
                clock=_CLOCKS.get(record.timing, "unknown"),
                region=record.region,
            )
        elif record.elapsed_ns is not None:
            yield KernelTrace(
                name=record.name,
                end_ns=record.elapsed_ns,
                grid=grid,
                block=block,
                stream_id=record.stream,
                clock=UNPOSITIONED,
                region=record.region,
            )


def _device_label(records: Sequence[LaunchRecord], gpu: Device | None) -> str:
    """Name the device the evidence came from: a named card, the host for CPU work, or explicitly unknown."""
    if gpu is not None:
        return gpu.name
    if not records:
        return "cpu"
    try:
        hosts = [Target.of(target).vendor == "cpu" for target in {record.target for record in records}]
    except KeyError:
        return "unknown"
    return "cpu" if all(hosts) else "unknown"

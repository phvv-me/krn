import os
import threading
from typing import TYPE_CHECKING, ClassVar

from .._constants import MILLISECONDS_PER_SECOND
from .._logging import logger
from ..result.models import ProcessReading

if TYPE_CHECKING:
    from ...hw.devices.base import Device
    from .recorder import SpanRecorder


class DeviceSampler:
    """Read the selected device's telemetry of this process, polling while spans are open.

    gpu: the one device a session reads, None until the session selects it.
    evidence: whether any reading so far contained this process.
    label: the unit name of the last reading that did.
    thread: the polling worker, kept while it lives even after `stop` gave up waiting for it.
    """

    _JOIN_SECONDS: ClassVar[float] = 2.0

    evidence: bool
    gpu: Device | None
    interval_ms: int
    label: str
    stop_event: threading.Event
    thread: threading.Thread | None

    def __init__(self, *, interval_ms: int) -> None:
        self.gpu = None
        self.interval_ms = interval_ms
        self.evidence = False
        self.label = ""
        self.stop_event = threading.Event()
        self.thread = None

    def _poll(self, recorder: SpanRecorder) -> None:
        """Poll target-process device telemetry while at least one span is open."""
        interval = self.interval_ms / MILLISECONDS_PER_SECOND
        while not self.stop_event.wait(interval):
            self._sample_once(recorder)

    def _sample_once(self, recorder: SpanRecorder) -> None:
        """Attach one device reading of the target process to every span `recorder` has open."""
        with recorder.lock:
            frames = tuple(recorder.frames.values())
            name = frames[-1].path if frames else ""
        if not frames:
            return
        snapshot = self.snapshot(name)
        if snapshot is None:
            return
        with recorder.lock:
            for frame in frames:
                frame.samples.append(snapshot)

    def snapshot(self, name: str) -> ProcessReading | None:
        """Read one process-scoped device reading, only when it contains this process."""
        gpu = self.gpu
        if gpu is None:
            return None
        try:
            raw = gpu.snapshot()
            processes = gpu.processes()
        except OSError, RuntimeError:
            logger.warning("device sampler skipped a failed snapshot", exc_info=True)
            return None
        process = next((item for item in processes or () if item.pid == os.getpid()), None)
        if process is None:
            return None
        self.evidence = True
        self.label = gpu.name
        return ProcessReading(
            unit_name=gpu.name,
            memory_used_bytes=process.memory_bytes,
            gpu_util_pct=raw.utilization_percent,
            memory_util_pct=raw.memory_utilization_percent,
            power_w=raw.power_w,
            temperature_c=raw.temperature_c,
        )

    def start(self, recorder: SpanRecorder) -> None:
        """Poll in the background, each reading going to every span `recorder` has open.

        Refused while an earlier worker still runs, so two workers never poll one session's spans.
        """
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("the previous device sampler is still blocked in a telemetry call")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._poll, args=(recorder,), daemon=True, name="krn-profiler")
        self.thread.start()

    def stop(self) -> None:
        """Stop the polling thread, releasing it only once it has finished.

        A worker blocked in a telemetry call outlives the wait and stays referenced, so `start` refuses
        to run beside it.
        """
        self.stop_event.set()
        if self.thread is None:
            return
        self.thread.join(timeout=self._JOIN_SECONDS)
        if self.thread.is_alive():
            logger.warning("device sampler is still blocked in a telemetry call; keeping its worker")
            return
        self.thread = None

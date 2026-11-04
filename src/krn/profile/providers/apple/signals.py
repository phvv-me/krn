# The Apple timing and counter signals, each with the reason and the alternative when this device lacks it.

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto

from ....compiler.errors import TargetError
from .capabilities import Capabilities


class Signal(StrEnum):
    ENCODER_TIME = auto()
    DISPATCH_TIME = auto()
    UTILIZATION = auto()
    CAPTURE = auto()


@dataclass(frozen=True, kw_only=True)
class _Rule:
    feature: str
    provided: Callable[[Capabilities], bool]
    reason: Callable[[Capabilities], str]
    instead: str


_RULES: dict[Signal, _Rule] = {
    Signal.ENCODER_TIME: _Rule(
        feature="GPU time per encoder from counter samples",
        provided=lambda caps: caps.timestamps and caps.stage_sampling,
        reason=lambda caps: (
            f"the timestamp set sampled at stage boundaries is needed, and the device has counter sets "
            f"[{caps.names()}] sampled at [{', '.join(sorted(caps.sampling)) or 'no points'}]"
        ),
        instead="read GPUStartTime and GPUEndTime of the command buffer, which krn's runtime records per launch",
    ),
    Signal.DISPATCH_TIME: _Rule(
        feature="GPU time per dispatch from counter samples",
        provided=lambda caps: caps.dispatch_sampling,
        reason=lambda caps: (
            f"counters are sampled at [{', '.join(sorted(caps.sampling)) or 'no points'}] only, "
            "not at dispatch boundaries"
        ),
        instead="give each dispatch its own encoder and command buffer, then read GPUStartTime and GPUEndTime",
    ),
    Signal.UTILIZATION: _Rule(
        feature="ALU, memory and occupancy counters",
        provided=lambda caps: caps.statistics,
        reason=lambda caps: f"no statistic or stage-utilization counter set is exposed (counter sets: {caps.names()})",
        instead="compare the measured GPU time with the bytes and FLOPs krn counts per launch (the roofline), or "
        "record with `xcrun xctrace record --template 'Metal System Trace'`, whose default counter profile holds only "
        "RT Unit Active",
    ),
    Signal.CAPTURE: _Rule(
        feature="programmatic GPU capture to a .gputrace document",
        provided=lambda caps: caps.capture,
        reason=lambda caps: (
            "MTLCaptureManager refuses the GPU trace destination for this unbundled process, and "
            "MTL_CAPTURE_ENABLED=1 does not change that"
        ),
        instead="record with `xcrun xctrace record --template 'Metal System Trace' --launch -- <command>`, or run the "
        "workload in an app bundle whose Info.plist sets MetalCaptureEnabled",
    ),
}


def require(signal: Signal, caps: Capabilities) -> None:
    """Return when `caps` provides `signal`; otherwise raise a TargetError naming why and what to use instead."""
    rule = _RULES[signal]
    if not rule.provided(caps):
        raise TargetError(rule.feature, caps.chip, rule.reason(caps), rule.instead)

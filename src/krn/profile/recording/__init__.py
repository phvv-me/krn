"""The span recorder and device sampler the profiling facade composes, kept apart from the stable
collection policy in `session` so that what the CLI imports stays stable."""

from .recorder import SpanRecorder
from .sampler import DeviceSampler

__all__ = ["DeviceSampler", "SpanRecorder"]

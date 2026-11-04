# Register strategies without requiring their optional libraries to be installed.
from .amd import tracer as amd_tracer
from .apple import tracer as apple_tracer
from .cpu import tracer as cpu_tracer
from .intel import tracer as intel_tracer
from .nvidia import tracer as nvidia_tracer

__all__ = ["amd_tracer", "apple_tracer", "cpu_tracer", "intel_tracer", "nvidia_tracer"]

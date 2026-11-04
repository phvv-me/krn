"""What krn raises and warns with: capture and target errors, kernel errors rebuilt on the host, and the two warnings.

    from krn.errors import CaptureError, TargetError

A `TargetError` is a `CaptureError` that names the feature, the target, why the hardware lacks it and what to do
instead. `PrecisionWarning` and `PerformanceWarning` are `UserWarning`s, so `warnings.filterwarnings` silences or
escalates them.
"""

from krn.compiler.errors import CaptureError, PrecisionWarning, SettingError, TargetError
from krn.compiler.performance import PerformanceWarning
from krn.exceptions import KernelError

__all__ = [
    "CaptureError",
    "KernelError",
    "PerformanceWarning",
    "PrecisionWarning",
    "SettingError",
    "TargetError",
]

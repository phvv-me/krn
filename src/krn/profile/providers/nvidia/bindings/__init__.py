# The accelerator bindings, each present only where its package is installed. A binding that did
# not import leaves its name unbound and its `has_` flag False, so each use is gated on the flag.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Local stubs describe the optional wheels without requiring them on macOS.
    from cuda.bindings import driver

    from . import _cupti as cupti
    from . import _nvtx as nvtx
    from . import _runtime as runtime

    has_nvtx = has_cupti = has_cuda = True
else:
    try:
        import nvtx
    except ImportError:
        has_nvtx = False
    else:
        has_nvtx = True
    try:
        from cupti import cupti
    except ImportError:
        has_cupti = False
    else:
        has_cupti = True
    try:
        from cuda.bindings import driver, runtime
    except ImportError, OSError:
        has_cuda = False
    else:
        has_cuda = True

__all__ = ["cupti", "driver", "has_cuda", "has_cupti", "has_nvtx", "nvtx", "runtime"]

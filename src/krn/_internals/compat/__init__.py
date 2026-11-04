"""The compatibility matrix: which features compile on which targets, as data.

`krn._internals.compat.analyze.analyze` builds a matrix from probes; `dumps` and `loads` write and read its JSON.
"""

from .model import Evidence, Feature, Hardware, Matrix, Measurement, Status, Support, dumps, loads
from .requires import Requires

__all__ = [
    "Evidence",
    "Feature",
    "Hardware",
    "Matrix",
    "Measurement",
    "Requires",
    "Status",
    "Support",
    "dumps",
    "loads",
]

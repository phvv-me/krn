"""host: everything the Python program around the kernels uses, grouped so that it does not crowd the kernel language.

from krn import host

host.Device.current()     the device kernels launch on, found from the machine, never registered
host.Stream, host.Event   in-order queues and awaitable points in them
host.Graph                recorded launches that replay with one call
host.Settings             immutable grouped settings snapshots
host.CompileOptions       the C++ standard and flags of each backend
host.Ladder               precision ladders
host.System.detect()      machine inventory and compiler target
"""

from krn.frontend.pipeline import (
    Channel,
    Lane,
    Latest,
    Memory,
    Move,
    Moved,
    Place,
    Slot,
    Ticket,
    Transfer,
    join,
    route,
)
from krn.frontend.precision import Ladder, Resolution, Route
from krn.hw.system import System

from .api import (
    Backend,
    BackendUnavailable,
    Bound,
    Configured,
    Device,
    Event,
    Graph,
    Launcher,
    Stream,
    current_stream,
    default_device,
)
from .compile_options import CompileOptions, Dialect, Opt
from .settings import Policy, Settings

__all__ = [
    "Backend",
    "BackendUnavailable",
    "Bound",
    "Channel",
    "CompileOptions",
    "Configured",
    "Device",
    "Dialect",
    "Event",
    "Graph",
    "Ladder",
    "Lane",
    "Latest",
    "Launcher",
    "Memory",
    "Move",
    "Moved",
    "Opt",
    "Place",
    "Policy",
    "Resolution",
    "Route",
    "Settings",
    "Slot",
    "Stream",
    "System",
    "Ticket",
    "Transfer",
    "current_stream",
    "default_device",
    "join",
    "route",
]

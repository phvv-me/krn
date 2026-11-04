from collections.abc import Callable, Sequence

from ....formats.protocols import RawActivity

class ActivityKind:
    CONCURRENT_KERNEL: int
    MEMCPY: int
    MEMSET: int
    SYNCHRONIZATION: int
    OVERHEAD: int
    MEMORY: int
    JIT: int
    RUNTIME: int
    DRIVER: int
    MEMORY_POOL: int

class CallbackDomain:
    RUNTIME_API: int
    DRIVER_API: int

def activity_disable(kind: int) -> None: ...
def activity_enable(kind: int) -> None: ...
def activity_flush_all(flag: int) -> None: ...
def activity_get_num_dropped_records(context: int, stream: int, pointer: int) -> None: ...
def activity_register_callbacks(
    requested: Callable[[], tuple[int, int]], completed: Callable[[Sequence[RawActivity]], None]
) -> None: ...
def get_device_id(context: int) -> int: ...
def get_context_id(context: int) -> int: ...
def get_callback_name(domain: int, callback: int) -> str: ...
def get_timestamp() -> int: ...

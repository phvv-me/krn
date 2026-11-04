from collections.abc import Collection
from dataclasses import dataclass

SUPPORTED_NVIDIA_LIBNAMES: Collection[str]

class DynamicLibNotFoundError(RuntimeError): ...
class DynamicLibUnknownError(RuntimeError): ...
class DynamicLibNotAvailableError(RuntimeError): ...

@dataclass(frozen=True, kw_only=True)
class LoadedDL:
    abs_path: str
    was_already_loaded_from_elsewhere: bool

def load_nvidia_dynamic_lib(libname: str) -> LoadedDL: ...

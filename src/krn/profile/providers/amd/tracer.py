# AMD annotation backend: ROCTx ranges, visible under `rocprofv3 --marker-trace`. The module
# imports only where `roctx` does, and the providers package skips it otherwise.

from ctypes import c_char_p, c_uint64
from typing import ClassVar

from ....compiler.errors import TargetError
from ....hw.libraries import Libraries, LibraryNotFound, LoadedLibrary, SonameLoader
from ...enums import Activity
from ...tracer import Marker, Tracer, Vendor


class ROCTxLoader(SonameLoader):
    names = frozenset({"roctx"})

    def candidates(self, name: str) -> tuple[str, ...]:
        return ("libroctx64.so",)

    def load(self, name: str) -> LoadedLibrary:
        try:
            return super().load(name)
        except LibraryNotFound as error:
            raise LibraryNotFound(
                name,
                error.names,
                error.places,
                "Install ROCm's ROCTx library and expose its library directory to the loader",
            ) from error


class ROCTxTracer(Tracer):
    """ROCTx start/stop ranges."""

    vendor: ClassVar[Vendor] = Vendor.AMD
    label: ClassVar[str] = "roctx"

    @classmethod
    def is_available(cls) -> bool:
        return Libraries.available("roctx")

    def start(self, name: str) -> Marker:
        """Open a correlatable range and return its exact closer."""
        handle = Libraries.load("roctx").handle
        handle.roctxRangeStartA.argtypes = [c_char_p]
        handle.roctxRangeStartA.restype = c_uint64
        handle.roctxRangeStop.argtypes = [c_uint64]
        range_id = handle.roctxRangeStartA(name.encode())
        return lambda: handle.roctxRangeStop(range_id)

    def supported(self) -> Activity:
        raise TargetError(
            "rocprofiler-sdk activity",
            "AMD",
            "a tested SDK callback adapter is not implemented",
            "profile on an NVIDIA or Apple target, or disable ACTIVITY to keep host spans",
        )

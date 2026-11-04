"""Optional Apple private readers; failures have no reading."""

import ctypes
import math
import struct
import time
from contextlib import ExitStack

from .libraries import Libraries, LibraryHandle, LibraryNotFound


def registry_int(property_name: str, child: str | None = None) -> int | None:
    try:
        return _registry_int(property_name, child)
    except OSError, LibraryNotFound, AttributeError, ValueError, ctypes.ArgumentError:
        return None


def _registry_int(property_name: str, child: str | None) -> int | None:
    """Read the AGXAccelerator core count, releasing every owned CF handle."""
    io, cf = Libraries.load("iokit").handle, Libraries.load("corefoundation").handle
    io.IOServiceMatching.restype = ctypes.c_void_p
    io.IOServiceMatching.argtypes = [ctypes.c_char_p]
    io.IOServiceGetMatchingService.restype = ctypes.c_uint32
    io.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    io.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
    io.IORegistryEntryCreateCFProperty.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    io.IOObjectRelease.argtypes = [ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    cf.CFNumberGetValue.restype = ctypes.c_bool
    cf.CFGetTypeID.restype = ctypes.c_ulong
    cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
    cf.CFNumberGetTypeID.restype = ctypes.c_ulong
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    service = io.IOServiceGetMatchingService(0, io.IOServiceMatching(b"AGXAccelerator"))
    if not service:
        return None
    key = cf.CFStringCreateWithCString(None, property_name.encode(), 0x08000100)
    value = io.IORegistryEntryCreateCFProperty(service, key, None, 0)
    child_key = None
    try:
        number = value
        if child is not None and value:
            cf.CFDictionaryGetTypeID.restype = ctypes.c_ulong
            if cf.CFGetTypeID(value) != cf.CFDictionaryGetTypeID():
                return None
            cf.CFDictionaryGetValue.restype = ctypes.c_void_p
            cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            child_key = cf.CFStringCreateWithCString(None, child.encode(), 0x08000100)
            number = cf.CFDictionaryGetValue(value, child_key)
        if not number or cf.CFGetTypeID(number) != cf.CFNumberGetTypeID():
            return None
        result = ctypes.c_int64()
        return result.value if cf.CFNumberGetValue(number, 4, ctypes.byref(result)) else None
    finally:
        if child_key:
            cf.CFRelease(child_key)
        if value:
            cf.CFRelease(value)
        cf.CFRelease(key)
        io.IOObjectRelease(service)


def gpu_core_count() -> int | None:
    return registry_int("gpu-core-count")


def performance_cores() -> tuple[int | None, int | None]:
    def read(name: bytes) -> int | None:
        try:
            call = Libraries.load("libsystem").handle.sysctlbyname
            value, size = ctypes.c_uint64(), ctypes.c_size_t(8)
            if call(name, ctypes.byref(value), ctypes.byref(size), None, 0):
                return None
            return value.value
        except OSError, LibraryNotFound, AttributeError, ValueError:
            return None

    return read(b"hw.perflevel0.physicalcpu"), read(b"hw.perflevel1.physicalcpu")


class SMCKeyInfo(ctypes.Structure):
    _fields_ = [("data_size", ctypes.c_uint32), ("data_type", ctypes.c_uint32), ("attributes", ctypes.c_uint8)]


class SMCKeyData(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_uint32),
        ("version", ctypes.c_uint8 * 6),
        ("limit", ctypes.c_uint8 * 16),
        ("info", SMCKeyInfo),
        ("result", ctypes.c_uint8),
        ("status", ctypes.c_uint8),
        ("command", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("data", ctypes.c_uint8 * 32),
    ]


def gpu_temperature() -> float | None:
    """Read GPU-attributed SMC float keys; system fans have no GPU attribution."""
    try:
        return _gpu_temperature()
    except OSError, LibraryNotFound, AttributeError, ValueError, ctypes.ArgumentError:
        return None


def _gpu_temperature() -> float | None:
    io, libc = Libraries.load("iokit").handle, Libraries.load("libsystem").handle
    io.IOServiceMatching.restype = ctypes.c_void_p
    io.IOServiceMatching.argtypes = [ctypes.c_char_p]
    io.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    io.IOServiceGetMatchingService.restype = ctypes.c_uint32
    io.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    io.IOConnectCallStructMethod.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    io.IOObjectRelease.argtypes = [ctypes.c_uint32]
    io.IOServiceClose.argtypes = [ctypes.c_uint32]
    libc.mach_task_self.restype = ctypes.c_uint32
    service = io.IOServiceGetMatchingService(0, io.IOServiceMatching(b"AppleSMC"))
    if not service:
        return None
    connection = ctypes.c_uint32()
    try:
        if io.IOServiceOpen(service, libc.mach_task_self(), 0, ctypes.byref(connection)):
            return None
        try:
            values = [_smc_float(io, connection.value, key) for key in ("Tg04", "Tg05", "Tg0K", "Tg0L", "Tg0R", "Tg0S")]
            return max((v for v in values if v is not None and 0 < v < 150), default=None)
        finally:
            io.IOServiceClose(connection.value)
    finally:
        io.IOObjectRelease(service)


def _smc_float(io: LibraryHandle, connection: int, key: str) -> float | None:
    request, response = SMCKeyData(), SMCKeyData()
    request.key, request.command = int.from_bytes(key.encode(), "big"), 9
    size = ctypes.c_size_t(ctypes.sizeof(response))
    call = io.IOConnectCallStructMethod
    if call(connection, 2, ctypes.byref(request), ctypes.sizeof(request), ctypes.byref(response), ctypes.byref(size)):
        return None
    if response.result or response.info.data_size != 4 or response.info.data_type != int.from_bytes(b"flt ", "big"):
        return None
    request.info, request.command = response.info, 5
    if call(connection, 2, ctypes.byref(request), ctypes.sizeof(request), ctypes.byref(response), ctypes.byref(size)):
        return None
    return None if response.result else struct.unpack("<f", bytes(response.data[:4]))[0]


def gpu_power(*, interval: float) -> float | None:
    """Watts from two IOReport GPU Energy samples, using the measured elapsed time."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be positive and finite")
    try:
        return _gpu_power(interval)
    except OSError, LibraryNotFound, AttributeError, ValueError, ctypes.ArgumentError:
        return None


def _gpu_power(interval: float) -> float | None:
    cf, report = Libraries.load("corefoundation").handle, Libraries.load("ioreport").handle
    pointer = ctypes.c_void_p
    cf.CFRelease.argtypes = [pointer]
    cf.CFStringCreateWithCString.argtypes = [pointer, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = pointer
    cf.CFStringGetCString.argtypes = [pointer, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFDictionaryGetValue.argtypes = [pointer, pointer]
    cf.CFDictionaryGetValue.restype = pointer
    cf.CFArrayGetCount.argtypes = [pointer]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetValueAtIndex.argtypes = [pointer, ctypes.c_long]
    cf.CFArrayGetValueAtIndex.restype = pointer
    report.IOReportCopyAllChannels.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    report.IOReportCopyAllChannels.restype = pointer
    report.IOReportCreateSubscription.argtypes = [pointer, pointer, ctypes.POINTER(pointer), ctypes.c_uint64, pointer]
    report.IOReportCreateSubscription.restype = pointer
    report.IOReportCreateSamples.argtypes = [pointer, pointer, pointer]
    report.IOReportCreateSamples.restype = pointer
    report.IOReportCreateSamplesDelta.argtypes = [pointer, pointer, pointer]
    report.IOReportCreateSamplesDelta.restype = pointer
    for name in ("IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel"):
        call = getattr(report, name)
        call.argtypes, call.restype = [pointer], pointer
    report.IOReportSimpleGetIntegerValue.argtypes = [pointer, ctypes.c_int32]
    report.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
    with ExitStack() as owned:

        def keep(handle: int) -> int:
            if not handle:
                raise ValueError("IOReport returned no handle")
            owned.callback(cf.CFRelease, handle)
            return handle

        channels = keep(report.IOReportCopyAllChannels(0, 0))
        subscribed = pointer()
        subscription = keep(report.IOReportCreateSubscription(None, channels, ctypes.byref(subscribed), 0, None))
        if subscribed.value:
            keep(subscribed.value)
        first = keep(report.IOReportCreateSamples(subscription, channels, None))
        started = time.monotonic()
        time.sleep(interval)
        second = keep(report.IOReportCreateSamples(subscription, channels, None))
        elapsed = time.monotonic() - started
        delta = keep(report.IOReportCreateSamplesDelta(first, second, None))
        key = keep(cf.CFStringCreateWithCString(None, b"IOReportChannels", 0x08000100))
        array = cf.CFDictionaryGetValue(delta, key)
        if not array:
            return None
        for index in range(cf.CFArrayGetCount(array)):
            channel = cf.CFArrayGetValueAtIndex(array, index)
            if _cf_string(cf, report.IOReportChannelGetChannelName(channel)) == "GPU Energy":
                unit = _cf_string(cf, report.IOReportChannelGetUnitLabel(channel))
                scale = {"nJ": 1e-9, "uJ": 1e-6, "mJ": 1e-3, "J": 1.0}.get(unit)
                energy = report.IOReportSimpleGetIntegerValue(channel, 0)
                return energy * scale / elapsed if scale is not None and elapsed > 0 and energy >= 0 else None
        return None


def _cf_string(cf: LibraryHandle, handle: int) -> str:
    buffer = ctypes.create_string_buffer(512)
    return buffer.value.decode() if handle and cf.CFStringGetCString(handle, buffer, 512, 0x08000100) else ""

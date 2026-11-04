import ctypes
import errno
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from krn.compiler.errors import TargetError
from krn.profile.providers.cpu import perf


class FakeLibc:
    """A C library whose syscall answers with the descriptor (or -1 with errno) it is given."""

    def __init__(self, answer: int, number: int = errno.EACCES, ioctl: int = 0) -> None:
        self.answer, self.number, self.ioctl_result = answer, number, ioctl
        self.calls: list[tuple[object, ...]] = []
        self.errno = ctypes.c_int(0)
        self.__dict__[perf.ERRNO_LOCATION] = lambda: ctypes.pointer(self.errno)  # a dunder method would be mangled

    def syscall(self, *args: object) -> int:
        self.calls.append(args)
        if self.answer < 0:
            self.errno.value = self.number
        return self.answer

    def ioctl(self, fd: int, request: int, value: int) -> int:
        self.calls.append((fd, request))
        return self.ioctl_result


def sample(tmp_path: Path, value: int, enabled: int, running: int) -> int:
    path = tmp_path / f"counter-{value}"
    path.write_bytes(struct.pack("<3Q", value, enabled, running))
    return os.open(path, os.O_RDONLY)


def pmu_tree(root: Path, **types: str) -> Path:
    for name, text in types.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "type").write_text(text)
    return root


def test_attr_matches_the_kernel_layout() -> None:
    assert ctypes.sizeof(perf._Attr) == perf.ATTR_SIZE == 112
    assert perf._Attr.config.offset == 8 and perf._Attr.flags.offset == 40
    assert perf._Attr.clockid.offset == 92 and perf._Attr.reserved.offset == 110


def test_reading_extrapolates_only_multiplexed_counters() -> None:
    assert perf.Reading(value=10, enabled=100, running=100).scaled == 10.0
    assert perf.Reading(value=10, enabled=100, running=50).scaled == 20.0
    assert perf.Reading(value=0, enabled=100, running=0).scaled == 0.0


def test_core_pmus_and_types_come_from_sysfs(tmp_path: Path) -> None:
    hybrid = pmu_tree(tmp_path, cpu_core="4\n", cpu_atom="10\n", i915="14")
    assert perf.core_pmus(hybrid) == {"cpu_core": 4, "cpu_atom": 10}
    assert perf.pmu_type("i915", hybrid) == 14
    assert perf.pmu_type("absent", hybrid) is None
    (hybrid / "cpu_core" / "type").write_text("not a number")
    assert perf.pmu_type("cpu_core", hybrid) is None
    assert perf.core_pmus(tmp_path / "missing") == {}
    assert perf.core_pmus(pmu_tree(tmp_path / "plain", cpu="8")) == {"cpu": 8}


def test_paranoid_level_is_read_or_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    level = tmp_path / "paranoid"
    monkeypatch.setattr(perf, "PARANOID", level)
    assert perf.paranoid() is None
    level.write_text("2\n")
    assert perf.paranoid() == 2
    level.write_text("two")
    assert perf.paranoid() is None


@pytest.mark.parametrize(
    ("number", "system_wide", "remedy"),
    [
        (errno.EACCES, True, "CAP_PERFMON to the Python binary"),
        (errno.EPERM, False, "2 or below"),
        (errno.ENOENT, False, "generic event"),
        (errno.EINVAL, False, "encoding"),
    ],
)
def test_refusals_name_the_remedy_for_their_scope(number: int, system_wide: bool, remedy: str) -> None:
    error = perf.refusal(number, "cycles counter", system_wide=system_wide)
    assert isinstance(error, TargetError) and error.feature == "cycles counter"
    assert remedy in error.instead


def test_open_raw_refuses_outside_linux() -> None:
    with pytest.raises(TargetError, match="Linux interface"):
        perf.open_raw(0, 0, pid=0, cpu=-1, feature="cycles", system="Darwin")


def test_open_linux_returns_the_descriptor_and_refuses_with_errno(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fd = sample(tmp_path, 1, 1, 1)
    libc = FakeLibc(fd)
    monkeypatch.setattr(perf, "_libc", lambda: libc)
    monkeypatch.setattr(perf.platform, "machine", lambda: "x86_64")
    assert perf.open_raw(4, 0x400000, pid=0, cpu=-1, feature="cycles", system="Linux") == fd
    number, _, pid, cpu, group, flags = libc.calls[0]
    assert (number, pid, cpu, group) == (298, 0, -1, -1) and flags == perf.FLAG_FD_CLOEXEC
    os.close(fd)
    refused = FakeLibc(-1, errno.EACCES)
    monkeypatch.setattr(perf, "_libc", lambda: refused)
    with pytest.raises(TargetError, match="system-wide perf_event_open was refused"):
        perf.open_raw(14, 0, pid=-1, cpu=0, feature="i915 rcs0-busy", system="Linux")
    monkeypatch.setattr(perf.platform, "machine", lambda: "riscv64")
    with pytest.raises(TargetError, match="riscv64"):
        perf.open_raw(0, 0, pid=0, cpu=-1, feature="cycles", system="Linux")


def test_counter_reads_its_descriptor_and_reports_failed_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fd = sample(tmp_path, 7, 20, 10)
    counter = perf.Counter(fd=fd, pmu="cpu_core", event="cycles")
    assert counter.read() == perf.Reading(value=7, enabled=20, running=10)
    libc = FakeLibc(0, ioctl=0)
    monkeypatch.setattr(perf, "_libc", lambda: libc)
    counter.control(perf.IOC_ENABLE)
    monkeypatch.setattr(perf, "_libc", lambda: FakeLibc(0, ioctl=-1))
    with pytest.raises(OSError):
        counter.control(perf.IOC_ENABLE)
    counter.close()


def test_open_hardware_names_the_core_pmu_only_on_hybrid_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def fake_open(kind: int, config: int, **keywords: object) -> int:
        seen.append({"kind": kind, "config": config, **keywords})
        return 99

    monkeypatch.setattr(perf, "open_raw", fake_open)
    hybrid = perf.open_hardware("instructions", "cpu_atom", 10, hybrid=True, pid=0, cpu=-1)
    plain = perf.open_hardware("instructions", "cpu", 10, hybrid=False, pid=0, cpu=-1)
    assert seen[0]["config"] == 1 | (10 << 32) and seen[0]["kind"] == perf.PERF_TYPE_HARDWARE
    assert seen[1]["config"] == 1 and hybrid.pmu == "cpu_atom" and plain.event == "instructions"


def test_hardware_counters_sum_pmus_and_release_on_refusal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown generic counter events"):
        perf.HardwareCounters(["cycles", "joules"])
    with pytest.raises(TargetError, match="no CPU core PMU"), perf.HardwareCounters(pmus={}):
        pass

    opened: list[perf.Counter] = []
    counts = {("cycles", "cpu_core"): 30, ("cycles", "cpu_atom"): 5, ("instructions", "cpu_core"): 60}

    def fake_open(event: str, pmu: str, kind: int, *, hybrid: bool, pid: int, cpu: int) -> perf.Counter:
        counter = perf.Counter(fd=sample(tmp_path, counts.get((event, pmu), 0), 4, 4), pmu=pmu, event=event)
        opened.append(counter)
        return counter

    monkeypatch.setattr(perf, "open_hardware", fake_open)
    monkeypatch.setattr(perf, "_libc", lambda: FakeLibc(0, ioctl=0))
    with perf.HardwareCounters(["cycles", "instructions"], pmus={"cpu_core": 4, "cpu_atom": 10}) as counters:
        totals = counters.read()
    assert totals == {"cycles": 35.0, "instructions": 60.0}
    assert perf.HardwareCounters.metrics(totals) == {"ipc": 60 / 35}
    assert perf.HardwareCounters.metrics({"cycles": 0.0, "branch_misses": 1.0}) == {}
    assert {c.pmu for c in opened} == {"cpu_core", "cpu_atom"}

    closed: list[int] = []
    monkeypatch.setattr(perf.Counter, "close", lambda self: closed.append(self.fd))
    failing = {"branches": True}

    def refusing_open(event: str, pmu: str, kind: int, *, hybrid: bool, pid: int, cpu: int) -> perf.Counter:
        if failing.get(event) and pmu == "cpu_atom":
            raise TargetError("branches counter", "CPU", "refused", "lower paranoid")
        return perf.Counter(fd=sample(tmp_path, 0, 1, 1), pmu=pmu, event=event)

    monkeypatch.setattr(perf, "open_hardware", refusing_open)
    with (
        pytest.raises(TargetError),
        perf.HardwareCounters(["cycles", "branches"], pmus={"cpu_core": 4, "cpu_atom": 10}),
    ):
        pass
    assert len(closed) == 3


def test_exit_disables_and_releases_even_when_disable_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        perf,
        "open_hardware",
        lambda event, pmu, kind, *, hybrid, pid, cpu: perf.Counter(fd=sample(tmp_path, 1, 1, 1), pmu=pmu, event=event),
    )
    monkeypatch.setattr(perf, "_libc", lambda: FakeLibc(0, ioctl=-1))
    counters = perf.HardwareCounters(["cycles"], pmus={"cpu": 8})
    with pytest.raises(OSError), counters:
        pass
    assert counters._counters == []


@pytest.mark.hardware
@pytest.mark.skipif(sys.platform != "linux", reason="perf_event_open is a Linux interface")
def test_real_counters_see_a_loop_on_this_thread() -> None:
    total = 0
    with perf.HardwareCounters(["cycles", "instructions"]) as counters:
        for i in range(2_000_000):
            total += i
        counts = counters.read()
    assert total > 0
    assert counts["instructions"] > 2_000_000
    assert perf.HardwareCounters.metrics(counts)["ipc"] > 0


def test_libc_comes_from_the_loader_with_errno_location(monkeypatch: pytest.MonkeyPatch) -> None:
    library = "/usr/lib/libSystem.B.dylib" if sys.platform == "darwin" else "libc.so.6"
    monkeypatch.setattr(perf.Libraries, "load", lambda name: SimpleNamespace(handle=ctypes.CDLL(library)))
    handle = perf._libc.__wrapped__()
    assert handle.syscall.restype is ctypes.c_long and handle.ioctl.restype is ctypes.c_int
    assert isinstance(perf._errno(), int)

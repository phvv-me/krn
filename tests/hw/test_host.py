from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from krn.hw import host
from krn.hw.host import CPU, OS, Limits, Memory, PythonRuntime, Volume


def test_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host.platform, "freedesktop_os_release", lambda: {"NAME": "Test", "VERSION_ID": "2"})
    assert OS.detect().distribution == "Test"
    monkeypatch.setattr(host.platform, "freedesktop_os_release", lambda: {})
    assert OS.detect().distribution == "Linux"
    monkeypatch.setattr(host.platform, "freedesktop_os_release", Mock(side_effect=OSError))
    assert OS.detect().distribution == "Linux"
    monkeypatch.setattr(
        host,
        "archspec",
        SimpleNamespace(
            host=lambda: SimpleNamespace(
                name="x86_64_v3",
                vendor="GenuineIntel",
                features={"avx2"},
                generation=3,
            )
        ),
    )
    monkeypatch.setattr(
        host,
        "psutil",
        SimpleNamespace(
            cpu_count=lambda logical: 8 if logical else 4,
            cpu_freq=lambda: SimpleNamespace(max=4000),
        ),
    )
    monkeypatch.setattr(host.native, "cpu_caches", lambda: (("l2-unified", 1024),))
    cpu = CPU.detect()
    assert cpu.physical_cores == 4 and cpu.max_mhz == 4000 and cpu.features == frozenset({"avx2"})
    assert cpu.microarchitecture == "x86_64_v3" and cpu.generation == 3
    monkeypatch.setattr(host.archspec, "host", Mock(side_effect=OSError))
    assert not CPU.detect().features
    monkeypatch.setattr(host, "archspec", None)
    monkeypatch.setattr(host.psutil, "cpu_freq", lambda: None)
    assert CPU.detect().max_mhz is None
    monkeypatch.setattr(host.psutil, "cpu_freq", Mock(side_effect=OSError))
    assert CPU.detect().max_mhz is None
    monkeypatch.setattr(host, "psutil", None)
    assert CPU.detect().physical_cores is None


def test_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host.platform, "system", lambda: "Other")
    with pytest.raises(ValueError, match="unknown system"):
        OS.detect()
    monkeypatch.setattr(host.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(host.platform, "mac_ver", lambda: ("27", ("", "", ""), "arm64"))
    assert OS.detect().distribution_version == "27"
    monkeypatch.setattr(host.apple_native, "performance_cores", lambda: (8, 4))
    monkeypatch.setattr(
        host.AppleProbe, "discover", lambda self: (SimpleNamespace(name="Apple M4", vendor=host.Vendor.APPLE),)
    )
    cpu = CPU.detect()
    assert cpu.name == "Apple M4" and cpu.performance_cores == 8 and cpu.efficiency_cores == 4
    assert cpu.max_mhz is None and not cpu.features
    monkeypatch.setattr(host.AppleProbe, "discover", lambda self: ())
    assert host.cpu_identity(system="Darwin")[1] == host.Vendor.UNKNOWN
    assert host.core_split() == (None, None) and host.caches() == ()


def test_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host.native, "cgroup_limits", lambda: (0.5, 1024))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert host.psutil is not None
    monkeypatch.setattr(host.psutil.Process, "cpu_affinity", lambda self: [1, 2], raising=False)
    limits = Limits.detect()
    assert limits.usable_cpus == 0.5 and limits.cpu_affinity == frozenset({1, 2})
    assert Memory.detect(limits).cgroup_limit == 1024
    monkeypatch.setattr(host.native, "cgroup_limits", lambda: (None, None))
    monkeypatch.setattr(host.psutil.Process, "cpu_affinity", Mock(side_effect=OSError))
    assert Limits.detect().cpu_affinity is None
    assert Memory.detect(Limits.detect()).cgroup_limit is None
    monkeypatch.setattr(host.psutil, "swap_memory", Mock(side_effect=NotImplementedError))
    assert Memory.detect(Limits.detect()).swap_used is None
    assert host.cgroup_limits() == (None, None)
    monkeypatch.setattr(host, "psutil", None)
    assert host.cpu_affinity() is None and host.frequency(system="Linux") is None
    assert Memory.detect(limits).total is None
    assert PythonRuntime.detect().version.startswith("3.14")


def test_volumes(monkeypatch: pytest.MonkeyPatch) -> None:
    def usage(path: str) -> SimpleNamespace:
        if path == "/absent":
            raise OSError
        return SimpleNamespace(total=100, free=60, used=40)

    monkeypatch.setattr(
        host,
        "psutil",
        SimpleNamespace(
            disk_partitions=lambda: [SimpleNamespace(mountpoint=p) for p in ("/", "/data", "/absent")],
            disk_usage=usage,
        ),
    )
    assert len(Volume.detect_all()) == 2
    assert Volume.detect().used_fraction == 0.4
    monkeypatch.setattr(host.psutil, "disk_usage", lambda path: SimpleNamespace(total=0, free=0, used=0))
    assert Volume.detect().used_fraction == 0
    monkeypatch.setattr(host, "psutil", None)
    assert Volume.detect_all() == ()
    with pytest.raises(OSError, match="psutil"):
        Volume.detect()

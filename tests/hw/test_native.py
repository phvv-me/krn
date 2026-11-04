from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from krn.hw import apple_native, native


def test_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "value"
    assert native.read_text(path) is None and native.read_int(path) is None
    for text, expected in (("42", 42), ("0x10de", 4318), ("max", None)):
        path.write_text(text)
        assert native.read_int(path) == expected
    monkeypatch.setattr(native, "read_text", lambda path: "0::/jobs/1")
    assert native.cgroup_root() == Path("/sys/fs/cgroup/jobs/1")
    monkeypatch.setattr(native, "read_text", lambda path: None)
    assert native.cgroup_root() == Path("/sys/fs/cgroup")


@pytest.mark.parametrize(
    "service,value,kind,child,number,success,expected",
    [
        (0, 0, 1, None, 0, 0, None),
        (1, 0, 1, None, 0, 0, None),
        (1, 2, 9, None, 0, 1, None),
        (1, 2, 1, None, 2, 0, None),
        (1, 2, 1, None, 2, 1, 0),
        (1, 2, 9, "key", 2, 1, None),
        (1, 2, 3, "key", 0, 1, None),
        (1, 2, 3, "key", 4, 1, 0),
    ],
)
def test_registry(
    service: int,
    value: int,
    kind: int,
    child: str | None,
    number: int,
    success: int,
    expected: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = SimpleNamespace(
        IOServiceMatching=Mock(return_value=1),
        IOServiceGetMatchingService=Mock(return_value=service),
        IORegistryEntryCreateCFProperty=Mock(return_value=value),
        IOObjectRelease=Mock(),
    )
    cf = SimpleNamespace(
        CFStringCreateWithCString=Mock(return_value=1),
        CFNumberGetTypeID=Mock(return_value=1),
        CFGetTypeID=Mock(side_effect=lambda ref: kind if ref == value else 1),
        CFNumberGetValue=Mock(return_value=success),
        CFDictionaryGetTypeID=Mock(return_value=3),
        CFDictionaryGetValue=Mock(return_value=number),
        CFRelease=Mock(),
    )
    monkeypatch.setattr(
        apple_native.Libraries, "load", lambda name: SimpleNamespace(handle=io if name == "iokit" else cf)
    )
    assert apple_native.registry_int("property", child) == expected
    if service:
        io.IOObjectRelease.assert_called_once_with(service)


def test_gpu_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apple_native, "registry_int", lambda name: 16)
    assert apple_native.gpu_core_count() == 16


def test_drm_processes(tmp_path: Path) -> None:
    directory = tmp_path / "12" / "fdinfo"
    directory.mkdir(parents=True)
    for index, content in enumerate(
        (
            "",
            "drm-pdev: bus",
            "drm-pdev: other\ndrm-client-id: 1",
            "drm-pdev: bus\ndrm-client-id: 1\ndrm-resident-vram: 2 KiB",
            "drm-pdev: bus\ndrm-client-id: 1\ndrm-resident-vram: 2 KiB",
            "drm-pdev: bus\ndrm-client-id: 2\ndrm-resident-vram: 3",
        )
    ):
        (directory / str(index)).write_text(content)
    assert native.drm_processes("bus", tmp_path) == ((12, 2051),)
    (directory / "6").write_text("drm-pdev: bus\ndrm-client-id: 3\ndrm-resident-vram: unknown")
    assert native.drm_processes("bus", tmp_path) == ((12, 2051),)


def test_caches(tmp_path: Path) -> None:
    directory = tmp_path / "index0"
    directory.mkdir()
    assert native.cpu_caches(tmp_path) == ()
    for key, value in {"level": "2", "type": "Unified", "size": "2M"}.items():
        (directory / key).write_text(value)
    assert native.cpu_caches(tmp_path) == (("l2-unified", 2 * 1024**2),)
    (directory / "size").write_text("100")
    assert native.cpu_caches(tmp_path)[0][1] == 100
    (directory / "size").write_text("unknown")
    assert native.cpu_caches(tmp_path) == ()


def test_cgroup_parents(tmp_path: Path) -> None:
    job = tmp_path / "job"
    child = job / "child"
    child.mkdir(parents=True)
    for directory, cpu, memory in (
        (tmp_path, "100000 100000", "1024"),
        (job, "max 100000", "max"),
        (child, "200000 100000", "2048"),
    ):
        (directory / "cpu.max").write_text(cpu)
        (directory / "memory.max").write_text(memory)
    assert native.cgroup_limits(child, tmp_path) == (1, 1024)

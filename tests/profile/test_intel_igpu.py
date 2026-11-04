from pathlib import Path

import pytest

from krn.compiler.errors import TargetError
from krn.hw import native
from krn.hw.platforms import system_key
from krn.profile.providers.cpu import perf
from krn.profile.providers.intel import igpu


def make_card(root: Path, name: str, *, vendor: str = "0x8086", gt: bool = True, bus: str = "0000:00:02.0") -> Path:
    card = root / name
    pci = root / "pci" / bus
    pci.mkdir(parents=True, exist_ok=True)
    card.mkdir(parents=True)
    (card / "device").symlink_to(pci, target_is_directory=True)
    (pci / "vendor").write_text(vendor)
    (pci / "device").write_text("0xa780")
    if gt:
        (card / "gt" / "gt0").mkdir(parents=True)
    return card


class Ticks:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def test_cards_are_intel_gpus_with_a_gt(tmp_path: Path) -> None:
    make_card(tmp_path, "card0", vendor="0x10de", gt=False)
    make_card(tmp_path, "card1")
    make_card(tmp_path, "card2", gt=False)
    (tmp_path / "card1-DP-1").mkdir()
    assert igpu.igpu_cards(tmp_path) == (tmp_path / "card1",)
    assert igpu.igpu_cards(tmp_path / "missing") == ()


def test_frequencies_are_read_at_card_level(tmp_path: Path) -> None:
    card = make_card(tmp_path, "card1")
    (card / "gt_act_freq_mhz").write_text("300")
    (card / "gt_RP0_freq_mhz").write_text("1650")
    assert igpu.frequencies(card) == igpu.Frequencies(actual_mhz=300.0, requested_mhz=None, max_mhz=1650.0)
    (card / "gt_act_freq_mhz").write_text("not a number")
    assert igpu.frequencies(card).actual_mhz is None


def test_rc6_meter_reports_the_awake_share_between_readings(tmp_path: Path) -> None:
    clock = Ticks()
    gt = tmp_path / "gt"
    gt.mkdir()
    meter = igpu.RC6Meter(gt, clock=clock)
    assert meter.awake_percent() is None
    (gt / "rc6_residency_ms").write_text("0")
    assert meter.awake_percent() is None
    clock.now = 1_000_000_000
    (gt / "rc6_residency_ms").write_text("750")
    assert meter.awake_percent() == pytest.approx(25.0)
    assert meter.awake_percent() is None
    clock.now = 2_000_000_000
    (gt / "rc6_residency_ms").write_text("1750")
    assert meter.awake_percent() == pytest.approx(0.0)
    clock.now = 3_000_000_000
    (gt / "rc6_residency_ms").write_text("1700")
    assert meter.awake_percent() == pytest.approx(100.0)
    (gt / "rc6_residency_ms").unlink()
    assert meter.awake_percent() is None


def test_device_reports_clock_and_awake_share_and_its_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = make_card(tmp_path, "card1")
    (card / "gt_act_freq_mhz").write_text("450")
    (card / "gt_RP0_freq_mhz").write_text("1650")
    (card / "gt" / "gt0" / "rc6_residency_ms").write_text("0")
    clock = Ticks()
    [device] = igpu.igpu_devices(tmp_path, clock=clock)
    assert (device.name, device.index, device.driver, device.source) == ("Intel GPU 0xa780", 1, "i915", "i915-sysfs")
    assert (
        device.pci_bus_id == "0000:00:02.0" and device.memory.unified is True and device.limits.max_clock_mhz == 1650.0
    )
    first = device.snapshot()
    assert first.clock_mhz == 450.0 and first.utilization_percent is None
    clock.now = 1_000_000_000
    (card / "gt" / "gt0" / "rc6_residency_ms").write_text("250")
    second = device.snapshot()
    assert second.utilization_percent == pytest.approx(75.0)
    assert dict(second.sources) == {
        "clock_mhz": "i915-sysfs",
        "max_clock_mhz": "i915-sysfs",
        "utilization_percent": "i915-sysfs",
    }
    monkeypatch.setattr(native, "drm_processes", lambda bus: ((4242, 1024), (4243, None)))
    assert device.processes() == (igpu.Process(pid=4242, memory_bytes=1024), igpu.Process(pid=4243, memory_bytes=None))


def test_i915_events_come_from_the_pmu_export(tmp_path: Path) -> None:
    events = tmp_path / "i915" / "events"
    events.mkdir(parents=True)
    (events / "rcs0-busy").write_text("config=0x0\n")
    (events / "actual-frequency").write_text("config=0x100000")
    assert igpu.i915_event("rcs0-busy", tmp_path) == igpu.I915Event(name="rcs0-busy", config=0)
    assert igpu.i915_event("actual-frequency", tmp_path).config == 0x100000
    with pytest.raises(TargetError, match="does not export this event"):
        igpu.i915_event("rc6-residency", tmp_path)


def test_i915_counter_needs_the_pmu_and_names_the_capability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    event = igpu.I915Event(name="rcs0-busy", config=0)
    with pytest.raises(TargetError, match="no i915 PMU"):
        igpu.open_i915(event, root=tmp_path)
    (tmp_path / "i915").mkdir()
    (tmp_path / "i915" / "type").write_text("14")
    seen: list[dict[str, object]] = []

    def fake_open(kind: int, config: int, **keywords: object) -> int:
        seen.append({"kind": kind, "config": config, **keywords})
        return 77

    monkeypatch.setattr(igpu, "open_raw", fake_open)
    counter = igpu.open_i915(event, root=tmp_path)
    assert counter == perf.Counter(fd=77, pmu="i915", event="rcs0-busy")
    assert seen == [{"kind": 14, "config": 0, "pid": -1, "cpu": 0, "feature": "i915 rcs0-busy", "system": system_key()}]


def test_busy_percent_is_clamped_and_needs_an_interval() -> None:
    assert igpu.busy_percent(0.0, 250.0, 1000.0) == pytest.approx(25.0)
    assert igpu.busy_percent(0.0, 2000.0, 1000.0) == 100.0
    assert igpu.busy_percent(500.0, 100.0, 1000.0) == 0.0
    with pytest.raises(ValueError, match="positive"):
        igpu.busy_percent(0.0, 1.0, 0.0)

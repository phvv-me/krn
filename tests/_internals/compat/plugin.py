"""Record the hardware runs that compare a kernel with the emulator, for the measured evidence of the matrix.

Enabled by `KRN_HARDWARE=1` and `--compat-out DIR`, and off otherwise. A test that runs on a device and compares its
result with the emulator is recorded as one run of one feature on one target: the feature is the row the matrix names
for the test's parameters, and the target is the device's. Each machine's runs go to `DIR/<machine>.json`, which
`krn._internals.compat.measured` merges into the matrix. A run that disagrees is written too, and the session then
fails, since `measured.attach` refuses a mismatch.
"""

import os
import platform
import socket
import subprocess
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

from experiments.compat_matrix import analyzed
from krn._internals.compat import measured
from krn._internals.compat.analyze import catalog_feature
from krn.facts import catalog
from krn.runtime.api import Device
from tests.compiler.test_intrinsic_families import CASES, Case

DEVICE_FIXTURES = {"family_metal": "metal", "metal_device": "metal", "family_card": "cuda", "gpu": "cuda"}
FAMILY_TESTS = {"test_family_metal", "test_family_cuda"}
PACKED_TESTS = {"test_packed_sub_sat_and_avg_match_lane_oracle_on_metal", "test_general_packed_runs_on_the_card"}
VIDEO_TESTS = {"test_every_name_runs_on_the_card"}
VIDEO_TARGET = "sm_120a"  # test_video captures its kernels for this target


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--compat-out", default=None, help="write the measured runs of the hardware tests to this directory"
    )


def pytest_configure(config: pytest.Config) -> None:
    out = config.getoption("compat_out")
    if out and os.environ.get("KRN_HARDWARE") == "1":
        config.pluginmanager.register(Recorder(Path(out)), "compat-recorder")


class Recorder:
    """Collects one record per measured test, and writes them and checks them at the end of the session."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.records: list[measured.Record] = []

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo) -> Iterator[pytest.TestReport | None]:
        report = yield
        if not isinstance(item, pytest.Function) or call.when != "call" or report is None or report.skipped:
            return report
        if (run := _identify(item)) is not None:
            feature, target, tolerance, device = run
            status = _outcome(call)
            self.records.append(
                measured.Record(
                    feature=feature,
                    target=target,
                    machine=f"{socket.gethostname().split('.')[0]} {device.target.arch}",
                    os=_os(),
                    driver=_driver(device.target.vendor),
                    status=status,
                    tolerance=tolerance,
                )
            )
        return report

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if not self.records:
            return
        writer = session.config.get_terminal_writer()
        for machine in sorted({record.machine for record in self.records}):
            runs = [record for record in self.records if record.machine == machine]
            writer.line(f"compat: wrote {measured.write(self.directory, machine, runs)}")
        try:
            measured.attach(analyzed(), self.records)
        except ValueError as error:
            writer.line(f"compat: {error}", red=True)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _outcome(call: pytest.CallInfo) -> measured.Outcome:
    if call.excinfo is None:
        return measured.Outcome.MATCH
    return measured.Outcome.MISMATCH if call.excinfo.errisinstance(AssertionError) else measured.Outcome.ERROR


def _identify(item: pytest.Function) -> tuple[str, str, str, Device] | None:
    """The feature, target, tolerance and device of a measured test, or None for a test the matrix does not measure."""
    device = next((item.funcargs[name] for name in DEVICE_FIXTURES if name in item.funcargs), None)
    if not isinstance(device, Device):
        return None
    args = item.funcargs
    if item.originalname in FAMILY_TESTS:
        case = args.get("case")
        if not isinstance(case, Case) or not case.supported(device.target):
            return None
        return _case_feature(case), device.target.arch, _tolerance(case), device
    if item.originalname in VIDEO_TESTS:
        kernel = args.get("name")
        if device.target.arch != VIDEO_TARGET or not isinstance(kernel, str):
            return None
        return f"simd.{kernel}", VIDEO_TARGET, "exact", device
    op, ty = args.get("op"), args.get("ty")
    if item.originalname in PACKED_TESTS and isinstance(op, str) and isinstance(ty, str):
        return f"krn.math.{op}[{ty}]", device.target.arch, "exact", device
    return None


def _case_feature(case: Case) -> str:
    """The row a family case is, as `experiments.compat_matrix` names it: the dtypes are in the id when the name has
    several cases."""
    dtypes = {c.dtypes for c in CASES if c.name == case.name}
    entry = catalog.BY_NAME[case.name]
    return catalog_feature(entry, case.dtypes if len(dtypes) > 1 else ()).id


def _tolerance(case: Case) -> str:
    """The check the case applies, as `Case.check` states it: bit-exact for the cache and rounding families, the fast
    family's relative tolerance, the default float tolerance for float operands, and exact for integers."""
    if case.name.startswith(("cache.", "rounded.")):
        return "bit-exact"
    if case.name.startswith("fast."):
        return "rtol 1e-4, atol 1e-6"
    if any(dtype in ("f32", "f64") for dtype in case.dtypes):
        return "rtol 1e-5, atol 1e-6"
    return "exact"


def _os() -> str:
    if platform.system() == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    return f"{platform.system()} {platform.release()}"


@cache
def _driver(vendor: str) -> str:
    """The driver the runs of one vendor used: the macOS release for Metal, else the NVIDIA driver of the machine."""
    if vendor == "apple":
        return f"Metal on macOS {platform.mac_ver()[0]}"
    version = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return f"NVIDIA driver {version.stdout.strip()}"

"""Settings snapshots survive scopes, concurrent compilation, and program-cache reuse."""

import asyncio
import inspect
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

import krn
from krn.compiler import performance
from krn.compiler.capture import capture
from krn.compiler.errors import SettingError
from krn.compiler.ir import Kernel as IrKernel
from krn.compiler.performance import Facts, PerformanceWarning, check, compilation
from krn.facts.targets import CPU, Target
from krn.runtime import settings as state
from krn.runtime.runtime import Argument, CPUDevice, CPULauncher
from krn.runtime.settings import BACKENDS, Settings
from tests.compiler.test_heuristics import THIS, update


def test_groups_copy_on_write_and_scopes() -> None:
    before = state.current_settings()
    krn.configure(lib="ours", warn=krn.Warn(max_registers_per_thread=192))
    chosen = state.current_settings()
    assert before.lib.mode == "auto" and chosen.lib.mode == "ours"
    assert krn.config.warn.max_registers_per_thread == 192
    with pytest.raises(FrozenInstanceError):
        chosen.lib.__setattr__("mode", "auto")
    with krn.configure(lib="vendor"):
        with krn.configure(**{"lib.matmul": "host"}):
            assert krn.config.lib.matmul == "host"
        assert krn.config.lib.matmul is None
        krn.config.lib.mode = "ours"
        assert krn.config.lib.mode == "ours"
    assert state.current_settings() is chosen
    krn.config.__setattr__("run", krn.Run(workers=2))
    krn.config.compile.cpu.opt = "3"
    assert state.current_settings().compile.cpu.opt == "3"
    with pytest.raises(SettingError, match="known setting"):
        krn.configure(unknown=True)
    with pytest.raises(SettingError, match="Lib"):
        krn.configure(lib=2)
    mapping = {"x.y": "1"}
    frozen = krn.Run(config=mapping)
    mapping["x.y"] = "2"
    assert frozen.config["x.y"] == "1"


def test_scopes_are_per_thread_and_async_task() -> None:
    async def worker(mode: str) -> str:
        with krn.configure(lib=mode):
            await asyncio.sleep(0)
            return state.current_settings().lib.mode

    async def tasks() -> list[str]:
        return list(await asyncio.gather(worker("ours"), worker("vendor")))

    assert asyncio.run(tasks()) == ["ours", "vendor"]
    with krn.configure(lib="ours"), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(lambda: state.current_settings().lib.mode).result() == "auto"
    assert krn.config.lib.mode == "auto"


@pytest.mark.parametrize(
    "name,value",
    [
        ("KRN_TARGET", "unknown"),
        ("KRN_WORKERS", "0"),
        ("KRN_WORKERS", "bad"),
        ("KRN_CACHE_DIR", ""),
        ("KRN_DEBUG", "maybe"),
        ("KRN_DEBUG_LINEINFO", "maybe"),
        ("KRN_DEBUG_KEEP", "bad"),
        ("KRN_DEBUG_DIAG", "bad"),
        ("KRN_DEBUG_EMULATE", "maybe"),
        ("KRN_DEBUG_DUMP_DIR", ""),
    ],
)
def test_invalid_seeds_name_the_variable(name: str, value: str) -> None:
    with pytest.raises(SettingError, match=name):
        Settings.from_environ({name: value})


def test_removed_variables_are_inert_and_environment_is_not_reparsed(monkeypatch: pytest.MonkeyPatch) -> None:
    removed = (
        "BLOCK",
        "MAXREGS",
        "UNROLL",
        "INLINE",
        "INLINE_LIMIT",
        "RESTRICT",
        "CONST",
        "LOG_LEVEL",
        "GRAPHS",
        "POLICY",
        "PRECISION",
        "PRECISION_FLOOR_BITS",
        "VARIANT",
        "CONFIG",
        "LIB",
        "LIB_MATMUL",
        "LIB_TUNE",
        "OPT",
        "CXX_STD",
        "MARCH",
        "LINEINFO",
        "KEEP",
        "DUMP_DIR",
        "DIAG",
        "EMULATE",
    )
    assert Settings.from_environ({f"KRN_{name}": "invalid" for name in removed}) == Settings()
    original = state.current_settings()
    monkeypatch.setenv("KRN_WORKERS", "999")
    assert state.current_settings() is original
    assert not {"maxregs", "inline", "inline_limit", "unroll", "restrict", "const", "block"} & {
        f.name for f in fields(Settings)
    }
    assert tuple(inspect.signature(krn.kernel).parameters) == ("fn",)
    assert tuple(inspect.signature(krn.items).parameters) == ("count",)
    assert not hasattr(krn, "noalias") and not hasattr(krn, "const")


def test_settings_metadata_is_complete() -> None:
    def visit(
        group: Settings
        | krn.Compile
        | krn.BackendCompile
        | krn.Warn
        | krn.Run
        | krn.Cache
        | krn.Lib
        | krn.Log
        | krn.Errors
        | krn.Precision
        | krn.Debug,
    ) -> None:
        for item in fields(group):
            assert set(item.metadata["backends"]) <= set(BACKENDS)

    snapshot = Settings()
    visit(snapshot)
    for group in (
        snapshot.debug,
        snapshot.warn,
        snapshot.lib,
        snapshot.cache,
        snapshot.run,
        snapshot.precision,
        snapshot.compile,
        snapshot.compile.cuda,
        snapshot.compile.amd,
        snapshot.compile.metal,
        snapshot.compile.cpu,
    ):
        visit(group)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_registers_per_thread": -1},
        {"max_spilled_bytes": True},
        {"min_threads_per_block": -1},
        {"max_unrolled_instructions": -1},
        {"min_occupancy": float("nan")},
        {"min_occupancy": 1.1},
        {"warn_on_unproven_aliasing": 1},
        {"warn_on_unfilled_warps": 0},
    ],
)
def test_warning_limits_validate(changes: dict[str, int | float | bool]) -> None:
    with pytest.raises(SettingError):
        krn.configure(**{f"warn.{key}": value for key, value in changes.items()})


def test_performance_warning_facts_filters_limits_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())
    ir = capture(THIS, "update")
    facts = Facts(
        registers=214,
        spilled_bytes=16,
        occupancy=0.125,
        threads=17,
        warp_size=32,
        thread_limit=32,
        unrolled_instructions=300,
        unproven_aliasing=True,
    )
    with pytest.warns(PerformanceWarning) as caught:
        check(ir, "sm_90a", krn.Warn(), facts)
    assert len(caught) == 8
    assert all(w.filename == ir.span.file and w.lineno == ir.span.line for w in caught)
    assert all("update on sm_90a" in str(w.message) for w in caught)
    assert "214 registers" in str(caught[0].message)
    with pytest.warns(PerformanceWarning):
        check(ir, "gfx942", krn.Warn(), facts)
    with warnings.catch_warnings(record=True) as quiet:
        check(ir, "sm_90a", krn.Warn(), facts)
        check(ir, "cpu", krn.Warn(), Facts())
        check(
            ir,
            "apple_m4",
            krn.Warn(
                max_registers_per_thread=300,
                min_occupancy=0,
                max_spilled_bytes=32,
                min_threads_per_block=0,
                max_unrolled_instructions=400,
                warn_on_unproven_aliasing=False,
                warn_on_unfilled_warps=False,
            ),
            facts,
        )
        assert not quiet
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=PerformanceWarning)
        with pytest.raises(PerformanceWarning):
            check(ir, "new_target", krn.Warn(), facts)
    check(ir, "whole_warp", krn.Warn(), Facts(threads=256, warp_size=32))
    check(ir, "unknown_width", krn.Warn(), Facts(threads=256))


def test_concurrent_compiles_retain_starting_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    krn.configure(run=krn.Run(target="avx512", workers=1), cache=krn.Cache(dir=tmp_path))
    device = CPUDevice(CPU.of("avx512"))
    workers = 3
    entered, changed = Barrier(workers + 1), Barrier(workers + 1)
    original = CPULauncher.captured

    def captured(self: CPULauncher, target: Target, args: Sequence[Argument]) -> IrKernel:
        snapshot = state.current_settings()
        entered.wait(timeout=30)
        changed.wait(timeout=30)
        assert state.current_settings() is snapshot
        return original(self, target, args)

    monkeypatch.setattr(CPULauncher, "captured", captured)
    x, y = np.arange(17, dtype=np.float32), np.ones(17, dtype=np.float32)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            builds = [pool.submit(device.launcher(update, krn.Blocks(i + 1)).program, x, y) for i in range(workers)]
            entered.wait(timeout=30)
            krn.configure(debug=krn.Debug(lineinfo=True), **{"compile.cpu.opt": "3"})
            changed.wait(timeout=30)
            first = [future.result(timeout=60) for future in builds]
        monkeypatch.setattr(CPULauncher, "captured", original)
        launcher = device.launcher(update)
        later = launcher.program(x, y)
        assert all(program is first[0] for program in first)
        assert later is not first[0]
        assert "#line " not in first[0].unit.source and "#line " in later.unit.source
        assert first[0].library != later.library
        krn.configure(debug=krn.Debug(), **{"compile.cpu.opt": None})
        assert launcher.program(x, y) is first[0]
    finally:
        device.close()


def test_ptxas_compile_resource_facts() -> None:
    result = compilation(
        b"cubin",
        """ptxas info : 24 bytes stack frame, 16 bytes spill stores, 32 bytes spill loads
ptxas info : Used 214 registers, used 0 barriers, 48 bytes smem
""",
    )
    assert (result.cubin, result.registers, result.spilled_bytes) == (b"cubin", 214, 48)
    unknown = compilation(b"cached", "NVRTC has no ptxas resource output")
    assert unknown.registers is None and unknown.spilled_bytes is None
    zero = compilation(b"cubin", "ptxas info : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads")
    assert zero.spilled_bytes == 0

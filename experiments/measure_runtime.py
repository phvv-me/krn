"""Measure the CPU runtime on the interpreter that runs this script, and merge the result into results/runtime.json.

    PYTHONPATH=.:../sketch:../experiments <python> measure_runtime.py run
    PYTHONPATH=.:../sketch:../experiments PYTHON_GIL=0 <python3.14t> measure_runtime.py run

Run it once per interpreter. Each result carries the load average read before and after it, because other agents share
the machine: read differences under 25 percent as noise. Latencies are per call, bracketed by `perf_counter_ns`, minus
the cost of an empty bracket.
"""

import asyncio
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import product
from pathlib import Path

import cyclopts
import numpy as np

from experiments.paths import RESULTS
from experiments.sketch.elementwise import Tiling, add, saxpy
from krn import Blocks, Items, Run, T, host, targets
from krn.arithmetic import ceildiv
from krn.cache import cpu as cpu_cache
from krn.compiler.api import CPUDevice, CPUTensor, capture
from krn.compiler.dtypes import Arithmetic

log = logging.getLogger("measure_runtime")
app = cyclopts.App()
HERE = Path(__file__).resolve().parent
SKETCH = HERE / "sketch"
TRIVIAL, LAUNCHES, GRAPH_LENGTH, TIMEOUT = 64, 20000, 10, 600
TILINGS = [Tiling(rows=rows, cols=cols) for rows, cols in product((8, 16, 32, 64), (16, 64))]
SHAPE = (96, 96)
EXPERIMENTS = HERE.parent / "experiments"

type Node = Summary | Load | float | str | bool | list[Node] | dict[str, Node] | None


@dataclass(frozen=True)
class Load:
    load1: float
    uptime: str

    @classmethod
    def read(cls) -> Load:
        return cls(os.getloadavg()[0], subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip())


@dataclass(frozen=True)
class Summary:
    median: float
    min: float
    max: float
    unit: str
    trials: int
    noisy: bool  # the slowest trial is over 25 percent above the fastest
    load_before: Load
    load_after: Load
    speedup: float | None = None


@dataclass
class Spread:
    """Repeated measurements of one quantity, with the load average read when it opened and when it closed."""

    unit: str
    trials: list[float] = field(default_factory=list)
    opened: Load = field(default_factory=Load.read)

    @classmethod
    def of(cls, unit: str, trials: Sequence[float]) -> Summary:
        return cls(unit, list(trials)).summary()

    def summary(self, speedup: float | None = None) -> Summary:
        low, high = min(self.trials), max(self.trials)
        return Summary(
            statistics.median(self.trials),
            low,
            high,
            self.unit,
            len(self.trials),
            high > 1.25 * low,
            self.opened,
            Load.read(),
            speedup,
        )


def current_device() -> CPUDevice:
    device = host.Device.current()
    assert isinstance(device, CPUDevice)
    return device


def bracket_cost_ns() -> float:
    """The median cost of reading the clock twice, which every per-call latency below subtracts."""
    return statistics.median(-(time.perf_counter_ns() - time.perf_counter_ns()) for _ in range(20000))


def latencies_us(call: Callable[[], None], count: int, cost_ns: float) -> list[float]:
    out = []
    for _ in range(count):
        started = time.perf_counter_ns()
        call()
        out.append((time.perf_counter_ns() - started - cost_ns) / 1e3)
    return out


class Overhead:
    """(a) and (d): what Python costs per launch of a trivial kernel, on the host and over an asyncio round trip."""

    def __init__(self) -> None:
        self.device, self.cost_ns = current_device(), bracket_cost_ns()
        self.x, self.y = self.device.empty(T.f32, TRIVIAL), self.device.empty(T.f32, TRIVIAL)
        self.x.array[:], self.y.array[:] = 1, 0
        self.launcher = host.Device.current().launcher(saxpy, Items(TRIVIAL))
        self.eager()
        self.settle()

    def eager(self) -> None:
        self.launcher(T.f32(2.0), self.x, self.y)

    def settle(self) -> None:
        self.device.default_stream.record().wait(TIMEOUT)

    def sustained_us(self, enqueue: Callable[[], None], count: int) -> float:
        """Microseconds per launch from the first enqueue to the last completion, the worker included."""
        started = time.perf_counter()
        for _ in range(count):
            enqueue()
        self.settle()
        return (time.perf_counter() - started) / count * 1e6

    def host_side(self) -> dict[str, Node]:
        bound = self.launcher.bind(T.f32(2.0), self.x, self.y)

        def ten(x: CPUTensor[T.f32, int], y: CPUTensor[T.f32, int]) -> None:
            for _ in range(GRAPH_LENGTH):
                self.eager()

        graph = self.device.graph(ten, self.x, self.y)

        def replay() -> None:
            graph(self.x, self.y)

        results: dict[str, Node] = {}
        for name, call, per in (
            ("eager", self.eager, 1),
            ("bound", bound, 1),
            ("graph_replay_of_10", replay, GRAPH_LENGTH),
        ):
            count = LAUNCHES // per
            trials = []
            for _ in range(3):
                trials.append(statistics.median(latencies_us(call, count, self.cost_ns)) / per)
                self.settle()
            results[name] = Spread.of("us per launch, host side", trials)
            results[f"{name}_sustained"] = Spread.of(
                "us per launch, first enqueue to last completion",
                [self.sustained_us(call, count) / per for _ in range(3)],
            )
        results["await_submit_pipelined"] = Spread.of(
            "us per event, 1000 submitted then awaited", asyncio.run(self.pipelined())
        )
        return results

    async def pipelined(self, events: int = 1000, trials: int = 10) -> list[float]:
        out = []
        for _ in range(trials):
            started = time.perf_counter()
            pending = [self.launcher.submit(T.f32(2.0), self.x, self.y) for _ in range(events)]
            for event in pending:
                await event
            out.append((time.perf_counter() - started) / events * 1e6)
        return out

    def sequential_awaits(self, count: int = 10000) -> dict[str, Node]:
        """One `await launcher.submit(...)` after another: the latency of a full round trip through the event loop."""

        async def main() -> list[float]:
            out = []
            for _ in range(count):
                started = time.perf_counter_ns()
                await self.launcher.submit(T.f32(2.0), self.x, self.y)
                out.append((time.perf_counter_ns() - started - self.cost_ns) / 1e3)
            return out

        runs = [asyncio.run(main()) for _ in range(3)]
        flat = sorted(value for run in runs for value in run)
        return {
            "median_us": Spread.of("us per awaited launch", [statistics.median(run) for run in runs]),
            "p99_us": flat[int(0.99 * (len(flat) - 1))],
            "mean_us": statistics.fmean(flat),
            "awaits_per_second": 1e6 / statistics.median(flat),
        }

    def linger(self) -> dict[str, Node]:
        """What a lane keeping its worker for a moment after running dry does to the two latencies it targets."""
        out: dict[str, Node] = {}
        pool = self.device.pool
        configured = pool.spin_ns
        try:
            for spin_ns in (0, 100_000):
                pool.spin_ns = spin_ns
                eager = []
                for _ in range(3):
                    eager.append(statistics.median(latencies_us(self.eager, LAUNCHES, self.cost_ns)))
                    self.settle()
                out[f"spin_{spin_ns}_ns"] = {
                    "eager_us": Spread.of("us per launch, host side", eager),
                    "await": self.sequential_awaits(3000),
                }
        finally:
            pool.spin_ns = configured
        return out


def grid_of(tiling: Tiling) -> Blocks:
    return Blocks(ceildiv(SHAPE[1], tiling.cols), ceildiv(SHAPE[0], tiling.rows))


def build_wall_seconds(threads: int) -> float:
    """Wall time to capture and compile the eight programs of `TILINGS` from `threads` threads, clang cache cold."""
    scratch = Path(tempfile.mkdtemp(prefix="krn_cold_"))
    previous, cpu_cache.CACHE = cpu_cache.CACHE, scratch
    device = CPUDevice(targets.CPU.of("avx512"), host.Settings(run=Run(workers=1)))
    a, b, c = (CPUTensor[T.f32, int, int](np.zeros(SHAPE, dtype=np.float32)) for _ in range(3))
    try:
        started = time.perf_counter()
        with ThreadPoolExecutor(threads) as pool:
            for future in [pool.submit(device.launcher(add, grid_of(t)).program, a, b, c, t) for t in TILINGS]:
                future.result(TIMEOUT)
        return time.perf_counter() - started
    finally:
        cpu_cache.CACHE = previous
        device.close()
        shutil.rmtree(scratch, ignore_errors=True)


def capture_wall_seconds(threads: int, repeats: int = 300) -> float:
    """Wall time of `repeats` captures of each of the eight kernels: pure Python, the part a GIL serializes."""
    path = SKETCH / "elementwise.py"
    target = targets.CPU.of("avx512")
    jobs = [{"tiling": {"rows": t.rows, "cols": t.cols}} for t in TILINGS] * repeats
    started = time.perf_counter()
    with ThreadPoolExecutor(threads) as pool:
        futures = [
            pool.submit(capture, path, "add", consts=c, target=target, arithmetic=Arithmetic.STRICT) for c in jobs
        ]
        for future in futures:
            future.result(TIMEOUT)
    return time.perf_counter() - started


def scaling(
    measure: Callable[[int], float], unit: str, threads: Sequence[int] = (1, 2, 4), trials: int = 3
) -> dict[str, Node]:
    """Wall time per thread count, with the counts interleaved across trials so a load spike hits all of them."""
    spreads = {t: Spread(unit) for t in threads}
    for _ in range(trials):
        for t in threads:
            spreads[t].trials.append(measure(t))
    base = statistics.median(spreads[threads[0]].trials)
    return {f"threads_{t}": spreads[t].summary(base / statistics.median(spreads[t].trials)) for t in threads}


def streams_throughput(jobs: int = 4, launches: int = 3, trials: int = 5, n: int = 1 << 24) -> dict[str, Node]:
    """(c) Four independent large saxpy jobs on four streams against the same four on one stream."""
    device = current_device()
    pairs = [(device.empty(T.f32, n), device.empty(T.f32, n)) for _ in range(jobs)]
    for x, y in pairs:
        x.array[:], y.array[:] = 1, 0
    launcher = host.Device.current().launcher(saxpy, Items(n))

    def wall(streams: int) -> float:
        lanes = [device.stream() for _ in range(streams)]
        started = time.perf_counter()
        for job, (x, y) in enumerate(pairs):
            with lanes[job % streams]:
                for _ in range(launches):
                    launcher(T.f32(1.0), x, y)
        for lane in lanes:
            lane.record().wait(TIMEOUT)
        return time.perf_counter() - started

    wall(jobs)
    one, many = Spread("seconds"), Spread("seconds")
    for _ in range(trials):
        one.trials.append(wall(1))
        many.trials.append(wall(jobs))
    traffic = jobs * launches * n * 12 / 1e9  # saxpy reads x and y and writes y: 12 bytes per element
    speedup = statistics.median(one.trials) / statistics.median(many.trials)
    return {
        "elements": float(n),
        "jobs": float(jobs),
        "launches_per_job": float(launches),
        "one_stream": one.summary(),
        "four_streams": many.summary(speedup),
        "gb_per_s_one_stream": traffic / statistics.median(one.trials),
        "gb_per_s_four_streams": traffic / statistics.median(many.trials),
    }


def fresh_process_ms(code: str, pythonpath: str, repeats: int = 7) -> Summary:
    spread = Spread("ms")
    for _ in range(repeats):
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, "-P", "-c", code],
            env={**os.environ, "PYTHONPATH": pythonpath},
            check=True,
            timeout=TIMEOUT,
        )
        spread.trials.append((time.perf_counter() - started) * 1e3)
    return spread.summary()


DTYPE_NAMES = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
)
LAZY_NUMPY = """import importlib.util as _import_util
import sys as _sys
try:
    np = _sys.modules["numpy"]
except KeyError:
    _numpy_spec = _import_util.find_spec("numpy")
    assert _numpy_spec is not None and _numpy_spec.loader is not None
    _numpy_spec.loader = _import_util.LazyLoader(_numpy_spec.loader)
    np = _import_util.module_from_spec(_numpy_spec)
    _sys.modules["numpy"] = np
    _numpy_spec.loader.exec_module(np)
"""
LAZY_EDITS = {
    "cpu.py": [
        ("import numpy as np", LAZY_NUMPY),
        *((f'np.dtype("{n}")', f'"{n}"') for n in DTYPE_NAMES),
    ],
    "runtime.py": [
        ("import numpy as np", LAZY_NUMPY),
        ("from numpy.lib.array_utils import byte_bounds", ""),
        ("byte_bounds(", "np.lib.array_utils.byte_bounds("),
        (
            "DTYPES = {dtype: proto for proto, dtype in NUMPY.items()}",
            "@cache\ndef _numpy_types() -> dict[np.dtype, Dtype]:\n"
            "    return {np.dtype(dtype): proto for proto, dtype in NUMPY.items()}",
        ),
        ("DTYPES", "_numpy_types()"),
        ("return NUMPY[from_sketch(dtype)]", "return np.dtype(NUMPY[from_sketch(dtype)])"),
    ],
}
FIRST_LAUNCH = """
import krn
from krn import Items, T, host
from experiments.sketch.elementwise import saxpy
d = host.Device.current(); x = d.empty(T.f32, 8); y = d.empty(T.f32, 8)
saxpy(2.0, x, y, launch=Items(8))
d.default_stream.record().wait(60)
"""


def import_times() -> dict[str, Node]:
    """(e) Import cost in a fresh process, against the same package with numpy behind `LazyLoader`.

    The lazy variant is a throwaway copy of the package in a temp directory: `cpu.py` and `runtime.py` import numpy
    lazily and keep their dtype table as names, so nothing loads numpy until a kernel's arguments do.
    """
    eager = f"{HERE.parent}:{HERE.parent / 'src'}"
    scratch = Path(tempfile.mkdtemp(prefix="krn_lazy_"))
    shutil.copytree(HERE.parent / "src" / "krn", scratch / "krn", ignore=shutil.ignore_patterns("__pycache__"))
    for name, edits in LAZY_EDITS.items():
        text = (scratch / "krn" / "runtime" / name).read_text()
        for old, new in edits:
            text = text.replace(old, new)
        (scratch / "krn" / "runtime" / name).write_text(text)
    lazy = f"{scratch}:{HERE.parent}"
    # LazyLoader registers a module before executing it; initialization restores the ordinary module class.
    probe = "import sys; import krn; sys.stdout.write(str(type(sys.modules.get('numpy')) is type(sys)))"
    try:
        loaded = subprocess.run(
            [sys.executable, "-P", "-c", probe],
            env={**os.environ, "PYTHONPATH": lazy},
            capture_output=True,
            text=True,
            check=True,
            timeout=TIMEOUT,
        )
        return {
            "interpreter_start": fresh_process_ms("pass", eager),
            "import_numpy": fresh_process_ms("import numpy", eager),
            "import_krn": fresh_process_ms("import krn", eager),
            "import_krn.compiler.api": fresh_process_ms("import krn", eager),
            "import_krn.compiler.api_lazy_numpy": fresh_process_ms("import krn", lazy),
            "numpy_loaded_by_lazy_import": loaded.stdout.strip() == "True",
            "import_and_first_launch": fresh_process_ms(FIRST_LAUNCH, eager),
            "import_and_first_launch_lazy_numpy": fresh_process_ms(FIRST_LAUNCH, lazy),
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def as_json(node: Node) -> Node | dict[str, Node]:
    return asdict(node) if is_dataclass(node) and not isinstance(node, type) else node


@app.command
def run(out: Path = RESULTS / "runtime.json", *, groups: str = "abcdef") -> None:
    """Measure this interpreter and merge the result into `out` under its label."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    label = "gil" if sys._is_gil_enabled() else "free-threaded"
    device = current_device()
    result: dict[str, Node] = {
        "label": label,
        "python": sys.version,
        "free_threading_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "cpus": float(os.cpu_count() or 0),
        "workers": float(device.settings.run.workers or os.cpu_count() or 1),
        "spin_ns": float(device.pool.spin_ns),
        "started": Load.read(),
    }
    if set("adf") & set(groups):
        overhead = Overhead()
        if "a" in groups:
            result["a_overhead_per_launch"] = overhead.host_side()
        if "d" in groups:
            result["d_sequential_await"] = overhead.sequential_awaits()
        if "f" in groups:
            result["f_linger"] = overhead.linger()
    if "b" in groups:
        result["b_parallel_build"] = scaling(build_wall_seconds, "seconds, capture and clang for 8 kernels, cold cache")
        result["b_parallel_capture_only"] = scaling(capture_wall_seconds, "seconds, 2400 captures, no clang")
    if "c" in groups:
        result["c_streams"] = streams_throughput()
    if "e" in groups:
        result["e_import"] = import_times()
    result["finished"] = Load.read()
    result["gil_enabled_at_end"] = sys._is_gil_enabled()
    document = json.loads(out.read_text()) if out.exists() else {"runs": {}}
    document["runs"].setdefault(label, {}).update(json.loads(json.dumps(result, default=as_json)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n")
    log.info("wrote %s (%s)", out, label)


if __name__ == "__main__":
    app()

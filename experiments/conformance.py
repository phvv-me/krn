"""The conformance matrix of krn.gpu: which kernels run on which target, and why the rest do not.

A SUBJECT is a kernel, an intrinsic, a column operation, an example or a language check, with a stable id built from
the pytest node ids of the prototype suite (the name in the parameter id when there is one, else the test function).
A TARGET is a host and a backend: `mac/c` (Apple CPU through clang), `mac/metal` (the M4 GPU), `silo/c` (Intel
i9-14900, AVX2), `silo/cuda` (RTX 5080, sm_120a), `crimson/c` (AMD Threadripper 3960X), `crimson/cuda` (RTX 3090,
sm_86), `gh200/c` (Miyabi-G Grace CPU), `gh200/cuda` (H100, sm_90a), and the targets that do not depend on the host:
`emulator`, `nvrtc` (compile only), `hip` (clang AMDGPU for gfx942 and gfx950, compile only), `msl` (MSL type check,
compile only) and `frontend` (capture, refusals, rendering and the host API).

Each test node has one ROLE, the target it exercises, taken from its function name and parameter id (the model is in
`conformance_model.py`). A cell of the matrix aggregates the nodes of one subject and one role on one host: `pass` when
every applicable node passed, `fail` when one failed, `partial` when some passed and others were skipped for a reason
that is not by design, `skip` when nothing ran, `na` when the subject has no node for the target. Skips are classified:
no hardware, no toolchain, by design, other platform, not implemented, other. Two percentages come out for a target:
the passing subjects over all kernel subjects of the suite, and over the ones the suite tests on it.

    mb run --no-install python -m experiments.conformance collect mac        # suite and examples on this machine
    mb run --no-install python -m experiments.conformance collect silo       # through mb, under the GPU lock
    mb run --no-install python -m experiments.conformance ingest gh200 junit.xml node.txt   # a run made through PBS
    mb run --no-install python -m experiments.conformance parse              # raw runs to results/conformance_DATE.json
    mb run --no-install python -m experiments.conformance report             # notes/round10/conformance-report.md
"""

import ast
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import cyclopts

import experiments.conformance_report as conformance_report
import experiments.results_schema as schema
from experiments.conformance_model import (
    HOSTS,
    KERNEL_FAMILIES,
    NOT_APPLICABLE,
    TARGETS,
    Family,
    Matrix,
    Node,
    Outcome,
    Percent,
    SkipClass,
    State,
    example_nodes,
    read_junit,
    short_reason,
)
from experiments.paths import RESULTS

LOG = logging.getLogger("conformance")
ROOT = Path(__file__).resolve().parents[1]

NOTE = ROOT / "docs" / "notes" / "round10" / "conformance-report.md"
RAW = Path(os.environ.get("KL_CONFORMANCE_RAW", "/tmp/kl-conf"))
BEGIN, END = "@@KL-BEGIN@@", "@@KL-END@@"
GPU_LOCK = Path("/tmp/kl-gpu.lock")
CPU_TARGET = {"mac": "avx512", "silo": "avx2", "crimson": "avx2", "gh200": "avx512"}
PACKAGES = (
    "numpy",
    "pytest",
    "cuda-core",
    "cuda-bindings",
    "nvmath-python",
    "nvidia-cuda-nvrtc",
    "nvidia-cublas",
    "nvidia-cusparse",
    "pyobjc-framework-Metal",
    "more-itertools",
    "toolz",
)
APP = cyclopts.App(name="conformance", help=__doc__.splitlines()[0])


def stream(command: list[str], timeout: int) -> str:
    """Run a command and return its output; a command that outlives its bound is killed."""
    LOG.info("run: %s", " ".join(command))
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if done.returncode:
        LOG.warning("exit %s: %s", done.returncode, done.stderr[-400:])
    return done.stdout + ("\n" + done.stderr if done.returncode else "")


def between(text: str, tag: str) -> str:
    """The JSON text a remote command printed between two markers."""
    _, begin, remaining = text.partition(f"{BEGIN}{tag}\n")
    payload, end, _ = remaining.partition(f"\n{END}{tag}")
    if not begin or not end:
        raise ValueError(f"no {tag} document in the output: {text[-600:]}")
    return payload


def emit(tag: str, payload: str) -> None:
    """Print JSON text between markers, for `between` to find in the output of `mb run`."""
    sys.stdout.write(f"\n{BEGIN}{tag}\n{payload}\n{END}{tag}\n")
    sys.stdout.flush()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except OSError, subprocess.TimeoutExpired:
        return ""


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def facts() -> schema.Facts:
    """What the host is, for the report: CPU, GPU, compilers and the library versions the suite depends on."""
    cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or next(
        (
            line.split(":", 1)[1].strip()
            for line in command_output(["lscpu"]).splitlines()
            if line.startswith("Model name")
        ),
        "",
    )
    query = ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
    if visible := os.environ.get("CUDA_VISIBLE_DEVICES"):
        query += ["-i", visible]  # a host that pins its card (crimson) reports the card the job sees
    gpus = [line.strip() for line in command_output(query).splitlines()]
    clang = command_output(["clang", "--version"]).splitlines()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "gpus": gpus or (["Apple M4 Pro GPU (Metal)"] if sys.platform == "darwin" else []),
        "python": platform.python_version(),
        "clang": clang[0] if clang else "",
        "packages": {name: package_version(name) for name in PACKAGES},
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def node_rows(nodes: Iterable[Node]) -> list[schema.NodeRow]:
    return [
        {
            "stem": n.stem,
            "function": n.function,
            "parameter": n.parameter,
            "outcome": n.outcome.value,
            "seconds": n.seconds,
            "message": n.message[:300],
        }
        for n in nodes
    ]


@APP.command
def suite(target: str, label: str, timeout: int = 3000, ignore: tuple[str, ...] = ()) -> None:
    """Run the prototype suite here and print its nodes as JSON (the remote half of `collect`).

    target: the CPU target the runtime tests use (`KRN_TARGET`), avx2 or avx512.
    label: the host name the nodes are recorded under.
    timeout: seconds for the whole run.
    ignore: test files to leave out, as pytest `--ignore` takes them (a file that needs the network on a PBS node).
    """
    with tempfile.TemporaryDirectory() as directory:
        junit = Path(directory) / "junit.xml"
        command = [sys.executable, "-m", "pytest", "--timeout=300", "-v", "-rs", "-p", "no:cacheprovider"]
        command += ["--continue-on-collection-errors", "-o", "addopts=", "-o", "junit_logging=no"]
        command += [f"--junitxml={junit}", *(f"--ignore={path}" for path in ignore)]
        started = time.time()
        env = os.environ | {"KRN_TARGET": target}
        done = subprocess.run(command, cwd=ROOT, env=env, timeout=timeout, check=False, capture_output=True, text=True)
        sys.stdout.write(done.stdout[-3000:])
        nodes = read_junit(junit, label)
    document: schema.SuiteDocument = {
        "label": label,
        "seconds": round(time.time() - started, 1),
        "target": target,
        "facts": facts(),
        "nodes": node_rows(nodes),
    }
    emit("suite", json.dumps(document, separators=(",", ":")))


def literal_set(path: Path, name: str) -> set[str]:
    """A module level set of strings of a Python file, read without importing it."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return set(ast.literal_eval(node.value))
    return set()


def first_error(output: str) -> str:
    """The last line of an example's output that names an error, else its last line."""
    lines = [line for line in output.splitlines() if line.strip()]
    named = (
        line
        for line in reversed(lines)
        if ("Error" in line.split(":")[0] or "Exception" in line.split(":")[0] or line.startswith("error:"))
    )
    return next(named, lines[-1] if lines else "")[:300]


@APP.command
def examples(target: str, label: str, per_example: int = 180) -> None:
    """Run every numbered example with the settings of `tests/test_examples.py` and print status, seconds and error.

    target: `KRN_TARGET` for the run, avx512 on the Mac and avx2 on the Intel and AMD hosts.
    label: the host name the rows are recorded under.
    per_example: seconds one example may take.
    """
    gate = ROOT / "tests" / "examples" / "test_examples.py"
    rights = literal_set(gate, "NEEDS_INSPECTION_RIGHTS")
    # the paths pytest adds from the `pythonpath` of prototype/pyproject.toml, which the test passes on through sys.path
    paths = [ROOT, ROOT / "src", ROOT / "tests" / "kernels", ROOT / "experiments" / "sketch"]
    pythonpath = os.pathsep.join([*map(str, paths), *sys.path])
    environment = os.environ | {"KRN_TARGET": target, "PYTHONPATH": pythonpath}
    rows: list[schema.ExampleRow] = []
    runnable = [
        *sorted((ROOT / "examples").glob("[0-9][0-9]_*/[0-9][0-9]_*.py")),
        *sorted((ROOT / "experiments" / "library_checks").glob("*.py")),
    ]
    for path in runnable:
        row: schema.ExampleRow = {"example": str(path.relative_to(ROOT)), "host": label, "status": ""}
        if path.name in rights and sys.platform == "darwin" and os.geteuid() != 0:
            row.update({"status": "skip", "reason": "macOS lets only root or an entitled process attach to another"})
        else:
            started = time.time()
            try:
                command = [sys.executable, str(path)]
                done = subprocess.run(
                    command, capture_output=True, text=True, env=environment, timeout=per_example, check=False
                )
                ok = done.returncode == 0
                row.update(
                    {"status": "pass" if ok else "fail", "error": "" if ok else first_error(done.stderr + done.stdout)}
                )
            except subprocess.TimeoutExpired:
                row.update({"status": "timeout", "error": f"no result in {per_example} s"})
            row["seconds"] = round(time.time() - started, 2)
        rows.append(row)
        LOG.info("%s %s", row["example"], row["status"])
    document: schema.ExamplesDocument = {"label": label, "target": target, "rows": rows}
    emit("examples", json.dumps(document, separators=(",", ":")))


def locked_remote(host: str, commands: list[list[str]], timeout: int) -> list[str]:
    """Run commands through mb on a host while holding the shared GPU lock, and release it afterwards."""
    while True:
        try:
            GPU_LOCK.mkdir()
            break
        except FileExistsError:
            LOG.info("waiting for %s", GPU_LOCK)
            time.sleep(60)
    try:
        stream(["mb", "host", "sync", host], 900)
        return [stream(["mb", "run", "--on", host, *command], timeout) for command in commands]
    finally:
        GPU_LOCK.rmdir()


@APP.command
def collect(host: str, skip_suite: bool = False, skip_examples: bool = False, timeout: int = 3300) -> None:
    """Run the suite and the examples on a host and keep the raw output under `/tmp/kl-conf`.

    host: mac (this machine, run directly), silo or crimson (through mb, under the GPU lock).
    skip_suite: leave the pytest run out.
    skip_examples: leave the example runs out.
    timeout: seconds for the suite.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    target = CPU_TARGET[host]
    commands = []
    if not skip_suite:
        commands.append(
            ["python", "-m", "experiments.conformance", "suite", target, host, "--timeout", str(timeout - 300)]
        )
    if not skip_examples:
        commands.append(["python", "-m", "experiments.conformance", "examples", target, host])
    if host == "mac":
        outputs = [stream(["mb", "run", "--no-install", *command], timeout) for command in commands]
    else:
        outputs = locked_remote(host, commands, timeout)
    for command, text in zip(commands, outputs, strict=True):
        name = command[2]
        (RAW / f"{host}.{name}.log").write_text(text)
        (RAW / f"{host}.{name}.json").write_text(between(text, name))
        LOG.info("kept %s", RAW / f"{host}.{name}.json")


def node_facts(text: str, junit: ET.Element) -> schema.Facts:
    """Facts of a PBS node from the `node.txt` its job wrote: host, CPU model, GPU line, Python version."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    host = next((line.split("=", 1)[1] for line in lines if line.startswith("host=")), "")
    cpu = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Model name")), "")
    gpu = next((line for line in lines if line.startswith("NVIDIA")), "")
    python = next((line.split()[1] for line in lines if line.startswith("Python ")), "")
    stamp = next(iter(junit.iter("testsuite")), junit).get("timestamp", "")
    return {
        "hostname": host,
        "platform": "Linux, PBS node of Miyabi-G",
        "machine": "aarch64",
        "cpu": cpu,
        "gpus": [gpu] if gpu else [],
        "python": python,
        "clang": "",
        "packages": {},
        "date": stamp,
    }


@APP.command
def ingest(host: str, junit: Path, node: Path, target: str = "avx512") -> None:
    """Turn the `--junitxml` file and `node.txt` of a suite run made elsewhere (a PBS job) into a raw suite file.

    host: the name to record the run under, gh200.
    junit: the junit file of the run.
    node: the `node.txt` the job wrote (host=, `Model name`, the GPU line, the Python version).
    target: the `KRN_TARGET` the run used; the conftest default is avx512.
    """
    RAW.mkdir(parents=True, exist_ok=True)
    root = ET.parse(junit).getroot()
    suite_node = next(iter(root.iter("testsuite")))
    document: schema.SuiteDocument = {
        "label": host,
        "seconds": float(suite_node.get("time", 0)),
        "target": target,
        "facts": node_facts(node.read_text(), root),
        "nodes": node_rows(read_junit(junit, host)),
    }
    (RAW / f"{host}.suite.json").write_text(json.dumps(document, separators=(",", ":")))
    LOG.info("kept %s: %s nodes", RAW / f"{host}.suite.json", len(document["nodes"]))


@APP.command
def keep(host: str, log: Path) -> None:
    """Keep the suite and examples documents found between markers in the output of a run made elsewhere.

    host: the name the run is recorded under.
    log: a file holding the output of `conformance.py suite` or `conformance.py examples` (a PBS job's log).
    """
    RAW.mkdir(parents=True, exist_ok=True)
    text = log.read_text()
    for name in ("suite", "examples"):
        if f"{BEGIN}{name}\n" in text:
            (RAW / f"{host}.{name}.json").write_text(between(text, name))
            LOG.info("kept %s", RAW / f"{host}.{name}.json")


def load_raw() -> tuple[dict[str, schema.HostFacts], list[Node], list[schema.ExampleRow]]:
    """The raw runs under `RAW`: facts by host, every node, every example row."""
    hosts: dict[str, schema.HostFacts] = {}
    nodes: list[Node] = []
    rows: list[schema.ExampleRow] = []
    for host in HOSTS:
        suite_file, examples_file = RAW / f"{host}.suite.json", RAW / f"{host}.examples.json"
        if suite_file.exists():
            suite_document: schema.SuiteDocument = json.loads(suite_file.read_text())
            hosts[host] = {
                **suite_document["facts"],
                "seconds": suite_document["seconds"],
                "target": suite_document["target"],
            }
            nodes += [
                Node(host, n["stem"], n["function"], n["parameter"], Outcome(n["outcome"]), n["seconds"], n["message"])
                for n in suite_document["nodes"]
            ]
        if examples_file.exists():
            examples_document: schema.ExamplesDocument = json.loads(examples_file.read_text())
            rows += examples_document["rows"]
            nodes += example_nodes(examples_document["rows"], host)
            if host in hosts:
                hosts[host]["examples_target"] = examples_document["target"]
    return hosts, nodes, rows


def skip_groups(matrix: Matrix) -> list[schema.Group]:
    """Skipped and failing cells grouped by target, class and reason, with the subjects behind each group."""
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for subject, cells in matrix.cells.items():
        for target, cell in cells.items():
            if cell.state is State.SKIP or cell.state is State.PARTIAL:
                kind, reason = cell.main_reason or (SkipClass.OTHER, "")
                groups[(target, kind.value, reason)].append(subject)
            elif cell.state is State.FAIL:
                groups[(target, "fail", cell.first_failure[:160])].append(subject)
    return [
        {
            "target": t,
            "class": k,
            "reason": r,
            "subjects": len(s),
            "kernel": sum(matrix.families[x] in KERNEL_FAMILIES for x in s),
            "sample": sorted(s)[:6],
        }
        for (t, k, r), s in sorted(groups.items(), key=lambda item: -len(item[1]))
    ]


def stem_outcomes(nodes: Iterable[Node]) -> dict[str, dict[str, dict[str, int]]]:
    """For each host and test file, the node outcomes; skips that leave a denominator are counted apart."""
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for node in nodes:
        if node.stem in ("test_examples", "examples_run"):
            continue
        leaves = node.outcome is Outcome.SKIP and node.skip_class in NOT_APPLICABLE
        counts[node.host][node.stem]["by_design" if leaves else node.outcome.value] += 1
    return {host: {stem: dict(counter) for stem, counter in sorted(per.items())} for host, per in counts.items()}


def host_skips(nodes: Iterable[Node]) -> dict[str, list[schema.HostSkip]]:
    """For each host, the skipped test nodes grouped by class and reason: what the host cannot verify."""
    counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for node in nodes:
        if node.outcome is Outcome.SKIP and node.stem != "test_examples":
            counts[node.host][(node.skip_class.value, short_reason(node.message))] += 1
    return {
        host: [{"class": kind, "reason": reason, "nodes": count} for (kind, reason), count in counter.most_common(30)]
        for host, counter in counts.items()
    }


def summarize(matrix: Matrix) -> dict[str, schema.TargetSummary]:
    """Pass rates per target: over every subject, over the kernel subjects, and per family."""
    sizes = Counter(matrix.families.values())
    kernels = sum(sizes[family] for family in KERNEL_FAMILIES)
    summary: dict[str, schema.TargetSummary] = {}
    for target in TARGETS:
        cells = {subject: row[target] for subject, row in matrix.cells.items() if target in row}
        summary[target] = {
            "overall": Percent.of(cells.values(), len(matrix.cells)),
            "kernels": Percent.of((c for s, c in cells.items() if matrix.families[s] in KERNEL_FAMILIES), kernels),
            "by_family": {
                family.value: Percent.of((c for s, c in cells.items() if matrix.families[s] is family), sizes[family])
                for family in Family
            },
        }
    return summary


@APP.command
def parse(
    output: Path | None = None, when: Annotated[str, cyclopts.Parameter(help="date of the file name")] = ""
) -> None:
    """Build `results/conformance_DATE.json` from the raw runs under `/tmp/kl-conf`.

    output: the file to write, `results/conformance_<date>.json` when unset.
    when: the date in the default name, today when unset.
    """
    hosts, nodes, rows = load_raw()
    matrix = Matrix()
    matrix.add(nodes)
    document: schema.Document = {
        "date": when or date.today().isoformat(),
        "hosts": hosts,
        "targets": list(TARGETS),
        "families": [family.value for family in Family],
        "summary": summarize(matrix),
        "node_outcomes": {host: dict(counter) for host, counter in matrix.node_counts.items()},
        "subjects": [
            {
                "id": subject,
                "family": matrix.families[subject].value,
                "named": subject in matrix.named,
                "cells": {t: c.as_json() for t, c in sorted(cells.items())},
            }
            for subject, cells in sorted(matrix.cells.items())
        ],
        "groups": skip_groups(matrix),
        "host_skips": host_skips(nodes),
        "stems": stem_outcomes(nodes),
        "examples": rows,
    }
    path = output or RESULTS / f"conformance_{document['date'].replace('-', '')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n")
    LOG.info("wrote %s: %s subjects", path, len(document["subjects"]))


def latest_result() -> Path:
    return max(RESULTS.glob("conformance_*.json"))


@APP.command
def report(source: Path | None = None, output: Path = NOTE) -> None:
    """Write the report note from the newest conformance file: numbers, skips, failures and requests.

    source: a `conformance_*.json`, the newest when unset.
    output: the note to write, `notes/round10/conformance-report.md` by default.
    """
    document: schema.Document = json.loads((source or latest_result()).read_text())
    output.write_text(conformance_report.render(document))
    LOG.info("wrote %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    APP()

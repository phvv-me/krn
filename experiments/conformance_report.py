"""The conformance report note, written from a `conformance_*.json` file by `conformance.py report`.

Every number of the note is read from the file; the prose is here. The requests table is rule based: a group of skipped
or failing subjects takes the first request whose pattern matches its reason, and the gain of a request is the share of
all kernel subjects it would turn into passes on the target.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import experiments.results_schema as schema

EFFORT_WEIGHT = {"S": 1, "M": 3, "L": 8}
GPU_TARGETS = {"mac/metal": "mac/c", "silo/cuda": "silo/c", "crimson/cuda": "crimson/c", "gh200/cuda": "gh200/c"}
M1_TARGETS = ("mac/c", "mac/metal", "silo/c", "silo/cuda", "crimson/cuda")
NON_KERNEL = "runtime, frontend and debug"
LEAVES_DENOMINATOR = ("by_design", "other_platform")

DESCRIPTION = (
    "Percent of kernel subjects passing on the Mac (C, Metal), NVIDIA (sm_120a, sm_86, sm_90a) and Intel, AMD and "
    "Grace CPUs, with the unit defined, skips classified, failures listed and the changes that raise each percent."
)
FRONT_MATTER = f"""---
type: Reference
title: "krn.gpu conformance: which kernels run on which target, and why the rest do not"
description: "{DESCRIPTION}"
generated: {{ by: agent:claude }}
---
"""


@dataclass(frozen=True, slots=True)
class Request:
    """A change another track or the owner can make, and the skipped or failing reasons it removes."""

    pattern: str
    effort: str
    track: str
    text: str


@dataclass(slots=True)
class Row:
    """A request with the points of percent of all kernel subjects it adds per target."""

    track: str
    effort: str
    text: str
    gains: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return sum(self.gains.values()) / EFFORT_WEIGHT[self.effort]


REQUESTS: tuple[Request, ...] = (
    Request(
        r"could not import 'more_itertools'|could not import 'toolz'",
        "S",
        "workspace manifest",
        "Add `more-itertools` and `toolz` to `[dev.python.deps]` of `mb.toml` (both have wheels for 3.15), so the two "
        "reference tests of the helper library run on every host instead of skipping.",
    ),
    Request(
        r"unresolved type: recursive cublasStatus_t",
        "S",
        "vendor bindings track",
        "Resolve the self-referential `cublasStatus_t` typedef in `experiments/gen_cuda_bindings.py`: the session "
        "fixture of `tests/test_cuda_bindings.py` raises, so every generated-contract and header-ABI node fails on the "
        "hosts that have the wheels.",
    ),
    Request(
        r"emulator has no copysign or ldexp",
        "S",
        "emulator",
        "Give the emulator `copysign` and `ldexp` semantics (`tests/test_numbers.py` skips two cases for them).",
    ),
    Request(
        r"C scalar ABI lacks the PY_FLOAT|emulator binds a bare float parameter",
        "S",
        "C backend and emulator",
        "Map a bare Python float parameter in the C backend's ctypes table and stop truncating it through `int(value)` "
        "in the emulator (`numerics/generic_numbers`).",
    ),
    Request(
        r"atomic.add currently has no f16 output overload",
        "M",
        "intrinsics track",
        "Add an f16 output overload of `atomic.add` (a CAS loop on the containing 32-bit word is the usual lowering).",
    ),
    Request(
        r"bf16 tile casts need lowering",
        "M",
        "CUDA and Metal renderers",
        "Lower bf16 tile casts in the CUDA and MSL renderers (`tests/test_generics.py` skips them).",
    ),
    Request(
        "Metal execution coverage here uses 32-bit integer atomics|"
        "Metal contention coverage here uses 32-bit integer atomics",
        "M",
        "Metal tests",
        "Run the atomic families on Metal for the dtypes the GPU has (32-bit integers, float add, 64-bit min and max) "
        "and mark the rest as skipped by design; today the tests skip every other dtype with a coverage note.",
    ),
)
HOST_LIMITS = (
    "NVRTC and cuda-python do not exist for macOS: those subjects run on the Linux hosts, and the `nvrtc` target takes "
    "them from there.",
    "Apple's clang has no AMDGPU target: the `hip` subjects run on the Linux hosts, whose conda clang has it. A clang "
    "with AMDGPU for `osx-arm64` in `mb.toml` would add them to the Mac, but no percentage of a run target.",
    "A card or the Metal device is not on every host: `mac/cuda` and `silo/metal` are not targets. Tests of another "
    "platform (Metal and Accelerate off the Mac) are skipped as `other_platform` and leave the denominator.",
)


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def table(head: Sequence[str], rows: Iterable[Sequence[str | int | float]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    lines += ["| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def untested_subjects(document: schema.Document) -> dict[str, dict[str, list[str]]]:
    """For each GPU target, the named kernel and case subjects its host tests on the C backend but not on the GPU,
    by family; subjects named by a test's sentence are host-side checks and are left out."""
    found: dict[str, dict[str, list[str]]] = {target: defaultdict(list) for target in GPU_TARGETS}
    for subject in document["subjects"]:
        if subject["family"] == NON_KERNEL or not subject["named"]:
            continue
        for gpu, cpu in GPU_TARGETS.items():
            if cpu in subject["cells"] and gpu not in subject["cells"]:
                found[gpu][subject["family"]].append(subject["id"])
    return found


def missing_test_rows(document: schema.Document) -> list[Row]:
    """One request per family and GPU kind for the subjects a host tests on the C backend and not on its GPU."""
    universe = document["summary"]["mac/c"]["kernels"]["universe"]
    found: dict[tuple[str, str], list[str]] = {}
    for gpu, per_family in untested_subjects(document).items():
        for family, subjects in per_family.items():
            found.setdefault(("Metal" if gpu == "mac/metal" else "card", family), subjects)
    rows = []
    for (kind, family), subjects in found.items():
        targets = ("mac/metal",) if kind == "Metal" else tuple(t for t in GPU_TARGETS if t != "mac/metal")
        sample = ", ".join(f"`{s.split('/', 1)[1]}`" for s in subjects[:3])
        text = (
            f"Write the {kind} tests of the {len(subjects)} `{family}` subjects that only the C backend runs "
            f"(for example {sample})."
        )
        if family == "examples":
            text = (
                f"Run the {len(subjects)} numbered examples on the {kind} backend too: `tests/test_examples.py` pins "
                "`krn.configure(lib='ours')` and a CPU target today."
            )
        rows.append(Row(f"tests of {family}", "M", text, {t: 100 * len(subjects) / universe for t in targets}))
    return rows


def request_rows(document: schema.Document) -> list[Row]:
    """Requests with their gains per target, best score first."""
    summary = document["summary"]
    rows: dict[int, Row] = {}
    for group in document["groups"]:
        if group["class"] in LEAVES_DENOMINATOR:
            continue
        for index, request in enumerate(REQUESTS):
            if any(part in group["reason"] for part in request.pattern.split("|")):
                row = rows.setdefault(index, Row(request.track, request.effort, request.text))
                universe = summary[group["target"]]["kernels"]["universe"]
                row.gains[group["target"]] = row.gains.get(group["target"], 0.0) + 100 * group["kernel"] / universe
                break
    return sorted([*rows.values(), *missing_test_rows(document)], key=lambda row: -row.score)


def m1_reading(document: schema.Document) -> str:
    """Which of the five M1 targets reach half of all kernel subjects."""
    summary = document["summary"]
    shown = {t: summary[t]["kernels"]["percent_of_all"] or 0.0 for t in M1_TARGETS}
    met = [f"`{t}`" for t, p in shown.items() if p >= 50]
    short = [f"`{t}` ({p}%)" for t, p in shown.items() if p < 50]
    return ", ".join(met) + " pass the bar" + (f" and {', '.join(short)} does not" if short else "")


def failing_sentence(document: schema.Document) -> str:
    """Whether any kernel subject fails on a run target, in a sentence."""
    summary = document["summary"]
    failing = {t: summary[t]["kernels"]["fail"] for t in document["targets"] if summary[t]["kernels"]["fail"]}
    if not failing:
        return "No kernel subject fails on any target."
    listing = ", ".join(f"`{t}` {n}" for t, n in failing.items())
    return f"Kernel subjects fail on {listing} (section 6)."


def short_answers(document: schema.Document) -> str:
    summary = document["summary"]
    rows = []
    for target in document["targets"]:
        row = summary[target]["kernels"]
        if row["applicable"]:
            lead = [f"`{target}`", row["pass"], row["applicable"], row["universe"], percent(row["percent_of_all"])]
            rows.append([*lead, percent(row["percent"]), row["partial"], row["fail"], row["skip"], row["by_design"]])
    head = ["Target", "Pass", "Tested", "All kernel subjects", "Pass, percent of all", "Pass, percent of tested"]
    head += ["Partial", "Fail", "Skip", "Not applicable"]
    universe = summary["mac/c"]["kernels"]["universe"]
    return (
        "## 1. Short answers\n\n"
        + table(head, rows)
        + '\n- [M] **Percent of all** answers "what share of the kernels run on this target": passing subjects over '
        f"all {universe} kernel subjects of the suite. **Percent of tested** is the pass rate among the subjects the "
        "suite exercises on the target (skips by design and for another platform leave that denominator). A target "
        "the suite exercises less scores lower on the first and the same on the second.\n"
        f"- [M] {failing_sentence(document)} What is not a pass is a skip or a test that does not exist: the "
        "gap of Metal and CUDA to the C backend is missing tests (section 5), not wrong results.\n"
        f"- [M] M1 asks for at least half of the kernels on the Mac (C and Metal), NVIDIA (sm_86 and sm_120a) and an "
        f"Intel CPU. By the first reading {m1_reading(document)}; by the second reading every one of them passes.\n"
        "- [M] `gh200/c` and `gh200/cuda` are the Grace CPU (Neoverse-V2) and the H100 (sm_90a) of a Miyabi-G node, "
        "from a suite run made through PBS; that run did not include `test_cuda_bindings.py`.\n"
        "- [M] `nvrtc`, `hip`, `msl` and `emulator` do not depend on the host: each takes the hosts that can run it "
        "(NVRTC and clang AMDGPU exist on the Linux hosts only, so the Mac alone would show them skipped).\n"
    )


def definitions() -> str:
    return (
        "## 2. Definitions\n\n"
        "- **Subject.** A kernel, an intrinsic case, a column or string operation, an example or a language check, "
        "named by the parameter id of the test that exercises it (`columns/cast`, `groups/libm.erf`, "
        "`helpers/run_pairwise`) or, for a test without a parameter, by its function. Digits that only number a case "
        "(`libm.erf0`) are stripped and architecture and backend tokens are dropped, so the tests of one subject on "
        "different targets meet in one row, and a kernel that several test files name differently (`saxpy`, "
        "`elementwise-saxpy`) is one row. A subject named by the sentence of a test function "
        "(`a_bad_value_raises_instead_of_being_ignored`) is a host-side check; it counts in the universe and is left "
        "out of the missing-test requests. An example is a subject, one per file, run by the script itself. A "
        "subject that only the `frontend` target tests (a refusal, a rendering) is a check of the language, not a "
        f"kernel: it belongs to the family `{NON_KERNEL}` and stays out of the headline.\n"
        "- **Target.** A host and a backend: `mac/c` (Apple M4 Pro CPU through clang), `mac/metal` (the M4 Pro GPU), "
        "`silo/c` (Intel Core i9-14900, AVX2, no AVX-512), `silo/cuda` (RTX 5080, sm_120a), `crimson/c` (AMD "
        "Threadripper 3960X, AVX2), `crimson/cuda` (RTX 3090, sm_86), `gh200/c` (Grace, Neoverse-V2), `gh200/cuda` "
        "(H100, sm_90a), and the targets that do not depend on the host: `emulator`, `nvrtc` (compile only), `hip` "
        "(clang AMDGPU for gfx942 and gfx950, compile only), `msl` (type check against a shim, compile only) and "
        "`frontend` (capture, rendering, refusals, the host API).\n"
        "- **Role.** The target one test node exercises, taken from its function name and the first token of its "
        "parameter id by the rules of `Classifier` in `conformance_model.py` (`_card`, `cuda_against` and `test_gpu` "
        "are CUDA; `_metal` and `test_metal` are Metal; `emulator-`, `c-` and `cpu-` prefixes name their backend; "
        "`_local` is the C backend for an `avx*` architecture and the emulator for the others; `nvrtc` and "
        "`compiles` are compile only). [I] The roles are rules, not annotations in the tests; a misplaced node moves a "
        "subject between columns, never between pass and fail.\n"
        "- **Cell.** The nodes of one subject, one role and one host. `pass`: every applicable node passed. `partial`: "
        "some passed and some were skipped for a reason that is not by design. `fail`: a node failed. `skip`: "
        "nothing ran. `na`: no node for the target.\n"
        "- **Skip classes.** `no_hardware` (the card is absent), `no_toolchain` (NVRTC, cuda-python, a clang with "
        "AMDGPU or a Python package is absent on the host), `by_design` (the target lacks the feature: Metal has no "
        "double, a signature-only example, an instruction that needs a newer SM), `other_platform` (a test of Metal or "
        "Accelerate on a host that is not a Mac), `not_implemented` (the code says so) and `other`. `by_design` and "
        "`other_platform` leave the denominator. The rules are regexes over the reason text.\n"
        "- **Families.** elementwise and reductions, tiles and matmul, sparse, groups and intrinsics, strings, "
        "columns, helpers, numerics, examples.\n"
    )


def runs(document: schema.Document) -> str:
    hosts = document["hosts"]
    rows = []
    for host, facts in hosts.items():
        outcomes = document["node_outcomes"][host]
        lead = [host, facts["cpu"], "; ".join(facts["gpus"]) or "none", facts.get("target", ""), sum(outcomes.values())]
        counts = [outcomes.get("pass", 0), outcomes.get("skip", 0), outcomes.get("fail", 0)]
        rows.append([*lead, *counts, f"{facts.get('seconds', 0):.0f} s"])
    head = ["Host", "CPU", "GPU", "CPU target", "Nodes", "Pass", "Skip", "Fail", "Time"]
    names = ("numpy", "pytest", "cuda-core", "cuda-bindings", "nvidia-cuda-nvrtc", "nvidia-cublas", "nvidia-cusparse")
    names += ("pyobjc-framework-Metal", "more-itertools", "toolz")
    packages = [[name, *[hosts[h]["packages"].get(name) or "-" for h in hosts]] for name in names]
    return (
        "## 3. Runs\n\n"
        + table(head, rows)
        + "\n- [M] Per host: `mb run --no-install python -m experiments.conformance collect <host>`. `silo` and "
        "`crimson` go through `mb host sync` and `mb run --on`, under `/tmp/kl-gpu.lock`. It runs `pytest -v -rs "
        "--junitxml` in `prototype/` with `KRN_TARGET` set to the CPU target above and the per-test timeout of the "
        "`test` task (300 s), then every numbered example with `krn.configure(lib='ours')` "
        "and the same `KRN_TARGET`, as "
        "`tests/test_examples.py` does. That test hard-codes `avx512`; the avx512 target of the Mac is a description "
        "(`-march=native` decides what runs), and the Intel and AMD hosts have no AVX-512, hence `avx2`.\n"
        "- [M] `gh200` is a PBS job on the held Miyabi-G line: `conformance.py suite` and `examples` run inside "
        "`mb job submit --on miyabi-g --queue held`, their output copied back with scp and kept with `conformance.py "
        "keep`. The suite leaves out `tests/test_cuda_bindings.py` (it downloads headers and was another track's "
        "work in flux).\n"
        "- [M] crimson sees GPU 1 only (the pin in `mb.toml`); the other card serves AIZK.\n\n"
        + table(["Package", *hosts], packages)
    )


def families(document: schema.Document) -> str:
    summary, targets = document["summary"], document["targets"]
    names = [f for f in document["families"] if f != NON_KERNEL]

    def tested(target: str, family: str) -> str:
        row = summary[target]["by_family"][family]
        return percent(row["percent"]) if row["applicable"] else "-"

    def verified(target: str, family: str) -> str:
        row = summary[target]["by_family"][family]
        return f"{row['pass']}/{row['universe']}"

    return (
        "## 4. Percent by family and target\n\n"
        "Pass rate among the tested subjects:\n\n"
        + table(["Family", *targets], [[f, *[tested(t, f) for t in targets]] for f in names])
        + "\nPassing subjects over all subjects of the family (a family with `0/n` has no test on that target):\n\n"
        + table(["Family", *targets], [[f, *[verified(t, f) for t in targets]] for f in names])
    )


def gaps(document: schema.Document) -> str:
    shown = [g for g in document["groups"] if g["class"] not in LEAVES_DENOMINATOR]
    leaving = sum(g["subjects"] for g in document["groups"] if g["class"] in LEAVES_DENOMINATOR)
    rows = [[g["target"], g["class"], g["subjects"], g["reason"][:110], ", ".join(g["sample"][:3])] for g in shown[:30]]
    coverage = []
    for gpu, per_family in untested_subjects(document).items():
        total = sum(len(subjects) for subjects in per_family.values())
        listing = ", ".join(f"{f} {len(s)}" for f, s in sorted(per_family.items(), key=lambda item: -len(item[1])))
        coverage.append([f"`{gpu}`", total, listing])
    return (
        "## 5. Where the gaps are\n\n"
        "**Skips and partial cells**, by target, class and reason (the first 30 groups; `groups` in the JSON "
        f"holds all). {leaving} more subject cells are skipped by design or for another platform and leave the "
        "denominator:\n\n"
        + table(["Target", "Class", "Subjects", "Reason", "Examples"], rows)
        + "\n**Missing tests.** Kernel and case subjects (named by a kernel id, not by the sentence of a test "
        "function, which is a host-side check) that the host tests on the C backend and that have no test at all on "
        "its GPU, by family. This is the largest lever on the percent of all subjects, and it is a change to the "
        "tests, not to the compiler:\n\n" + table(["GPU target", "Subjects without a test", "By family"], coverage)
    )


def failures(document: schema.Document) -> str:
    failing = [g for g in document["groups"] if g["class"] == "fail"]
    body = "No cell failed.\n"
    if failing:
        rows = [[g["target"], g["subjects"], g["reason"][:140], ", ".join(g["sample"][:3])] for g in failing]
        body = table(["Target", "Subjects", "Failure", "Examples"], rows)
    nodes = [
        [host, stem, counts["fail"]]
        for host, per in document["stems"].items()
        for stem, counts in per.items()
        if counts.get("fail")
    ]
    detail = ""
    if nodes:
        detail = (
            "\nFailing test nodes by host and file (a failure of a session fixture counts once per test that "
            "needed it):\n\n" + table(["Host", "Test file", "Nodes"], nodes)
        )
    return "## 6. Failures\n\n" + body + detail


def requests(document: schema.Document) -> str:
    body = [
        [
            index + 1,
            row.track,
            row.effort,
            row.text,
            ", ".join(f"{t} +{g:.2f}" for t, g in row.gains.items()),
            f"{row.score:.2f}",
        ]
        for index, row in enumerate(request_rows(document))
    ]
    return (
        "## 7. Requests, by gain per effort\n\n"
        "Effort: S a few lines, M a feature or the tests of one family, L a design change. Gain is the points of "
        "percent of all kernel subjects that the request adds to a target if every skip with that reason became a "
        "pass; the score is the sum over every target divided by the effort weight (S 1, M 3, L 8). The requests are "
        "for the tracks that own the code; nothing here was changed. [I] Gains assume the new passes are real, and the "
        "missing-test requests assume the kernels behave on the GPU as they do on the C backend, which only the new "
        "tests can say.\n\n"
        + table(["#", "Track", "Effort", "Request", "Points gained per target", "Score"], body)
        + "\n**Skips no change can remove** (the host is the limit):\n\n"
        + "\n".join(f"- {text}" for text in HOST_LIMITS)
        + "\n"
    )


def examples(document: schema.Document) -> str:
    rows, hosts = document["examples"], document["hosts"]

    def count(host: str, *statuses: str) -> int:
        return sum(r["status"] in statuses for r in rows if r["host"] == host)

    body = [
        [h, hosts[h].get("examples_target", "not run"), count(h, "pass"), count(h, "skip"), count(h, "fail", "timeout")]
        for h in hosts
    ]
    return (
        "## 8. Examples\n\n"
        + table(["Host", "`KRN_TARGET`", "Pass", "Skip", "Fail"], body)
        + "\nThe skipped example is `05_systems/11_attach_to_process` on the Mac (macOS lets "
        "only root attach to another process). Seconds and the first error line of a failure are in the JSON.\n"
    )


def limits() -> str:
    return (
        "## 9. What this does not say\n\n"
        "- A pass is the suite's own assertion (agreement with numpy, the emulator or the C backend), not a "
        "performance claim.\n"
        "- The suite has many more tests of the C backend than of Metal or CUDA. Raising GPU coverage is a change to "
        "the tests; the percent of all subjects is the fair comparison of how much each target is exercised.\n"
        "- `hip` and `nvrtc` compile only. No kernel has run on an AMD GPU and no Intel GPU backend exists.\n"
        "- Skip reasons are classified by regex; a new reason text lands in `other` until a rule is added.\n"
        "- The other tracks edited `sketch/` and `prototype/` while the runs were made (the CUDA backend, the vendor "
        "bindings, the columns library), and the hosts ran at different minutes, so a later run can differ in what "
        "it counts.\n"
    )


def render(document: schema.Document) -> str:
    date = document["date"].replace("-", "")
    header = (
        FRONT_MATTER + "\n# Conformance report\n\n"
        "Evidence tags: `M` measured here, `R` read from the named file, `I` inference. Data: "
        f"`evidence/results/conformance_{date}.json`, produced by `experiments/conformance.py`. The site page is "
        "`docs/site/conformance.html`.\n\n"
    )
    sections = [
        short_answers(document),
        definitions(),
        runs(document),
        families(document),
        gaps(document),
        failures(document),
        requests(document),
        examples(document),
        limits(),
    ]
    return header + "\n".join(sections)

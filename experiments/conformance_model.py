"""The model of the conformance matrix: nodes, roles, subjects, cells and pass rates. No collection, no files.

A test NODE of the prototype suite has one ROLE, the target it exercises, from its function name and parameter id. A
SUBJECT is a kernel, an intrinsic, a column operation, an example or a language check, named by the parameter id of the
test that exercises it. A CELL aggregates the nodes of one subject, one role and one host.
"""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path

import experiments.results_schema as schema


class Outcome(StrEnum):
    PASS = auto()
    FAIL = auto()
    SKIP = auto()
    XFAIL = auto()


class SkipClass(StrEnum):
    NO_HARDWARE = auto()
    NO_TOOLCHAIN = auto()
    BY_DESIGN = auto()
    OTHER_PLATFORM = auto()
    NOT_IMPLEMENTED = auto()
    OTHER = auto()


# classes of skip that leave the denominator: the target lacks the feature, or the test is for another platform
NOT_APPLICABLE = (SkipClass.BY_DESIGN, SkipClass.OTHER_PLATFORM)


class State(StrEnum):
    PASS = auto()
    PARTIAL = auto()
    FAIL = auto()
    SKIP = auto()
    NA = auto()


class Role(StrEnum):
    C = auto()
    CUDA = auto()
    METAL = auto()
    EMULATOR = auto()
    NVRTC = auto()
    HIP = auto()
    MSL = auto()
    FRONTEND = auto()


class Family(StrEnum):
    ELEMENTWISE = "elementwise and reductions"
    TILES = "tiles and matmul"
    SPARSE = "sparse"
    GROUPS = "groups and intrinsics"
    STRINGS = "strings"
    COLUMNS = "columns"
    HELPERS = "helpers"
    NUMERICS = "numerics"
    EXAMPLES = "examples"
    RUNTIME = "runtime, frontend and debug"


KERNEL_FAMILIES = tuple(family for family in Family if family is not Family.RUNTIME)
HOSTS = ("mac", "silo", "crimson", "gh200")
RUN_TARGETS = ("mac/c", "mac/metal", "silo/c", "silo/cuda", "crimson/c", "crimson/cuda", "gh200/c", "gh200/cuda")
SHARED_TARGETS = ("emulator", "nvrtc", "hip", "msl", "frontend")
TARGETS = RUN_TARGETS + SHARED_TARGETS
HOST_ROLE_TARGET = {
    ("mac", Role.C): "mac/c",
    ("mac", Role.METAL): "mac/metal",
    ("silo", Role.C): "silo/c",
    ("silo", Role.CUDA): "silo/cuda",
    ("crimson", Role.C): "crimson/c",
    ("crimson", Role.CUDA): "crimson/cuda",
    ("gh200", Role.C): "gh200/c",
    ("gh200", Role.CUDA): "gh200/cuda",
}
SHARED_ROLES = {role: role.value for role in (Role.EMULATOR, Role.NVRTC, Role.HIP, Role.MSL, Role.FRONTEND)}

SKIP_RULES: tuple[tuple[str, SkipClass], ...] = (
    (r"no CUDA card|libcuda|no card|no NVIDIA", SkipClass.NO_HARDWARE),
    (r"no Metal|MTLCreateSystemDefaultDevice|Metal device|needs a Mac|Apple GPU", SkipClass.NO_HARDWARE),
    (
        r"No module named 'Metal'|Metal runtime is unavailable|no Metal|No Metal|macOS framework|not a macOS"
        r"|macOS only|only on Apple",
        SkipClass.OTHER_PLATFORM,
    ),
    (
        r"Metal has no|has no double|signatures only|has its own atomic SpMV|needs a newer SM"
        r"|dedicated matrix cases|does not apply|documented only for",
        SkipClass.BY_DESIGN,
    ),
    (
        r"NVRTC|nvrtc|cuda-core|cuda-bindings|cuda-python|no AMDGPU|could not import|not installed|unavailable"
        r"|no cuSPARSE|no cuBLAS",
        SkipClass.NO_TOOLCHAIN,
    ),
    (
        r"need lowering|currently has no|lacks the|has no .* semantics yet|through int\(|coverage here"
        r"|not implemented|yet",
        SkipClass.NOT_IMPLEMENTED,
    ),
)
UNIT_STEMS = {"test_cuda", "test_hip", "test_msl"}
BY_FUNCTION_STEMS = {
    "test_generics", "test_lib", "test_lib_vendors", "test_lib_sparse", "test_runtime", "test_cuda_bindings",
    "test_nvmath_bindings", "test_gen_cuda_bindings",
    "test_capture", "test_surface", "test_debug", "test_cuda_backend", "test_cuda_gdb",
}  # fmt: skip
ROLE_PREFIX = {
    "emulator": Role.EMULATOR,
    "c": Role.C,
    "cpu": Role.C,
    "metal": Role.METAL,
    "cuda": Role.CUDA,
    "card": Role.CUDA,
}


def cpu_arch(token: str) -> bool:
    return token == "spr" or (token.startswith("avx") and (not token[3:] or token[3:].isdigit()))


def architecture(token: str) -> bool:
    return cpu_arch(token) or any(
        token.startswith(prefix)
        and (token[len(prefix) :].removesuffix("a") if prefix == "sm_" else token[len(prefix) :]).isdigit()
        for prefix in ("sm_", "gfx", "apple_m")
    )


STEM_ROLE = {
    "test_gpu": Role.CUDA,
    "test_cuda_backend": Role.CUDA,
    "test_cuda_gdb": Role.CUDA,
    "test_metal": Role.METAL,
    "test_hip": Role.HIP,
    "test_msl": Role.MSL,
    "test_emulate": Role.EMULATOR,
}
STEM_DEFAULT_ROLE = {
    "test_cpu": Role.C,
    "test_runtime": Role.C,
    "test_ops": Role.C,
    "test_validity": Role.C,
    "test_intrinsics": Role.C,
    "test_lib": Role.C,
    "test_lib_sparse": Role.C,
    "test_lib_vendors": Role.C,
    "examples_run": Role.C,
}


def name_role(function: str, stem: str) -> Role | None:
    if "hip" in function:
        return Role.HIP
    if any(part in function for part in ("nvrtc", "header_free_and_compiles", "_compiles_")) or function.endswith(
        "_compiles"
    ):
        return Role.NVRTC
    if "msl_shapes" in function:
        return Role.MSL
    if stem != "test_cuda" and (
        "card" in function.split("_")
        or function.endswith("_cuda")
        or any(
            part in function for part in ("cuda_against", "cublaslt_multiplies", "cusparse_multiplies", "cusparse_on")
        )
        or function == "family_cuda"
    ):
        return Role.CUDA
    if (
        function == "metal"
        or function.endswith("_metal")
        or any(
            part in function
            for part in ("metal_gpu", "metal_against", "on_the_gpu", "mps_multiplies", "on_the_metal", "simdgroup")
        )
    ):
        return Role.METAL
    if (
        function == "c"
        or function.endswith("_c")
        or "metal_packed_algorithm_on_c" in function
        or "cg_c_" in function
        or any(
            part == "c" and not following.startswith("against")
            for part, following in zip(function.split("_"), function.split("_")[1:], strict=False)
        )
    ):
        return Role.C
    if (
        function.endswith("_emulator")
        or function.startswith(("emulator", "cg_emulator"))
        or any(part in function for part in ("emulator_against", "wave64_emulation"))
    ):
        return Role.EMULATOR
    return None


SPARSE_NAMES = ("spmv", "spmm", "csr", "ell", "bsr", "coo", "sparse", "sddmm", "spgemm", "cusparse")
TILE_NAMES = ("gemm", "tile", "matmul", "epilogue", "fp16_gemm", "mma")


def classify_skip(reason: str) -> SkipClass:
    """The class of a skip reason; the first matching rule wins and anything unmatched is `other`."""
    for pattern, kind in SKIP_RULES:
        if any(part.replace("\\(", "(") in reason for part in pattern.split("|")) or (
            "has no .* semantics yet" in pattern and "has no " in reason and " semantics yet" in reason
        ):
            return kind
    return SkipClass.OTHER


def short_reason(text: str) -> str:
    """A skip reason without the file position and without the dlopen search paths."""
    text = text.strip()
    file, separator, rest = text.partition(":")
    number, colon, message = rest.partition(": ")
    if separator and colon and number.isdigit() and all(c.isalnum() or c in "_/. -" for c in file):
        text = message
    before, dlopen, _ = text.partition("dlopen(")
    if dlopen:
        text = before + "no libcuda.so.1 on this host"
    text = text.partition("(no such file")[0]
    return text.splitlines()[0][:200] if text else "no reason given"


@dataclass(frozen=True, slots=True)
class Node:
    """One test node of one host's run."""

    host: str
    stem: str
    function: str
    parameter: str
    outcome: Outcome
    seconds: float
    message: str

    @property
    def tokens(self) -> list[str]:
        return self.parameter.split("-") if self.parameter else []

    @property
    def skip_class(self) -> SkipClass:
        return classify_skip(self.message)


@dataclass(frozen=True, slots=True)
class Placement:
    """Where a node lands in the matrix: the subject's name, its family, whether a rule chose the family (`sure`) or
    the default did, and the role."""

    name: str
    named: bool
    family: Family
    sure: bool
    role: Role

    def subject(self) -> str:
        return f"{self.family.name.lower()}/{self.name}"


class Classifier:
    """Map a node to its subject, family and role. The rules are the tables above; nothing here reads the tests."""

    def place(self, node: Node) -> Placement:
        name, named = self._name(node)
        family, sure = self._family(node, name)
        return Placement(name, named, family, sure, self._role(node))

    def _role(self, node: Node) -> Role:
        function = node.function.removeprefix("test_")
        tokens = node.tokens
        if node.stem in STEM_ROLE:
            return STEM_ROLE[node.stem]
        if tokens and tokens[0] in ROLE_PREFIX and node.stem not in UNIT_STEMS:
            return ROLE_PREFIX[tokens[0]]
        if function.endswith("_local"):
            return Role.C if any(cpu_arch(token) for token in tokens) else Role.EMULATOR
        if function == "render" and any(token.startswith("sm_") for token in tokens):
            return Role.NVRTC  # the unit rendered for an NVIDIA architecture is also compiled by NVRTC
        if role := name_role(function, node.stem):
            return role
        if node.stem in UNIT_STEMS:
            return Role.NVRTC if node.stem == "test_cuda" else Role.MSL
        return STEM_DEFAULT_ROLE.get(node.stem, Role.FRONTEND)

    def _name(self, node: Node) -> tuple[str, bool]:
        """The subject's name and whether it is the id of a kernel or case (else the sentence of a test function)."""
        function = node.function.removeprefix("test_")
        tokens = [token for token in node.tokens if not architecture(token) and token not in ROLE_PREFIX]
        if tokens and (node.stem in UNIT_STEMS or (node.stem == "test_capture" and "examples_capture" in function)):
            return next((token for token in tokens[1:2] if token), tokens[0]), True  # the kernel of `group-kernel`
        # an id that is a message (a refusal text, a source line) or a bare number names no subject: the function does
        named = (
            tokens
            and tokens[0]
            and all(c.isalnum() or c in "_.-/" for c in node.parameter)
            and not (tokens[0] in {"True", "False"} or all(c.isdigit() or c in "._" for c in tokens[0]))
        )
        by_function = node.stem in BY_FUNCTION_STEMS
        if named and not by_function:
            name = tokens[0]
            if node.stem in {"test_intrinsic_families", "test_strings"} or "." in name:
                trimmed = name.rstrip("0123456789")
                if trimmed and (trimmed[-1].islower() or trimmed[-1] == "_"):
                    name = trimmed
            return name, True
        for suffix in ("_on_the_metal_gpu", "_on_the_card", "_on_the_gpu", "_metal_gpu", "_card", "_gpu"):
            if function.endswith(suffix):
                function = function.removesuffix(suffix)
                break
        function = (
            function.replace("simdgroup", "warp").replace("the_card", "the_device").replace("the_gpu", "the_device")
        )
        function = function.removeprefix("the_") if function.startswith("the_metal_gpu") else function
        function = function.replace("metal_gpu", "device")
        return function.removesuffix("_against_numpy_and_emulator").removesuffix("_against_numpy"), False

    def _family(self, node: Node, name: str) -> tuple[Family, bool]:
        """The family of a node and whether a rule chose it; the default, elementwise, is not a choice."""
        stem = node.stem
        by_stem = {
            "test_tools": Family.HELPERS,
            "test_strings": Family.STRINGS,
            "test_columns": Family.COLUMNS,
            "test_numbers": Family.NUMERICS,
            "test_iteration": Family.NUMERICS,
            "test_groups": Family.GROUPS,
            "test_intrinsic_families": Family.GROUPS,
            "test_intrinsics": Family.GROUPS,
            "test_sparse": Family.SPARSE,
            "test_generics": Family.SPARSE,
            "test_lib_sparse": Family.SPARSE,
            "test_lib": Family.TILES,
            "test_lib_vendors": Family.TILES,
            "examples_run": Family.EXAMPLES,
            "test_capture": Family.RUNTIME,
            "test_surface": Family.RUNTIME,
            "test_debug": Family.RUNTIME,
            "test_runtime": Family.RUNTIME,
            "test_cuda_backend": Family.RUNTIME,
            "test_cuda_bindings": Family.RUNTIME,
            "test_nvmath_bindings": Family.RUNTIME,
            "test_gen_cuda_bindings": Family.RUNTIME,
            "test_cuda_gdb": Family.RUNTIME,
        }
        if stem in by_stem and not (stem == "test_capture" and "examples_capture" in node.function):
            return by_stem[stem], True
        group = node.tokens[0] if node.tokens and (stem in UNIT_STEMS or stem == "test_capture") else ""
        if group == "intrinsics":
            return Family.GROUPS, True
        if group in ("elementwise", "extra", "ragged", "automaton", "softmax"):
            return Family.ELEMENTWISE, True
        if group == "gemm" or any(part in f"{name} {node.function}".lower() for part in TILE_NAMES):
            return Family.TILES, True
        if any(part in name.lower() for part in SPARSE_NAMES):
            return Family.SPARSE, True
        return Family.ELEMENTWISE, False


@dataclass(slots=True)
class Cell:
    """The aggregate of the nodes of one subject, one role, one host."""

    counts: Counter[str] = field(default_factory=Counter)
    skips: Counter[SkipClass] = field(default_factory=Counter)
    reasons: Counter[tuple[str, str]] = field(default_factory=Counter)
    seconds: float = 0.0
    first_failure: str = ""
    host: str = ""

    def add(self, node: Node) -> None:
        self.seconds += node.seconds
        self.host = node.host
        match node.outcome:
            case Outcome.PASS | Outcome.XFAIL:
                self.counts["pass"] += 1
            case Outcome.FAIL:
                self.counts["fail"] += 1
                self.first_failure = self.first_failure or node.message
            case Outcome.SKIP:
                kind = node.skip_class
                self.counts["skip"] += 1
                self.skips[kind] += 1
                self.reasons[(kind.value, short_reason(node.message))] += 1

    @property
    def applicable_skips(self) -> int:
        return sum(count for kind, count in self.skips.items() if kind not in NOT_APPLICABLE)

    @property
    def state(self) -> State:
        passed, failed = self.counts["pass"], self.counts["fail"]
        if failed:
            return State.FAIL
        if passed and not self.applicable_skips:
            return State.PASS
        if passed:
            return State.PARTIAL
        return State.SKIP if self.counts["skip"] else State.NA

    @property
    def main_reason(self) -> tuple[SkipClass, str] | None:
        """The most common skip of the cell, the ones that are not by design first."""
        ranked = sorted(self.reasons.items(), key=lambda item: (SkipClass(item[0][0]) in NOT_APPLICABLE, -item[1]))
        return (SkipClass(ranked[0][0][0]), ranked[0][0][1]) if ranked else None

    @property
    def dominant_class(self) -> SkipClass | None:
        reason = self.main_reason
        return reason[0] if reason else None

    def as_json(self) -> schema.Cell:
        main = self.main_reason
        return {
            "state": self.state.value,
            "pass": self.counts["pass"],
            "fail": self.counts["fail"],
            "skip": {kind.value: count for kind, count in self.skips.items()},
            "class": main[0].value if main else "",
            "reason": main[1] if main else "",
            "failure": self.first_failure[:240],
            "seconds": round(self.seconds, 3),
        }

    @classmethod
    def merge(cls, cells: Sequence[Cell]) -> Cell:
        merged = cls()
        for cell in cells:
            merged.counts.update(cell.counts)
            merged.skips.update(cell.skips)
            merged.reasons.update(cell.reasons)
            merged.seconds += cell.seconds
            merged.first_failure = merged.first_failure or cell.first_failure
        return merged


def shared_state(cells: Sequence[Cell]) -> Cell:
    """A host independent target runs on every host that can: the cell is the best host's, or fails on any failure."""
    if failing := [cell for cell in cells if cell.state is State.FAIL]:
        return Cell.merge(failing)
    for wanted in (State.PASS, State.PARTIAL):
        if picked := [cell for cell in cells if cell.state is wanted]:
            return max(picked, key=lambda cell: cell.counts["pass"])
    return max(cells, key=lambda cell: cell.counts["skip"])


def read_junit(path: Path, host: str) -> list[Node]:
    """The nodes of a pytest `--junitxml` file."""
    nodes = []
    for case in ET.parse(path).getroot().iter("testcase"):
        if not case.get("classname"):
            continue  # a test file skipped at collection (a Metal module on Linux) has no tests to record
        stem = case.get("classname", "").rsplit(".", 1)[-1]
        name = case.get("name", "")
        function, _, rest = name.partition("[")
        outcome, message = Outcome.PASS, ""
        for child in case:
            if child.tag in ("failure", "error"):
                outcome, message = Outcome.FAIL, (child.get("message") or child.text or "").splitlines()[0][:300]
            elif child.tag == "skipped":
                xfail = child.get("type") == "pytest.xfail"
                outcome = Outcome.XFAIL if xfail else Outcome.SKIP
                message = child.get("message") or (child.text or "")
        nodes.append(Node(host, stem, function, rest.removesuffix("]"), outcome, float(case.get("time", 0)), message))
    return nodes


def example_nodes(rows: Iterable[schema.ExampleRow], host: str) -> list[Node]:
    """The direct runs of the examples as nodes of the examples family, one subject per example file."""
    outcomes = {"pass": Outcome.PASS, "fail": Outcome.FAIL, "timeout": Outcome.FAIL, "skip": Outcome.SKIP}
    return [
        Node(
            host,
            "examples_run",
            "test_run",
            Path(row["example"]).stem,
            outcomes[row["status"]],
            row.get("seconds", 0.0),
            row.get("error") or row.get("reason") or "",
        )
        for row in rows
    ]


@dataclass(slots=True)
class Matrix:
    """Subjects by targets, built from the nodes of every host."""

    cells: dict[str, dict[str, Cell]] = field(default_factory=lambda: defaultdict(dict))
    families: dict[str, Family] = field(default_factory=dict)
    named: set[str] = field(default_factory=set)  # subjects named by the id of a kernel or case, not by a sentence
    node_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    classifier: Classifier = field(default_factory=Classifier)

    def add(self, nodes: Iterable[Node]) -> None:
        placed = [(node, self.classifier.place(node)) for node in nodes if node.stem != "test_examples"]
        chosen: dict[str, set[Family]] = defaultdict(set)
        for _, place in placed:
            if place.sure:
                chosen[place.name].add(place.family)
        per_cell: dict[tuple[str, str, str], Cell] = defaultdict(Cell)
        for node, place in placed:
            # a node that only the default placed joins the family a rule chose for the same name, when it is one
            sure = chosen[place.name]
            family = next(iter(sure)) if not place.sure and len(sure) == 1 else place.family
            subject = f"{family.name.lower()}/{place.name}"
            self.families[subject] = family
            if place.named:
                self.named.add(subject)
            target = HOST_ROLE_TARGET.get((node.host, place.role)) or SHARED_ROLES.get(place.role)
            self.node_counts[node.host][node.outcome.value] += 1
            if target is not None:
                per_cell[(subject, target, node.host)].add(node)
        grouped: dict[tuple[str, str], list[Cell]] = defaultdict(list)
        for (subject, target, _), cell in per_cell.items():
            grouped[(subject, target)].append(cell)
        for (subject, target), cells in grouped.items():
            self.cells[subject][target] = shared_state(cells) if target in SHARED_TARGETS else cells[0]
        self._move_language_checks()

    def _move_language_checks(self) -> None:
        """A subject that only the frontend target tests (a refusal, a rendering, a binding) is a check of the
        language, not a kernel: it moves to the runtime family, and its id follows."""
        for subject in [s for s, cells in self.cells.items() if set(cells) <= {"frontend"}]:
            name = subject.split("/", 1)[1]
            moved = f"{Family.RUNTIME.name.lower()}/{name}"
            self.cells[moved] = {**self.cells.pop(subject), **self.cells.get(moved, {})}
            if subject in self.named:
                self.named.add(moved)
            self.families.pop(subject)
            self.families[moved] = Family.RUNTIME


class Percent:
    """Pass rates of a set of subjects on a target: over all `universe` subjects, and over the applicable ones."""

    @staticmethod
    def of(cells: Iterable[Cell], universe: int) -> schema.FullStats:
        counter: Counter[str] = Counter()
        for cell in cells:
            counter[cell.state.value] += 1
            if cell.state is State.SKIP and cell.dominant_class in NOT_APPLICABLE:
                counter["by_design"] += 1
        tested = counter["pass"] + counter["partial"] + counter["fail"] + counter["skip"]
        applicable = tested - counter["by_design"]
        return {
            "subjects": tested,
            "applicable": applicable,
            "pass": counter["pass"],
            "partial": counter["partial"],
            "fail": counter["fail"],
            "skip": counter["skip"],
            "by_design": counter["by_design"],
            "percent": round(100 * counter["pass"] / applicable, 1) if applicable else None,
            "universe": universe,
            "percent_of_all": round(100 * counter["pass"] / universe, 1) if universe else None,
        }

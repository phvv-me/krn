"""Complexity of a kernel source: Lizard NLOC, CCN and tokens, an LLM token count, and the API surface read.

A source is a repository, a path and a line range, exactly what the corpus records. The API surface is the number of
distinct symbols of the framework that the range uses: names imported from it, attributes read from its module aliases,
and, for our language, the members of its classes. For non-Python sources it is approximate (qualified names, double
underscore intrinsics, imports) and flagged.

Run: python metrics.py sketch <examples dir>   or   python metrics.py corpus <corpus.json> <src root> <out.json>
"""

import ast
import json
import logging
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import lizard
import tiktoken

from experiments.text import words

type Row = dict[str, str | int | bool | list[str]]
log = logging.getLogger("metrics")
ENCODING = tiktoken.get_encoding("o200k_base")

LIZARD_SUFFIX = {
    ".py": ".py", ".cu": ".cpp", ".cuh": ".cpp", ".cpp": ".cpp", ".cc": ".cpp", ".h": ".cpp", ".hpp": ".cpp",
    ".c": ".cpp", ".hip": ".cpp", ".metal": ".cpp", ".slang": ".cpp", ".rs": ".rs", ".mojo": ".py", ".zig": ".zig",
}  # fmt: skip
APPROXIMATE = {".mojo", ".slang", ".metal", ".hip", ".cu", ".cuh"}  # parsed as a neighbor language
DSL_ROOTS = {
    "triton": {"triton"}, "tilelang": {"tilelang"}, "tilus": {"tilus"}, "cute-dsl": {"cutlass"}, "cutile": {"cuda"},
    "warp": {"warp"}, "helion": {"helion"}, "pallas": {"jax"}, "numba": {"numba"}, "taichi": {"taichi"},
    "tinygrad": {"tinygrad"}, "krn-gpu": {"krn"}, "krn-cuda": {"krn"}, "mlx": {"mlx"},
}  # fmt: skip


@dataclass(frozen=True)
class Metrics:
    framework: str
    path: str
    start: int
    end: int
    nloc: int
    ccn_total: int
    ccn_max: int
    lizard_tokens: int
    llm_tokens: int
    characters: int
    api_symbols: int
    api_names: list[str]
    approximate: bool


def read_range(path: Path, start: int, end: int) -> str:
    return "".join(path.read_text(errors="replace").splitlines(keepends=True)[start - 1 : end])


def lizard_numbers(code: str, suffix: str) -> tuple[int, int, int, int]:
    """NLOC, total and maximum cyclomatic complexity over the functions found, and Lizard's token count."""
    info = lizard.analyze_file.analyze_source_code(f"unit{LIZARD_SUFFIX.get(suffix, '.cpp')}", code)
    ccns = [f.cyclomatic_complexity for f in info.function_list] or [1]
    return info.nloc, sum(ccns), max(ccns), info.token_count


def python_api(code: str, full_text: str, roots: set[str], members: set[str]) -> list[str]:
    """Names of the framework used by a Python range: imports, module-alias attributes and known member names."""
    aliases: dict[str, str] = {}
    imported: dict[str, str] = {}
    for node in ast.walk(ast.parse(full_text)):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or a.name.split(".")[0]: a.name for a in node.names if a.name.split(".")[0] in roots}
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in roots:
            imported |= {a.asname or a.name: f"{node.module}.{a.name}" for a in node.names}
            aliases |= {a.asname or a.name: f"{node.module}.{a.name}" for a in node.names}
    found: set[str] = set()
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        tree = ast.parse("pass")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name) and base.id in aliases:
                found.add(f"{aliases[base.id]}.{node.attr}")
            elif node.attr in members:
                found.add(f"member.{node.attr}")
        elif isinstance(node, ast.Name) and node.id in imported:
            found.add(imported[node.id])
    return sorted(found)


def cpp_like_api(code: str) -> list[str]:
    """Qualified names (ns::name, T.name), double-underscore intrinsics and launch syntax: an approximation."""
    pieces = "".join(c if c.isalnum() or c in "_:" else " " for c in code).split()
    qualified = {part for part in pieces if "::" in part and part.split("::")[0].isidentifier()}
    intrinsics = {part for part in words(code) if part.startswith("__") and len(part) > 2}
    imports = {
        line.partition("import ")[2]
        for raw in code.splitlines()
        if (line := raw.strip()).startswith(("import ", "from ")) and "import " in line
    }
    names = {n.strip() for chunk in imports for n in chunk.strip("() ").split(",") if n.strip()}
    return sorted(qualified | intrinsics | names)


def api_members(sources: list[Path]) -> set[str]:
    """Method, function and class names defined in a framework's own sources (for attribute matching on values)."""
    names: set[str] = set()
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                names.add(node.name)
    return {n for n in names if not n.startswith("_")}


def measure(framework: str, path: Path, start: int, end: int, members: set[str] | None = None) -> Metrics:
    code = read_range(path, start, end)
    suffix = path.suffix
    nloc, ccn_total, ccn_max, tokens = lizard_numbers(textwrap.dedent(code), suffix)
    if suffix == ".py":
        names = python_api(code, path.read_text(errors="replace"), DSL_ROOTS.get(framework, set()), members or set())
    else:
        names = cpp_like_api(code)
    return Metrics(framework, str(path), start, end, nloc, ccn_total, ccn_max, tokens, len(ENCODING.encode(code)),
                   len(code), len(names), names, suffix in APPROXIMATE or suffix != ".py")  # fmt: skip


def kernels_in(path: Path) -> list[tuple[str, int, int]]:
    """Decorated functions and classes of one of our sketch examples: name, first and last line."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.decorator_list:
            found.append((node.name, min(d.lineno for d in node.decorator_list), node.end_lineno or node.lineno))
    return sorted(found, key=lambda k: k[1])


def sketch(examples: Path) -> list[Row]:
    members = api_members(sorted((examples.parent / "krn").glob("*.py")))
    rows = []
    for path in sorted(examples.glob("*.py")):
        if path.name in {"__init__.py", "mistakes.py"}:
            continue
        for name, start, end in kernels_in(path):
            m = measure("krn-gpu", path, start, end, members)
            rows.append({"kernel": f"{path.stem}.{name}", **{k: v for k, v in asdict(m).items() if k != "api_names"}})
    return rows


ALIASES = {
    "transformerengine": "te",
    "cudalibrarysamples": "cudalibsamples",
    "ao": "torchao",
    "gpu-rodinia": "rodinia",
    "intel-xpu-backend-for-triton": "intel-xpu-triton",
}
MONOREPO = Path(__file__).resolve().parents[3]


def clone_name(directory: str) -> str:
    """The repository name a clone directory stands for: lowercase, without the lane prefix and `-repo` suffix."""
    name = directory.lower().removesuffix("-repo")
    return next((name.removeprefix(prefix) for prefix in ("amd-", "lp-", "prof-") if name.startswith(prefix)), name)


def resolve(root: Path, repo: str, relative: str) -> Path | None:
    """The local file of `relative` in `repo`: clones sit at src/<name>, src/<name>/repo or src/round3/<name>."""
    name = repo.rstrip("/").split("/")[-1].removesuffix(".git").lower()
    if (MONOREPO / relative).is_file() and name == MONOREPO.name:
        return MONOREPO / relative
    wanted = {name, ALIASES.get(name, name)}
    for base in (root, root / "round3"):
        for entry in base.iterdir() if base.is_dir() else []:
            if clone_name(entry.name) in wanted:
                for candidate in (entry, entry / "repo"):
                    if (candidate / relative).is_file():
                        return candidate / relative
    return None


def corpus(manifest: Path, root: Path) -> list[Row]:
    rows: list[Row] = []
    for item in json.loads(manifest.read_text()):
        for src in item.get("sources", []):
            path = resolve(root, src.get("repo", ""), src["path"])
            if path is None:
                rows.append(
                    {
                        "corpus_id": item.get("corpus_id", item["id"]),
                        "framework": src["framework"],
                        "error": "file not found",
                    }
                )
                continue
            m = measure(src["framework"], path, int(src["start_line"]), int(src["end_line"]))
            rows.append({"corpus_id": item.get("corpus_id", item["id"]), "kernel": item["name"], **asdict(m)})
    return rows


def table(rows: list[Row]) -> str:
    keys = ["kernel", "nloc", "ccn_total", "ccn_max", "lizard_tokens", "llm_tokens", "api_symbols"]
    lines = ["  ".join(f"{k:>14}" for k in keys)]
    lines += ["  ".join(f"{r.get(k, '')!s:>14}" for k in keys) for r in rows if "error" not in r]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if sys.argv[1] == "sketch":
        out = sketch(Path(sys.argv[2]))
    else:
        out = corpus(Path(sys.argv[2]), Path(sys.argv[3]))
        destination = Path(sys.argv[4])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(out, indent=2) + "\n")
    log.info(table(out))

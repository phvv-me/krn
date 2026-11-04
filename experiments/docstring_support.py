"""Write the "Supported on" block at the end of each public function docstring of krn.math, krn.fast.math and
krn.intrinsics, from `../krn.dev/data/compat/matrix.json`.

Run from the repository root with `mb run python -m experiments.docstring_support --write`, after
`experiments.compat_matrix` has written the matrix. Without `--write` the script checks the docstrings and exits
non-zero when one disagrees with the matrix; tests/_internals/compat runs that check. A function's block reads each
vendor's targets over all the function's operand types: a target whose operand types differ names them after its best
status.
"""

import ast
import importlib
import logging
import sys
import textwrap
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cyclopts

from experiments.compat_matrix import OUTPUT
from krn._internals.compat import Feature, Hardware, Matrix, Status, Support, loads

ROOT = Path(__file__).resolve().parents[1]
MARKER = "Supported on (generated):"
MAX_LINES = 8
VENDORS = {"nvidia": "NVIDIA", "apple": "Apple", "amd": "AMD", "cpu": "CPU"}
RANK = {Status.NATIVE: 0, Status.COMPOSED: 1, Status.REFUSED: 2, Status.EMULATOR: 3, Status.UNKNOWN: 4}
LOG = logging.getLogger(__name__)
APP = cyclopts.App(name="docstring_support", help=__doc__.splitlines()[0])


@dataclass(frozen=True, kw_only=True)
class Source:
    """A public module: its import name, its file, and the prefix its matrix rows carry before the function's name."""

    module: str
    path: Path
    prefix: str = ""


SOURCES = (
    Source(module="krn.math", path=ROOT / "src" / "krn" / "math" / "__init__.py"),
    Source(module="krn.fast.math", path=ROOT / "src" / "krn" / "fast" / "math.py", prefix="fast."),
    Source(module="krn.intrinsics", path=ROOT / "src" / "krn" / "intrinsics" / "__init__.py"),
)


def public_functions(source: Source) -> list[str]:
    """The public functions the module defines, sorted."""
    module = importlib.import_module(source.module)
    tree = ast.parse(source.path.read_text())
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    return sorted(
        name for name in defined if not name.startswith("_") and getattr(module, name).__module__ == source.module
    )


def rows(matrix: Matrix, source: Source, function: str) -> list[Feature]:
    """The matrix rows of one function: one per operand type, or the single row of a fast function."""
    name = f"{source.prefix}{function}"
    found = [f for f in matrix.features if f.module == source.module and f.name.split("[")[0] == name]
    if not found:
        raise LookupError(f"{source.module}.{function} has no row in the matrix: run experiments.compat_matrix")
    return found


def operand(feature: Feature) -> str:
    """The operand type a row names in brackets, or the row's name when it names none."""
    return feature.id[feature.id.index("[") + 1 : -1] if "[" in feature.id else feature.name


def text_of(support: Support, length: int, arch: str) -> str:
    """A status as the block reads it: `native`, `composed (N instructions)`, or `refused` with the head of its reason.
    The target's own name is left out of the reason, so the targets of a vendor agree and share one run."""
    match support.status:
        case Status.COMPOSED:
            return f"composed ({length} instruction{'' if length == 1 else 's'})"
        case Status.REFUSED:
            clause = support.note.split(";")[0].replace(arch, "this target")
            return f"refused ({textwrap.shorten(clause, 72, placeholder='...')})"
        case other:
            return other.value


def label(arch: str, pairs: Sequence[tuple[str, Support]]) -> str:
    """A target's reading of a function over its operand rows: the status most operand types have, then the rows that
    differ from it, named when there are few and counted when there are many."""
    counts = Counter(support.status for _, support in pairs)
    main = min(counts, key=lambda status: (-counts[status], RANK[status]))
    length = max((s.length or 0 for _, s in pairs if s.status is Status.COMPOSED), default=0)
    text = text_of(next(s for _, s in pairs if s.status is main), length, arch)
    others = [(name, support) for name, support in pairs if support.status is not main]
    if not others:
        return text
    if len(others) <= 3:
        return f"{text} ({'; '.join(f'{name} {s.status.value}' for name, s in others)})"
    kinds = " or ".join(sorted({s.status.value for _, s in others}))
    return f"{text}, {len(others)} of {len(pairs)} operand types {kinds}"


def span_text(run: Sequence[str], last: str) -> str:
    """`sm_80` for one target, `sm_80+` for a run that reaches the vendor's last target, else `sm_80..sm_89`."""
    if len(run) == 1:
        return run[0]
    return f"{run[0]}+" if run[-1] == last else f"{run[0]}..{run[-1]}"


def runs(targets: Sequence[Hardware], labels: dict[str, str]) -> list[str]:
    """The consecutive targets with one label, each as `range label`."""
    grouped: list[list[str]] = []
    for target in targets:
        if grouped and labels[grouped[-1][-1]] == labels[target.arch]:
            grouped[-1].append(target.arch)
        else:
            grouped.append([target.arch])
    return [f"{span_text(run, targets[-1].arch)} {labels[run[0]]}" for run in grouped]


def requires_line(matrix: Matrix, supports: dict[str, list[tuple[str, Support]]]) -> str:
    """`Requires: NVIDIA from sm_80: CUDA 11.0, driver 450.80.02, PTX ISA 7.0; ...`: each vendor's first native target
    that has documented requirements, with those requirements."""
    parts = []
    for vendor, name in VENDORS.items():
        for target in (t for t in matrix.targets if t.vendor == vendor):
            native = [s.requires for _, s in supports[target.arch] if s.status is Status.NATIVE and s.requires]
            if native:
                parts.append(f"{name} from {target.arch}: {native[0].summary()}")
                break
    return f"Requires: {'; '.join(parts)}." if parts else ""


def block_lines(matrix: Matrix, source: Source, function: str) -> list[str]:
    """The block of one function at the docstring's indentation of zero, ending with the blank line that closes it."""
    features = rows(matrix, source, function)
    supports = {t.arch: [(operand(f), matrix.cells[f.id][t.arch]) for f in features] for t in matrix.targets}
    labels = {arch: label(arch, pairs) for arch, pairs in supports.items()}
    if len(set(labels.values())) == 1:
        text = next(iter(labels.values()))
        body = [f"    {text[0].upper()}{text[1:]} on every target"]
    else:
        body = []
        for vendor, name in VENDORS.items():
            targets = [t for t in matrix.targets if t.vendor == vendor]
            if targets:
                body += textwrap.wrap(
                    f"{name}: {'; '.join(runs(targets, labels))}",
                    width=110,
                    initial_indent="    ",
                    subsequent_indent="        ",
                )
    if requires := requires_line(matrix, supports):
        body += textwrap.wrap(requires, width=110, initial_indent="", subsequent_indent="    ")
    if len(body) + 2 > MAX_LINES:
        raise ValueError(f"the block of {source.module}.{function} is {len(body) + 2} lines, over {MAX_LINES}")
    return [MARKER, *body, ""]


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def docstring_span(tree: ast.Module, function: str) -> tuple[int, int]:
    """The first and last line, 1-based, of the docstring of the function, the one that has a docstring when overloads
    share its name."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function and _is_docstring(node.body[0]):
            return node.body[0].lineno, node.body[0].end_lineno or node.body[0].lineno
    raise ValueError(f"{function} has no docstring to carry its block")


def splice(lines: list[str], block: list[str]) -> list[str]:
    """The docstring's lines with `block` at its end, in place of a block an earlier run wrote. The docstring keeps its
    opening and closing quotes; a one-line docstring grows a line for the closing quotes."""
    indent = lines[0][: lines[0].index('"""')]
    if len(lines) == 1:
        lines = [f'{indent}"""{lines[0].strip()[3:-3]}', f'{indent}"""']
    first, last = lines[0][len(indent) + 3 :], lines[-1][: lines[-1].rindex('"""')]
    body = [first, *lines[1:-1]]
    closing = indent
    if last.strip():
        body.append(last)
    else:
        closing = last
    kept: list[str] = []
    skipping = False
    for line in body:
        if line.strip() == MARKER:
            skipping = True
        elif skipping:
            skipping = line.strip() != ""
        else:
            kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    head = f'{indent}"""{kept[0] if kept else ""}'
    indented = [f"{indent}{line}" if line else "" for line in block]
    return [head, *kept[1:], "", *indented, f'{closing}"""']


def rewrite(source: Source, matrix: Matrix) -> str:
    """The module's text with the block of every public function written from the matrix."""
    text = source.path.read_text()
    tree = ast.parse(text)
    lines = text.split("\n")
    edits = [(docstring_span(tree, f), block_lines(matrix, source, f)) for f in public_functions(source)]
    for (first, last), block in sorted(edits, reverse=True):
        lines[first - 1 : last] = splice(lines[first - 1 : last], block)
    return "\n".join(lines)


def stale_modules(matrix: Matrix, write: bool) -> list[str]:
    """The modules whose docstrings disagree with the matrix. With `write`, their text is replaced."""
    stale = []
    for source in SOURCES:
        expected = rewrite(source, matrix)
        if expected != source.path.read_text():
            stale.append(source.module)
            if write:
                source.path.write_text(expected)
    return stale


@APP.default
def main(
    *,
    write: Annotated[bool, cyclopts.Parameter(help="rewrite the blocks from the matrix")] = False,
    check: Annotated[bool, cyclopts.Parameter(help="exit non-zero when a block disagrees (the default)")] = False,
) -> None:
    """Check the public docstrings against the matrix, or with `--write` rewrite their blocks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    if write and check:
        raise SystemExit("choose --write or --check")
    stale = stale_modules(loads(OUTPUT.read_text()), write)
    if stale and not write:
        LOG.error("out of date: %s (run `mb run python -m experiments.docstring_support --write`)", ", ".join(stale))
        raise SystemExit(1)
    LOG.info("%s", f"wrote {', '.join(stale)}" if stale else "docstrings match the matrix")


if __name__ == "__main__":
    APP()

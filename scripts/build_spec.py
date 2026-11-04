"""Write the release spec of krn: its API, callouts, compatibility matrix, examples and hardware, as one JSON file.

Run from the repository root with `mb run python -m scripts.build_spec`. The file `krn-spec-<version>.json` goes to
`--out`, which defaults to `KRN_SPEC_OUT` and then to the site's `data/spec`. The API is read statically with griffe, so
no module is imported for its docstrings, and the compatibility matrix comes from `experiments.compat_matrix`. `--check`
fails when the file in `--out` is not what the builder writes.
"""

import ast
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Annotated

import cyclopts
import griffe

from experiments import compat_matrix
from krn._internals.compat import Matrix, dumps

type Json = str | int | float | bool | list[Json] | dict[str, Json] | None

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
EXAMPLES = ROOT / "examples"
SCHEMA_ID = "krn.spec"
SCHEMA_VERSION = 1
SCHEMA_FILE = Path(__file__).with_name("spec.schema.json")
UNCOMMITTED = "uncommitted"
DEFAULT_OUT = Path(os.environ.get("KRN_SPEC_OUT", ROOT / "../krn.dev/data/spec")).resolve()
DOCUMENTED_MODULES = ("krn.math", "krn.fast.math", "krn.intrinsics", "krn.errors", "krn.emulate")
LEVELS = 5
NO_NEEDS = "none"
DEPRECATED_CALLS = {"warnings.deprecated", "deprecated"}
MARKER = "Supported on (generated):"
DIRECTIVE = re.compile(r"^(\s*)\.\. (versionadded|versionchanged|deprecated)::(.*)$")
VERSION = re.compile(r"\d+(\.\d+)+")
VENDOR_LINE = re.compile(r"^(?P<vendor>NVIDIA|Apple|AMD|CPU): (?P<runs>.*)$")
REQUIREMENT = re.compile(r"^(?P<vendor>\w+) from (?P<target>\S+): (?P<text>.+)$")
LOG = logging.getLogger(__name__)
APP = cyclopts.App(name="build_spec", help=__doc__.splitlines()[0])


class SpecError(ValueError):
    """A docstring or decorator that breaks the spec's rules. The build stops and names the object."""


@dataclass(frozen=True, kw_only=True)
class Callout:
    """A `.. versionadded::`, `.. versionchanged::` or `.. deprecated::` directive of a docstring."""

    kind: str
    version: str
    text: str


@dataclass(frozen=True, kw_only=True)
class DocParameter:
    name: str
    annotation: str | None
    description: str


@dataclass(frozen=True, kw_only=True)
class DocRaise:
    exception: str
    description: str


@dataclass(frozen=True, kw_only=True)
class Doc:
    """The prose of a docstring: its summary paragraph, the rest of its text and the Google sections."""

    summary: str
    description: str
    parameters: tuple[DocParameter, ...] = ()
    returns: str | None = None
    raises: tuple[DocRaise, ...] = ()
    examples: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Run:
    """One run of targets in a vendor row, as `sm_80+` and what the targets do there."""

    targets: str
    reading: str


@dataclass(frozen=True, kw_only=True)
class VendorRow:
    vendor: str
    runs: tuple[Run, ...]


@dataclass(frozen=True, kw_only=True)
class Requirement:
    vendor: str
    target: str
    text: str


@dataclass(frozen=True, kw_only=True)
class Supported:
    """The generated "Supported on" block: one reading for every target, or a row per vendor, and the requirements."""

    everywhere: str | None = None
    vendors: tuple[VendorRow, ...] = ()
    requires: tuple[Requirement, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Argument:
    name: str
    kind: str
    annotation: str | None
    default: str | None


@dataclass(frozen=True, kw_only=True)
class Signature:
    parameters: tuple[Argument, ...]
    returns: str | None


def read_docstring(owner: str, text: str) -> tuple[Doc, tuple[Callout, ...], Supported | None]:
    """The prose, the callouts and the generated block of one docstring. `owner` names the object in errors."""
    lines, callouts, support = _split(owner, text.splitlines())
    sections = griffe.Docstring("\n".join(lines).strip()).parse("google")
    return _doc(owner, sections), tuple(callouts), _supported(owner, support)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _body_end(lines: list[str], start: int, indent: int | None) -> int:
    """The index after the non-blank lines under line `start`, which must be indented past `indent` when it is given."""
    end = start + 1
    while end < len(lines) and lines[end].strip() and (indent is None or _indent(lines[end]) > indent):
        end += 1
    return end


def _split(owner: str, lines: list[str]) -> tuple[list[str], list[Callout], list[str]]:
    """The lines that are prose, the callouts, and the lines of the generated block, each in the order they appear."""
    prose: list[str] = []
    callouts: list[Callout] = []
    support: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if match := DIRECTIVE.match(line):
            end = _body_end(lines, index, _indent(line))
            callouts.append(_callout(owner, match, lines[index + 1 : end]))
            index = end
        elif line.strip() == MARKER:
            end = _body_end(lines, index, None)
            support.extend(lines[index + 1 : end])
            index = end
        else:
            prose.append(line)
            index += 1
    return prose, callouts, support


def _callout(owner: str, match: re.Match[str], body: list[str]) -> Callout:
    kind, version = match.group(2), match.group(3).strip()
    if not VERSION.fullmatch(version):
        raise SpecError(f"{owner}: `.. {kind}::` needs a dotted version, got {version!r}")
    return Callout(kind=kind, version=version, text=" ".join(line.strip() for line in body))


def _split_top(text: str) -> list[str]:
    """The `; `-separated parts of a line, not counting the separators inside brackets."""
    parts: list[str] = []
    depth = start = 0
    for index, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if char == ";" and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    return [*parts, text[start:].strip()]


def _supported(owner: str, lines: list[str]) -> Supported | None:
    """The generated block: a vendor line starts a row, a `Requires:` line starts the requirements, and any other line
    continues the row or the requirements above it."""
    if not lines:
        return None
    rows: list[str] = []
    for line in (line.strip() for line in lines):
        if line.startswith("Requires: ") or VENDOR_LINE.match(line) or line.endswith(" on every target") or not rows:
            rows.append(line)
        else:
            rows[-1] += f" {line}"
    everywhere = None
    vendors: list[VendorRow] = []
    requires: tuple[Requirement, ...] = ()
    for row in rows:
        if match := VENDOR_LINE.match(row):
            runs = (
                Run(targets=part.split(" ", 1)[0], reading=part.partition(" ")[2]) for part in _split_top(match["runs"])
            )
            vendors.append(VendorRow(vendor=match["vendor"], runs=tuple(runs)))
        elif row.startswith("Requires: "):
            requires = tuple(
                _requirement(owner, part) for part in _split_top(row.removeprefix("Requires: ").rstrip("."))
            )
        elif row.endswith(" on every target"):
            everywhere = row
        else:
            raise SpecError(f"{owner}: unknown line in the Supported on block: {row!r}")
    return Supported(everywhere=everywhere, vendors=tuple(vendors), requires=requires)


def _requirement(owner: str, part: str) -> Requirement:
    if not (match := REQUIREMENT.match(part)):
        raise SpecError(f"{owner}: a requirement reads `VENDOR from TARGET: text`, got {part!r}")
    return Requirement(vendor=match["vendor"], target=match["target"], text=match["text"])


def _paragraphs(text: str) -> list[str]:
    return [" ".join(paragraph.split()) for paragraph in text.split("\n\n") if paragraph.strip()]


def _doc(owner: str, sections: Sequence[griffe.DocstringSection]) -> Doc:
    """The Google sections of a docstring as data. A section griffe reads that the spec does not keep is an error."""
    paragraphs: list[str] = []
    parameters: list[DocParameter] = []
    returns: str | None = None
    raises: list[DocRaise] = []
    examples: list[str] = []
    notes: list[str] = []
    for section in sections:
        match section:
            case griffe.DocstringSectionText():
                paragraphs += _paragraphs(section.value)
            case griffe.DocstringSectionParameters():
                parameters = [
                    DocParameter(name=p.name, annotation=_text(p.annotation), description=p.description)
                    for p in section.value
                ]
            case griffe.DocstringSectionReturns():
                returns = " ".join(item.description for item in section.value)
            case griffe.DocstringSectionRaises():
                raises = [
                    DocRaise(exception=_text(item.annotation) or "", description=item.description)
                    for item in section.value
                ]
            case griffe.DocstringSectionExamples():
                examples = [text for _, text in section.value]
            case griffe.DocstringSectionAdmonition():
                notes.append(section.value.text)
            case _:
                raise SpecError(f"{owner}: the spec does not keep the `{section.kind.value}` section")
    return Doc(
        summary=paragraphs[0] if paragraphs else "",
        description="\n\n".join(paragraphs[1:]),
        parameters=tuple(parameters),
        returns=returns,
        raises=tuple(raises),
        examples=tuple(examples),
        notes=tuple(notes),
    )


def _text(expression: griffe.Expr | str | None) -> str | None:
    return None if expression is None else str(expression)


def _signature(function: griffe.Function) -> Signature:
    arguments = tuple(
        Argument(name=p.name, kind=p.kind.value, annotation=_text(p.annotation), default=_text(p.default))
        for p in function.parameters
    )
    return Signature(parameters=arguments, returns=_text(function.returns))


def _decorator_message(obj: griffe.Object) -> str | None:
    """The message of a `warnings.deprecated` decorator, read from the source of the decorator."""
    decorators = obj.decorators if isinstance(obj, griffe.Function | griffe.Class) else ()
    for decorator in decorators:
        call = ast.parse(str(decorator.value), mode="eval").body
        if isinstance(call, ast.Call) and ast.unparse(call.func) in DEPRECATED_CALLS:
            return ast.literal_eval(call.args[0])
    return None


def _deprecation(obj: griffe.Object, callouts: Sequence[Callout]) -> str | None:
    """The decorator's message, once it agrees with the `.. deprecated::` directive: one of the two without the other is
    an error, since the decorator is the source of truth and the directive is what the docs show."""
    message = _decorator_message(obj)
    directives = [callout for callout in callouts if callout.kind == "deprecated"]
    if (message is None) != (not directives) or len(directives) > 1:
        raise SpecError(f"{obj.path}: the warnings.deprecated decorator and the `.. deprecated::` directive disagree")
    return message


def _final(obj: griffe.Object | griffe.Alias) -> griffe.Object | None:
    """The object an alias names, or None when it names something outside krn, which the spec does not describe."""
    try:
        return obj.final_target if isinstance(obj, griffe.Alias) else obj
    except griffe.AliasResolutionError:
        return None


class Api:
    """The public API: the modules `krn.__all__` exports and the documented modules, each with its public members."""

    def __init__(self, package: griffe.Module) -> None:
        self.package = package

    def bound(self, module: griffe.Module, name: str) -> griffe.Object | None:
        """What `name` is in `module` at run time: an import of that name wins over a submodule that shares it."""
        if (target := module.imports.get(name, "")).startswith("krn."):
            return _final(self.package[target.removeprefix("krn.")])
        return _final(module.members[name])

    def modules(self) -> list[dict[str, Json]]:
        found = {"krn": self.package}
        for name in self.package.exports or []:
            if isinstance(target := self.bound(self.package, name), griffe.Module):
                found[target.path] = target
        for path in DOCUMENTED_MODULES:
            found[path] = _final(self.package[path.removeprefix("krn.")])
        return [{"module": path, "members": self.members(found[path])} for path in sorted(found)]

    def members(self, module: griffe.Module) -> list[dict[str, Json]]:
        names = module.exports if module.exports is not None else [n for n in module.members if not n.startswith("_")]
        members = []
        for name in sorted(names):
            obj = self.bound(module, name)
            if obj is None or isinstance(obj, griffe.Module) or obj.path.split(".")[0] != "krn":
                continue
            members.append(self.member(name, obj))
        return members

    def member(self, name: str, obj: griffe.Object) -> dict[str, Json]:
        doc, callouts, supported = self.read(obj)
        entry: dict[str, Json] = {
            "name": name,
            "kind": "",
            "defined_in": obj.module.path,
            "source": {"path": obj.filepath.relative_to(ROOT).as_posix(), "line": obj.lineno or 0},
            "doc": asdict(doc) if doc else None,
            "callouts": [asdict(callout) for callout in callouts],
            "deprecated": _deprecation(obj, callouts),
            "supported": asdict(supported) if supported else None,
            "signature": None,
            "overloads": [],
            "methods": [],
            "annotation": None,
            "value": None,
        }
        match obj:
            case griffe.Function():
                entry |= {"kind": "function", "signature": asdict(_signature(obj))}
                entry["overloads"] = [asdict(_signature(overload)) for overload in obj.overloads or ()]
            case griffe.Class():
                entry |= {"kind": "class", "methods": [self.method(m) for m in _methods(obj)]}
                if isinstance(init := obj.members.get("__init__"), griffe.Function):
                    entry["signature"] = asdict(_signature(init))
            case griffe.TypeAlias():
                entry |= {"kind": "type alias", "value": _text(obj.value)}
            case griffe.Attribute():
                entry |= {"kind": "constant", "annotation": _text(obj.annotation), "value": _text(obj.value)}
            case _:
                raise SpecError(f"{obj.path}: the spec has no kind for {type(obj).__name__}")
        return entry

    def method(self, obj: griffe.Function) -> dict[str, Json]:
        return self.member(obj.name, obj)

    def read(self, obj: griffe.Object) -> tuple[Doc | None, tuple[Callout, ...], Supported | None]:
        if not obj.docstring or not obj.docstring.value.strip():
            return None, (), None
        return read_docstring(obj.path, obj.docstring.value)


def _methods(cls: griffe.Class) -> list[griffe.Function]:
    return [m for name, m in sorted(cls.members.items()) if isinstance(m, griffe.Function) and not name.startswith("_")]


@dataclass(frozen=True, kw_only=True)
class ExampleHeader:
    """What the docstring of an example says: title, teaching text and the metadata block at its end."""

    title: str
    description: str
    level: int
    topics: tuple[str, ...]
    needs: tuple[str, ...]


def read_example_header(name: str, docstring: str) -> ExampleHeader:
    """The header of an example docstring: title, teaching text, then `Level:`, `Topics:` and `Needs:` lines.

    name: the example, for errors.
    """
    title, _, rest = docstring.partition("\n")
    if not title.strip():
        raise SpecError(f"{name} needs a module docstring whose first line is its title")
    body, _, block = rest.rpartition("\n\n")
    fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
    if {key.strip() for key in fields} != {"Level", "Topics", "Needs"}:
        raise SpecError(f"{name}: the docstring must end with a block of `Level:`, `Topics:` and `Needs:` lines")
    level = fields["Level"].strip()
    if not level.isdigit() or not 1 <= int(level) <= LEVELS:
        raise SpecError(f"{name}: `Level:` is a number from 1 to {LEVELS}, got {level!r}")
    needs = fields["Needs"].strip()
    return ExampleHeader(
        title=title.strip(),
        description=" ".join(body.split()),
        level=int(level),
        topics=tuple(topic.strip() for topic in fields["Topics"].split(",") if topic.strip()),
        needs=() if needs == NO_NEEDS else tuple(need.strip() for need in needs.split(",")),
    )


def examples(folder: Path) -> list[dict[str, Json]]:
    """Every example of `folder`, which holds one folder per level (`02_tiles_and_types`) of files named `NN_name.py`.

    An id is the path without the extension (`02_tiles_and_types/01_softmax_rows`). The folder number is the level, and
    each `Needs:` entry must be the id of an example.
    """
    paths = sorted(folder.glob("[0-9][0-9]_*/[0-9][0-9]_*.py"))
    entries: list[dict[str, Json]] = []
    for path in paths:
        group, number, name = path.parent.name, int(path.name[:2]), path.stem[3:]
        ident = f"{group}/{path.stem}"
        source = path.read_text()
        header = read_example_header(ident, ast.get_docstring(ast.parse(source)) or "")
        if header.level != int(group[:2]):
            raise SpecError(f"{ident}: `Level: {header.level}` but its folder {group} is level {int(group[:2])}")
        entries.append(
            {
                "id": ident,
                "group": group,
                "level": header.level,
                "number": number,
                "name": name,
                "title": header.title,
                "description": header.description,
                "topics": list(header.topics),
                "needs": list(header.needs),
                "path": path.relative_to(folder.parent).as_posix(),
                "source": source,
            }
        )
    known = {entry["id"] for entry in entries}
    for entry in entries:
        if missing := [need for need in entry["needs"] if need not in known]:
            raise SpecError(f"{entry['id']}: `Needs:` names {missing}, which are not examples")
    return entries


def hardware(matrix: Matrix) -> dict[str, Json]:
    """The targets of the matrix, and each machine that measured a cell of it."""
    machines = {m.machine: m for row in matrix.cells.values() for s in row.values() for m in s.measured}
    return {
        "targets": [asdict(target) for target in matrix.targets],
        "machines": [asdict(machines[name]) for name in sorted(machines)],
    }


def version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def git_sha() -> str:
    """The commit the tree is at, or `uncommitted` for a working tree with no commit to name (a local preview)."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    return result.stdout.strip() if result.returncode == 0 else UNCOMMITTED


def build(matrix: Matrix) -> dict[str, Json]:
    """The spec of the working tree, with the compatibility matrix the caller computed."""
    package = griffe.load("krn", search_paths=[SOURCE], allow_inspection=False)
    return {
        "schema": f"{SCHEMA_ID}/{SCHEMA_VERSION}",
        "version": version(),
        "git_sha": git_sha(),
        "date": date.today().isoformat(),
        "python": platform.python_version(),
        "api": Api(package).modules(),
        "compat": json.loads(dumps(matrix)),
        "examples": examples(EXAMPLES),
        "hardware": hardware(matrix),
    }


def dumps_spec(spec: dict[str, Json]) -> str:
    """The spec as JSON with sorted keys, so the same tree always gives the same bytes."""
    return json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def callout_count(spec: dict[str, Json]) -> int:
    members = [m for module in spec["api"] for m in module["members"]]
    members += [method for member in members for method in member["methods"]]
    return sum(len(member["callouts"]) for member in members)


def summarize(spec: dict[str, Json], text: str) -> None:
    for module in spec["api"]:
        LOG.info("%-18s %4d members", module["module"], len(module["members"]))
    LOG.info(
        "callouts %d, examples %d, compat %d bytes",
        callout_count(spec),
        len(spec["examples"]),
        len(json.dumps(spec["compat"])),
    )
    LOG.info("spec %d bytes", len(text.encode()))


@APP.default
def main(
    *,
    out: Annotated[Path, cyclopts.Parameter(help="directory of the spec file")] = DEFAULT_OUT,
    check: Annotated[
        bool, cyclopts.Parameter(help="fail when the file in --out is not what the builder writes")
    ] = False,
) -> None:
    """Write `krn-spec-<version>.json`, or with `--check` exit non-zero when the file in `--out` differs from it."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    matrix = compat_matrix.build()
    spec = build(matrix)
    text = dumps_spec(spec)
    path = out / f"krn-spec-{spec['version']}.json"
    summarize(spec, text)
    if check:
        if not path.exists() or path.read_text() != text:
            LOG.error("%s is stale: run `mb run python -m scripts.build_spec`", path)
            raise SystemExit(1)
        return
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    LOG.info("wrote %s", path)


if __name__ == "__main__":
    APP()

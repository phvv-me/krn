"""Generate ctypes bindings from public wheel headers with clang.

NVIDIA is covered by nvmath.bindings since 2026-11-02. This generator remains the strategy for vendors without
official bindings (HIP, Intel and Apple C frameworks); its NVIDIA catalog is a historical worked example.

Regenerate from the workspace (replace <abs> with its absolute path):
mb run --no-install env PYTHONPATH=<abs>/src:<abs> python \
experiments/gen_cuda_bindings.py generate --headers /tmp/kl-b2-fresh --out /tmp/kl-generated --download
"""

import io
import json
import keyword
import logging
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from cyclopts import App

from experiments.text import words
from krn.compiler.text import replace_word

WHEELS = {
    "nvidia-cublas": "13.8.2.12",
    "nvidia-cusparse": "12.8.6.72",
    "nvidia-cufft": "12.4.0.43",
    "nvidia-cusolver": "12.3.5.3",
    "nvidia-curand": "10.4.4.72",
    "nvidia-cuda-runtime": "13.4.92",
    "nvidia-cuda-crt": "13.4.92",
}
PRELUDE = """#define __CUDA_FP16_H__
#define __CUDA_BF16_H__
#define __CUDA_FP8_H__
typedef struct __half __half;
typedef struct __nv_bfloat16 __nv_bfloat16;
"""


@dataclass(frozen=True)
class Spec:
    """One generated library: its top-level header, C name prefix, wheel, soname macro and status handling."""

    header: str
    prefix: str
    wheel: str
    version: str
    status: str
    error: str
    message: str = ""

    @property
    def constants(self) -> str:
        """The prefix of the library's own object-like macros."""
        return self.prefix.upper() + "_"


MODULES = {
    "cublas": Spec(
        "cublas_v2.h",
        "cublas",
        "nvidia-cublas",
        "CUBLAS_VER_MAJOR",
        "cublasStatus_t",
        "CublasError",
        "cublasGetStatusString",
    ),
    "cublaslt": Spec(
        "cublasLt.h",
        "cublasLt",
        "nvidia-cublas",
        "CUBLAS_VER_MAJOR",
        "cublasStatus_t",
        "CublasLtError",
        "cublasLtGetStatusString",
    ),
    "cusparse": Spec(
        "cusparse.h",
        "cusparse",
        "nvidia-cusparse",
        "CUSPARSE_VER_MAJOR",
        "cusparseStatus_t",
        "CusparseError",
        "cusparseGetErrorString",
    ),
    "cufft": Spec("cufft.h", "cufft", "nvidia-cufft", "CUFFT_VER_MAJOR", "cufftResult", "CufftError"),
    "cusolver": Spec(
        "cusolverDn.h",
        "cusolver",
        "nvidia-cusolver",
        "CUSOLVER_VER_MAJOR",
        "cusolverStatus_t",
        "CusolverError",
    ),
    "curand": Spec("curand.h", "curand", "nvidia-curand", "CURAND_VER_MAJOR", "curandStatus_t", "CurandError"),
}
HOMES = {
    "cublas_api.h": "cublas",
    "cublasLt.h": "cublaslt",
    "cusparse.h": "cusparse",
    "cufft.h": "cufft",
    "cusolverDn.h": "cusolver",
    "cusolver_common.h": "cusolver",
    "curand.h": "curand",
}
SCALARS = {
    "char": ("c_int8", "int"),
    "signed char": ("c_int8", "int"),
    "unsigned char": ("c_uint8", "int"),
    "short": ("c_int16", "int"),
    "unsigned short": ("c_uint16", "int"),
    "int": ("c_int32", "int"),
    "unsigned int": ("c_uint32", "int"),
    "long": ("c_int64", "int"),
    "unsigned long": ("c_uint64", "int"),
    "long long": ("c_int64", "int"),
    "unsigned long long": ("c_uint64", "int"),
    "float": ("c_float", "float"),
    "double": ("c_double", "float"),
    "_Bool": ("c_bool", "bool"),
    "int8_t": ("c_int8", "int"),
    "uint8_t": ("c_uint8", "int"),
    "int16_t": ("c_int16", "int"),
    "uint16_t": ("c_uint16", "int"),
    "int32_t": ("c_int32", "int"),
    "uint32_t": ("c_uint32", "int"),
    "int64_t": ("c_int64", "int"),
    "uint64_t": ("c_uint64", "int"),
    "size_t": ("c_uint64", "int"),
    "ptrdiff_t": ("c_int64", "int"),
    "intptr_t": ("c_int64", "int"),
    "uintptr_t": ("c_uint64", "int"),
    "void": ("None", "None"),
}
PYTHON_SCALARS = {"int", "float", "bool"}
NUMERIC = {
    "cuComplex",
    "cuFloatComplex",
    "cuDoubleComplex",
    "float2",
    "double2",
    "__half",
    "__nv_bfloat16",
}
RESERVED = {
    "ctypes",
    "POINTER",
    "IntEnum",
    "Address",
    "Ref",
    "Library",
    "LibraryError",
    "property_version",
    "_library",
}
logger = logging.getLogger(__name__)
app = App()
type Json = str | int | float | bool | list[Json] | dict[str, Json] | None


class RangeFile(io.RawIOBase):
    """Seekable HTTP byte ranges; reject servers that would download an entire wheel."""

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url, self.position = url, 0
        request = Request(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"})
        with urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise OSError("wheel server does not support HTTP Range")
            self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        position = offset + (self.position if whence == io.SEEK_CUR else self.size if whence == io.SEEK_END else 0)
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END) or position < 0:
            raise ValueError("invalid seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        end = self.size if size < 0 else min(self.position + size, self.size)
        if self.position >= end:
            return b""
        request = Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{end - 1}", "Accept-Encoding": "identity"},
        )
        with urlopen(request, timeout=30) as response:
            expected = f"bytes {self.position}-{end - 1}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise OSError("invalid wheel byte range response")
            data = response.read(end - self.position + 1)
        if len(data) != end - self.position:
            raise OSError("truncated wheel byte range")
        self.position = end
        return data


def download_headers(headers: Path) -> None:
    """Fetch only includes and wheel metadata, publishing a folder after successful extraction."""
    headers.mkdir(parents=True, exist_ok=True)
    for wheel, version in WHEELS.items():
        target = headers / wheel
        if target.exists():
            continue
        with urlopen(f"https://pypi.org/pypi/{wheel}/{version}/json", timeout=30) as response:
            metadata = Node(json.load(response))
        wheels = [
            item
            for item in metadata.children("urls")
            if "manylinux" in item.text("filename")
            and "x86_64" in item.text("filename")
            and item.text("filename").endswith(".whl")
        ]
        selected = sorted(wheels, key=lambda item: item.text("filename"))[0]
        with TemporaryDirectory(dir=headers) as temporary:
            tree = Path(temporary) / "tree"
            with RangeFile(selected.text("url")) as source:
                try:
                    archive = zipfile.ZipFile(source)
                except zipfile.BadZipFile as error:
                    raise OSError("invalid or truncated wheel archive") from error
                with archive:
                    for info in archive.infolist():
                        parts = Path(info.filename).parts
                        if ".." in parts or Path(info.filename).is_absolute():
                            raise ValueError("unsafe wheel path")
                        if not info.is_dir() and (
                            "/include/" in info.filename or info.filename.endswith(".dist-info/METADATA")
                        ):
                            path = tree / info.filename
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(archive.read(info))
            Path(temporary).rename(target)
        logger.info("extracted %s %s headers", wheel, version)


@dataclass
class Node:
    """Read the small part of clang's heterogeneous JSON needed for C declarations."""

    data: dict[str, Json]

    def text(self, key: str) -> str:
        value = self.data.get(key, "")
        return value if isinstance(value, str) else ""

    def child(self, key: str) -> Node:
        value = self.data.get(key, {})
        return Node(value if isinstance(value, dict) else {})

    def children(self, key: str = "inner") -> list[Node]:
        value = self.data.get(key, [])
        return [Node(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    @property
    def ctype(self) -> str:
        return self.child("type").text("qualType")

    def descendants(self) -> list[Node]:
        return [self, *(item for child in self.children() for item in child.descendants())]


@dataclass
class Declaration:
    """A type the headers declare: `kind` is enum, record, handle, callback or alias."""

    name: str
    home: str
    kind: str
    node: Node
    target: str = ""
    spelling: str = ""


@dataclass
class Counts:
    """What one generated module holds and what the generator left out, with the reason."""

    declared: int = 0
    functions: int = 0
    enums: int = 0
    enum_members: int = 0
    structs: int = 0
    unions: int = 0
    handles: int = 0
    callbacks: int = 0
    aliases: int = 0
    constants: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    skipped_types: dict[str, str] = field(default_factory=dict)
    skipped_macros: Counter[str] = field(default_factory=Counter)
    status_functions: list[str] = field(default_factory=list)


@dataclass
class Report:
    libraries: dict[str, Counts]
    declarations: dict[str, Declaration]


def include_dirs(headers: Path) -> list[Path]:
    return sorted(path for path in headers.glob("**/nvidia/cu*/include") if path.is_dir())


def clang_arguments(headers: Path, prelude: Path) -> list[str]:
    clang = shutil.which("clang")
    if clang is None:
        raise FileNotFoundError("clang is required to parse CUDA headers")
    return [
        clang,
        "-x",
        "c",
        "-include",
        str(prelude),
        *(f"-I{path}" for path in include_dirs(headers)),
        "-Wno-everything",
    ]


def clean(ctype: str) -> str:
    for qualifier in ("const", "volatile", "restrict", "__restrict", "__restrict__"):
        ctype = replace_word(ctype, qualifier, "")
    return " ".join(ctype.split())


def identifier(name: str) -> str:
    return name + "_" if keyword.iskeyword(name) or name in RESERVED or name.startswith("c_") else name


class Catalog:
    """Every declaration of the six public headers, found by walking clang's JSON AST in document order."""

    def __init__(self, headers: Path) -> None:
        self.headers = headers
        self.declarations: dict[str, Declaration] = {}
        self.lookup: dict[str, str] = {}
        self.tags: dict[str, str] = {}
        self.system: dict[str, str] = {}
        self.functions: dict[str, dict[str, Node]] = {module: {} for module in MODULES}
        self.macros: dict[str, dict[str, str]] = {}
        self.constants: dict[str, dict[str, int]] = {module: {} for module in MODULES}
        self.skipped_macros: dict[str, Counter[str]] = {module: Counter() for module in MODULES}
        self.versions: dict[str, str] = {}
        self.wheel_roots: dict[str, Path] = {}

    def parse(self, temporary: Path) -> None:
        prelude = temporary / "prelude.h"
        prelude.write_text(PRELUDE)
        arguments = clang_arguments(self.headers, prelude)
        self._read_versions()
        for module, spec in MODULES.items():
            header = next(path / spec.header for path in include_dirs(self.headers) if (path / spec.header).exists())
            ast = subprocess.run(
                [*arguments, "-Xclang", "-ast-dump=json", "-fsyntax-only", str(header)],
                capture_output=True,
                check=True,
                timeout=90,
            )
            self._collect(Node(json.loads(ast.stdout)).children())
            macros = subprocess.run(
                [*arguments, "-dM", "-E", str(header)],
                capture_output=True,
                text=True,
                check=True,
                timeout=90,
            )
            self.macros[module] = macro_definitions(macros.stdout)
            self._evaluate_macros(module, spec, header, arguments, temporary)

    def _evaluate_macros(self, module: str, spec: Spec, header: Path, arguments: list[str], temporary: Path) -> None:
        """Library macros that are integer constant expressions, valued by the compiler as enumerators.

        An enumerator per macro in a file of its own, since a macro that is not an integer (a string, an empty include
        guard, an attribute) makes clang report an error for that one enumerator and no value, and no other.
        """
        functions, skipped = self.functions[module], self.skipped_macros[module]
        candidates = []
        for macro, body in sorted(self.macros[module].items()):
            if not macro.startswith(spec.constants) or body in functions:
                continue
            if "(" in macro:
                skipped["function-like"] += 1
            elif not body:
                skipped["empty, an include guard"] += 1
            else:
                candidates.append(macro)
        probe = temporary / f"{module}.c"
        probe.write_text(
            f'#include "{header}"\n' + "".join(f"enum {{ krn_{i} = ({name}) }};\n" for i, name in enumerate(candidates))
        )
        ast = subprocess.run(
            [*arguments, "-Xclang", "-ast-dump=json", "-fsyntax-only", str(probe)],
            capture_output=True,
        )
        values = {
            member.text("name"): item.text("value")
            for node in Node(json.loads(ast.stdout)).children()
            for member in node.children()
            if member.text("kind") == "EnumConstantDecl" and member.text("name").startswith("krn_")
            for item in member.descendants()
            if item.text("kind") == "ConstantExpr" and item.text("value")
        }
        for index, macro in enumerate(candidates):
            if value := values.get(f"krn_{index}"):
                self.constants[module][macro] = int(value)
            else:
                skipped["not an integer constant expression"] += 1

    def _read_versions(self) -> None:
        for metadata in sorted(self.headers.glob("**/*.dist-info/METADATA")):
            fields = dict(line.split(": ", 1) for line in metadata.read_text().splitlines() if ": " in line)
            wheel = "-".join(
                part for part in fields["Name"].lower().replace("_", "-").replace(".", "-").split("-") if part
            )
            if wheel in WHEELS:
                self.versions[wheel] = fields["Version"]
                self.wheel_roots[wheel] = self.headers / metadata.relative_to(self.headers).parts[0]
        for wheel in WHEELS:
            if wheel not in self.versions:
                raise FileNotFoundError(f"{wheel} METADATA not found below {self.headers}")

    def _home(self, file: str) -> str:
        if home := HOMES.get(Path(file).name):
            return home
        if any(str(self.wheel_roots[wheel]) in file for wheel in ("nvidia-cuda-runtime", "nvidia-cuda-crt")):
            return "common"
        return ""

    def _collect(self, nodes: list[Node]) -> None:
        """clang names a file only when it changes, so the current file is carried along the walk."""
        file = ""
        tagged: dict[str, tuple[Node, str]] = {}
        typedefs: list[tuple[Node, str]] = []
        for node in nodes:
            for location in (
                node.child("range").child("begin"),
                node.child("range").child("end"),
                node.child("loc"),
            ):
                for loc in (
                    location,
                    location.child("spellingLoc"),
                    location.child("expansionLoc"),
                ):
                    file = loc.text("file") or file
            home, kind, name = self._home(file), node.text("kind"), node.text("name")
            if kind in {"RecordDecl", "EnumDecl"}:
                tagged[node.text("id")] = node, home
            if kind == "TypedefDecl":
                if home:
                    typedefs.append((node, home))
                elif name:
                    self.system[name] = node.ctype
            if home in self.functions and kind == "FunctionDecl":
                self.functions[home].setdefault(name, node)
        owners = self._tag_owners(tagged, [node for node, _ in typedefs])
        self._add_tags(tagged, owners)
        self._add_typedefs(typedefs)

    @staticmethod
    def _tag_owners(tagged: dict[str, tuple[Node, str]], typedefs: list[Node]) -> dict[str, str]:
        """The typedef that names each anonymous struct or enum, by the id of the tag it declares.

        clang 21 says so with `ownedTagDecl` on the typedef's type; clang 23 leaves it null and names the tag's `decl`.
        """
        owners: dict[str, str] = {}
        for node in typedefs:
            tag = next(
                (
                    found
                    for desc in node.descendants()
                    for found in (
                        desc.child("ownedTagDecl").text("id"),
                        desc.child("decl").text("id"),
                    )
                    if found
                ),
                "",
            )
            direct = clean(node.ctype)
            if not tag and direct.startswith(("struct ", "union ", "enum ")) and "*" not in direct:
                tag = next(
                    (key for key, (item, _) in tagged.items() if item.text("name") == direct.split(" ", 1)[1]),
                    "",
                )
            if tag and "*" not in direct:
                owners.setdefault(tag, node.text("name"))
        return owners

    def _add_tags(self, tagged: dict[str, tuple[Node, str]], owners: dict[str, str]) -> None:
        for key, (node, home) in tagged.items():
            name = owners.get(key, node.text("name"))
            if not home or not name:
                continue
            kind = "enum" if node.text("kind") == "EnumDecl" else "record"
            spelling = f"{node.text('tagUsed') or 'enum'} {node.text('name')}" if node.text("name") else name
            declaration = Declaration(name, home, kind, node, spelling=spelling)
            if name not in self.declarations or node.data.get("completeDefinition"):
                self.declarations[name] = declaration
            self.lookup[name] = name
            if node.text("name"):
                self.tags[node.text("name")] = name

    def _add_typedefs(self, typedefs: list[tuple[Node, str]]) -> None:
        for node, home in typedefs:
            name, target = node.text("name"), clean(node.ctype)
            if name in self.declarations and self.declarations[name].kind in {"enum", "record"}:
                continue
            kind = "callback" if "(*" in target else "alias"
            if target.endswith("*") and target.count("*") == 1:
                base = target[:-1].strip().removeprefix("struct ").removeprefix("union ")
                canonical = self.lookup.get(base, base)
                if target.startswith("struct ") or (
                    canonical in self.declarations and self.declarations[canonical].kind == "record"
                ):
                    kind = "handle"
            self.declarations.setdefault(name, Declaration(name, home, kind, node, target, name))
            self.lookup[name] = name

    def enum_values(self, declaration: Declaration) -> tuple[dict[str, int], list[str]]:
        """The members of an enum with their values, and the names that are not Python identifiers."""
        values: dict[str, int] = {}
        invalid: list[str] = []
        value = -1
        for member in declaration.node.children():
            if member.text("kind") != "EnumConstantDecl":
                continue
            name = member.text("name")
            if not name.isidentifier() or keyword.iskeyword(name):
                invalid.append(name)
                continue
            explicit = next(
                (
                    item.text("value")
                    for item in member.descendants()
                    if item.text("kind") == "ConstantExpr" and item.text("value")
                ),
                "",
            )
            value = int(explicit) if explicit else value + 1
            values[name] = value
        return values, invalid


class Resolver:
    """Map a C type string to (ctypes expression, Python annotation), recording which declarations a module uses."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.used: dict[str, set[str]] = defaultdict(set)

    def resolve(self, ctype: str, module: str, trail: tuple[str, ...] = ()) -> tuple[str, str]:
        catalog = self.catalog
        value = clean(ctype)
        if "(*" in value:
            raise ValueError("function-pointer parameter")
        array, separator, size = value.removesuffix("]").rpartition("[")
        if value.endswith("]") and separator and array and size.isdecimal():
            native, _ = self.resolve(array.strip(), module, trail)
            return f"{native} * {size}", f"ctypes.Array[{native}]"
        value = value.removesuffix("[]") + ("*" if value.endswith("[]") else "")
        depth = value.count("*")
        base = value.replace("*", "").strip()
        tagged = base.startswith(("struct ", "union ", "enum "))
        base = base.partition(" ")[2] if tagged else base
        canonical = (catalog.tags if tagged else catalog.lookup).get(base, base)
        declaration = catalog.declarations.get(canonical)
        if depth:
            return self._pointer(depth, base, canonical, declaration, module)
        if canonical in SCALARS:
            return SCALARS[canonical]
        if declaration:
            self.used[module].add(canonical)
            if declaration.kind == "enum":
                return "c_int32", canonical
            if declaration.kind in {"record", "handle", "callback"}:
                return canonical, "CFuncPtr" if declaration.kind == "callback" else canonical
            if canonical in trail:
                raise ValueError(f"unresolved type: recursive {canonical}")
            native, annotation = self.resolve(declaration.target, module, (*trail, canonical))
            return native, canonical if annotation in PYTHON_SCALARS else annotation
        if canonical in catalog.system:
            return self.resolve(catalog.system[canonical], module, (*trail, canonical))
        raise ValueError(f"unresolved type: {value}")

    def _pointer(
        self, depth: int, base: str, canonical: str, declaration: Declaration | None, module: str
    ) -> tuple[str, str]:
        """A device address or host out-parameter is `c_void_p`; only handles and descriptor structs keep a type."""
        if depth == 1 and base == "char":
            return "c_char_p", "bytes | None"
        if depth >= 2:
            if base == "char":
                return "POINTER(c_char_p)", "Ref[ctypes.c_char_p]"
            return "POINTER(c_void_p)", "Ref[ctypes.c_void_p]"
        described = declaration is not None and (
            declaration.kind == "handle"
            or (
                declaration.kind == "record"
                and declaration.node.data.get("completeDefinition")
                and canonical not in NUMERIC
            )
        )
        if described:
            self.used[module].add(canonical)
            return f"POINTER({canonical})", f"Ref[{canonical}]"
        return "c_void_p", "Address"


class Writer:
    """Render types and functions as Python source, assemble one module per library and format them."""

    def __init__(self, catalog: Catalog, resolver: Resolver) -> None:
        self.catalog, self.resolver = catalog, resolver
        self.counts = {module: Counts() for module in ("common", *MODULES)}

    def write(self, out: Path) -> Report:
        bodies = {module: self._functions(module) for module in MODULES}
        rendered = self._types()
        out.mkdir(parents=True, exist_ok=True)
        modules = ("common", *MODULES)
        for module in modules:
            local = {
                name: rendered[name]
                for name, declaration in self.catalog.declarations.items()
                if declaration.home == module and rendered.get(name)
            }
            (out / f"{module}.py").write_text(self._module(module, local, bodies.get(module, [])))
        self._format([out / f"{module}.py" for module in modules])
        return Report(self.counts, self.catalog.declarations)

    def _functions(self, module: str) -> list[str]:
        counts, bodies = self.counts[module], []
        counts.declared = len(self.catalog.functions[module])
        for name, node in self.catalog.functions[module].items():
            try:
                bodies.append(self._function(name, node, module))
                counts.functions += 1
            except ValueError as error:
                counts.skipped[name] = str(error)
        self.resolver.resolve(MODULES[module].status, module)
        self.resolver.resolve("libraryPropertyType", module)
        return bodies

    def _types(self) -> dict[str, str]:
        """Every type of a library header, and the runtime types something reaches, rendered from the used sets."""
        used = self.resolver.used
        rendered: dict[str, str] = {}
        pending = list(self.catalog.declarations)
        while pending:
            name = pending.pop(0)
            declaration = self.catalog.declarations[name]
            if declaration.home == "common" and not any(name in reached for reached in used.values()):
                continue
            try:
                rendered[name] = self._type(declaration, declaration.home)
            except ValueError as error:
                self.counts[declaration.home].skipped_types[name] = str(error)
                rendered[name] = ""
            pending += [
                dependency
                for dependency in self.catalog.declarations
                if dependency in used[declaration.home]
                and dependency not in rendered
                and dependency not in pending
                and dependency != name
            ]
        return rendered

    def _type(self, declaration: Declaration, module: str) -> str:
        name, kind = declaration.name, declaration.kind
        if kind == "enum":
            values, invalid = self.catalog.enum_values(declaration)
            self.counts[module].skipped_types.update({member: "invalid enum member identifier" for member in invalid})
            return f"class {name}(IntEnum):\n" + (
                "\n".join(f"    {key} = {value}" for key, value in values.items()) or "    pass"
            )
        if kind == "handle":
            return f"class {name}(ctypes.c_void_p):\n    pass"
        if kind == "record":
            return self._record(declaration, module)
        if kind == "callback":
            result, marker, arguments = declaration.target.partition(" (*)(")
            if not result or not marker or not arguments.endswith(")"):
                raise ValueError(f"unresolved type: callback {declaration.target}")
            resolve = self.resolver.resolve
            parameters = [resolve(value.strip(), module)[0] for value in arguments[:-1].split(",") if value != "void"]
            return f"{name} = ctypes.CFUNCTYPE({', '.join((resolve(result, module)[0], *parameters))})"
        return self._alias(declaration, module)

    def _record(self, declaration: Declaration, module: str) -> str:
        name = declaration.name
        if not declaration.node.data.get("completeDefinition"):
            return ""
        annotations, members = [], []
        for item in (item for item in declaration.node.children() if item.text("kind") == "FieldDecl"):
            native, annotation = self.resolver.resolve(item.ctype, module)
            if item.data.get("isBitfield"):
                raise ValueError(f"unresolved type: bitfield {name}.{item.text('name')}")
            annotations.append(f"    {identifier(item.text('name'))}: {annotation}")
            members.append(f'("{item.text("name")}", {native})')
        parent = "Union" if declaration.node.text("tagUsed") == "union" else "Structure"
        alignment = next(
            (
                desc.text("value")
                for attr in declaration.node.children()
                if attr.text("kind") == "AlignedAttr"
                for desc in attr.descendants()
                if desc.text("kind") == "ConstantExpr"
            ),
            "",
        )
        if alignment:
            annotations.insert(0, f"    _align_ = {int(alignment)}")
        return f"class {name}(ctypes.{parent}):\n" + "\n".join(annotations) + f"\n    _fields_ = [{', '.join(members)}]"

    def _alias(self, declaration: Declaration, module: str) -> str:
        name = declaration.name
        declarations = self.catalog.declarations
        native, annotation = self.resolver.resolve(declaration.target, module, (name,))
        if annotation in declarations and declarations[annotation].kind == "enum":
            return f"{name} = {annotation}"
        if annotation in PYTHON_SCALARS or annotation in declarations:
            if native not in declarations or annotation in PYTHON_SCALARS:
                return f"type {name} = {annotation}"
            return f"{name} = {annotation}"
        return f"{name} = {native}"

    def _function(self, name: str, node: Node, module: str) -> str:
        if node.data.get("inline") or node.text("storageClass") == "static":
            raise ValueError("inline/header-only")
        if node.ctype.endswith("...)"):
            raise ValueError("variadic")
        declarations = self.catalog.declarations
        resolve = self.resolver.resolve
        parameters = [child for child in node.children() if child.text("kind") == "ParmVarDecl"]
        arguments, annotations = [], []
        for index, parameter in enumerate(parameters):
            native, annotation = resolve(parameter.ctype, module)
            arguments.append(native)
            annotations.append(f"{identifier(parameter.text('name') or f'arg{index}')}: {annotation}")
        result = node.ctype.partition("(")[0].strip()
        if result == MODULES[module].status:
            self.counts[module].status_functions.append(name)
            decorator, annotation = f"status({', '.join(arguments)})", "None"
        else:
            native, annotation = resolve(result, module)
            if native == "c_int32" and annotation in declarations:
                annotation = "int"
            if annotation not in {"None", "bytes | None", *PYTHON_SCALARS} and not (
                annotation in declarations and declarations[annotation].kind == "handle"
            ):
                raise ValueError(f"unresolved return type: {result}")
            decorator = f"returns({', '.join((native, *arguments))})"
        implementation = "..." if annotation in {"None", "bytes | None"} else "\n    raise NotImplementedError"
        return f"@_library.{decorator}\ndef {name}({', '.join(annotations)}) -> {annotation}: {implementation}"

    def _order(self, local: dict[str, str]) -> list[str]:
        """Dependencies first: a type is defined after every type its source mentions."""
        ordered: list[str] = []
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in ordered:
                return
            if name in visiting:
                raise ValueError(f"cyclic by-value declarations: {name}")
            visiting.add(name)
            for dependency in local:
                if dependency != name and dependency in words(local[name]):
                    visit(dependency)
            visiting.remove(name)
            ordered.append(name)

        for name in local:
            visit(name)
        return ordered

    def _module(self, module: str, local: dict[str, str], functions: list[str]) -> str:
        counts = self.counts[module]
        ordered = self._order(local)
        for name in ordered:
            self._count(counts, self.catalog.declarations[name])
        body = "\n\n\n".join(local[name] for name in ordered)
        if module == "common":
            wheel, header = "nvidia-cuda-runtime", "reachable CUDA runtime headers"
        else:
            spec = MODULES[module]
            wheel, header = spec.wheel, spec.header
            body += self._constants(module) + self._library(module, spec, functions)
        version = self.catalog.versions[wheel]
        banner = (
            f'"""{module}: names and signatures of {header} in {wheel} {version}.\n\n'
            'Generated by experiments/gen_cuda_bindings.py; do not edit.\n"""\n\n'
        )
        imports = "\n".join(self._imports(module, body))
        return banner + imports.rstrip() + f'\n\nHEADER_VERSION = "{version}"\n\n\n' + body + "\n"

    def _count(self, counts: Counts, declaration: Declaration) -> None:
        if declaration.kind == "enum":
            counts.enums += 1
            counts.enum_members += len(self.catalog.enum_values(declaration)[0])
        elif declaration.kind == "record":
            if declaration.node.text("tagUsed") == "union":
                counts.unions += 1
            else:
                counts.structs += 1
        else:
            attribute = {"handle": "handles", "callback": "callbacks", "alias": "aliases"}[declaration.kind]
            setattr(counts, attribute, getattr(counts, attribute) + 1)

    def _constants(self, module: str) -> str:
        """The integer macros the compiler evaluated, as module constants."""
        counts, catalog = self.counts[module], self.catalog
        counts.constants, counts.skipped_macros = (
            catalog.constants[module],
            catalog.skipped_macros[module],
        )
        if not counts.constants:
            return ""
        return "\n\n\n" + "\n".join(f"{name} = {value}" for name, value in counts.constants.items())

    def _library(self, module: str, spec: Spec, functions: list[str]) -> str:
        """The error class, the `Library`, its functions, the macro names and `version()` of one module."""
        counts = self.counts[module]
        macros = self.catalog.macros[module]
        options = f', message="{spec.message}"' if spec.message else ""
        soname = f"lib{spec.prefix}.so.{macros[spec.version]}"
        text = (
            f"\n\n\nclass {spec.error}(LibraryError):\n"
            f'    """A {spec.prefix} call returned a nonzero {spec.status}."""\n'
            f'\n\n_library = Library("{spec.prefix}", "{soname}", "{spec.wheel}", '
            f"status={spec.status}, error={spec.error}{options})\n\n\n" + "\n\n\n".join(functions)
        )
        for alias, target in sorted(macros.items()):
            if target in self.catalog.functions[module] and target not in counts.skipped and alias != target:
                text += f"\n\n{alias} = {target}"
                counts.aliases += 1
        property_name = f"{spec.prefix}GetProperty"
        return text + (
            f"\n\n\ndef version() -> tuple[int, int, int]:\n"
            f'    """(major, minor, patch) from {property_name}."""\n'
            f"    return property_version({property_name}, libraryPropertyType)\n"
        )

    def _imports(self, module: str, body: str) -> list[str]:
        imports = []
        if "ctypes." in body:
            imports.append("import ctypes")
        if "CFuncPtr" in body:
            imports.append("from _ctypes import CFuncPtr")
        known = {"POINTER", "c_char_p", "c_void_p", *(native for native, _ in SCALARS.values())}
        names = sorted(
            set(unqualified_words(body)) & known,
            key=numeric_key,
        )
        if names:
            imports.append(f"from ctypes import {', '.join(names)}")
        if "IntEnum" in body:
            imports.append("from enum import IntEnum")
        imports.append("")
        origins: dict[str, set[str]] = defaultdict(set)
        for name in self.resolver.used[module]:
            origin = self.catalog.declarations[name].home
            if origin != module and name in words(body):
                origins[origin].add(name)
        imports += [f"from .{origin} import {', '.join(sorted(origins[origin]))}" for origin in sorted(origins)]
        loader = [
            name for name in ("Address", "Library", "LibraryError", "Ref", "property_version") if name in words(body)
        ]
        if loader:
            imports.append(f"from .loader import {', '.join(loader)}")
        return imports

    @staticmethod
    def _format(paths: list[Path]) -> None:
        ruff = shutil.which("ruff")
        if ruff is None:
            raise FileNotFoundError("ruff is required to format generated bindings")
        subprocess.run(
            [
                ruff,
                "format",
                "--isolated",
                "--line-length",
                "120",
                "--target-version",
                "py314",
                *map(str, paths),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )


def generate(headers: Path, out: Path) -> Report:
    """Write one module per library below `out` from the headers below `headers`."""
    catalog = Catalog(headers)
    with TemporaryDirectory(prefix="krn-cuda-") as temporary:
        catalog.parse(Path(temporary))
    report = Writer(catalog, Resolver(catalog)).write(out)
    logger.info("library    functions/declared enums members structs unions handles callbacks aliases constants")
    for module, counts in report.libraries.items():
        logger.info(
            "%-10s %4d/%-4d %5d %7d %7d %6d %7d %9d %7d %9d",
            module,
            counts.functions,
            counts.declared,
            counts.enums,
            counts.enum_members,
            counts.structs,
            counts.unions,
            counts.handles,
            counts.callbacks,
            counts.aliases,
            len(counts.constants),
        )
        for name, reason in (counts.skipped | counts.skipped_types).items():
            logger.info("  skipped %s: %s", name, reason)
        for reason, count in counts.skipped_macros.items():
            logger.info("  skipped %d macros: %s", count, reason)
    return report


@app.command(name="generate")
def command(*, headers: Path, out: Path, download: bool = False) -> None:
    """Generate the bindings; `--download` first fills `headers` with the pinned wheels' include trees."""
    if download:
        download_headers(headers)
    generate(headers, out)


def macro_definitions(source: str) -> dict[str, str]:
    """Read object-like and function-like definitions from clang's macro dump."""
    result: dict[str, str] = {}
    for line in source.splitlines():
        if not line.startswith("#define "):
            continue
        definition = line.removeprefix("#define ")
        name, separator, value = definition.partition(" ")
        if "(" in name and ")" not in name:
            head, closing, tail = definition.partition(")")
            if not closing or not tail.startswith(" "):
                continue
            name, separator, value = head + closing, " ", tail[1:]
        if separator:
            result[name] = value
    return result


def unqualified_words(source: str) -> list[str]:
    """Bare tokens without a preceding attribute dot."""
    result: list[str] = []
    start = 0
    while start < len(source):
        if not (source[start].isalnum() or source[start] == "_"):
            start += 1
            continue
        end = start + 1
        while end < len(source) and (source[end].isalnum() or source[end] == "_"):
            end += 1
        if start == 0 or source[start - 1] != ".":
            result.append(source[start:end])
        start = end
    return result


def numeric_key(name: str) -> str:
    """Pad digit runs for stable natural ordering of ctypes names."""
    parts: list[str] = []
    digits = ""
    for char in name:
        if char.isdecimal():
            digits += char
        else:
            if digits:
                parts.append(digits.zfill(6))
                digits = ""
            parts.append(char)
    if digits:
        parts.append(digits.zfill(6))
    return "".join(parts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app()

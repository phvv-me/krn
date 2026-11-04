"""Reads a kernel file with `ast`, never importing it (the API bodies are signature-only), and evaluates annotation
ASTs against the names the file imports from `krn`, so forward references and names that cannot be imported are
safe. Those names are the real objects of the package (`T.f16`, `T.fp.min`, `AUTO`, `Feature`), found by import."""

import ast
import inspect
import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import cache
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path
from types import FunctionType
from typing import cast

from krn.frontend.kernel import Kernel

from .dtypes import BOOL, PY_FLOAT, PY_INT, Dtype, Sketch, from_sketch, is_sketch_type
from .errors import CaptureError, Span
from .ir import TensorType

type FuncOrClass = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
type Resolved = Dtype | TensorType | StructDef

TENSOR_RANKS = {"Vector": 1, "Matrix": 2, "Tensor3": 3}
PY_NAMES = {"int": PY_INT, "float": PY_FLOAT, "bool": BOOL}
KERNEL_DECORATORS = frozenset({"kernel", "family", "variant"})
FOLDABLE = frozenset({"int", "bool"})


class Role(StrEnum):
    """What a use of a function makes it. Capture records the roles a function played; none is declared."""

    ENTRY = auto()  # launched from the host
    CALLEE = auto()  # called from a kernel and expanded where it is called
    CONFIGURED = auto()  # its first parameter is a `@struct` with `Const` fields: one specialization per configuration
    FOLDED = auto()  # called with capture-time constants only and evaluated while capturing


def plain[T](value: T) -> T | FunctionType:
    """The function behind a `Kernel`: capture reads its source, and the launch object is not part of it."""
    return cast(FunctionType, value.fn) if isinstance(value, Kernel) else value  # a Kernel wraps the function it marks


def decorators(node: FuncOrClass) -> list[str]:
    """Decorator names, looking through calls: `@requires(...)` is `requires`."""
    names = (d.func if isinstance(d, ast.Call) else d for d in node.decorator_list)
    return [n.id if isinstance(n, ast.Name) else n.attr for n in names if isinstance(n, ast.Name | ast.Attribute)]


def foldable(fn: ast.FunctionDef) -> bool:
    """Whether a call with capture-time constants is evaluated while capturing: the body is one `return` of an
    expression (after an optional docstring) and every parameter and the result is an `int` or a `bool`."""
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    arguments = fn.args
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and body[0].value is not None
        and not (arguments.defaults or arguments.kwonlyargs or arguments.vararg or arguments.kwarg)
        and all(
            isinstance(a, ast.Name) and a.id in FOLDABLE for a in [*(p.annotation for p in arguments.args), fn.returns]
        )
    )


@dataclass(frozen=True, slots=True)
class FieldDef:
    name: str
    annotation: ast.expr
    default: ast.expr | None
    span: Span

    @property
    def is_const(self) -> bool:
        return isinstance(self.annotation, ast.Subscript) and ast.unparse(self.annotation.value).endswith("Const")

    @property
    def is_float(self) -> bool:
        """Whether a `Const[float]` field, the one kind of constant that is not an integer."""
        return isinstance(self.annotation, ast.Subscript) and ast.unparse(self.annotation.slice) == "float"


@dataclass(frozen=True, slots=True)
class StructDef:
    name: str
    type_params: tuple[str, ...]
    fields: tuple[FieldDef, ...]
    methods: dict[str, ast.FunctionDef]
    module: Module
    span: Span


class Module:
    """One parsed source file. Names it imports with a relative import are found lazily in the sibling files."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = str(path)
        self.tree = ast.parse(path.read_text(), filename=self.file)
        self.defs: dict[str, FuncOrClass] = {
            n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        self.imports: dict[str, tuple[Path, str]] = {
            (a.asname or a.name): (self._sibling(n), a.name)
            for n in self.tree.body
            if isinstance(n, ast.ImportFrom) and n.level
            for a in n.names
        }
        self.names = self._package_names()

    def _sibling(self, node: ast.ImportFrom) -> Path:
        base = self.path.parent.joinpath(*[".."] * (node.level - 1))
        return base.joinpath(*(node.module or "").split(".")).with_suffix(".py")

    def _package_names(self) -> dict[str, Sketch]:
        """Resolve `krn` imports and itertools' chain spelling. Missing package names stay undefined."""
        found: dict[str, Sketch] = {}
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "itertools":
                        found[alias.asname or alias.name] = itertools
                    elif alias.name == "krn" or alias.name.startswith("krn."):
                        package = import_module(alias.name)
                        found[alias.asname or "krn"] = package if alias.asname else import_module("krn")
            if isinstance(node, ast.ImportFrom) and not node.level and node.module == "itertools":
                for alias in node.names:
                    if alias.name == "chain":
                        found[alias.asname or alias.name] = itertools.chain
            if not isinstance(node, ast.ImportFrom):
                continue
            module_name = node.module or ""
            root = Path(__file__).resolve().parents[2]
            if node.level and self.path.is_relative_to(root / "krn"):
                package_name = ".".join(self.path.parent.relative_to(root).parts)
                module_name = resolve_name("." * node.level + module_name, package_name)
            if module_name.split(".")[0] == "krn":
                package = import_module(module_name)
                for alias in node.names:
                    try:
                        found[alias.asname or alias.name] = plain(getattr(package, alias.name))
                    except AttributeError:
                        try:
                            found[alias.asname or alias.name] = import_module(f"{package.__name__}.{alias.name}")
                        except ModuleNotFoundError:
                            continue
        return found

    @classmethod
    @cache
    def read(cls, path: Path) -> Module:
        return cls(path.resolve())

    def span(self, node: ast.AST) -> Span:
        return Span.of(self.file, node)

    def lookup(self, name: str) -> tuple[Module, FuncOrClass] | None:
        if name in self.defs:
            return self, self.defs[name]
        if name in self.imports:
            file, original = self.imports[name]
            found = Module.read(file).lookup(original) if file.is_file() else None
            package = Path(__file__).resolve().parents[1]
            if (
                file.resolve().is_relative_to(package)
                and found
                and isinstance(found[1], ast.FunctionDef)
                and not KERNEL_DECORATORS & set(decorators(found[1]))
            ):
                return None
            return found
        value = self.names.get(name)
        if isinstance(value, FunctionType):
            file = inspect.getsourcefile(value)
            if file is not None:
                found = Module.read(Path(file)).lookup(value.__name__)
                if found and KERNEL_DECORATORS & set(decorators(found[1])):
                    return found
        return None

    def kernel(self, name: str) -> ast.FunctionDef:
        module, fn = self.lookup(name) or (self, None)
        if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            module.check_decorators(fn)
        if isinstance(fn, ast.AsyncFunctionDef):
            raise CaptureError(module.span(fn), "async kernels are not lowered in this slice")
        if not isinstance(fn, ast.FunctionDef) or not KERNEL_DECORATORS & set(decorators(fn)):
            raise CaptureError(Span(self.file, 1, 1), f"no @kernel function named `{name}`")
        return fn

    def check_decorators(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for name in decorators(fn):
            if name in {"device", "template", "comptime"}:
                raise CaptureError(self.span(fn), f"decorator {name} was removed; write @kernel")

    def struct(self, node: ast.ClassDef) -> StructDef:
        """Merge fields and methods from bases in this file.

        Generic inheritance is positional: subclass and base declare the same parameter names in the same order,
        and a subscripted base passes those names unchanged.
        """
        parameters = tuple(p.name for p in node.type_params if isinstance(p, ast.TypeVar))
        bases = [
            (base, self.struct(parent))
            for base in node.bases
            for name in [base.value if isinstance(base, ast.Subscript) else base]
            if isinstance(name, ast.Name) and isinstance(parent := self.defs.get(name.id), ast.ClassDef)
        ]
        for annotation, base in bases:
            arguments = annotation.slice if isinstance(annotation, ast.Subscript) else None
            names = (
                tuple(ast.unparse(a) for a in arguments.elts)
                if isinstance(arguments, ast.Tuple)
                else ((ast.unparse(arguments),) if arguments is not None else ())
            )
            if base.type_params and (parameters != base.type_params or names != parameters):
                raise CaptureError(
                    self.span(annotation),
                    f"subclass {node.name} and generic base {base.name} must have the same number of type parameters "
                    "in the same order, passed positionally unchanged",
                )
        fields = tuple(
            FieldDef(n.target.id, n.annotation, n.value, self.span(n))
            for n in node.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        )
        methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
        return StructDef(
            node.name,
            parameters,
            (*(field for _, base in bases for field in base.fields), *fields),
            {name: method for _, base in bases for name, method in base.methods.items()} | methods,
            self,
            self.span(node),
        )

    def sketch(self, node: ast.expr) -> Sketch | None:
        """What a name or a dotted name refers to in `krn` through this file's imports, or None."""
        match node:
            case ast.Name(id=name):
                return self.names.get(name)
            case ast.Attribute(value=base, attr=attr) if (head := self.sketch(base)) is not None:
                return plain(getattr(head, attr, None))
        return None

    def evaluate(self, node: ast.expr) -> Sketch | None:
        """A compile-time value built from literals and `krn` names only: `T.fp.emax(8)`, `T.uint[T.L[3]]`."""
        try:
            return eval(compile(ast.Expression(node), self.file, "eval"), {"__builtins__": {}}, self.names)
        except (NameError, AttributeError, TypeError, LookupError) as error:
            raise CaptureError(self.span(node), f"cannot evaluate `{ast.unparse(node)}` at capture: {error}") from None

    def resolve(self, node: ast.expr, types: Mapping[str, Dtype] | None = None) -> Resolved:
        """Evaluate an annotation: dtypes, `T.uint[T.L[N]]`, `T.fp[...]`, `Vector/Matrix/Tensor[...]`, `Const[...]`,
        structs of this file or its relative imports, type parameters bound in `types`, and forward references."""
        match node:
            case ast.Constant(value=str() as text):
                return self.resolve(ast.parse(text, mode="eval").body, types)
            case ast.Name(id=name) if types and name in types:
                return types[name]
            case ast.Name(id=name) if name in PY_NAMES:
                return PY_NAMES[name]
            case ast.BinOp(left=left, op=ast.BitOr(), right=right) if self.python_number(right) or self.python_number(
                left
            ):
                # `u32 | int`: the type, and a Python number of its kind is welcome where it is expected
                typed = left if self.python_number(right) else right
                return self.resolve(typed, types)
            case ast.Name(id=name) if getattr(self.names.get(name), "__name__", None) == "Idx":
                return PY_INT  # an index of any integer width: the caller's type is used
            case ast.Name(id=name) if (found := self.lookup(name)) and isinstance(found[1], ast.ClassDef):
                return found[0].struct(found[1])
            case ast.Subscript(value=ast.Name(id=name)) if (found := self.lookup(name)) and isinstance(
                found[1], ast.ClassDef
            ):
                return found[0].struct(found[1])
            case ast.Subscript(value=head, slice=arg) if (kind := getattr(self.sketch(head), "__name__", None)) in (
                "Vector",
                "Matrix",
                "Tensor3",
                "Tensor",
                "Const",
            ):
                args = arg.elts if isinstance(arg, ast.Tuple) else [arg]
                match kind:
                    case "Const":
                        return self.resolve(args[0], types)
                    case "Vector" | "Matrix" | "Tensor3":
                        return TensorType(self.dtype(args[0], types), TENSOR_RANKS[kind])
                    case _:
                        return TensorType(self.dtype(args[0], types), len(args) - 1)
        return self.sketch_dtype(node)

    def python_number(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id in ("int", "float")

    def sketch_dtype(self, node: ast.expr) -> Dtype:
        """The dtype a type expression of `krn` names stands for: `T.f16`, `T.uint[T.L[3]]`."""
        try:
            found = self.evaluate(node)
        except CaptureError:
            found = None
        if not is_sketch_type(found):
            raise CaptureError(self.span(node), f"unsupported annotation `{ast.unparse(node)}`")
        return from_sketch(found)

    def dtype(self, node: ast.expr, types: Mapping[str, Dtype] | None = None) -> Dtype:
        resolved = self.resolve(node, types)
        if not isinstance(resolved, Dtype):
            raise CaptureError(self.span(node), f"`{ast.unparse(node)}` is not a number type")
        return resolved

    def literal(self, node: ast.expr) -> int:
        """The integer of `L[3]`, `T.L[3]`, `Literal[3]` or a bare 3."""
        match node:
            case ast.Constant(value=int() as value) | ast.Subscript(slice=ast.Constant(value=int() as value)):
                return value
        raise CaptureError(self.span(node), f"expected a type-level integer like L[8], got `{ast.unparse(node)}`")

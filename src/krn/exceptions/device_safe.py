"""Which exception classes a kernel can construct on a device, and the reason when they cannot."""

import ast
import inspect
import textwrap
from annotationlib import Format
from collections.abc import Hashable
from dataclasses import MISSING, Field, dataclass, field, fields, make_dataclass
from types import BuiltinFunctionType, FunctionType, MethodDescriptorType, NoneType, WrapperDescriptorType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

from .schemas import SCHEMAS, Schema, is_builtin

type Member = Hashable | list | dict | set

CONSTANTS = (int, float, bool, str, NoneType)
PARAMETER_TYPES = (int, float, bool, str)
HOOKS = (
    "__new__",
    "__setattr__",
    "__delattr__",
    "__getattribute__",
    "__getattr__",
    "__str__",
    "__post_init__",
    "__init_subclass__",
    "__class_getitem__",
)
ALLOWED = "only `self.name = x` and `super().__init__(x, ...)` are allowed, with each x a parameter or a constant"


@dataclass(frozen=True, kw_only=True)
class Reason:
    """Why a class cannot be raised on a device: `text` names the construct that cannot run there."""

    text: str


def device_safe(cls: type[BaseException]) -> Reason | None:
    """None when a kernel may raise cls on a device, else the first construct that cannot run there.

    The construction path is the metaclass, every constructor hook and Python __init__ in the MRO, the calls each
    __init__ makes to its parent, and the class attributes an instance reads or writes. A dataclass's generated code is
    accepted only when its bytecode equals what dataclasses generates for the same fields.
    """
    return _ConstructionPath(cls).reason()


class _ConstructionPath:
    """The checks of one class's construction path, in the order their reasons are most useful."""

    def __init__(self, cls: type[BaseException]) -> None:
        self._cls = cls
        self._python = [klass for klass in cls.__mro__ if not is_builtin(klass)]
        self._terminal = next(klass for klass in cls.__mro__ if is_builtin(klass))
        self._schema = SCHEMAS.get(self._terminal)

    def reason(self) -> Reason | None:
        if type(self._cls) is not type:
            return Reason(
                text=f"{self._cls.__qualname__} has metaclass {type(self._cls).__qualname__}, "
                "which runs Python code when the class is called"
            )
        if self._schema is None:
            return Reason(
                text=f"{self._terminal.__qualname__} has no device constructor schema, so a kernel cannot raise "
                f"{self._cls.__qualname__}"
            )
        return self._hooks() or self._inits() or self._attributes()

    def _hooks(self) -> Reason | None:
        for klass in self._python:
            for name in HOOKS:
                if name in vars(klass) and not _c_callable(vars(klass)[name]) and not _generated(klass, name):
                    return Reason(
                        text=f"{klass.__qualname__}.{name} is defined in Python, which cannot run on a device"
                    )
        return None

    def _inits(self) -> Reason | None:
        for klass in self._python:
            if "__init__" in vars(klass) and (reason := self._init(klass, vars(klass)["__init__"])):
                return reason
        return None

    def _init(self, klass: type, init: Member) -> Reason | None:
        if _c_callable(init):
            return None
        if not isinstance(init, FunctionType):
            return Reason(text=f"{klass.__qualname__}.__init__ is a {type(init).__name__}, which runs Python code")
        if hasattr(init, "__wrapped__"):
            return Reason(text=f"{klass.__qualname__}.__init__ wraps another function, which cannot be checked")
        if _generated(klass, "__init__"):
            return _generated_fields(klass) or self._signature(klass, init)
        return self._signature(klass, init) or self._body(klass, init)

    def _signature(self, klass: type, init: FunctionType) -> Reason | None:
        parameters = list(inspect.signature(init, annotation_format=Format.FORWARDREF).parameters.values())
        if not parameters or parameters[0].kind not in _POSITIONAL:
            return Reason(text=f"{klass.__qualname__}.__init__ must take self as a positional parameter")
        for parameter in parameters[1:]:
            if parameter.kind not in _POSITIONAL:
                return Reason(
                    text=f"{klass.__qualname__}.__init__ parameter '{parameter.name}' is not positional; "
                    "a device raise passes positional arguments only"
                )
            if parameter.annotation is inspect.Parameter.empty:
                return Reason(text=f"{klass.__qualname__}.__init__ parameter '{parameter.name}' is not annotated")
            if parameter.annotation not in PARAMETER_TYPES:
                return Reason(
                    text=f"{klass.__qualname__}.__init__ parameter '{parameter.name}' is annotated "
                    f"{parameter.annotation!r}; a device error takes int, float, bool, or a constant str"
                )
        return None

    def _body(self, klass: type, init: FunctionType) -> Reason | None:
        try:
            source = inspect.getsource(init)
        except OSError, TypeError:
            return Reason(text=f"{klass.__qualname__}.__init__ has no source to check")
        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError:
            return Reason(text=f"{klass.__qualname__}.__init__ has source that does not parse")
        function = tree.body[0]
        if not isinstance(function, ast.FunctionDef) or function.decorator_list:
            return Reason(text=f"{klass.__qualname__}.__init__ is not a plain def, so it cannot be checked")
        arguments = [*function.args.posonlyargs, *function.args.args]
        self_name = arguments[0].arg
        parameters = {argument.arg for argument in arguments[1:]}
        for statement in function.body:
            if not _device_statement(statement, self_name, parameters, self._schema):
                return Reason(
                    text=f"{klass.__qualname__}.__init__ statement '{ast.unparse(statement)}' at line "
                    f"{statement.lineno} runs code on a device; {ALLOWED}"
                )
        return None

    def _attributes(self) -> Reason | None:
        for klass in self._python:
            for name, value in vars(klass).items():
                if _dunder(name) or isinstance(value, FunctionType):
                    continue
                if not _constant(value):
                    return Reason(
                        text=f"{klass.__qualname__}.{name} is a {type(value).__name__}; a device class may hold only "
                        "int, float, bool, str, or None constants"
                    )
        return None


_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _c_callable(value: Member) -> bool:
    return isinstance(value, WrapperDescriptorType | BuiltinFunctionType | MethodDescriptorType)


def _dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _device_statement(statement: ast.stmt, self_name: str, parameters: set[str], schema: Schema | None) -> bool:
    match statement:
        case ast.Pass() | ast.Expr(value=ast.Constant(value=str())):
            return True
        case ast.Assign(targets=[ast.Attribute(value=ast.Name(id=target))], value=value):
            return target == self_name and _plain(value, parameters)
        case ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=ast.Call(func=ast.Name(id="super"), args=[], keywords=[]), attr="__init__"),
                args=args,
                keywords=[],
            )
        ):
            return schema is not None and schema.accepts(len(args)) and all(_plain(a, parameters) for a in args)
    return False


def _plain(node: ast.expr, parameters: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return _constant(node.value)
    return isinstance(node, ast.Name) and node.id in parameters


def _constant(value: Member) -> bool:
    """Whether value is exactly one of the constant types: a subclass may carry a descriptor that runs on assignment."""
    return type(value) in CONSTANTS


def _fields(klass: type) -> tuple[Field, ...]:
    """The fields of a class that _generated has shown dataclasses decorated, which is what makes this cast true."""
    return fields(cast("type[DataclassInstance]", klass))


def _generated_fields(klass: type) -> Reason | None:
    for item in _fields(klass):
        if item.default_factory is not MISSING:
            return Reason(
                text=f"{klass.__qualname__} field '{item.name}' has a default_factory, which runs Python code "
                "when the field is omitted"
            )
        if item.default is not MISSING and not _constant(item.default):
            return Reason(text=f"{klass.__qualname__} field '{item.name}' has a default that is not a constant")
    return None


def _generated(klass: type, name: str) -> bool:
    """Whether klass defines its own `name` as the code dataclasses generates for its fields.

    Compares bytecode with a reference dataclass built from the same fields and flags, so a hand-written function
    with the same signature does not pass. A class without its own dataclass markers is never generated.
    """
    if "__dataclass_params__" not in vars(klass) or name not in vars(klass):
        return False
    reference = vars(_reference(klass))
    value = vars(klass)[name]
    return name in reference and isinstance(value, FunctionType) and _bytecode(value) == _bytecode(reference[name])


def _reference(klass: type) -> type:
    params = vars(klass)["__dataclass_params__"]
    specs = []
    for item in _fields(klass):
        if item.default_factory is MISSING:
            specs.append((item.name, item.type, field(default=item.default, init=item.init, kw_only=item.kw_only)))
        else:
            factory = field(default_factory=item.default_factory, init=item.init, kw_only=item.kw_only)
            specs.append((item.name, item.type, factory))
    return make_dataclass("Reference", specs, frozen=params.frozen, bases=())


def _bytecode(function: FunctionType) -> tuple[Hashable, ...]:
    code = function.__code__
    return (code.co_code, code.co_names, code.co_consts, code.co_varnames, code.co_freevars, code.co_argcount)

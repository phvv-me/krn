"""Which exception classes a kernel may construct on a device, and the reason text when they may not."""

import builtins
import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from krn.exceptions.device_safe import device_safe
from krn.exceptions.schemas import SCHEMAS

GLOBAL_LIMIT = 4


def compute(value: int) -> int:
    return value + 1


def compile_class(name: str, source: str, **names: type) -> type[BaseException]:
    """The class `name` from source text, which has no file to read its constructor from."""
    namespace: dict[str, type] = dict(names)
    exec(source, namespace)
    found = namespace[name]
    assert isinstance(found, type) and issubclass(found, BaseException)
    return found


class Stub:
    def __get__(self, instance: RuntimeError | None, owner: type | None = None) -> int:
        return 1


def deco(function: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(function)
    def wrapper(*args: object, **kwargs: object) -> None:
        function(*args, **kwargs)

    return wrapper


def plain_deco(function: Callable[..., None]) -> Callable[..., None]:
    def wrapper(*args: int) -> None:
        function(*args)

    return wrapper


class Stores(RuntimeError):
    """Stores its scalars and a constant."""

    label = "budget"
    limit_scale = 2.5

    def __init__(self, limit: int, strict: bool = True) -> None:
        super().__init__()
        self.limit = limit
        self.strict = strict
        self.kind = "budget"


class Documented(RuntimeError):
    """A docstring and a pass are allowed."""

    def __init__(self, limit: int) -> None:
        "Store the limit."
        pass


class Stringy(RuntimeError):
    def __init__(self, what: str) -> None:
        super().__init__(what)


class Methodical(RuntimeError):
    def describe(self) -> str:
        return "methods are not constants"


class CInit(RuntimeError):
    __init__ = BaseException.__init__
    __str__ = BaseException.__str__


class Scaled(RuntimeError):
    def __init__(self, limit: int) -> None:
        super().__init__(compute(limit))


class PositionalOnly(RuntimeError):
    def __init__(self, limit: int, /) -> None:
        super().__init__(limit)


class Payload(RuntimeError):
    def __init__(self, payload: dict[str, int]) -> None:
        self.payload = payload


class Keyword(RuntimeError):
    def __init__(self, *, limit: int) -> None:
        self.limit = limit


class Variadic(RuntimeError):
    def __init__(self, *values: int) -> None:
        self.values = values


class Global(RuntimeError):
    def __init__(self, limit: int) -> None:
        self.limit = compute(limit)


class GlobalName(RuntimeError):
    def __init__(self, limit: int) -> None:
        self.limit = GLOBAL_LIMIT


class Named(RuntimeError):
    def __init__(self, limit: int) -> None:
        local = limit
        self.limit = local


class Branching(RuntimeError):
    def __init__(self, limit: int) -> None:
        if limit:
            self.limit = limit


class Bytes(RuntimeError):
    def __init__(self, limit: int) -> None:
        self.data = b"x"


class Called(RuntimeError):
    def __init__(self, limit: int) -> None:
        compute(limit)


class Unpacked(RuntimeError):
    def __init__(self, limit: int) -> None:
        self.limit, self.strict = limit, True


class Loose(RuntimeError):
    """A constructor that takes keywords as well, so that a subclass can pass them for the type checker to accept."""

    def __init__(self, *args: int, **kwargs: int) -> None:
        super().__init__(*args)


class Keyworded(Loose):
    def __init__(self, limit: int) -> None:
        super().__init__(limit=compute(limit))


class Explicit(RuntimeError):
    def __init__(self, limit: int) -> None:
        super(RuntimeError, self).__init__(limit)


class LooseSyntax(SyntaxError):
    def __init__(self, *args: int) -> None:
        super().__init__(*args)


class Pairy(LooseSyntax):
    def __init__(self, first: int, second: int) -> None:
        super().__init__(first, second)


class Base(Loose):
    def __init__(self, limit: int) -> None:
        super().__init__(limit=limit)


class Child(Base):
    def __init__(self, limit: int) -> None:
        super().__init__(limit)


class Stringifies(RuntimeError):
    def __str__(self) -> str:
        return "text"


class Holds(RuntimeError):
    items: ClassVar[list[int]] = [1, 2]


class Inherits(Holds):
    pass


class Odd(RuntimeError):
    def __new__(cls, limit: int) -> Odd:
        return super().__new__(cls)


class Watched(RuntimeError):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def __setattr__(self, name: str, value: int | str) -> None:
        super().__setattr__(name, value)


class Hooked(RuntimeError):
    def __init_subclass__(cls, **kwargs: int) -> None:
        super().__init_subclass__(**kwargs)


class DescribedInit(RuntimeError):
    __init__ = Stub()


class DescribedAttribute(RuntimeError):
    limit = Stub()


class Settable(int):
    def __set__(self, instance: RuntimeError, value: int) -> None:
        pass


class SettableAttribute(RuntimeError):
    limit = Settable()


DescribedDefault = compile_class(
    "DescribedDefault",
    "from dataclasses import dataclass\n@dataclass\nclass DescribedDefault(RuntimeError):\n    limit: int = Stub()\n",
    Stub=Stub,
)


class Prop(RuntimeError):
    @property
    def limit(self) -> int:
        return 1


class Slotted(RuntimeError):
    __slots__ = ("limit",)

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit


class Meta(type):
    def __call__(cls, *args: int, **kwargs: int) -> type:
        return super().__call__(*args, **kwargs)


class WithMeta(RuntimeError, metaclass=Meta):
    pass


class Wrapped(RuntimeError):
    @deco
    def __init__(self, limit: int) -> None:
        super().__init__()


class PlainWrapped(RuntimeError):
    @plain_deco
    def __init__(self, limit: int) -> None:
        super().__init__()


class Asynchronous(RuntimeError):
    async def __init__(self, limit: int) -> None:
        super().__init__()


NoArguments = compile_class("NoArguments", "class NoArguments(RuntimeError):\n    def __init__():\n        pass\n")
Unannotated = compile_class(
    "Unannotated", "class Unannotated(RuntimeError):\n    def __init__(self, limit):\n        pass\n"
)


@dataclass(frozen=True)
class Budget(RuntimeError):
    limit: int
    used: float
    strict: bool = True
    label: str = "budget"


@dataclass
class Listed(RuntimeError):
    items: list[int] = field(default_factory=list)


@dataclass
class Factory(RuntimeError):
    limit: int = field(default_factory=int)


@dataclass
class Checked(RuntimeError):
    limit: int

    def __post_init__(self) -> None:
        self.limit = self.limit


@dataclass
class Nullable(RuntimeError):
    limit: int
    note: str | None = None


@dataclass
class Constructed(RuntimeError):
    items: tuple[int, ...] = (1, 2)


@dataclass
class DataWatched(RuntimeError):
    limit: int

    def __setattr__(self, name: str, value: int) -> None:
        pass


@pytest.mark.parametrize(
    "cls",
    [ValueError, KeyError, BaseException, Budget, Stores, Stringy, Documented, CInit, Methodical, PositionalOnly],
)
def test_device_safe_classes(cls: type[BaseException]) -> None:
    assert device_safe(cls) is None


@pytest.mark.parametrize(
    "cls,fragment",
    [
        (Scaled, "super().__init__(compute(limit))"),
        (Payload, "parameter 'payload' is annotated"),
        (Keyword, "parameter 'limit' is not positional"),
        (Variadic, "parameter 'values' is not positional"),
        (NoArguments, "must take self as a positional parameter"),
        (Global, "self.limit = compute(limit)"),
        (GlobalName, "self.limit = GLOBAL_LIMIT"),
        (Named, "local = limit"),
        (Branching, "if limit"),
        (Bytes, "self.data = b'x'"),
        (Called, "compute(limit)"),
        (Unpacked, "self.limit, self.strict = (limit, True)"),
        (Keyworded, "super().__init__(limit=compute(limit))"),
        (Explicit, "super(RuntimeError, self).__init__(limit)"),
        (Pairy, "super().__init__(first, second)"),
        (Child, "Base.__init__ statement 'super().__init__(limit=limit)'"),
        (Stringifies, "Stringifies.__str__ is defined in Python"),
        (Holds, "Holds.items is a list"),
        (Inherits, "Holds.items is a list"),
        (Odd, "Odd.__new__ is defined in Python"),
        (Watched, "Watched.__setattr__ is defined in Python"),
        (Hooked, "Hooked.__init_subclass__ is defined in Python"),
        (DescribedInit, "DescribedInit.__init__ is a Stub"),
        (DescribedAttribute, "DescribedAttribute.limit is a Stub"),
        (Prop, "Prop.limit is a property"),
        (Slotted, "Slotted.limit is a member_descriptor"),
        (WithMeta, "metaclass Meta"),
        (Wrapped, "wraps another function"),
        (PlainWrapped, "must take self as a positional parameter"),
        (Asynchronous, "is not a plain def"),
        (Listed, "field 'items' has a default_factory"),
        (Factory, "field 'limit' has a default_factory"),
        (Checked, "Checked.__post_init__ is defined in Python"),
        (Nullable, "parameter 'note' is annotated"),
        (Constructed, "field 'items' has a default that is not a constant"),
        (DescribedDefault, "DescribedDefault.limit is a Stub"),
        (SettableAttribute, "SettableAttribute.limit is a Settable"),
        (DataWatched, "DataWatched.__setattr__ is defined in Python"),
    ],
)
def test_device_unsafe_classes_name_the_offender(cls: type[BaseException], fragment: str) -> None:
    reason = device_safe(cls)
    assert reason is not None and fragment in reason.text


def test_dataclass_generated_hooks_pass_provenance() -> None:
    assert device_safe(Budget) is None


def test_builtins_whose_constructor_takes_bytes_have_no_schema() -> None:
    for name in ("UnicodeDecodeError", "UnicodeEncodeError", "UnicodeTranslateError"):
        reason = device_safe(getattr(builtins, name))
        assert reason is not None and f"{name} has no device constructor schema" in reason.text


def test_exception_groups_have_no_schema() -> None:
    for cls in (ExceptionGroup, BaseExceptionGroup):
        reason = device_safe(cls)
        assert reason is not None and "no device constructor schema" in reason.text


def test_every_schema_class_is_device_safe() -> None:
    assert all(device_safe(cls) is None for cls in SCHEMAS)


def test_python_init_without_source_cannot_be_checked() -> None:
    opaque = compile_class("Opaque", "class Opaque(RuntimeError):\n    def __init__(self, limit: int):\n        pass\n")
    reason = device_safe(opaque)
    assert reason is not None and "has no source to check" in reason.text


def test_unannotated_parameter_is_refused_before_its_source_is_read() -> None:
    reason = device_safe(Unannotated)
    assert reason is not None and "parameter 'limit' is not annotated" in reason.text


def test_source_that_does_not_parse_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inspect, "getsource", lambda _: "def (")
    reason = device_safe(Stringy)
    assert reason is not None and "source that does not parse" in reason.text


def test_dataclass_look_alike_with_a_hand_written_init_is_not_generated() -> None:
    reason = device_safe(
        compile_class(
            "Faked",
            "from dataclasses import dataclass\n"
            "@dataclass(frozen=True)\n"
            "class Faked(RuntimeError):\n"
            "    limit: int\n"
            "    used: float\n"
            "    strict: bool = True\n"
            "    label: str = 'budget'\n"
            "    def __init__(self, limit: int, used: float, strict: bool = True, label: str = 'budget'):\n"
            "        self.limit = limit\n"
            "        print('ran on the host')\n",
        )
    )
    assert reason is not None and "has no source to check" in reason.text


def test_a_mutated_base_brings_its_constructor_into_the_check() -> None:
    class Safe(RuntimeError):
        pass

    class Derived(Safe):
        pass

    assert device_safe(Derived) is None

    class Unsafe(Loose):
        def __init__(self, limit: int) -> None:
            super().__init__(limit=limit)

    Derived.__bases__ = (Unsafe,)
    reason = device_safe(Derived)
    assert reason is not None and "super().__init__(limit=limit)" in reason.text


def test_a_replaced_constructor_is_checked_again() -> None:
    class Replaced(RuntimeError):
        def __init__(self, limit: int) -> None:
            super().__init__(limit)

    assert device_safe(Replaced) is None
    Replaced.__init__ = lambda self, limit: None
    reason = device_safe(Replaced)
    assert reason is not None and "parameter 'limit' is not annotated" in reason.text

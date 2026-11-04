"""The function bodies a set of modules owns, which automatic annotation watches."""

import importlib
import inspect
from collections.abc import Sequence
from types import CodeType, FunctionType, ModuleType


def module_codes(modules: Sequence[str]) -> set[CodeType]:
    """Find owned function bodies, including nested functions but not comprehensions.

    Traverse generated code too, since it may contain an explicitly written lambda.
    """
    loaded = [importlib.import_module(name) for name in modules]
    found: set[CodeType] = set()
    pending = {code for module in loaded for code in owned_codes(module)}
    while pending:
        code = pending.pop()
        found.add(code)
        pending.update(item for item in code.co_consts if isinstance(item, CodeType) and item not in found)
    return {code for code in found if code.co_name not in ("<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>")}


def owned_codes(module: ModuleType) -> tuple[CodeType, ...]:
    """Return unique unwrapped function bodies owned by a module or its classes."""
    owned = [
        value
        for value in module.__dict__.values()
        if isinstance(value, FunctionType | type) and value.__module__ == module.__name__
    ]
    functions = [value for value in owned if isinstance(value, FunctionType)]
    methods = [
        member.__func__ if isinstance(member, staticmethod | classmethod) else member
        for cls in owned
        if isinstance(cls, type)
        for member in cls.__dict__.values()
        if isinstance(member, FunctionType | staticmethod | classmethod)
    ]
    return tuple(
        dict.fromkeys(
            function.__code__
            for candidate in (*functions, *methods)
            if isinstance(function := inspect.unwrap(candidate), FunctionType)
            and function.__module__ == module.__name__
        )
    )

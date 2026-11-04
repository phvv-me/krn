"""Token and literal assertions for generated source and compiler messages."""

from collections.abc import Generator
from contextlib import contextmanager
from itertools import pairwise

import pytest


def words(source: str) -> list[str]:
    """Identifier and number tokens, in source order."""
    result: list[str] = []
    current = ""
    for char in source:
        if char.isalnum() or char == "_":
            current += char
        elif current:
            result.append(current)
            current = ""
    if current:
        result.append(current)
    return result


def descriptor(source: str, name: str) -> bool:
    """Whether assembly declares the named AMD kernel descriptor."""
    tokens = words(source)
    return any(left == "amdhsa_kernel" and right == name for left, right in pairwise(tokens))


@contextmanager
def raises_literal[E: BaseException](kind: type[E], message: str) -> Generator[pytest.ExceptionInfo[E]]:
    """Require the exception's message to contain the literal expected text."""
    with pytest.raises(kind) as captured:
        yield captured
    assert message in str(captured.value)


def forbidden_msl(source: str) -> bool:
    """Whether generated Metal source contains unsupported C spellings."""
    tokens = words(source)
    return (
        "long long" in source
        or "__builtin" in source
        or bool(set(tokens) & {"double", "printf", "unsigned", "expf", "sqrtf"})
        or any(
            token.endswith(suffix) and len(token) > len(suffix) and token[-len(suffix) - 1].isdigit()
            for token in tokens
            for suffix in ("ull", "ll")
        )
    )

"""Identifier tokens used by offline research tooling."""


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

"""Word-preserving transformations for generated C-family source."""


def word_character(char: str) -> bool:
    """Whether a character belongs to an identifier or numeric token."""
    return char.isalnum() or char == "_"


def replace_word(source: str, old: str, new: str) -> str:
    """Replace complete spellings without changing containing identifiers."""
    parts: list[str] = []
    start = 0
    while (index := source.find(old, start)) >= 0:
        end = index + len(old)
        boundary = (index == 0 or not word_character(source[index - 1])) and (
            end == len(source) or not word_character(source[end])
        )
        parts.extend((source[start:index], new if boundary else old))
        start = end
    parts.append(source[start:])
    return "".join(parts)


def integer_suffixes(source: str) -> str:
    """Use Metal's long suffixes while preserving nonnumeric identifiers."""
    for old, new in (("ull", "ul"), ("ll", "l")):
        parts: list[str] = []
        start = 0
        while (index := source.find(old, start)) >= 0:
            end = index + len(old)
            numeric = (
                index > 0 and source[index - 1].isdigit() and (end == len(source) or not word_character(source[end]))
            )
            parts.extend((source[start:index], new if numeric else old))
            start = end
        parts.append(source[start:])
        source = "".join(parts)
    return source

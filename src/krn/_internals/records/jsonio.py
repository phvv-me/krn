"""Reading the JSON that sites and messages dump: each value is checked, so a bad table names its key."""

type JSON = str | int | bool | list[JSON] | dict[str, JSON] | None


def json_object(value: JSON) -> dict[str, JSON]:
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object, not {value!r}")
    return value


def json_list(item: dict[str, JSON], key: str) -> list[JSON]:
    value = item[key]
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be a JSON array, not {value!r}")
    return value


def json_text(item: dict[str, JSON], key: str) -> str:
    value = item[key]
    if not isinstance(value, str):
        raise ValueError(f"{key!r} must be a string, not {value!r}")
    return value


def json_int(item: dict[str, JSON], key: str) -> int:
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer, not {value!r}")
    return value


def json_optional_int(item: dict[str, JSON], key: str) -> int | None:
    return None if item[key] is None else json_int(item, key)


def json_bool(item: dict[str, JSON], key: str) -> bool:
    value = item[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key!r} must be true or false, not {value!r}")
    return value

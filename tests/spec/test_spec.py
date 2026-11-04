"""The spec of the working tree: it validates against its schema, builds to the same bytes, and describes every public
function of `krn.math` with a signature and a summary."""

import json
from pathlib import Path

import jsonschema
import pytest

import krn.math
from krn._internals.compat import Matrix
from scripts import build_spec

EMPTY = Matrix(features=(), targets=(), cells={})


@pytest.fixture(scope="module")
def spec_text() -> str:
    return build_spec.dumps_spec(build_spec.build(EMPTY))


@pytest.fixture(scope="module")
def document(spec_text: str):
    """The spec as the site reads it: the file's JSON, decoded."""
    return json.loads(spec_text)


def _members(document, module: str):
    return next(entry["members"] for entry in document["api"] if entry["module"] == module)


def test_the_spec_validates_against_its_schema(document) -> None:
    schema = json.loads(build_spec.SCHEMA_FILE.read_text())
    jsonschema.Draft202012Validator(schema).validate(document)


def test_building_twice_gives_identical_bytes(spec_text: str) -> None:
    assert build_spec.dumps_spec(build_spec.build(EMPTY)) == spec_text


def test_every_public_function_of_krn_math_has_a_signature_and_a_summary(document) -> None:
    public = {name for name in krn.math.__all__ if callable(getattr(krn.math, name))}
    functions = {m["name"]: m for m in _members(document, "krn.math") if m["kind"] == "function"}
    assert set(functions) == public
    for member in functions.values():
        assert member["signature"] is not None
        assert member["doc"] is not None and member["doc"]["summary"], member["name"]


def test_the_packed_lane_functions_carry_their_versionadded_callout(document) -> None:
    members = {m["name"]: m for m in _members(document, "krn.math")}
    for name in ("add_sat", "sub_sat", "abs_diff", "avg"):
        assert members[name]["callouts"] == [{"kind": "versionadded", "version": "0.0.1", "text": ""}], name


def test_every_example_is_in_the_spec_with_its_header(document) -> None:
    files = sorted(build_spec.EXAMPLES.glob("[0-9][0-9]_*/[0-9][0-9]_*.py"))
    assert [e["path"] for e in document["examples"]] == [f.relative_to(build_spec.ROOT).as_posix() for f in files]
    for example in document["examples"]:
        assert example["id"] == f"{example['group']}/{example['path'].rsplit('/', 1)[-1].removesuffix('.py')}"
        assert example["level"] == int(example["group"][:2])
        assert example["title"] and example["topics"], example["id"]


def test_the_needs_of_an_example_are_examples_that_come_before_it(document) -> None:
    order = {example["id"]: index for index, example in enumerate(document["examples"])}
    for example in document["examples"]:
        assert all(order[need] < order[example["id"]] for need in example["needs"]), example["id"]


def test_the_levels_are_the_five_folders_in_order(document) -> None:
    assert sorted({example["group"] for example in document["examples"]}) == [
        "01_first_kernels",
        "02_tiles_and_types",
        "03_memory_and_sync",
        "04_libraries",
        "05_systems",
    ]


HEADER = (
    "Title of the example\n\nWhat it teaches.\nOn two lines.\n\n"
    "Level: 2\nTopics: tiles, strict arithmetic\nNeeds: {needs}"
)


def test_a_header_gives_title_text_level_topics_and_needs() -> None:
    header = build_spec.read_example_header(
        "x", HEADER.format(needs="01_first_kernels/01_saxpy, 01_first_kernels/02_gelu")
    )
    assert header.title == "Title of the example"
    assert header.description == "What it teaches. On two lines."
    assert (header.level, header.topics) == (2, ("tiles", "strict arithmetic"))
    assert header.needs == ("01_first_kernels/01_saxpy", "01_first_kernels/02_gelu")
    assert build_spec.read_example_header("x", HEADER.format(needs="none")).needs == ()


@pytest.mark.parametrize(
    "docstring",
    [
        "",
        "Title only",
        "Title\n\nText with no block.",
        "Title\n\nText.\n\nLevel: 2\nTopics: a",
        "Title\n\nText.\n\nLevel: 6\nTopics: a\nNeeds: none",
        "Title\n\nText.\n\nLevel: two\nTopics: a\nNeeds: none",
    ],
)
def test_a_header_that_breaks_the_form_stops_the_build(docstring: str) -> None:
    with pytest.raises(build_spec.SpecError):
        build_spec.read_example_header("x", docstring)


def _example(folder: Path, group: str, name: str, level: int, needs: str = "none") -> None:
    (folder / group).mkdir(exist_ok=True)
    (folder / group / name).write_text(f'''"""Title\n\nText.\n\nLevel: {level}\nTopics: a\nNeeds: {needs}\n"""\n''')


def test_a_level_that_disagrees_with_its_folder_stops_the_build(tmp_path: Path) -> None:
    _example(tmp_path, "02_tiles_and_types", "01_a.py", 3)
    with pytest.raises(build_spec.SpecError, match="its folder 02_tiles_and_types is level 2"):
        build_spec.examples(tmp_path)


def test_a_need_that_is_not_an_example_stops_the_build(tmp_path: Path) -> None:
    _example(tmp_path, "02_tiles_and_types", "01_a.py", 2, needs="01_first_kernels/01_none")
    with pytest.raises(build_spec.SpecError, match="not examples"):
        build_spec.examples(tmp_path)


def test_the_homes_of_the_errors_and_the_emulator_are_documented(document) -> None:
    assert {m["name"] for m in _members(document, "krn.errors")} >= {"CaptureError", "TargetError", "PrecisionWarning"}
    assert {m["name"] for m in _members(document, "krn.emulate")} == {"Deadlock", "emulate"}

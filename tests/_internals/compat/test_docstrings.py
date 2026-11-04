"""The generated "Supported on" blocks: the docstrings agree with the committed matrix, a rewrite replaces the block a
previous run wrote, and the block words a target's status from the data."""

from pathlib import Path

import pytest

from experiments import docstring_support as docs
from experiments.compat_matrix import OUTPUT
from krn._internals.compat import Status, Support, loads

NATIVE = Support(status=Status.NATIVE)
REFUSED = Support(status=Status.REFUSED, note="`x` needs a unit avx2 lacks; later", instead="use f16")


committed = pytest.mark.skipif(not OUTPUT.exists(), reason="the site data is not checked out here")


@committed
def test_the_docstrings_agree_with_the_committed_matrix() -> None:
    assert docs.stale_modules(loads(OUTPUT.read_text()), write=False) == []


@committed
def test_a_rewrite_replaces_the_block_and_is_idempotent(tmp_path: Path) -> None:
    matrix = loads(OUTPUT.read_text())
    original = docs.SOURCES[0].path.read_text()
    stale = tmp_path / "math.py"
    stale.write_text(original.replace(f"{docs.MARKER}\n", f"{docs.MARKER}\n        a stale line\n", 1))
    assert docs.rewrite(docs.Source(module="krn.math", path=stale), matrix) == original


def test_splice_replaces_an_earlier_block_and_keeps_the_text() -> None:
    lines = ['    """Sine.', "", f"    {docs.MARKER}", "        NVIDIA: old", "", '    """']
    block = [docs.MARKER, "    Native on every target", ""]
    once = docs.splice(lines, block)
    assert once == ['    """Sine.', "", f"    {docs.MARKER}", "        Native on every target", "", '    """']
    assert docs.splice(once, block) == once


def test_splice_grows_a_one_line_docstring_to_its_closing_quotes() -> None:
    block = [docs.MARKER, "    Native on every target", ""]
    assert docs.splice(['    """Sine."""'], block) == [
        '    """Sine.',
        "",
        f"    {docs.MARKER}",
        "        Native on every target",
        "",
        '    """',
    ]


def test_a_refusal_reads_without_its_target_name_so_the_targets_of_a_vendor_share_a_run() -> None:
    on_avx2 = Support(status=Status.REFUSED, note="`x` needs a unit avx2 lacks", instead="use f16")
    on_avx512 = Support(status=Status.REFUSED, note="`x` needs a unit avx512 lacks", instead="use f16")
    assert docs.text_of(on_avx2, 0, "avx2") == "refused (`x` needs a unit this target lacks)"
    assert docs.text_of(on_avx2, 0, "avx2") == docs.text_of(on_avx512, 0, "avx512")


def test_a_label_names_few_rows_that_differ_and_counts_many() -> None:
    assert docs.label("sm_80", [("f16", NATIVE), ("f64", REFUSED)]) == "native (f64 refused)"
    many = [(f"t{i}", NATIVE) for i in range(6)] + [(f"u{i}", REFUSED) for i in range(4)]
    assert docs.label("sm_80", many) == "native, 4 of 10 operand types refused"
    assert docs.label("avx2", [("f32", Support(status=Status.COMPOSED, length=1))]) == "composed (1 instruction)"
    assert docs.label("avx2", [("f32", Support(status=Status.COMPOSED, length=3))]) == "composed (3 instructions)"


def test_a_run_of_targets_is_written_as_its_first_and_last_names() -> None:
    assert docs.span_text(["sm_80"], "sm_121a") == "sm_80"
    assert docs.span_text(["sm_80", "sm_86"], "sm_121a") == "sm_80..sm_86"
    assert docs.span_text(["sm_80", "sm_86"], "sm_86") == "sm_80+"

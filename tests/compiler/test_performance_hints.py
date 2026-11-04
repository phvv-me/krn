"""The fused multiply-add hint: a multiply whose only use is an add or subtract, read from the captured IR."""

import ast
import re
import textwrap
import warnings
from pathlib import Path

import pytest

from krn.compiler import performance
from krn.compiler.api import CaptureError, capture
from krn.compiler.dtypes import F32, Dtype
from krn.compiler.errors import Span
from krn.compiler.hints import _Pattern, _Spelling
from krn.compiler.ir import Binary, Kernel, Op, Sync, Zeros
from krn.compiler.performance import PerformanceWarning
from krn.compiler.source import KERNEL_DECORATORS
from krn.facts.targets import Target
from krn.runtime.settings import configure

ROOT = Path(__file__).resolve().parents[2]
SCAN = (ROOT / "src/krn/lib", ROOT / "src/krn/data", ROOT / "src/krn/collections", ROOT / "examples")
MESSAGE = "two roundings; math.fma(a, b, c) is a single fused instruction and more accurate"
HEADER = """\
from krn import T, Vector, grid, items, kernel, math


@kernel
def k(a: Vector[T.{ty}], b: Vector[T.{ty}], c: Vector[T.{ty}], out: Vector[T.{ty}]{extra}) -> None:
"""
THREAD = "i = grid.thread_rank()\nif i < out.shape[0]:\n"


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(performance, "_seen", set())


def thread(statements: str) -> str:
    return THREAD + textwrap.indent(textwrap.dedent(statements), "    ")


def source(directory: Path, body: str, *, ty: str = "f32", extra: str = "") -> Path:
    file = directory / "kernel.py"
    file.write_text(HEADER.format(ty=ty, extra=extra) + textwrap.indent(textwrap.dedent(body), "    "))
    return file


def hints(file: Path, target: str) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        capture(file, "k", target=Target.of(target))
    messages = [str(w.message) for w in caught if issubclass(w.category, PerformanceWarning)]
    return [message for message in messages if "fusable" in message]


SHAPES = [
    pytest.param("out[i] = a[i] * b[i] + c[i]", "out[i] = math.fma(a[i], b[i], c[i])", id="a*b+c"),
    pytest.param("out[i] = c[i] + a[i] * b[i]", "out[i] = math.fma(a[i], b[i], c[i])", id="c+a*b"),
    pytest.param("out[i] = c[i] - a[i] * b[i]", "out[i] = math.fma(-a[i], b[i], c[i])", id="c-a*b"),
    pytest.param("out[i] = a[i] * b[i] - c[i]", "out[i] = math.fma(a[i], b[i], -c[i])", id="a*b-c"),
    pytest.param("t = a[i] * b[i]\nout[i] = t + c[i]", "out[i] = math.fma(a[i], b[i], c[i])", id="temp+c"),
    pytest.param("t = a[i] * b[i]\nout[i] = c[i] - t", "out[i] = math.fma(-a[i], b[i], c[i])", id="c-temp"),
]

ACCUMULATE = [
    pytest.param("acc += a[i] * b[i]", "acc = math.fma(a[i], b[i], acc)", id="acc+=a*b"),
    pytest.param("acc -= a[i] * b[i]", "acc = math.fma(-a[i], b[i], acc)", id="acc-=a*b"),
]

OPERANDS = [
    pytest.param("out[i] = a[i] * 2.0 + c[i]", "out[i] = math.fma(a[i], 2.0, c[i])", id="const"),
    pytest.param(
        "out[i] = a[i] * b[i] + T.f32(out.shape[0])",
        "out[i] = math.fma(a[i], b[i], T.f32(out.shape[0]))",
        id="dim",
    ),
    pytest.param(
        "out[i] = a[i] * b[i] + T.f32(grid.thread_rank())",
        "out[i] = math.fma(a[i], b[i], T.f32(grid.thread_rank()))",
        id="intrinsic",
    ),
    pytest.param("out[i] = -a[i] * b[i] + c[i]", "out[i] = math.fma(-a[i], b[i], c[i])", id="unary"),
    pytest.param("out[i] = math.sqrt(a[i]) * b[i] + c[i]", "out[i] = math.fma(math.sqrt(a[i]), b[i], c[i])", id="math"),
    pytest.param(
        "out[i] = a[i] * (b[i] if i > 0 else c[i]) + c[i]",
        "out[i] = math.fma(a[i], (b[i] if i > 0 else c[i]), c[i])",
        id="select",
    ),
    pytest.param("out[i] = a[i] - (b[i] - c[i]) * c[i]", "out[i] = math.fma(-(b[i] - c[i]), c[i], a[i])", id="negated"),
    pytest.param("out[i] = (a[i] + b[i]) * c[i] + c[i]", "out[i] = math.fma(a[i] + b[i], c[i], c[i])", id="sum-factor"),
    pytest.param("out[i] = a[i] / b[i] * c[i] + c[i]", "out[i] = math.fma(a[i] / b[i], c[i], c[i])", id="quotient"),
    pytest.param(
        "out[i] = a[i] * b[i] + T.f32(i // 2)", "out[i] = math.fma(a[i], b[i], T.f32(i // 2))", id="integer-quotient"
    ),
]

SILENT = [
    pytest.param("out[i] = math.fma(a[i], b[i], c[i])", "f32", id="math.fma"),
    pytest.param("out[i] = a[i] * b[i] + c[i]", "i32", id="integers"),
    pytest.param("t = a[i] * b[i]\nout[i] = t + c[i]\nout[i] = out[i] + t", "f32", id="used-twice"),
    pytest.param("out[i] = a[i] * b[i]", "f32", id="no-add"),
    pytest.param("t = a[i] * b[i]\nout[i] = t * c[i]", "f32", id="feeds-multiply"),
]


@pytest.mark.parametrize(("body", "spelled"), SHAPES + ACCUMULATE)
def test_each_shape_warns_with_the_fused_spelling(tmp_path: Path, body: str, spelled: str) -> None:
    lines = body.splitlines()
    if lines[0].startswith("acc"):
        loop = "for i in items(out.shape[0]):\n    acc = T.f32(0)\n" + textwrap.indent(body, "    ")
        body = loop + "\n    out[i] = acc"
    else:
        body = thread(body)
    [message] = hints(source(tmp_path, body), "apple_m4")
    assert message.startswith("k on apple_m4: has 1 fusable multiply-add pattern; the first, at line ")
    assert MESSAGE in message
    assert f"`{spelled}`" in message


@pytest.mark.parametrize(("body", "spelled"), OPERANDS)
def test_the_replacement_spells_each_operand_kind(tmp_path: Path, body: str, spelled: str) -> None:
    [message] = hints(source(tmp_path, thread(body)), "apple_m4")
    assert f"`{spelled}`" in message


def test_a_scalar_parameter_operand_is_spelled_by_name(tmp_path: Path) -> None:
    body = thread("out[i] = alpha * a[i] + c[i]")
    [message] = hints(source(tmp_path, body, extra=", alpha: T.f32"), "apple_m4")
    assert "`out[i] = math.fma(alpha, a[i], c[i])`" in message


def test_a_condition_is_spelled_as_the_call_alone(tmp_path: Path) -> None:
    body = thread("if a[i] * b[i] + c[i] > 0.0:\n    out[i] = 1.0")
    [message] = hints(source(tmp_path, body), "apple_m4")
    assert "is `if a[i] * b[i] + c[i] > 0.0:`" in message
    assert "spell that line `math.fma(a[i], b[i], c[i])`" in message


def test_the_message_counts_the_patterns_and_quotes_the_first_line(tmp_path: Path) -> None:
    body = thread("out[i] = a[i] * b[i] + c[i]\nout[i] = out[i] - a[i] * c[i]")
    file = source(tmp_path, body)
    [message] = hints(file, "apple_m4")
    line = file.read_text().splitlines().index("        out[i] = a[i] * b[i] + c[i]") + 1
    assert (
        f"has 2 fusable multiply-add patterns; the first, at line {line}, is `out[i] = a[i] * b[i] + c[i]`" in message
    )
    assert "spell that line `out[i] = math.fma(a[i], b[i], c[i])`" in message


@pytest.mark.parametrize(("body", "ty"), SILENT)
def test_shapes_without_a_fusable_multiply_stay_silent(tmp_path: Path, body: str, ty: str) -> None:
    assert not hints(source(tmp_path, thread(body), ty=ty), "apple_m4")


@pytest.mark.parametrize(
    ("ty", "target", "warns"),
    [
        pytest.param("f16", "apple_m4", True, id="f16-metal"),
        pytest.param("f16", "avx512", False, id="f16-cpu-without-native-half-fma"),
        pytest.param("f16", "sm_90a", True, id="f16-nvidia"),
        pytest.param("f64", "sm_90a", True, id="f64-nvidia"),
        pytest.param("f32", "avx512", True, id="f32-cpu"),
    ],
)
def test_only_targets_with_a_fused_instruction_are_hinted(tmp_path: Path, ty: str, target: str, warns: bool) -> None:
    file = source(tmp_path, thread("out[i] = a[i] * b[i] + c[i]"), ty=ty)
    assert bool(hints(file, target)) is warns


def test_warns_once_per_kernel_kind_and_target(tmp_path: Path) -> None:
    file = source(tmp_path, thread("out[i] = a[i] * b[i] + c[i]\nout[i] = out[i] - a[i] * c[i]"))
    assert len(hints(file, "apple_m4")) == 1
    assert not hints(file, "apple_m4")
    assert len(hints(file, "sm_90a")) == 1


def test_the_hint_can_be_silenced(tmp_path: Path) -> None:
    file = source(tmp_path, thread("out[i] = a[i] * b[i] + c[i]"))
    with configure(**{"warn.hint_fusable_multiply_add": False}):
        assert not hints(file, "apple_m4")


def test_the_minimum_count_raises_the_bar(tmp_path: Path) -> None:
    file = source(tmp_path, thread("out[i] = a[i] * b[i] + c[i]\nout[i] = out[i] - a[i] * c[i]"))
    with configure(**{"warn.min_fusable_multiply_adds": 3}):
        assert not hints(file, "apple_m4")
    with configure(**{"warn.min_fusable_multiply_adds": 2}):
        assert len(hints(file, "apple_m4")) == 1


def test_the_hint_can_be_escalated_to_an_error(tmp_path: Path) -> None:
    file = source(tmp_path, thread("out[i] = a[i] * b[i] + c[i]"))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=PerformanceWarning, message=".*fusable.*")
        with pytest.raises(PerformanceWarning, match="two roundings"):
            capture(file, "k", target=Target.of("apple_m4"))


def test_an_operand_without_a_source_form_prints_an_ellipsis() -> None:
    span = Span("kernel.py", 1, 1)
    zero = Zeros(F32, span=span)
    add = Binary(F32, Op.ADD, zero, zero, span=span)
    spelling = _Spelling(pattern=_Pattern(statement=Sync(span=span), consumer=add, product=add, product_left=True))
    assert spelling.expr(zero) == "..."


def dtypes_for(error: CaptureError, bound: dict[str, Dtype]) -> dict[str, Dtype]:
    """Bind the tensor paths a capture error asks for to f32, or none when it names none."""
    keys = [key for key in re.findall(r"'([^']+)': \.\.\.", error.message) if key not in bound]
    return {key: F32 for key in keys}


def captured(path: Path, name: str) -> tuple[Kernel, list[warnings.WarningMessage]] | None:
    """The kernel captured for apple_m4, binding the element types a generic kernel asks for to f32."""
    bound: dict[str, Dtype] = {}
    while True:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                kernel = capture(path, name, target=Target.of("apple_m4"), dtypes=bound)
            except CaptureError as error:
                if not (more := dtypes_for(error, bound)):
                    return None
                bound |= more
                continue
            except KeyError:
                return None  # capture has no visitor for some operator (`in`); the kernel is skipped, not hidden
        return kernel, caught


def kernel_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d).split("(")[0].split(".")[-1] in KERNEL_DECORATORS for d in node.decorator_list)
    ]


def test_library_kernels_that_still_warn(tmp_path: Path) -> None:
    """Scan library/example hints without modifying repository documentation."""
    scanned = captures = 0
    warned: list[str] = []
    for root in SCAN:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            for name in kernel_functions(path):
                scanned += 1
                performance._seen.clear()
                if (result := captured(path, name)) is None:
                    continue
                captures += 1
                hits = [str(w.message) for w in result[1] if "fusable" in str(w.message)]
                if hits:
                    warned.append(f"- `{path.relative_to(ROOT)}` `{name}`: {hits[0].split(': ', 1)[1]}")
    assert scanned
    lines = [
        "# FMA hint status",
        "",
        f"Scanned {scanned} `@kernel` functions in src/krn/lib, src/krn/data, src/krn/collections and examples on "
        f"apple_m4. {captures} captured standalone, with generic element types bound to f32 where the error names "
        "them; the rest are skipped (they need arguments the scan cannot supply, or use `in`, which capture does not "
        f"support yet). {len(warned)} captured kernels still warn with kind `fusable-multiply-add`:",
        "",
        *warned,
    ]
    (tmp_path / "fma-hints.md").write_text("\n".join(lines) + "\n")

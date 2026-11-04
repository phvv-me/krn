import inspect

from krn.diagnostics.models import Diagnostic, Rule


def test_contract() -> None:
    diagnostic = Diagnostic(
        id="registers",
        severity="warning",
        message="High pressure",
        suggested_fix="Reduce live values",
        targets=("nvidia",),
    )
    assert diagnostic.severity == "warning"
    assert inspect.isabstract(Rule)

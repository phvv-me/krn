"""The machine inventory as text: facts, hardware insights and the opt-in install line."""

import json
from dataclasses import asdict

from .install import install_plan
from .system import System


def report(system: System) -> str:
    """Describe `system`, list its insights with their fixes and give the `pip install` line its hardware calls for.

    system: the machine to describe, usually `System.detect()`.
    """
    insights = (
        f"{insight.severity} [{insight.code}] {insight.subject}: {insight.message}. {insight.fix}"
        for insight in system.insights()
    )
    plan = json.dumps(asdict(install_plan(system)), indent=2, default=sorted)
    return "\n".join((system.describe(), *insights, plan))

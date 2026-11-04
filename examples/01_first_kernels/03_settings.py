"""Settings and targets: ask what the machine is, then change one setting for one block

`host.System.detect()` finds the machine and its compiler target, and `Settings` is an immutable snapshot of every
group (run, precision, compile, debug). `with configure(...) as settings:` applies a change for the block and
restores it after, so a program can ask what a precision ladder resolves to on an `sm_100a` or a `gfx942` without
owning one. The ladder lists formats narrowest first, and the policy decides how hard it pushes.

Level: 1
Topics: settings, targets, precision ladder
Needs: none
"""

import logging

from krn import Precision, Run, Settings, configure, formats, host, targets

LADDER = host.Ladder("weights", (formats.Plain16(), formats.PlainBF16(), formats.Plain32()))


def show(settings: Settings) -> None:
    target = targets.Target.of(settings.run.target) if settings.run.target else host.System.detect().target
    logging.info("target %s, ladder %s", target.arch, LADDER.resolve(target, settings))
    logging.info(
        "policy %s, workers %s, graphs %s", settings.precision.policy, settings.run.workers, settings.run.graphs
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    show(Settings())
    with configure(run=Run(target="sm_100a"), precision=Precision(policy=host.Policy.MEMORY_BOUND)) as settings:
        show(settings)
    with configure(run=Run(target="gfx942", graphs=False), precision=Precision(pins={"weights": "fp16"})) as settings:
        show(settings)

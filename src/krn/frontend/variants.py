"""Kernel families: one contract, several implementations, picked by what the target can do.

A family starts from a portable kernel that every target can run. Specialized variants are added next to it, each with
a rule that says where it applies, and the rules compose the three dispatch styles of `krn`:

    @gemm.variant(targets.Nvidia)                      by type of target, like type_dispatch
    @gemm.variant(arch={"sm_90a"})                     by value, like value_dispatch
    @gemm.variant(requires={Feature.TENSOR_MEMORY})    by capability, which no type or value expresses

Selection picks the most specific rule that matches: a pinned architecture beats a feature set, more required features
beat fewer, and a vendor type beats the portable kernel. Two rules of equal specificity on one target are an error,
and `audit` runs the selection over a list of targets so a missing or ambiguous case fails in CI, not on the cluster.
Every variant has the signature of the portable kernel, which the type checker verifies, so callers never learn which
one runs. `krn.configure(**{"run.variants": {"gemm": "portable"}})` pins a family
when a specialized variant misbehaves on one system.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Concatenate, cast, overload

from krn.facts.targets import Feature, Target
from krn.runtime.api import Device, HostArgument, Launcher
from krn.runtime.settings import Settings, current_settings, retained

from .geometry import Geometry
from .kernel import Kernel


class AmbiguousVariant(LookupError):
    """Two variants of one family match a target with the same specificity."""


@dataclass(frozen=True, slots=True)
class Rule:
    """Where a variant applies. Empty fields do not restrict."""

    vendors: tuple[type[Target], ...] = ()
    archs: frozenset[str] = frozenset()
    requires: frozenset[Feature] = frozenset()

    def matches(self, target: Target) -> bool:
        return (
            (not self.vendors or isinstance(target, self.vendors))
            and (not self.archs or target.arch in self.archs)
            and self.requires <= target.features
        )

    @property
    def specificity(self) -> tuple[int, int, int]:
        return (int(bool(self.archs)), len(self.requires), int(bool(self.vendors)))


@dataclass(frozen=True, slots=True)
class Selection[S, **P]:
    """The variant chosen for a target and the reason, for logs and for the kernel report."""

    kernel: Kernel[Concatenate[S, P]]
    name: str
    reason: str


@dataclass(slots=True)
class Family[S, **P]:
    """Configured kernels with one configuration type and one signature, one per class of hardware."""

    name: str
    portable: Kernel[Concatenate[S, P]]
    variants: dict[str, tuple[Rule, Kernel[Concatenate[S, P]]]] = field(default_factory=dict)

    def variant(
        self, *vendors: type[Target], arch: Iterable[str] = (), requires: Iterable[Feature] = ()
    ) -> Callable[[Callable[Concatenate[S, P], None]], Kernel[Concatenate[S, P]]]:
        """Register a specialized implementation. Its signature must be the portable kernel's."""
        rule = Rule(vendors, frozenset(arch), frozenset(requires))

        def register(fn: Callable[Concatenate[S, P], None]) -> Kernel[Concatenate[S, P]]:
            kernel = Kernel(fn)
            self.variants[kernel.name] = (rule, kernel)
            return kernel

        return register

    def select(self, target: Target, settings: Settings | None = None) -> Selection[S, P]:
        settings = settings or current_settings()
        if pinned := settings.run.variants.get(self.name):
            return self._pinned(pinned)
        matching = [(rule, name, kernel) for name, (rule, kernel) in self.variants.items() if rule.matches(target)]
        if not matching:
            return Selection(self.portable, "portable", f"no variant applies to {target.arch}")
        best = max(rule.specificity for rule, _, _ in matching)
        winners = [(name, kernel) for rule, name, kernel in matching if rule.specificity == best]
        if len(winners) > 1:
            raise AmbiguousVariant(f"{self.name} on {target.arch}: {', '.join(name for name, _ in winners)} tie")
        name, kernel = winners[0]
        return Selection(kernel, name, f"most specific rule for {target.arch}")

    def _pinned(self, name: str) -> Selection[S, P]:
        if name == "portable":
            return Selection(self.portable, name, "pinned by run.variants")
        return Selection(self.variants[name][1], name, "pinned by run.variants")

    def audit(self, targets: Iterable[Target], settings: Settings | None = None) -> dict[str, str]:
        """The variant chosen on every target. Raises on an ambiguous case, which is the point of running it in CI."""
        return {target.arch: self.select(target, settings).name for target in targets}

    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...
    @overload
    def __call__(self, *args: HostArgument, launch: Geometry, **kwargs: HostArgument) -> None: ...
    @retained
    def __call__(self, *args: HostArgument, launch: Geometry | None = None, **kwargs: HostArgument) -> None:
        device = Device.current()
        # Family's P omits the inspected configuration parameter of the selected kernel.
        selected = cast(Kernel[Concatenate[S, P], None, S, P], self.select(device.target).kernel)
        selected(*args, launch=launch, **kwargs)

    @retained
    def __getitem__(self, spec: S) -> Launcher[P]:
        """Launch the variant selected for the device of the calling thread or task, configured as for a template."""
        device = Device.current()
        # Selection preserves the portable signature, including its configuration parameter.
        selected = cast(Kernel[Concatenate[S, P], None, S, P], self.select(device.target).kernel)
        return selected[spec]


def family[S, **P](fn: Callable[Concatenate[S, P], None]) -> Family[S, P]:
    """Declare a family from its portable template kernel."""
    portable = Kernel(fn)
    return Family(portable.name, portable)

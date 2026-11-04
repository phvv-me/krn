# A swept study: one collection policy, many input points, one row each, point beside profile.

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ...hw.devices.base import Device
from ..profiler import Collection, Profiler
from ..result import Profile


@dataclass(frozen=True, kw_only=True)
class Point(ABC):
    """One input configuration a study visits.

    A model, so its conditions travel with the measurement; what it configures is the domain's
    business.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """A short name identifying this point among the others."""


@dataclass(frozen=True, kw_only=True)
class Row[P: Point]:
    """One point's conditions beside what was observed there.

    A throughput number without its input specification is hard to reproduce and easy to
    misattribute. Generic over the caller's point type, so a facet reading a domain field off
    `row.point` sees that field rather than the bare `Point`.
    """

    label: str
    point: P
    profile: Profile
    seconds: float = 0.0

    @property
    def has_evidence(self) -> bool:
        """Whether this point captured evidence, independently of any scientific verdict."""
        return bool(
            self.profile.summaries
            or self.profile.windows
            or self.profile.launches
            or self.profile.kernels
            or self.profile.memcpys
            or self.profile.activities
        )


@dataclass(frozen=True, kw_only=True)
class Study[P: Point]:
    """A collection policy and the points to apply it at.

    collection: held once, so two points cannot silently differ in how they were measured.
    gpus: the devices every point's `Profiler` may use (mainboard.probe is not a dependency of
        profiling, so the caller resolves them).
    """

    collection: Collection = field(default_factory=Collection)
    points: tuple[P, ...] = ()
    gpus: Sequence[Device] = ()

    @classmethod
    def over(
        cls,
        points: Sequence[P],
        *,
        collection: Collection | None = None,
        gpus: Sequence[Device] = (),
    ) -> Study[P]:
        return cls(collection=collection or Collection(), points=tuple(points), gpus=gpus)

    def run(self, work: Callable[[P], None], *, warm: bool = True) -> tuple[Row[P], ...]:
        """Measure `work` at every point, returning one row each.

        Work and warmup exceptions propagate, since a collected span does not make failed work
        successful; use parametrized pytest trials when points need independent failure
        handling and durable receipts.

        warm: run the first point once, unmeasured, so what a target compiles or allocates on
            its first call is not charged to that point. Without it the first row of a GPU sweep
            read 4630 ms against its neighbours' 2.5.
        """
        rows = []
        if warm and self.points:
            work(self.points[0])
        for point in self.points:
            started = time.perf_counter()
            with Profiler.under(self.collection, gpus=self.gpus) as profiler:
                work(point)
            rows.append(
                Row(
                    label=point.label,
                    point=point,
                    profile=profiler.result(),
                    seconds=time.perf_counter() - started,
                )
            )
        return tuple(rows)

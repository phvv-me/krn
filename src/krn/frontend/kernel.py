"""Kernel objects, structs, targets, intrinsic registry, launch geometry and raw assembly escapes."""

import inspect
from collections.abc import Callable, Iterator
from dataclasses import fields, is_dataclass
from string.templatelib import Template as TString
from typing import TYPE_CHECKING, Concatenate, cast, get_origin, overload

from krn.compiler.debug import Explanation
from krn.facts.targets import Feature, Target
from krn.runtime.api import Configured, Device, HostArgument, Launcher
from krn.runtime.settings import current_settings, retained

from .geometry import Geometry, Tiles
from .numerics import Scalar, todo
from .records import Const

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class Kernel[**P, R = None, S = HostArgument, **Q = ...]:
    """A function that runs on the device. Capture decides its role from how it is used.

    A host call infers the geometry from the body: `saxpy(a, x, y)`. Brackets carry only a configuration,
    `gemm[Tiling(bm=64)](a, b, c)`, and `launch=` overrides the inferred geometry. A configured kernel, whose first
    parameter is a `@struct` with `Const` fields, uses the defaults of that struct when the call omits it. Each distinct
    configuration is its own specialization, a template argument in the C++ sense, and a field may be a type selector
    such as `fp.min` that the target resolves. Called from another kernel it is a callee, expanded into the caller.

    P and R describe an entry signature and its result. S and Q describe the configuration and remaining signature
    when a family supplies them. Other configured host calls are validated at runtime. Scalar and generator callees keep
    their complete function type through the decorator overload, including generic argument/result relationships.
    """

    __slots__ = ("_shape", "fn", "name")

    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn
        self.name = getattr(fn, "__name__", type(fn).__name__)  # a partial or callable object has no __name__
        self._shape: tuple[int, type[DataclassInstance] | None] | None = None

    def _inspect(self) -> tuple[int, type[DataclassInstance] | None]:
        """The parameter count and the configuration struct of the first parameter (a dataclass with `Const` fields),
        if it has one. Read on the first launch: annotations evaluate lazily and may name what follows the kernel."""
        if self._shape is None:
            parameters = tuple(inspect.signature(self.fn).parameters.values())
            first = parameters[0].annotation if parameters else None
            match get_origin(first) or first:
                case type() as struct if is_dataclass(struct) and any(
                    get_origin(f.type) is Const for f in fields(struct)
                ):
                    self._shape = len(parameters), struct
                case _:
                    self._shape = len(parameters), None
        return self._shape

    @overload
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    @overload
    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> R: ...
    @overload
    def __call__(self, *args: HostArgument, launch: Geometry | None = None, **kwargs: HostArgument) -> R: ...
    @retained
    def __call__[Return](
        self: Kernel[P, Return, S, Q],
        *args: HostArgument,
        launch: Geometry | None = None,
        **kwargs: HostArgument,
    ) -> Return | None:
        """Launch on the current device, evaluating the body's iteration space on these arguments."""
        # Binding Return through self keeps pyrefly from substituting R's None default in overload consistency checks.
        # A host launcher ignores the return type used when this kernel is captured as a callee.
        device = Device.current()
        launcher = device.launcher(self, launch.blocks if isinstance(launch, Tiles) else launch)
        arity, configuration = self._inspect()
        if configuration is not None and len(args) + len(kwargs) == arity - 1:
            try:
                config = current_settings().configuration(self.name, configuration)
            except TypeError:
                raise TypeError(
                    f"{self.name}: configuration needs arguments; write {self.name}[Config(...)](...)"
                ) from None
            # The inspected first parameter is this dataclass; the remaining signature belongs to Configured.
            launcher = Configured(cast("Launcher[Concatenate[DataclassInstance, P]]", launcher), config)
        # ParamSpec arguments are opaque here; the overloads check the host call's signature.
        return cast(Callable[..., Return | None], launcher)(*args, **kwargs)

    @retained
    def explain(self, *args: HostArgument, launch: Geometry | None = None) -> Explanation:
        device = Device.current()
        launcher = device.launcher(self, launch.blocks if isinstance(launch, Tiles) else launch)
        arity, configuration = self._inspect()
        if configuration is not None and len(args) == arity - 1:
            args = (current_settings().configuration(self.name, configuration), *args)
        return cast(Callable[..., Explanation], launcher.explain)(*args)

    @retained
    def __getitem__(self, configuration: S) -> Launcher[Q]:
        """Specialize on a configuration; launch geometry belongs in `launch=`."""
        _, kind = self._inspect()
        if kind is None or not isinstance(configuration, kind):
            raise TypeError(f"{self.name}: brackets take a configuration; pass geometry with launch=")
        launcher = Device.current().launcher(self)
        return Configured(cast(Launcher[Concatenate[S, Q]], launcher), configuration)


@overload
def kernel[F: Callable[..., Scalar | int | float | bool | Iterator]](fn: F) -> F: ...
@overload
def kernel[**P, R](fn: Callable[P, R]) -> Kernel[P, R]: ...
def kernel[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """The one decorator of device code; capture finds out from the uses whether a function is an entry point, a
    callee, configured or folded. An `async def` kernel awaits `Pending` copies and multiplies, which is how a pipeline
    is written. The value-returning overload preserves the whole callable type because splitting its generic
    signature across a ParamSpec and a result type loses that relationship in pyrefly."""
    return Kernel(fn)


target: Target = Target.of("sm_90a")  # the target of the kernel being captured; a constant in every kernel body


def static_assert(condition: bool, message: str) -> None:
    """Fail the capture with the user's file and line when the condition is false. The condition must be constant."""


def static_warn(condition: bool, message: str) -> None:
    """Warn at capture, once per kernel specialization, when the condition is false."""


def requires[**P, R](*features: Feature) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Fail the capture for any target missing one of the features, listing the targets that have all of them."""
    return todo()


class Intrinsic[**P, R]:
    """One typed contract with one lowering per target. A portable fallback is registered with `Target`."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return todo()

    def lower[T: Target](
        self, target: type[T], *, when: Callable[[T], bool] | None = None
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        return todo()


def intrinsic[**P, R](fn: Callable[P, R]) -> Intrinsic[P, R]:
    return todo()


class Out:
    """Marks the result operand inside an asm template."""


OUT = Out()


def ptx[T: Scalar](template: TString, out: type[T]) -> T:
    """Inline PTX. Interpolations become operands and their dtypes pick the register constraints."""
    return todo()


def amdgcn[T: Scalar](template: TString, out: type[T]) -> T:
    """Inline GCN assembly."""
    return todo()


def msl[T: Scalar](template: TString, out: type[T]) -> T:
    """A raw MSL expression, since Metal has no inline assembly."""
    return todo()

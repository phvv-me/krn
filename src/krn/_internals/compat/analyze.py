"""Decide every cell of the compatibility matrix by compiling a probe kernel for each target.

A probe is a feature, the kernel source that calls it (None when the row is read from the tables alone), and a judge.
For each target the analyzer captures and emits the kernel. The outcome is:
- a TargetError: REFUSED, with the error's reason and alternative;
- any other CaptureError: REFUSED, with the message as the reason and `ALTERNATIVE` as the alternative, since krn's own
  refusal names no alternative of its own;
- NotImplementedError or ValueError: UNKNOWN, with the message, because krn has not classified it as a refusal;
- success: the judge decides NATIVE or COMPOSED with a length, from the lowering table or the dispatcher.

Warnings are silenced: the analysis reads the outcome, and a fallback warning reports the same fact in another form.
"""

import inspect
import tempfile
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

from krn import intrinsics, math
from krn.compiler import narrow
from krn.compiler.api import CaptureError, capture, emit
from krn.compiler.dtypes import F32, Dtype, from_sketch
from krn.compiler.errors import Span, TargetError
from krn.compiler.ir import Const
from krn.dispatch import registry as dispatch
from krn.facts import catalog
from krn.facts.targets import AMD, Apple, Nvidia, Target
from krn.facts.targets import Feature as TargetFeature
from krn.frontend import numerics

from .model import Feature, Hardware, Matrix, Status, Support
from .requires import requirement

ALTERNATIVE = "write the operation in a type the target computes natively, or convert the operands to f16, bf16 or f32"
SPAN = Span("<compat>", 1, 1)
CDNA = {"gfx942": "CDNA3", "gfx950": "CDNA4"}
FLOAT_TYPES = ("f64", "f32", "f16", "bf16")
GENERAL = ("add_sat", "sub_sat", "avg", "abs_diff", "maximum")
EXCLUDED = frozenset({"kernel", "sum", "todo"})  # not functions a kernel calls on values
PREDICATES = ("isnan", "isfinite")  # a boolean result, stored as one byte
NUMBER_TYPES = (
    "f64", "f32", "f16", "bf16", "f8e4m3fn", "f8e5m2", "f8e4m3fnuz", "f8e5m2fnuz", "f6e2m3", "f6e3m2", "f4e2m1",
    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64",
)  # fmt: skip
PACKED_TYPES = ("u8x4", "i8x4", "u16x2", "i16x2")
OPERATIONS = {
    "convert": "convert to f32 and back",
    "load_store": "load one element and store it",
    "arithmetic": "add two elements (a packed word also subtracts them, lane by lane)",
    "matmul": "multiply and accumulate a 32 by 32 tile into float32",
    "lanes": "unpack the lanes of a word and pack them again",
}
TOKENS = {"f6e2m3": "f6", "f6e3m2": "f6"}  # facts.targets names both fp6 formats with one capability token


type Judge = Callable[[Target], Support]


@dataclass(frozen=True, kw_only=True)
class Probe:
    """A feature, its kernel (`entry` names the function in `source`) and the judge of a compile that succeeded."""

    feature: Feature
    judge: Judge
    source: str | None = None
    entry: str = "probe"


def combine(supports: Iterable[Support]) -> Support:
    """One support for the parts of a feature: the first refusal or unknown, else NATIVE, else COMPOSED with the sum."""
    parts = list(supports)
    for part in parts:
        if part.status not in (Status.NATIVE, Status.COMPOSED):
            return part
    composed = [part for part in parts if part.status is Status.COMPOSED]
    if not composed:
        return Support(status=Status.NATIVE)
    notes = "; ".join(dict.fromkeys(part.note for part in composed))
    return Support(status=Status.COMPOSED, note=notes, length=sum(part.length or 0 for part in composed) or None)


def spelled(entry: catalog.Entry, target: Target) -> Support:
    """The catalog's spelling on the target. No spelling is a refusal, a software helper is composed with its statement
    count as the length, and a direct spelling is native."""
    if isinstance(target, Nvidia) and target.sm < entry.sm:
        note = f"`{entry.name}` needs sm_{entry.sm} or later"
    elif entry.feature is not None and not target.has(entry.feature):
        note = f"`{entry.name}` needs {entry.feature.value}, which {target.arch} lacks"
    elif not entry.lowering(target):
        note = f"`{entry.name}` has no lowering for {target.arch}: {entry.note or 'no instruction in this backend'}"
    else:
        note = ""
    if note:
        instead = "use a portable expression or a target with this intrinsic"
        return Support(status=Status.REFUSED, note=note, instead=instead)
    if not (helper := entry.helper_for(target.vendor)):
        return Support(status=Status.NATIVE)
    note = f"`{entry.name}` calls a software helper on {target.arch}"
    return Support(status=Status.COMPOSED, note=note, length=dispatch.statements(helper))


def catalog_judge(entry: catalog.Entry) -> Judge:
    """The judge of a catalog row: the dispatcher's lowering for a `general.*` row, else the spelling of the row, or the
    combined spellings of the rows it is made of."""
    if entry.name.startswith("general."):
        _, name, operand = entry.name.split(".")
        return dispatched(name, next(d for d in dispatch.SAMPLES if str(d) == operand))
    parts = [catalog.BY_NAME[name] for name in entry.parts] or [entry]
    return lambda target: combine(spelled(part, target) for part in parts)


def dispatched(name: str, dtype: Dtype) -> Judge:
    """The judge of a general function on one operand type, from the dispatcher's own lowering. A software sequence of
    one operation is that operation, so it counts as native."""

    def judge(target: Target) -> Support:
        overload = dispatch.resolve(name, dtype, target, SPAN)
        operand = Const(dtype, 0, span=SPAN)
        built = dispatch.lower(overload, (operand, operand), SPAN, target)
        if built.native or built.instructions <= 1:
            return Support(status=Status.NATIVE)
        note = f"{overload.name} on {dtype} is a software sequence on {target.arch}"
        return Support(status=Status.COMPOSED, note=note, length=built.instructions)

    return judge


def conversion(dtype: Dtype) -> Judge:
    """Conversion to and from f32: a narrow float is decoded or encoded in software without its instruction."""

    def judge(target: Target) -> Support:
        parts = []
        for source, dest in ((dtype, F32), (F32, dtype)):
            for side in (source, dest):
                if narrow.is_narrow(side) and not narrow.native(target, side):
                    note = f"{narrow.label(side)} converts in software on {target.arch}"
                    length = narrow.statements(narrow.FORMATS[side], decoding=side == source)
                    parts.append(Support(status=Status.COMPOSED, note=note, length=length))
        return combine(parts or [Support(status=Status.NATIVE)])

    return judge


def matmul(dtype: Dtype) -> Judge:
    """A tile product into float32: native where a matrix unit multiplies this type, else composed on the vector units.
    A type the compiler refuses never reaches this judge."""
    token = TOKENS.get(narrow.label(dtype), narrow.label(dtype))

    def judge(target: Target) -> Support:
        if target.has(TargetFeature.TENSOR_CORE) and token in target.mma:
            return Support(status=Status.NATIVE)
        return Support(status=Status.COMPOSED, note=f"{dtype} multiplies on the vector units of {target.arch}")

    return judge


def native(target: Target) -> Support:
    """A load, a store, or a library or instruction call that the lowering emits directly."""
    return Support(status=Status.NATIVE)


def type_probes() -> list[Probe]:
    """Four operations on every number type of `krn.frontend.numerics`, each a kernel over that type. A packed word has
    no matrix multiply: its rows are the lane functions instead."""
    probes = []
    for name in NUMBER_TYPES:
        dtype = from_sketch(getattr(numerics, name))
        judges: dict[str, Judge] = {
            "convert": conversion(dtype),
            "load_store": native,
            "arithmetic": native,
            "matmul": matmul(dtype),
        }
        probes += [_type_probe(name, operation, judge) for operation, judge in judges.items()]
    for name in PACKED_TYPES:
        judges = {"convert": native, "load_store": native, "arithmetic": native, "lanes": native}
        probes += [_type_probe(name, operation, judge) for operation, judge in judges.items()]
    return probes


def _type_probe(name: str, operation: str, judge: Judge) -> Probe:
    feature = Feature(id=f"{name}.{operation}", module="types", group=name, name=operation, doc=OPERATIONS[operation])
    return Probe(feature=feature, source=_type_source(name, operation), judge=judge)


def _type_source(name: str, operation: str) -> str:
    header = "from krn import T, Matrix, Vector, block, grid, kernel, math, zeros\n@kernel\n"
    guard = "    i = grid.thread_rank()\n    if i < out.shape[0]:\n"
    if operation == "arithmetic" and name in PACKED_TYPES:
        body = (
            f"def probe(a: Vector[T.{name}], b: Vector[T.{name}], out: Vector[T.{name}], diff: Vector[T.{name}])"
            " -> None:\n"
            f"{guard}        out[i] = a[i] + b[i]\n        diff[i] = a[i] - b[i]\n"
        )
    elif operation == "lanes":
        lanes = ", ".join(f"w{k}" for k in range(4 if name.endswith("x4") else 2))
        body = (
            f"def probe(a: Vector[T.{name}], out: Vector[T.{name}]) -> None:\n"
            f"{guard}        {lanes} = math.unpack(a[i])\n        out[i] = math.pack({lanes})\n"
        )
    elif operation == "matmul":
        body = (
            f"def probe(a: Matrix[T.{name}], b: Matrix[T.{name}], c: Matrix[T.f32]) -> None:\n"
            "    rows, cols = block.tile(32, 32)\n    acc = zeros(T.f32, 32, 32)\n"
            "    for ks in range(0, a.shape[1], 32):\n"
            "        acc += a[rows, slice(ks, ks + 32)] @ b[slice(ks, ks + 32), cols]\n"
            "    c[rows, cols] = acc.to(T.f32)\n"
        )
    elif operation == "arithmetic":
        body = (
            f"def probe(a: Vector[T.{name}], b: Vector[T.{name}], out: Vector[T.{name}]) -> None:\n"
            f"{guard}        out[i] = a[i] + b[i]\n"
        )
    elif operation == "convert":
        body = (
            f"def probe(a: Vector[T.{name}], out: Vector[T.{name}]) -> None:\n"
            f"{guard}        x = T.f32(a[i])\n        out[i] = T.{name}(x)\n"
        )
    else:
        body = f"def probe(a: Vector[T.{name}], out: Vector[T.{name}]) -> None:\n{guard}        out[i] = a[i]\n"
    return header + body


def function_probes() -> list[Probe]:
    """The public functions of `krn.math` and `krn.intrinsics`, each called on each operand type of its row."""
    probes = [_call_probe(math, name, operand) for name in _public(math) for operand in _operand_types(name)]
    probes += [_call_probe(intrinsics, name, "u32") for name in _public(intrinsics)]
    return probes


def _public(module: ModuleType) -> list[str]:
    return sorted(
        name
        for name, function in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_") and function.__module__ == module.__name__ and name not in EXCLUDED
    )


def _operand_types(name: str) -> tuple[str, ...]:
    match name:
        case _ if name in GENERAL:
            return tuple(str(dtype) for dtype in dispatch.SAMPLES)
        case "pack":
            return ("u8", "i8", "u16", "i16")
        case "unpack":
            return ("u8x4", "i8x4", "u16x2", "i16x2")
        case _:
            return FLOAT_TYPES


def _call_probe(module: ModuleType, name: str, operand: str) -> Probe:
    leaf = module.__name__.rpartition(".")[2]
    function = getattr(module, name)
    feature = Feature(
        id=f"{module.__name__}.{name}[{operand}]",
        module=module.__name__,
        group=leaf,
        name=name,
        signature=str(inspect.signature(function)),
        doc=_summary(function),
    )
    if module is math and name in GENERAL:
        judge = dispatched(name, next(d for d in dispatch.SAMPLES if str(d) == operand))
    else:
        judge = native
    return Probe(feature=feature, source=_call_source(module, name, operand), judge=judge)


def _call_source(module: ModuleType, name: str, operand: str) -> str:
    """A kernel that calls `module.name` on one operand type and stores the result in `out`."""
    leaf = module.__name__.rpartition(".")[2]
    if name == "pack":
        lanes = 4 if operand.endswith("8") else 2
        params = [f"a{i}: Vector[T.{operand}]" for i in range(lanes)]
        statement = f"out[i] = {leaf}.pack({', '.join(f'a{i}[i]' for i in range(lanes))})"
        out = f"out: Vector[T.{operand}x{lanes}]"
    elif name == "unpack":
        lanes = int(operand[-1])
        names = ", ".join(f"w{i}" for i in range(lanes))
        params = [f"a0: Vector[T.{operand}]"]
        statement = f"{names} = {leaf}.unpack(a0[i])\n        out[i] = w0"
        out = f"out: Vector[T.{operand[: -len(f'x{lanes}')]}]"
    else:
        arity = sum(1 for p in inspect.signature(getattr(module, name)).parameters.values() if p.default is p.empty)
        types = [operand, "i32"] if name == "ldexp" else [operand] * arity
        params = [f"a{i}: Vector[T.{t}]" for i, t in enumerate(types)]
        result = _result(name, operand)
        call = f"{leaf}.{name}({', '.join(f'a{i}[i]' for i in range(arity))})"
        statement = f"out[i] = T.{result}({call})" if name in PREDICATES else f"out[i] = {call}"
        out = f"out: Vector[T.{result}]"
    signature = ", ".join([*params, out])
    return (
        f"from krn import T, Vector, grid, kernel, {leaf}\n@kernel\ndef probe({signature}) -> None:\n"
        f"    i = grid.thread_rank()\n    if i < out.shape[0]:\n        {statement}\n"
    )


def _result(name: str, operand: str) -> str:
    """The element type a call stores: `u8` for a predicate, the unsigned twin of a signed magnitude, else the
    operand."""
    if name in PREDICATES:
        return "u8"
    return "u" + operand[1:] if name == "abs_diff" and operand.startswith("i") else operand


def _summary(function: Callable[..., object]) -> str:
    return (inspect.getdoc(function) or "").partition("\n")[0]


def catalog_module(name: str) -> str:
    """The surface a catalog row is reached through: `fast.*` rows are the krn.fast.math calls and `general.*` rows the
    krn.math general functions. Every other row is the catalog itself, reached through krn.intrinsics or krn.math."""
    if name.startswith("fast."):
        return "krn.fast.math"
    if name.startswith("general."):
        return "krn.math"
    return "catalog"


def catalog_feature(entry: catalog.Entry, dtypes: Sequence[str] = ()) -> Feature:
    """The row of a catalog entry, with its operand types in the id when the entry has one row per type."""
    suffix = f"[{','.join(dtypes)}]" if dtypes else ""
    signature = ""
    if entry.signature is not None:
        params = ", ".join(f"{p.name}: {p.kind}" for p in entry.signature.params)
        signature = f"({params}) -> {entry.signature.result}"
    return Feature(
        id=f"{entry.name}{suffix}",
        module=catalog_module(entry.name),
        group=entry.name.split(".")[0],
        name=entry.name,
        signature=signature,
        doc=entry.summary,
    )


def family(target: Target) -> str:
    """The family that decides what the target has, as a name a reader knows: sm_90, CDNA3, Apple family 9, x86-64."""
    match target:
        case Nvidia():
            return f"sm_{target.sm}"
        case AMD():
            return CDNA.get(target.arch, target.arch)
        case Apple():
            return f"Apple family {target.family}"
        case _:
            return "x86-64"


def _cell(probe: Probe, source: Path | None, target: Target) -> Support:
    try:
        if source is not None:
            emit(capture(source, probe.entry, target=target), target)
        support = probe.judge(target)
    except TargetError as error:
        return Support(status=Status.REFUSED, note=error.reason, instead=error.instead)
    except CaptureError as error:
        return Support(status=Status.REFUSED, note=error.message, instead=ALTERNATIVE)
    except (NotImplementedError, ValueError) as error:
        return Support(status=Status.UNKNOWN, note=str(error))
    if support.status is Status.NATIVE:
        return replace(support, requires=requirement(target))
    return support


def analyze(probes: Sequence[Probe], targets: Sequence[Target], generated_from: Mapping[str, str]) -> Matrix:
    """The matrix of `probes` on `targets`. A feature id that is listed twice is an error."""
    features: dict[str, Feature] = {}
    cells: dict[str, dict[str, Support]] = {}
    with tempfile.TemporaryDirectory() as scratch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for index, probe in enumerate(probes):
            if probe.feature.id in features:
                raise ValueError(f"feature {probe.feature.id} is listed twice")
            features[probe.feature.id] = probe.feature
            source = None
            if probe.source is not None:
                source = Path(scratch) / f"probe{index}.py"
                source.write_text(probe.source)
            cells[probe.feature.id] = {target.arch: _cell(probe, source, target) for target in targets}
    return Matrix(
        features=tuple(features[key] for key in sorted(features)),
        targets=tuple(Hardware(arch=t.arch, vendor=t.vendor, family=family(t)) for t in targets),
        cells=cells,
        generated_from=dict(generated_from),
    )

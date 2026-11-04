"""Atomic read-modify-write on one element of a tensor, with the same names and the same result on every vendor.

Every function takes the tensor, the index (an int, or a tuple with one int per axis) and the operand, applies the
operation to `tensor[index]` as one indivisible step and returns the value the element had before:

    atomic.add(counts, bin, 1)              histogram
    atomic.max(best, 0, score)              running maximum
    old = atomic.cas(lock, 0, 0, 1)         try to take a lock: 0 means this thread got it

The eleven names are the ones of numba-cuda-mlir's `cuda.atomic` (`and_` and `or_` keep the underscore of the keyword),
and the accepted element types are a superset of its page: unsigned add, subtract, minimum and maximum are there too,
because the hardware has them. What differs between vendors is only how an operation is lowered, and `plan(...)` says:

    NATIVE     one instruction (atomicAdd, atomic_fetch_add_explicit, __atomic_fetch_add)
    CAS_LOOP   a compare-and-swap loop over the bit pattern of the element, correct but slower under contention
    REFUSED    the capture fails, naming the target and the reason (a 64-bit atomic on Apple, an add on a bool)

Which instructions exist is an inference from the vendors' documentation and is checked only where the prototype can
compile the lowering (CUDA through NVRTC, the CPU through clang). `reference(...)` is the semantics in plain Python, the
oracle of the tests and of the emulator.
"""

from dataclasses import dataclass
from enum import StrEnum, auto

from krn.facts.targets import Feature, Target

from .memory import Tensor
from .numerics import Integer, Scalar, todo


class AtomicOp(StrEnum):
    ADD = auto()
    SUB = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    EXCH = auto()
    INC = auto()  # wraps to zero: 0 if old >= value else old + 1
    DEC = auto()  # wraps to the limit: value if old == 0 or old > value else old - 1
    MAX = auto()
    MIN = auto()
    CAS = auto()

    def apply(self, old: int | float, value: int | float) -> int | float:
        """The new value of the element for every operation except `CAS`, which needs the expected value."""
        match self:
            case AtomicOp.ADD:
                return old + value
            case AtomicOp.SUB:
                return old - value
            case AtomicOp.AND:
                return int(old) & int(value)
            case AtomicOp.OR:
                return int(old) | int(value)
            case AtomicOp.XOR:
                return int(old) ^ int(value)
            case AtomicOp.EXCH:
                return value
            case AtomicOp.INC:
                return 0 if old >= value else old + 1
            case AtomicOp.DEC:
                return value if old == 0 or old > value else old - 1
            case AtomicOp.MAX:
                return max(old, value)
            case AtomicOp.MIN:
                return min(old, value)
            case AtomicOp.CAS:
                raise TypeError("compare-and-swap takes the expected value; use `reference`")


class Strategy(StrEnum):
    NATIVE = auto()
    CAS_LOOP = auto()
    REFUSED = auto()


@dataclass(frozen=True, slots=True)
class Plan:
    strategy: Strategy
    reason: str = ""
    flushes: bool = False  # the instruction exists but flushes subnormals, so a loop runs instead


_INTS = frozenset({"i32", "u32", "i64", "u64"})
_FLOATS = frozenset({"f32", "f64"})
OPERANDS: dict[AtomicOp, frozenset[str]] = {
    AtomicOp.ADD: _INTS | _FLOATS,
    AtomicOp.SUB: _INTS | _FLOATS,
    AtomicOp.AND: _INTS,
    AtomicOp.OR: _INTS,
    AtomicOp.XOR: _INTS,
    AtomicOp.EXCH: _INTS,
    AtomicOp.INC: frozenset({"u32", "u64"}),
    AtomicOp.DEC: frozenset({"u32", "u64"}),
    AtomicOp.MAX: _INTS | _FLOATS,
    AtomicOp.MIN: _INTS | _FLOATS,
    AtomicOp.CAS: _INTS,
}
_GPU_NATIVE: dict[AtomicOp, frozenset[str]] = {
    AtomicOp.ADD: _INTS | _FLOATS,
    AtomicOp.SUB: _INTS | _FLOATS,  # subtraction is an addition of the negation
    AtomicOp.AND: _INTS,
    AtomicOp.OR: _INTS,
    AtomicOp.XOR: _INTS,
    AtomicOp.EXCH: _INTS,
    AtomicOp.INC: frozenset({"u32"}),
    AtomicOp.DEC: frozenset({"u32"}),
    AtomicOp.MAX: _INTS,
    AtomicOp.MIN: _INTS,
    AtomicOp.CAS: _INTS,
}
NATIVE: dict[str, dict[AtomicOp, frozenset[str]]] = {
    "nvidia": _GPU_NATIVE,
    "amd": _GPU_NATIVE,
    "apple": {
        AtomicOp.ADD: _INTS,
        AtomicOp.SUB: _INTS,
        AtomicOp.AND: _INTS,
        AtomicOp.OR: _INTS,
        AtomicOp.XOR: _INTS,
        AtomicOp.EXCH: _INTS,
        AtomicOp.MAX: _INTS,
        AtomicOp.MIN: _INTS,
        AtomicOp.CAS: _INTS,
    },
    "cpu": {
        op: _INTS
        for op in (
            AtomicOp.ADD,
            AtomicOp.SUB,
            AtomicOp.AND,
            AtomicOp.OR,
            AtomicOp.XOR,
            AtomicOp.EXCH,
            AtomicOp.CAS,
        )
    },
}  # the instruction exists for these element types; anything else in OPERANDS is a CAS loop
FLUSHING: dict[str, dict[AtomicOp, frozenset[str]]] = {
    "nvidia": {AtomicOp.ADD: frozenset({"f32"}), AtomicOp.SUB: frozenset({"f32"})},
}  # PTX `atom.add.f32` without `.noftz` flushes subnormals to zero on global memory; generic pointers keep the loop


def plan(op: AtomicOp, dtype: str, target: Target) -> Plan:
    """How `op` on elements of `dtype` is lowered for `target`; `dtype` is the name of a number type: `f32`, `u64`."""
    if dtype not in OPERANDS[op]:
        return Plan(
            Strategy.REFUSED,
            f"atomic.{op} is not defined for {dtype}; it takes {sorted(OPERANDS[op])}",
        )
    if dtype.endswith("64") and not target.has(Feature.ATOMICS_64):
        return Plan(Strategy.REFUSED, f"{target.arch} has no 64-bit atomics, and {dtype} is 64 bits wide")
    if dtype == "f64" and not target.has(Feature.FLOAT64):
        return Plan(Strategy.REFUSED, f"{target.arch} has no float64")
    if dtype in FLUSHING.get(target.vendor, {}).get(op, frozenset()):
        return Plan(
            Strategy.CAS_LOOP,
            f"{target.arch}'s native {op} for {dtype} flushes subnormal inputs and results to zero",
            flushes=True,
        )
    if dtype in NATIVE[target.vendor].get(op, frozenset()):
        return Plan(Strategy.NATIVE)
    return Plan(Strategy.CAS_LOOP, f"{target.arch} has no {op} instruction for {dtype}")


def reference(op: AtomicOp, old: int | float, value: int | float, expected: int | float | None = None) -> int | float:
    """The new value of the element, as the oracle of the tests: `CAS` stores `value` only where `old == expected`."""
    if op is AtomicOp.CAS:
        return value if old == expected else old
    return op.apply(old, value)


type Index = Integer | int | tuple[Integer | int, ...]


class AtomicScope:
    """The eleven operations, called as `atomic.add(...)`. Each returns the element as it was before the update."""

    def add[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, value: T | float) -> T:
        """Atomically add, returning the old value.

        Floating-point contenders arrive in nondeterministic order, so the rounded sum
        may differ between runs. A deterministic alternative is a fixed-order sum with
        one writer; sparse products can select Determinism.STABLE and a row layout.
        """
        return todo()

    def sub[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, value: T | float) -> T:
        """`tensor[index] -= value`."""
        return todo()

    def and_[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, value: T | float) -> T:
        """`tensor[index] &= value` on integers."""
        return todo()

    def or_[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, value: T | float) -> T:
        """`tensor[index] |= value` on integers."""
        return todo()

    def xor[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, value: T | float) -> T:
        """`tensor[index] ^= value` on integers."""
        return todo()

    def exch[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, value: T | float) -> T:
        """`tensor[index] = value`, and the old value comes back."""
        return todo()

    def inc[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, limit: T | float) -> T:
        """`tensor[index] = 0 if tensor[index] >= limit else tensor[index] + 1`, a counter that wraps."""
        return todo()

    def dec[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, limit: T | float) -> T:
        """`tensor[index] = limit if it is 0 or above limit else tensor[index] - 1`."""
        return todo()

    def max[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, value: T | float) -> T:
        """`tensor[index] = max(tensor[index], value)`."""
        return todo()

    def min[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, value: T | float) -> T:
        """`tensor[index] = min(tensor[index], value)`."""
        return todo()

    def cas[T: Scalar, *S](self, tensor: Tensor[T, *S], index: Index, expected: T | float, value: T | float) -> T:
        """`if tensor[index] == expected: tensor[index] = value`, as one step. The old value says whether it worked."""
        return todo()


atomic = AtomicScope()

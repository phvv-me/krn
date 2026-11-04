"""System-scope atomic operations on tensor elements, returning the old value."""

from krn.frontend.numerics import Scalar, todo


def add[T: Scalar](target: T, value: T) -> T:
    return todo()


def sub[T: Scalar](target: T, value: T) -> T:
    return todo()


def and_[T: Scalar](target: T, value: T) -> T:
    return todo()


def or_[T: Scalar](target: T, value: T) -> T:
    return todo()


def xor[T: Scalar](target: T, value: T) -> T:
    return todo()


def exch[T: Scalar](target: T, value: T) -> T:
    return todo()


def inc[T: Scalar](target: T, limit: T) -> T:
    return todo()


def dec[T: Scalar](target: T, limit: T) -> T:
    return todo()


def max[T: Scalar](target: T, value: T) -> T:
    return todo()


def min[T: Scalar](target: T, value: T) -> T:
    return todo()


def cas[T: Scalar](target: T, expected: T, value: T) -> T:
    return todo()

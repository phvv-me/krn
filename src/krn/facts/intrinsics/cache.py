"""Load, store, and prefetch hints that preserve values on every backend."""

from krn.frontend.numerics import Scalar, todo


def ldg[T: Scalar](x: T) -> T:
    return todo()


def ldcg[T: Scalar](x: T) -> T:
    return todo()


def ldca[T: Scalar](x: T) -> T:
    return todo()


def ldcs[T: Scalar](x: T) -> T:
    return todo()


def ldlu[T: Scalar](x: T) -> T:
    return todo()


def ldcv[T: Scalar](x: T) -> T:
    return todo()


def stcg[T: Scalar](x: T, value: T) -> None:
    todo()


def stcs[T: Scalar](x: T, value: T) -> None:
    todo()


def stwb[T: Scalar](x: T, value: T) -> None:
    todo()


def stwt[T: Scalar](x: T, value: T) -> None:
    todo()


def prefetch_l1[T: Scalar](x: T) -> None:
    todo()


def prefetch_l2[T: Scalar](x: T) -> None:
    todo()

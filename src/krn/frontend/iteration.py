"""Kernel iteration: thread items, generator folds, and fixed register sequences.

    for i in items(a.rows):
        y[i] = sum(v * x[c] for c, v in a.row(i))

`sum`, `min`, `max`, `any` and `all` fold generators or register sequences in order. Numeric folds take the type of
their terms; float extrema propagate NaN. Generator methods inline where they yield. `enumerate`, `zip`, `reversed`
and signed constant range steps compose with loops. Lists, tuples and comprehensions have a length known at capture:
their elements are snapshots, indexing is constant, and iteration unrolls. Build a new sequence to change its layout.
`match` selects one arm at capture from a literal, dtype, target or `Const` field; runtime conditions use `if`.
"""

from collections.abc import Iterable, Iterator

from .numerics import Integer, Scalar, todo


def items[N: Integer | int](count: N) -> Iterator[N]:
    """The items this thread owns out of `count`: thread `t` of `T` gets `t`, `t + T`, `t + 2T` and so on."""
    return todo()


def sum[S: Scalar](terms: Iterable[S], /) -> S:
    """The terms added in order, in the type of the terms."""
    return todo()

"""Iteration vocabulary parsed by capture, including flattening register sequences on Python 3.14."""

from itertools import chain

from krn import AUTO, Auto, Const, T, Vector, kernel, struct, target, targets


@kernel
def fold_u8(x: Vector[T.u8], n: T.u32, out: Vector[T.u8]) -> None:
    out[0] = sum(x[i] for i in range(n))
    out[1] = min(x[i] for i in range(n))
    out[2] = max(x[i] for i in range(n))


@kernel
def constant_folds(out: Vector[T.u8]) -> None:
    out[0] = min([3, 1, 2])
    out[1] = max((3, 1, 2))
    out[2] = min(T.u8(3), 2)
    out[3] = max(2, T.u8(3))
    out[4] = sum(range(0))


@kernel
def fold_i32(x: Vector[T.i32], n: T.i32, out: Vector[T.i32]) -> None:
    out[0] = sum(x[i] for i in range(n))
    out[1] = min(x[i] for i in range(n))
    out[2] = max(x[i] for i in range(n))


@kernel
def fold_f32(x: Vector[T.f32], n: T.u32, out: Vector[T.f32]) -> None:
    out[0] = sum(x[i] for i in range(n))
    out[1] = min(x[i] for i in range(n))
    out[2] = max(x[i] for i in range(n))


@kernel
def fold_clauses(x: Vector[T.i32], n: T.i32, out: Vector[T.i32]) -> None:
    out[0] = sum(x[i] + x[j] for i in range(n) if i != 1 for j in range(i) if j != 2)
    out[1] = min(x[i] + x[j] for i in range(n) if i > 0 for j in range(i) if j != 2)
    out[2] = max(x[i] - x[j] for i in range(n) if i > 0 for j in range(i) if j != 2)
    out[3] = sum(v + T.i32(k) for k, v in enumerate([x[0], x[1]]))
    out[4] = sum(x[i] + x[j] for i, j in zip(range(n), range(1, n), strict=False))
    out[5] = sum(range(n))
    out[6] = max(range(n))


@kernel
def fold_logic(x: Vector[T.i32], n: T.i32, out: Vector[T.i32]) -> None:
    out[0] = T.i32(any(x[i] > 0 for i in range(n)))
    out[1] = T.i32(all(x[i] > 0 for i in range(n)))
    out[2] = T.i32(any(x[i] < 0 for i in range(n)))
    out[3] = T.i32(all(x[i] < 0 for i in range(n)))
    out[4] = sum([])
    out[5] = T.i32(any(()))
    out[6] = T.i32(all([]))
    out[7] = T.i32(any(x[i] == x[j] for i in range(n) if i > 0 for j in range(i)))
    out[8] = T.i32(all(x[i] == x[j] for i in range(n) if i > 0 for j in range(i)))


@kernel
def range_loops(n: T.u32, first: T.u32, out: Vector[T.u32]) -> None:
    total = 0
    for k, i in enumerate(range(n), first):
        total += k + i
    out[0] = total
    total = 0
    for i in reversed(range(n)):
        total += i
    out[1] = total
    total = 0
    for i in reversed(range(1, n, 2)):
        total = total * 10 + i
    out[2] = total
    total = 0
    for i in range(n, 0, -1):
        total += i
    out[3] = total
    total = 0
    for a, b in zip(range(n), range(1, n, 2), strict=False):
        total += a + b
    out[4] = total
    total = 0
    for a, b in zip(range(n, 0, -2), range(1, n, 3), strict=False):
        total += a + b
    out[5] = total
    total = 0
    for k, (a, b) in enumerate(zip(range(n), range(n, 0, -1), strict=False), start=first):
        total += k + a + b
    out[6] = total


@kernel
def signed_loops(n: T.i32, out: Vector[T.i32]) -> None:
    total = 0
    for i in range(n, -3, -2):
        total += i
    out[0] = total
    total = 0
    for i in reversed(range(-2, n, 3)):
        total = total * 10 + i
    out[1] = total
    total = 0
    for i in reversed(range(n, -3, -2)):
        total = total * 10 + i
    out[2] = total
    for i in range(0, 4, -1):
        total += 1000
    out[3] = total


@kernel
def values(n: int):
    for i in range(n):
        yield i + 1


@kernel
def pairs(n: int):
    for i in range(n):
        yield i, i + 1


@struct
class Rows:
    n: T.u32

    def row(self):
        for i in range(self.n):
            yield i, i + 2


@kernel
def generators(n: T.u32, rows: Rows, out: Vector[T.u32]) -> None:
    total = 0
    for k, v in enumerate(values(n), start=T.u32(3)):
        total += k + v
    out[0] = total
    total = 0
    for k, (c, v) in enumerate(pairs(n)):
        total += k + c + v
    out[1] = total
    out[2] = sum(v for k, (c, v) in enumerate(rows.row()) if k == c)


@kernel
def sequences(x: Vector[T.i32], out: Vector[T.i32]) -> None:
    a = x[0]
    xs = [a, a + 1, x[1]]
    a += 10
    x[1] = 99
    out[0] = xs[0]
    out[1] = xs[-1]
    out[2] = len(xs)
    out[3] = sum(xs[1:] + xs[::-1])
    out[4] = sum(2 * xs[:1] + xs[:1] * 3)
    a, b = x[0], x[2]
    a, b = b, a
    out[5] = a
    out[6] = b
    first, *rest = xs
    *init, last = xs
    first, *_ = xs
    out[7] = first + sum(rest) + sum(init) + last
    ys = [*xs, a]
    out[8] = sum(ys)
    out[9] = min(xs)
    out[10] = max(xs)
    out[11] = T.i32(any([False, True])) + T.i32(all((True, True)))
    out[12] = T.i32(any([False, False])) + T.i32(all((False, True)))
    total = 0
    for i, v in enumerate(xs):
        total += v + xs[i]
        v += 1
    out[13] = total
    total = 0
    for pair in zip(xs, xs[::-1], strict=False):
        a, b = pair
        total += a + b
    out[14] = total
    total = 0
    for a, *middle, b in [(1, 2, 3, 4), (5, 6, 7, 8)]:
        total += a + sum(middle) + b
    out[15] = total
    total = 0
    for v in reversed(tuple(xs)):
        total = total * 10 + v
    out[16] = total
    total = 0
    for i, (a, b) in enumerate([(1, 2), (3, 4)]):
        total += i + a + b
    out[17] = total


@kernel
def comprehensions(x: Vector[T.i32], out: Vector[T.i32]) -> None:
    i = 77
    xs = [x[i] for i in range(4) if i != 1]
    out[0] = i
    out[1] = sum(xs)
    rows = [[x[i] + j for j in (0, 1)] for i in range(2)]
    out[2] = sum(list(chain.from_iterable(rows)))
    out[3] = sum(tuple(chain.from_iterable(rows)))
    suffix = [1]
    out[4] = sum(list(chain.from_iterable(row + suffix for row in rows)))
    out[5] = sum(tuple(v for row in rows for v in row))
    out[6] = sum(list(v for row in rows for v in row))
    out[7] = sum([a + b for a, *rest, b in [(1, 2, 3), (4, 5, 6)] if a != 4])
    out[8] = sum([a + b for a, b in zip(range(4), [1, 2, 3], strict=False)])
    out[9] = sum([i + v for i, v in enumerate([2, 3], start=5)])
    out[10] = sum([v for v in reversed(range(4))])
    out[11] = sum(list(range(4))) + sum(tuple(range(3)))


@struct
class Choice:
    count: Const[int] = 2
    dtype: Const[type] = T.f32
    mode: Const[int | Auto] = AUTO


@kernel
def matches(x: Vector[T.f32], choice: Choice, out: Vector[T.i32]) -> None:
    match x.dtype:
        case T.f16 | T.bf16:
            out[0] = sum(x[0])
        case T.f32 as dtype:
            out[0] = 32
        case _:
            out[0] = sum(x[0])
    match choice.count:
        case 1:
            out[1] = sum(x[0])
        case value if value > 2:
            out[1] = value + 10
        case 2:
            out[1] = 2
    match choice.dtype:
        case T.f16 | T.bf16:
            out[2] = 16
        case _:
            out[2] = 32
    match target:
        case targets.CPU():
            match 3:
                case 3:
                    out[3] = 1
                case _:
                    out[3] = sum(x[0])
        case targets.Nvidia():
            match 4:
                case 3:
                    out[3] = sum(x[0])
                case _:
                    out[3] = 2
    match choice.mode:
        case 4:
            out[4] = sum(x[0])
        case _:
            out[4] = 1
    match True:
        case False:
            out[5] = sum(x[0])
        case True:
            out[5] = 1


@kernel
def runtime_empty(n: T.u32, out: Vector[T.u32]) -> None:
    out[0] = min(i for i in range(n))


@kernel
def plain_range(n: T.u32, out: Vector[T.u32]) -> None:
    for i in range(n):
        out[i] = i


@kernel
def bound_snapshot(x: Vector[T.u32], out: Vector[T.u32]) -> None:
    total = 0
    for i in reversed(range(x[0])):
        total += i
        x[0] = 0
    out[0] = total


@kernel
def scopes(x: Vector[T.i32], out: Vector[T.i32]) -> None:
    a = x[0]
    values = [a for a in (x[1], x[2])]
    out[0] = a
    out[1] = sum(values)
    xs = [1, 2]
    ys = [xs for xs in xs]
    out[2] = sum(xs) + sum(ys)
    out[3] = sum([i for i in range(len(xs)) if not False])
    out[4] = sum(v for _, v in enumerate(values))
    types = (T.i32, x.dtype)
    total = 0
    for dtype in types:
        total += dtype(1)
    out[5] = total
    tensors = (x, x)
    total = 0
    for tensor in tensors:
        total += tensor[0]
    out[6] = total


@kernel
def constant_loop(out: Vector[T.u32]) -> None:
    for i in range(4):
        out[i] = i


@kernel
def long_ranges(out: Vector[T.u32]) -> None:
    total = 0
    for i in reversed(range(1000)):
        total += i
    for k, i in enumerate(range(1000)):
        total += k
    for a, b in zip(range(1000), range(1000), strict=False):
        total += a + b
    out[0] = total


@kernel
def match_facts(x: Vector[T.f32], out: Vector[T.i32]) -> None:
    match target.warp_size:
        case 32:
            out[0] = 32
        case _:
            out[0] = 16
    match T.acc(x.dtype):
        case T.f32:
            out[1] = 32
    match 1:
        case value if not True:
            out[2] = sum(x[0])
        case 1 as one:
            out[2] = one
    match 7:
        case 8:
            out[3] = sum(x[0])
    out[3] = 3

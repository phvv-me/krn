"""`async def` inside a kernel: `await` is the wait on an asynchronous copy, and warp roles are tasks of a task group.

A GPU pipeline is concurrency: loads in flight while earlier tiles are multiplied, and on Hopper and Blackwell separate
warps that only produce or only consume. Python already has the words for it, so the kernel language uses them
instead of inventing barrier phases and parity bits (the thing Gluon, TLX and Pallas Mosaic GPU make users write).

Two styles, the same meaning as the plain `pipelined` loop of the other examples:

1. `await` on `Pending`. `copy(slice)` issues a load and returns at once; `await pending` is the wait that makes
   the data usable (`cp.async.wait_group` on Ampere and Ada, an mbarrier wait on Hopper and Blackwell, `s_waitcnt` on
   AMD, a threadgroup barrier on Apple, nothing on a CPU). Writing the next `copy` before awaiting the current
   tile is double buffering, and the compiler sinks each await as late as the dependences allow.
2. Roles as a `TaskGroup`. `async with block.roles() as roles:` splits the warps of a block, and
   `roles.spawn(coro, warps=n)` runs a coroutine on `n` warps. A `ring` is the channel between them: the producer's
   `await ring.put(tile)` and the consumer's `await ring.get()` are the empty and full mbarriers of a TMA pipeline,
   and the consumer is where `acc += ...` issues the MMA. On a target without independent warps the roles run one
   after the other and the result is the same.

The kernels use `@kernel` (configuration in brackets) and are `async def`, so a type checker sees a coroutine and the
launch is unchanged: `gemm_awaited[Tiling(), Tiles(c.shape, (128, 128))](a, b, c)`. The prototype rejects `async def`
kernels today with a clear error; this is the design the typed surface already supports, not a lowering that exists.

Python features used: `async def`, `await`, `async with`, task groups as the model for concurrent roles, unpacking of
an iterable with a starred target, and `itertools.pairwise`-style lookahead written with plain variables.
"""

from krn import AUTO, Const, Matrix, T, block, chunks, copy, kernel, ring, struct, zeros


@struct
class Tiling:
    bm: Const[int] = 128
    bn: Const[int] = 128
    bk: Const[int] = 32


@kernel
async def gemm_awaited(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """Double buffering with awaits: issue the next tile's loads, then wait for the current one and multiply."""
    rows, cols = block.tile(t.bm, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)
    first, *rest = chunks(a.shape[1], t.bk)
    a_next, b_next = copy(a[rows, first]), copy(b[first, cols])
    for ks in rest:
        a_now, b_now = await a_next, await b_next
        a_next, b_next = copy(a[rows, ks]), copy(b[ks, cols])  # in flight while the product below runs
        acc += a_now @ b_now
    last_a, last_b = await a_next, await b_next
    c[rows, cols] = acc.add(last_a @ last_b)


@kernel
async def gemm_roles(t: Tiling, a: Matrix[T.f16], b: Matrix[T.f16], c: Matrix[T.f32]) -> None:
    """Warp specialization: one warp produces tiles into a ring, two warps consume them."""
    rows, cols = block.tile(t.bm, t.bn)
    a_ring, b_ring = ring(T.f16, AUTO, t.bm, t.bk), ring(T.f16, AUTO, t.bk, t.bn)
    acc = zeros(T.f32, t.bm, t.bn)

    async def produce() -> None:
        for ks in chunks(a.shape[1], t.bk):
            await a_ring.put(a[rows, ks])
            await b_ring.put(b[ks, cols])

    async def consume() -> None:
        nonlocal acc
        for _ in chunks(a.shape[1], t.bk):
            acc += await a_ring.get() @ await b_ring.get()

    async with block.roles() as roles:
        roles.spawn(produce(), warps=1)
        roles.spawn(consume(), warps=2)
    c[rows, cols] = acc

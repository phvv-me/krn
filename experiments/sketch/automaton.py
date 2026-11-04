"""Structs as kernel arguments and byte-level SIMT code, the shape of the cutok kernels.

TileLang cannot pass a struct to a kernel.
"""

from krn import Items, T, Tensor, grid, kernel, struct


@struct
class Dfa:
    table: Tensor[T.u16, int]
    accept: Tensor[T.u8, int]
    start: T.u16

    def step(self, state: T.u16, byte: T.u8) -> T.u16:
        return self.table[T.u32(state) * 256 + T.u32(byte)]


@struct
class Segments:
    text: Tensor[T.u8, int]
    offsets: Tensor[T.u32, int]
    out: Tensor[T.u32, int]


@kernel
def run_dfa(dfa: Dfa, segs: Segments) -> None:
    i = grid.thread_rank()
    if i + 1 < segs.offsets.shape[0]:
        state = dfa.start
        for p in range(segs.offsets[i], segs.offsets[i + 1]):
            state = dfa.step(state, segs.text[p])
        segs.out[i] = T.u32(dfa.accept[state])


def run(dfa: Dfa, segs: Segments) -> None:
    run_dfa(dfa, segs, launch=Items(segs.offsets.shape[0] - 1))

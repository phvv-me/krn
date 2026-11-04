"""The kernels every GPU renderer is checked on: elementwise, tile, struct-argument, collective and ragged ones."""

RAGGED = (
    "row_max",
    "row_min",
    "row_sum",
    "row_prod",
    "softmax_rows",
    "centered",
    "larger_target",
    "scores_max",
)
COMMON = [
    ("elementwise", "saxpy", None),
    ("elementwise", "add", {"tiling": {"rows": 32, "cols": 32}}),
    ("softmax", "softmax", {"t": {"rows": 2, "cols": 1024}}),
    ("gemm", "gemm", {"t": {"bm": 64, "bn": 64, "bk": 16}}),
    ("automaton", "run_dfa", None),
    ("extra", "wrap", None),
    ("extra", "collatz", None),
    ("extra", "shadow", None),
    ("extra", "scale", None),
    *(
        ("intrinsics", name, None)
        for name in (
            "histogram", "largest", "wrapping_counter", "take_lock", "bit_tricks", "math_helpers", "warp_sum", "votes",
            "ordered", "debug_aids", "split_barrier", "tiles_wide",
        )
    ),
    *(("ragged", name, None) for name in RAGGED),
]  # fmt: skip

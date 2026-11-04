"""Execution function modules, geometry, explicit calls, and segmented warp semantics."""

import inspect
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

from krn import Blocks, block, catalog, cluster, grid, thread, warp
from krn.compiler.api import CaptureError, capture, emit
from krn.compiler.capture import INTRINSICS
from krn.compiler.dtypes import F32, I32, U32, Dtype
from krn.compiler.ir import Const, Intrinsic, Kernel, Store
from krn.emulator.emulate import GEOMETRY, Thread, WarpRequest, build, np_dtype
from krn.facts.targets import Target, all_targets
from krn.runtime.cpu import CLANG, CPUProgram, Value
from krn.runtime.nvrtc import Nvidia as Nvrtc
from tests.conftest import KERNELS, build_kernel
from tests.helpers.devices import CUDARun, MetalRun, agree, arrays, fresh
from tests.helpers.text import descriptor

if TYPE_CHECKING:
    from krn.runtime.cuda_backend import CUDADevice
    from krn.runtime.metal import MetalDevice

FILE = KERNELS / "intrinsics.py"
GPU = Target.of("sm_90a")
WITHOUT_CATALOG = {
    "block.tile",
    "block.roles",
    "block.sum",
    "warp.reserve",
    "warp.sum",
    "warp.exclusive_sum",
    "warp.argmax",
}


def test_group_functions_and_lowered_names_agree() -> None:
    functions: set[str] = set()
    for module in (thread, block, grid, warp, cluster):
        assert isinstance(module, ModuleType)
        scope = module.__name__.rsplit(".", 1)[1]
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ != module.__name__:
                continue
            path = f"{scope}.{name}"
            functions.add(path)
            entries = (
                {f"{path}.{a}" for a in "xyz"}
                if path in {"thread.index", "block.index", "block.dim", "grid.dim"}
                else {path}
            )
            if path in WITHOUT_CATALOG:
                assert entries.isdisjoint(catalog.BY_NAME)
            else:
                assert entries <= INTRINSICS.keys() & catalog.BY_NAME.keys()
                spec = INTRINSICS[next(iter(entries))]
                assert spec.parameters == tuple(inspect.signature(function).parameters)
    assert functions >= WITHOUT_CATALOG
    assert {
        name for name in catalog.BY_NAME if name.split(".")[0] in {"thread", "block", "grid", "warp", "cluster"}
    } == set(INTRINSICS)
    removed = {f"{s}.id.{a}" for s in ("thread", "block") for a in "xyz"}
    removed |= {"grid.thread_id", *(f"warp.shuffle{s}" for s in ("", "_up", "_down", "_xor"))}
    removed |= {f"fence.{s}" for s in ("block", "device", "system")}
    assert removed.isdisjoint(catalog.BY_NAME.keys() | INTRINSICS.keys())


@pytest.mark.parametrize("arch", ["sm_90a", "gfx942", "apple_m4", "avx512"])
def test_rendered_ranks_use_every_axis(arch: str) -> None:
    target = Target.of(arch)
    source = emit(capture(FILE, "group_ranks", target=target), target, lineinfo=False).source
    if target.vendor in {"nvidia", "amd"}:
        rank = "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)"
        blocks = "(blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)"
        assert f"{rank} % {target.warp_size}" in source
        assert f"{rank} / {target.warp_size}" in source
        assert f"(unsigned long long){blocks} * (blockDim.x * blockDim.y * blockDim.z) + {rank}" in source
    elif target.vendor == "apple":
        rank = "(_thread.x + _thread.y * _blockdim.x + _thread.z * _blockdim.x * _blockdim.y)"
        blocks = "(_block.x + _block.y * _griddim.x + _block.z * _griddim.x * _griddim.y)"
        assert rank in source
        assert f"(ulong){blocks} * (_blockdim.x * _blockdim.y * _blockdim.z) + {rank}" in source
        assert "(ulong)(_lane)" in source and "(ulong)(_simdgroup)" in source
    else:
        assert "((unsigned)tx)" in source
        assert "bx + by * gx + bz * gx * gy" in source
        assert "* bdim + tx" in source


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("warp.lane", "`warp.lane` is a function; write `warp.lane()`"),
        ("warp.size", "`warp.size` is a function; write `warp.size()`"),
        ("thread.index", "`thread.index` is a function; write `thread.index()`"),
        (
            "thread.index()",
            "`thread.index()` needs an axis; write `thread.index().x`, `.y` or `.z`",
        ),
        ("warp.shfl_xor(1, 1, surprise=8)", "unexpected argument `surprise` to `warp.shfl_xor`"),
        ("warp.shfl_xor(1, 1, 3)", "`warp.shfl_xor` width must be a power of two up to 32"),
        ("warp.shfl_xor(1, 1, 64)", "`warp.shfl_xor` width must be a power of two up to 32"),
    ],
)
def test_bad_calls_fail_at_the_users_line(tmp_path: Path, expression: str, reason: str) -> None:
    file = tmp_path / "bad_call.py"
    file.write_text(
        "from krn import T, Vector, kernel, thread, warp\n@kernel\ndef bad(out: Vector[T.u64]):\n"
        f"    out[0] = T.u64({expression})\n"
    )
    with pytest.raises(CaptureError) as error:
        capture(file, "bad", target=GPU)
    assert f"{file}:4:" in str(error.value)
    assert reason in str(error.value)


@pytest.mark.parametrize(("arch", "name"), [("avx512", "warp_sum"), ("apple_m4", "matching_lanes")])
def test_refusals_name_the_reason_and_every_supported_architecture(arch: str, name: str) -> None:
    path = "warp.shfl_xor" if name == "warp_sum" else "warp.match_all"
    entry = catalog.BY_NAME[path]
    supported = ", ".join(
        t.arch for t in all_targets() if entry.lowering(t) and (entry.feature is None or t.has(entry.feature))
    )
    with pytest.raises(CaptureError) as error:
        capture(FILE, name, target=Target.of(arch))
    message = str(error.value)
    assert f"targets that have it: {supported}" in message
    assert (
        "needs WARP_SHUFFLE, which target avx512 lacks" if arch == "avx512" else "has no lowering for target apple_m4"
    ) in message


@pytest.mark.parametrize("arch", ["sm_90a", "gfx942"])
@pytest.mark.parametrize(
    ("expression", "width", "mask"),
    [
        ("warp.shfl_xor(x[0], lane_mask=1, mask=T.u64(15))", None, 15),
        ("warp.shfl_xor(mask=None, lane_mask=1, x=x[0], width=None)", None, None),
        ("warp.shfl_xor(mask=T.u64(15), width=4, lane_mask=1, x=x[0])", 4, 15),
    ],
)
def test_operands_bind_in_signature_order(
    tmp_path: Path, arch: str, expression: str, width: int | None, mask: int | None
) -> None:
    target = Target.of(arch)
    file = tmp_path / "arguments.py"
    file.write_text(
        "from krn import T, Vector, kernel, warp\n@kernel\n"
        "def bind(x: Vector[T.i32], out: Vector[T.i32]):\n"
        f"    out[0] = {expression}\n"
        "    out[1] = warp.size()\n"
    )
    ir = capture(file, "bind", target=target)
    shuffle, size = ir.body
    assert isinstance(shuffle, Store) and isinstance(shuffle.value, Intrinsic)
    assert all(isinstance(value, Const) for value in shuffle.value.args[1:])
    assert [value.value for value in shuffle.value.args[1:] if isinstance(value, Const)] == [
        1,
        width or target.warp_size,
        mask if mask is not None else (1 << target.warp_size) - 1,
    ]
    assert isinstance(size, Store) and isinstance(size.value, Const)
    assert size.value.value == target.warp_size


@pytest.mark.parametrize("width", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("amount", [1, 3, 16, 32])
def test_shuffles_obey_segment_boundaries(width: int, amount: int) -> None:
    x = np.arange(64, dtype=np.int32) * 7 + 3
    out = np.zeros((64, 4), np.int32)
    build(capture(FILE, "segmented_shuffles", target=GPU), GPU).run(Blocks(1, threads=64), [x, out, width, amount])
    lanes = np.arange(64) % 32
    starts = lanes // width * width
    warp_starts = np.arange(64) // 32 * 32
    for column, source in enumerate((lanes ^ amount, lanes - amount, lanes + amount, starts + amount % width)):
        source = np.where((source >= starts) & (source < starts + width), source, lanes)
        np.testing.assert_array_equal(out[:, column], x[warp_starts + source])


@pytest.mark.parametrize("arch", ["sm_90a", "gfx942", "avx512"])
def test_lane_masks_match_numpy_at_both_warp_sizes(arch: str) -> None:
    target = Target.of(arch)
    size = target.warp_size
    out = np.zeros((size * 2, 5), np.uint64)
    build(capture(FILE, "lane_masks", target=target), target).run(Blocks(1, threads=size * 2), [out])
    lanes = np.arange(size * 2, dtype=np.uint64) % size
    bits = np.left_shift(np.uint64(1), lanes)
    below = bits - np.uint64(1)
    through = bits | below
    full = np.uint64((1 << size) - 1)
    np.testing.assert_array_equal(out, np.column_stack((below, through, bits, full ^ below, full ^ through)))


def test_emulator_and_c_cover_multidimensional_grids() -> None:
    launch = Blocks(2, 3, 2, threads=16)
    count = 2 * 3 * 2 * 16
    emulated, compiled = np.zeros((count, 7), np.uint64), np.zeros((count, 7), np.uint64)
    build(capture(FILE, "group_ranks", target=GPU), GPU).run(launch, [emulated])
    build_kernel(FILE, "group_ranks")(compiled, launch=launch)
    ranks = np.arange(count, dtype=np.uint64)
    expected = np.column_stack(
        (
            ranks % 16,
            ranks % 16,
            ranks * 0,
            ranks // 16,
            ranks * 0 + 16,
            ranks * 0 + count,
            ranks * 0 + 12,
        )
    )
    np.testing.assert_array_equal(emulated, expected)
    # The C target has a 16-lane warp here; the block has no second warp on either target.
    np.testing.assert_array_equal(compiled, expected)


def test_emitter_retains_all_axes_for_multidimensional_blocks() -> None:
    program = build(capture(FILE, "group_ranks", target=GPU), GPU)
    out = np.zeros((48, 7), np.uint64)
    for z, y, x in product(range(2), range(3), range(8)):
        geometry = dict(
            zip(
                (
                    "_bx",
                    "_by",
                    "_bz",
                    "_tx",
                    "_ty",
                    "_tz",
                    "_gx",
                    "_gy",
                    "_gz",
                    "_bdx",
                    "_bdy",
                    "_bdz",
                ),
                (0, 0, 0, x, y, z, 1, 1, 1, 8, 3, 2),
                strict=True,
            )
        )
        assert list(program.function(out=out, out_n0=48, out_n1=7, **geometry)) == []
    ranks = np.arange(48, dtype=np.uint64)
    np.testing.assert_array_equal(out[:, :3], np.column_stack((ranks, ranks % 32, ranks // 32)))
    np.testing.assert_array_equal(out[:, 4:6], np.full((48, 2), 48, np.uint64))


def test_grid_stride_covers_each_item_once_over_every_axis() -> None:
    launch = Blocks(2, 3, 2, threads=4)
    emulated, compiled = np.zeros(275, np.uint32), np.zeros(275, np.uint32)
    build(capture(FILE, "group_items", target=GPU), GPU).run(launch, [emulated])
    build_kernel(FILE, "group_items")(compiled, launch=launch)
    np.testing.assert_array_equal(emulated, np.ones(275, np.uint32))
    np.testing.assert_array_equal(compiled, np.ones(275, np.uint32))


def test_complete_mask_groups_are_answered_in_the_same_round() -> None:
    program = build(capture(FILE, "masked_groups", target=GPU), GPU)
    out = np.zeros(32, np.uint64)
    threads = [
        Thread(lane, program.function(**program.arguments([out]), **dict.fromkeys(GEOMETRY, 0))) for lane in range(32)
    ]
    for lane, state in enumerate(threads):
        state.request = WarpRequest("ballot", 255 if lane < 8 else 0xFFFFFF00, (True,))
    assert program.answer_warps(threads)
    assert all(state.request is None for state in threads)
    assert [state.reply for state in threads] == [255] * 8 + [0xFFFFFF00] * 24
    # An incomplete low group must not keep a complete high group waiting.
    for lane, state in enumerate(threads):
        state.request = WarpRequest("ballot", 255 if lane < 8 else 0xFFFFFF00, (True,))
    threads[0].request = None
    assert program.answer_warps(threads)
    assert threads[1].request is not None
    assert all(state.request is None for state in threads[8:])
    program.run(Blocks(1, threads=32), [out])
    np.testing.assert_array_equal(out, [255] * 8 + [0xFFFFFF00] * 24)


def test_match_all_returns_the_mask_and_predicate() -> None:
    x = np.ones(64, np.int32)
    x[32] = 2
    masks, unanimous = np.zeros(64, np.uint64), np.zeros(64, np.uint32)
    build(capture(FILE, "matching_lanes", target=GPU), GPU).run(Blocks(1, threads=64), [x, masks, unanimous])
    np.testing.assert_array_equal(masks, [0xFFFFFFFF] * 32 + [0] * 32)
    np.testing.assert_array_equal(unanimous, [1] * 32 + [0] * 32)


try:
    from krn.runtime import metal
except ImportError:
    metal = None

CG_FILE = KERNELS / "groups.py"
CLANG_HAS_AMDGPU = (
    "amdgcn"
    in subprocess.run(
        [CLANG[0], "--print-targets"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    ).stdout
)


@dataclass(frozen=True)
class GroupCase:
    name: str
    width: int = 8
    dtype: Dtype = U32
    operation: int = 0
    collective: int = 0
    threads: int = 64

    @property
    def configured(self) -> bool:
        return self.name.startswith("tile_") or self.name in {"block_collective", "partitioned"}

    @property
    def matching(self) -> bool:
        return self.name in {"partitioned", "tile_matching", "coalesced_matching"}

    @property
    def cpu(self) -> bool:
        return self.name in {"tile_indices", "block_indices", "grid_indices"}

    @property
    def blocks(self) -> int:
        return 1 if self.name == "copied" else 4 if "grid_barrier" in self.name else 3

    def capture(self, target: Target) -> Kernel:
        return capture(
            CG_FILE,
            self.name,
            target=target,
            consts={
                "t": {
                    "lanes": self.width,
                    "operation": self.operation,
                    "collective": self.collective,
                }
            },
            dtypes={"x": self.dtype, "out": self.dtype},
        )

    def arguments(self) -> list[Value]:
        count = self.threads * self.blocks
        x = (np.arange(count) % 13).astype(np_dtype(self.dtype))
        if self.dtype == I32:
            x -= 6
        if self.dtype == F32:
            x = np.random.default_rng(17).uniform(0.25, 1.75, count).astype(np.float32)
        shape = {
            "tile_indices": (count, 5),
            "tile_votes": (count, 3),
            "tile_shuffles": (count, 4),
            "coalesced": (count, 4),
            "block_indices": (count, 2),
            "coalesced_functions": (count, 10),
            "grid_indices": (count, 4),
        }.get(self.name, (count,))
        dtype = (
            np.uint64
            if self.name
            in {
                "tile_indices",
                "tile_votes",
                "tile_matching",
                "coalesced_functions",
                "coalesced_matching",
                "grid_indices",
            }
            else np_dtype(self.dtype)
        )
        out = np.zeros(shape, dtype)
        if self.name == "copied":
            x, out = np.arange(256, dtype=np.uint32), np.zeros(256, np.uint32)
        args: list[Value] = [out] if self.cpu else [x, out]
        return [{}, *args] if self.configured else args

    def expected(self, arguments: list[Value], warp_size: int) -> np.ndarray:
        out = arguments[-1]
        assert isinstance(out, np.ndarray)
        count = out.shape[0]
        rank = np.arange(count)
        if self.name == "tile_indices":
            base = rank % warp_size // self.width * self.width
            result = np.zeros_like(out)
            result[:, 0] = self.width
            result[:, 1] = rank % self.width
            result[:, 2] = rank % self.threads // self.width
            result[:, 3] = self.threads // self.width
            result[:, 4] = np.array([((1 << self.width) - 1) << int(b) for b in base], np.uint64)
            return result
        if self.name == "block_indices":
            return np.column_stack((rank * 0 + self.threads, rank % self.threads))
        if self.name == "grid_indices":
            return np.column_stack((rank // self.threads, rank * 0 + self.blocks, rank * 0 + count, rank * 0 + count))
        x = arguments[-2]
        assert isinstance(x, np.ndarray)
        if self.name == "copied":
            return x[::-1]
        if self.name == "grid_barrier":
            return ((rank + 64) % count).astype(np.uint32)
        if self.name == "repeated_grid_barrier":
            return (((rank + 64) % count) // 64 * 64 + 63 - rank % 64 + 2).astype(np.uint32)
        if self.name in {"tile_collective", "block_collective", "invoked"}:
            width = self.width if self.name.startswith("tile") else self.threads
            tiles = x.reshape(-1, width)
            op = (np.add, np.minimum, np.maximum, np.bitwise_and, np.bitwise_or, np.bitwise_xor)[self.operation]
            if self.collective == 0:
                totals = op.reduce(tiles, axis=1, dtype=x.dtype)
                if self.name == "invoked":
                    result = np.zeros_like(out)
                    result[::width] = totals
                    return result
                return np.repeat(totals, width)
            result = op.accumulate(tiles, axis=1, dtype=x.dtype)
            if self.collective == 2:
                result = np.column_stack((np.zeros(len(tiles), x.dtype), result[:, :-1]))
            return result.reshape(-1)
        if self.name == "tile_shuffles":
            base = rank // self.width * self.width
            lane = rank % self.width
            return np.column_stack(
                (
                    x[base + 1],
                    x[base + np.maximum(lane - 1, 0)],
                    x[base + np.minimum(lane + 1, self.width - 1)],
                    x[rank ^ 1],
                )
            )
        if self.name in {"tile_votes", "tile_matching"}:
            result = np.zeros_like(out)
            for start in range(0, len(x), self.width):
                tile = x[start : start + self.width]
                if self.name == "tile_votes":
                    ballot = sum(1 << i for i, value in enumerate(tile) if value > 0)
                    result[start : start + self.width] = (
                        ballot,
                        np.any(tile > 0),
                        np.all(tile > 0),
                    )
                else:
                    result[start : start + self.width] = [
                        sum(1 << j for j, other in enumerate(tile) if other == value) for value in tile
                    ]
            return result
        result = np.zeros_like(out)
        if self.name == "coalesced":
            result.fill(-1)
        for start in range(0, len(x), warp_size):
            values = x[start : start + warp_size]
            labels = values % 3 if self.operation == 0 else values > 0
            if self.name == "coalesced":
                positive = np.flatnonzero(values > 0)
                selected = values[positive]
                result[start + positive, 0] = np.sum(selected, dtype=x.dtype)
                result[start + positive, 1] = np.arange(len(positive))
                result[start + positive, 2] = len(positive)
                result[start + positive, 3] = np.cumsum(selected, dtype=x.dtype)
            elif self.name in {"coalesced_functions", "coalesced_matching"}:
                positive = np.flatnonzero(values > 0)
                selected = values[positive]
                ranks = np.arange(len(selected))
                if not len(selected):
                    continue
                if self.name == "coalesced_matching":
                    result[start + positive] = [
                        sum(1 << j for j, other in enumerate(selected) if other % 3 == value % 3) for value in selected
                    ]
                else:
                    ballot = sum(1 << j for j, value in enumerate(selected) if value % 3 == 0)
                    columns = (
                        np.full(len(selected), selected[1] if len(selected) > 1 else selected[0]),
                        selected[np.where(ranks >= 2, ranks - 2, ranks)],
                        selected[np.where(ranks + 2 < len(selected), ranks + 2, ranks)],
                        selected,
                        selected,
                        np.full(len(selected), ballot),
                        np.full(len(selected), np.any(selected % 3 == 0)),
                        np.full(len(selected), np.all(selected % 3 == 0)),
                        np.cumsum(selected, dtype=x.dtype) - selected,
                        np.full(len(selected), sum(1 << int(lane) for lane in positive), np.uint64),
                    )
                    result[start + positive] = np.column_stack([np.asarray(column, np.uint64) for column in columns])
            else:
                for label in np.unique(labels):
                    members = np.flatnonzero(labels == label)
                    result[start + members] = np.sum(values[members], dtype=x.dtype)
        return result

    def run(
        self,
        target: Target,
        backend: str,
        launcher: Callable[[Kernel], CUDARun | MetalRun] | None = None,
    ) -> None:
        ir = self.capture(target)
        args = self.arguments()
        expected = self.expected(args, target.warp_size)
        launch = Blocks(self.blocks, threads=self.threads)
        oracle = [fresh(arg) for arg in args]
        build(ir, target).run(launch, oracle)
        if expected.dtype.kind == "f":
            np.testing.assert_allclose(arrays(oracle)[-1], expected, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(oracle[-1], expected)
        if backend == "emulator":
            return
        actual = [fresh(arg) for arg in args]
        if backend == "c":
            CPUProgram(ir, emit(ir, target))(*actual, launch=launch)
        else:
            assert launcher is not None
            launcher(ir).run(launch, actual)
        agree(arrays(oracle), arrays(actual))
        if expected.dtype.kind == "f":
            np.testing.assert_allclose(arrays(actual)[-1], expected, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(actual[-1], expected)


TILE_OPERATIONS = [
    (0, F32),
    (0, I32),
    (1, F32),
    (1, I32),
    (2, F32),
    (2, I32),
    (3, U32),
    (4, U32),
    (5, U32),
]
SCAN_OPERATIONS = [(0, F32, 1), (0, I32, 1), (0, F32, 2), (0, I32, 2), (2, I32, 1)]
GROUP_CASES = [
    *[GroupCase("tile_collective", width, dtype, op) for width in (2, 4, 8, 16, 32) for op, dtype in TILE_OPERATIONS],
    *[
        GroupCase("tile_collective", width, dtype, op, coll)
        for width in (2, 4, 8, 16, 32)
        for op, dtype, coll in SCAN_OPERATIONS
    ],
    *[
        GroupCase(name, width)
        for width in (2, 4, 8, 16, 32)
        for name in ("tile_indices", "tile_votes", "tile_matching", "tile_shuffles")
    ],
    *[
        GroupCase("block_collective", dtype=dtype, operation=op, threads=threads)
        for threads in (32, 64, 128, 256)
        for op, dtype in ((0, F32), (0, I32), (1, I32), (2, I32))
    ],
    *[
        GroupCase("block_collective", dtype=dtype, collective=coll, threads=threads)
        for threads in (32, 64, 128, 256)
        for dtype in (I32, F32)
        for coll in (1, 2)
    ],
    *[GroupCase("block_indices", threads=threads) for threads in (32, 64, 128, 256)],
    GroupCase("coalesced", dtype=I32),
    GroupCase("partitioned"),
    GroupCase("partitioned", dtype=I32, operation=1),
    GroupCase("copied"),
    GroupCase("invoked"),
    GroupCase("grid_barrier"),
    GroupCase("repeated_grid_barrier"),
    GroupCase("coalesced_functions"),
    GroupCase("coalesced_matching"),
    GroupCase("grid_indices"),
]
WAVE64_CASES = [replace(case, width=64) for case in GROUP_CASES if case.name.startswith("tile_") and case.width == 32]


@pytest.mark.parametrize("case", GROUP_CASES)
def test_cg_emulator_against_numpy(case: GroupCase) -> None:
    case.run(GPU, "emulator")


@pytest.mark.parametrize("case", GROUP_CASES)
def test_cg_c_indices_and_collective_refusals(case: GroupCase) -> None:
    target = Target.of("avx512")
    if case.width > target.warp_size and case.name.startswith("tile_"):
        with pytest.raises(CaptureError, match="larger tile needs shared memory"):
            case.capture(target)
    elif case.cpu:
        case.run(target, "c")
    else:
        with pytest.raises(CaptureError) as error:
            case.capture(target)
        message = str(error.value)
        assert "one after the other" in message or "needs WARP_" in message or "needs a cooperative launch" in message
        assert "called from " in message
        assert str(CG_FILE) in message
        assert "targets that have it:" in message or "barrier would not wait" in message


@pytest.fixture(scope="module")
def cg_metal() -> MetalDevice:
    metal = pytest.importorskip("krn.runtime.metal")
    try:
        return metal.MetalDevice(Target.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("case", GROUP_CASES)
def test_cg_metal_against_numpy_and_emulator(cg_metal: MetalDevice, case: GroupCase) -> None:
    if case.matching or "grid_barrier" in case.name:
        with pytest.raises(CaptureError) as error:
            case.capture(cg_metal.target)
        message = str(error.value)
        assert "has no lowering for target apple_" in message
        assert "targets that have it:" in message and "sm_90a" in message and "gfx942" in message
        assert "called from " in message
    else:
        assert metal is not None
        program_type = MetalRun
        case.run(cg_metal.target, "metal", lambda ir: program_type(cg_metal, ir))


@pytest.fixture(scope="module")
def cg_card(gpu: CUDADevice) -> CUDADevice:
    return gpu


@pytest.mark.parametrize("case", GROUP_CASES)
def test_cg_cuda_against_numpy_and_emulator(cg_card: CUDADevice, case: GroupCase) -> None:
    case.run(cg_card.target, "cuda", lambda ir: CUDARun(cg_card, ir))


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a"])
@pytest.mark.parametrize("case", GROUP_CASES)
def test_cg_cuda_compiles(nvrtc: Nvrtc, arch: str, case: GroupCase) -> None:
    target = Target.of(arch)
    unit = emit(case.capture(target), target)
    assert nvrtc.compile(unit.source, arch, "cubin").data[:4] == b"\x7fELF"


@pytest.mark.skipif(not CLANG_HAS_AMDGPU, reason="this clang has no AMDGPU backend")
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("case", [*GROUP_CASES, *WAVE64_CASES])
def test_cg_hip_compiles(tmp_path: Path, arch: str, case: GroupCase) -> None:
    target = Target.of(arch)
    source = emit(case.capture(target), target, lineinfo=False).source
    file = tmp_path / "group.hip"
    file.write_text(source)
    done = subprocess.run(
        [
            CLANG[0],
            "-x",
            "hip",
            f"--offload-arch={arch}",
            "--offload-device-only",
            "-nogpulib",
            "-nogpuinc",
            "-I",
            str(Path(__file__).resolve().parents[1] / "hip_shim"),
            "-S",
            "-o",
            str(tmp_path / "group.s"),
            str(file),
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert descriptor((tmp_path / "group.s").read_text(), case.name)


@pytest.mark.parametrize("lanes", [0, 3, 33, 64, 128])
def test_invalid_cg_tiles_include_the_target_and_user_site(tmp_path: Path, lanes: int) -> None:
    file = tmp_path / "tile.py"
    file.write_text(
        "from krn import cg, kernel\n@kernel\ndef bad():\n"
        f"    tile = cg.tiled_partition(cg.this_thread_block(), {lanes})\n"
    )
    with pytest.raises(CaptureError) as error:
        capture(file, "bad", target=GPU)
    message = str(error.value)
    assert "larger tile needs shared memory" in message
    assert "target warp size 32" in message
    assert f"called from {file}:4:" in message


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("p = Pair(7, b=9)\n    out[0] = p.a + p.b", 16),
        ("p = Pair(a=7)\n    out[0] = p.a + p.b", 12),
        ("p = make(7)\n    out[0] = p.a + p.b", 12),
        ("out[0] = generic(out, i32(7))", 7),
        ("op = add\n    out[0] = apply(i32(7), op)", 14),
        ("out[0] = apply(i32(7), cg.plus)", 14),
        ("out[0] = i32(cg.this_thread_block().size())", 16),
    ],
)
def test_source_library_bindings(tmp_path: Path, body: str, expected: int) -> None:
    file = tmp_path / "bindings.py"
    file.write_text(
        "from collections.abc import Callable\n"
        "from krn import Vector, Scalar, cg, kernel, struct, shared_array\n"
        "from krn.frontend.types import i32\n"
        "@struct\nclass Pair:\n    a: i32\n    b: i32 = 5\n"
        "@kernel\ndef make(a: i32) -> Pair:\n    return Pair(a)\n"
        "@kernel\ndef generic[T: Scalar](x: Vector[T], a: T) -> T:\n"
        "    zero = T(0)\n    slots = shared_array(T, 1)\n    slots[0] = a\n    return slots[0] + zero\n"
        "@kernel\ndef add(a: i32, b: i32) -> i32:\n    return a + b\n"
        "@kernel\ndef apply[T: Scalar](a: T, fn: Callable[[T, T], T]) -> T:\n    return fn(a, a)\n"
        "@kernel\ndef run(out: Vector[i32]):\n    " + body + "\n"
    )
    program = build_kernel(file, "run")
    out = np.zeros(1, np.int32)
    program(out, launch=Blocks(1, threads=16))
    assert out.tolist() == [expected]


@pytest.mark.parametrize("expression", ["Pair()", "Pair(1, 2, 3)", "Pair(1, a=2)", "Pair(c=1)"])
def test_struct_constructor_reports_bad_bindings(tmp_path: Path, expression: str) -> None:
    file = tmp_path / "constructor.py"
    file.write_text(
        "from krn import kernel, struct\nfrom krn.frontend.types import i32\n"
        "@struct\nclass Pair:\n    a: i32\n    b: i32 = 0\n"
        f"@kernel\ndef bad():\n    p = {expression}\n"
    )
    with pytest.raises(CaptureError) as error:
        capture(file, "bad", target=GPU)
    assert error.value.span is not None and error.value.span.file == str(file)
    assert error.value.span is not None and error.value.span.line == 9


@pytest.mark.parametrize("annotation", ["T", "Vector[T]"])
def test_generic_device_type_conflicts_fail_at_the_call(tmp_path: Path, annotation: str) -> None:
    file = tmp_path / "generic.py"
    a, b = ("x[0]", "y[0]") if annotation == "T" else ("x", "y")
    file.write_text(
        "from krn import Vector, Scalar, kernel\nfrom krn.frontend.types import f32, i32\n"
        f"@kernel\ndef pair[T: Scalar](a: {annotation}, b: {annotation}):\n    pass\n"
        f"@kernel\ndef bad(x: Vector[f32], y: Vector[i32]):\n    pair({a}, {b})\n"
    )
    with pytest.raises(CaptureError, match="type parameter `T` is f32 and `b` is i32") as error:
        capture(file, "bad", target=GPU)
    assert error.value.span is not None and error.value.span.line == 8


def test_struct_collective_expression_is_materialized_once(tmp_path: Path) -> None:
    file = tmp_path / "once.py"
    file.write_text(
        "from krn import Vector, grid, kernel, struct, warp\n"
        "from krn.frontend.types import u64\n@struct\nclass Members:\n    bits: u64\n"
        "@kernel\ndef make() -> Members:\n    return Members(warp.active_mask())\n"
        "@kernel\ndef run(out: Vector[u64]):\n    p = make()\n"
        "    out[grid.thread_rank()] = p.bits + p.bits\n"
    )
    ir = capture(file, "run", target=GPU)
    program = build(ir, GPU)
    assert program.source.count("WarpRequest('active_mask'") == 1
    out = np.zeros(32, np.uint64)
    program.run(Blocks(1, threads=32), [out])
    np.testing.assert_array_equal(out, np.full(32, 2 * 0xFFFFFFFF, np.uint64))


def test_coalesced_if_and_else_have_distinct_active_sets(tmp_path: Path) -> None:
    file = tmp_path / "divergence.py"
    file.write_text(
        "from krn import Matrix, cg, grid, kernel\nfrom krn.frontend.types import u64\n"
        "@kernel\ndef run(out: Matrix[u64]):\n    i = grid.thread_rank()\n"
        "    if i % 3 == 0:\n        a = cg.coalesced_threads()\n        out[i, 0] = a.mask()\n"
        "    else:\n        b = cg.coalesced_threads()\n        out[i, 0] = b.mask()\n"
    )
    program = build(capture(file, "run", target=GPU), GPU)
    out = np.zeros((64, 1), np.uint64)
    program.run(Blocks(1, threads=64), [out])
    for i in range(64):
        start = i // 32 * 32
        assert out[i, 0] == sum(1 << lane for lane in range(32) if ((start + lane) % 3 == 0) == (i % 3 == 0))


def test_header_free_grid_and_cluster_helpers() -> None:
    for arch in ("sm_90a", "gfx942"):
        target = Target.of(arch)
        source = emit(capture(CG_FILE, "grid_barrier", target=target), target).source
        assert "cooperative_groups::" not in source
        assert source.count("__device__ unsigned long long _grid_arrived;") == 1
        assert "atomicAdd(&_grid_arrived, 1ull)" in source
        assert "(ticket / blocks + 1) * blocks" in source
    ir = capture(CG_FILE, "clustered", target=GPU)
    source = emit(ir, GPU).source
    assert "%%cluster_ctarank" in source and "%%cluster_nctarank" in source
    assert "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" in source
    with pytest.raises(NotImplementedError, match=r"one block at a time.*cluster launch"):
        build(ir, GPU)


def test_cg_cluster_compiles(nvrtc: Nvrtc) -> None:
    unit = emit(capture(CG_FILE, "clustered", target=GPU), GPU)
    assert nvrtc.compile(unit.source, "sm_90a", "cubin").data[:4] == b"\x7fELF"


@pytest.mark.parametrize("arch", ["avx512", "apple_m4", "gfx942"])
def test_cluster_refusals_name_feature_and_user_site(arch: str) -> None:
    with pytest.raises(CaptureError) as error:
        capture(CG_FILE, "clustered", target=Target.of(arch))
    assert "CLUSTERS" in str(error.value)
    assert "targets that have it:" in str(error.value)
    assert f"called from {CG_FILE}:" in str(error.value)


@pytest.mark.parametrize("case", [c for c in GROUP_CASES if c.matching or c.name == "grid_barrier"])
def test_apple_refusals_without_a_device(case: GroupCase) -> None:
    with pytest.raises(CaptureError) as error:
        case.capture(Target.of("apple_m4"))
    message = str(error.value)
    assert "has no lowering for target apple_m4" in message
    assert "sm_90a" in message and "gfx942" in message
    assert f"called from {CG_FILE}:" in message
    if case.name == "grid_barrier":
        assert "needs a cooperative launch" in message


@pytest.mark.parametrize("arch", ["sm_90a", "sm_120a", "gfx942", "gfx950", "apple_m4"])
@pytest.mark.parametrize(
    "case",
    GROUP_CASES,
    ids=lambda case: f"{case.name}-{case.width}-{case.dtype}-{case.operation}-{case.collective}-{case.threads}",
)
def test_cg_sources_render_without_a_device(arch: str, case: GroupCase) -> None:
    target = Target.of(arch)
    if target.vendor == "apple" and (case.matching or "grid_barrier" in case.name):
        with pytest.raises(CaptureError, match="has no lowering for target apple_m4"):
            case.capture(target)
    else:
        unit = emit(case.capture(target), target)
        assert unit.entry == case.name
        assert "cooperative_groups::" not in unit.source
        if target.vendor == "nvidia":
            assert "#include" not in unit.source


@pytest.mark.parametrize("case", WAVE64_CASES)
def test_64_lane_tiles_capture_and_render(case: GroupCase) -> None:
    target = Target.of("gfx942")
    unit = emit(case.capture(target), target)
    assert unit.entry == case.name
    if case.name == "tile_indices":
        assert "18446744073709551615ull" in unit.source


def test_runtime_tile_size_diagnostic_explains_the_limit(tmp_path: Path) -> None:
    file = tmp_path / "runtime_size.py"
    file.write_text(
        "from krn import cg, kernel\nfrom krn.frontend.types import u32\n"
        "@kernel\ndef bad(lanes: u32):\n"
        "    tile = cg.tiled_partition(cg.this_thread_block(), lanes)\n"
    )
    with pytest.raises(CaptureError) as error:
        capture(file, "bad", target=GPU)
    message = str(error.value)
    assert "larger tile needs shared memory" in message and "target warp size 32" in message
    assert f"called from {file}:5:" in message


@pytest.mark.parametrize("name", ["block_collective", "grid_barrier", "copied"])
def test_cg_c_block_and_grid_barrier_diagnostic(name: str) -> None:
    case = GroupCase(name, dtype=I32 if name == "block_collective" else U32)
    with pytest.raises(CaptureError, match="this barrier would not wait") as error:
        case.capture(Target.of("avx512"))
    assert "called from " in str(error.value)


@pytest.mark.parametrize("case", [*WAVE64_CASES, GroupCase("coalesced_functions"), GroupCase("coalesced_matching")])
def test_wave64_emulation_preserves_all_mask_bits(case: GroupCase) -> None:
    case.run(Target.of("gfx942"), "emulator")


@pytest.mark.parametrize("arch", ["sm_90a", "gfx942"])
def test_coalesced_singletons_empty_warps_and_high_mask_bits(arch: str) -> None:
    target = Target.of(arch)
    case = GroupCase("coalesced_functions")
    x = np.zeros(192, np.uint32)
    x[[17, 63, 137]] = [7, 6, 9]
    out = np.zeros((192, 10), np.uint64)
    expected = case.expected([x, out], target.warp_size)
    build(case.capture(target), target).run(Blocks(3, threads=64), [x, out])
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("placement", ["factory", "branch"])
def test_nested_coalescing_uses_the_whole_inline_call_site(tmp_path: Path, placement: str) -> None:
    file = tmp_path / "nested.py"
    make = "@kernel\ndef make() -> cg.Coalesced:\n    return cg.coalesced_threads()\n"
    factory = "make()" if placement == "factory" else "cg.coalesced_threads()"
    body = (
        "    i = grid.thread_rank()\n    if i % 3 == 0:\n"
        f"        a = {factory}\n        out[i] = a.mask()\n"
        f"    else:\n        b = {factory}\n        out[i] = b.mask()\n"
    )
    helper = "@kernel\ndef write(out: Vector[u64]):\n" + body if placement == "branch" else ""
    kernel_body = "    write(out)\n" if placement == "branch" else body
    file.write_text(
        "from krn import Vector, cg, grid, kernel\nfrom krn.frontend.types import u64\n"
        + make
        + helper
        + "@kernel\ndef run(out: Vector[u64]):\n"
        + kernel_body
    )
    out = np.zeros(64, np.uint64)
    build(capture(file, "run", target=GPU), GPU).run(Blocks(1, threads=64), [out])
    for i in range(64):
        start = i // 32 * 32
        assert out[i] == sum(1 << lane for lane in range(32) if ((start + lane) % 3 == 0) == (i % 3 == 0))


def test_struct_constructor_checks_runtime_field_types(tmp_path: Path) -> None:
    file = tmp_path / "field_type.py"
    file.write_text(
        "from krn import kernel, struct\nfrom krn.frontend.types import i32, f32\n"
        "@struct\nclass Value:\n    number: i32\n@kernel\ndef bad():\n    v = Value(f32(1))\n"
    )
    with pytest.raises(CaptureError, match="field `number`") as error:
        capture(file, "bad", target=GPU)
    assert error.value.span is not None and error.value.span.file == str(file)
    assert error.value.span is not None and error.value.span.line == 8


def test_generic_device_functions_can_return_a_dtype_binding(tmp_path: Path) -> None:
    file = tmp_path / "dtype.py"
    file.write_text(
        "from krn import Vector, Scalar, kernel\nfrom krn.frontend.types import i32\n"
        "@kernel\ndef dtype[T: Scalar](x: Vector[T]) -> type[T]:\n    return T\n"
        "@kernel\ndef run(out: Vector[i32]):\n    dt = dtype(out)\n    out[0] = dt(7)\n"
    )
    out = np.zeros(1, np.int32)
    build_kernel(file, "run")(out, launch=Blocks(1, threads=1))
    np.testing.assert_array_equal(out, [7])


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "apple_m4"])
@pytest.mark.parametrize("name", ["warp.ballot", "warp.any", "warp.all", "warp.eq"])
def test_masked_vote_spellings_keep_only_the_calling_group(arch: str, name: str) -> None:
    spelling = catalog.spell(name, Target.of(arch), "predicate", "members")
    assert "& (members)" in spelling
    if name == "warp.all":
        assert "== ((members) &" in spelling
    elif name == "warp.any":
        assert "!= 0" in spelling
    elif name == "warp.eq":
        assert "== 0" in spelling and "== ((members) &" in spelling

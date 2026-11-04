---
type: Reference
title: "krn examples"
description: "Runnable examples of krn in five folders by complexity, each with a title, a short teaching text, a level, topics and prerequisites that the release spec and the website guides read."
generated: { by: agent:claude, at: 2026-11-03T16:00:00+00:00 }
---
# examples

Start at the top and stop where you have what you need. The folders are the levels and the files are numbered inside a
folder in order of difficulty, so each file needs only the files before it. Every example is a runnable script, and the
test suite runs all of them on the CPU backend and, where the example launches kernels, on this Mac's GPU (and on a CUDA
card when there is one). They import only `krn` and the public modules listed in `tests/test_public_surface.py`
(`krn.errors`, `krn.emulate`, `krn.compiler.api`, `krn.frontend.atomic`, `krn.data.sparse`, `krn.data.sparse_build`
and the documented library modules), never `krn._internals` and never another example.

Run one from the repository root. Most examples read their results back through numpy views, which the CPU backend
gives, so they select it with `KRN_TARGET` (the test suite sets `avx512`, and an AVX2 machine uses `avx2`):

```bash
KRN_TARGET=avx512 mb run python examples/01_first_kernels/01_saxpy.py
KRN_TARGET=avx512 mb run python examples/03_memory_and_sync/05_warp_shuffles.py
```

## How a file is written

The docstring is a teaching text: a title on the first line, three to eight lines that explain the idea, and a block of
metadata at the end in exactly this form, which `scripts/build_spec.py` parses and the website reads.

```text
Level: 2
Topics: tiles, strict arithmetic
Needs: 01_first_kernels/01_saxpy
```

`Level` is the number of the folder, `Topics` are comma separated, and `Needs` lists example ids (the path without
`.py`) or `none`. The build stops when a block is missing, a level disagrees with its folder, or a need is not an
example.

## Conventions the examples follow

- **One decorator.** `@kernel` marks device code. Capture classifies entry points, callees, configurations and constant
  folding by use. A guarded helper runs only when its guard is true.
- **Config in brackets, data in parentheses.** `gemm(a, b, c)` uses the configuration defaults and infers geometry from
  the body. `gemm[Tiling(bm=64, bn=128)](a, b, c)` chooses a configuration; `launch=` overrides geometry.
- **Groups, not a flat root.** `from krn import T, host, targets, formats, linalg, catalog, debug` and the kernel
  language at the root. `T.f16`, `T.fp.min`, `T.acc(T.f16)` are types and selectors; `host.Device`, `host.Settings`
  the Python around the kernels; `targets.Nvidia`, `targets.Target.of("sm_100a")` the hardware tables.
- **Python-style names for asynchronous variants.** A name with a synchronous twin gets the async one with an `a`
  prefix, as `aiter` and `anext` do. A name with no synchronous twin keeps the plain name.
- **Say what the compiler can decide as `AUTO`.** `pipelined(K, bk)` yields K tiles as slices and picks the stage count
  and the schedule from the target; a number or a schedule you pass is a request that the compiler checks.
- **Tiles have torch-like methods.** `x.exp()`, `x.exp_()` in place on a named tile, `.softmax()`, `.sum(keepdims=True)`.
- **Python's statements keep their meaning in a kernel.** `breakpoint()`, `print(...)` and `assert` are the device
  versions (see `03_memory_and_sync/03_print_assert_bounds` and `05_systems/02_breakpoints`).

## Level 1, first kernels

Write a kernel, run it, read the machine and the errors.

| File | What it teaches | Topics | Needs |
| --- | --- | --- | --- |
| [`01_first_kernels/01_saxpy.py`](01_first_kernels/01_saxpy.py) | SAXPY in the SIMT style, and what strict arithmetic asks of the code you write | SIMT, strict arithmetic, launch geometry | none |
| [`01_first_kernels/02_gelu.py`](01_first_kernels/02_gelu.py) | GELU with a bias in the tile style: a slice read is a load, a slice write a store | tiles, tile methods, configuration, invalid lanes | `01_saxpy` |
| [`01_first_kernels/03_settings.py`](01_first_kernels/03_settings.py) | Settings and targets: ask what the machine is, then change one setting for one block | settings, targets, precision ladder | none |
| [`01_first_kernels/04_compile_errors.py`](01_first_kernels/04_compile_errors.py) | Compile-time errors that name the missing feature, the line and the targets that would work | diagnostics, TargetError, capture, targets | `01_saxpy` |

## Level 2, tiles and types

The tile style, number types, and the Python a kernel can use.

| File | What it teaches | Topics | Needs |
| --- | --- | --- | --- |
| [`02_tiles_and_types/01_softmax_rows.py`](02_tiles_and_types/01_softmax_rows.py) | Softmax for rows longer than a tile, with no padding value and no mask | tiles, reductions, invalid lanes, chunked loops | `02_gelu` |
| [`02_tiles_and_types/02_tile_methods.py`](02_tiles_and_types/02_tile_methods.py) | Torch-style tile methods: chain them, and update in place with a trailing underscore | tile methods, reductions, in-place updates | `02_gelu` |
| [`02_tiles_and_types/03_bit_tricks.py`](02_tiles_and_types/03_bit_tricks.py) | Integer bit tricks and scalar math helpers: popcount, reversed bits, fma, cbrt | integer types, bit intrinsics, math helpers | `01_saxpy` |
| [`02_tiles_and_types/04_python_in_kernels.py`](02_tiles_and_types/04_python_in_kernels.py) | Python's own vocabulary inside a kernel: folds, enumerate, zip, register sequences and match | Python vocabulary, generators, register sequences, match | `01_saxpy` |
| [`02_tiles_and_types/05_helper_functions.py`](02_tiles_and_types/05_helper_functions.py) | Helper functions in kernels: itertools generators that cost what the hand-written loop costs | helper functions, itertools, generators | `04_python_in_kernels` |

## Level 3, memory and synchronization

Atomics, shared memory, barriers and warps, with the emulator to see them.

| File | What it teaches | Topics | Needs |
| --- | --- | --- | --- |
| [`03_memory_and_sync/01_histogram_atomics.py`](03_memory_and_sync/01_histogram_atomics.py) | Atomic updates: a histogram, a running maximum, a wrapping ticket counter and a lock | atomics, CPU backend, atomic plan | `01_saxpy` |
| [`03_memory_and_sync/02_block_reduction.py`](03_memory_and_sync/02_block_reduction.py) | A block-wide sum through shared memory and barriers, run thread by thread in the emulator | shared memory, barriers, block votes, emulator | `01_saxpy` |
| [`03_memory_and_sync/03_print_assert_bounds.py`](03_memory_and_sync/03_print_assert_bounds.py) | print, assert and out-of-bounds checks in a kernel | print, assert, bounds checks, emulator | `02_block_reduction` |
| [`03_memory_and_sync/04_deadlock.py`](03_memory_and_sync/04_deadlock.py) | Two paths of a block at two different barriers: the emulator reports the deadlock | barriers, deadlock, emulator | `02_block_reduction` |
| [`03_memory_and_sync/05_warp_shuffles.py`](03_memory_and_sync/05_warp_shuffles.py) | Warp collectives: a butterfly sum, an inclusive scan, a ballot and a vote | warps, shuffles, ballot, votes, emulator | `02_block_reduction` |
| [`03_memory_and_sync/06_cooperative_groups.py`](03_memory_and_sync/06_cooperative_groups.py) | Cooperative groups: tile sums and a block scan with the collectives of a group | cooperative groups, reductions, scans, emulator | `05_warp_shuffles` |
| [`03_memory_and_sync/07_intrinsic_catalog.py`](03_memory_and_sync/07_intrinsic_catalog.py) | Every device intrinsic with its spelling on NVIDIA, AMD, Apple and the CPU | intrinsics, catalog, vendors | `05_warp_shuffles` |

## Level 4, libraries and performance

GEMM, dispatch by target and precision, vendor libraries, sparse matrices, a real application.

| File | What it teaches | Topics | Needs |
| --- | --- | --- | --- |
| [`04_libraries/01_gemm_tiles.py`](04_libraries/01_gemm_tiles.py) | GEMM from the algorithm alone: Tilus's minimal Blackwell matmul, written once for every target | GEMM, tiles, pipelined loops, tensor cores | `02_gelu` |
| [`04_libraries/02_gemm_selectors.py`](04_libraries/02_gemm_selectors.py) | A GEMM written once as a template: configuration in brackets, data in parentheses, float type by the target | templates, configuration, float type selectors, targets | `01_gemm_tiles` |
| [`04_libraries/03_gemm_variants.py`](04_libraries/03_gemm_variants.py) | One GEMM, five implementations, picked by what the machine can do | families, dispatch by target, capabilities, audit | `02_gemm_selectors` |
| [`04_libraries/04_precision_ladder.py`](04_libraries/04_precision_ladder.py) | One linear layer that stores its weights as FP4 where the machine has FP4 and widens gracefully elsewhere | precision ladders, FP4, FP8, policies, formats | `02_gemm_selectors` |
| [`04_libraries/05_vendor_matmul.py`](04_libraries/05_vendor_matmul.py) | One call that reaches cuBLASLt, hipBLASLt, Accelerate or oneDNN when they are better, and our kernels when not | vendor libraries, dispatch, lib.matmul, measured selection | `01_gemm_tiles` |
| [`04_libraries/06_sparse.py`](04_libraries/06_sparse.py) | Sparse vectors and matrices as kernels of a few lines: CSR, Ellpack, blocked CSR, COO and 2:4 sparsity | sparse formats, SpMV, atomics, warp reductions | `01_histogram_atomics`, `05_warp_shuffles` |
| [`04_libraries/07_cutok_tiles.py`](04_libraries/07_cutok_tiles.py) | cutok's Tiles rebuilt: spans cut into tiles, a warp to a span and a lane to a tile | structs of arrays, warp per item, real application | `05_warp_shuffles` |
| [`04_libraries/08_flash_attention.py`](04_libraries/08_flash_attention.py) | Flash attention 1 forward: the online softmax over the key tiles of every head, causal masks and ragged sequences in one kernel of about twenty lines | attention, online softmax, batched heads, causal masks, ragged tiles | `01_gemm_tiles` |

## Level 5, systems

Debugging for people and agents, compile flags, streams, graphs and launch cost.

| File | What it teaches | Topics | Needs |
| --- | --- | --- | --- |
| [`05_systems/01_emulator_vs_backend.py`](05_systems/01_emulator_vs_backend.py) | The emulator against the backend: a differential test in ten lines | emulator, differential testing, launch inference | `02_block_reduction` |
| [`05_systems/02_breakpoints.py`](05_systems/02_breakpoints.py) | breakpoint() inside a kernel: stop, look at the locals of one thread, step through the kernel's own lines | breakpoints, pdb, emulator | `01_emulator_vs_backend` |
| [`05_systems/03_agent_breakpoints.py`](05_systems/03_agent_breakpoints.py) | Breakpoints an agent can use: a hook that writes the locals to a file and carries on | breakpoints, agents, PYTHONBREAKPOINT, recorded locals | `02_breakpoints` |
| [`05_systems/04_line_info_artifacts.py`](05_systems/04_line_info_artifacts.py) | Line information and saved artifacts: generated code that points back at the Python line that wrote it | line info, artifacts, compile flags, inspection | `01_saxpy` |
| [`05_systems/05_native_debuggers.py`](05_systems/05_native_debuggers.py) | Command lines for the native tools: cuda-gdb, compute-sanitizer, rocgdb and the Metal validation layers | native debuggers, sanitizers, cuda-gdb, rocgdb | `04_line_info_artifacts` |
| [`05_systems/06_compile_flags.py`](05_systems/06_compile_flags.py) | Compiler flags per backend: the C++ standard each toolchain accepts, and the options that refuse a target | compile options, flags, settings, TargetError | `03_settings` |
| [`05_systems/07_cpu_backend.py`](05_systems/07_cpu_backend.py) | Run for real: eight batches of GEMM then SAXPY on the CPU backend, eager, as async tasks and as graph replays | CPU backend, streams, tasks, graphs, free-threaded Python | `01_gemm_tiles` |
| [`05_systems/08_async_streams.py`](05_systems/08_async_streams.py) | Overlapping device work and Python work with async/await, for the free-threaded build of Python | async launches, streams, TaskGroup, free-threaded Python | `07_cpu_backend` |
| [`05_systems/09_graphs.py`](05_systems/09_graphs.py) | Record a sequence of launches once and replay it with one call: a graph on CUDA, HIP, Metal and CPUs | graphs, replay, typed recording, CPU task DAG | `07_cpu_backend` |
| [`05_systems/10_fast_launch.py`](05_systems/10_fast_launch.py) | Keeping the Python in the loop out of the way: the slow, fast and bound launch paths | launch overhead, bind, rebind, specialization cache | `07_cpu_backend` |
| [`05_systems/11_attach_to_process.py`](05_systems/11_attach_to_process.py) | Attach to a process that is already running, ask it what it is doing, and dump its stacks when it is stuck | attach, snapshot, stuck processes, remote debugging | `07_cpu_backend` |

## Not here

Four kinds of file left this folder in the regrouping of 2026-11-03, because they do not run or are not teaching
examples. The [public API roadmap](../ROADMAP.html#language) tracks their remaining blockers.

- `experiments/sketch/` holds the typed design sketches: API shapes the type checkers verify and capture cannot lower
  yet (`@intrinsic` with a lowering per vendor, `async def` kernels, abstract epilogues, `linalg.cholesky` with a
  backward, the serving engine contracts, a warp `argmax`), and the small kernels the test suite and the benchmarks
  capture. Move a sketch back into a level folder when it runs.
- `experiments/library_checks/` holds the checks of the data and helper libraries (UTF-8 strings, nullable columns, the
  `krn.itertools` helpers against their originals). They build programs through compiler internals, so they wait for a
  default launcher in `krn.data`. `tests/examples` still runs them.

## Check

```bash
mb run lint                                  # ruff, ty and pyrefly over src, tests, examples and experiments
mb run pytest -q tests/examples tests/test_public_surface.py
```

Only Python 3.14 is supported. A kernel body is captured, not executed.

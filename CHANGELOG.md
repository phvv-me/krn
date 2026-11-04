# Changelog

- Add the tile operations attention needs, on the emulator, the C backend and the Apple GPU: `t.transpose()` (the
  compiler reads the swapped element, and the validity of a ragged edge swaps with it), the comparisons `<`, `<=`, `>`,
  `>=` on tiles giving bool tiles, `where(condition, x, y)` as a broadcast select, `arange(extent, axis, rank=, start=)`
  for an int32 tile of positions, the bare product `a @ b` into a fresh float32 tile (`acc += a @ b` is unchanged), and
  `t.shape[axis]`, the static extent of a tile. `where` and `arange` are public in `krn`. Refused: `&` and `|` on bool
  tiles, `acc.mma(a, b)` (its stub is not lowered), `argmax`, `cumsum`, `sort`, `gather`, tile slicing and reshape.
- Tensors take a scalar index mixed with slices: `x[h, r0:r1, c0:c1]` is a rank-2 tile, the scalar axis is an axis of
  the tensor and not of the tile, and a store is symmetric. The validity of the tile is the tensor's own edge, so a
  batch of sequences of different lengths in one tensor needs no padding. Before, mixing a scalar and a slice was
  refused.
- Add `examples/04_libraries/08_flash_attention.py`, flash attention 1 forward: one `@kernel` over the query tiles and
  the heads, the running maximum, sum and output kept in registers while the key and value tiles stream past, causal
  masks by `where`, and ragged sequences with no mask at all. The kernel is 17 lines of body. It is checked against numpy
  on the CPU backend and through the Metal mirror, and in `tests/frontend/test_tile_ops.py` on the emulator and the C
  backend over ragged sequences.
- Add `krn.profile.providers.apple`: `Capabilities.probe` reads the Metal counter API of one device, `StageTimer` times each
  encoder from the timestamp counter at its stage boundaries (no measurable GPU overhead), and `signals.require` raises an
  explained `TargetError` for dispatch-boundary sampling, ALU/memory/occupancy counters and `.gputrace` capture, which the
  M4 Pro does not provide to this process. `experiments/zero_cost_apple.py` compares krn's generated Metal with hand-written
  Metal and MLX on the Mac. Per-launch counters in `Profiler` records remain unavailable on Apple.
- Add `krn.profile.providers.cpu.perf.HardwareCounters`, per-thread perf_event hardware counters on Linux (cycles,
  instructions, cache and branch events, summed over hybrid core PMUs) with derived `ipc`, `cache_miss_rate` and
  `branch_miss_rate`. The struct layout is spelled and tested; refusals name `kernel.perf_event_paranoid` or
  `CAP_PERFMON`, and macOS raises `TargetError` rather than reporting zeros. `RegionSummary` gains an empty-by-default
  `counters` field. Per-launch counters in `Profiler` records wait for the runtime hook.
- Add `krn.profile.providers.intel.igpu`: device-level health of the Intel integrated GPU from i915 sysfs (awake share
  from rc6 residency, actual and requested frequency), plus the i915 PMU opener that refuses with the capability it needs.
  Kernel-level Intel tracing waits for the Intel backend.
- Add `experiments/zero_cost_cpu.py`, which compares krn's generated C with hand-written C and BLAS on one thread through
  the same counters.
- Fix a CUDA program compiled or launched from a thread other than the one that first used the device, such as
  `asyncio.to_thread(launcher.bind, ...)` or a thread pool. The device's context was bound only to that first thread, so
  loading the program failed with `CUDA_ERROR_INVALID_CONTEXT`. Each thread now binds the context once, before its first
  driver call, and every driver call of the backend does so: loads, launches, graph capture and replay, copies and
  stream or event creation. Launch semantics are unchanged.
- Fix `mean` over a ragged tile. It divided the sum of the valid lanes by the tile width, so 1000 ones in a 1024-wide
  tile gave 0.9765625; it now divides by the number of valid lanes and gives 1.0 on the emulator, the C backend and the
  Apple GPU. The tile method `mean` is `Reduction.MEAN` in the IR; the scalar `krn.math.mean(x, start, stop)` is
  unchanged. A lane with no valid operand lane is NaN, as numpy's mean of nothing is, and a mean is valid on
  every lane, so that NaN is stored: an element past the edge of its operand (a row past the tensor, or every column
  past the edge) gives NaN in the output where that output exists. Full tiles keep their bits. The denormal-flush
  warning on Apple still names the mean's division.
- Regroup the examples by complexity into `examples/01_first_kernels`, `02_tiles_and_types`, `03_memory_and_sync`,
  `04_libraries` and `05_systems`, with short names (`01_saxpy`, `02_gelu`, `01_softmax_rows`, `01_histogram_atomics`
  ...) numbered in order of difficulty. Each docstring is a title, three to eight lines of teaching text and a block
  `Level:`, `Topics:`, `Needs:` that the release spec reads; `examples/README.md` is the table of contents. Examples that
  did not run moved out: the typed design sketches to `experiments/sketch` (with the old `examples/sketch` kernels that
  the tests capture) and the data and helper library checks to `experiments/library_checks`. The first examples now
  launch kernels and check them against numpy (`01_saxpy`, `02_gelu`, `01_softmax_rows`, `02_tile_methods`,
  `05_helper_functions`, `08_async_streams`, `09_graphs`, `10_fast_launch`).
- Add the public modules `krn.errors` (`CaptureError`, `TargetError`, `KernelError`, `SettingError`, `PrecisionWarning`,
  `PerformanceWarning`) and `krn.emulate` (`emulate`, `Deadlock`), and export `inspect_kernel` from `krn.compiler.api`.
  Examples import only the modules listed in `tests/test_public_surface.py`; the release spec documents the two new
  modules and gives each example an `id`, `level`, `topics`, `needs` and `description`.
- Rename the kernel-side conversion method `.astype(dtype)` to `.to(dtype)`, as in torch, on scalars, tiles, tensors and
  vectors inside kernels. The meaning is unchanged: round to nearest even, and saturate for the narrow floats. There is
  no alias: `x.astype(T.f16)` is now a kernel error, and host-side numpy `.astype` is untouched. `.to` takes one dtype;
  any other argument (a device, `copy`, `non_blocking`) is refused at capture with "krn's `.to` takes a dtype; devices
  are chosen with the launch". The narrow-float and product refusals name `.to` in their alternatives.
- Add the compatibility matrix's requirements, measured runs and docstring blocks.
  Each native cell names what the machine needs, from the vendor pages it cites: for NVIDIA the PTX ISA
  version that added the target, the CUDA toolkit that shipped it and that toolkit's minimum driver; for Apple the GPU
  family; for AMD the gfx architecture; for x86 the instruction-set extensions. A requirement the documents do not state
  is left empty. Hardware-gated tests record each run that matched the emulator with `--compat-out DIR` (with
  `KRN_HARDWARE=1`), and a run that did not match fails the session; the Apple M4 Pro runs are recorded, the NVIDIA
  runs on silo and crimson are not yet. The packed words `i8x4`, `u8x4`, `i16x2` and `u16x2` have rows for conversion,
  load and store, addition and subtraction, and the lane functions. The block-scaled formats and tf32 have no rows,
  since the compiler does not capture `QMatrix` operands and krn has no tf32 type. `experiments/docstring_support.py`
  writes a "Supported on" block at the end of each public function of `krn.math`, `krn.fast.math` and
  `krn.intrinsics`, and the test suite checks it against the matrix. The compatibility page marks measured cells, shows
  each cell's requirement in its tooltip, and details a selected feature with its requirement and evidence per target.
- A float32 `atomic.add` into global memory on sm_90a and sm_120a is one `atom.global.add.noftz.f32` instead of a
  compare-and-swap loop. Measured on an RTX 5080 (driver 615.78.08, NVRTC 13.4) and a GH200 (driver 615.71.09), the
  noftz form keeps every subnormal bit in 22 of 22 runs, where the plain `atomicAdd` flushes them, and costs about
  1.4 ns (RTX 5080) and 1.9 ns (GH200) per atomic under contention against about 990 ns and 1537 ns for the loop.
  Float32 `atomic.sub`, shared memory, block and system scope and every other architecture keep the loop (sm_80 and
  sm_86 reject the form). The probe is `experiments/noftz_probe.py`; these are historical measurements,
  not a fresh validation of the current source.
- Refuse what the narrow floats and FNUZ fp8 cannot do, and make `f64` on Apple a target refusal. Arithmetic,
  comparisons and math on `f8e4m3fn`, `f8e5m2`, `f6e2m3`, `f6e3m2` and `f4e2m1` raise a `TargetError` at the operation
  on every target, since they would add byte codes, not values; convert to `f16` or `f32` first. A plain `@` on fp6 or
  fp4 raises a `TargetError` that names the block-scaled `QMatrix` on sm_120a. `f8e4m3fnuz` and `f8e5m2fnuz` are dtypes
  of their own, and every conversion or product of them is refused with an explained `TargetError`. An `f64` value on
  an Apple GPU fails at capture with a `TargetError` instead of a `NotImplementedError` at emit. The coverage omit
  lists the API-sketch modules `krn/intrinsics/__init__.py` and `krn/math/__init__.py`, so the gate is 100 percent.
- Fix hardware detection on real machines: NVML devices map to CUDA ordinals by PCI bus id so a `CUDA_VISIBLE_DEVICES`
  mask no longer crashes the probe, display connector entries no longer break the DRM probe (which hid the Intel iGPU
  on Linux), OpenVINO and WebGPU adapters are attributed to the right vendor and merged with the probed device, software
  renderers are a separate `DeviceKind.SOFTWARE`, and a missing Level Zero or OpenCL library is reported for each Intel
  GPU. The `gpu-throttling` insight fires only for a thermal or power-brake reason.
- Run the narrow-float conversions in the Python emulator, bit for bit with the compiled code. fp8 conversions on gfx942
  are refused with an explained `TargetError` because gfx942 stores FNUZ fp8, which krn does not model yet. Narrow
  float dtypes print as `f8e4m3fn`, `f8e5m2`, `f6e2m3`, `f6e3m2` and `f4e2m1`.
- Add `krn.profile`: frozen results with schema versioning (mainboard-era files load), runtime launch timing on CPU,
  Metal, and CUDA events, optional CUPTI/NVTX/signpost tracing, per-dtype roofline verdicts, bounded retention with
  explicit capture-loss errors, and explained unsupported-feature errors.
- Convert the narrow floats `f8e4m3fn`, `f8e5m2`, `f6e2m3`, `f6e3m2` and `f4e2m1` to and from `f32` on every target:
  the NVIDIA `cvt` instructions where the card has them (fp8 from sm_89; fp4 and fp6 on sm_100a, sm_101a and sm_120a)
  and a composed integer sequence everywhere else, rounding to nearest even with the PTX `.satfinite` rule. A composed
  conversion on a GPU warns with the `software-fallback` kind (`warn.warn_software_fallback`,
  `warn.software_fallback_min_instructions`); a narrow float with no composition is refused with an explained
  `TargetError`. A narrow value is one byte of storage. Narrow matrix products are unchanged and still refused where
  they have no unit. Further storage and product work is tracked in [the roadmap](ROADMAP.html#quantization).
- Add the general functions `krn.math.sub_sat` and `avg` on packed words and plain integers, the lane tests
  `all(a OP b)` and `any(a OP b)` over packed comparisons, and the raw NVIDIA `vsads2`, `vsads4`, `vsadu2` and
  `vsadu4` sums of absolute differences and the 36 `__vi*` video SIMD instructions in `krn.intrinsics`
  (`viaddmax`, `viaddmin`, `vimax3`, `vimin3`, the `_relu` forms, and the predicated `vibmax` and `vibmin` returning
  their value and predicates as a tuple). `avg` rounds half up without overflow, and its signed packed form is a
  composed lane loop on NVIDIA. On Apple, packed `add_sat`, `maximum` and `abs_diff` are native Metal vector calls
  (`addsat`, `max`, `absdiff`) and no longer warn as `software-fallback`.
- Name every device log record after its logger, `krn.<logger>` (`log.get("saxpy")` delivers on `krn.saxpy`, the kernel
  is an extra), so the client's own `logging` configuration controls both the compile-time level and delivery. A record
  that cannot be rendered becomes an ERROR record in its place with the raw argument words instead of failing the whole
  drain; the ring overflow warning goes to the fixed logger `krn`.
- Add packed element types `i8x4`, `u8x4`, `i16x2` and `u16x2` (one 32-bit word of lanes, wrapping `+ -`, lane masks from
  comparisons, `pack`, `unpack` and `.lanes`), the general functions `krn.math.add_sat`, `maximum` (two or three
  operands) and `abs_diff` resolved by operand type and target through `krn.dispatch`, `krn.intrinsics` for the raw
  NVIDIA SIMD instructions, and the `software-fallback` warning with `krn.Warn.warn_software_fallback` and
  `software_fallback_min_instructions`.
- Add `krn.log` host side (messages from f-strings, t-strings and `{}`, sites, record layout, the ring and asynchronous
  drain into the client's own `logging` handlers, the compile-time level from the client's logger) and `krn.Log`
  settings; add `krn.exceptions` host side (any `BaseException` subclass in a class table with subclass sets, a
  device-safe check, error records, and rebuilding the same class on the host or an `ExceptionGroup`) and `krn.Errors`
  settings. Kernel capture and the device buffers come next.
- Enforce the strict numerical flags at the NVRTC boundary and pass the resolved options to inspection builds; a caller
  option that relaxes them is refused with an explained error.
- Fire the matmul reduced-accumulation `PrecisionWarning` once per call site, and refuse the unverified AMD lowerings of
  `krn.fast.math` with an explained `TargetError`.
- Move the precision-losing functions to `krn.fast.math`, and give each one a strict `krn.math` counterpart with the
  same name and signature. `krn.fast` holds only operations that raise a `PrecisionWarning`. `fma` leaves the fast
  namespace: `krn.math.fma` is strict, rounds once and is exact on every backend. `saturate` is gone; its strict
  meaning is `krn.math.clip(x, 0.0, 1.0)`. Names follow torch: `recip` is `reciprocal`, `clamp` is `clip` (also the
  tile methods `x.clip(low, high)` and `clip_`). New strict `krn.math` functions: `rsqrt`, `reciprocal`, `div`, `exp10`,
  `tan`, `pow`. Strict `rsqrt` is `1 / sqrt(x)` as two correctly rounded operations, so CUDA no longer uses `rsqrtf`
  for it. No aliases remain for the old names.
- Make arithmetic strict on every backend and replace global math modes with the real
  `krn.fast.math` subpackage. Fast calls have target-specific contracts, source-located
  `PrecisionWarning`, and compiler decisions; explicit FMA rounds once in the emulator.
- Move library matrix precision to `matmul(..., precision=...)`, use pedantic cuBLAS
  float32 by default, remove the unused reduced-accumulation boolean, and enforce ladder floors.
- Refuse scalar atomics embedded in tile broadcasts; bind the result to a local first.
- Define integer zero divisors and invalid shift counts to return zero across backends;
  spell sparse matrix multiplication's fused update explicitly.

## Settings migration — 2026-11-02

- Added immutable grouped `krn.config` / `krn.configure` snapshots with thread/task scopes and program-cache isolation.
- Debug seeds now use `KRN_DEBUG_*`; only target, cache, workers and debug settings have import-time seeds.
- Removed environment compiler/library/policy controls and all user register, unroll, inline and alias knobs.
- Added `krn.Warn` budgets and source-located `PerformanceWarning` diagnostics, controllable through Python warnings.
- The debug logger is `krn.debug`; applications own logging levels.


## Unreleased

- `krn.profile` on NVIDIA: `Profiler.result()` flushes the CUPTI collector before reading, so kernels still in CUPTI's
  buffers are no longer missing or replaced silently by event timings. A CUPTI window is refused only when its own start
  was evicted or CUPTI reported lost records; eviction of older records no longer refuses later windows. Verified on an
  RTX 5080 and an RTX 3090 (see the profile page); hardware counters remain blocked by the driver's profiling setting.
- Warn with `PerformanceWarning` of kind `fusable-multiply-add` where a multiply's only use is an add or subtract, once
  per kernel and target, quoting the first line and its `math.fma` spelling; `krn.Warn` gains
  `hint_fusable_multiply_add` and `min_fusable_multiply_adds`. Generated code and numerics are unchanged.
- Raise compiler-style `TargetError` diagnostics for unsupported explicit target settings, vendor library choices,
  Metal debugging tools, binary artifacts and graph rebinding. Preserve reasons for atomic, intrinsic, fp8 matrix
  and shared-memory refusals. Stop clamping C++ standards and dropping compiler options; wire CPU compiler options
  and persistent cache directories, and Metal fast math. Declare settings support beside every field and document
  backend behavior. Reject explicit pipeline depths, specialized schedules and tile storage without a matching
  lowering. Unset settings and documented prefetch hints retain their target defaults.

- Correct Apple atomic capabilities: no Apple target advertises the full 64-bit atomic set. Measured on M4 Pro
  with MSL 3.0–4.1, only void unsigned min/max in device memory compile; the returning atomic catalog refuses all
  64-bit operations. Remove invalid ulong lowering and restore 32-bit collection paths and 64-bit Metal skips.
- Emit `precise::sqrt` and precise math helpers for Metal while keeping fast math disabled and numerical oracles
  unchanged. Preserve raw BF16 buffer binding and unused scalar omission. Add refusal and publication source checks.

- Wire the Apple Metal backend into `host.Device.current()`, with shared-buffer tensors, queues, awaitable events,
  bound launches, graph replay, and a source/version program cache. Metal libraries compiled from source stay in memory.
- Record Apple7 through Apple10 for M1 through M5; keep 64-bit integer atomics disabled for all Apple targets.
  Note M5 tensor operations and Neural Accelerators without claiming renderer support.

- Centralize native loading in `krn.hw.Libraries`, with registered NVIDIA pathfinder, framework, soname,
  and wheel strategies, cached failures, lookup without loading, and actionable missing-library insights.

- Replace command-output hardware probing with `krn.hw.System`: frozen host and device facts, fresh telemetry and
  process readings, DRM fallback discovery, visibility-aware selection, actionable insights, and hardware-gated extras.
  Retire `facts.machine`, command fixtures, and the old probe functions.
- Replace model bases with frozen dataclasses, explicit value validation, and `dataclasses.replace` in tests.
  Prefer psutil, Linux archspec, Metal/MLX, CUDA, HIP/SMI, OpenVINO/Sysman, wgpu, and hwloc for inventory.
  Isolate permitted Apple native reads, retain cached vendor library discovery, attribute telemetry sources,
  and provide explicit timed energy sampling. Recommend Intel extras and optional portable probes.

- Add `krn.collections.Set` and `Map` for integer keys and values, bounded splitmix64 linear probing, an all-ones side slot, explicit insertion counts, and deterministic last-occurrence map assignment. Add cross-backend Python oracles and a C bulk benchmark. Support boolean tensor storage, host tensor snapshots, and blocking events in the shared runtime contracts. All Apple targets reject 64-bit CAS.

- Add `krn.hashlib` scalar kernels: splitmix64, XXH32/XXH64, byte-range XXH64, MurmurHash3 x86_32, and ordered combination. Add `hash_strings_xxh64`; retain the superseded polynomial string hash.
- Copy examples, experiments, study tooling, and research notes into this workspace; retire external example paths.
- Centralize local evidence under gitignored evidence/results; merge checks and retain the docs-study task.

- Copy the research typed API, compiler, emulator, runtimes, libraries, and data kernels into divisions.
- Export the kernel API and bare dtypes from `krn`; add top-level math, itertools, and functools helpers.
- Rename tools to hw; add emulator, cache, data, and the reserved collections seam; remove std.
- Extract tuning and CPU object caches, retain PATOS settings, and copy/reorganize prototype tests.
- Add a per-file migration map, offline verification report, platform test dependencies, and explicit coverage omissions.
- Retarget the 0.0.1 scaffold to `krn` and Python 3.14, including CI and docs.
- Use nvmath-python for CUDA libraries; add optional device APIs, CCCL, and references.
- Report optional extras separately and keep vendor packages out of the default test environment.
- Add immutable hardware facts, injected probes, tool discovery, and `krn doctor`.
- Recommend opt-in vendor extras from observed devices and drivers.
- Add captured machine tests, diagnostic contracts, and documented package seams.
- Align Python 3.14 checks, documentation, mainboard tasks, and dependency hygiene.

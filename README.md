# krn

A typed Python kernel language. Write a GPU kernel as a small, ordinary Python function; `krn` captures it, checks its types, and renders it for the hardware you have.

Status: pre-release. Python 3.14 only. Nothing is published yet. The plan is in [ROADMAP.html](ROADMAP.html).

The 0.0.1 candidate is experimental. Building or importing it does not establish every backend or operation. Isolated installation checks currently cover a small scalar CPU kernel on local Arm64; fresh installed-package GPU qualification remains pending. Rust and JavaScript starters provide host sizing utilities only; they do not yet share this compiler.

Tools and settings are generic: an explicit choice a target cannot honor raises `krn.errors.TargetError`
with the feature, architecture, hardware or backend limitation, and an alternative. Unset values retain target defaults.
See [settings and target diagnostics](https://krn.dev/start/settings/) for backend support and compiler options.

`krn.hw.report(System.detect())` returns a command-free machine inventory, hardware insights, and the exact optional-package install line. krn has no command line.
Use `from krn.hw import System` and `System.detect()` for frozen OS, CPU, memory, volume, Python, process-limit, and GPU
facts. Static device properties are cached; `device.telemetry()`, `device.processes()`, and `device.nvlinks()` read fresh
state. Unavailable readings are `None`. Visibility masks affect selection without removing physical devices from the
inventory. Hardware values are frozen dataclasses; use `dataclasses.replace` to derive a modified snapshot.
`system.libraries` caches available vendor kernel libraries, and `system.probe_failures` explains unavailable sources.

`krn.hw.Libraries` is the shared native library loader: `load("nvml")` returns a frozen value with its path,
ctypes handle, and loader source; `find("nvml")` only locates, and `available("nvml")` checks loading.
NVIDIA uses optional `cuda.pathfinder`; other runtimes use exact frameworks, sonames, or installed wheel manifests.
Successes and failures are cached until `Libraries.clear_cache()`. `LibraryNotFound` records names, search
locations, and an installation hint for `krn.hw.report`.
`system.topology` exposes optional Linux hwloc NUMA, cache, and PCI snapshots. `device.sample(interval=0.1)` derives
power from two timed energy readings where supported; `telemetry.sources` records field provenance. See the [hardware roadmap](ROADMAP.html#hardware) for remaining coverage and verification work.

```python
import asyncio

import numpy as np

from krn import T, Vector, items, kernel, math
from krn.runtime import host


@kernel
def square_root(x: Vector[T.f32], y: Vector[T.f32]) -> None:
    for i in items(x.shape[0]):
        y[i] = math.sqrt(x[i])


async def main() -> None:
    device = host.Device.current()
    x = device.to_device(np.arange(8, dtype=np.float32))
    y = device.empty(T.f32, 8)
    square_root(x, y)  # the launch geometry is inferred from the body
    await device.synchronize()
    np.testing.assert_allclose(y.to_host(), np.sqrt(np.arange(8, dtype=np.float32)))


asyncio.run(main())
```

Kernels are captured from the source file, so run this from a file. On an Apple GPU Mac, `host.Device.current()` selects Metal. Shared buffers expose an `array` view without launch-time copies; synchronize before reading or changing storage used by pending work. Metal compiles with fast math disabled and uses `precise::sqrt` for float32 square roots. Apple targets refuse 64-bit atomics: Metal only offers unsigned min/max in device memory without a result; collections use 32-bit keys on Apple. BF16 tensors store raw `uint16` words. `KRN_TARGET` can select an explicit target.

Tiles take the matrix product `s = q @ k.transpose()`, `where`, comparisons and `arange` for masks, and the reductions and
normalizations with `keepdims`. `examples/04_libraries/08_flash_attention.py` is flash attention in about twenty lines.

## Targets

These are historical source-tree results, not qualification of every operation in the installed 0.0.1 candidate.

| Target | Route | State |
| --- | --- | --- |
| CPU (x86 AVX2, Arm) | C through clang | tested |
| NVIDIA | CUDA through NVRTC; tensor cores through `mma.sync` | tested on sm_86, sm_90a and sm_120a |
| Apple | Metal Shading Language through PyObjC | tested on an M4 Pro |
| AMD | HIP | compile-checked only, never run on a card |
| Intel | Level Zero with SPIR-V | planned |

A Python emulator is the oracle every backend is compared with. Vendor libraries (cuBLASLt, cuSPARSE, MPS) are fast paths behind the same calls, never a requirement.

## Modules

`krn.math`, `krn.itertools`, `krn.functools`, and `krn.hashlib` are the kernel helpers. `krn.collections` provides fixed `Set` and `Map` tables for 32- and 64-bit integer keys, with batched insertion, lookup, explicit overflow counts, and unordered snapshots. Map updates take the last value for duplicate keys. See [the collections reference](https://krn.dev/libraries/collections/) for synchronization and backend limits. `krn.hashlib` provides splitmix64, fixed-width XXH32/XXH64 and MurmurHash3, byte-range XXH64, and ordered field-hash combination. Callers canonicalize floats before hashing their bits. See [the hash reference](https://krn.dev/libraries/hashlib/). `krn.profile` times Python spans, runtime launches, and optional native vendor kernels; see [profiling](https://krn.dev/profile/). Never put `src/krn` on `sys.path`: its module names would shadow the standard library.

## Install

No release exists yet. From a checkout, `mb dep install`, then `mb run python -c 'from krn.hw import System, report; print(report(System.detect()))'`. The report reads the machine and prints the exact `pip install` line that matches its hardware; it never installs anything.

| Extra | Pulls | For |
| --- | --- | --- |
| `cuda` | cuda-core, cuda-bindings, NVRTC | Linux with an NVIDIA GPU |
| `cuda-libs` | nvmath-python (cuBLAS, cuSPARSE, cuFFT and more) | with `cuda` |
| `cuda-dx`, `cccl` | device-side math APIs, device-wide algorithms | optional NVIDIA development |
| `refs` | CuPy, CUTLASS DSL, FlashAttention and other oracles | reference tests and benchmarks only |
| `metal` | PyObjC Metal, MetalPerformanceShaders, MLX | a Mac |
| `hip` | hip-python, amdsmi | Linux with an AMD GPU and ROCm |
| `intel` | pyzes, OpenVINO | Intel GPUs and NPUs |
| `probe` | wgpu, Linux pyhwloc | optional portable adapter and topology facts |
| `dev` | ruff, ty, pyrefly, pytest, Hypothesis | development |

## Layout

| Folder | Holds |
| --- | --- |
| `src/krn` | the package, one folder per division: facts, hw, frontend, compiler, emulator, runtime, lib, cache, data, diagnostics, collections |
| `tests` | tests mirroring the divisions |
| `examples` | runnable examples in five folders by complexity, from a first kernel to systems; `examples/README.md` is the table of contents |
| `experiments` | benchmarks and conformance runs; results go to the lake, not the tree |
| `development` | task ownership, explicit dependencies and acceptance records; historical research is preserved in the Mainboard lake |

The website is not in this repository. It is the app `../krn.dev`, with its own repository and mb workspace, and it reads nothing from here.

## Development

Everything runs through mainboard from the full source checkout: `mb run lint`, `mb run test`, `mb run ci`. Ruff, ty and pyrefly cover src, tests, examples and experiments. Hardware hosts are declared in `mb.toml`: `mb host sync <host>`, then `mb run --on <host> python -m pytest --no-cov tests`. Coverage reports the complete library without a forced percentage; skipped hardware paths remain gaps, and passing host tests do not establish accelerator support. Use `krn.configure` or `krn.config` for current grouped immutable settings; the planned flat API is not implemented yet. Only target, cache, workers and `KRN_DEBUG*` seed settings at import. See [CONTRIBUTING.md](CONTRIBUTING.md) for dependency changes.

Settings change with `krn.configure(lib="ours", warn=krn.Warn(max_registers_per_thread=192))`,
`krn.config.lib.matmul = "vendor"`, or a scoped `with krn.configure(lib="ours"):`. Compilations retain their starting
snapshot; later work uses the new settings and matching program-cache keys. The compiler owns register, unroll,
inlining and alias choices. Slow computed choices emit `krn.PerformanceWarning` at the kernel source; change tile sizes,
live values or `Blocks(threads=...)`, adjust `krn.Warn` thresholds, or use `warnings.filterwarnings` to silence/escalate.
See the [settings guide](https://krn.dev/start/settings/) for groups, defaults and backend support.

Numerics are strict by default: `a * b + c` has two roundings. Opt in at a single call
with `krn.fast.math.sqrt(x)`, whose strict twin is `krn.math.sqrt`. A fused multiply-add
rounds once and gives up no precision, so it is strict: `krn.math.fma(a, b, c)`. Compilation explains the
selected target contract in `PrecisionWarning` and `Decisions`; ordinary warning filters
control its display. See [Numerics](https://krn.dev/language/numerics/).

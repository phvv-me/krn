---
type: Reference
title: "kernel-languages experiments"
description: "CPU-only measurements behind STUDY.md: scalar semantics, inline PTX validity, NVRTC and AMDGPU compile latency, and prebuilt library linking."
generated: { by: agent:claude, at: 2026-10-30T12:00:00+00:00 }
---
# experiments

All scripts are typed Python 3.14 and clean under `ruff`, `ty` and `pyrefly` (see the repository `pyproject.toml`). The first seven measure
compilers, not kernels, and ran without a GPU. `sparse_bench.py` launches kernels: on the RTX 5080 of silo and on the CPU
backend of the Mac. The run commands are in the parent `README.md`, and in its own docstring.

| Script | Question | Result file |
| --- | --- | --- |
| `scalar_semantics.py` | Where do strict same-dtype and C promotion disagree in cutok's device code? | `evidence/results/scalar_semantics.json` |
| `ptx_asm.py` | Which inline PTX does `ptxas` accept on which target? | `evidence/results/ptx_asm.json` |
| `compile_latency.py` | How long do NVRTC and nvJitLink take, cold and warm, by unit size and headers? | `evidence/results/nvrtc_latency.json` |
| `amd_latency.py` | How long does the LLVM beneath hipRTC take for gfx942 and gfx950? | `evidence/results/amd_latency.json` |
| `split_link.py` | Does a prebuilt device library plus a tiny kernel beat recompiling? | `evidence/results/split_link.json` |
| `promotion.py` | How do torch, C and a lossless rule promote integer types? (needs torch) | `evidence/results/promotion.json` |
| `metrics.py` | How large and how complex is a kernel: NLOC, Lizard CCN and tokens, LLM tokens, API symbols read? | printed table, or the `corpus` mode output |
| `gemm_gap.py` | How far is our FMA lowering of `@` from cuBLASLt over Tilus's autotune space, on a card? | `evidence/results/gemm_gap_silo_{2048,4096}.json` |
| `sparse_bench.py` | How fast are our sparse products against cuSPARSE (`cuda`, on a card) and Accelerate (`cpu`), on a matrix set whose rule is in its docstring? | `evidence/results/sparse_{spmv,spmm16}_{silo,mac}.json` |

`nvidia.py` is the ctypes layer for NVRTC and nvJitLink and `units.py` holds the translation units. `metrics.py` takes either
`sketch <examples dir>` (our kernels) or `corpus <corpus.json> <src root> <out.json>` (original kernels at the exact
repository, path and line range the corpus records); it needs `lizard` and `tiktoken`. Limits: single runs of
medians on a shared 4-vCPU Xeon, synthetic units in place of real generated cutok C++, a clang process standing in for
hipRTC, nothing on Metal, and no kernel was ever launched.

Two folders hold kernels instead of scripts. `sketch/` has the small kernels that the tests and the benchmarks capture by
name (`elementwise`, `softmax`, `gemm`, `automaton`, `target_variables`, `mistakes`) and the typed design sketches that
capture cannot lower yet (`intrinsic_lowerings`, `warp_argmax`, `async_roles`, `gemm_epilogues`, `cholesky_autograd`,
`serving_engine`); a sketch moves back into `examples/` when it runs. `library_checks/` holds the checks of the string,
column and helper libraries that build their programs through compiler internals, until `krn.data` has a default
launcher. `tests/examples` runs both folders. See the [public API roadmap](../ROADMAP.html#language).

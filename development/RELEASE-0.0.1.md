# krn 0.0.1 release record

The owner authorized publishing the existing Python history to https://github.com/phvv-me/krn and publishing 0.0.1 to PyPI. The existing history is public and PyPI 0.0.1 publication succeeded through trusted publishing from source 7484e19. A fresh installation from the public PyPI index passed the CPU smoke. Rust and npm publication remain separate pending actions.

## Candidate verification

- Ruff, format, ty and pyrefly pass over source, tests, examples and experiments.
- Local Python 3.14.10 suite: 10,999 passed, 6,020 skipped, 18 hardware tests deselected and 7 expected failures; no executed test failed.
- Full-library coverage reports 89.78% combined, 92.30% statements and 81.01% branches across 19,797 statements and 5,688 branches. No files are omitted; 825 lines remain excluded by the documented syntax rules. Coverage is evidence of executed Python paths, not target support.
- Separate Python 3.14.7 environments passed uv wheel and pip source installation plus the installed-package affine CPU smoke at lengths 1 and 17.
- The C smoke runs on macOS Arm64 using the existing avx2 capability profile; it does not establish x86 ISA support.
- Fresh installed-package CUDA, Metal, Linux and free-threaded qualification remains unverified. The release is experimental and planned APIs remain planned.
- The documentation site builds 93 pages with no local link failures; the library and app plans preserve historical task states through two audit rounds.

## Publication contract

GitHub CI runs the checked-in host test and static-check gate before PyPI trusted publishing. The workflow tags only after upload. Verify the public registry metadata and install the public wheel before calling publication complete. If OIDC authentication fails, configure the owner, repository krn, workflow publish.yml and environment pypi in the PyPI account outside chat; never put a token in task records.

The Rust no_std checked ceildiv/align_up starter and typed JavaScript ESM starter were packaged and verified previously; neither is a compiler binding and neither has been published by this task. Retain their prior artifact receipts in the lake.

## Local candidate artifacts

- `krn-0.0.1-py3-none-any.whl`: 547292 bytes; SHA-256 `c476b08d6d2775e7c25162e0f366aab4c3bc39b4ca8b62204192123c34cf12bf`.
- `krn-0.0.1.tar.gz`: 537276 bytes; SHA-256 `b03e01520fca0e643b264622624682b9aa9e7adba481c3572ef30dd08b0f344c`.

## Published receipt

- [PyPI 0.0.1](https://pypi.org/project/krn/0.0.1/) and [GitHub release](https://github.com/phvv-me/krn/releases/tag/v0.0.1).
- [Release workflow](https://github.com/phvv-me/krn/actions/runs/41786394375): Linux 10,836 passed and 6,161 skipped; macOS 9,552 passed and 7,445 skipped; each had 15 deselected hardware tests and 7 expected failures. Static checks passed on both.
- Public-wheel installation used an empty environment and uncached PyPI resolution, followed by isolated import/version and CPU execution checks.
- Wheel SHA-256: `4aeff2d9a250a9db94de35886970e705e7767ea6efda8c3fbe61890c55629cdb`.
- Source SHA-256: `b2203213ae58d8fb0b76fdf8d9b56a076894aa217def46ad0d9c16df9e3871e8`.
- Raw registry metadata, CI logs and research receipts are preserved in the Mainboard lake under `evidence/krn-roadmap-r8`. Local candidate hashes above refer to the earlier local build, not the independently built registry artifacts.

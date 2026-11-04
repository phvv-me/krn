# krn 0.0.1 release record

The owner authorized publishing the existing Python history to https://github.com/phvv-me/krn and publishing 0.0.1 to PyPI. The repository and its pypi environment exist; the release workflow must still finish before publication is claimed. Rust and npm publication remain separate pending actions.

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

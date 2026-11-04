# Initial baseline

This source snapshot records unfinished work. It is not a release candidate or a claim of full hardware support.

- `mb run --no-install lint`: fails with 12 Ruff findings in existing asyncio, compiler, emulator and runtime files.
- `mb run --no-install lint-format`: two files need formatting, `compiler/heuristics.py` and `runtime/settings.py`.
- `mb run --no-install lint-ty`: 13 diagnostics, including a deprecated annotation warning.
- `mb run --no-install lint-pyrefly`: 12 errors; most concern list arguments passed to the tuple-based CUDA call interface, plus the Metal device waiter annotation.
- Focused library-loader, log-site and exception-site tests: 35 passed, one warning.
- CPU affine and dtype-rejection smoke cases: five passed, one expected performance warning.
- Metal subgroup-reduction smoke case: one passed, with the existing subnormal-flush precision warning.
- No remote CUDA, Intel GPU, AMD, free-threaded, full-suite or performance campaign ran during this planning revision.

Focused tests used `--no-cov`; they establish no coverage percentage. The existing `fail_under = 100` setting still needs the roadmap's coverage-policy reconciliation before unattended rounds.

The roadmap dependency graph is acyclic. It retains earlier task IDs and completion checks, with exactly five example designs and cumulative verification per milestone. Three independent reviewers each audited the revised specification twice; corrections were integrated, but this does not prove every future task has no ambiguity.

Historical notes were ingested and verified before eviction: 332 files, 14,907,384 bytes, under the original workspace path `krn/krn.py/notes`. Recover them with `mb data lake materialize notes` from this repository when source archaeology is needed; they are not current instructions. Original source comments may still name these archived paths.

Initial local commits were authorized separately. Automatic Mainboard commits remain disabled; later commits, pushes, releases and paid compute still require approval. Git and jj share each active repository; the app's jj snapshot limit admits its existing exported specification files.

## Release preparation follow-up

The initial failures above remain historical evidence. Release preparation now passes all static checks and 10,999 local tests; 6,020 skips and 7 expected failures remain explicit. The migration omission list and forced 100% threshold were removed. Combined whole-library coverage is 89.78%, with syntax exclusions still reported. See RELEASE-0.0.1.md for candidate scope and registry state; none of these checks establishes all-hardware support.

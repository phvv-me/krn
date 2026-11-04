# krn

A typed Python kernel language, in development.

## Working rules

- Keep the public surface small and documented.
- Code in `krn._internals` is private: public modules may use it, users and examples never import it, and nothing in it is re-exported.
- Update README.md and CHANGELOG.md when behavior changes, and the site in `../krn.dev` (its own repository) when its pages or data change.
- Follow development/README.md for task claims, dependencies and evidence. Local integration after checks is authorized; commits, pushes and releases require owner approval.
- Unattended compute uses owned runners. Paid hosted jobs, rentals and existing GPU credits require an owner-approved campaign and cap.
- No commits, tags, pushes, or publishing without explicit instructions.
- Use the currently admitted Python version through Mainboard; the roadmap targets 3.14 and 3.14t plus newer minors after compatibility checks, without claiming those checks already pass. macOS and Linux only: Windows is not supported, so write no Windows branch, table or test (owner, 2026-11-02).
- Acronyms stay uppercase inside class names (PEP 8): `OS`, `CPU`, `GPU`, `NPU`, `PCILink`, `CUDADevice`, `HIPRenderer`, `AMDDevice`, never `Os`, `Cpu`, `Cuda`. A brand that is not an acronym keeps normal capitalization (`Nvidia`, `Apple`, `Intel`). Function, method and variable names stay snake_case.
- krn has no command line and no verbs (owner, 2026-11-02); the only cyclopts code is the scripts in experiments/. Use frozen dataclasses, logging, and precise types. krn has no pydantic dependency (owner, 2026-11-02): a value type is `@dataclass(frozen=True, kw_only=True)` with an explicit `__post_init__` for any invariant, and static facts that are computed once are `functools.cached_property` (so no `slots=True`). Never use `typing.NamedTuple` for these.
- No print, typing.Any, object annotations, or type ignores.
- Tools and settings are generic across hardware (owner, 2026-11-02). When a path does not exist on a target (a register cap on Metal, a 64-bit atomic on Apple, a vendor library on another vendor) the compiler or runtime raises a compilation error that names the setting or feature, the target, why the hardware lacks it, and what to do instead. Never ignore it silently, never downgrade it to a warning, never skip it.

## Commands

- Install dependencies with `mb dep install`.
- Change dependencies in mb.toml and pyproject.toml, then `mb dep lock` and `mb dep install`.
- Lint and typecheck with `mb run lint` (ruff, ty, and pyrefly over src, tests, examples, experiments).
- Test with `mb run test`.
- Check a change with `mb run ci`.
- Build documentation with `mb run docs`.
- Bound commands with `mb proc timeout <seconds> <command>`.

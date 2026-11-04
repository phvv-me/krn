"""Every device intrinsic with its spelling on NVIDIA, AMD, Apple and the CPU

`catalog.ENTRIES` is the one table the renderers read: a format string per vendor, an empty string where the vendor
has no instruction, and a note that says what would be needed. `catalog.spell(name, target, ...)` gives one
spelling, and `catalog.missing()` lists the names of numba-cuda-mlir's `cuda` namespace that have no answer (none:
each is an entry or a plain Python form such as `a if p else b`). A kernel that calls an intrinsic the target has no
spelling for fails at capture, at your line, naming the target and the note.

Level: 3
Topics: intrinsics, catalog, vendors
Needs: 03_memory_and_sync/05_warp_shuffles
"""

import logging

from krn import catalog, targets

log = logging.getLogger("catalog")
SHOWN = ("sm_90a", "gfx942", "apple_m5", "avx512")


def column(name: str, target: targets.Target) -> str:
    try:
        return catalog.spell(name, target, "a", "b", "c", bits=32)
    except LookupError:
        return "-"


def main() -> None:
    chosen = [targets.Target.of(arch) for arch in SHOWN]
    log.info(
        "%d intrinsics in the catalog; unmatched numba-cuda-mlir names: %s", len(catalog.ENTRIES), catalog.missing()
    )
    for entry in catalog.ENTRIES:
        if entry.name.startswith(("thread.index", "block.index", "block.dim", "grid.dim")) and not entry.name.endswith(
            ".x"
        ):
            continue  # one row per axis family is enough
        log.info("%s  (%s)", entry.name, entry.summary)
        for target in chosen:
            log.info("    %-9s %s", target.arch, column(entry.name, target))
        if entry.note:
            log.info("    note: %s", entry.note)
    log.info("plain Python instead of a function:")
    for name in ("shared.array", "local.array", "const.array_like", "selp", "is_float16_supported"):
        log.info("  %-22s %s", name, catalog.OPERATORS[name])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

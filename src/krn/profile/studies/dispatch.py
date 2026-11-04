from collections.abc import Mapping

from ...hw.devices.base import Device


def arch_config[T](table: Mapping[str, T], *, default: T, gpu: Device | None = None) -> T:
    """Select the entry of `table` for `gpu`'s architecture.

    table: maps a `GPU.arch_key` (e.g. `sm_90`) to that generation's config, such as
        tile sizes or a Helion config.
    default: returned on a CPU-only host (`gpu` is None) or when the key is absent from `table`.
    """
    return default if gpu is None else table.get(gpu.arch_key, default)

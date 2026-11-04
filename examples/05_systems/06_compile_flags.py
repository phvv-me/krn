"""Compiler flags per backend: the C++ standard each toolchain accepts, and the options that refuse a target

`host.CompileOptions.resolve(target, settings=...)` turns the settings into the flags of one backend: the newest C++
standard its toolchain accepts, line tables when `Debug(lineinfo=True)`, and the optimization a backend group asks
for. An option a backend cannot honor raises `TargetError` naming the setting, the target, why the hardware or
toolchain lacks it and what to do instead, never a silent downgrade. The file prints the flags of four targets under
three settings.

Level: 5
Topics: compile options, flags, settings, TargetError
Needs: 01_first_kernels/03_settings
"""

import logging

from krn import BackendCompile, Compile, Debug, Settings, host, targets
from krn.errors import TargetError


def show(title: str, settings: Settings) -> None:
    logging.info("%s", title)
    for arch in ("sm_120a", "gfx942", "apple_m5", "spr"):
        try:
            options = host.CompileOptions.resolve(targets.Target.of(arch), settings=settings)
            logging.info("%s: %s", arch, options.flags())
        except TargetError as error:
            logging.info("%s", error)


if __name__ == "__main__":
    show("defaults", Settings())
    show("line tables", Settings(debug=Debug(lineinfo=True)))
    show("CUDA fast compilation", Settings(compile=Compile(cuda=BackendCompile(opt=host.Opt.FAST_COMPILE))))

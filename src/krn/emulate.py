"""The emulator: run a kernel thread by thread in Python, with breakpoints, bounds checks and deadlock reports.

    from krn.emulate import emulate

`emulate(kernel, Blocks(...), *arguments)` updates the arrays in `arguments` in place. `Deadlock` is raised when threads
wait at barriers or collectives that the others never reach.
"""

from krn.emulator.emulate import Deadlock, emulate

__all__ = ["Deadlock", "emulate"]

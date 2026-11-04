class HardwareError(RuntimeError):
    """Unavailable hardware, with an actionable hint."""

    def __init__(self, message: str, fix: str) -> None:
        self.fix = fix
        super().__init__(f"{message}. {fix}")


class ProbeError(HardwareError):
    """A native source could not be loaded or queried."""

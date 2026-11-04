"""Fixed table capacity errors."""


class CapacityError(RuntimeError):
    """One or more input keys could not fit in the table."""

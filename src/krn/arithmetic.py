"""Integer arithmetic the divisions share and the standard library lacks. Host side; kernels use `krn.math`."""


def ceildiv(numerator: int, denominator: int) -> int:
    """The smallest integer not below `numerator / denominator`, for any nonzero denominator.

    Floor division rounds toward negative infinity, so negating before and after turns it into a ceiling.
    """
    return -(-numerator // denominator)


def align_up(value: int, multiple: int) -> int:
    """The smallest multiple of `multiple` that is not below `value`."""
    return ceildiv(value, multiple) * multiple

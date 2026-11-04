"""Helpers for kernels, in the shape of itertools, more-itertools, toolz and es-toolkit.

Each module holds `@kernel` functions that capture inlines at the call site. Rows are vectors plus half-open bounds;
runtime-length results are generators or caller-owned tensors. Sizes and steps are positive capture-time constants.
"""

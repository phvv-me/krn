"""The execution hierarchy as modules of functions, in the names of CUDA and Cooperative Groups.

`thread`, `warp`, `block`, `grid` and `cluster` hold the coordinates and the collectives of one level each, and `cg`
holds group values (tiles, blocks, active threads) that run a collective over a named set of threads. The modules have
no state, so they are modules and not classes; every function has a lowering per target in `catalog.py`.
"""

from . import block, cg, cluster, grid, thread, warp

__all__ = ["block", "cg", "cluster", "grid", "thread", "warp"]

"""The thread block cluster of sm_90a and later: blocks that run together and can reach each other's shared memory.

The names are those of `cooperative_groups::this_cluster()`. Only the barrier and the ranks are lowered; distributed
shared memory (`map_shared_rank`) needs a pointer value the language does not have. A target without `Feature.CLUSTERS`
refuses these calls at capture, and a cluster needs a cluster launch. Bodies are signature-only."""

from ..numerics import todo, u32


def block_rank() -> u32:
    """Linear rank of this block inside its cluster."""
    return todo()


def num_blocks() -> u32:
    """Blocks in this cluster."""
    return todo()


def sync() -> None:
    """A barrier for every thread of every block of the cluster."""
    return todo()

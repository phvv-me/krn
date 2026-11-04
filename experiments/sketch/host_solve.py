"""A low-precision solve as a PyTorch op with a backward. The forward factors in a block-scaled format and refines.

The backward is the implicit derivative: one more solve with A^T under the same recipe, so no tape runs through the
factorization or the refinement loop. The registration is the executed pattern of notes/backward-autodiff.md (gradcheck
passes in torch 2.15.0), and torch.func.grad through a register_autograd op fails there.

One op per recipe, because the recipe lives in the closure: torch types neither FunctionCtx attributes nor a
saved_tensors longer than one (a stub quirk that list() unpacks around), so the context carries tensors only.
"""

import torch
from torch import Tensor
from torch.autograd.function import BackwardCFunction, FunctionCtx

from krn.data import formats as formats


def lu_refine(a: Tensor, b: Tensor, factor: formats.BlockFormat, refine: int) -> Tensor:
    """Blocked LU with pivot_search and trailing_update (examples/lu.py), then `refine` residual corrections."""
    raise NotImplementedError


def lowprec_solve_op(name: str, factor: formats.BlockFormat, refine: int = 8) -> torch.library.CustomOpDef:
    @torch.library.custom_op(f"krn::{name}", mutates_args=())
    def solve(a: Tensor, b: Tensor) -> Tensor:
        return lu_refine(a, b, factor, refine)

    @solve.register_fake
    def _fake(a: Tensor, b: Tensor) -> Tensor:
        return torch.empty_like(b)

    def _save(ctx: FunctionCtx, inputs: tuple[Tensor, Tensor], output: Tensor) -> None:
        ctx.save_for_backward(inputs[0], output)

    def _backward(ctx: BackwardCFunction, grad_x: Tensor) -> tuple[Tensor, Tensor]:
        a, x = list(ctx.saved_tensors)
        grad_b = solve(a.mH.contiguous(), grad_x)
        return -grad_b @ x.mH, grad_b

    solve.register_autograd(_backward, setup_context=_save)
    return solve

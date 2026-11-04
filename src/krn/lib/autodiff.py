"""Backward passes attached to host functions, so a kernel pipeline trains inside PyTorch.

A differentiable function composes launches. Its adjoint is a second function of the saved input, the output and the
output gradient, written with the same kernels. Registration is typed: the adjoint must accept what the forward
takes and returns, and must return the gradient with the input's type. The binding to `torch.library.custom_op` and
`torch.autograd` is generated, so the same function works eagerly and under `torch.compile`.
"""

from collections.abc import Callable


class Differentiable[A, R]:
    def __init__(self, forward: Callable[[A], R]) -> None:
        self.forward = forward
        self.adjoint: Callable[[A, R, R], A] | None = None

    def __call__(self, a: A) -> R:
        return self.forward(a)

    def backward(self, adjoint: Callable[[A, R, R], A]) -> Callable[[A, R, R], A]:
        """Register the adjoint: `(input, output, grad_output) -> grad_input`."""
        self.adjoint = adjoint
        return adjoint


def differentiable[A, R](forward: Callable[[A], R]) -> Differentiable[A, R]:
    return Differentiable(forward)

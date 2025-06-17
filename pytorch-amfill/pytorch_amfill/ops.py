import torch
from torch import Tensor

__all__ = ["amfill", "amfill_"]


def amfill(im: Tensor, mask: Tensor) -> Tensor:
    """Fills the masked elements of im into new tensor."""
    return torch.ops.pytorch_amfill.amfill.default(im, mask)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("pytorch_amfill::amfill")
def _(im, mask):
    torch._check(im.shape == mask.shape)
    torch._check(im.dtype == torch.float)
    torch._check(mask.dtype == torch.bool)
    torch._check(im.device == mask.device)
    return torch.empty_like(im)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.pytorch_amfill.amfill.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.pytorch_amfill.amfill.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "pytorch_amfill::amfill", _backward, setup_context=_setup_context)

def amfill_(iom: Tensor, mask: Tensor) -> Tensor:
    """Fills the masked elements of im and returns it."""
    torch.ops.pytorch_amfill.amfill_.default(iom, mask)

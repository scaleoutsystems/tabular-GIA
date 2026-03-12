"""Utilities for temporarily changing model mode behavior."""
from contextlib import contextmanager

from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm


@contextmanager
def bn_eval_mode(model: Module):
    """Temporarily switch BatchNorm layers to eval mode and then restore them.

    This hardens gradient generation against per-batch BN statistic leakage while
    leaving non-BN modules (e.g., Dropout) unchanged.
    """
    switched = []
    for module in model.modules():
        if isinstance(module, _BatchNorm) and module.training:
            module.eval()
            switched.append(module)
    try:
        yield
    finally:
        for module in switched:
            module.train()

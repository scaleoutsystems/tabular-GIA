"""Optimizer objects used for FL training without higher-order graph retention."""
import torch
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import Tensor
from torch.autograd import grad

from leakpro.utils.import_helper import Dict, Self, Tuple


class FLOptimizer(ABC):
    """Abstract FL Optimizer."""

    def __init__(self: "FLOptimizer") -> None:
        """Initialize the FLOptimizer."""
        raise NotImplementedError("This is an abstract class and should not be instantiated directly.")

    @abstractmethod
    def step(self: "FLOptimizer", loss: Tensor, params: Dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """Perform a single optimization step.

        Args:
        ----
            loss (torch.Tensor): The loss value calculated from the model's output.
            params (Dict[str, torch.Tensor]): A dictionary of model parameters to be updated.

        Returns:
        -------
            OrderedDict[str, torch.Tensor]: A new set of parameters which have been updated.

        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class FLSGD(FLOptimizer):
    """Implementation of SGD that performs a step to a new set of parameters."""

    def __init__(self: Self, lr: float = 1e-2) -> None:
        """Init."""
        self.lr = lr

    def step(self: Self, loss: Tensor, params: Dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """Perform a single optimization step."""
        grad_params = [(name, param) for name, param in params.items() if param.requires_grad]

        # Compute gradients only for grad params.
        grads = grad(
            loss,
            [param for _, param in grad_params],
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=True,
        )

        grad_iter = iter(grads)
        updated_params = OrderedDict()

        for name, param in params.items():
            if param.requires_grad:
                grad_value = next(grad_iter)
                if grad_value is None:
                    grad_value = 1
                updated_params[name] = param - self.lr * grad_value
            else:
                # Leave params that do not require grad as they are.
                updated_params[name] = param
        return updated_params


class FLAdam(FLOptimizer):
    """Implementation of Adam that performs a step to a new set of parameters."""

    def __init__(
        self: Self,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        """Initializes the FLAdam optimizer."""
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self: Self, loss: Tensor, params: Dict[str, Tensor]) -> OrderedDict[str, Tensor]:
        """Perform a single optimization step."""
        grad_params = [(name, param) for name, param in params.items() if param.requires_grad]
        gradients = grad(
            loss,
            [param for _, param in grad_params],
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )

        if self.weight_decay != 0:
            gradients = [
                grad_value + self.weight_decay * param
                for grad_value, (_, param) in zip(gradients, grad_params)
            ]

        # Initialize m and v.
        if not self.m:
            self.m = {name: torch.zeros_like(param) for name, param in params.items()}
            self.v = {name: torch.zeros_like(param) for name, param in params.items()}
        self.t += 1

        gradient_iter = iter(gradients)
        new_params = OrderedDict()
        for name, param in params.items():
            if param.requires_grad:
                gradient = next(gradient_iter)
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * gradient
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (gradient**2)

                m_hat = self.m[name] / (1 - self.beta1**self.t)
                v_hat = self.v[name] / (1 - self.beta2**self.t)
                adam_grad = m_hat / (v_hat.sqrt() + self.eps)

                new_params[name] = param - self.lr * adam_grad
            else:
                new_params[name] = param
        return new_params
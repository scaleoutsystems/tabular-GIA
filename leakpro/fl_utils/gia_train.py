"""Train function that keeps the computational graph intact."""
from collections import OrderedDict

import torch
from torch import Tensor, cuda, device
from torch.autograd import grad
from torch.func import functional_call, grad as func_grad, vmap
from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_module_to_functional import MetaModule
from leakpro.fl_utils.gia_optimizers import MetaOptimizer


def train(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer,
    criterion: Module,
    epochs: int,
) -> list:
    """Model training procedure for GIA.

    This training will create a computational graph through multiple steps, which is necessary
    for backpropagating to an input image.

    Requires a meta optimizer that performs step to a new set of parameters to keep a functioning
    graph.

    Training does not update the original model, but returns a norm of what the update would have been.
    """
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    model.to(gpu_or_cpu)
    patched_model = MetaModule(model)
    outputs = None
    for _ in range(epochs):
        for inputs, labels in data:
            inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), (labels.to(gpu_or_cpu, non_blocking=True) if
                                                                        isinstance(labels, Tensor) else labels)
            outputs = patched_model(inputs, patched_model.parameters)
            loss = criterion(outputs, labels).sum()
            patched_model.parameters = optimizer.step(loss, patched_model.parameters)
    model_delta = OrderedDict((name, param - param_origin)
                                            for ((name, param), (name_origin, param_origin))
                                            in zip(patched_model.parameters.items(),
                                                    OrderedDict(model.named_parameters()).items()))
    return list(model_delta.values())

def train_nostep(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer,  # noqa: ARG001
    criterion: Module,
    epochs: int,
) -> list:
    """Model training procedure for GIA.

    This training will create a computational graph through multiple steps, which is necessary
    for backpropagating to an input image.

    Requires a meta optimizer that performs step to a new set of parameters to keep a functioning
    graph.

    Training does not update the original model, but returns a norm of what the update would have been.
    """
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    model.to(gpu_or_cpu)
    outputs = None
    for _ in range(epochs):
        for inputs, labels in data:
            inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), (labels.to(gpu_or_cpu, non_blocking=True) if
                                                                        isinstance(labels, Tensor) else labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels).sum()
            grads = grad(
                loss,list(model.parameters()),
                retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True
            )
    return grads

def train_nostep_vectorized(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer | None,  # noqa: ARG001
    criterion: Module,
    epochs: int,
) -> list[Tensor | None]:
    """Vectorized train_nostep-style gradient extraction across clients."""
    gpu_or_cpu = next(model.parameters()).device
    model.to(gpu_or_cpu)
    named_params = list(model.named_parameters())
    trainable_pos = [idx for idx, (_, p) in enumerate(named_params) if p.requires_grad]
    num_params = len(named_params)
    if not trainable_pos:
        return [None for _ in range(num_params)]

    trainable_names = [named_params[idx][0] for idx in trainable_pos]
    trainable_params = tuple(named_params[idx][1] for idx in trainable_pos)
    frozen_params = OrderedDict((name, p) for idx, (name, p) in enumerate(named_params) if idx not in trainable_pos)
    buffers = OrderedDict(model.named_buffers())

    def loss_for_client(trainable_values: tuple[Tensor, ...], x_c: Tensor, y_c: Tensor) -> Tensor:
        state = OrderedDict(zip(trainable_names, trainable_values))
        state.update(frozen_params)
        state.update(buffers)
        out = functional_call(model, state, (x_c,))
        return criterion(out, y_c).sum()

    grad_fn = func_grad(loss_for_client)
    client_gradients_trainable: tuple[Tensor, ...] = tuple()
    for _ in range(epochs):
        for inputs_by_client, labels_by_client in data:
            inputs_by_client = inputs_by_client.to(gpu_or_cpu, non_blocking=True)
            labels_by_client = (
                labels_by_client.to(gpu_or_cpu, non_blocking=True)
                if isinstance(labels_by_client, Tensor)
                else torch.as_tensor(labels_by_client, device=gpu_or_cpu)
            )
            client_gradients_trainable = vmap(
                grad_fn,
                in_dims=(None, 0, 0),
                randomness="different",
            )(trainable_params, inputs_by_client, labels_by_client)
    client_gradients = [None for _ in range(num_params)]
    for pos, grad in zip(trainable_pos, client_gradients_trainable):
        client_gradients[pos] = grad
    return client_gradients

def trainyolo(
    model: Module,
    data: DataLoader,
    optimizer: MetaOptimizer,
    criterion: Module,
    epochs: int,
) -> list:
    """Model training procedure for GIA.

    This training will create a computational graph through multiple steps, which is necessary
    for backpropagating to an input image.

    Requires a meta optimizer that performs step to a new set of parameters to keep a functioning
    graph.

    Training does not update the original model, but returns a norm of what the update would have been.
    """
    gpu_or_cpu = device("cuda" if cuda.is_available() else "cpu")
    model.to(gpu_or_cpu)
    patched_model = MetaModule(model)
    outputs = None
    for _ in range(epochs):
        for inputs, labels, _ in data:
            inputs, labels = inputs.to(gpu_or_cpu, non_blocking=True), (labels.to(gpu_or_cpu, non_blocking=True) if
                                                                        isinstance(labels, Tensor) else labels)
            outputs = patched_model(inputs, patched_model.parameters)
            loss = criterion(outputs, labels).sum()
            patched_model.parameters = optimizer.step(loss, patched_model.parameters)
    model_delta = OrderedDict((name, param - param_origin)
                                            for ((name, param), (name_origin, param_origin))
                                            in zip(patched_model.parameters.items(),
                                                    OrderedDict(model.named_parameters()).items()))
    return list(model_delta.values())

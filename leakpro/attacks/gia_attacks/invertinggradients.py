"""Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?."."""
import os
from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
from optuna.trial import Trial
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from leakpro.fl_utils.data_utils import CustomTensorDataset
from leakpro.fl_utils.data_utils import GiaImageExtension, GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train, train_nostep_vectorized
from leakpro.fl_utils.similarity_measurements import (
    cosine_similarity_weights,
    total_variation,
)
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.import_helper import Callable, Self
from leakpro.utils.logger import logger


@dataclass
class InvertingConfig:
    """Possible configs for the Inverting Gradients attack."""

    # total variation scale for smoothing the reconstructions after each iteration
    tv_reg: float = 1.0e-06
    # learning rate on the attack optimizer
    attack_lr: float = 0.1
    # iterations for the attack steps
    at_iterations: int = 8000
    # MetaOptimizer, see MetaSGD for implementation
    optimizer: object = field(default_factory=lambda: MetaSGD())
    # Client loss function
    criterion: object = field(default_factory=lambda: CrossEntropyLoss())
    # Data modality extension
    data_extension: object = field(default_factory=lambda: GiaImageExtension())
    # Number of epochs for the client attack
    epochs: int = 1
    # If True, attack optimization uses true labels from attacked batch.
    # If False, labels are replaced with dummy labels (label-unknown setting).
    label_known: bool = True
    # if to use median pool 2d on images, can improve attack on high higher resolution (100+)
    median_pooling: bool = False
    # if we compare difference only for top 10 layers with largest changes. Potentially good for larger models.
    top10norms: bool = False

class InvertingGradients(AbstractGIA):
    """Gradient inversion attack by Geiping et al."""

    def __init__(
        self: Self,
        model: Module,
        client_loader: DataLoader,
        data_mean: Tensor,
        data_std: Tensor,
        train_fn: Optional[Callable] = None,
        configs: Optional[InvertingConfig] = None,
        optuna_trial_data: list = None,
        observed_client_gradient: Optional[list[Tensor | None]] = None,
    ) -> None:
        super().__init__()
        self.original_model = model
        self.model = deepcopy(self.original_model)
        self.client_loader = client_loader
        self.train_fn = train_fn if train_fn is not None else train
        self.data_mean = data_mean
        self.data_std = data_std
        self.configs = configs if configs is not None else InvertingConfig()
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        # required for optuna to save the best hyperparameters
        self.attack_cache_folder_path = "leakpro_output/attacks/inverting_grad"
        os.makedirs(self.attack_cache_folder_path, exist_ok=True)
        # optuna trial data
        self.optuna_trial_data = optuna_trial_data

        # tabular
        self.is_tabular = False
        self._optimizer_prototype = deepcopy(self.configs.optimizer)
        self.observed_client_gradient = observed_client_gradient
        self.vectorized_last_client_losses: Optional[Tensor] = None
        self.vectorized_best_client_losses: Optional[Tensor] = None

        logger.info("Inverting gradient initialized.")

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Inverting gradients"
        reference_str = """Geiping, Jonas, et al. Inverting gradients-how easy is it to
            break privacy in federated learning? Neurips, 2020."""
        summary_str = ""
        detailed_str = ""
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare the attack.

        Args:
        ----
            self (Self): The instance of the class.

        Returns:
        -------
            None

        """
        self.model.eval()
        self.is_tabular = isinstance(self.configs.data_extension, GiaTabularExtension)
        if self.is_tabular:
            self._prepare_attack_tabular()
            return
        (
            self.client_loader,
            self.original,
            self.reconstruction,
            self.reconstruction_labels,
            self.reconstruction_loader
        ) = self.configs.data_extension.get_at_data(self.client_loader)
        self.reconstruction.requires_grad = True
        client_gradient = self.train_fn(
            self.model,
            self.client_loader,
            self.configs.optimizer,
            self.configs.criterion,
            self.configs.epochs,
        )
        self.client_gradient = [p.detach() for p in client_gradient]

    def run_attack(self:Self) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Run the attack and return the combined metric result.

        Returns
        -------
            GIAResults: Container for results on GIA attacks.

        """
        return self.generic_attack_loop(self.configs, self.gradient_closure, self.configs.at_iterations, self.reconstruction,
                                self.data_mean, self.data_std, self.configs.attack_lr, self.configs.median_pooling,
                                self.client_loader, self.reconstruction_loader, is_tabular=self.is_tabular)


    def gradient_closure(self: Self, optimizer: torch.optim.Optimizer) -> Callable:
        """Returns a closure function that performs a gradient descent step.

        The closure function computes the gradients, calculates the reconstruction loss,
        adds a total variation regularization term, and then performs backpropagation.
        """
        def closure() -> torch.Tensor:
            """Computes the reconstruction loss and performs backpropagation.

            This function zeroes out the gradients of the optimizer and the model,
            computes the gradient and reconstruction loss, logs the reconstruction loss,
            optionally adds a total variation term, performs backpropagation, and optionally
            modifies the gradient of the input image.

            Returns
            -------
                torch.Tensor: The reconstruction loss.

            """
            optimizer.zero_grad()
            self.model.zero_grad()
            if self.is_tabular:
                self.vectorized_last_client_losses = None
                rec_loss = self._tabular_reconstruction_loss()
                rec_loss.backward()
                return rec_loss

            gradient = self.train_fn(
                self.model,
                self.reconstruction_loader,
                self.configs.optimizer,
                self.configs.criterion,
                self.configs.epochs,
            )
            rec_loss = cosine_similarity_weights(gradient, self.client_gradient, self.configs.top10norms)
            tv_reg = (self.configs.tv_reg * total_variation(self.reconstruction))
            # Add the TV loss term to penalize large variations between pixels, encouraging smoother images.
            rec_loss += tv_reg
            rec_loss.backward()
            return rec_loss
        return closure

    def _new_meta_optimizer(self) -> object:
        """Return a fresh meta-optimizer with reset internal state."""
        return deepcopy(self._optimizer_prototype)

    def _prepare_attack_tabular(self: Self) -> None:
        """Prepare tabular attack tensors and observed gradients."""
        self.configs.data_extension.label_known = self.configs.label_known
        model_device = next(self.model.parameters()).device
        (
            self.client_loader,
            self.original,
            self.reconstruction,
            self.reconstruction_labels,
            self.reconstruction_loader
        ) = self.configs.data_extension.get_at_data(self.client_loader)

        self.reconstruction = self.reconstruction.to(model_device, non_blocking=True)
        if hasattr(self.model, "to_gia_space"):
            with torch.no_grad():
                if self.reconstruction.dim() == 2:
                    reconstruction_gia = self.model.to_gia_space(self.reconstruction)
                elif self.reconstruction.dim() == 3:
                    c, b, d = self.reconstruction.shape
                    reconstruction_flat = self.reconstruction.reshape(c * b, d)
                    reconstruction_gia = self.model.to_gia_space(reconstruction_flat).reshape(c, b, -1)
                else:
                    raise ValueError(
                        f"Unexpected tabular reconstruction rank {self.reconstruction.dim()} "
                        f"for shape {tuple(self.reconstruction.shape)}"
                    )
            self.reconstruction = reconstruction_gia.detach().clone()
        if isinstance(self.reconstruction_labels, list):
            self.reconstruction_labels = [
                label.to(model_device, non_blocking=True) if isinstance(label, Tensor) else label
                for label in self.reconstruction_labels
            ]
        self.reconstruction_loader = DataLoader(
            CustomTensorDataset(self.reconstruction, self.reconstruction_labels),
            batch_size=self.reconstruction_loader.batch_size or 32,
            shuffle=False,
            drop_last=self.reconstruction_loader.drop_last,
        )
        self.reconstruction.requires_grad = True
        if self.observed_client_gradient is not None:
            self.client_gradient = [p.detach() if p is not None else None for p in self.observed_client_gradient]
            return
        if self.reconstruction.dim() == 3:
            client_gradient = train_nostep_vectorized(
                self.model,
                self.client_loader,
                self._new_meta_optimizer(),
                self.configs.criterion,
                self.configs.epochs,
            )
        else:
            client_gradient = self.train_fn(
                self.model,
                self.client_loader,
                self._new_meta_optimizer(),
                self.configs.criterion,
                self.configs.epochs,
            )
        self.client_gradient = [p.detach() for p in client_gradient]

    def _tabular_reconstruction_loss(self: Self) -> Tensor:
        """Compute tabular reconstruction objective, optionally vectorized across clients."""
        if self.reconstruction.dim() == 2:
            gradient = self.train_fn(
                self.model,
                self.reconstruction_loader,
                self._new_meta_optimizer(),
                self.configs.criterion,
                self.configs.epochs,
            )
            return cosine_similarity_weights(gradient, self.client_gradient, self.configs.top10norms)

        if self.reconstruction.dim() != 3:
            raise ValueError(
                f"Tabular attack expects reconstruction rank 2 or 3, got {tuple(self.reconstruction.shape)}"
            )
        if self.client_gradient is None:
            raise ValueError("Missing observed gradients for vectorized tabular attack.")
        rec_grads_full = train_nostep_vectorized(
            self.model,
            self.reconstruction_loader,
            None, # self._new_meta_optimizer(),
            self.configs.criterion,
            self.configs.epochs,
        )

        if len(rec_grads_full) != len(self.client_gradient):
            raise ValueError(
                "Reconstructed and observed gradient lists must have the same length for vectorized tabular attack."
            )

        client_losses = []
        num_clients = int(self.reconstruction.shape[0])
        for c_idx in range(num_clients):
            rec_client = []
            obs_client = []
            for rec_grad, obs_grad in zip(rec_grads_full, self.client_gradient):
                if rec_grad is None or obs_grad is None:
                    continue
                rec_client.append(rec_grad[c_idx])
                obs_client.append(obs_grad[c_idx])
            if not rec_client:
                raise ValueError(
                    "No aligned parameter gradients available for vectorized tabular cosine loss."
                )
            client_losses.append(
                cosine_similarity_weights(rec_client, obs_client, self.configs.top10norms).reshape(())
            )
        per_client_losses = torch.stack(client_losses, dim=0)
        self.vectorized_last_client_losses = per_client_losses.detach()
        return per_client_losses.sum()

    def _configure_attack(self: Self, configs: dict) -> None:
        pass

    def suggest_parameters(self: Self, trial: Trial) -> None:
        """Suggest parameters to chose and range for optimization for the Inverting Gradient attack."""
        total_variation = trial.suggest_float("total_variation", 1e-8, 1e-1, log=True)
        attack_lr = trial.suggest_float("attack_lr", 1e-4, 100.0, log=True)
        median_pooling = trial.suggest_int("median_pooling", 0, 1)
        top10norms = trial.suggest_int("top10norms", 0, 1)
        self.configs.attack_lr = attack_lr
        self.configs.tv_reg = total_variation
        self.configs.median_pooling = median_pooling
        self.configs.top10norms = top10norms
        if self.optuna_trial_data is not None:
            trial_data_idx = trial.suggest_categorical(
                "trial_data",
                list(range(len(self.optuna_trial_data)))
            )
            self.client_loader = self.optuna_trial_data[trial_data_idx]
            logger.info(f"Next experiment on trial data idx: {trial_data_idx}")
        logger.info(f"Chosen parameters:\
                    total_variation: {total_variation} \
                    attack_lr: {attack_lr} \
                    median_pooling: {median_pooling} \
                    top10norms: {top10norms}")

    def reset_attack(self: Self, new_config:dict) -> None:  # noqa: ARG002
        """Reset attack to initial state."""
        self.best_loss = float("inf")
        self.best_reconstruction = None
        self.best_reconstruction_round = None
        self.model = deepcopy(self.original_model)
        super().reset_attack()
        self.prepare_attack()
        logger.info("Inverting attack reset to initial state.")

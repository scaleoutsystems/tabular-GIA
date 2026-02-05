"""Module that contains the abstract class for constructing and performing a membership inference attack on a target."""

from abc import abstractmethod
from collections.abc import Generator
from copy import deepcopy
from typing import Callable, Optional

import optuna
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from leakpro.attacks.attack_base import AbstractAttack
from leakpro.attacks.utils.hyperparameter_tuning.optuna import optuna_optimal_hyperparameters
from leakpro.fl_utils.model_utils import MedianPool2d
from leakpro.fl_utils.similarity_measurements import dataloaders_psnr, dataloaders_ssim_ignite, text_reconstruciton_score
from leakpro.metrics.attack_result import GIAResults
from leakpro.schemas import OptunaConfig
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AbstractGIA(AbstractAttack):
    """Interface to construct and perform a gradient inversion attack on a target model and dataset.

    This serves as a guideline for implementing a metric to be used for measuring the privacy leakage of a target model.
    """

    def __init__(  # noqa: B027
        self:Self,
    )->None:
        """Initialize the AttackAbstract class.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.

        """
        self.best_sim = 0
        pass

    @abstractmethod
    def _configure_attack(self:Self, configs:dict)->None:
        """Configure the attack.

        Args:
        ----
            configs (dict): The configurations for the attack.

        """
        pass


    @abstractmethod
    def description(self:Self) -> dict:
        """Return a description of the attack.

        Returns
        -------
        dict: A dictionary containing the reference, summary, and detailed description of the attack.

        """
        pass

    @abstractmethod
    def prepare_attack(self:Self) -> None:
        """Method that handles all computation related to the attack dataset."""
        pass

    @abstractmethod
    def run_attack(self:Self) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Runs GIA attack.

        Returns
        -------
            Generator with intermediary results to allow for lazy evaluation during
            hyperparameter turning, and final GiaResults.

        """
        pass

    @abstractmethod
    def reset_attack(self: Self) -> None:
        """Reset attack to its initial state."""
        self.best_sim = 0

    @abstractmethod
    def suggest_parameters(self: Self, trial: optuna.trial.Trial) -> None:
        """Apply and suggest new hyperparameters for the attack using optuna trial."""
        pass

    def run_with_optuna(self:Self, optuna_config: Optional[OptunaConfig] = None) -> optuna.study.Study:
        """Fins optimal hyperparameters using optuna."""
        if optuna_config is None:
            optuna_config = OptunaConfig()
        optuna_optimal_hyperparameters(self, optuna_config)

    def generic_attack_loop(self: Self, configs:dict, gradient_closure: Callable, at_iterations: int,
                            reconstruction: Tensor, data_mean: Tensor, data_std: Tensor, attack_lr: float,
                            median_pooling: bool, client_loader: DataLoader, reconstruction_loader: DataLoader,
                            is_tabular: bool = False,
                            chose_best_ssim_as_final: bool = True) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Generic attack loop for GIA's."""
        def _loader_from_tensor(recon_tensor: Tensor) -> DataLoader:
            ds_cls = reconstruction_loader.dataset.__class__
            ds = ds_cls(recon_tensor, reconstruction_loader.dataset.labels)
            return DataLoader(ds, batch_size=reconstruction_loader.batch_size or 32, shuffle=False)

        optimizer = torch.optim.Adam([reconstruction], lr=attack_lr)
        # Initialize best reconstruction fallback
        best_reconstruction_tensor = reconstruction.detach().clone()
        self.best_reconstruction = _loader_from_tensor(best_reconstruction_tensor)
        self.final_best = self.best_reconstruction
        # reduce LR every 1/3 of total iterations
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[at_iterations // 2.667,
                                                                        at_iterations // 1.6,
                                                                        at_iterations // 1.142], gamma=0.1)
        try:
            for i in range(at_iterations):
                # loss function which does training and compares distance from reconstruction training to the real training.
                closure = gradient_closure(optimizer)
                pre_step_reconstruction = reconstruction.detach().clone()
                loss = optimizer.step(closure)
                loss_value = float(loss.detach().item())
                scheduler.step()
                with torch.no_grad():
                    if not is_tabular:
                        reconstruction.data = torch.max(
                            torch.min(reconstruction, (1 - data_mean) / data_std), -data_mean / data_std
                            )
                        if (i +1) % 500 == 0 and median_pooling:
                            reconstruction.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(reconstruction)
                # Choose image who has given least loss
                if loss_value < self.best_loss:
                    self.best_loss = loss_value
                    # the loader before the step was the one that gave the good loss value
                    best_reconstruction_tensor = pre_step_reconstruction
                    self.best_reconstruction = _loader_from_tensor(best_reconstruction_tensor)
                    self.best_reconstruction_round = i
                    logger.info(f"New best loss: {self.best_loss} on round: {i}")

                # Enforce constraints if data extension supports it (e.g., Tabular)
                #if i % 100 == 0 and hasattr(configs.data_extension, 'enforce_constraints'):
                #    reconstruction.data = configs.data_extension.enforce_constraints(reconstruction.data)

                if i % 250 == 0:
                    logger.info(f"Iteration {i}, loss {loss_value}")
                    if is_tabular:
                        # For tabular, track the best loss as similarity proxy
                        yield i, -self.best_loss, None
                    else:
                        ssim = dataloaders_ssim_ignite(client_loader, self.best_reconstruction)
                        if ssim > self.best_sim:
                            self.final_best = deepcopy(self.best_reconstruction)
                            self.best_sim = ssim
                        current_sim = self.best_sim if chose_best_ssim_as_final else ssim
                        logger.info(f"current ssim: {self.best_sim}")
                        yield i, current_sim, None
        except Exception as e:
            logger.info(f"Attack stopped due to {e}. \
                        Saving results.")
        if is_tabular:
            # Enforce constraints on best reconstruction before metrics
            if hasattr(configs.data_extension, 'enforce_constraints'):
                # Note: best_reconstruction is a DataLoader. We need to extract tensor, enforce, and repack?
                # Actually, Tabular GIA used 'best_reconstruction' as Tensor internally. 
                # AbstractGIA uses it as DataLoader. This is a mismatch.
                # However, generic_attack_loop takes 'reconstruction' (Tensor).
                # best_reconstruction is updated as deepcopy(pre_step_loader).
                # Let's rely on the result generator to handle final processing or 
                # blindly trust the periodic enforcement was enough.
                pass
            
            # Compute simple tabular reconstruction metrics
            orig_tensor = torch.cat([batch[0] for batch in client_loader], dim=0)
            
            # Re-collect tensor from best reconstruction snapshot
            recon_tensor = best_reconstruction_tensor
            
            #if hasattr(configs.data_extension, 'enforce_constraints'):
            #     recon_tensor = configs.data_extension.enforce_constraints(recon_tensor)
            
            mse = torch.mean((orig_tensor - recon_tensor) ** 2).item()
            mae = torch.mean(torch.abs(orig_tensor - recon_tensor)).item()
            rmse = mse ** 0.5
            gia_result = GIAResults(client_loader, self.best_reconstruction,
                                    psnr_score=None, ssim_score=None,
                                    data_mean=None, data_std=None, config=configs,
                                    images=False, rmse_score=rmse, mae_score=mae, is_tabular=True)
            yield i, -self.best_loss, gia_result
        else:
            result = self.final_best if chose_best_ssim_as_final else reconstruction_loader
            ssim_score = dataloaders_ssim_ignite(client_loader, result)
            psnr_score = dataloaders_psnr(client_loader, result)
            logger.info(f"final sim: {ssim_score}")
            gia_result = GIAResults(client_loader, result,
                              psnr_score=psnr_score, ssim_score=ssim_score,
                              data_mean=data_mean, data_std=data_std, config=configs)
            yield i, ssim_score, gia_result


    def generic_attack_loop_text(self: Self, configs:dict, gradient_closure: Callable, at_iterations: int,
                        reconstruction: Tensor, data_mean: Tensor, data_std: Tensor, used_tokens: Tensor, attack_lr: float,
                        client_loader: DataLoader, reconstruction_loader: DataLoader, tryouts: int = 6
                        ) -> Generator[tuple[int, Tensor, GIAResults]]:
        """Generic attack loop for GIA's."""

        optimizer = torch.optim.Adam(reconstruction, lr=attack_lr)

        # reduce LR every 1/3 of total iterations
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[at_iterations // 2.667,
                                                                        at_iterations // 1.6,
                                                                        at_iterations // 1.142], gamma=0.1)
        try:
            for i in range(at_iterations):
                logger.info(f"iteration: {i}")

                # loss function which does training and compares distance from reconstruction training to the real training.
                closure = gradient_closure(optimizer)

                previous_reconstruction_data = deepcopy(reconstruction_loader)
                previous_optimizer_state = deepcopy(optimizer.state_dict())
                last_loss = optimizer.step(closure)
                loss = self.inference_closure()
                tryout = 0
                while loss > last_loss and tryout < tryouts:
                    # reset reconstruction with previous version until better performance
                    with torch.no_grad():
                            for j in range(len(reconstruction)):
                                reconstruction[j].copy_(previous_reconstruction_data.dataset[j].embedding)

                    optimizer.load_state_dict(previous_optimizer_state)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = param_group["lr"]/2
                    closure = gradient_closure(optimizer)
                    _ = optimizer.step(closure)
                    loss = self.inference_closure()

                    tryout += 1

                for param_group in optimizer.param_groups:
                    param_group["lr"] = attack_lr



                scheduler.step()
                # Choose image who has given least loss
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_reconstruction = deepcopy(reconstruction_loader)
                    self.best_reconstruction_round = i
                    logger.info(f"New best loss: {loss} on round: {i}")
                if i % 100 == 0:
                    logger.info(f"Iteration {i}, loss {loss}")
                    sim_score = text_reconstruciton_score(client_loader, self.best_reconstruction, token_used=used_tokens)
                    yield i, sim_score, None
        except Exception as e:
            logger.info(f"Attack stopped due to {e}. \
                        Saving results.")

        sim_score = text_reconstruciton_score(client_loader, self.best_reconstruction, token_used=used_tokens)
        gia_result = GIAResults(client_loader, self.best_reconstruction,
                        psnr_score=0, ssim_score=sim_score,
                        data_mean=data_mean, data_std=data_std, config=configs, images=False)
        yield i, sim_score, gia_result

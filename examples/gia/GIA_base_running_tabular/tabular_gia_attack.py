"""Tabular gradient inversion attack for federated learning."""

from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from collections.abc import Generator

from leakpro.attacks.gia_attacks.abstract_gia import AbstractGIA
from tabular_metrics import evaluate_reconstruction, count_constraint_violations
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.metrics.attack_result import GIAResults
from leakpro.utils.logger import logger
import optuna


class TabularGIA(AbstractGIA):
    """Optimization-based gradient inversion attack for tabular data.
    
    Implements DLG-style gradient matching with tabular-specific constraints.
    Uses GiaTabularExtension for data handling and initialization.
    """
    
    def __init__(
        self,
        model: Module,
        client_loader: DataLoader,
        criterion: Module,
        config: Optional[Dict] = None,
        data_extension: Optional[GiaTabularExtension] = None
    ):
        """Initialize tabular GIA."""
        super().__init__()
        
        self.model = model
        self.client_loader = client_loader
        self.criterion = criterion
        self.config = config or {}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Use provided extension or default
        self.data_extension = data_extension or GiaTabularExtension()
        
        # Attack hyperparameters
        self.attack_lr = self.config.get('attack_lr', 0.1)
        self.iterations = self.config.get('iterations', 1000)
        self.regularization_weight = self.config.get('reg_weight', 0.01)
        self.label_known = self.config.get('label_known', True)
        
        # Best reconstruction tracking
        self.best_loss = float('inf')
        self.best_reconstruction = None
        self.best_score = 0.0
        
        # Get feature info for constraints if available
        # In this new flow, we expect config or data_extension to potentially carry it, or pass it explicitly
        # We will check if configured
        self.feature_info = self.config.get('encoder_meta', None)


    def _enforce_constraints(self, data: Tensor) -> Tensor:
        """Project data to valid space (one-hot for categorical)."""
        if self.feature_info is None:
            return data
            
        # Using tabular.py metadata structure
        # meta['cat_dummy_cols'] list of all one-hot cols
        # meta['cat_categories'] dict of {col: [cats]}
        # meta['num_cols'] list of num cols
        
        # If we don't have cat_dummy_cols, we can't easily project.
        if 'cat_dummy_cols' not in self.feature_info:
            return data

        with torch.no_grad():
            data_np = data.detach().cpu().numpy()
            
            num_cols = len(self.feature_info.get('num_cols', []))
            cat_dummy_cols = self.feature_info.get('cat_dummy_cols', [])
            
            # Map original categorical features to their one-hot indices
            # cat_dummy_cols are like ['workclass_Private', 'workclass_Self-emp', ...]
            # We need to group them by prefix or using cat_categories to know which belong together
            
            current_idx = num_cols
            cat_categories = self.feature_info.get('cat_categories', {})
            
            # Constraint 1: Clip numerical features to reasonable range (normalized)
            if num_cols > 0:
                data_np[:, :num_cols] = np.clip(data_np[:, :num_cols], -5.0, 5.0)

            # Constraint 2: One-hot projection for categoricals
            # Iterate through keys in cat_categories (original feature names)
            for cat_feat, cats in cat_categories.items():
                num_cats = len(cats)
                # Indices for this feature
                indices = list(range(current_idx, current_idx + num_cats))
                
                if current_idx + num_cats > data_np.shape[1]:
                    break # Safety check
                
                # Extract and project
                onehot_vectors = data_np[:, indices]
                max_indices = onehot_vectors.argmax(axis=1)
                
                data_np[:, indices] = 0.0
                for sample_idx, max_idx in enumerate(max_indices):
                    data_np[sample_idx, indices[max_idx]] = 1.0
                    
                current_idx += num_cats
            
            data.copy_(torch.tensor(data_np, dtype=torch.float32, device=data.device))
        
        return data

    def run_attack(self) -> Generator[tuple[int, float, Optional[GIAResults]]]:
        """Run gradient inversion attack."""
        # 1. Prepare data using extension (handles Prior Initialization)
        (
            self.org_loader, 
            self.original, 
            self.reconstruction, 
            self.reconstruction_labels, 
            self.reconstruction_loader
        ) = self.data_extension.get_at_data(self.client_loader, feature_meta=self.feature_info)
        
        self.reconstruction = self.reconstruction.to(self.device).requires_grad_(True)
        self.reconstruction_labels = [l.to(self.device) for l in self.reconstruction_labels]
        
        stacked = torch.stack(self.reconstruction_labels).float()
        if stacked.dim() == 1:
             dummy_labels = stacked.unsqueeze(1)
        else:
             dummy_labels = stacked
             
        # Get true gradients (target)
        self.model.zero_grad()
        true_outputs = self.model(self.original.to(self.device))
        if self.label_known:
            # Assuming binary classification for Adult
             true_loss = self.criterion(true_outputs, dummy_labels)
        else:
             print("Warning: Label unknown cases not fully implemented for Adult check")
             true_loss = self.criterion(true_outputs, torch.zeros_like(true_outputs)) # Placeholder

        self.observed_gradients = torch.autograd.grad(true_loss.sum(), self.model.parameters())
        self.observed_gradients = [g.detach() for g in self.observed_gradients]
        
        # Optimizer
        optimizer = torch.optim.Adam([self.reconstruction], lr=self.attack_lr)
        
        for iteration in range(self.iterations):
            optimizer.zero_grad()
            self.model.zero_grad()
            
            # Forward pass
            dummy_outputs = self.model(self.reconstruction)
            dummy_loss = self.criterion(dummy_outputs, dummy_labels)
            
            # Compute gradients
            dummy_gradients = torch.autograd.grad(
                dummy_loss.sum(),
                self.model.parameters(),
                create_graph=True
            )
            
            # Gradient matching loss
            grad_loss = 0.0
            for dummy_grad, observed_grad in zip(dummy_gradients, self.observed_gradients):
                grad_loss += ((dummy_grad - observed_grad) ** 2).sum()
                
            # Regularization
            reg_loss = (self.reconstruction ** 2).mean()
            total_loss = grad_loss + self.regularization_weight * reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Constraints
            if iteration % 10 == 0:
                self.reconstruction.data = self._enforce_constraints(self.reconstruction)
                
            # Track best
            if total_loss.item() < self.best_loss:
                self.best_loss = total_loss.item()
                self.best_reconstruction = self.reconstruction.detach().clone()
                
            # Log
            if iteration % 100 == 0:
                # Quick metric check if feature info available
                score = 0.0
                if self.feature_info:
                     metrics = evaluate_reconstruction(
                         self.original.to(self.device), 
                         self.best_reconstruction.to(self.device), 
                         self.feature_info, 
                         return_per_feature=False
                     )
                     score = metrics['aggregate']['overall_score']
                     self.best_score = score
                     
                logger.info(f"Iteration {iteration}, Loss: {total_loss.item():.4f}, Score: {score:.4f}")
                yield iteration, score, None
                
        # Final result
        final_recon = self._enforce_constraints(self.best_reconstruction)
        
        results = None
        if self.feature_info:
            full_metrics = evaluate_reconstruction(
                self.original.to(self.device), 
                final_recon.to(self.device), 
                self.feature_info, 
                return_per_feature=True
            )
            violations = count_constraint_violations(final_recon, self.feature_info)
            results = {
                'reconstruction_metrics': full_metrics,
                'constraint_violations': violations,
                'final_loss': self.best_loss
            }
        
        logger.info(f"Attack complete. Final score: {self.best_score:.4f}")
        yield self.iterations, self.best_score, results

    def prepare_attack(self) -> None:
        pass
    
    def reset_attack(self) -> None:
        super().reset_attack()
        self.best_loss = float('inf')
        self.best_reconstruction = None
        
    def _configure_attack(self, configs: dict) -> None:
        self.config.update(configs)
        
    def description(self) -> dict:
        return {'summary': 'Tabular GIA with constraints'}

    def suggest_parameters(self, trial: optuna.trial.Trial) -> None:
        self.attack_lr = trial.suggest_float('attack_lr', 0.01, 1.0, log=True)

"""Simple FedSGD simulator for gradient extraction in federated learning."""

from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.fl_utils.gia_optimizers import MetaOptimizer


class FedSGDSimulator:
    """Simulates FedSGD protocol to extract client gradients for GIA.
    
    In FedSGD, clients compute gradients on their local data and send these
    gradients to the server. An honest-but-curious server can observe these
    gradients and attempt reconstruction.
    """
    
    def __init__(
        self,
        model: Module,
        criterion: Module,
        optimizer: MetaOptimizer
    ):
        """Initialize FedSGD simulator.
        
        Args:
            model: Global model
            criterion: Loss function
            optimizer: Meta optimizer for gradient computation
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
    def compute_client_gradients(
        self,
        client_loader: DataLoader,
        epochs: int = 1
    ) -> List[Tensor]:
        """Compute gradients for a client's data batch.
        
        Args:
            client_loader: DataLoader with client's local data
            epochs: Number of local epochs (typically 1 for FedSGD)
            
        Returns:
            List of gradient tensors (one per model parameter)
        """
        # For tabular data with BCEWithLogitsLoss, we need to handle label shapes
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        for _ in range(epochs):
            for inputs, labels in client_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Reshape labels for BCEWithLogitsLoss if needed
                # Binary classification with BCE needs labels shape (batch, 1)
                if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                    labels = labels.float().unsqueeze(1)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                if loss.dim() > 0:
                    loss = loss.sum()
                
                # Compute gradients
                gradients = torch.autograd.grad(
                    loss,
                    list(self.model.parameters()),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )
        
        return list(gradients)
    
    def simulate_round(
        self,
        client_loaders: List[DataLoader],
        aggregate: bool = True
    ) -> Tuple[List[List[Tensor]], List[Tensor]]:
        """Simulate one round of FedSGD.
        
        Args:
            client_loaders: List of client DataLoaders
            aggregate: If True, aggregate gradients; otherwise return individual gradients
            
        Returns:
            Tuple of (individual_gradients, aggregated_gradients)
            - individual_gradients: List of gradient lists (one per client)
            - aggregated_gradients: Averaged gradients across clients
        """
        all_client_gradients = []
        
        # Compute gradients for each client
        for client_loader in client_loaders:
            client_grads = self.compute_client_gradients(client_loader)
            all_client_gradients.append(client_grads)
        
        if not aggregate:
            return all_client_gradients, None
        
        # Aggregate gradients (simple averaging for FedSGD)
        aggregated_gradients = []
        num_clients = len(all_client_gradients)
        
        for param_idx in range(len(all_client_gradients[0])):
            # Average gradients for this parameter across all clients
            param_grads = [client_grads[param_idx] for client_grads in all_client_gradients]
            avg_grad = torch.stack(param_grads).mean(dim=0)
            aggregated_gradients.append(avg_grad)
        
        return all_client_gradients, aggregated_gradients
    
    def extract_client_gradient(
        self,
        client_loader: DataLoader
    ) -> List[Tensor]:
        """Convenience method to extract gradient from a single client.
        
        This is what an honest-but-curious server would observe.
        
        Args:
            client_loader: Single client's DataLoader
            
        Returns:
            List of gradient tensors
        """
        return self.compute_client_gradients(client_loader, epochs=1)

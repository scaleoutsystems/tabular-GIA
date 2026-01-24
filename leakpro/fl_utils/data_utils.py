"""Util functions relating to data."""
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor, cat, mean, randn, std
from torch.utils.data import DataLoader, Dataset

from leakpro.utils.import_helper import Any, Self


class GiaDataModalityExtension(ABC):
    """Abstract class for data modality extensions for GIA."""

    @abstractmethod
    def get_at_data() -> Tensor:
        """Get a dataloader mimicing the shape of the original data used for recreating."""
        pass

class CustomTensorDataset(Dataset):
    """Custom generic tensor dataset."""

    def __init__(self:Self, reconstruction: torch.Tensor, labels: list) -> None:
        self.reconstruction = reconstruction
        self.labels = labels

    def __len__(self: Self) -> int:
        """Dataset length."""
        return self.reconstruction.size(0)

    def __getitem__(self: Self, index: int) -> tuple[Tensor, Any]:
        """Get item from index."""
        return self.reconstruction[index], self.labels[index]

class CustomYoloTensorDataset(Dataset):
    """Custom generic tensor dataset."""

    def __init__(self:Self, reconstruction: torch.Tensor, labels: list) -> None:
        self.reconstruction = reconstruction
        self.labels = labels

    def __len__(self: Self) -> int:
        """Dataset length."""
        return self.reconstruction.size(0)

    def __getitem__(self: Self, index: int) -> tuple[Tensor, Any]:
        """Get item from index."""
        return self.reconstruction[index], self.labels[index], 1

class GiaImageExtension(GiaDataModalityExtension):
    """Image extension for GIA."""

    def get_at_data(self: Self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same COCO labels."""
        original = torch.stack([
            img.clone()
            for img, _ in client_loader.dataset
        ], dim=0)
        org_labels = []
        for _, label in client_loader:
            if isinstance(label, Tensor):
                org_labels.extend(deepcopy(label))
            else:
                org_labels.append(deepcopy(label))

        # 3) re-make your dataset & loader
        org_dataset = CustomTensorDataset(original, org_labels)
        org_loader  = DataLoader(org_dataset, batch_size=32, shuffle=True)
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        labels = []
        for _, label in client_loader:
            if isinstance(label, Tensor):
                labels.extend(deepcopy(label))
            else:
                labels.append(deepcopy(label))
        reconstruction_dataset = CustomTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)


        return org_loader, original, reconstruction, labels, reconstruction_loader

class GiaImageYoloExtension(GiaDataModalityExtension):
    """Image extension for GIA."""

    def get_at_data(self: Self, client_loader: DataLoader) -> DataLoader:
        """DataLoader with random noise images of the same shape as the client_loader's dataset, using the same COCO labels."""
        img_shape = client_loader.dataset[0][0].shape
        num_images = len(client_loader.dataset)
        reconstruction = randn((num_images, *img_shape))
        labels = []
        for _, label, _ in client_loader:
            if isinstance(label, Tensor):
                labels.extend(deepcopy(label))
            else:
                labels.append(deepcopy(label))
        reconstruction_dataset = CustomYoloTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=32, shuffle=True)
        original = torch.stack([
            img.clone()
            for img, _, _ in client_loader.dataset
        ], dim=0)
        org_labels = []
        for _, label, _ in client_loader:
            if isinstance(label, Tensor):
                org_labels.extend(deepcopy(label))
            else:
                org_labels.append(deepcopy(label))
        org_dataset = CustomYoloTensorDataset(original, org_labels)
        org_loader = DataLoader(org_dataset, batch_size=32, shuffle=True)
        return org_loader, original, reconstruction, labels, reconstruction_loader

class GiaTabularExtension(GiaDataModalityExtension):
    """Tabular extension for GIA (minimal, tensor-only)."""

    def get_at_data(self: Self, client_loader: DataLoader) -> tuple[DataLoader, Tensor, Tensor, list, DataLoader]:
        """Create reconstruction tensors matching tabular samples and labels.
        Assumes client_loader yields (features, labels) where features are numeric tensors.
        """
        batch_size = client_loader.batch_size or 32
        originals = []
        labels: list = []
        for batch in client_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                feats, lbls = batch[0], batch[1]
            else:  # fallback: treat entire batch as features with dummy labels
                feats, lbls = batch, torch.zeros(len(batch))
            originals.append(feats.clone())

            if isinstance(lbls, Tensor):
                labels.extend(deepcopy(lbls))
            else:
                labels.extend(deepcopy(lbls))

        original = torch.cat(originals, dim=0)
        reconstruction = torch.randn_like(original)
        org_dataset = CustomTensorDataset(original, labels)
        org_loader = DataLoader(org_dataset, batch_size=batch_size, shuffle=True)
        reconstruction_dataset = CustomTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=batch_size, shuffle=True)

        return org_loader, original, reconstruction, labels, reconstruction_loader

class GiaTabularExtension2(GiaDataModalityExtension):
    """Tabular extension for GIA (minimal, tensor-only)."""

    def get_at_data(self: Self, client_loader: DataLoader, feature_meta: dict = None) -> tuple[DataLoader, Tensor, Tensor, list, DataLoader]:
        """Create reconstruction tensors matching tabular samples and labels.

        Args:
            client_loader: DataLoader providing original data shapes
            feature_meta: Optional dictionary containing feature metadata (means, stds, categories)
                          for smart initialization.
        """
        batch_size = client_loader.batch_size or 32

        # Use injected metadata if not provided as argument
        if feature_meta is not None:
            self.feature_meta = feature_meta
        else:
            feature_meta = getattr(self, 'feature_meta', None)

        originals = []
        labels: list = []
        for batch in client_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                feats, lbls = batch[0], batch[1]
            else:  # fallback: treat entire batch as features with dummy labels
                feats, lbls = batch, torch.zeros(len(batch))

            originals.append(feats.clone())
            if isinstance(lbls, Tensor):
                labels.extend(deepcopy(lbls))
            else:
                labels.extend(deepcopy(lbls))

        original = torch.cat(originals, dim=0)
        
        # Smart initialization if metadata is provided
        if feature_meta:
            reconstruction_np = np.zeros_like(original.cpu().numpy())
            n_samples = reconstruction_np.shape[0]
            
            # 1. Numerical Initialization (Gaussian around mean/std)
            if 'num_mean' in feature_meta and 'num_std' in feature_meta:
                means = feature_meta['num_mean']
                stds = feature_meta['num_std']
                
                # Convert to numpy if tensor
                if isinstance(means, torch.Tensor):
                    means = means.cpu().numpy()
                elif hasattr(means, 'values'): # Pandas
                    means = means.values
                elif isinstance(means, list):
                    means = np.array(means)
                    
                if isinstance(stds, torch.Tensor):
                    stds = stds.cpu().numpy()
                elif hasattr(stds, 'values'):
                    stds = stds.values
                elif isinstance(stds, list):
                    stds = np.array(stds)
                
                num_cols = len(means) # Now safe
                # Assuming numerical columns are first in the concatenation (standard in tabular.py)
                for i in range(num_cols):
                    reconstruction_np[:, i] = np.random.normal(means[i], stds[i], size=n_samples)
            
            # 2. Categorical Initialization (Frequency based)
            # We expect feature_meta to potentially contain 'cat_frequencies': { 'col_name': [prob_cat1, prob_cat2, ...] }
            cat_cols = feature_meta.get('cat_cols', [])
            cat_frequencies = feature_meta.get('cat_frequencies', {})
            cat_categories = feature_meta.get('cat_categories', {})
            
            # Current index tracks where we are in the flattened feature vector
            # Assuming numerical columns are first (handled above)
            current_idx = len(feature_meta.get('num_cols', []))
            
            for col in cat_cols:
                # Get categories for this column
                cats = cat_categories.get(col, [])
                if isinstance(cats, dict): cats = cats.get('categories', []) # Handle old format if any
                n_cats = len(cats)
                
                # Indices in the one-hot vector
                indices = list(range(current_idx, current_idx + n_cats))
                
                if current_idx + n_cats > reconstruction_np.shape[1]:
                    break
                
                # Check for frequencies
                probs = cat_frequencies.get(col, None)
                
                # If we have valid probabilities, use them
                if probs is not None and len(probs) == n_cats:
                    # Normalize if needed
                    probs = np.array(probs)
                    msg_sum = probs.sum()
                    if msg_sum > 0:
                        probs = probs / msg_sum
                    else:
                        probs = None
                
                # Sample
                if probs is not None:
                    # Sample generic indices [0, ..., n_cats-1] based on probs
                    choices = np.random.choice(n_cats, size=n_samples, p=probs)
                else:
                    # Uniform random fallback
                    choices = np.random.randint(0, n_cats, size=n_samples)
                
                # Set one-hot
                reconstruction_np[:, indices] = 0
                for i, choice in enumerate(choices):
                    reconstruction_np[i, indices[choice]] = 1.0
                    
                current_idx += n_cats
                
            # Fill any remaining with random noise (safety)
            if current_idx < reconstruction_np.shape[1]:
                 reconstruction_np[:, current_idx:] = np.random.randn(n_samples, reconstruction_np.shape[1] - current_idx)

            reconstruction = torch.tensor(reconstruction_np, dtype=torch.float32)
        else:
            reconstruction = torch.randn_like(original)

        org_dataset = CustomTensorDataset(original, labels)
        org_loader = DataLoader(org_dataset, batch_size=batch_size, shuffle=True)

        reconstruction_dataset = CustomTensorDataset(reconstruction, labels)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=batch_size, shuffle=True)

        return org_loader, original, reconstruction, labels, reconstruction_loader

    def enforce_constraints(self: Self, data: Tensor) -> Tensor:
        """Project data to valid space (one-hot for categorical, clipped for numerical)."""
        feature_meta = getattr(self, 'feature_meta', None)
        if feature_meta is None:
            return data
            
        # Using tabular.py metadata structure
        if 'cat_dummy_cols' not in feature_meta:
            return data

        with torch.no_grad():
            data_np = data.detach().cpu().numpy()
            
            num_cols = len(feature_meta.get('num_cols', []))
            
            # Constraint 1: Clip numerical features to reasonable range (normalized)
            if num_cols > 0:
                data_np[:, :num_cols] = np.clip(data_np[:, :num_cols], -5.0, 5.0)

            # Constraint 2: One-hot projection for categoricals
            current_idx = num_cols
            cat_categories = feature_meta.get('cat_categories', {})
            
            # Iterate through keys in cat_categories
            for cat_feat, cats in cat_categories.items():
                if isinstance(cats, dict): cats = cats.get('categories', []) # Handle old format
                num_cats = len(cats)
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


def get_meanstd(trainset: Dataset, axis_to_reduce: tuple=(-2,-1)) -> tuple[Tensor, Tensor]:
    """Get mean and std of a dataset."""
    cc = cat([trainset[i][0].unsqueeze(0) for i in range(len(trainset))], dim=0)
    cc = cc.float()
    axis_to_reduce += (0,)
    data_mean = mean(cc, dim=axis_to_reduce).tolist()
    data_std = std(cc, dim=axis_to_reduce).tolist()
    return data_mean, data_std

def get_used_tokens(model: torch.nn.Module, client_gradient: Tensor) -> np.array:
    """Get used tokens."""

    # searching for layer index corresponding to the embedding layer
    for i, name in enumerate(model.named_parameters()):
        if name[0] == "embedding_layer.weight":
            embedding_layer_idx = i


    upd_embedding = client_gradient[embedding_layer_idx].detach().cpu().numpy()

    diff = np.sum(abs(upd_embedding),0)
    return np.where(diff>0)[0]

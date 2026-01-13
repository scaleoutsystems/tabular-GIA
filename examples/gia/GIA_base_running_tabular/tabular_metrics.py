"""Evaluation metrics for tabular gradient inversion attacks."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error


def normalized_rmse(true_values: np.ndarray, reconstructed_values: np.ndarray) -> float:
    """Compute normalized RMSE for numerical features."""
    rmse = np.sqrt(mean_squared_error(true_values, reconstructed_values))
    value_range = true_values.max() - true_values.min() 
    
    if value_range == 0:
        return 0.0
    
    return rmse / value_range


def categorical_accuracy(true_onehot: np.ndarray, reconstructed_onehot: np.ndarray) -> float:
    """Compute accuracy for one-hot encoded categorical features."""
    true_labels = np.argmax(true_onehot, axis=1)
    reconstructed_labels = np.argmax(reconstructed_onehot, axis=1)
    return accuracy_score(true_labels, reconstructed_labels)


def evaluate_reconstruction(
    ground_truth: Tensor,
    reconstructed: Tensor,
    feature_info: Dict,
    return_per_feature: bool = True
) -> Dict:
    """Evaluate reconstruction quality for tabular data."""
    gt_np = ground_truth.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    results = {}
    
    numeric_scores = []
    if return_per_feature: results['numerical'] = {}
    
    # Generic metadata support
    # Try keys: 'numerical_features' (old) OR 'num_cols' (new generic)
    num_feats = feature_info.get('numerical_features') or feature_info.get('num_cols', [])
    cat_feats = feature_info.get('categorical_features') or feature_info.get('cat_cols', [])
    
    # Mappings
    # Old: dec_to_onehot[i]
    # New: We need to infer indices.
    # In tabular.py: num cols come first, then cat cols.
    
    current_idx = 0
    # Numerical
    for i, feature in enumerate(num_feats):
        # Allow old style detailed mapping if present
        if 'dec_to_onehot' in feature_info:
             # Need to find index. If feature_info has old structure, use it.
             # Assuming old structure aligns numericals first 0..N
             # We rely on specific structure.
             
             # If Generic (tabular.py):
             # Num cols are always single column at the start (if normalized/scaled)
             indices = [current_idx]
             current_idx += 1
        else:
             # Assume Tabular.py generic structure: Num cols are 1:1 at start
             indices = [current_idx]
             current_idx += 1
             
        nrmse = normalized_rmse(gt_np[:, indices[0]], recon_np[:, indices[0]])
        numeric_scores.append(nrmse)
        if return_per_feature:
            results['numerical'][feature] = {'nrmse': float(nrmse)}
            
    cat_scores = []
    if return_per_feature: results['categorical'] = {}
    
    # Categorical
    # Need to know how many columns each cat feature takes.
    # Tabular.py: 'cat_categories' dict tells us this.
    cat_categories = feature_info.get('category_mappings') or feature_info.get('cat_categories', {})
    
    for i, feature in enumerate(cat_feats):
        # Find categories
        if feature in cat_categories:
             cats = cat_categories[feature]
             # Tabular.py stores list of cats.
             # Old util stored dict with 'categories' key.
             if isinstance(cats, dict): cats = cats['categories']
             
             num_cats = len(cats)
             indices = list(range(current_idx, current_idx + num_cats))
             current_idx += num_cats
             
             acc = categorical_accuracy(gt_np[:, indices], recon_np[:, indices])
             cat_scores.append(acc)
             if return_per_feature:
                results['categorical'][feature] = {'accuracy': float(acc)}
            
    # Aggregate
    num_score = 1.0 - np.mean(numeric_scores) if numeric_scores else 0.5
    cat_score = np.mean(cat_scores) if cat_scores else 0.5
    overall = 0.5 * num_score + 0.5 * cat_score
    
    results['aggregate'] = {
        'numerical_score': float(num_score),
        'categorical_score': float(cat_score),
        'overall_score': float(overall)
    }
    
    return results


def count_constraint_violations(reconstructed: Tensor, feature_info: Dict) -> Dict:
    """Count invalid one-hot encodings."""
    recon_np = reconstructed.detach().cpu().numpy()
    violations = 0
    
    cat_feats = feature_info.get('categorical_features') or feature_info.get('cat_cols', [])
    total = reconstructed.shape[0] * len(cat_feats)
    
    # Calculate offset for categorical columns
    num_feats = feature_info.get('numerical_features') or feature_info.get('num_cols', [])
    current_idx = len(num_feats)
    
    cat_categories = feature_info.get('category_mappings') or feature_info.get('cat_categories', {})
    
    for feature in cat_feats:
        if feature in cat_categories:
             cats = cat_categories[feature]
             if isinstance(cats, dict): cats = cats['categories']
             
             num_cats = len(cats)
             indices = list(range(current_idx, current_idx + num_cats))
             current_idx += num_cats
             
             vecs = recon_np[:, indices]
             
             # Check sum close to 1 and max close to 1
             sums = vecs.sum(axis=1)
             maxs = vecs.max(axis=1)
        
             invalid = np.logical_or(np.abs(sums - 1.0) > 0.1, maxs < 0.5)
             violations += invalid.sum()
        
    return {'total_violations': int(violations), 'total_checks': total}

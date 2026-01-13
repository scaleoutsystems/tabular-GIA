"""Minimal tabular GIA demo using a pre-trained global model."""

import sys
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.utils.seed import seed_everything

# Local imports
# Local imports
from tabular_metrics import evaluate_reconstruction, count_constraint_violations

from train import train_global_model
from model import TabularMLP
from tabular import get_tabular_loaders, load_tabular_config

from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients, InvertingConfig


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def compute_prior_stats(loader: DataLoader, encoder_meta: dict) -> dict:
    """Compute categorical frequencies from a loader (Prior Knowledge)."""
    cat_cols = encoder_meta.get('cat_cols', [])
    cat_frequencies = {}
    
    if not cat_cols:
        return {}
    
    logger.info("Computing prior statistics (frequencies) from training data...")
    
    num_cols = encoder_meta.get('num_cols', [])
    cat_categories = encoder_meta.get('cat_categories', {})
    
    # Initialize counts
    counts = {col: np.zeros(len(cat_categories.get(col, []))) for col in cat_cols}
    total_samples = 0
    
    start_idx = len(num_cols)
    
    for batch in loader:
        features = batch[0] # [B, D]
        features_np = features.numpy()
        total_samples += features.shape[0]
        
        curr = start_idx
        for col in cat_cols:
            cats = cat_categories.get(col, [])
            n_cats = len(cats)
            indices = list(range(curr, curr + n_cats))
            
            # Sum up one-hot vectors
            # features_np[:, indices] is [B, n_cats]
            batch_counts = features_np[:, indices].sum(axis=0)
            if col in counts:
                 counts[col] += batch_counts
            
            curr += n_cats
            
    # Normalize
    for col in counts:
        if total_samples > 0:
            counts[col] /= total_samples
        cat_frequencies[col] = counts[col].tolist()
        
    return cat_frequencies


def load_config() -> tuple[dict, Path]:
    config_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = load_tabular_config(config_path)
    data_path = Path(cfg["data_path"])
    dataset_dir = data_path.parent
    trainer_cfg = cfg.get("trainer", {}) or {}
    ckpt_dir_cfg = trainer_cfg.get("checkpoint_dir")
    ckpt_dir = Path(ckpt_dir_cfg) if ckpt_dir_cfg else dataset_dir / "checkpoints"
    if not ckpt_dir.is_absolute():
        ckpt_dir = dataset_dir / ckpt_dir
    ckpt_path = ckpt_dir / f"{data_path.stem}.ckpt"
    logger.info("Loaded config from %s", config_path)
    logger.info("Expecting checkpoint at %s", ckpt_path)
    return cfg, ckpt_path


def load_model(
    ckpt_path: Path,
    default_cfg: dict,
) -> tuple[TabularMLP, dict, torch.Tensor | None, torch.Tensor | None, dict | None]:
    """Load a pre-trained global model; assumes checkpoint exists."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or default_cfg
    meta = ckpt.get("meta", {})
    feat_dim = meta.get("feature_dim")
    n_classes = meta.get("num_classes")
    if feat_dim is None or n_classes is None:
        raise ValueError("Checkpoint missing feature_dim/num_classes metadata")
    
    # Check for binary classification (output dim 1)
    out_classes = 1 if n_classes == 2 else n_classes
    
    model = TabularMLP(d_in=feat_dim, num_classes=out_classes)
    model.load_state_dict(ckpt["model_state"])
    data_mean = ckpt.get("data_mean")
    data_std = ckpt.get("data_std")
    encoder_meta = ckpt.get("encoder_meta")
    if encoder_meta is not None:
        encoder_meta["num_classes"] = n_classes
    logger.info("Loaded global model from %s (feature_dim=%s num_classes=%s out_dim=%s)", ckpt_path, feat_dim, n_classes, out_classes)
    return model, cfg, data_mean, data_std, encoder_meta


def get_set_dataloaders(cfg: dict, saved_encoder_meta: dict | None) -> tuple[dict, torch.Tensor, torch.Tensor]:
    loaders, _ = get_tabular_loaders(cfg, encoder_meta=saved_encoder_meta)
    logger.info(
        "Loaders built: client=%d train=%d val=%d test=%d features=%d classes=%d",
        len(loaders["client_loader"].dataset),
        len(loaders["train_loader"].dataset),
        len(loaders["val_loader"].dataset) if loaders.get("val_loader") else 0,
        len(loaders["test_loader"].dataset) if loaders.get("test_loader") else 0,
        loaders["n_features"],
        loaders["num_classes"],
    )
    return loaders, loaders["data_mean"], loaders["data_std"]


def run_attack(model: TabularMLP, client_loader: DataLoader, data_mean: torch.Tensor, data_std: torch.Tensor, encoder_meta: dict, train_loader: DataLoader) -> None:
    
    # 1. Compute Public/Prior Stats
    cat_freqs = compute_prior_stats(train_loader, encoder_meta)
    encoder_meta['cat_frequencies'] = cat_freqs

    # Criterion 
    criterion = torch.nn.BCEWithLogitsLoss()
    
    data_extension = GiaTabularExtension()
    data_extension.feature_meta = encoder_meta # Inject meta for constraints initialization

    # Configure InvertingGradients
    attack_config = InvertingConfig(
        attack_lr=0.01,
        at_iterations=1000,
        tv_reg=0.01, # This will be ignored by generic_attack_loop because is_tabular will be true
        criterion=criterion,
        data_extension=data_extension
    )
    
    attacker = InvertingGradients(
        model=model,
        client_loader=client_loader,
        data_mean=data_mean, 
        data_std=data_std,
        configs=attack_config
    )
    
    logger.info("Starting Tabular GIA (using InvertingGradients): steps=%d lr=%.4f", attack_config.at_iterations, attack_config.attack_lr)
    
    attacker.prepare_attack()
    
    # 2. Prior Baseline Check (Before optimization)
    gen = attacker.run_attack()
    
    last_score = 0
    last_result = None
    scores = []
    
    try:
        iter_0, score_0, _ = next(gen)
        logger.info(f"--- Prior Baseline Score: {score_0:.4f} ---")
        scores.append(score_0)
        last_score = score_0
    except StopIteration:
        pass

    for i, score, result in gen:
        last_score = score
        scores.append(score)
        if result:
            last_result = result
            
    if last_result is None:
        logger.warning("No result produced.")
        return

    logger.info(f"Attack complete. Final Score: {last_score.item():.4f}")
    
    # Detailed logging
    # InvertingGradients stores best_reconstruction as DataLoader (copied from reconstruction_loader)
    # We need to extract the tensor
    if attacker.best_reconstruction:
        recon_tensor = torch.cat([batch[0] for batch in attacker.best_reconstruction], dim=0).detach().cpu()
    else:
        recon_tensor = torch.zeros_like(attacker.original.cpu())

    orig_tensor = attacker.original.cpu()
    
    # De-standardize if mean/std available
    if data_mean is not None and data_std is not None:
        data_mean = data_mean.cpu()
        data_std = data_std.cpu()
        # Safe inverse transform
        orig_raw = orig_tensor * data_std + data_mean
        recon_raw = recon_tensor * data_std + data_mean
    else:
        orig_raw = orig_tensor
        recon_raw = recon_tensor

    # Show metrics
    if last_result:
        # last_result is GIAResults object
        # It contains rmse_score, mae_score etc for tabular
        logger.info(f"Final Metrics: RMSE={last_result.rmse_score:.4f}, MAE={last_result.mae_score:.4f}")
        
        # We can also re-calculate our custom detailed metrics if desired
        full_metrics = evaluate_reconstruction(
             attacker.original.to(attacker.original.device), 
             recon_tensor.to(attacker.original.device), 
             encoder_meta, 
             return_per_feature=True
        )
        agg = full_metrics['aggregate']
        logger.info(f"Detailed Metrics: Numerical Score={agg['numerical_score']:.4f}, Categorical Score={agg['categorical_score']:.4f}")
        print(f"FINAL_METRICS: Numerical={agg['numerical_score']:.4f} Categorical={agg['categorical_score']:.4f}")

    # Show best row
    errors = torch.sum(torch.abs(orig_tensor - recon_tensor), dim=1)
    idx_best = int(errors.argmin().item())
    logger.info("Best row (idx=%d) original: %s", idx_best, orig_raw[idx_best].numpy())
    logger.info("Best row (idx=%d) reconstructed: %s", idx_best, recon_raw[idx_best].numpy())



def main(protocol: str = "fedsgd") -> None:
    seed_everything(42)
    base_cfg, ckpt_path = load_config()
    if not ckpt_path.exists():
        ckpt_path, _ = train_global_model(base_cfg)

    model, _, ckpt_mean, ckpt_std, saved_encoder_meta = load_model(ckpt_path, default_cfg=base_cfg)
    loaders, data_mean_loader, data_std_loader = get_set_dataloaders(base_cfg, saved_encoder_meta)

    client_loader = loaders["client_loader"]
    if protocol == "fedsgd":
        xb, yb = next(iter(client_loader))
        fedsgd_ds = TensorDataset(xb, yb)
        client_loader = DataLoader(fedsgd_ds, batch_size=len(xb), shuffle=False)
        logger.info("Protocol=fedsgd: using single batch of size %d from client split (orig size=%d)", len(xb), len(loaders["client_loader"].dataset))
    else:
        logger.info("Protocol=fedavg: using full client split (size=%d)", len(client_loader.dataset))

    data_mean = ckpt_mean if ckpt_mean is not None else data_mean_loader
    data_std = ckpt_std if ckpt_std is not None else data_std_loader
    logger.info("Using data_mean=%s data_std=%s (checkpoint overrides loader if present)",
        "ckpt" if ckpt_mean is not None else "loader",
        "ckpt" if ckpt_std is not None else "loader",
    )

    run_attack(model, client_loader, data_mean, data_std, saved_encoder_meta, loaders["train_loader"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular GIA demo")
    parser.add_argument("--protocol", choices=["fedavg", "fedsgd"], default="fedsgd", help="Full client split (fedavg) or single mini-batch (fedsgd)")
    parsed = parser.parse_args()
    main(protocol=parsed.protocol)

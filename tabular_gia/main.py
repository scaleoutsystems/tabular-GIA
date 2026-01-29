"""Minimal tabular GIA demo using a pre-trained global model."""

import sys
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients, InvertingConfig
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.utils.seed import seed_everything

from tabular_metrics import apply_matching, compute_metrics, nearest_neighbor_distance, tab_leak_accuracy, write_results_table

from train import train_global_model
from model import TabularMLP
from tabular import get_tabular_loaders, load_tabular_config, denormalize_features




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
        features_np = features.cpu().numpy()
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


def _move_loader_to_device(loader: DataLoader, device: torch.device) -> DataLoader:
    """In-place move of TensorDataset tensors to match model device."""
    ds = loader.dataset
    if hasattr(ds, "tensors"):
        ds.tensors = tuple(t.to(device) for t in ds.tensors)
    return loader


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
    task = meta.get("task")
    head_out = meta.get("head_out")
    if feat_dim is None or n_classes is None:
        raise ValueError("Checkpoint missing feature_dim/num_classes metadata")

    # For binary tasks we train a single-logit BCE head; honor stored head_out to reload correctly.
    out_dim = head_out if head_out is not None else (1 if task == "binary" else n_classes)
    model = TabularMLP(d_in=feat_dim, num_classes=out_dim)
    model.load_state_dict(ckpt["model_state"])
    data_mean = ckpt.get("data_mean")
    data_std = ckpt.get("data_std")
    encoder_meta = ckpt.get("encoder_meta")
    logger.info(
        "Loaded global model from %s (feature_dim=%s targets=%s head_out=%s task=%s)",
        ckpt_path,
        feat_dim,
        n_classes,
        out_dim,
        task,
    )
    if encoder_meta is not None:
        encoder_meta["num_classes"] = n_classes
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


def evaluate_inversion(
    attacker: InvertingGradients,
    orig_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
    encoder_meta: dict,
    train_loader: DataLoader,
    results_path: str,
    label_tensor: torch.Tensor | None,
) -> None:
    """Centralized evaluation/logging for tabular inversion results."""
    num_cols = encoder_meta.get("num_cols", [])
    cat_cols = encoder_meta.get("cat_cols", [])
    cat_categories = encoder_meta.get("cat_categories", {}) or {}

    num_count = len(num_cols)
    num_eps = np.full(num_count, 0.319, dtype=np.float32)

    nn_batch = nearest_neighbor_distance(recon_tensor, orig_tensor, encoder_meta)
    logger.info(
        "NN distance (target batch): min=%.4f mean=%.4f median=%.4f",
        nn_batch["min"],
        nn_batch["mean"],
        nn_batch["median"],
    )

    if hasattr(train_loader, "dataset"):
        train_count = len(train_loader.dataset)
        if train_count * recon_tensor.shape[1] <= 2.0e7:
            train_tensor = torch.cat([batch[0] for batch in train_loader], dim=0).detach().cpu()
            nn_train = nearest_neighbor_distance(recon_tensor, train_tensor, encoder_meta)
            logger.info(
                "NN distance (train set): min=%.4f mean=%.4f median=%.4f",
                nn_train["min"],
                nn_train["mean"],
                nn_train["median"],
            )
        else:
            logger.info(
                "Skipped NN distance over train set (rows=%d features=%d).",
                train_count,
                recon_tensor.shape[1],
            )

    per_row_acc, _ = tab_leak_accuracy(
        orig_tensor,
        recon_tensor,
        num_cols,
        cat_cols,
        cat_categories,
        None,
    )

    # de-normalize before writing results
    num_mean = encoder_meta.get("num_mean")
    num_std = encoder_meta.get("num_std")
    if num_count > 0 and num_mean is not None and num_std is not None:
        orig_raw = denormalize_features(orig_tensor, num_mean, num_std, num_count)
        recon_raw = denormalize_features(recon_tensor, num_mean, num_std, num_count)
        num_std_np = np.asarray(num_std).reshape(-1)
        if num_std_np.size >= num_count:
            num_eps_raw = 0.319 * num_std_np[:num_count]
        else:
            num_eps_raw = num_eps
    else:
        orig_raw = orig_tensor
        recon_raw = recon_tensor
        num_eps_raw = num_eps

    # compute rmse and mae across the reconstruction batch on the de-normalized data
    metrics = compute_metrics(orig_raw, recon_raw, encoder_meta)

    write_results_table(
        results_path,
        orig_raw,
        recon_raw,
        num_cols,
        cat_cols,
        cat_categories,
        num_eps_raw,
        label_tensor,
        per_row_acc,
        nn_batch,
        metrics,
    )


def run_attack(model: TabularMLP, client_loader: DataLoader, data_mean: torch.Tensor, data_std: torch.Tensor, encoder_meta: dict, train_loader: DataLoader) -> None:
    
    # 1. Compute Public/Prior Stats
    cat_freqs = compute_prior_stats(train_loader, encoder_meta)
    encoder_meta['cat_frequencies'] = cat_freqs

    # Choose task-aware criterion
    num_classes = encoder_meta.get("num_classes")
    target_mode = encoder_meta.get("target_mode")
    if target_mode == "classification":
        task = "binary" if num_classes == 2 else "multiclass"
    else:
        task = target_mode or ("binary" if num_classes == 2 else "multiclass" if num_classes and num_classes > 2 else "regression")

    if task == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
    
    data_extension = GiaTabularExtension()
    data_extension.feature_meta = encoder_meta # Inject meta for constraints initialization

    # Configure InvertingGradients
    attack_config = InvertingConfig(
        tv_reg=0.01, # This will be ignored by generic_attack_loop because is_tabular will be true
        attack_lr=0.03,
        at_iterations=10000,
        #optimizer: object = lambda : MetaSGD(),
        criterion=criterion,
        data_extension=data_extension,
        epochs=1,
        #median_pooling = False,
        #top10norms = False
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
            
    logger.info(f"Attack complete. Final Score: {float(last_score):.4f}")
    
    return attacker, last_result, data_mean, data_std, encoder_meta, train_loader


def main(protocol: str = "fedsgd", results_path: str = "results.txt") -> None:
    seed_everything(42)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(device))
    else:
        logger.info("Using CPU")
    base_cfg, ckpt_path = load_config()
    if not ckpt_path.exists():
        ckpt_path, _ = train_global_model(base_cfg)

    model, _, ckpt_mean, ckpt_std, saved_encoder_meta = load_model(ckpt_path, default_cfg=base_cfg)
    model = model.to(device)
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
    if data_mean is not None:
        data_mean = data_mean.to(device)
    if data_std is not None:
        data_std = data_std.to(device)
    logger.info("Using data_mean=%s data_std=%s (checkpoint overrides loader if present)",
        "ckpt" if ckpt_mean is not None else "loader",
        "ckpt" if ckpt_std is not None else "loader",
    )

    # run the attack on the model with the client data
    attacker, last_result, data_mean, data_std, encoder_meta, train_loader = run_attack(model, client_loader, data_mean, data_std, saved_encoder_meta, loaders["train_loader"])

    # metrics from attack
    # last_result is GIAResults object. it contains rmse_score, mae_score, etc for tabular
    if last_result is not None:
        logger.info("Final Metrics: RMSE=%.4f, MAE=%.4f", last_result.rmse_score, last_result.mae_score)

    # Extract tensors for evaluation
    if attacker.best_reconstruction:
        recon_tensor = torch.cat([batch[0] for batch in attacker.best_reconstruction], dim=0).detach().cpu()
    else:
        recon_tensor = torch.zeros_like(attacker.original.cpu())
    orig_tensor = attacker.original.detach().cpu()

    labels = getattr(attacker, "reconstruction_labels", None)
    if labels is None:
        label_tensor = None
    elif isinstance(labels, list):
        label_tensor = torch.stack(labels).view(-1).detach().cpu()
    else:
        label_tensor = torch.as_tensor(labels).view(-1).detach().cpu()

    # apply hungarian matching
    orig_tensor, recon_tensor, label_tensor = apply_matching(orig_tensor, recon_tensor, label_tensor)
    # run evaluation of the attack
    evaluate_inversion(attacker, orig_tensor, recon_tensor, encoder_meta, train_loader, results_path, label_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular GIA demo")
    parser.add_argument("--protocol", choices=["fedavg", "fedsgd"], default="fedsgd", help="Full client split (fedavg) or single mini-batch (fedsgd)")
    parser.add_argument("--results-path", default="results.txt", help="Results path")
    parsed = parser.parse_args()
    main(protocol=parsed.protocol, results_path=parsed.results_path)

"""Minimal tabular GIA demo using a pre-trained global model."""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.utils.seed import seed_everything

from train import train_global_model
from model import TabularMLP
from tabular import get_tabular_loaders, load_tabular_config


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


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
    model = TabularMLP(d_in=feat_dim, num_classes=n_classes)
    model.load_state_dict(ckpt["model_state"])
    data_mean = ckpt.get("data_mean")
    data_std = ckpt.get("data_std")
    encoder_meta = ckpt.get("encoder_meta")
    logger.info("Loaded global model from %s (feature_dim=%s num_classes=%s)", ckpt_path, feat_dim, n_classes)
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


def run_attack(model: TabularMLP, client_loader: DataLoader, data_mean: torch.Tensor, data_std: torch.Tensor) -> None:
    attack_cfg = InvertingConfig(
        data_extension=GiaTabularExtension(),
        median_pooling=False,
        tv_reg=0.0,
        at_iterations=500,
        attack_lr=0.1,
    )
    attack = InvertingGradients(model, client_loader, data_mean, data_std, configs=attack_cfg)
    logger.info("Starting attack: steps=%d lr=%.4f", attack_cfg.at_iterations, attack_cfg.attack_lr)
    attack.prepare_attack()
    last_result = None
    for _, _, result in attack.run_attack():
        if result is not None:
            last_result = result
    if last_result is None:
        logger.warning("No result produced.")
        return

    logger.info("Attack complete: rmse=%.4f mae=%.4f", last_result.rmse_score, last_result.mae_score)
    orig_tensor = torch.cat([batch[0] for batch in last_result.original_data], dim=0)
    orig_labels = torch.cat([batch[1] for batch in last_result.original_data], dim=0).view(-1)
    recon_tensor = torch.cat([batch[0] for batch in last_result.recreated_data], dim=0)

    per_feat_mse = torch.mean((orig_tensor - recon_tensor) ** 2, dim=0)
    per_feat_rmse = torch.sqrt(per_feat_mse)
    norm_rmse = (per_feat_rmse / data_std).detach().cpu().numpy()
    logger.info("Per-feature RMSE: %s", per_feat_rmse.detach().cpu().numpy())
    logger.info("Per-feature normalized RMSE (RMSE / std): %s", norm_rmse)

    errors = torch.sum(torch.abs(orig_tensor - recon_tensor), dim=1)
    idx_best = int(errors.argmin().item())
    if data_mean is not None and data_std is not None:
        orig_row = (orig_tensor[idx_best] * data_std + data_mean).detach().cpu().numpy()
        recon_row = (recon_tensor[idx_best] * data_std + data_mean).detach().cpu().numpy()
    else:
        orig_row = orig_tensor[idx_best].detach().cpu().numpy()
        recon_row = recon_tensor[idx_best].detach().cpu().numpy()
    logger.info("Best row (idx=%d) original: %s", idx_best, orig_row)
    logger.info("Best row (idx=%d) reconstructed: %s", idx_best, recon_row)

    try:
        model.eval()
        with torch.no_grad():
            logits = model(recon_tensor)
            if logits.ndim > 1 and logits.shape[1] > 1:
                preds = logits.argmax(dim=1)
            else:
                preds = (torch.sigmoid(logits).view(-1) > 0.5).long()
            label_acc = (preds.view(-1).cpu() == orig_labels.cpu()).float().mean().item()
        logger.info("Label accuracy on reconstructed features: %.4f", label_acc)
    except Exception:
        logger.warning("Could not compute label accuracy (model mismatch)")


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

    run_attack(model, client_loader, data_mean, data_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular GIA demo")
    parser.add_argument("--protocol", choices=["fedavg", "fedsgd"], default="fedsgd", help="Full client split (fedavg) or single mini-batch (fedsgd)")
    parsed = parser.parse_args()
    main(protocol=parsed.protocol)

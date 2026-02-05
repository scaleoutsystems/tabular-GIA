"""Experiment runner that wraps existing FL/GIA components."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml
import torch
from tqdm import tqdm

from leakpro.attacks.gia_attacks.invertinggradients import InvertingGradients, InvertingConfig
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD
from leakpro.fl_utils.gia_train import train_nostep, train
from leakpro.utils.seed import seed_everything

from tabular_gia.fl.dataloader.tabular_dataloader import load_dataset
from tabular_gia.fl.fedsgd import run_fedsgd
from tabular_gia.fl.metrics.fl_metrics import eval_epoch, infer_task_from_criterion
from tabular_gia.model import TabularMLP1, TabularMLP2, TabularMLP
from tabular_gia.tabular_metrics import evaluate_batch_rows


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("leakpro").setLevel(logging.WARNING)


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(base_dir: Path, maybe_path: str | None) -> Path | None:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return p if p.is_absolute() else base_dir / p


def _build_model(model_cfg: dict, feature_schema: dict, task: str) -> torch.nn.Module:
    name = (model_cfg.get("name") or "mlp1").lower()
    d_hidden = int(model_cfg.get("d_hidden", 128))
    dropout = float(model_cfg.get("dropout", 0.1))
    n_blocks = int(model_cfg.get("n_blocks", 3))
    out_dim = 1 if task == "binary" else feature_schema["num_classes"]

    if name in ("mlp2", "tabularmlp2"):
        return TabularMLP2(
            d_in=feature_schema["num_features"],
            d_hidden=d_hidden,
            num_classes=out_dim,
            dropout=dropout,
        )
    if name in ("mlp", "tabularmlp"):
        return TabularMLP(
            d_in=feature_schema["num_features"],
            d_hidden=max(d_hidden, 64),
            num_classes=out_dim,
            dropout=dropout,
            n_blocks=n_blocks,
        )
    return TabularMLP1(
        d_in=feature_schema["num_features"],
        d_hidden=d_hidden,
        num_classes=out_dim,
    )


def run_experiment(
    base_cfg_path: str,
    dataset_cfg_path: str,
    model_cfg_path: str,
    fl_cfg_path: str,
    gia_cfg_path: str,
    results_base_path: str | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    base_dir = Path(__file__).resolve().parents[1]
    base_cfg_path = _resolve_path(base_dir, base_cfg_path)
    dataset_cfg_path = _resolve_path(base_dir, dataset_cfg_path)
    model_cfg_path = _resolve_path(base_dir, model_cfg_path)
    fl_cfg_path = _resolve_path(base_dir, fl_cfg_path)
    gia_cfg_path = _resolve_path(base_dir, gia_cfg_path)

    base_cfg = _load_yaml(base_cfg_path)
    protocol = base_cfg.get("protocol", "fedsgd")
    seed = int(base_cfg.get("seed", 42))

    seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    ds_cfg = _load_yaml(dataset_cfg_path)
    dataset_path = _resolve_path(base_dir, ds_cfg.get("dataset_path"))
    dataset_meta_path = _resolve_path(base_dir, ds_cfg.get("dataset_meta_path"))
    if dataset_path is not None:
        ds_cfg["dataset_path"] = str(dataset_path)
    if dataset_meta_path is not None:
        ds_cfg["dataset_meta_path"] = str(dataset_meta_path)
    model_cfg = _load_yaml(model_cfg_path)
    fl_cfg = _load_yaml(fl_cfg_path)
    fl_cfg["batch_size"] = ds_cfg.get("batch_size")

    gia_cfg = _load_yaml(gia_cfg_path)
    gia_cfg = gia_cfg.get(protocol, {}).get("invertingconfig", {})

    if quick:
        ds_cfg["batch_size"] = min(int(ds_cfg.get("batch_size", 32)), 16)
        ds_cfg["num_workers"] = 0
        ds_cfg["pin_memory"] = False
        ds_cfg["persistent_workers"] = False
        fl_cfg["epochs"] = min(int(fl_cfg.get("epochs", 1)), 1)
        fl_cfg["client_participation"] = min(float(fl_cfg.get("client_participation", 1.0)), 0.2)
        gia_cfg["at_iterations"] = min(int(gia_cfg.get("at_iterations", 10)), 10)

    if results_base_path is None:
        results_base_path = gia_cfg.get("results_dir", "leakpro_output/tabular_gia")
    results_dir = Path(results_base_path) / protocol
    results_dir.mkdir(parents=True, exist_ok=True)

    num_clients = int(fl_cfg.get("num_clients", 1))
    ds_cfg["seed"] = seed
    client_dataloaders, val_loader, test_loader, feature_schema = load_dataset(
        **ds_cfg,
        num_clients=num_clients,
    )

    task = feature_schema["task"]
    if task == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    model = _build_model(model_cfg, feature_schema, task).to(device)

    optimizer = MetaSGD() if gia_cfg.get("optimizer") == "MetaSGD" else MetaAdam()
    attack_cfg = InvertingConfig(
        attack_lr=gia_cfg.get("attack_lr"),
        at_iterations=gia_cfg.get("at_iterations"),
        optimizer=optimizer,
        criterion=criterion,
        data_extension=GiaTabularExtension(),
    )

    def attack_fn(att_model, batch_loader, epoch_idx, round_idx, client_idx):
        if quick and (epoch_idx > 1 or round_idx > 1):
            return
        attacker = InvertingGradients(
            att_model,
            batch_loader,
            None,
            None,
            train_fn=train_nostep,
            configs=attack_cfg,
        )
        attacker.prepare_attack()
        total_iters = int(attack_cfg.at_iterations or 0)
        last_i = -1
        with tqdm(
            total=total_iters,
            desc=f"GIA Epoch {epoch_idx} Round {round_idx} Client {client_idx}",
            leave=False,
            disable=quick,
        ) as bar:
            for i, score, _ in attacker.run_attack():
                step = max(0, int(i) - last_i)
                if step:
                    bar.update(step)
                last_i = int(i)
                loss = -float(score) if score is not None else float("nan")
                best_loss = float(attacker.best_loss) if hasattr(attacker, "best_loss") else float("nan")
                bar.set_postfix(loss=f"{loss:.6f}", best=f"{best_loss:.6f}")
            if total_iters > 0 and last_i + 1 < total_iters:
                bar.update(total_iters - (last_i + 1))
        out_dir = results_dir / f"epoch_{epoch_idx}" / f"round_{round_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / f"client_{client_idx}.txt"
        evaluate_batch_rows(attacker, feature_schema, str(results_path), client_idx)
        tqdm.write(f"GIA done: epoch={epoch_idx} round={round_idx} client={client_idx}")

    if protocol == "fedsgd":
        run_fedsgd(fl_cfg, model, criterion, attack_fn, client_dataloaders, val_loader, test_loader)
    else:
        raise NotImplementedError("FedAvg not wired yet in experiments runner.")

    task = infer_task_from_criterion(criterion)
    train_stats = eval_epoch(client_dataloaders, model, criterion, task)
    val_stats = eval_epoch([val_loader], model, criterion, task) if val_loader is not None else None
    test_stats = eval_epoch([test_loader], model, criterion, task)

    return {
        "model": model,
        "metrics": {"train": train_stats, "val": val_stats, "test": test_stats},
        "feature_schema": feature_schema,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    default_cfg_dir = base_dir / "configs"
    parser = argparse.ArgumentParser(description="Run a tabular GIA experiment")
    parser.add_argument("--base", default=str(default_cfg_dir / "base.yaml"))
    parser.add_argument("--dataset", default=str(default_cfg_dir / "dataset" / "dataset.yaml"))
    parser.add_argument("--model", default=str(default_cfg_dir / "model" / "model.yaml"))
    parser.add_argument("--fl", default=str(default_cfg_dir / "fl" / "fedsgd.yaml"))
    parser.add_argument("--gia", default=str(default_cfg_dir / "gia" / "gia.yaml"))
    parser.add_argument("--results", default=None)
    parser.add_argument("--quick", action="store_true", help="Run a small, fast config for local sanity checks.")
    args = parser.parse_args()

    run_experiment(args.base, args.dataset, args.model, args.fl, args.gia, args.results, args.quick)


if __name__ == "__main__":
    main()

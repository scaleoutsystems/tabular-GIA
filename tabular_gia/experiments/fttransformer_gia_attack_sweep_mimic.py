from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABULAR_GIA_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, TABULAR_GIA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.base import BaseConfig
from configs.dataset.dataset import DatasetConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig
from configs.model.model import ModelConfig
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaSGD
from leakpro.fl_utils.gia_train import train_nostep, train_nostep_vectorized
from leakpro.utils.seed import seed_everything
from tabular_gia.metrics.tabular_metrics import compute_reconstruction_metrics, prepare_tensors_for_metrics
from tabular_gia.runner.run import RunConfig, build_runtime


@dataclass(frozen=True)
class AttackSetting:
    gia_forward_mode: str
    gia_soft_temperature: float | None
    gia_init_logit_scale: float | None

    @property
    def tag(self) -> str:
        if self.gia_forward_mode == "probabilities":
            return "probabilities"
        tau = "none" if self.gia_soft_temperature is None else f"{self.gia_soft_temperature:g}"
        scale = "none" if self.gia_init_logit_scale is None else f"{self.gia_init_logit_scale:g}"
        return f"logits_tau{tau}_scale{scale}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MIMIC FTTransformer initialized-checkpoint attack sweep comparing the probability "
            "GIA path against the default logits GIA path (tau=1, scale=5)."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/binary/mimic_admission_tier3_binary/mimic_admission_tier3_binary.train.csv",
        help="Relative dataset CSV path.",
    )
    parser.add_argument(
        "--dataset-meta-path",
        type=str,
        default="data/binary/mimic_admission_tier3_binary/mimic_admission_tier3_binary.yaml",
        help="Relative dataset metadata YAML path.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="Client batch sizes to test.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument(
        "--client-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Client indices whose first local batch will be attacked.",
    )
    parser.add_argument(
        "--logits-configs",
        type=str,
        nargs="+",
        default=["1.0:5.0"],
        help="Explicit logits tau:scale pairs to test. Default matches the paper setting.",
    )
    parser.add_argument(
        "--include-probabilities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the probability-simplex GIA path alongside the logits setting(s).",
    )
    parser.add_argument(
        "--attack-iterations",
        type=int,
        default=10000,
        help="Number of attack iterations per attacked batch.",
    )
    parser.add_argument(
        "--attack-lr",
        type=float,
        default=0.06,
        help="Attack optimizer learning rate.",
    )
    parser.add_argument(
        "--local-lr",
        type=float,
        default=0.01,
        help="FedSGD local learning rate used for observed gradients.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tabular_gia/results/ablations/fttransformer_gia_attack_sweep_mimic",
        help="Directory for JSON and CSV outputs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Number of parallel worker processes across batch sizes. 0 uses one worker per batch size.",
    )
    return parser.parse_args()


def _configure_repro(seed: int) -> None:
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _run_config(
    *,
    dataset_path: str,
    dataset_meta_path: str,
    batch_size: int,
    seed: int,
    num_clients: int,
) -> RunConfig:
    return RunConfig(
        base_cfg=BaseConfig(seed=int(seed), protocol="fedsgd"),
        dataset_cfg=DatasetConfig(
            dataset_path=dataset_path,
            dataset_meta_path=dataset_meta_path,
            batch_size=int(batch_size),
        ),
        model_cfg=ModelConfig(preset="fttransformer"),
        fl_cfg=FedSGDConfig(
            local_steps=1,
            local_epochs=1,
            num_clients=int(num_clients),
            min_exposure=25.0,
            optimizer="MetaSGD",
            lr=0.01,
            vectorized_clients=True,
        ),
        gia_cfg=GiaConfig(),
        results_dir=Path("/tmp/tabular_gia_fttransformer_attack_sweep_mimic"),
        fl_only=False,
    )


def _settings(args: argparse.Namespace) -> list[AttackSetting]:
    logits_pairs: list[tuple[float, float]] = []
    for raw in args.logits_configs:
        text = str(raw).strip()
        if ":" not in text:
            raise ValueError(f"Invalid logits config '{text}'. Expected tau:scale.")
        tau_raw, scale_raw = text.split(":", 1)
        logits_pairs.append((float(tau_raw), float(scale_raw)))

    out: list[AttackSetting] = []
    if bool(args.include_probabilities):
        out.append(
            AttackSetting(
                gia_forward_mode="probabilities",
                gia_soft_temperature=None,
                gia_init_logit_scale=None,
            )
        )
    for tau, scale in logits_pairs:
        out.append(
            AttackSetting(
                gia_forward_mode="logits",
                gia_soft_temperature=float(tau),
                gia_init_logit_scale=float(scale),
            )
        )
    return out


def _make_single_batch_loader(loader: DataLoader) -> DataLoader:
    xb, yb = next(iter(loader))
    return DataLoader(TensorDataset(xb, yb), batch_size=int(xb.shape[0]), shuffle=False)


def _make_vectorized_batch_loader(loaders: list[DataLoader]) -> tuple[DataLoader, list[torch.Tensor], list[torch.Tensor], int]:
    x_per_client: list[torch.Tensor] = []
    y_per_client: list[torch.Tensor] = []
    expected_rows: int | None = None
    for loader in loaders:
        xb, yb = next(iter(loader))
        rows = int(xb.shape[0])
        if expected_rows is None:
            expected_rows = rows
        elif rows != expected_rows:
            raise ValueError("Vectorized client attack requires same number of rows in every selected client batch.")
        x_per_client.append(xb.detach().cpu())
        y_per_client.append(yb.detach().cpu())
    if expected_rows is None:
        raise ValueError("No client loaders provided for vectorized attack.")
    x_stack = torch.stack(x_per_client, dim=0)
    y_stack = torch.stack(y_per_client, dim=0)
    batch_loader = DataLoader(
        TensorDataset(x_stack, y_stack),
        batch_size=int(x_stack.shape[0]),
        shuffle=False,
    )
    return batch_loader, x_per_client, y_per_client, int(expected_rows)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str, float | None, float | None], list[dict]] = {}
    for row in rows:
        key = (
            int(row["batch_size"]),
            str(row["gia_forward_mode"]),
            _to_float(row["gia_soft_temperature"]),
            _to_float(row["gia_init_logit_scale"]),
        )
        grouped.setdefault(key, []).append(row)

    metric_keys = [
        "tableak_acc",
        "num_acc",
        "cat_acc",
        "prior_tableak_acc",
        "gain_tableak_over_prior",
        "random_tableak_acc",
        "gain_tableak_over_random",
        "emr",
        "emr_90",
        "emr_80",
        "emr_60",
        "dist_conf",
        "num_dist_conf",
        "num_within_1std",
        "num_within_2std",
        "cat_dist_conf",
        "best_loss",
    ]

    out: list[dict] = []
    for key, group in sorted(grouped.items()):
        batch_size, mode, tau, scale = key
        row = {
            "batch_size": batch_size,
            "gia_forward_mode": mode,
            "gia_soft_temperature": tau,
            "gia_init_logit_scale": scale,
            "num_attacked_batches": len(group),
        }
        for metric in metric_keys:
            values = [float(g[metric]) for g in group if g.get(metric) is not None]
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            row[f"{metric}_mean"] = float(tensor.mean().item())
            row[f"{metric}_std"] = float(tensor.std(unbiased=False).item()) if len(values) > 1 else 0.0
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_vectorized_attack(
    *,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    feature_schema: dict,
    client_loaders: list[DataLoader],
    client_indices: list[int],
    local_lr: float,
    attack_lr: float,
    attack_iterations: int,
    random_seed: int,
) -> list[dict]:
    batch_loader, _x_per_client, y_per_client, expected_rows = _make_vectorized_batch_loader(client_loaders)

    # Match the real FedSGD path. Observed gradients are collected from a train-mode model.
    model.train()
    observed_client_gradient = train_nostep_vectorized(
        model,
        batch_loader,
        None,
        criterion,
        epochs=1,
        client_batch_size=int(expected_rows),
    )
    observed_client_gradient = [g.detach().clone() if g is not None else None for g in observed_client_gradient]

    attack_cfg = InvertingConfig(
        attack_lr=float(attack_lr),
        at_iterations=int(attack_iterations),
        optimizer=MetaSGD(lr=float(local_lr)),
        criterion=criterion,
        data_extension=GiaTabularExtension(),
        epochs=1,
        label_known=True,
        vectorized_client_batch_size=int(expected_rows),
    )
    attacker = InvertingGradients(
        model,
        batch_loader,
        None,
        None,
        train_fn=train_nostep,
        configs=attack_cfg,
        observed_client_gradient=observed_client_gradient,
    )
    attacker.prepare_attack()
    for _ in attacker.run_attack():
        pass

    best_x_rec = attacker.best_reconstruction.dataset.reconstruction.detach().cpu()
    best_client_losses = attacker.vectorized_best_client_losses
    rows: list[dict] = []
    for c_idx, client_idx in enumerate(client_indices):
        client_labels = y_per_client[c_idx].detach().cpu()
        recon_loader = DataLoader(
            TensorDataset(best_x_rec[c_idx].detach().cpu(), client_labels),
            batch_size=int(best_x_rec.shape[1]),
            shuffle=False,
        )
        orig_tensor, recon_tensor, label_tensor = prepare_tensors_for_metrics(
            original=attacker.original[c_idx],
            best_reconstruction=recon_loader,
            reconstruction_labels=attacker.reconstruction_labels[c_idx],
            model=attacker.model,
            feature_schema=feature_schema,
            client_idx=int(client_idx),
        )
        metrics = compute_reconstruction_metrics(
            orig_tensor,
            recon_tensor,
            feature_schema,
            int(client_idx),
            random_baseline_seed=int(random_seed),
        )["aggregate_metrics"]

        row = {key: _to_float(value) for key, value in metrics.items()}
        row["client_idx"] = int(client_idx)
        row["rows"] = int(orig_tensor.shape[0])
        if best_client_losses is not None and c_idx < int(best_client_losses.shape[0]):
            row["best_loss"] = float(best_client_losses[c_idx].detach().cpu().item())
        else:
            row["best_loss"] = float(attacker.best_loss)
        if label_tensor is not None:
            row["label_count"] = int(label_tensor.shape[0])
        rows.append(row)
    return rows


def _run_batch_size(
    *,
    batch_size: int,
    dataset_path: str,
    dataset_meta_path: str,
    seed: int,
    num_clients: int,
    client_indices: list[int],
    local_lr: float,
    attack_lr: float,
    attack_iterations: int,
    settings_payload: list[dict],
) -> list[dict]:
    _configure_repro(int(seed))
    settings = [AttackSetting(**payload) for payload in settings_payload]
    run_cfg = _run_config(
        dataset_path=str(dataset_path),
        dataset_meta_path=str(dataset_meta_path),
        batch_size=int(batch_size),
        seed=int(seed),
        num_clients=int(num_clients),
    )
    runtime = build_runtime(run_cfg)
    model = runtime.model_wrapper
    criterion = model.criterion
    selected_loaders = [_make_single_batch_loader(runtime.client_dataloaders[int(client_idx)]) for client_idx in client_indices]

    rows: list[dict] = []
    for setting in settings:
        model.gia_forward_mode = str(setting.gia_forward_mode)
        if setting.gia_soft_temperature is not None:
            model.gia_soft_temperature = float(setting.gia_soft_temperature)
        if setting.gia_init_logit_scale is not None:
            model.gia_init_logit_scale = float(setting.gia_init_logit_scale)

        setting_rows = _run_vectorized_attack(
            model=model,
            criterion=criterion,
            feature_schema=runtime.feature_schema,
            client_loaders=selected_loaders,
            client_indices=client_indices,
            local_lr=float(local_lr),
            attack_lr=float(attack_lr),
            attack_iterations=int(attack_iterations),
            random_seed=int(seed),
        )
        for row in setting_rows:
            row["batch_size"] = int(batch_size)
            row["gia_forward_mode"] = str(setting.gia_forward_mode)
            row["gia_soft_temperature"] = _to_float(setting.gia_soft_temperature)
            row["gia_init_logit_scale"] = _to_float(setting.gia_init_logit_scale)
            row["setting_tag"] = setting.tag
            rows.append(row)
            print(
                (
                    f"completed batch_size={int(batch_size)} client={int(row['client_idx'])} "
                    f"setting={setting.tag} tableak_acc={row.get('tableak_acc', float('nan')):.6f}"
                ),
                flush=True,
            )
    return rows


def main() -> None:
    args = parse_args()
    _configure_repro(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = _settings(args)
    manifest = {
        "dataset": "mimic",
        "model": "fttransformer",
        "checkpoint": "initialized",
        "seed": int(args.seed),
        "num_clients": int(args.num_clients),
        "client_indices": [int(v) for v in args.client_indices],
        "batch_sizes": [int(v) for v in args.batch_sizes],
        "attack_iterations": int(args.attack_iterations),
        "attack_lr": float(args.attack_lr),
        "local_lr": float(args.local_lr),
        "dataset_path": str(args.dataset_path),
        "dataset_meta_path": str(args.dataset_meta_path),
        "settings": [asdict(setting) for setting in settings],
        "max_workers": int(args.max_workers),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    per_batch_rows: list[dict] = []
    settings_payload = [asdict(setting) for setting in settings]
    max_workers = int(args.max_workers) if int(args.max_workers) > 0 else len(args.batch_sizes)
    if max_workers <= 1:
        for batch_size in args.batch_sizes:
            per_batch_rows.extend(
                _run_batch_size(
                    batch_size=int(batch_size),
                    dataset_path=str(args.dataset_path),
                    dataset_meta_path=str(args.dataset_meta_path),
                    seed=int(args.seed),
                    num_clients=int(args.num_clients),
                    client_indices=[int(v) for v in args.client_indices],
                    local_lr=float(args.local_lr),
                    attack_lr=float(args.attack_lr),
                    attack_iterations=int(args.attack_iterations),
                    settings_payload=settings_payload,
                )
            )
    else:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _run_batch_size,
                    batch_size=int(batch_size),
                    dataset_path=str(args.dataset_path),
                    dataset_meta_path=str(args.dataset_meta_path),
                    seed=int(args.seed),
                    num_clients=int(args.num_clients),
                    client_indices=[int(v) for v in args.client_indices],
                    local_lr=float(args.local_lr),
                    attack_lr=float(args.attack_lr),
                    attack_iterations=int(args.attack_iterations),
                    settings_payload=settings_payload,
                ): int(batch_size)
                for batch_size in args.batch_sizes
            }
            for future in concurrent.futures.as_completed(futures):
                batch_size = futures[future]
                rows = future.result()
                per_batch_rows.extend(rows)
                print(f"finished all settings for batch_size={batch_size}", flush=True)

    summary_rows = _aggregate_rows(per_batch_rows)
    _write_csv(output_dir / "per_batch_results.csv", per_batch_rows)
    _write_csv(output_dir / "summary_by_setting.csv", summary_rows)
    (output_dir / "per_batch_results.json").write_text(json.dumps(per_batch_rows, indent=2))
    (output_dir / "summary_by_setting.json").write_text(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()

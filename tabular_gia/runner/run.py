import json
import logging
import sys
import csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sweep import build_run_specs
from helper.helpers import write_yaml
from fl.dataloader.tabular_dataloader import load_dataset
from fl.fedavg import run_fedavg
from fl.fedsgd import run_fedsgd
from leakpro.attacks.gia_attacks.invertinggradients import InvertingConfig, InvertingGradients
from leakpro.fl_utils.data_utils import GiaTabularExtension
from leakpro.fl_utils.gia_optimizers import MetaAdam, MetaSGD
from leakpro.fl_utils.gia_train import train, train_nostep
from leakpro.utils.seed import restore_rng_state, seed_everything
from model.model import TabularMLP
from metrics.tabular_metrics import (
    compute_reconstruction_metrics,
    prepare_tensors_for_metrics,
    summarize_epoch,
    summarize_run,
    write_debug_reconstruction_txt,
    write_epoch_summary,
    write_round_summary,
    write_run_summary,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("leakpro").setLevel(logging.WARNING)


def run(
    protocol: str,
    dataset_cfg: dict,
    model_cfg: dict,
    fl_cfg: dict,
    gia_cfg: dict,
    results_dir: Path,
):
    fl_cfg["batch_size"] = dataset_cfg["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(device))
    else:
        logger.info("Using CPU")

    num_clients = fl_cfg.get("num_clients")
    if num_clients < 1:
        raise ValueError(f"Invalid num_clients in FL config: {num_clients}")

    client_dataloaders, val_loader, test_loader, feature_schema = load_dataset(
        **dataset_cfg,
        num_clients=num_clients,
    )
    task = feature_schema["task"]
    if task == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    out_dim = 1 if task == "binary" else feature_schema["num_classes"]
    preset_name = model_cfg.get("use_preset")
    if preset_name is not None:
        presets = model_cfg.get("presets", {})
        preset_cfg = presets.get(preset_name)
        if preset_cfg is None:
            raise ValueError(f"Preset '{preset_name}' not found in model config.")
        model_cfg = preset_cfg

    model = TabularMLP(
        d_in=feature_schema["num_features"],
        d_out=out_dim,
        **model_cfg,
    )
    model.to(device)

    lr = fl_cfg.get("lr")
    optimizer_name = fl_cfg.get("optimizer")
    logger.info("Optimizer: %s", optimizer_name)

    def _optimizer_fn(optimizer_name: str | None, lr: float):
        return MetaAdam(lr=lr) if optimizer_name == "MetaAdam" else MetaSGD(lr=lr)

    invertingconfig = gia_cfg.get("invertingconfig")
    if invertingconfig is None:
        raise ValueError("Missing GIA invertingconfig.")
    invertingconfig = dict(invertingconfig)
    invertingconfig.pop("data_extension", None)
    attack_cfg_base = InvertingConfig(
        optimizer=_optimizer_fn(optimizer_name, lr),
        criterion=criterion,
        data_extension=GiaTabularExtension(),
        epochs=fl_cfg.get("local_epochs"),
        **invertingconfig
    )
    attack_schedule = str(gia_cfg.get("attack_schedule", "all")).strip().lower()
    round_summaries_by_epoch: dict[int, list[dict]] = {}
    epoch_summaries_by_epoch: dict[int, dict] = {}

    def round_summary_fn(epoch_idx: int, round_idx: int, metrics_list: list[dict]) -> None:
        round_summary = write_round_summary(results_dir, epoch_idx, round_idx, metrics_list)
        if round_summary is not None:
            round_summaries_by_epoch.setdefault(epoch_idx, []).append(round_summary)

    def epoch_summary_fn(epoch_idx: int) -> None:
        epoch_rounds = round_summaries_by_epoch.get(epoch_idx, [])
        epoch_summary = summarize_epoch(epoch_idx, epoch_rounds)
        if epoch_summary is None:
            return
        write_epoch_summary(results_dir, epoch_summary)
        epoch_summaries_by_epoch[epoch_idx] = epoch_summary

    def finalize_summaries() -> dict | None:
        epoch_summaries = [epoch_summaries_by_epoch[k] for k in sorted(epoch_summaries_by_epoch)]
        run_summary = summarize_run(epoch_summaries)
        if run_summary is not None:
            write_run_summary(results_dir, run_summary)
        return run_summary

    def _make_attack_fn(train_fn, mismatch_label: str):
        def attack_fn(att_model, batch_loader, epoch_idx, round_idx, client_idx, client_updates, rng_pre, rng_post):
            if attack_schedule == "pow2":
                r = int(round_idx)
                if r < 1 or (r & (r - 1)) != 0:
                    return None

            attack_cfg = deepcopy(attack_cfg_base)
            attacker = InvertingGradients(
                att_model,
                batch_loader,
                None,
                None,
                train_fn=train_fn,
                configs=attack_cfg,
            )
            attacker.replay_rng_state = rng_pre
            attacker.prepare_attack()
            restore_rng_state(rng_post)

            if len(client_updates) != len(attacker.client_gradient):
                raise AssertionError(
                    f"{mismatch_label} length mismatch: "
                    f"client={len(client_updates)} attacker={len(attacker.client_gradient)}"
                )
            for idx, (a, b) in enumerate(zip(client_updates, attacker.client_gradient)):
                if a is None or b is None:
                    continue
                if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                    raise AssertionError(f"{mismatch_label} mismatch at param {idx}")

            total_iters = int(attack_cfg.at_iterations or 0)
            last_i = -1
            with tqdm(
                total=total_iters,
                desc=f"GIA Epoch {epoch_idx} Round {round_idx} Client {client_idx}",
                leave=False,
            ) as bar:
                for i, score, _ in attacker.run_attack():
                    step = max(0, int(i) - last_i)
                    if step:
                        bar.update(step)
                    last_i = int(i)
                    loss = -float(score) if score is not None else float("nan")
                    best_loss = float(attacker.best_loss)
                    bar.set_postfix(loss=f"{loss:.6e}", best=f"{best_loss:.6e}")
                if total_iters > 0 and last_i + 1 < total_iters:
                    bar.update(total_iters - (last_i + 1))

            out_dir = results_dir / f"epoch_{epoch_idx}" / f"round_{round_idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            results_path = out_dir / f"client_{client_idx}.txt"
            orig_tensor, recon_tensor, label_tensor = prepare_tensors_for_metrics(attacker, feature_schema, client_idx)
            reconstruction_metrics = compute_reconstruction_metrics(
                orig_tensor,
                recon_tensor,
                feature_schema,
                client_idx,
            )
            write_debug_reconstruction_txt(
                str(results_path),
                orig_tensor,
                recon_tensor,
                reconstruction_metrics["per_row_metrics"],
                feature_schema,
                client_idx,
                label_tensor,
            )
            metrics = dict(reconstruction_metrics["aggregate_metrics"])
            metrics["client_idx"] = int(client_idx)
            tqdm.write(
                f"GIA done: epoch={epoch_idx} round={round_idx} client={client_idx} "
                f"best={float(attacker.best_loss):.6e}"
            )
            return metrics

        return attack_fn

    if protocol == "fedsgd":
        attack_fn = _make_attack_fn(train_nostep, "Gradient")
        run_fedsgd(
            fl_cfg,
            model,
            criterion,
            attack_fn,
            client_dataloaders,
            val_loader,
            test_loader,
            round_summary_fn,
            epoch_summary_fn,
        )
        return finalize_summaries()

    if protocol == "fedavg":
        attack_fn = _make_attack_fn(train, "Delta")
        run_fedavg(
            fl_cfg,
            model,
            criterion,
            _optimizer_fn,
            attack_fn,
            client_dataloaders,
            val_loader,
            test_loader,
            round_summary_fn,
            epoch_summary_fn,
        )
        return finalize_summaries()

    raise ValueError(f"Unsupported protocol: {protocol}")


def _extract_run_metadata(spec: dict, run_dir: Path) -> dict:
    base_cfg = spec.get("base_cfg", {})
    dataset_cfg = spec.get("dataset_cfg", {})
    fl_cfg = spec.get("fl_cfg", {})
    gia_cfg = spec.get("gia_cfg", {})
    inverting_cfg = gia_cfg.get("invertingconfig", {}) or {}
    model_override = (spec.get("overrides", {}) or {}).get("model", {}) or {}

    return {
        "run_dir": str(run_dir),
        "protocol": base_cfg.get("protocol"),
        "seed": base_cfg.get("seed"),
        "attack_schedule": gia_cfg.get("attack_schedule", base_cfg.get("attack_schedule")),
        "dataset_path": dataset_cfg.get("dataset_path"),
        "dataset_meta_path": dataset_cfg.get("dataset_meta_path"),
        "batch_size": dataset_cfg.get("batch_size"),
        "partition_strategy": dataset_cfg.get("partition_strategy", "iid"),
        "dirichlet_alpha": dataset_cfg.get("dirichlet_alpha"),
        "num_clients": fl_cfg.get("num_clients"),
        "client_participation": fl_cfg.get("client_participation"),
        "full_dataset_passes": fl_cfg.get("full_dataset_passes"),
        "local_steps": fl_cfg.get("local_steps"),
        "local_epochs": fl_cfg.get("local_epochs"),
        "optimizer": fl_cfg.get("optimizer"),
        "lr": fl_cfg.get("lr"),
        "label_known": inverting_cfg.get("label_known"),
        "at_iterations": inverting_cfg.get("at_iterations"),
        "attack_lr": inverting_cfg.get("attack_lr"),
        "model_name": model_override.get("name"),
    }


def _write_sweep_results_csv(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_sweep(
    sweep_cfg: dict,
    base_cfg: dict,
    dataset_cfg: dict,
    model_cfg: dict,
    config_dir: Path,
    results_dir: Path,
) -> None:
    experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    experiment_dir = results_dir / experiment_id
    run_specs = build_run_specs(
        sweep_cfg=sweep_cfg,
        base_cfg=base_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        config_dir=config_dir
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_records = []
    sweep_result_rows: list[dict] = []
    for spec in run_specs:
        run_id = int(spec["run_id"])
        protocol = str(spec["base_cfg"].get("protocol", ""))
        run_seed = int(spec["base_cfg"].get("seed", 42))
        if not protocol:
            raise ValueError(f"Missing protocol for run_id={run_id}")

        # Re-seed each run for reproducible multi-seed sweeps.
        seed_everything(run_seed)

        dataset_cfg_run = deepcopy(spec["dataset_cfg"])
        dataset_cfg_run["seed"] = run_seed

        run_dir = experiment_dir / f"run_{run_id:04d}"
        cfg_dir = run_dir / "configs"
        base_cfg_run_path = cfg_dir / "base.yaml"
        dataset_cfg_run_path = cfg_dir / "dataset" / "dataset.yaml"
        model_cfg_run_path = cfg_dir / "model" / "model.yaml"
        gia_cfg_run_path = cfg_dir / "gia" / "gia.yaml"
        fl_cfg_run_path = cfg_dir / "fl" / f"{protocol}.yaml"

        write_yaml(base_cfg_run_path, spec["base_cfg"])
        write_yaml(dataset_cfg_run_path, dataset_cfg_run)
        write_yaml(model_cfg_run_path, spec["model_cfg"])
        write_yaml(gia_cfg_run_path, {protocol: spec["gia_cfg"]})
        write_yaml(fl_cfg_run_path, spec["fl_cfg"])

        run_summary = run(
            protocol=protocol,
            dataset_cfg=dataset_cfg_run,
            model_cfg=deepcopy(spec["model_cfg"]),
            fl_cfg=deepcopy(spec["fl_cfg"]),
            gia_cfg=deepcopy(spec["gia_cfg"]),
            results_dir=run_dir / "artifacts",
        )

        metadata = _extract_run_metadata(spec, run_dir)
        row = {
            "run_id": run_id,
            **metadata,
            "overrides_json": json.dumps(spec.get("overrides", {}), sort_keys=True),
        }
        if run_summary is not None:
            row.update(run_summary)
        sweep_result_rows.append(row)

        run_records.append(
            {
                "run_id": run_id,
                "protocol": protocol,
                "seed": run_seed,
                "run_dir": str(run_dir),
                "overrides": spec.get("overrides", {}),
            }
        )

    with open(experiment_dir / "sweep_runs.json", "w", encoding="utf-8") as f:
        json.dump(run_records, f, indent=2)
    _write_sweep_results_csv(experiment_dir / "sweep_results.csv", sweep_result_rows)

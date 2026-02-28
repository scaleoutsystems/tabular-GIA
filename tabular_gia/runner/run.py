import json
import logging
import sys
import csv
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

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
    summarize_round,
    summarize_run,
    write_debug_reconstruction_txt,
    write_rounds_summary,
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
    results_dir.mkdir(parents=True, exist_ok=True)
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
    attack_mode = str(gia_cfg.get("attack_mode", "round_checkpoint")).strip().lower()
    if attack_mode not in {"round_checkpoint", "fixed_batch"}:
        raise ValueError("Unknown attack_mode. Use one of: round_checkpoint, fixed_batch.")

    fixed_batch_k = int(gia_cfg.get("fixed_batch_k", 1))
    if fixed_batch_k <= 0:
        raise ValueError("fixed_batch_k must be > 0.")

    attack_schedule = str(gia_cfg.get("attack_schedule", "all")).strip().lower()
    min_exposure = fl_cfg.get("min_exposure")
    if min_exposure is None:
        raise ValueError("Missing required FL config field: min_exposure")
    min_exposure = float(min_exposure)
    if min_exposure <= 0:
        raise ValueError(f"min_exposure must be > 0, got {min_exposure}")
    exposure_targets: list[float] = []
    next_exposure_idx = 0
    exposure_round_labels: dict[int, list[float]] = {}
    if attack_schedule == "exposure":
        raw = gia_cfg.get("attack_exposure_milestones", [0.25, 0.5, 0.75, 1.0])
        if not isinstance(raw, list):
            raise ValueError("attack_schedule='exposure' requires gia.attack_exposure_milestones as a list.")
        exposure_targets = sorted({
            (float(m) * min_exposure if float(m) <= 1.0 else float(m))
            for m in raw
            if float(m) > 0
        })
        if not exposure_targets:
            raise ValueError("attack_schedule='exposure' produced no valid exposure milestones.")

    planned_num_rounds: int | None = None
    checkpoint_rounds: set[int] | None = None

    def _build_checkpoint_rounds(total_rounds: int) -> set[int] | None:
        if attack_schedule == "all":
            return None

        if attack_schedule == "pow2":
            rounds: set[int] = set()
            r = 1
            while r <= total_rounds:
                rounds.add(r)
                r <<= 1
            return rounds

        if attack_schedule == "fixed":
            raw = gia_cfg.get("attack_rounds", [])
            if not isinstance(raw, list):
                raise ValueError("attack_schedule='fixed' requires gia.attack_rounds as a list of round indices.")
            rounds = {int(r) for r in raw if 1 <= int(r) <= total_rounds}
            if not rounds:
                raise ValueError("attack_schedule='fixed' produced no valid rounds within planned round range.")
            return rounds

        if attack_schedule in {"log", "logspace"}:
            count = int(gia_cfg.get("attack_num_checkpoints", 8))
            if count <= 0:
                raise ValueError("attack_num_checkpoints must be > 0 for logspace schedule.")
            if count == 1:
                return {total_rounds}
            rounds = {
                int(round(math.exp((i / (count - 1)) * math.log(total_rounds))))
                for i in range(count)
            }
            rounds = {min(total_rounds, max(1, r)) for r in rounds}
            return rounds

        if attack_schedule == "exposure":
            return set()

        raise ValueError(
            f"Unknown attack_schedule '{attack_schedule}'. "
            "Use one of: all, pow2, fixed, logspace, exposure."
        )

    def num_rounds_fn(total_rounds: int) -> None:
        nonlocal planned_num_rounds, checkpoint_rounds
        planned_num_rounds = int(total_rounds)
        checkpoint_rounds = _build_checkpoint_rounds(planned_num_rounds)
        if checkpoint_rounds is None:
            logger.info("Attack checkpoints: all rounds (1..%d)", planned_num_rounds)
        else:
            logger.info("Attack checkpoints: schedule=%s selected=%d/%d", attack_schedule, len(checkpoint_rounds), planned_num_rounds)

    def exposure_progress_fn(round_idx: int, exp_min_prev: float, exp_min_curr: float) -> None:
        nonlocal next_exposure_idx
        if attack_schedule != "exposure" or checkpoint_rounds is None:
            return
        crossed: list[float] = []
        while next_exposure_idx < len(exposure_targets) and exposure_targets[next_exposure_idx] <= exp_min_curr:
            milestone = exposure_targets[next_exposure_idx]
            if exp_min_prev < milestone:
                crossed.append(float(milestone))
            next_exposure_idx += 1
        if crossed:
            checkpoint_rounds.add(int(round_idx))
            exposure_round_labels[int(round_idx)] = crossed

    fixed_replay_loaders: dict[int, list[DataLoader]] = {}
    if attack_mode == "fixed_batch":
        local_steps_raw = fl_cfg.get("local_steps", 1)
        local_steps_all = isinstance(local_steps_raw, str) and local_steps_raw.strip().lower() == "all"
        local_steps = None if local_steps_all else max(1, int(local_steps_raw))
        base_seed = int(dataset_cfg.get("seed", 42))

        def _sample_fixed_units(
            num_effective: int,
            unit_size: int,
            k_units: int,
            seed: int,
        ) -> list[list[int]]:
            if unit_size <= 0:
                raise ValueError("Fixed Batch unit_size must be > 0.")
            max_units = num_effective // unit_size
            if k_units > max_units:
                raise ValueError(
                    f"fixed_batch_k={k_units} exceeds non-overlapping capacity={max_units} "
                    f"(n_eff={num_effective}, unit_size={unit_size})."
                )
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            perm = torch.randperm(num_effective, generator=generator).tolist()
            selected = perm[: k_units * unit_size]
            return [selected[i * unit_size : (i + 1) * unit_size] for i in range(k_units)]

        def _build_replay_loader(client_loader: DataLoader, sample_ids: list[int]) -> DataLoader:
            if not isinstance(client_loader.dataset, TensorDataset):
                raise TypeError("Fixed Batch mode requires TensorDataset client datasets.")
            if len(client_loader.dataset.tensors) != 2:
                raise ValueError("Expected client TensorDataset to contain (inputs, labels).")
            x_all, y_all = client_loader.dataset.tensors
            idx = torch.as_tensor(sample_ids, dtype=torch.long)
            replay_ds = TensorDataset(x_all.index_select(0, idx), y_all.index_select(0, idx))
            return DataLoader(replay_ds, batch_size=int(client_loader.batch_size), shuffle=False)

        for client_idx, client_loader in enumerate(client_dataloaders):
            batch_size = int(client_loader.batch_size)
            num_effective = int(len(client_loader) * batch_size)
            if num_effective <= 0:
                raise ValueError(f"Client {client_idx} has no effective samples for Fixed Batch mode.")

            if protocol == "fedsgd":
                sequence_batches = 1
            else:
                sequence_batches = int(len(client_loader)) if local_steps_all else int(local_steps)

            unit_size = int(sequence_batches * batch_size)
            seed = base_seed + (client_idx * 1_000_003)
            unit_sample_ids = _sample_fixed_units(
                num_effective=num_effective,
                unit_size=unit_size,
                k_units=fixed_batch_k,
                seed=seed,
            )
            client_units: list[DataLoader] = []
            for sample_ids in unit_sample_ids:
                client_units.append(_build_replay_loader(client_loader, sample_ids))
            fixed_replay_loaders[client_idx] = client_units

        logger.info(
            "Fixed Batch mode prepared: clients=%d fixed_batch_k=%d",
            len(fixed_replay_loaders),
            fixed_batch_k,
        )

    round_summaries: list[dict] = []
    attack_id_counter = 0
    attacks_csv_path = results_dir / "attacks.csv"
    fl_csv_path = results_dir / "fl.csv"

    def append_attack_row(attack_row: dict) -> None:
        write_header = not attacks_csv_path.exists() or attacks_csv_path.stat().st_size == 0
        fieldnames = list(attack_row.keys())
        with open(attacks_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(attack_row)

    def fl_metrics_fn(payload: dict) -> None:
        write_header = not fl_csv_path.exists() or fl_csv_path.stat().st_size == 0
        with open(fl_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(payload)

    def round_summary_fn(round_idx: int, metrics_list: list[dict]) -> None:
        if not metrics_list:
            return
        round_metric_rows = [
            {
                k: v
                for k, v in row.items()
                if k not in {
                    "attack_id",
                    "round",
                    "artifact_path",
                    "attack_mode",
                    "fixed_batch_id",
                    "checkpoint_type",
                    "checkpoint_label",
                }
            }
            for row in metrics_list
        ]
        round_summary = summarize_round(round_metric_rows, round_idx)
        if round_summary is not None:
            round_summaries.append(round_summary)

    def finalize_summaries() -> dict | None:
        write_rounds_summary(results_dir, round_summaries)
        run_summary = summarize_run(round_summaries)
        if run_summary is not None:
            write_run_summary(results_dir, run_summary)
        return run_summary

    def _make_attack_fn(train_fn, mismatch_label: str):
        def _run_single_attack(
            att_model,
            batch_loader,
            round_idx: int,
            client_idx: int,
            client_updates,
            rng_pre,
            rng_post,
            fixed_batch_id: int,
            checkpoint_type: str,
            checkpoint_label: str,
            attack_context: dict | None,
        ):
            nonlocal attack_id_counter
            attack_id_counter += 1
            attack_id = attack_id_counter

            attack_cfg = deepcopy(attack_cfg_base)
            attacker = InvertingGradients(
                att_model,
                batch_loader,
                None,
                None,
                train_fn=train_fn,
                configs=attack_cfg,
            )
            if rng_pre is not None:
                attacker.replay_rng_state = rng_pre
            attacker.prepare_attack()
            if rng_post is not None:
                restore_rng_state(rng_post)

            if client_updates is not None:
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
                desc=f"GIA Round {round_idx} Client {client_idx}",
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

            results_path = results_dir / f"attack_{attack_id:06d}_round_{int(round_idx):06d}_client_{int(client_idx):03d}.txt"
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
            metrics["attack_id"] = int(attack_id)
            metrics["round"] = int(round_idx)
            metrics["client_idx"] = int(client_idx)
            metrics["attack_mode"] = attack_mode
            metrics["fixed_batch_id"] = int(fixed_batch_id)
            if attack_context is not None:
                metrics["exp_min"] = float(attack_context["exp_min"])
                metrics["exp_avg"] = float(attack_context["exp_avg"])
                metrics["exp_max"] = float(attack_context["exp_max"])
                metrics["client_exp"] = float(attack_context["client_exp"])
            metrics["checkpoint_type"] = checkpoint_type
            metrics["checkpoint_label"] = checkpoint_label
            metrics["artifact_path"] = results_path.name
            append_attack_row(metrics)
            if fixed_batch_id >= 0:
                tqdm.write(
                    f"GIA done: round={round_idx} client={client_idx} fixed_batch_id={fixed_batch_id} "
                    f"best={float(attacker.best_loss):.6e}"
                )
            else:
                tqdm.write(
                    f"GIA done: round={round_idx} client={client_idx} "
                    f"best={float(attacker.best_loss):.6e}"
                )
            return metrics

        def attack_fn(att_model, batch_loader, round_idx, client_idx, client_updates, rng_pre, rng_post, attack_context):
            if attack_schedule != "all":
                if checkpoint_rounds is None:
                    raise RuntimeError("Attack checkpoint schedule is missing.")
                if int(round_idx) not in checkpoint_rounds:
                    return None
            round_i = int(round_idx)
            if attack_schedule == "exposure":
                checkpoint_type = "exposure"
                crossed = exposure_round_labels.get(round_i, [])
                checkpoint_label = ";".join(f"{m:.6g}" for m in crossed) if crossed else "exposure"
            else:
                checkpoint_type = "round"
                checkpoint_label = "all" if attack_schedule == "all" else str(round_i)

            if attack_mode == "fixed_batch":
                replay_units = fixed_replay_loaders.get(int(client_idx), [])
                metrics_list = []
                for fixed_batch_id, replay_loader in enumerate(replay_units):
                    metrics = _run_single_attack(
                        att_model=att_model,
                        batch_loader=replay_loader,
                        round_idx=int(round_idx),
                        client_idx=int(client_idx),
                        client_updates=None,
                        rng_pre=None,
                        rng_post=None,
                        fixed_batch_id=int(fixed_batch_id),
                        checkpoint_type=checkpoint_type,
                        checkpoint_label=checkpoint_label,
                        attack_context=attack_context,
                    )
                    if metrics is not None:
                        metrics_list.append(metrics)
                return metrics_list if metrics_list else None

            return _run_single_attack(
                att_model=att_model,
                batch_loader=batch_loader,
                round_idx=int(round_idx),
                client_idx=int(client_idx),
                client_updates=client_updates,
                rng_pre=rng_pre,
                rng_post=rng_post,
                fixed_batch_id=-1,
                checkpoint_type=checkpoint_type,
                checkpoint_label=checkpoint_label,
                attack_context=attack_context,
            )

        return attack_fn

    if protocol == "fedsgd":
        attack_fn = _make_attack_fn(train_nostep, "Gradient")
        run_fedsgd(
            cfg=fl_cfg,
            global_model=model,
            criterion=criterion,
            attack_fn=None,
            client_dataloaders=client_dataloaders,
            val=val_loader,
            test=test_loader,
            round_summary_fn=round_summary_fn,
            num_rounds_fn=num_rounds_fn,
            exposure_progress_fn=exposure_progress_fn,
            fl_metrics_fn=fl_metrics_fn,
        )
        return finalize_summaries()

    if protocol == "fedavg":
        attack_fn = _make_attack_fn(train, "Delta")
        run_fedavg(
            cfg=fl_cfg,
            global_model=model,
            criterion=criterion,
            optimizer_fn=_optimizer_fn,
            attack_fn=None,
            client_dataloaders=client_dataloaders,
            val=val_loader,
            test=test_loader,
            round_summary_fn=round_summary_fn,
            num_rounds_fn=num_rounds_fn,
            exposure_progress_fn=exposure_progress_fn,
            fl_metrics_fn=fl_metrics_fn,
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
        "attack_mode": gia_cfg.get("attack_mode", "round_checkpoint"),
        "fixed_batch_k": gia_cfg.get("fixed_batch_k", 1),
        "dataset_path": dataset_cfg.get("dataset_path"),
        "dataset_meta_path": dataset_cfg.get("dataset_meta_path"),
        "batch_size": dataset_cfg.get("batch_size"),
        "partition_strategy": dataset_cfg.get("partition_strategy", "iid"),
        "dirichlet_alpha": dataset_cfg.get("dirichlet_alpha"),
        "num_clients": fl_cfg.get("num_clients"),
        "local_steps": fl_cfg.get("local_steps"),
        "local_epochs": fl_cfg.get("local_epochs"),
        "min_exposure": fl_cfg.get("min_exposure"),
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

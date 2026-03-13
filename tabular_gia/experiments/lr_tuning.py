from pathlib import Path
from math import log10
import csv
import math
from typing import Any

from experiments.sweep_runner import SweepExperimentRunner, SweepRunResults
from helper.helpers import write_json


LR_MIN = 1e-4
LR_MAX = 1e-1
LR_POINTS = 9

TRIALS_CSV_FIELDS = (
    "protocol",
    "model_preset",
    "lr",
    "seed",
    "objective",
    "metric",
    "direction",
)

SUMMARY_CSV_FIELDS = (
    "protocol",
    "model_preset",
    "lr",
    "metric",
    "direction",
    "mean_objective",
    "std_objective",
)


def build_sweep_cfg() -> dict[str, Any]:
    base = {
        "default": {
            "protocol": "fedavg",
            "seed": 42,
        },
        "grid": {
            "protocol": ["fedsgd", "fedavg"],
            "seed": [7, 13, 42],
        },
    }

    dataset = {
        "default": {
            "dataset_path_and_meta_path": [
                "data/binary/adult/adult.csv",
                "data/binary/adult/adult.yaml",
            ],
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
            "batch_size": 256,
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
            "partition_strategy": "iid",
            "dirichlet_alpha": 0.3,
            "min_client_samples": 1,
            "dirichlet_max_attempts": 50,
        },
        "grid": {},
    }

    model = {
        "default": {
            "preset": "small",
            "arch": "mlp",
            "d_hidden": 32,
            "n_hidden_layers": 1,
            "norm": "none",
            "dropout": 0.0,
            "activation": "relu",
            "presets": {
                "small": {
                    "arch": "mlp",
                    "d_hidden": 32,
                    "n_hidden_layers": 1,
                    "norm": "layernorm",
                    "dropout": 0.0,
                    "activation": "gelu",
                },
                "medium": {
                    "arch": "mlp",
                    "d_hidden": 64,
                    "n_hidden_layers": 2,
                    "norm": "layernorm",
                    "dropout": 0.1,
                    "activation": "gelu",
                },
                "large": {
                    "arch": "mlp",
                    "d_hidden": 128,
                    "n_hidden_layers": 3,
                    "norm": "layernorm",
                    "dropout": 0.2,
                    "activation": "gelu",
                },
                "fttransformer": {"arch": "fttransformer"},
                "resnet": {"arch": "resnet"},
            },
        },
        "grid": {
            # For LR tuning we keep MLP presets by default.
            "preset": ["small", "medium", "large", "fttransformer", "resnet"],
        },
    }

    fl = {
        "fedsgd": {
            "default": {
                "local_steps": 1,
                "local_epochs": 1,
                "num_clients": 10,
                "min_exposure": 100.0,
                "optimizer": "MetaSGD",
                "lr": 0.01,
            },
            "grid": {},
        },
        "fedavg": {
            "default": {
                "local_steps": "all",
                "local_epochs": 1,
                "max_client_dataset_examples": 64,
                "num_clients": 3,
                "min_exposure": 100.0,
                "optimizer": "MetaAdam",
                "lr": 0.01,
            },
            "grid": {},
        },
    }

    gia = {
        "default": {
            "attack_mode": "round_checkpoint",
            "fixed_batch_k": 1,
            "attack_schedule": "auto",
            "auto_checkpoints": 6,
            "attack_exposure_milestones": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
            "invertingconfig": {
                "label_known": True,
                "attack_lr": 0.03,
                "at_iterations": 1,
                "data_extension": "GiaTabularExtension",
            },
        },
        "grid": {},
    }

    return {
        "base": base,
        "dataset": dataset,
        "model": model,
        "fl": fl,
        "gia": gia,
    }


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _extract_checkpoint_objective(fl_rows: list[dict], metric_name: str, direction: str) -> float:
    values: list[float] = []
    for row in fl_rows:
        if str(row.get("phase", "")).strip() != "checkpoint":
            continue
        parsed = _parse_float(row.get(metric_name))
        if parsed is not None:
            values.append(parsed)
    if not values:
        raise ValueError(f"No checkpoint values found for metric '{metric_name}'.")
    if direction == "maximize":
        return max(values)
    if direction == "minimize":
        return min(values)
    raise ValueError(f"Unknown objective direction '{direction}'.")


def _infer_objective_spec(fl_rows: list[dict]) -> tuple[str, str]:
    # Task-aware objective inference from available FL metrics.
    if any(_parse_float(row.get("val_roc_auc")) is not None for row in fl_rows):
        return "val_roc_auc", "maximize"  # binary
    if any(_parse_float(row.get("val_f1_macro")) is not None for row in fl_rows):
        return "val_f1_macro", "maximize"  # multiclass
    if any(_parse_float(row.get("val_r2")) is not None for row in fl_rows):
        return "val_r2", "maximize"  # regression
    if any(_parse_float(row.get("val_loss")) is not None for row in fl_rows):
        return "val_loss", "minimize"
    raise ValueError("Could not infer objective metric from FL logs.")


def _write_rows_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    if not rows:
        return
    for row in rows:
        unknown = sorted(set(row.keys()) - set(fieldnames))
        if unknown:
            raise ValueError(f"Unexpected CSV fields in {path.name}: {unknown}")
    ordered_rows = [{field: row.get(field, "") for field in fieldnames} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(row)


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean_value = _mean(values)
    ss = sum((value - mean_value) ** 2 for value in values)
    return float(math.sqrt(ss / float(len(values) - 1)))


def _is_better(candidate: float, incumbent: float, direction: str) -> bool:
    if not math.isfinite(incumbent):
        return True
    if direction == "maximize":
        return candidate > incumbent
    if direction == "minimize":
        return candidate < incumbent
    raise ValueError(f"Unknown objective direction '{direction}'.")


def _write_lr_reports(sweep_results: SweepRunResults, lr_values: list[float]) -> None:
    experiment_dir = sweep_results.experiment_dir
    trial_rows: list[dict] = []
    grouped_objectives: dict[tuple[str, str, float, str, str], list[float]] = {}

    for group in sweep_results.groups:
        if not group.seed_runs:
            continue

        protocol = str(group.protocol)
        first_run_cfg = group.seed_runs[0].run_config
        model_preset = str(first_run_cfg.model_cfg.preset)
        lr_value = float(first_run_cfg.fl_cfg.lr)

        objective_metric = ""
        objective_direction = ""
        for seed_run in group.seed_runs:
            seed = int(seed_run.seed)
            fl_rows = seed_run.run_result.fl_rows
            metric_name, direction = _infer_objective_spec(fl_rows)
            if objective_metric == "":
                objective_metric = metric_name
                objective_direction = direction
            elif objective_metric != metric_name or objective_direction != direction:
                raise ValueError(
                    f"Inconsistent objective for run_group_id={group.run_group_id}: "
                    f"'{objective_metric}/{objective_direction}' vs '{metric_name}/{direction}'."
                )

            objective = _extract_checkpoint_objective(fl_rows, metric_name, direction)
            trial_rows.append(
                {
                    "protocol": protocol,
                    "model_preset": model_preset,
                    "lr": lr_value,
                    "seed": seed,
                    "objective": objective,
                    "metric": metric_name,
                    "direction": direction,
                }
            )
            group_key = (protocol, model_preset, lr_value, metric_name, direction)
            if group_key not in grouped_objectives:
                grouped_objectives[group_key] = []
            grouped_objectives[group_key].append(objective)

    summary_rows: list[dict] = []
    best_by_protocol_model: dict[str, dict] = {}
    for group_key, objectives in grouped_objectives.items():
        protocol, model_preset, lr_value, metric_name, direction = group_key
        mean_objective = _mean(objectives)
        std_objective = _sample_std(objectives)
        summary_rows.append(
            {
                "protocol": protocol,
                "model_preset": model_preset,
                "lr": lr_value,
                "metric": metric_name,
                "direction": direction,
                "mean_objective": mean_objective,
                "std_objective": std_objective,
            }
        )

        if protocol not in best_by_protocol_model:
            best_by_protocol_model[protocol] = {}
        if model_preset not in best_by_protocol_model[protocol]:
            best_by_protocol_model[protocol][model_preset] = {
                "metric": metric_name,
                "direction": direction,
                "best_lr": lr_value,
                "best_mean_objective": mean_objective,
                "lr_values": [float(v) for v in lr_values],
            }
        else:
            incumbent = best_by_protocol_model[protocol][model_preset]
            if _is_better(mean_objective, float(incumbent["best_mean_objective"]), direction):
                incumbent["best_lr"] = lr_value
                incumbent["best_mean_objective"] = mean_objective

    summary_rows.sort(key=lambda row: (row["protocol"], row["model_preset"], float(row["lr"])))
    trial_rows.sort(key=lambda row: (row["protocol"], row["model_preset"], float(row["lr"]), int(row["seed"])))

    _write_rows_csv(experiment_dir / "trials_per_seed.csv", trial_rows, TRIALS_CSV_FIELDS)
    _write_rows_csv(experiment_dir / "lr_summary.csv", summary_rows, SUMMARY_CSV_FIELDS)
    write_json(experiment_dir / "best_lr_by_protocol_model.json", best_by_protocol_model)


def run_experiment(
    *,
    project_root: Path,
    results_dir: Path,
    fl_only: bool = True,
) -> None:
    if LR_POINTS < 2:
        raise ValueError(f"LR_POINTS must be >= 2, got {LR_POINTS}")
    if LR_MIN <= 0 or LR_MAX <= 0 or LR_MIN >= LR_MAX:
        raise ValueError(f"Invalid LR range: min={LR_MIN}, max={LR_MAX}")

    log_low = log10(LR_MIN)
    log_high = log10(LR_MAX)
    step = (log_high - log_low) / float(LR_POINTS - 1)
    lr_values = [10 ** (log_low + i * step) for i in range(LR_POINTS)]

    sweep_cfg = build_sweep_cfg()
    sweep_cfg["fl"]["fedsgd"]["grid"]["lr"] = lr_values
    sweep_cfg["fl"]["fedavg"]["grid"]["lr"] = lr_values

    runner = SweepExperimentRunner(
        sweep_cfg=sweep_cfg,
        results_dir=results_dir / "tuning",
        fl_only=fl_only,
    )
    sweep_results = runner.run()
    experiment_dir = sweep_results.experiment_dir

    write_json(
        experiment_dir / "frozen_tuning_config.json",
        {
            "lr_min": LR_MIN,
            "lr_max": LR_MAX,
            "lr_points": LR_POINTS,
            "lr_values": lr_values,
            "fl_only": fl_only,
        },
    )
    _write_lr_reports(sweep_results, lr_values)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    run_experiment(
        project_root=project_root,
        results_dir=project_root / "results",
        fl_only=True,
    )

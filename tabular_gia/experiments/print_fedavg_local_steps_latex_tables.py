from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

MODEL_CLEAN_NAMES = {
    "fttransformer": "FTTransformer",
    "resnet": "ResNet",
    "small": "Small MLP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print LaTeX tables for FedAvg initialized and trained checkpoint reconstruction accuracy."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to a FedAvg experiment directory containing sweep_runs.json and per-run aggregated stats.",
    )
    parser.add_argument(
        "--metric",
        default="tableak_acc",
        help="Metric to print from rounds_summary_stats.csv. Default: tableak_acc",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimals to print in each cell. Default: 3",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name override for the caption.",
    )
    return parser.parse_args()


def _clean_model_name(model_name: str) -> str:
    return MODEL_CLEAN_NAMES.get(model_name, model_name)


def _dataset_name_from_dir(experiment_dir: Path) -> str:
    name = experiment_dir.name
    for candidate in ("adult", "california", "pandemic", "mimic", "california_housing"):
        if candidate in name:
            return candidate
    return name


def _ordered_models(model_names: list[str]) -> list[str]:
    preferred = ["fttransformer", "resnet", "small"]
    ordered = [name for name in preferred if name in model_names]
    ordered.extend(sorted(name for name in model_names if name not in preferred))
    return ordered


def _load_run_metadata(experiment_dir: Path) -> pd.DataFrame:
    sweep_runs_path = experiment_dir / "sweep_runs.json"
    if not sweep_runs_path.exists():
        raise FileNotFoundError(f"Missing sweep runs file: {sweep_runs_path}")

    sweep_runs = json.loads(sweep_runs_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for row in sweep_runs:
        overrides = row.get("overrides", {})
        dataset_overrides = overrides.get("dataset", {})
        model_overrides = overrides.get("model", {})
        fl_overrides = overrides.get("fl", {})
        batch_size = int(dataset_overrides["batch_size"])
        local_epochs = int(fl_overrides["local_epochs"])
        max_client_dataset_examples = int(fl_overrides["max_client_dataset_examples"])
        number_of_batches = int(math.ceil(max_client_dataset_examples / batch_size))
        rows.append(
            {
                "run_id": int(row["run_id"]),
                "protocol": str(row.get("protocol", "")).strip(),
                "model_name": str(model_overrides.get("preset") or model_overrides.get("arch") or "unknown"),
                "batch_size": batch_size,
                "local_epochs": local_epochs,
                "max_client_dataset_examples": max_client_dataset_examples,
                "number_of_batches": number_of_batches,
            }
        )
    return pd.DataFrame(rows)


def _load_checkpoint_summary_stats(experiment_dir: Path, metadata: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in metadata.to_dict(orient="records"):
        run_id = int(record["run_id"])
        protocol = str(record["protocol"])
        stats_path = experiment_dir / protocol / f"run_{run_id:04d}" / "aggregated" / "rounds_summary_stats.csv"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing rounds summary stats file: {stats_path}")

        stats = pd.read_csv(stats_path)
        if stats.empty:
            raise ValueError(f"Rounds summary stats file is empty: {stats_path}")

        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        if mean_col not in stats.columns or std_col not in stats.columns:
            raise ValueError(
                f"Metric '{metric_name}' not found in {stats_path}. "
                f"Expected columns '{mean_col}' and '{std_col}'."
            )

        stats = stats.sort_values("round").reset_index(drop=True)
        initialized_row = stats.iloc[0]
        trained_row = stats.iloc[-1]
        for stage_name, stats_row in (("Initialized", initialized_row), ("Trained", trained_row)):
            rows.append(
                {
                    "run_id": run_id,
                    "stage": stage_name,
                    "metric_mean": float(stats_row[mean_col]),
                    "metric_std": float(stats_row[std_col]),
                }
            )
    return pd.DataFrame(rows)


def build_stage_frame(experiment_dir: Path, metric_name: str, stage_name: str) -> pd.DataFrame:
    metadata = _load_run_metadata(experiment_dir)
    stats = _load_checkpoint_summary_stats(experiment_dir, metadata, metric_name)
    merged = stats.merge(metadata, on="run_id", how="inner")
    merged = merged[merged["stage"] == stage_name].copy()
    pivot = merged.pivot(
        index="number_of_batches",
        columns=["local_epochs", "model_name"],
        values=["metric_mean", "metric_std"],
    )

    number_of_batches = sorted(int(index) for index in pivot.index.tolist())
    local_epochs = sorted(int(epoch) for epoch in merged["local_epochs"].unique().tolist())
    model_names = _ordered_models(merged["model_name"].astype(str).unique().tolist())
    ordered_cols = pd.MultiIndex.from_product([["metric_mean", "metric_std"], local_epochs, model_names])
    pivot = pivot.reindex(index=number_of_batches, columns=ordered_cols)
    return pivot


def format_stage_table(
    frame: pd.DataFrame,
    stage_name: str,
    dataset_name: str,
    decimals: int,
    metric_name: str,
) -> str:
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex frame with metric_mean/metric_std by local_epochs and model.")

    number_of_batches = [int(index) for index in frame.index.tolist()]
    local_epochs = [int(value) for value in frame.columns.get_level_values(1).unique().tolist()]
    model_names = [str(value) for value in frame.columns.get_level_values(2).unique().tolist()]

    col_spec = "l" + "c" * (len(local_epochs) * len(model_names))
    label_suffix = stage_name.lower()
    lines: list[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Reconstruction accuracy at the {stage_name.lower()} attack checkpoint for {dataset_name}. "
        f"Initialized denotes the attack before any global model aggregation, and Trained denotes the final attacked checkpoint. "
        f"Each cell reports mean $\\pm$ standard deviation across 3 seeds.}}"
    )
    lines.append(f"\\label{{tab:fedavg-{dataset_name}-{metric_name.replace('_', '-')}-{label_suffix}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    header_top = "\\multicolumn{1}{l}{}"
    for epoch in local_epochs:
        epoch_label = "local epoch" if epoch == 1 else "local epochs"
        header_top += f" & \\multicolumn{{{len(model_names)}}}{{c}}{{{epoch} {epoch_label}}}"
    header_top += " \\\\"
    lines.append(header_top)

    header = "n. batches"
    for _epoch in local_epochs:
        for model_name in model_names:
            header += f" & {_clean_model_name(model_name)}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for n_batches in number_of_batches:
        row = frame.loc[n_batches]
        values: list[str] = []
        for epoch in local_epochs:
            for model_name in model_names:
                mean_value = row[("metric_mean", epoch, model_name)]
                std_value = row[("metric_std", epoch, model_name)]
                if pd.isna(mean_value):
                    values.append("--")
                else:
                    values.append(f"{float(mean_value):.{decimals}f} $\\pm$ {float(std_value):.{decimals}f}")
        lines.append(f"{n_batches} & " + " & ".join(values) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    dataset_name = args.dataset_name or _dataset_name_from_dir(experiment_dir)

    initialized_frame = build_stage_frame(experiment_dir, args.metric, "Initialized")
    trained_frame = build_stage_frame(experiment_dir, args.metric, "Trained")

    print(format_stage_table(initialized_frame, "Initialized", dataset_name, args.decimals, args.metric))
    print()
    print(format_stage_table(trained_frame, "Trained", dataset_name, args.decimals, args.metric))


if __name__ == "__main__":
    main()

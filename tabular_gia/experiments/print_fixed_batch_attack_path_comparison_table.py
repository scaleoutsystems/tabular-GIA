from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_ALIASES = {
    "reconstruction": "tableak_acc",
    "recon": "tableak_acc",
    "tableak": "tableak_acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print a LaTeX comparison table for FTTransformer across two attack "
            "parameterization experiment directories."
        )
    )
    parser.add_argument("left_experiment_dir", type=Path)
    parser.add_argument("right_experiment_dir", type=Path)
    parser.add_argument("--metric", default="tableak_acc")
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--dataset-path", default=None, help="Optional explicit dataset path key to compare.")
    parser.add_argument("--left-name", default="Logits ($\\tau=1$, scale 5)")
    parser.add_argument("--right-name", default="Probabilities")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def _resolve_metric_name(metric: str) -> str:
    return METRIC_ALIASES.get(metric.strip().lower(), metric.strip())


def _load_runs(experiment_dir: Path) -> list[dict]:
    path = experiment_dir / "sweep_runs.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    runs: list[dict] = []
    for run_config_path in sorted(experiment_dir.glob("fedsgd/run_*/run_config.json")):
        run_id = int(run_config_path.parent.name.split("_")[1])
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run_id": run_id,
                "protocol": "fedsgd",
                "dataset": payload.get("dataset", {}),
                "model": payload.get("model", {}),
                "overrides": payload.get("overrides", {}),
            }
        )
    if runs:
        return runs

    for run_config_path in sorted(experiment_dir.glob("fedavg/run_*/run_config.json")):
        run_id = int(run_config_path.parent.name.split("_")[1])
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run_id": run_id,
                "protocol": "fedavg",
                "dataset": payload.get("dataset", {}),
                "model": payload.get("model", {}),
                "overrides": payload.get("overrides", {}),
            }
        )
    if runs:
        return runs

    raise FileNotFoundError(
        f"Missing sweep_runs.json and no per-run run_config.json files found in: {experiment_dir}"
    )


def _dataset_key(run_row: dict) -> str:
    overrides = run_row.get("overrides", {})
    dataset_overrides = overrides.get("dataset", {})
    pair = dataset_overrides.get("dataset_path_and_meta_path")
    if pair and len(pair) >= 1:
        return str(pair[0])
    dataset_cfg = run_row.get("dataset", {})
    dataset_path = dataset_cfg.get("dataset_path")
    if dataset_path:
        return str(dataset_path)
    return "unknown"


def _model_name(run_row: dict) -> str:
    overrides = run_row.get("overrides", {})
    model_overrides = overrides.get("model", {})
    return str(model_overrides.get("preset") or model_overrides.get("arch") or "unknown").strip().lower()


def _run_metadata(experiment_dir: Path) -> pd.DataFrame:
    rows = []
    for row in _load_runs(experiment_dir):
        overrides = row.get("overrides", {})
        dataset_overrides = overrides.get("dataset", {})
        rows.append(
            {
                "run_id": int(row["run_id"]),
                "protocol": str(row.get("protocol", "")).strip(),
                "dataset_key": _dataset_key(row),
                "batch_size": int(dataset_overrides["batch_size"]),
                "model_name": _model_name(row),
            }
        )
    return pd.DataFrame(rows)


def _infer_dataset_key(left_meta: pd.DataFrame, right_meta: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    left_keys = sorted(left_meta["dataset_key"].astype(str).unique().tolist())
    right_keys = sorted(right_meta["dataset_key"].astype(str).unique().tolist())
    if len(left_keys) == 1 and left_keys[0] in right_keys:
        return left_keys[0]
    if len(right_keys) == 1 and right_keys[0] in left_keys:
        return right_keys[0]
    common = sorted(set(left_keys).intersection(right_keys))
    if len(common) == 1:
        return common[0]
    raise ValueError(
        "Could not infer a unique dataset to compare. "
        f"Left datasets={left_keys}, right datasets={right_keys}. "
        "Pass --dataset-path explicitly."
    )


def _load_checkpoint_summary_stats(
    experiment_dir: Path,
    metadata: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in metadata.to_dict(orient="records"):
        run_id = int(record["run_id"])
        protocol = str(record["protocol"])
        stats_path = experiment_dir / protocol / f"run_{run_id:04d}" / "aggregated" / "rounds_summary_stats.csv"
        if not stats_path.exists():
            continue
        stats = pd.read_csv(stats_path)
        if stats.empty:
            continue

        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        if mean_col not in stats.columns or std_col not in stats.columns:
            raise ValueError(
                f"Metric '{metric_name}' not found in {stats_path}. "
                f"Expected '{mean_col}' and '{std_col}'."
            )

        stats = stats.sort_values("round").reset_index(drop=True)
        init_row = stats.iloc[0]
        trained_row = stats.iloc[-1]
        for stage_name, stats_row in (("Initialized", init_row), ("Trained", trained_row)):
            rows.append(
                {
                    "run_id": run_id,
                    "stage": stage_name,
                    "metric_mean": float(stats_row[mean_col]),
                    "metric_std": float(stats_row[std_col]),
                }
            )
    return pd.DataFrame(rows)


def _build_frame(experiment_dir: Path, dataset_key: str, metric_name: str) -> pd.DataFrame:
    metadata = _run_metadata(experiment_dir)
    metadata = metadata[
        (metadata["dataset_key"] == dataset_key)
        & (metadata["model_name"] == "fttransformer")
    ].copy()
    if metadata.empty:
        raise ValueError(f"No FTTransformer runs for dataset '{dataset_key}' in {experiment_dir}.")
    stats = _load_checkpoint_summary_stats(experiment_dir, metadata, metric_name)
    merged = stats.merge(metadata, on="run_id", how="inner")
    pivot = merged.pivot(index="batch_size", columns=["stage"], values=["metric_mean", "metric_std"])
    ordered_batches = sorted(int(index) for index in pivot.index.tolist())
    stage_order = ["Initialized", "Trained"]
    ordered_cols = pd.MultiIndex.from_product([["metric_mean", "metric_std"], stage_order])
    return pivot.reindex(index=ordered_batches, columns=ordered_cols)


def _dataset_name_from_key(dataset_key: str) -> str:
    key = dataset_key.lower()
    if "adult" in key:
        return "adult"
    if "mimic" in key:
        return "mimic"
    if "pandemic" in key:
        return "pandemic"
    if "california" in key:
        return "california"
    return Path(dataset_key).stem


def _comparison_kind(left_experiment_dir: Path, right_experiment_dir: Path) -> str:
    joined = f"{left_experiment_dir.name.lower()} {right_experiment_dir.name.lower()}"
    if "fixedbatch" in joined or "fixed_batch" in joined:
        return "fixed-batch"
    return "batch-size"


def _combined_frame(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    left = left.copy()
    left.columns = pd.MultiIndex.from_tuples([(metric, stage, left_name) for metric, stage in left.columns])
    right = right.copy()
    right.columns = pd.MultiIndex.from_tuples([(metric, stage, right_name) for metric, stage in right.columns])
    index = sorted(set(int(v) for v in left.index.tolist()).intersection(int(v) for v in right.index.tolist()))
    if not index:
        raise ValueError("No overlapping batch sizes between the two experiment directories.")
    stage_order = ["Initialized", "Trained"]
    col_order = pd.MultiIndex.from_product(
        [["metric_mean", "metric_std"], stage_order, [left_name, right_name]]
    )
    combined = pd.concat([left, right], axis=1).reindex(index=index, columns=col_order)
    return combined


def format_latex_table(
    frame: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    decimals: int,
    caption: str,
    label: str,
) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append(
        "\\multicolumn{1}{l}{} & \\multicolumn{2}{c}{Initialized} & \\multicolumn{2}{c}{Trained} \\\\"
    )
    lines.append(f"Batch size & {left_name} & {right_name} & {left_name} & {right_name} \\\\")
    lines.append("\\hline")
    for batch_size in frame.index.tolist():
        row = frame.loc[batch_size]
        vals: list[str] = []
        for stage in ("Initialized", "Trained"):
            for attack_name in (left_name, right_name):
                mean_value = row[("metric_mean", stage, attack_name)]
                std_value = row[("metric_std", stage, attack_name)]
                if pd.isna(mean_value):
                    vals.append("--")
                else:
                    vals.append(f"{float(mean_value):.{decimals}f} $\\pm$ {float(std_value):.{decimals}f}")
        lines.append(f"{int(batch_size)} & " + " & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    left_experiment_dir = args.left_experiment_dir.resolve()
    right_experiment_dir = args.right_experiment_dir.resolve()
    metric_name = _resolve_metric_name(args.metric)

    left_meta = _run_metadata(left_experiment_dir)
    right_meta = _run_metadata(right_experiment_dir)
    dataset_key = _infer_dataset_key(left_meta, right_meta, args.dataset_path)

    left_frame = _build_frame(left_experiment_dir, dataset_key, metric_name)
    right_frame = _build_frame(right_experiment_dir, dataset_key, metric_name)
    combined = _combined_frame(left_frame, right_frame, args.left_name, args.right_name)

    dataset_name = args.dataset_name or _dataset_name_from_key(dataset_key)
    comparison_kind = _comparison_kind(left_experiment_dir, right_experiment_dir)
    caption = args.caption
    if caption is None:
        caption = (
            f"Reconstruction accuracy at the initialized and trained attack checkpoints for {dataset_name} "
            f"under two FTTransformer attack parameterizations. Initialized denotes the attack before any "
            f"global model aggregation, and Trained denotes the final attacked checkpoint. Each cell reports "
            f"mean $\\pm$ standard deviation across 3 seeds."
        )
    label = args.label
    if label is None:
        label = f"tab:{comparison_kind}-{dataset_name}-{metric_name.replace('_', '-')}-attack-paths"

    print(
        format_latex_table(
            combined,
            left_name=args.left_name,
            right_name=args.right_name,
            decimals=args.decimals,
            caption=caption,
            label=label,
        )
    )


if __name__ == "__main__":
    main()

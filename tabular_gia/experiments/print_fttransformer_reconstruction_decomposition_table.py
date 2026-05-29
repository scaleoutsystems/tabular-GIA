from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
SEED_DIRS = ["seed_0007", "seed_0013", "seed_0042"]
PROTOCOL = "fedsgd"
METRICS = [
    ("tableak_acc", "Recon. acc."),
    ("num_acc", "Num. acc."),
    ("cat_acc", "Cat. acc."),
]
STAGES = ["Initialized", "Trained"]
DROPOUTS = [
    ("on", "Dropout on"),
    ("off", "Dropout off"),
]
ATTACK_PATHS = [
    ("logits", "Logits ($\\tau=1$, scale 5)"),
    ("probabilities", "Probabilities"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the MIMIC-IV FTTransformer reconstruction decomposition appendix tables."
    )
    parser.add_argument(
        "experiment_dirs",
        type=Path,
        nargs="+",
        help="Experiment directories for the four FTTransformer attack path and dropout conditions.",
    )
    return parser.parse_args()


def _get_nested(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    sweep_runs_path = experiment_dir / "sweep_runs.json"
    if sweep_runs_path.exists():
        return json.loads(sweep_runs_path.read_text(encoding="utf-8"))

    runs: list[dict[str, Any]] = []
    for run_config_path in sorted((experiment_dir / PROTOCOL).glob("run_*/run_config.json")):
        run_id = int(run_config_path.parent.name.split("_")[1])
        payload = json.loads(run_config_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run_id": run_id,
                "protocol": PROTOCOL,
                "dataset": payload.get("dataset", {}),
                "model": payload.get("model", {}),
                "base": payload.get("base", {}),
                "overrides": payload.get("overrides", {}),
            }
        )
    if not runs:
        raise FileNotFoundError(f"No run metadata found under {experiment_dir}")
    return runs


def _batch_size(run_row: dict[str, Any]) -> int:
    batch_size = _get_nested(run_row, "overrides.dataset.batch_size")
    if batch_size is None:
        batch_size = _get_nested(run_row, "dataset.batch_size")
    if batch_size is None:
        raise ValueError(f"Could not read batch size for run {run_row.get('run_id')}")
    return int(batch_size)


def _model_name(run_row: dict[str, Any]) -> str:
    model_name = _get_nested(run_row, "overrides.model.preset")
    if model_name is None:
        model_name = _get_nested(run_row, "overrides.model.arch")
    if model_name is None:
        model_name = _get_nested(run_row, "model.preset") or _get_nested(run_row, "model.arch")
    return str(model_name or "unknown").strip().lower()


def _protocol(run_row: dict[str, Any]) -> str:
    protocol = run_row.get("protocol") or _get_nested(run_row, "base.protocol") or PROTOCOL
    return str(protocol).strip()


def _run_metadata(experiment_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_row in _load_runs(experiment_dir):
        rows.append(
            {
                "run_id": int(run_row["run_id"]),
                "protocol": _protocol(run_row),
                "batch_size": _batch_size(run_row),
                "model_name": _model_name(run_row),
            }
        )
    return pd.DataFrame(rows)


def _condition_from_dir(experiment_dir: Path) -> tuple[str, str]:
    name = experiment_dir.name.lower()
    dropout = "off" if "dropout0" in name or "dropout_0" in name else "on"
    attack_path = "probabilities" if "prob" in name else "logits"
    return dropout, attack_path


def _load_endpoint_rows(experiment_dir: Path) -> pd.DataFrame:
    dropout, attack_path = _condition_from_dir(experiment_dir)
    metadata = _run_metadata(experiment_dir)
    metadata = metadata[
        (metadata["protocol"] == PROTOCOL)
        & (metadata["model_name"] == "fttransformer")
        & (metadata["batch_size"].isin(BATCH_SIZES))
    ].copy()
    if metadata.empty:
        raise ValueError(f"No matching FTTransformer runs found in {experiment_dir}")

    rows: list[dict[str, Any]] = []
    for record in metadata.to_dict(orient="records"):
        run_id = int(record["run_id"])
        run_dir = experiment_dir / PROTOCOL / f"run_{run_id:04d}"
        for seed_dir_name in SEED_DIRS:
            summary_path = run_dir / seed_dir_name / "artifacts" / "rounds_summary.csv"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing seed-level rounds summary file: {summary_path}")

            summary = pd.read_csv(summary_path)
            if summary.empty:
                raise ValueError(f"Seed-level rounds summary file is empty: {summary_path}")

            missing = [metric for metric, _label in METRICS if metric not in summary.columns]
            if missing:
                raise ValueError(f"Missing columns {missing} in {summary_path}")

            summary = summary.sort_values("round").reset_index(drop=True)
            for stage, summary_row in (("Initialized", summary.iloc[0]), ("Trained", summary.iloc[-1])):
                row = {
                    "dropout": dropout,
                    "attack_path": attack_path,
                    "stage": stage,
                    "batch_size": int(record["batch_size"]),
                    "seed": seed_dir_name,
                }
                for metric, _label in METRICS:
                    row[metric] = float(summary_row[metric])
                rows.append(row)

    return pd.DataFrame(rows)


def build_table_frame(experiment_dirs: list[Path]) -> pd.DataFrame:
    endpoints = pd.concat([_load_endpoint_rows(path) for path in experiment_dirs], axis=0, ignore_index=True)
    expected_n = len(BATCH_SIZES) * len(SEED_DIRS)
    counts = endpoints.groupby(["stage", "dropout", "attack_path"], sort=False).size()
    if (counts != expected_n).any():
        raise ValueError(f"Expected {expected_n} seed and batch endpoint rows per condition. Got {counts.to_dict()}")

    grouped = endpoints.groupby(["stage", "dropout", "attack_path"], sort=False)
    means = grouped[[metric for metric, _label in METRICS]].mean().reset_index()
    stds = grouped[[metric for metric, _label in METRICS]].std(ddof=1).reset_index()
    stds = stds.rename(columns={metric: f"{metric}_std" for metric, _label in METRICS})
    return means.merge(stds, on=["stage", "dropout", "attack_path"], how="inner")


def _cell(frame: pd.DataFrame, *, stage: str, dropout: str, attack_path: str, metric: str) -> str:
    row = frame[
        (frame["stage"] == stage)
        & (frame["dropout"] == dropout)
        & (frame["attack_path"] == attack_path)
    ]
    if len(row) != 1:
        raise ValueError(
            f"Expected one row for stage={stage}, dropout={dropout}, attack_path={attack_path}. Found {len(row)}."
        )
    mean_value = float(row.iloc[0][metric])
    std_value = float(row.iloc[0][f"{metric}_std"])
    return f"{mean_value:.3f} $\\pm$ {std_value:.3f}"


def format_stage_table(frame: pd.DataFrame, stage: str) -> str:
    label_stage = stage.lower()
    caption_stage = "initialized" if stage == "Initialized" else "trained"
    lines: list[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(
        f"  \\caption{{{stage} FT-Transformer reconstruction decomposition on MIMIC-IV. "
        "Values average endpoint results over client batch sizes "
        "$1$, $2$, $4$, $8$, $16$, $32$, $64$, and $128$. "
        "Each cell reports mean $\\pm$ standard deviation across seed and batch endpoint values.}"
    )
    lines.append(f"  \\label{{tab:fttransformer-reconstruction-decomposition-{caption_stage}}}")
    lines.append("  \\begin{tabular}{lcccccc}")
    lines.append("  \\hline")
    lines.append(
        "  \\multicolumn{1}{l}{} & \\multicolumn{3}{c}{Dropout on} & \\multicolumn{3}{c}{Dropout off} \\\\"
    )
    metric_header = " & ".join(label for _metric, label in METRICS)
    lines.append(f"  Attack path & {metric_header} & {metric_header} \\\\")
    lines.append("  \\hline")
    for attack_path, attack_label in ATTACK_PATHS:
        values: list[str] = []
        for dropout, _dropout_label in DROPOUTS:
            for metric, _metric_label in METRICS:
                values.append(_cell(frame, stage=stage, dropout=dropout, attack_path=attack_path, metric=metric))
        lines.append(f"  {attack_label} & " + " & ".join(values) + " \\\\")
    lines.append("  \\hline")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    frame = build_table_frame([path.resolve() for path in args.experiment_dirs])
    print("\n\n".join(format_stage_table(frame, stage) for stage in STAGES))


if __name__ == "__main__":
    main()

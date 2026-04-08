from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_TASK_BY_NAME = {
    "adult": "binary",
    "pandemic_movement_office": "multiclass",
    "california_housing": "regression",
}

DEFAULT_Y_BY_TASK = {
    "binary": "val_roc_auc",
    "multiclass": "val_f1_macro",
    "regression": "val_r2",
}

PROTOCOL_CLEAN_NAMES = {
    "fedsgd": "FedSGD",
    "fedavg": "FedAvg",
}

MODEL_CLEAN_NAMES = {
    "fttransformer": "FTTransformer",
    "resnet": "ResNet",
    "small": "Small MLP",
}

DATASET_CLEAN_NAMES = {
    "adult": "Adult",
    "pandemic_movement_office": "Pandemic Movement Office",
    "california_housing": "California Housing",
}

TASK_CLEAN_NAMES = {
    "binary": "Binary Classification",
    "multiclass": "Multiclass Classification",
    "regression": "Regression",
}

X_AXIS_CLEAN_NAMES = {
    "exp_min": "Minimum Exposure",
}

FL_METRIC_CLEAN_NAMES = {
    "val_f1_macro": "Validation F1 Macro",
    "val_roc_auc": "Validation ROC-AUC",
    "val_r2": "Validation R2",
    "val_acc": "Validation Accuracy",
    "val_loss": "Validation Loss",
    "test_f1_macro": "Test F1 Macro",
    "test_roc_auc": "Test ROC-AUC",
    "test_r2": "Test R2",
    "train_acc": "Training Accuracy",
    "train_loss": "Training Loss",
    "test_acc": "Test Accuracy",
    "test_loss": "Test Loss",
}

FL_BOUNDED_METRICS = {
    metric
    for metric in FL_METRIC_CLEAN_NAMES
    if metric.endswith("_f1_macro") or metric.endswith("_roc_auc") or metric.endswith("_acc")
}

PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
DEFAULT_BOUNDED_METRIC_YMAX = 1.03


def _is_numeric_like(value: int | str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def ordered_plot_groups(groups: list[int | str]) -> list[int | str]:
    if all(_is_numeric_like(group) for group in groups):
        return sorted(groups, key=lambda value: float(value))
    return sorted(groups, key=lambda value: str(value))


def clean_name(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def infer_task_objective(dataset_name: str, dataset_path: str) -> str:
    if dataset_name in DATASET_TASK_BY_NAME:
        return DATASET_TASK_BY_NAME[dataset_name]
    path_parts = set(Path(dataset_path).parts)
    if "binary" in path_parts:
        return "binary"
    if "multiclass" in path_parts:
        return "multiclass"
    if "regression" in path_parts:
        return "regression"
    raise ValueError(f"Could not infer task objective for dataset '{dataset_name}' from path '{dataset_path}'.")


def metric_for_final_test(metric: str) -> str:
    if metric.startswith("val_"):
        return "test_" + metric[len("val_"):]
    return metric


def set_plot_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def plot_group_styles(groups: list[int | str]) -> dict[int | str, tuple[Any, str]]:
    cmap = plt.get_cmap("viridis")
    ordered = ordered_plot_groups(groups)
    colors = cmap(np.linspace(0.1, 0.95, max(1, len(ordered))))
    return {group: (colors[idx], PLOT_MARKERS[idx % len(PLOT_MARKERS)]) for idx, group in enumerate(ordered)}


def fl_metric_limits(
    metric: str,
    values: pd.Series,
    *,
    bounded_metric_ymax: float = DEFAULT_BOUNDED_METRIC_YMAX,
) -> tuple[float, float] | None:
    if metric in FL_BOUNDED_METRICS:
        return (0.0, bounded_metric_ymax)
    if metric in {"val_r2", "test_r2"}:
        lo = float(values.min())
        hi = float(values.max())
        pad = max(0.03, 0.08 * max(1e-6, hi - lo))
        return (min(lo - pad, 0.0), min(1.0, hi + pad) if hi <= 1.0 else hi + pad)
    return None


def is_bounded_fl_metric(metric: str) -> bool:
    return metric in FL_BOUNDED_METRICS

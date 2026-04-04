from __future__ import annotations

from pathlib import Path

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
    "exp_min": "Minimum Exposure ($\\mathrm{exp}_{\\min}$)",
}

METRIC_CLEAN_NAMES = {
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

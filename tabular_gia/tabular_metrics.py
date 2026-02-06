"""Evaluation metrics for tabular gradient inversion attacks."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tabulate import tabulate
from scipy.optimize import linear_sum_assignment


def _format_value(value: object) -> str:
    """Format values for table output."""
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def render_table(rows: list[list[object]], headers: list[str]) -> str:
    """Render table with tabulate."""
    rows_fmt = [[_format_value(cell) for cell in row] for row in rows]
    headers_fmt = [_format_value(h) for h in headers]
    return tabulate(rows_fmt, headers=headers_fmt, tablefmt="grid", stralign="center")


def hungarian_match(orig: Tensor, recon: Tensor) -> np.ndarray:
    """Hungarian matching on L1 cost between rows."""
    a = orig.detach().cpu().numpy()
    b = recon.detach().cpu().numpy()
    cost = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind


def apply_matching(orig: Tensor, recon: Tensor, labels: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
    """Reorder recon (and labels) to best match orig rows."""
    perm = hungarian_match(orig, recon)
    recon = recon[perm]
    if labels is not None:
        labels = labels[perm]
    return orig, recon, labels


def compute_metrics(
    ground_truth: Tensor,
    reconstructed: Tensor,
    feature_info: Dict,
) -> Dict:
    """Compute batch-level numeric error (mae/rmse) and categorical accuracy."""
    gt_np = ground_truth.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    num_feats = feature_info.get('numerical_features') or feature_info.get('num_cols', [])
    cat_feats = feature_info.get('categorical_features') or feature_info.get('cat_cols', [])
    cat_categories = feature_info.get('category_mappings') or feature_info.get('cat_categories', {})

    num_count = len(num_feats)
    if num_count > 0:
        gt_num = gt_np[:, :num_count]
        recon_num = recon_np[:, :num_count]
        mae = float(np.mean(np.abs(gt_num - recon_num)))
        rmse = float(np.sqrt(np.mean((gt_num - recon_num) ** 2)))
    else:
        mae = 0.0
        rmse = 0.0

    cat_hits = []
    current_idx = num_count
    for feature in cat_feats:
        cats = cat_categories.get(feature)
        if cats is None:
            continue
        if isinstance(cats, dict):
            cats = cats.get('categories', [])
        n_cats = len(cats)
        if n_cats <= 0 or current_idx + n_cats > gt_np.shape[1]:
            continue
        idxs = list(range(current_idx, current_idx + n_cats))
        hit = (np.argmax(gt_np[:, idxs], axis=1) == np.argmax(recon_np[:, idxs], axis=1)).astype(float)
        cat_hits.append(hit)
        current_idx += n_cats

    cat_acc = float(np.mean(np.concatenate(cat_hits))) if cat_hits else 0.0
    return {"mae": mae, "rmse": rmse, "categorical_accuracy": cat_acc}


def _get_feature_groups(feature_info: Dict, total_dim: int) -> tuple[list[int], list[list[int]]]:
    """Build numeric indices and categorical one-hot groups from encoder metadata."""
    num_feats = feature_info.get('numerical_features') or feature_info.get('num_cols', [])
    cat_feats = feature_info.get('categorical_features') or feature_info.get('cat_cols', [])
    cat_categories = feature_info.get('category_mappings') or feature_info.get('cat_categories', {})

    num_count = len(num_feats)
    num_indices = list(range(min(num_count, total_dim)))
    current_idx = num_count
    cat_groups: list[list[int]] = []

    for feature in cat_feats:
        cats = cat_categories.get(feature)
        if cats is None:
            continue
        if isinstance(cats, dict):
            cats = cats.get('categories', [])
        num_cats = len(cats)
        if num_cats <= 0:
            continue
        if current_idx + num_cats > total_dim:
            break
        cat_groups.append(list(range(current_idx, current_idx + num_cats)))
        current_idx += num_cats

    return num_indices, cat_groups


def nearest_neighbor_distance(
    reconstructed: Tensor,
    reference: Tensor | DataLoader,
    feature_info: Dict,
    num_weight: float = 0.5,
    cat_weight: float = 0.5,
    chunk_size: int = 2048,
) -> Dict:
    """Compute nearest-neighbor distance for each reconstructed row to a reference set."""
    recon_np = reconstructed.detach().cpu().numpy()
    num_idx, cat_groups = _get_feature_groups(feature_info, recon_np.shape[1])
    n_num = len(num_idx)
    n_cat = len(cat_groups)

    if n_num == 0 and n_cat == 0:
        return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "per_record": []}

    if n_num > 0 and n_cat > 0:
        weight_sum = num_weight + cat_weight
        if weight_sum <= 0:
            num_weight, cat_weight = 0.5, 0.5
        else:
            num_weight /= weight_sum
            cat_weight /= weight_sum

    rec_num = recon_np[:, num_idx] if n_num > 0 else None
    rec_cat_labels = [np.argmax(recon_np[:, idxs], axis=1) for idxs in cat_groups] if n_cat > 0 else []
    min_dist = np.full((recon_np.shape[0],), np.inf, dtype=np.float32)

    if isinstance(reference, DataLoader):
        ref_iter = (batch[0].detach().cpu().numpy() for batch in reference)
    else:
        ref_np = reference.detach().cpu().numpy()
        ref_iter = (ref_np[i:i + chunk_size] for i in range(0, ref_np.shape[0], chunk_size))

    for ref_chunk in ref_iter:
        if ref_chunk.size == 0:
            continue
        if n_num > 0:
            ref_num = ref_chunk[:, num_idx]
            num_dist = np.zeros((recon_np.shape[0], ref_chunk.shape[0]), dtype=np.float32)
            for col_idx in range(n_num):
                num_dist += np.abs(rec_num[:, None, col_idx] - ref_num[None, :, col_idx])
            num_dist /= n_num
        else:
            num_dist = None

        if n_cat > 0:
            cat_dist = np.zeros((recon_np.shape[0], ref_chunk.shape[0]), dtype=np.float32)
            for group_idx, idxs in enumerate(cat_groups):
                rec_labels = rec_cat_labels[group_idx]
                ref_labels = np.argmax(ref_chunk[:, idxs], axis=1)
                cat_dist += (rec_labels[:, None] != ref_labels[None, :]).astype(np.float32)
            cat_dist /= n_cat
        else:
            cat_dist = None

        if num_dist is not None and cat_dist is not None:
            dist = num_weight * num_dist + cat_weight * cat_dist
        elif num_dist is not None:
            dist = num_dist
        else:
            dist = cat_dist

        min_dist = np.minimum(min_dist, dist.min(axis=1))

    return {
        "min": float(np.min(min_dist)),
        "mean": float(np.mean(min_dist)),
        "median": float(np.median(min_dist)),
        "per_record": min_dist.tolist(),
    }


def tab_leak_accuracy(
    x_true: Tensor,
    x_pred: Tensor,
    num_cols: list[str],
    cat_cols: list[str],
    cat_categories: Dict,
    num_std: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """TabLeak per-row accuracy with numeric tolerance and categorical exact match."""
    gt = x_true.detach().cpu().numpy()
    pred = x_pred.detach().cpu().numpy()
    num_count = len(num_cols)

    if num_count > 0:
        gt_num = gt[:, :num_count]
        pred_num = pred[:, :num_count]
        if num_std is not None and num_std.size >= num_count:
            eps = 0.319 * num_std[:num_count]
        else:
            eps = np.full(num_count, 0.319, dtype=np.float32)
        num_hits = (np.abs(gt_num - pred_num) <= eps[None, :]).sum(axis=1).astype(float)
    else:
        num_hits = np.zeros(gt.shape[0], dtype=float)

    cat_hits = np.zeros(gt.shape[0], dtype=float)
    current_idx = num_count
    for col in cat_cols:
        cats = cat_categories.get(col, [])
        if isinstance(cats, dict):
            cats = cats.get("categories", [])
        n_cats = len(cats)
        if n_cats <= 0 or current_idx + n_cats > gt.shape[1]:
            continue
        idxs = list(range(current_idx, current_idx + n_cats))
        cat_hits += (np.argmax(gt[:, idxs], axis=1) == np.argmax(pred[:, idxs], axis=1)).astype(float)
        current_idx += n_cats

    denom = num_count + len(cat_cols)
    if denom == 0:
        raise ValueError("Need at least one numeric or categorical feature to compute TabLeak accuracy.")
    per_row = (num_hits + cat_hits) / denom
    return per_row, float(per_row.mean())


def write_results_table(
    results_path: str,
    orig_tensor: Tensor,
    recon_tensor: Tensor,
    num_cols: list[str],
    cat_cols: list[str],
    cat_categories: Dict,
    num_eps: np.ndarray,
    label_tensor: Tensor | None,
    per_row_acc: np.ndarray,
    nn_batch: Dict,
    metrics: Dict,
) -> None:
    """Write per-row reconstruction tables plus batch summary."""
    total_dim = orig_tensor.shape[1]
    num_count = len(num_cols)
    orig_num_raw = orig_tensor[:, :num_count].numpy() if num_count > 0 else np.zeros((orig_tensor.shape[0], 0), dtype=np.float32)
    recon_num_raw = recon_tensor[:, :num_count].numpy() if num_count > 0 else np.zeros((orig_tensor.shape[0], 0), dtype=np.float32)

    cat_groups: list[dict] = []
    current_idx = num_count
    for col in cat_cols:
        cats = cat_categories.get(col, [])
        if isinstance(cats, dict):
            cats = cats.get("categories", [])
        n_cats = len(cats)
        if current_idx + n_cats <= total_dim and n_cats > 0:
            cat_groups.append({"name": col, "indices": list(range(current_idx, current_idx + n_cats)), "cats": cats})
            current_idx += n_cats
        else:
            if current_idx < total_dim:
                cat_groups.append({"name": col, "indices": [current_idx], "cats": cats})
                current_idx += 1

    with open(results_path, "w", encoding="utf-8") as f:
        for row_idx in range(orig_tensor.shape[0]):
            row_values = []
            recon_values = []
            acc_values = []

            if num_count > 0:
                for j in range(num_count):
                    row_values.append(orig_num_raw[row_idx, j])
                    recon_values.append(recon_num_raw[row_idx, j])
                    acc_values.append(float(abs(orig_num_raw[row_idx, j] - recon_num_raw[row_idx, j]) <= num_eps[j]))

            for group in cat_groups:
                idxs = group["indices"]
                cats = group["cats"]
                if len(idxs) == 1:
                    true_code = int(round(orig_tensor[row_idx, idxs[0]].item()))
                    pred_code = int(round(recon_tensor[row_idx, idxs[0]].item()))
                else:
                    true_code = int(torch.argmax(orig_tensor[row_idx, idxs]).item())
                    pred_code = int(torch.argmax(recon_tensor[row_idx, idxs]).item())
                true_label = cats[true_code] if 0 <= true_code < len(cats) else true_code
                pred_label = cats[pred_code] if 0 <= pred_code < len(cats) else pred_code
                row_values.append(true_label)
                recon_values.append(pred_label)
                acc_values.append(float(true_code == pred_code))

            if label_tensor is not None:
                true_label = label_tensor[row_idx].item()
                row_values.append(true_label)
                recon_values.append(true_label)
                acc_values.append(1.0)

            total_acc = float(per_row_acc[row_idx]) if row_idx < len(per_row_acc) else float("nan")
            headers = num_cols + [g["name"] for g in cat_groups]
            if label_tensor is not None:
                headers.append("target")
            headers.append("total_acc")

            rows = [
                ["Original", *row_values, ""],
                ["Reconstruction", *recon_values, ""],
                ["TabLeak accuracy", *acc_values, total_acc],
            ]
            f.write(f"Reconstruction {row_idx + 1}\n")
            f.write(render_table(rows, ["", *headers]))
            f.write("\n\n")

        batch_acc = float(per_row_acc.mean()) if len(per_row_acc) else float("nan")
        final_headers = ["batch_acc", "nn_mean", "nn_median", "nn_min", "mae", "rmse", "categorical_acc"]
        final_rows = [[
            batch_acc,
            nn_batch["mean"],
            nn_batch["median"],
            nn_batch["min"],
            metrics.get("mae", float("nan")),
            metrics.get("rmse", float("nan")),
            metrics.get("categorical_accuracy", float("nan")),
        ]]
        f.write("Batch reconstruction results\n")
        f.write(render_table(final_rows, final_headers))
        f.write("\n")


def write_results_table_rows(
    results_path: str,
    orig_tensor: Tensor,
    recon_tensor: Tensor,
    num_cols: list[str],
    cat_cols: list[str],
    cat_categories: Dict,
    num_eps: np.ndarray,
    label_tensor: Tensor | None,
    per_row_acc: np.ndarray,
) -> None:
    """Write per-row reconstruction tables only (no aggregate summary)."""
    total_dim = orig_tensor.shape[1]
    num_count = len(num_cols)
    orig_num_raw = orig_tensor[:, :num_count].numpy() if num_count > 0 else np.zeros((orig_tensor.shape[0], 0), dtype=np.float32)
    recon_num_raw = recon_tensor[:, :num_count].numpy() if num_count > 0 else np.zeros((orig_tensor.shape[0], 0), dtype=np.float32)

    cat_groups: list[dict] = []
    current_idx = num_count
    for col in cat_cols:
        cats = cat_categories.get(col, [])
        if isinstance(cats, dict):
            cats = cats.get("categories", [])
        n_cats = len(cats)
        if current_idx + n_cats <= total_dim and n_cats > 0:
            cat_groups.append({"name": col, "indices": list(range(current_idx, current_idx + n_cats)), "cats": cats})
            current_idx += n_cats
        else:
            if current_idx < total_dim:
                cat_groups.append({"name": col, "indices": [current_idx], "cats": cats})
                current_idx += 1

    with open(results_path, "w", encoding="utf-8") as f:
        for row_idx in range(orig_tensor.shape[0]):
            row_values = []
            recon_values = []
            acc_values = []

            if num_count > 0:
                for j in range(num_count):
                    row_values.append(orig_num_raw[row_idx, j])
                    recon_values.append(recon_num_raw[row_idx, j])
                    acc_values.append(float(abs(orig_num_raw[row_idx, j] - recon_num_raw[row_idx, j]) <= num_eps[j]))

            for group in cat_groups:
                idxs = group["indices"]
                cats = group["cats"]
                if len(idxs) == 1:
                    true_code = int(round(orig_tensor[row_idx, idxs[0]].item()))
                    pred_code = int(round(recon_tensor[row_idx, idxs[0]].item()))
                else:
                    true_code = int(torch.argmax(orig_tensor[row_idx, idxs]).item())
                    pred_code = int(torch.argmax(recon_tensor[row_idx, idxs]).item())
                true_label = cats[true_code] if 0 <= true_code < len(cats) else true_code
                pred_label = cats[pred_code] if 0 <= pred_code < len(cats) else pred_code
                row_values.append(true_label)
                recon_values.append(pred_label)
                acc_values.append(float(true_code == pred_code))

            if label_tensor is not None:
                true_label = label_tensor[row_idx].item()
                row_values.append(true_label)
                recon_values.append(true_label)
                acc_values.append(1.0)

            total_acc = float(per_row_acc[row_idx]) if row_idx < len(per_row_acc) else float("nan")
            headers = num_cols + [g["name"] for g in cat_groups]
            if label_tensor is not None:
                headers.append("target")
            headers.append("total_acc")

            rows = [
                ["Original", *row_values, ""],
                ["Reconstruction", *recon_values, ""],
                ["TabLeak accuracy", *acc_values, total_acc],
            ]
            f.write(f"Reconstruction {row_idx + 1}\n")
            f.write(render_table(rows, ["", *headers]))
            f.write("\n\n")


def evaluate_batch_rows(
    attacker: object,
    feature_schema: Dict,
    results_path: str,
    client_idx: int,
) -> Dict:
    """Evaluate a single attack batch and write per-row reconstruction tables."""
    num_cols = feature_schema.get("num_cols", [])
    cat_cols = feature_schema.get("cat_cols", [])
    cat_categories = feature_schema.get("cat_categories", {}) or {}

    if getattr(attacker, "best_reconstruction", None):
        recon_tensor = torch.cat([batch[0] for batch in attacker.best_reconstruction], dim=0).detach().cpu()
    else:
        recon_tensor = torch.zeros_like(attacker.original.detach().cpu())
    orig_tensor = attacker.original.detach().cpu()

    labels = getattr(attacker, "reconstruction_labels", None)
    if labels is None:
        label_tensor = None
    elif isinstance(labels, list):
        label_tensor = torch.stack(labels).view(-1).detach().cpu()
    else:
        label_tensor = torch.as_tensor(labels).view(-1).detach().cpu()

    orig_tensor, recon_tensor, label_tensor = apply_matching(orig_tensor, recon_tensor, label_tensor)

    num_count = len(num_cols)
    if num_count > 0:
        means = feature_schema.get("client_num_mean", [])
        stds = feature_schema.get("client_num_std", [])
        if client_idx < len(means) and client_idx < len(stds):
            mean = torch.tensor(means[client_idx], dtype=orig_tensor.dtype)
            std = torch.tensor(stds[client_idx], dtype=orig_tensor.dtype)
            orig_tensor = orig_tensor.clone()
            recon_tensor = recon_tensor.clone()
            orig_tensor[:, :num_count] = (orig_tensor[:, :num_count] * std) + mean
            recon_tensor[:, :num_count] = (recon_tensor[:, :num_count] * std) + mean
            num_std = std.detach().cpu().numpy()
            num_eps = 0.319 * num_std
        else:
            num_std = None
            num_eps = np.full(num_count, 0.319, dtype=np.float32)
    else:
        num_std = None
        num_eps = np.array([], dtype=np.float32)

    per_row_acc, batch_acc = tab_leak_accuracy(
        orig_tensor,
        recon_tensor,
        num_cols,
        cat_cols,
        cat_categories,
        num_std,
    )

    metrics = compute_metrics(orig_tensor, recon_tensor, feature_schema)

    write_results_table_rows(
        results_path,
        orig_tensor,
        recon_tensor,
        num_cols,
        cat_cols,
        cat_categories,
        num_eps,
        label_tensor,
        per_row_acc,
    )
    return {
        "batch_acc": float(batch_acc),
        "mae": float(metrics.get("mae", 0.0)),
        "rmse": float(metrics.get("rmse", 0.0)),
        "categorical_accuracy": float(metrics.get("categorical_accuracy", 0.0)),
    }

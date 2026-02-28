"""Evaluation metrics for tabular gradient inversion attacks."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tabulate import tabulate
from scipy.optimize import linear_sum_assignment
from tabular_gia.fl.dataloader.tabular_dataloader import denormalize_numeric


def _format_value(value: object) -> str:
    """Format values for table output."""
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _render_table(rows: list[list[object]], headers: list[str]) -> str:
    """Render table with tabulate."""
    rows_fmt = [[_format_value(cell) for cell in row] for row in rows]
    headers_fmt = [_format_value(h) for h in headers]
    return tabulate(rows_fmt, headers=headers_fmt, tablefmt="grid", stralign="center")


def _hungarian_match(orig: Tensor, recon: Tensor) -> np.ndarray:
    """Hungarian matching on L1 cost between rows."""
    a = orig.detach().cpu().numpy()
    b = recon.detach().cpu().numpy()
    cost = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind


def _apply_matching(orig: Tensor, recon: Tensor, labels: Tensor | None) -> tuple[Tensor, Tensor, Tensor | None]:
    """Reorder recon (and labels) to best match orig rows."""
    perm = _hungarian_match(orig, recon)
    recon = recon[perm]
    if labels is not None:
        labels = labels[perm]
    return orig, recon, labels


def prepare_tensors_for_metrics(
    attacker: object,
    feature_schema: Dict,
    client_idx: int,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Prepare matched and denormalized tensors for reconstruction metrics."""
    num_cols = feature_schema.get("num_cols", [])

    orig_tensor = attacker.original.detach().cpu()
    recon_tensor = torch.cat([batch[0] for batch in attacker.best_reconstruction], dim=0).detach().cpu()

    labels = attacker.reconstruction_labels
    if labels is None:
        label_tensor = None
    elif isinstance(labels, list):
        label_tensor = torch.stack(labels).view(-1).detach().cpu()
    else:
        label_tensor = torch.as_tensor(labels).view(-1).detach().cpu()

    orig_tensor, recon_tensor, label_tensor = _apply_matching(orig_tensor, recon_tensor, label_tensor)

    num_count = len(num_cols)
    if num_count > 0:
        means = feature_schema.get("client_num_mean", [])
        stds = feature_schema.get("client_num_std", [])
        mean = means[client_idx]
        std = stds[client_idx]
        orig_tensor = denormalize_numeric(orig_tensor, num_count, mean, std)
        recon_tensor = denormalize_numeric(recon_tensor, num_count, mean, std)
    return orig_tensor, recon_tensor, label_tensor


def compute_reconstruction_metrics(
    orig_tensor: Tensor,
    recon_tensor: Tensor,
    feature_schema: Dict,
    client_idx: int,
) -> Dict:
    """Compute per-row and aggregate reconstruction metrics."""
    num_cols = feature_schema.get("num_cols", [])
    cat_cols = feature_schema.get("cat_cols", [])
    cat_categories = feature_schema.get("cat_categories", {}) or {}
    num_count = len(num_cols)
    has_num = num_count > 0
    has_cat = len(cat_cols) > 0
    num_std = None
    if has_num:
        stds = feature_schema.get("client_num_std", [])
        num_std = np.asarray(stds[client_idx], dtype=np.float32)

    num_acc, cat_acc, tableak_acc = _tab_leak_accuracy(
        orig_tensor,
        recon_tensor,
        num_cols,
        cat_cols,
        cat_categories,
        num_std,
    )
    nn_dist = _nearest_neighbor_distance(
        recon_tensor,
        orig_tensor,
        {"num_cols": num_cols, "cat_cols": cat_cols, "cat_categories": cat_categories},
    )

    per_row_metrics = {
        "tableak_acc": tableak_acc,
        "nn_dist": nn_dist,
    }
    if has_num:
        per_row_metrics["num_acc"] = num_acc
    if has_cat:
        per_row_metrics["cat_acc"] = cat_acc

    prior_metrics = _prior_baseline_metrics(orig_tensor, feature_schema, client_idx, num_std)
    distribution_metrics = _distribution_confidence_metrics(recon_tensor, feature_schema, client_idx)

    aggregate_metrics = {
        "tableak_acc": float(np.nanmean(tableak_acc)),
        "emr": _emr(tableak_acc),
        "emr_90": _emr(tableak_acc, 0.9),
        "emr_80": _emr(tableak_acc, 0.8),
        "emr_60": _emr(tableak_acc, 0.6),
        "nn_mean": float(np.mean(nn_dist)),
        "nn_median": float(np.median(nn_dist)),
        "nn_min": float(np.min(nn_dist)),
        "num_rows": int(orig_tensor.shape[0]),
        "prior_tableak_acc": prior_metrics["prior_tableak_acc"],
        "gain_tableak_over_prior": float(np.nanmean(tableak_acc)) - prior_metrics["prior_tableak_acc"],
        "dist_conf": distribution_metrics["dist_conf"],
    }
    if has_num:
        aggregate_metrics["num_acc"] = float(np.mean(num_acc))
        if "prior_num_acc" in prior_metrics:
            aggregate_metrics["prior_num_acc"] = prior_metrics["prior_num_acc"]
            aggregate_metrics["gain_num_over_prior"] = aggregate_metrics["num_acc"] - prior_metrics["prior_num_acc"]
        aggregate_metrics["num_dist_conf"] = distribution_metrics["num_dist_conf"]
        aggregate_metrics["num_within_1std"] = distribution_metrics["num_within_1std"]
        aggregate_metrics["num_within_2std"] = distribution_metrics["num_within_2std"]
    if has_cat:
        aggregate_metrics["cat_acc"] = float(np.mean(cat_acc))
        if "prior_cat_acc" in prior_metrics:
            aggregate_metrics["prior_cat_acc"] = prior_metrics["prior_cat_acc"]
            aggregate_metrics["gain_cat_over_prior"] = aggregate_metrics["cat_acc"] - prior_metrics["prior_cat_acc"]
        aggregate_metrics["cat_dist_conf"] = distribution_metrics["cat_dist_conf"]

    return {
        "per_row_metrics": per_row_metrics,
        "aggregate_metrics": aggregate_metrics,
    }


def _emr(tableak_acc: np.ndarray, min_tableak_acc: float = 1.0) -> float:
    if min_tableak_acc >= 1.0:
        return float(np.mean(np.isclose(tableak_acc, 1.0)))
    return float(np.mean(tableak_acc >= min_tableak_acc))


def _iter_cat_groups(feature_schema: Dict, total_dim: int, num_count: int) -> list[dict]:
    cat_cols = feature_schema.get("cat_cols", [])
    cat_categories = feature_schema.get("cat_categories", {}) or {}
    groups: list[dict] = []
    current_idx = num_count
    for col in cat_cols:
        cats = cat_categories.get(col, [])
        if isinstance(cats, dict):
            cats = cats.get("categories", [])
        n_cats = len(cats)
        if n_cats <= 0 or current_idx + n_cats > total_dim:
            continue
        groups.append({"name": col, "indices": list(range(current_idx, current_idx + n_cats)), "cats": cats})
        current_idx += n_cats
    return groups


def _prior_baseline_metrics(
    orig_tensor: Tensor,
    feature_schema: Dict,
    client_idx: int,
    num_std: np.ndarray | None,
) -> Dict[str, float]:
    """Compute prior-only baseline reconstruction scores and attack gains."""
    num_cols = feature_schema.get("num_cols", [])
    cat_cols = feature_schema.get("cat_cols", [])
    num_count = len(num_cols)
    prior_tensor = torch.zeros_like(orig_tensor)

    # Numeric prior: client mean for each numerical feature.
    if num_count > 0:
        means = feature_schema.get("client_num_mean", [])
        if client_idx < len(means):
            mean_arr = np.asarray(means[client_idx], dtype=np.float32)[:num_count]
            prior_tensor[:, :num_count] = torch.as_tensor(mean_arr, dtype=orig_tensor.dtype, device=orig_tensor.device)

    # Categorical prior: client marginal mode for each categorical feature.
    client_cat_probs = feature_schema.get("client_cat_probs", [])
    probs_map = client_cat_probs[client_idx] if client_idx < len(client_cat_probs) else {}
    cat_groups = _iter_cat_groups(feature_schema, orig_tensor.shape[1], num_count)
    for group in cat_groups:
        idxs = group["indices"]
        probs = np.asarray(probs_map.get(group["name"], []), dtype=np.float32)
        if probs.size != len(idxs) or float(np.sum(probs)) <= 0:
            probs = np.full(len(idxs), 1.0 / len(idxs), dtype=np.float32)
        best = int(np.argmax(probs))
        prior_tensor[:, idxs] = 0.0
        prior_tensor[:, idxs[best]] = 1.0

    prior_num_acc, prior_cat_acc, prior_tableak = _tab_leak_accuracy(
        orig_tensor,
        prior_tensor,
        num_cols,
        cat_cols,
        feature_schema.get("cat_categories", {}) or {},
        num_std,
    )
    metrics: Dict[str, float] = {
        "prior_tableak_acc": float(np.nanmean(prior_tableak)),
    }
    if num_count > 0:
        metrics["prior_num_acc"] = float(np.mean(prior_num_acc))
    if len(cat_cols) > 0:
        metrics["prior_cat_acc"] = float(np.mean(prior_cat_acc))
    return metrics


def _distribution_confidence_metrics(
    recon_tensor: Tensor,
    feature_schema: Dict,
    client_idx: int,
) -> Dict[str, float]:
    """Estimate how likely the reconstructed batch is under client feature marginals."""
    num_cols = feature_schema.get("num_cols", [])
    num_count = len(num_cols)
    cat_groups = _iter_cat_groups(feature_schema, recon_tensor.shape[1], num_count)
    recon_np = recon_tensor.detach().cpu().numpy()

    num_within_1std = float("nan")
    num_within_2std = float("nan")
    num_conf = float("nan")
    if num_count > 0:
        means = feature_schema.get("client_num_mean", [])
        stds = feature_schema.get("client_num_std", [])
        if client_idx < len(means) and client_idx < len(stds):
            mean = np.asarray(means[client_idx], dtype=np.float32)[:num_count]
            std = np.asarray(stds[client_idx], dtype=np.float32)[:num_count]
            std = np.where(std <= 1e-8, 1e-8, std)
            z = np.abs((recon_np[:, :num_count] - mean[None, :]) / std[None, :])
            num_within_1std = float(np.mean(z <= 1.0))
            num_within_2std = float(np.mean(z <= 2.0))
            # Gaussian-kernel confidence in [0, 1]
            num_conf = float(np.mean(np.exp(-0.5 * (z ** 2))))

    cat_conf = float("nan")
    if cat_groups:
        client_cat_probs = feature_schema.get("client_cat_probs", [])
        probs_map = client_cat_probs[client_idx] if client_idx < len(client_cat_probs) else {}
        probs_list: list[np.ndarray] = []
        for group in cat_groups:
            idxs = group["indices"]
            probs = np.asarray(probs_map.get(group["name"], []), dtype=np.float32)
            if probs.size != len(idxs) or float(np.sum(probs)) <= 0:
                probs = np.full(len(idxs), 1.0 / len(idxs), dtype=np.float32)
            else:
                probs = probs / float(np.sum(probs))
            pred = np.argmax(recon_np[:, idxs], axis=1)
            probs_list.append(probs[pred])
        if probs_list:
            cat_conf = float(np.mean(np.concatenate([p.reshape(-1) for p in probs_list])))

    combined = []
    if not np.isnan(num_conf):
        combined.append(num_conf)
    if not np.isnan(cat_conf):
        combined.append(cat_conf)
    dist_conf = float(np.mean(combined)) if combined else float("nan")

    return {
        "dist_conf": dist_conf,
        "num_dist_conf": num_conf,
        "cat_dist_conf": cat_conf,
        "num_within_1std": num_within_1std,
        "num_within_2std": num_within_2std,
    }


def _weighted_metric_means(
    records: List[Dict],
    weights: np.ndarray,
    excluded_keys: set[str],
) -> Dict[str, float]:
    metric_keys = [k for k in records[0].keys() if k not in excluded_keys]
    means: Dict[str, float] = {}
    for k in metric_keys:
        values = np.array([r[k] for r in records], dtype=float)
        valid = ~np.isnan(values)
        if not valid.any():
            means[k] = float("nan")
            continue
        if k == "nn_min":
            means[k] = float(np.min(values[valid]))
            continue
        w_valid = weights[valid]
        denom = float(np.sum(w_valid))
        if denom <= 0:
            means[k] = float("nan")
            continue
        means[k] = float(np.sum(values[valid] * w_valid) / denom)
    return means


def summarize_round(
    metrics_list: List[Dict],
    round_idx: int,
) -> Dict | None:
    if not metrics_list:
        return None

    rows = np.array([m["num_rows"] for m in metrics_list], dtype=float)
    total_rows = float(rows.sum()) if rows.size else 0.0
    weights = rows / total_rows if total_rows > 0 else np.zeros_like(rows)
    metric_means = _weighted_metric_means(metrics_list, weights, {"client_idx", "num_rows"})

    return {
        "round": int(round_idx),
        "num_clients": int(len(metrics_list)),
        "total_rows": int(total_rows),
        **metric_means,
    }


def summarize_run(round_summaries: List[Dict]) -> Dict | None:
    if not round_summaries:
        return None
    rows = np.array([r["total_rows"] for r in round_summaries], dtype=float)
    total_rows = float(rows.sum()) if rows.size else 0.0
    weights = rows / total_rows if total_rows > 0 else np.zeros_like(rows)
    metric_means = _weighted_metric_means(
        round_summaries,
        weights,
        {"round", "num_clients", "total_rows"},
    )
    return {
        "num_rounds": int(len(round_summaries)),
        "total_rows": int(total_rows),
        **metric_means,
    }


def _write_rows_csv(out_path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in fieldnames) + "\n")


def write_rounds_summary(results_dir: Path | str, round_summaries: List[Dict]) -> None:
    out_path = Path(results_dir) / "rounds_summary.csv"
    _write_rows_csv(out_path, round_summaries)


def write_run_summary(results_dir: Path | str, run_summary: Dict) -> None:
    out_path = Path(results_dir) / "run_summary.csv"
    summary_keys = list(run_summary.keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(summary_keys) + "\n")
        f.write(",".join(str(run_summary[k]) for k in summary_keys) + "\n")


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


def _nearest_neighbor_distance(
    reconstructed: Tensor,
    reference: Tensor | DataLoader,
    feature_info: Dict,
    num_weight: float = 0.5,
    cat_weight: float = 0.5,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Compute per-row nearest-neighbor distance to a reference set."""
    recon_np = reconstructed.detach().cpu().numpy()
    num_idx, cat_groups = _get_feature_groups(feature_info, recon_np.shape[1])
    n_num = len(num_idx)
    n_cat = len(cat_groups)

    if n_num == 0 and n_cat == 0:
        return np.full((recon_np.shape[0],), np.nan, dtype=np.float32)

    if n_num > 0 and n_cat > 0:
        weight_sum = num_weight + cat_weight
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

    return min_dist


def _tab_leak_accuracy(
    x_true: Tensor,
    x_pred: Tensor,
    num_cols: list[str],
    cat_cols: list[str],
    cat_categories: Dict,
    num_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row numeric, categorical, and TabLeak accuracies."""
    gt = x_true.detach().cpu().numpy()
    pred = x_pred.detach().cpu().numpy()
    num_count = len(num_cols)

    if num_count > 0:
        gt_num = gt[:, :num_count]
        pred_num = pred[:, :num_count]
        eps = 0.319 * np.asarray(num_std[:num_count], dtype=np.float32)
        num_hits = (np.abs(gt_num - pred_num) <= eps[None, :]).sum(axis=1).astype(float)
        num_acc = num_hits / num_count
    else:
        num_hits = np.zeros(gt.shape[0], dtype=float)
        num_acc = np.zeros(gt.shape[0], dtype=float)

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

    if len(cat_cols) > 0:
        cat_acc = cat_hits / len(cat_cols)
    else:
        cat_acc = np.full(gt.shape[0], np.nan, dtype=float)

    denom = num_count + len(cat_cols)
    if denom > 0:
        tableak_acc = (num_hits + cat_hits) / denom
    else:
        tableak_acc = np.full(gt.shape[0], np.nan, dtype=float)
    return num_acc, cat_acc, tableak_acc


def write_debug_reconstruction_txt(
    results_path: str | Path,
    orig_tensor: Tensor,
    recon_tensor: Tensor,
    per_row_metrics: Dict,
    feature_schema: Dict,
    client_idx: int,
    label_tensor: Tensor | None = None,
) -> None:
    """Write human-readable per-row reconstruction tables."""
    num_cols = feature_schema.get("num_cols", [])
    cat_cols = feature_schema.get("cat_cols", [])
    cat_categories = feature_schema.get("cat_categories", {}) or {}
    num_std = None
    num_count = len(num_cols)
    if num_count > 0:
        stds = feature_schema.get("client_num_std", [])
        num_std = np.asarray(stds[client_idx], dtype=np.float32)
        num_eps = 0.319 * num_std[:num_count]
    else:
        num_eps = np.array([], dtype=np.float32)
    per_row_acc = np.asarray(per_row_metrics["tableak_acc"], dtype=float)

    total_dim = orig_tensor.shape[1]
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
            f.write(_render_table(rows, ["", *headers]))
            f.write("\n\n")

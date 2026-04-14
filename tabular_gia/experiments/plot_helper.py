from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

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
DEFAULT_BOUNDED_METRIC_YMAX = 1.0
PLOT_LINEWIDTH = 0.5
PLOT_MARKERSIZE = 4.0
PLOT_SCATTER_SIZE = 20.0
PLOT_MARKER_EDGECOLOR = "none"
PLOT_MARKER_EDGEWIDTH = 0.0
PLOT_LINE_ZORDER = 3.0
PLOT_MARKER_ZORDER = 4.0
PLOT_BAND_ZORDER = 1.0
ATTACK_METRIC_LABEL = "Reconstruction Accuracy"
SMALL_GROUP_COLORS_2 = ["#004488", "#BB5566"]
SMALL_GROUP_COLORS_3 = ["#004488", "#BB5566", "#DDAA33"]
GROUP_COLORMAP_NAME = "viridis"
CHECKPOINT_COLORMAP_NAME = "cividis"
GROUP_COLORMAP_RANGE = (0.0, 0.9)
CHECKPOINT_COLORMAP_RANGE = (0.0, 0.8)
DEFAULT_RESULTS_ROOTS = (
    Path("tabular_gia/results/experiments"),
    Path("results/experiments"),
)
DEFAULT_LEGEND_FRAME_ALPHA = 0.92


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


def resolve_experiment_dir(results_root: str, experiment_dir: str) -> Path:
    path = Path(experiment_dir)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {path}")
        return path

    if results_root:
        candidate = Path(results_root) / experiment_dir
        if not candidate.exists():
            raise FileNotFoundError(f"Experiment directory not found: {candidate}")
        return candidate

    for root in DEFAULT_RESULTS_ROOTS:
        candidate = root / experiment_dir
        if candidate.exists():
            return candidate

    searched = ", ".join(str(root / experiment_dir) for root in DEFAULT_RESULTS_ROOTS)
    raise FileNotFoundError(f"Experiment directory not found. Checked: {searched}")


def get_nested(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def base_frame(metadata: dict[str, Any], size: int) -> pd.DataFrame:
    return pd.DataFrame({key: [value] * size for key, value in metadata.items()})


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


def load_basic_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    dataset_path = str(get_nested(cfg, "dataset.dataset_path", "unknown"))
    dataset_name = Path(dataset_path).stem
    model_name = str(get_nested(cfg, "model.preset") or get_nested(cfg, "model.arch") or "unknown")
    task_objective = infer_task_objective(dataset_name=dataset_name, dataset_path=dataset_path)
    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "batch_size": int(get_nested(cfg, "dataset.batch_size")),
        "model_name": model_name,
        "task_objective": task_objective,
        "utility_metric": DEFAULT_Y_BY_TASK.get(task_objective, "val_loss"),
        "run_id": run_dir.name,
    }


def metric_for_final_test(metric: str) -> str:
    if metric.startswith("val_"):
        return "test_" + metric[len("val_"):]
    return metric


def sanitize_for_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or "unknown"


def set_plot_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": PLOT_LINEWIDTH,
            "lines.scale_dashes": False,
            "lines.dashed_pattern": [3.0, 1.0],
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.xmargin": 0.0,
            "axes.ymargin": 0.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def style_legend(legend: Any) -> Any:
    if legend is not None:
        legend.get_frame().set_alpha(DEFAULT_LEGEND_FRAME_ALPHA)
    return legend


def sample_colormap(name: str, n: int, *, start: float, end: float) -> np.ndarray:
    cmap = plt.get_cmap(name)
    return cmap(np.linspace(start, end, max(1, n)))


def sample_group_colors(n: int) -> list[Any]:
    if n == 2:
        return SMALL_GROUP_COLORS_2
    if n == 3:
        return SMALL_GROUP_COLORS_3
    return list(sample_colormap(GROUP_COLORMAP_NAME, n, start=GROUP_COLORMAP_RANGE[0], end=GROUP_COLORMAP_RANGE[1]))


def sample_checkpoint_colors(n: int) -> np.ndarray:
    return sample_colormap(
        CHECKPOINT_COLORMAP_NAME,
        n,
        start=CHECKPOINT_COLORMAP_RANGE[0],
        end=CHECKPOINT_COLORMAP_RANGE[1],
    )


def plot_group_styles(groups: list[int | str]) -> dict[int | str, tuple[Any, str]]:
    ordered = ordered_plot_groups(groups)
    colors = sample_group_colors(len(ordered))
    return {group: (colors[idx], PLOT_MARKERS[idx % len(PLOT_MARKERS)]) for idx, group in enumerate(ordered)}


def batch_styles(batch_sizes: list[int]) -> dict[int, tuple[Any, str]]:
    return {
        int(group): (color, marker)
        for group, (color, marker) in plot_group_styles([int(batch_size) for batch_size in batch_sizes]).items()
    }


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
        return (lo, hi)
    return None


def is_bounded_fl_metric(metric: str) -> bool:
    return metric in FL_BOUNDED_METRICS


def ordered_model_names(model_names: list[str]) -> list[str]:
    preferred = ["small", "resnet", "fttransformer"]
    seen = set(model_names)
    ordered = [name for name in preferred if name in seen]
    ordered.extend(sorted(name for name in model_names if name not in preferred))
    return ordered


def rename_plot_columns(rows: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    out = rows.rename(columns=rename_map).copy()
    sort_cols = [col for col in ("batch_size", "checkpoint_idx", "exp_min", "exp_min_mean") if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)


def annotate_checkpoint_progress(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    out = rows.sort_values(["dataset_name", "model_name", "batch_size", "run_id", "exp_min"]).reset_index(drop=True).copy()
    group_cols = ["dataset_name", "model_name", "batch_size", "run_id"]
    out["checkpoint_idx"] = out.groupby(group_cols).cumcount() + 1
    checkpoint_count = out.groupby(group_cols)["checkpoint_idx"].transform("max")
    out["checkpoint_progress_pct"] = np.where(
        checkpoint_count <= 1,
        100.0,
        ((out["checkpoint_idx"] - 1) / (checkpoint_count - 1)) * 100.0,
    )
    return out


def clip_ci_upper(y: np.ndarray, ci: np.ndarray, upper_bound: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    lower = y - ci
    upper = np.minimum(y + ci, upper_bound)
    return lower, upper


def clip_errorbar_upper(y: np.ndarray, ci: np.ndarray, upper_bound: float = 1.0) -> np.ndarray:
    lower_err = ci
    upper_err = np.maximum(0.0, np.minimum(ci, upper_bound - y))
    return np.vstack([lower_err, upper_err])


def ci_bounds(y: np.ndarray, ci: np.ndarray, *, bounded: bool) -> tuple[np.ndarray, np.ndarray]:
    if bounded:
        return clip_ci_upper(y, ci)
    return y - ci, y + ci


def errorbar_yerr(y: np.ndarray, ci: np.ndarray, *, bounded: bool) -> np.ndarray | np.ndarray:
    if bounded:
        return clip_errorbar_upper(y, ci)
    return ci


def plot_line_with_band(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    y: np.ndarray,
    ci: np.ndarray,
    color: Any,
    marker: str,
    linewidth: float,
    markersize: float,
    alpha: float,
    label: str | None = None,
    bounded: bool,
    linestyle: str = "--",
) -> None:
    lower, upper = ci_bounds(y, ci, bounded=bounded)
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        markeredgecolor=PLOT_MARKER_EDGECOLOR,
        markeredgewidth=PLOT_MARKER_EDGEWIDTH,
        label=label,
        zorder=PLOT_MARKER_ZORDER,
        clip_on=False,
    )
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0, zorder=PLOT_BAND_ZORDER)


def apply_axis_finish(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(x=0.0, y=0.0)
    if ylim is not None:
        ax.set_ylim(*ylim)
        if np.isclose(ylim[0], 0.0) and np.isclose(ylim[1], 1.0):
            ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.grid(True, which="major")


def prepare_plot_dirs(plots_dir: Path, *, with_root_dirs: bool = True) -> dict[str, Path]:
    dirs = {
        "data": plots_dir / "data",
        "image_singles": plots_dir / "image" / "singles",
        "image_panels": plots_dir / "image" / "panels",
        "pdf_singles": plots_dir / "pdf" / "singles",
        "pdf_panels": plots_dir / "pdf" / "panels",
    }
    if with_root_dirs:
        dirs["image"] = plots_dir / "image"
        dirs["pdf"] = plots_dir / "pdf"
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, png_base_path: Path, *, pdf_base_path: Path | None = None) -> tuple[Path, Path]:
    png_path = png_base_path.with_suffix(".png")
    pdf_path = (pdf_base_path or png_base_path).with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)

    suptitle_artist = getattr(fig, "_suptitle", None)
    suptitle_text = None
    if suptitle_artist is not None:
        suptitle_text = suptitle_artist.get_text()
        suptitle_artist.set_text("")

    single_axis_title = None
    if len(fig.axes) == 1:
        axis = fig.axes[0]
        single_axis_title = axis.get_title()
        axis.set_title("")

    fig.savefig(pdf_path)

    if suptitle_artist is not None and suptitle_text is not None:
        suptitle_artist.set_text(suptitle_text)
    if len(fig.axes) == 1 and single_axis_title is not None:
        fig.axes[0].set_title(single_axis_title)
    plt.close(fig)
    return png_path, pdf_path


def load_attack_curve(
    run_dir: Path,
    metadata: dict[str, Any],
    *,
    attack_metric_cols: list[str] | None = None,
) -> pd.DataFrame | None:
    stats_csv = run_dir / "aggregated" / "rounds_summary_stats.csv"
    summary_csv = run_dir / "aggregated" / "rounds_summary.csv"
    metric_cols = attack_metric_cols or [
        "tableak_acc",
        "num_acc",
        "cat_acc",
        "prior_tableak_acc",
        "prior_num_acc",
        "prior_cat_acc",
        "random_tableak_acc",
        "random_num_acc",
        "random_cat_acc",
    ]

    if stats_csv.exists():
        df = pd.read_csv(stats_csv)
        required = {"exp_min_mean", "tableak_acc_mean"}
        if not required.issubset(df.columns):
            return None
        out = pd.DataFrame({"exp_min": pd.to_numeric(df["exp_min_mean"], errors="coerce")})
        for metric in metric_cols:
            out[metric] = pd.to_numeric(df.get(f"{metric}_mean"), errors="coerce")
            out[f"{metric}_ci95"] = pd.to_numeric(df.get(f"{metric}_ci95"), errors="coerce")
    elif summary_csv.exists():
        df = pd.read_csv(summary_csv)
        required = {"exp_min", "tableak_acc"}
        if not required.issubset(df.columns):
            return None
        out = pd.DataFrame({"exp_min": pd.to_numeric(df["exp_min"], errors="coerce")})
        for metric in metric_cols:
            out[metric] = pd.to_numeric(df.get(metric), errors="coerce")
            out[f"{metric}_ci95"] = np.nan
    else:
        return None

    out = out.dropna(subset=["exp_min", "tableak_acc"]).sort_values("exp_min").reset_index(drop=True)
    if out.empty:
        return None
    return pd.concat([base_frame(metadata, len(out)), out], axis=1)


def load_utility_curve(run_dir: Path, metadata: dict[str, Any]) -> pd.DataFrame | None:
    metric = metadata["utility_metric"]
    stats_csv = run_dir / "aggregated" / "fl_stats.csv"
    fl_csv = run_dir / "aggregated" / "fl.csv"

    if stats_csv.exists():
        df = pd.read_csv(stats_csv)
        df = df[df["phase"] == "checkpoint"].copy()
        x_col = "exp_min_mean"
        y_col = f"{metric}_mean"
        ci_col = f"{metric}_ci95"
        if x_col not in df.columns or y_col not in df.columns:
            return None
        out = pd.DataFrame(
            {
                "exp_min": pd.to_numeric(df[x_col], errors="coerce"),
                "utility_value": pd.to_numeric(df[y_col], errors="coerce"),
                "utility_ci95": pd.to_numeric(df.get(ci_col), errors="coerce"),
            }
        )
    elif fl_csv.exists():
        df = pd.read_csv(fl_csv)
        df = df[df["phase"] == "checkpoint"].copy()
        if "exp_min" not in df.columns or metric not in df.columns:
            return None
        out = pd.DataFrame(
            {
                "exp_min": pd.to_numeric(df["exp_min"], errors="coerce"),
                "utility_value": pd.to_numeric(df[metric], errors="coerce"),
                "utility_ci95": np.nan,
            }
        )
    else:
        return None

    out = out.dropna(subset=["exp_min", "utility_value"]).sort_values("exp_min").reset_index(drop=True)
    if out.empty:
        return None
    return pd.concat([base_frame(metadata, len(out)), out], axis=1)


def load_final_test_point(run_dir: Path, metadata: dict[str, Any]) -> pd.DataFrame | None:
    metric = metric_for_final_test(str(metadata["utility_metric"]))
    stats_csv = run_dir / "aggregated" / "fl_stats.csv"
    fl_csv = run_dir / "aggregated" / "fl.csv"

    if stats_csv.exists():
        df = pd.read_csv(stats_csv)
        df = df[df["phase"] == "final_test"].copy()
        x_col = "exp_min_mean"
        y_col = f"{metric}_mean"
        ci_col = f"{metric}_ci95"
        if x_col not in df.columns or y_col not in df.columns or df.empty:
            return None
        out = pd.DataFrame(
            {
                "exp_min": pd.to_numeric(df[x_col], errors="coerce"),
                "utility_value": pd.to_numeric(df[y_col], errors="coerce"),
                "utility_ci95": pd.to_numeric(df.get(ci_col), errors="coerce"),
                "utility_metric": metric,
            }
        )
    elif fl_csv.exists():
        df = pd.read_csv(fl_csv)
        df = df[df["phase"] == "final_test"].copy()
        if "exp_min" not in df.columns or metric not in df.columns or df.empty:
            return None
        out = pd.DataFrame(
            {
                "exp_min": pd.to_numeric(df["exp_min"], errors="coerce"),
                "utility_value": pd.to_numeric(df[metric], errors="coerce"),
                "utility_ci95": np.nan,
                "utility_metric": metric,
            }
        )
    else:
        return None

    out = out.dropna(subset=["exp_min", "utility_value"]).sort_values("exp_min").reset_index(drop=True)
    if out.empty:
        return None
    base = base_frame(metadata, len(out))
    base["utility_metric"] = [metric] * len(out)
    return pd.concat([base, out[["exp_min", "utility_value", "utility_ci95"]]], axis=1)


def interpolate_series(frame: pd.DataFrame, x_col: str, y_col: str, targets: np.ndarray) -> np.ndarray:
    src = frame[[x_col, y_col]].dropna().sort_values(x_col)
    if src.empty:
        return np.full(len(targets), np.nan, dtype=float)
    src = src.groupby(x_col, as_index=False).mean(numeric_only=True)
    x = src[x_col].to_numpy(dtype=float)
    y = src[y_col].to_numpy(dtype=float)
    if len(x) == 1:
        out = np.full(len(targets), y[0], dtype=float)
        out[(targets < x[0]) | (targets > x[0])] = np.nan
        return out

    out = np.interp(targets, x, y, left=np.nan, right=np.nan)
    out[(targets < x.min()) | (targets > x.max())] = np.nan
    return out


def align_attack_and_utility(
    attack_curve: pd.DataFrame,
    utility_curve: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    targets = attack_curve["exp_min"].to_numpy(dtype=float)
    utility_interp = interpolate_series(utility_curve, "exp_min", "utility_value", targets)
    out = attack_curve[["exp_min", "tableak_acc"]].copy()
    out["utility_value"] = utility_interp
    out = out.dropna(subset=["utility_value"]).reset_index(drop=True)
    if out.empty:
        return out
    return pd.concat([base_frame(metadata, len(out)), out], axis=1)


def sample_utility_at_attack_checkpoints(
    attack_rows: pd.DataFrame,
    utility_rows: pd.DataFrame,
) -> pd.DataFrame:
    if attack_rows.empty or utility_rows.empty:
        return pd.DataFrame()

    join_cols = ["dataset_name", "dataset_path", "batch_size", "model_name", "task_objective", "utility_metric", "run_id"]
    sampled_frames: list[pd.DataFrame] = []

    for key, attack_sub in attack_rows.groupby(join_cols, dropna=False, sort=False):
        utility_sub = utility_rows
        for col, value in zip(join_cols, key, strict=False):
            utility_sub = utility_sub[utility_sub[col] == value]
        if utility_sub.empty:
            continue

        utility_sub = utility_sub.sort_values("exp_min").reset_index(drop=True)
        utility_x = utility_sub["exp_min"].to_numpy(dtype=float)
        utility_y = utility_sub["utility_value"].to_numpy(dtype=float)
        utility_ci = utility_sub["utility_ci95"].to_numpy(dtype=float)

        attack_sub = attack_sub.sort_values("exp_min").reset_index(drop=True).copy()
        targets = attack_sub["exp_min"].to_numpy(dtype=float)
        nearest_idx = np.abs(utility_x[:, None] - targets[None, :]).argmin(axis=0)

        attack_sub["utility_value"] = utility_y[nearest_idx]
        attack_sub["utility_ci95"] = utility_ci[nearest_idx]
        attack_sub["utility_exp_min"] = utility_x[nearest_idx]
        sampled_frames.append(attack_sub)

    if not sampled_frames:
        return pd.DataFrame()
    return pd.concat(sampled_frames, axis=0, ignore_index=True)


def collect_run_tables(
    experiment_dir: Path,
    protocol_subdir: str,
    metadata_loader,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol_dir = experiment_dir / protocol_subdir
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    attack_frames: list[pd.DataFrame] = []
    utility_frames: list[pd.DataFrame] = []
    pareto_frames: list[pd.DataFrame] = []
    final_test_frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for run_dir in sorted(protocol_dir.glob("run_*")):
        metadata = metadata_loader(run_dir)
        if metadata is None:
            skipped.append(f"{run_dir.name}:missing_run_config")
            continue
        attack_curve = load_attack_curve(run_dir, metadata)
        utility_curve = load_utility_curve(run_dir, metadata)
        if attack_curve is None or utility_curve is None:
            skipped.append(f"{run_dir.name}:missing_attack_or_utility")
            continue

        attack_frames.append(attack_curve)
        utility_frames.append(utility_curve)
        final_test_point = load_final_test_point(run_dir, metadata)
        if final_test_point is not None:
            final_test_frames.append(final_test_point)

        pareto = align_attack_and_utility(attack_curve, utility_curve, metadata)
        if not pareto.empty:
            pareto_frames.append(pareto)

    if not attack_frames or not utility_frames:
        raise FileNotFoundError(f"No usable run curves found under {protocol_dir}")

    if skipped:
        print(f"Skipped {len(skipped)} run(s): {', '.join(skipped)}")

    return (
        pd.concat(attack_frames, axis=0, ignore_index=True),
        pd.concat(utility_frames, axis=0, ignore_index=True),
        pd.concat(pareto_frames, axis=0, ignore_index=True) if pareto_frames else pd.DataFrame(),
        pd.concat(final_test_frames, axis=0, ignore_index=True) if final_test_frames else pd.DataFrame(),
    )

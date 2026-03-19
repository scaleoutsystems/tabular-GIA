from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def _latest_experiment_dir(experiments_root: Path) -> Path:
    candidates = [p for p in experiments_root.glob("experiment_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No experiment_* folders found in {experiments_root}")
    return sorted(candidates)[-1]


def _load_runs(experiment_dir: Path) -> list[dict]:
    with open(experiment_dir / "sweep_runs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_round_exposure_tableak(rounds_csv: Path) -> tuple[list[int], list[float], list[float]]:
    rounds: list[int] = []
    exposures: list[float] = []
    values: list[float] = []
    with open(rounds_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(float(row["round"])))
            exposures.append(float(row["exp_min"]))
            values.append(float(row["tableak_acc"]))
    return rounds, exposures, values


def plot_tableak_vs_exposure(experiment_dir: Path, runs: list[dict], out_path: Path) -> Path:
    series: list[tuple[int, list[float], list[float]]] = []
    for run in runs:
        batch_size = int(run["overrides"]["dataset"]["batch_size"])
        run_dir = Path(run["run_dir"])
        rounds_csv = run_dir / "artifacts" / "rounds_summary.csv"
        if not rounds_csv.exists():
            continue
        _, exposures, values = _read_round_exposure_tableak(rounds_csv)
        series.append((batch_size, exposures, values))

    if not series:
        raise FileNotFoundError(f"No rounds_summary.csv files found under {experiment_dir}")

    series.sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for batch_size, exposures, values in series:
        ax.plot(exposures, values, marker="o", linewidth=1.5, label=f"batch_size={batch_size}")

    ax.set_xlabel("Exposure (exp_min)")
    ax.set_ylabel("TabLeak accuracy")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_fl_acc_vs_exposure(experiment_dir: Path, runs: list[dict], out_path: Path) -> Path:
    val_series: list[tuple[int, list[float], list[float]]] = []
    test_series: list[tuple[int, list[float], list[float]]] = []

    for run in runs:
        batch_size = int(run["overrides"]["dataset"]["batch_size"])
        fl_csv = Path(run["run_dir"]) / "artifacts" / "fl.csv"
        if not fl_csv.exists():
            continue

        rows = _read_csv_rows(fl_csv)
        val_x: list[float] = []
        val_y: list[float] = []
        test_x: list[float] = []
        test_y: list[float] = []

        for row in rows:
            exp_min = _to_float(row.get("exp_min", ""))
            if exp_min is None:
                continue
            phase = row.get("phase", "")
            if phase == "checkpoint":
                val_acc = _to_float(row.get("val_acc", ""))
                if val_acc is not None:
                    val_x.append(exp_min)
                    val_y.append(val_acc)
            elif phase == "final_test":
                test_acc = _to_float(row.get("test_acc", ""))
                if test_acc is not None:
                    test_x.append(exp_min)
                    test_y.append(test_acc)

        if val_x:
            order = sorted(range(len(val_x)), key=lambda i: val_x[i])
            val_series.append((batch_size, [val_x[i] for i in order], [val_y[i] for i in order]))
        if test_x:
            order = sorted(range(len(test_x)), key=lambda i: test_x[i])
            test_series.append((batch_size, [test_x[i] for i in order], [test_y[i] for i in order]))

    if not val_series and not test_series:
        raise FileNotFoundError(f"No usable FL metrics found under {experiment_dir}")

    val_series.sort(key=lambda t: t[0])
    test_series.sort(key=lambda t: t[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    for batch_size, x, y in val_series:
        ax1.plot(x, y, marker="o", linewidth=1.5, label=f"batch_size={batch_size}")
    ax1.set_ylabel("val_acc")
    ax1.set_title("Validation accuracy vs exp_min")
    ax1.legend()

    for batch_size, x, y in test_series:
        ax2.plot(x, y, marker="o", linewidth=1.5, label=f"batch_size={batch_size}")
    ax2.set_xlabel("Exposure (exp_min)")
    ax2.set_ylabel("test_acc")
    ax2.set_title("Test accuracy vs exp_min")
    ax2.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_accuracy_decomposition_by_batch(experiment_dir: Path, runs: list[dict], out_path: Path) -> Path:
    series: list[tuple[int, list[float], list[float], list[float], list[float]]] = []
    for run in runs:
        batch_size = int(run["overrides"]["dataset"]["batch_size"])
        rounds_csv = Path(run["run_dir"]) / "artifacts" / "rounds_summary.csv"
        if not rounds_csv.exists():
            continue

        rows = _read_csv_rows(rounds_csv)
        x: list[float] = []
        y_tableak: list[float] = []
        y_num: list[float] = []
        y_cat: list[float] = []

        for row in rows:
            exp_min = _to_float(row.get("exp_min", ""))
            tableak = _to_float(row.get("tableak_acc", ""))
            num_acc = _to_float(row.get("num_acc", ""))
            cat_acc = _to_float(row.get("cat_acc", ""))
            if exp_min is None or tableak is None:
                continue
            x.append(exp_min)
            y_tableak.append(tableak)
            y_num.append(float("nan") if num_acc is None else num_acc)
            y_cat.append(float("nan") if cat_acc is None else cat_acc)

        if x:
            order = sorted(range(len(x)), key=lambda i: x[i])
            series.append(
                (
                    batch_size,
                    [x[i] for i in order],
                    [y_tableak[i] for i in order],
                    [y_num[i] for i in order],
                    [y_cat[i] for i in order],
                )
            )

    if not series:
        raise FileNotFoundError(f"No usable rounds_summary.csv files found under {experiment_dir}")

    series.sort(key=lambda t: t[0])
    n = len(series)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharex=True, sharey=True)

    if isinstance(axes, plt.Axes):
        axes_flat = [axes]
    else:
        axes_flat = list(axes.flat)

    for ax, (batch_size, x, y_tableak, y_num, y_cat) in zip(axes_flat, series):
        ax.plot(x, y_tableak, marker="o", linewidth=1.5, label="tableak_acc")
        ax.plot(x, y_num, marker="o", linewidth=1.5, label="num_acc")
        ax.plot(x, y_cat, marker="o", linewidth=1.5, label="cat_acc")
        ax.set_title(f"batch_size={batch_size}")
        ax.grid(alpha=0.2)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for ax in axes_flat[:n]:
        ax.set_xlabel("Exposure (exp_min)")
        ax.set_ylabel("Accuracy")

    axes_flat[0].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot latest experiment metrics")
    parser.add_argument("--experiments-root", default="tabular_gia/results/experiments")
    parser.add_argument("--experiment-dir", default="", help="Optional explicit experiment directory")
    parser.add_argument("--plot", choices=["tableak", "fl", "decomp", "all"], default="all")
    parser.add_argument("--out", default="", help="Optional output path (only for single-plot modes)")
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else _latest_experiment_dir(experiments_root)
    runs = _load_runs(experiment_dir)

    if args.plot == "tableak":
        out_path = Path(args.out) if args.out else experiment_dir / "tableak_vs_exposure_by_batch.png"
        print(plot_tableak_vs_exposure(experiment_dir, runs, out_path))
        return

    if args.plot == "fl":
        out_path = Path(args.out) if args.out else experiment_dir / "fl_acc_vs_exposure.png"
        print(plot_fl_acc_vs_exposure(experiment_dir, runs, out_path))
        return

    if args.plot == "decomp":
        out_path = Path(args.out) if args.out else experiment_dir / "acc_decomposition_by_batch_vs_exposure.png"
        print(plot_accuracy_decomposition_by_batch(experiment_dir, runs, out_path))
        return

    print(plot_tableak_vs_exposure(experiment_dir, runs, experiment_dir / "tableak_vs_exposure_by_batch.png"))
    print(plot_fl_acc_vs_exposure(experiment_dir, runs, experiment_dir / "fl_acc_vs_exposure.png"))
    print(
        plot_accuracy_decomposition_by_batch(
            experiment_dir,
            runs,
            experiment_dir / "acc_decomposition_by_batch_vs_exposure.png",
        )
    )


if __name__ == "__main__":
    main()

"""Plot utility/leakage from sweep_results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _as_bool_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": "label-known", "false": "label-unknown"}).fillna(series.astype(str))


def _pick_default_leakage(df: pd.DataFrame) -> str:
    for candidate in ("tableak_acc", "gain_tableak_over_prior", "cat_acc", "num_acc"):
        if candidate in df.columns:
            return candidate
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for leakage plot.")
    return numeric_cols[0]


def _group_mean_std(df: pd.DataFrame, by: list[str], value: str) -> pd.DataFrame:
    grouped = df.groupby(by, dropna=False)[value].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _plot_protocol_label_bar(df: pd.DataFrame, leakage_col: str, out_dir: Path) -> None:
    if "protocol" not in df.columns or "label_known" not in df.columns:
        return

    plot_df = _group_mean_std(df, ["protocol", "label_known"], leakage_col)
    protocols = sorted(plot_df["protocol"].astype(str).unique().tolist())
    labels = ["label-known", "label-unknown"]

    x = np.arange(len(protocols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lbl in enumerate(labels):
        sub = plot_df[_as_bool_label(plot_df["label_known"]) == lbl]
        means = []
        stds = []
        for p in protocols:
            row = sub[sub["protocol"].astype(str) == p]
            means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            stds.append(float(row["std"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + (i - 0.5) * width, means, width=width, yerr=stds, capsize=3, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(protocols)
    ax.set_ylabel(leakage_col)
    ax.set_title(f"Leakage by Protocol and Label Assumption ({leakage_col})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "leakage_protocol_label_known.png", dpi=160)
    plt.close(fig)


def _plot_attack_budget(df: pd.DataFrame, leakage_col: str, out_dir: Path) -> None:
    if "at_iterations" not in df.columns:
        return
    vals = pd.to_numeric(df["at_iterations"], errors="coerce").dropna().unique()
    if len(vals) < 2:
        return

    plot_df = df.copy()
    plot_df["at_iterations"] = pd.to_numeric(plot_df["at_iterations"], errors="coerce")
    by_cols = ["at_iterations"]
    if "protocol" in plot_df.columns:
        by_cols.append("protocol")
    if "label_known" in plot_df.columns:
        by_cols.append("label_known")
    grouped = _group_mean_std(plot_df.dropna(subset=["at_iterations"]), by_cols, leakage_col)

    fig, ax = plt.subplots(figsize=(8, 5))
    if "protocol" in grouped.columns and "label_known" in grouped.columns:
        grouped["series"] = grouped["protocol"].astype(str) + " | " + _as_bool_label(grouped["label_known"])
    elif "protocol" in grouped.columns:
        grouped["series"] = grouped["protocol"].astype(str)
    elif "label_known" in grouped.columns:
        grouped["series"] = _as_bool_label(grouped["label_known"])
    else:
        grouped["series"] = "all"

    for series_name, g in grouped.groupby("series", dropna=False):
        g = g.sort_values("at_iterations")
        ax.plot(g["at_iterations"], g["mean"], marker="o", label=str(series_name))
        ax.fill_between(g["at_iterations"], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2)

    ax.set_xlabel("at_iterations")
    ax.set_ylabel(leakage_col)
    ax.set_title(f"Leakage vs Attack Iterations ({leakage_col})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "leakage_vs_at_iterations.png", dpi=160)
    plt.close(fig)


def _plot_utility_vs_leakage(df: pd.DataFrame, utility_col: str, leakage_col: str, out_dir: Path) -> None:
    if utility_col not in df.columns:
        return
    plot_df = df[[utility_col, leakage_col] + [c for c in ("protocol", "label_known") if c in df.columns]].copy()
    plot_df[utility_col] = pd.to_numeric(plot_df[utility_col], errors="coerce")
    plot_df[leakage_col] = pd.to_numeric(plot_df[leakage_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[utility_col, leakage_col])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    if "label_known" in plot_df.columns:
        label_text = _as_bool_label(plot_df["label_known"])
    else:
        label_text = pd.Series(["run"] * len(plot_df))
    if "protocol" in plot_df.columns:
        colors = pd.Categorical(plot_df["protocol"]).codes
    else:
        colors = np.zeros(len(plot_df))
    markers = np.where(label_text == "label-known", "o", "x")

    for marker in ("o", "x"):
        mask = markers == marker
        ax.scatter(
            plot_df.loc[mask, utility_col],
            plot_df.loc[mask, leakage_col],
            c=colors[mask],
            cmap="tab10",
            alpha=0.8,
            marker=marker,
            label="label-known" if marker == "o" else "label-unknown",
        )

    ax.set_xlabel(utility_col)
    ax.set_ylabel(leakage_col)
    ax.set_title(f"Utility vs Leakage ({utility_col} vs {leakage_col})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "utility_vs_leakage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument("--csv", required=True, help="Path to sweep_results.csv")
    parser.add_argument("--out", required=True, help="Output directory for plots")
    parser.add_argument("--leakage", default="", help="Leakage metric column (auto if omitted)")
    parser.add_argument("--utility", default="", help="Utility metric column (optional)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    leakage_col = args.leakage or _pick_default_leakage(df)
    if leakage_col not in df.columns:
        raise ValueError(f"Leakage column not found: {leakage_col}")

    _plot_protocol_label_bar(df, leakage_col, out_dir)
    _plot_attack_budget(df, leakage_col, out_dir)
    if args.utility:
        _plot_utility_vs_leakage(df, args.utility, leakage_col, out_dir)

    # Save an aggregated table for quick downstream analysis.
    group_cols = [c for c in ("protocol", "label_known", "partition_strategy", "dirichlet_alpha", "num_clients", "seed") if c in df.columns]
    agg = _group_mean_std(df, group_cols if group_cols else ["run_id"], leakage_col)
    agg.to_csv(out_dir / "leakage_grouped_stats.csv", index=False)


if __name__ == "__main__":
    main()

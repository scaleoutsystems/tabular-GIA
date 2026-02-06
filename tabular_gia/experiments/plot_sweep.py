"""Plot utility and leakage from sweep CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep results")
    parser.add_argument("--csv", required=True, help="Path to sweep_results.csv")
    parser.add_argument("--out", required=True, help="Output directory for plots")
    parser.add_argument("--x", default="model.name", help="Column to use on x-axis")
    parser.add_argument("--utility", default="test_acc", help="Utility metric column")
    parser.add_argument("--leakage", default="leakage_batch_acc", help="Leakage metric column")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.utility not in df.columns:
        raise ValueError(f"Utility column not found: {args.utility}")
    if args.leakage not in df.columns:
        raise ValueError(f"Leakage column not found: {args.leakage}")
    if args.x not in df.columns:
        raise ValueError(f"X column not found: {args.x}")

    # Utility plot
    plt.figure()
    df.groupby(args.x)[args.utility].mean().plot(kind="bar")
    plt.ylabel(args.utility)
    plt.title("Utility by configuration")
    plt.tight_layout()
    plt.savefig(out_dir / "utility_by_config.png")

    # Leakage plot
    plt.figure()
    df.groupby(args.x)[args.leakage].mean().plot(kind="bar")
    plt.ylabel(args.leakage)
    plt.title("Leakage by configuration")
    plt.tight_layout()
    plt.savefig(out_dir / "leakage_by_config.png")


if __name__ == "__main__":
    main()

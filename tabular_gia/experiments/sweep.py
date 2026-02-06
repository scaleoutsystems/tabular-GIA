"""Experiment sweep runner."""

from __future__ import annotations

import argparse
import copy
import itertools
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tabular_gia.experiments.runner import _run_experiment_cfgs


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _set_path(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = cfg
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _iter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _resolve_path(base_dir: Path, maybe_path: str | None) -> Path | None:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return p if p.is_absolute() else base_dir / p


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    default_cfg_dir = base_dir / "configs"

    parser = argparse.ArgumentParser(description="Run a sweep of tabular GIA experiments")
    parser.add_argument("--sweep", default=str(default_cfg_dir / "sweep.yaml"))
    parser.add_argument("--quick", action="store_true", help="Force quick mode for each run.")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    sweep_cfg = _load_yaml(Path(args.sweep))
    results_base = sweep_cfg.get("results", "leakpro_output/tabular_gia_sweeps")
    results_dir = Path(results_base)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = _resolve_path(base_dir, sweep_cfg.get("base", str(default_cfg_dir / "base.yaml")))
    dataset_cfg_path = _resolve_path(base_dir, sweep_cfg.get("dataset", str(default_cfg_dir / "dataset" / "dataset.yaml")))
    model_cfg_path = _resolve_path(base_dir, sweep_cfg.get("model", str(default_cfg_dir / "model" / "model.yaml")))
    gia_cfg_path = _resolve_path(base_dir, sweep_cfg.get("gia", str(default_cfg_dir / "gia" / "gia.yaml")))

    base_cfg = _load_yaml(base_cfg_path)
    ds_cfg = _load_yaml(dataset_cfg_path)
    model_cfg = _load_yaml(model_cfg_path)
    gia_cfg_root = _load_yaml(gia_cfg_path)

    grid = sweep_cfg.get("grid", {})
    fl_by_protocol = sweep_cfg.get("fl_config_by_protocol", {})
    overrides = _iter_grid(grid)

    runs: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for i, combo in enumerate(overrides, start=1):
        if args.max_runs is not None and i > args.max_runs:
            break
        base_cfg_run = copy.deepcopy(base_cfg)
        ds_cfg_run = copy.deepcopy(ds_cfg)
        model_cfg_run = copy.deepcopy(model_cfg)
        gia_cfg_root_run = copy.deepcopy(gia_cfg_root)

        for key, value in combo.items():
            if key.startswith("base."):
                _set_path(base_cfg_run, key[len("base.") :], value)
            elif key.startswith("dataset."):
                _set_path(ds_cfg_run, key[len("dataset.") :], value)
            elif key.startswith("model."):
                _set_path(model_cfg_run, key[len("model.") :], value)
            elif key.startswith("gia."):
                _set_path(gia_cfg_root_run, key[len("gia.") :], value)
            elif key.startswith("fl."):
                # stored for later after selecting FL config
                pass
            else:
                raise ValueError(f"Unknown grid override prefix: {key}")

        protocol = base_cfg_run.get("protocol", "fedsgd")
        fl_cfg_path = fl_by_protocol.get(protocol)
        if fl_cfg_path is None:
            fl_cfg_path = sweep_cfg.get("fl", str(default_cfg_dir / "fl" / "fedsgd.yaml"))
        fl_cfg_path = _resolve_path(base_dir, fl_cfg_path)
        fl_cfg_run = _load_yaml(fl_cfg_path)

        for key, value in combo.items():
            if key.startswith("fl."):
                _set_path(fl_cfg_run, key[len("fl.") :], value)

        result = _run_experiment_cfgs(
            base_cfg_run,
            ds_cfg_run,
            model_cfg_run,
            fl_cfg_run,
            gia_cfg_root_run,
            results_base_path=str(results_dir),
            quick=args.quick,
            base_dir=base_dir,
        )

        run_record = {
            "run_id": i,
            "overrides": combo,
            "metrics": result["metrics"],
        }
        runs.append(run_record)

        out_path = results_dir / "sweep_results.jsonl"
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")

        metrics = result["metrics"]
        row = {"run_id": i, **combo}
        for split in ("train", "val", "test"):
            split_stats = metrics.get(split) or {}
            for key, value in split_stats.items():
                row[f"{split}_{key}"] = value
        leakage = result.get("leakage") or {}
        for key, value in leakage.items():
            row[f"leakage_{key}"] = value
        csv_rows.append(row)

    summary_path = results_dir / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    if csv_rows:
        csv_path = results_dir / "sweep_results.csv"
        fieldnames = sorted({k for row in csv_rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABULAR_GIA_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, TABULAR_GIA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.base import BaseConfig
from configs.dataset.dataset import DatasetConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig, InvertingConfig
from configs.model.model import ModelConfig
from helper.helpers import write_json
from leakpro.utils.seed import seed_everything
from tabular_gia.runner.run import RunConfig, RunEngine, build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adult FTTransformer GIA forward parameter sweep across batch sizes."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--logit-scales",
        type=float,
        nargs="+",
        default=[1.0, 2.5, 5.0, 10.0, 20.0],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--attack-iterations", type=int, default=10000)
    parser.add_argument("--attack-lr", type=float, default=0.06)
    parser.add_argument("--local-lr", type=float, default=0.01)
    parser.add_argument("--min-exposure", type=float, default=25.0)
    parser.add_argument(
        "--attack-milestones",
        type=float,
        nargs="+",
        default=[0.0, 25.0],
        help="Exposure checkpoints to attack. Use 0 and final exposure for initialized and trained.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on number of parameter combinations to execute. 0 means all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional experiment directory. Defaults under tabular_gia/results/experiments.",
    )
    return parser.parse_args()


def _experiment_dir(output_dir: str) -> Path:
    if output_dir:
        return Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return TABULAR_GIA_ROOT / "results" / "experiments" / f"experiment_fttransformer_gia_forward_adult_{stamp}"


def _run_config(
    *,
    batch_size: int,
    seed: int,
    num_clients: int,
    attack_iterations: int,
    attack_lr: float,
    local_lr: float,
    min_exposure: float,
    attack_milestones: list[float],
    run_dir: Path,
) -> RunConfig:
    return RunConfig(
        base_cfg=BaseConfig(seed=int(seed), protocol="fedsgd"),
        dataset_cfg=DatasetConfig(
            dataset_path="data/binary/adult/adult.csv",
            dataset_meta_path="data/binary/adult/adult.yaml",
            batch_size=int(batch_size),
        ),
        model_cfg=ModelConfig(preset="fttransformer"),
        fl_cfg=FedSGDConfig(
            local_steps=1,
            local_epochs=1,
            num_clients=int(num_clients),
            min_exposure=float(min_exposure),
            optimizer="MetaSGD",
            lr=float(local_lr),
            vectorized_clients=True,
        ),
        gia_cfg=GiaConfig(
            attack_mode="round_checkpoint",
            fixed_batch_k=1,
            attack_schedule="exposure",
            attack_exposure_milestones=[float(v) for v in attack_milestones],
            vectorized_attacks=True,
            invertingconfig=InvertingConfig(
                label_known=True,
                attack_lr=float(attack_lr),
                at_iterations=int(attack_iterations),
                data_extension="GiaTabularExtension",
            ),
        ),
        results_dir=run_dir,
        fl_only=False,
    )


def _metric_value(summary: dict | None, key: str) -> float | None:
    if summary is None:
        return None
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _extract_checkpoint_summaries(round_summaries: list[dict]) -> tuple[dict | None, dict | None]:
    if not round_summaries:
        return None, None
    ordered = sorted(round_summaries, key=lambda row: int(row["round"]))
    return ordered[0], ordered[-1]


def _summarize_run(*, batch_size: int, temperature: float, logit_scale: float, run_result) -> dict:
    init_summary, trained_summary = _extract_checkpoint_summaries(run_result.round_summaries)
    run_summary = run_result.run_summary or {}
    return {
        "batch_size": int(batch_size),
        "gia_soft_temperature": float(temperature),
        "gia_init_logit_scale": float(logit_scale),
        "initialized_round": None if init_summary is None else int(init_summary["round"]),
        "trained_round": None if trained_summary is None else int(trained_summary["round"]),
        "initialized_tableak_acc": _metric_value(init_summary, "tableak_acc"),
        "initialized_num_acc": _metric_value(init_summary, "num_acc"),
        "initialized_cat_acc": _metric_value(init_summary, "cat_acc"),
        "trained_tableak_acc": _metric_value(trained_summary, "tableak_acc"),
        "trained_num_acc": _metric_value(trained_summary, "num_acc"),
        "trained_cat_acc": _metric_value(trained_summary, "cat_acc"),
        "run_tableak_acc": _metric_value(run_summary, "tableak_acc"),
        "run_num_acc": _metric_value(run_summary, "num_acc"),
        "run_cat_acc": _metric_value(run_summary, "cat_acc"),
    }


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))

    experiment_dir = _experiment_dir(args.output_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "adult",
        "model": "fttransformer",
        "seed": int(args.seed),
        "num_clients": int(args.num_clients),
        "attack_iterations": int(args.attack_iterations),
        "attack_lr": float(args.attack_lr),
        "local_lr": float(args.local_lr),
        "min_exposure": float(args.min_exposure),
        "batch_sizes": [int(v) for v in args.batch_sizes],
        "temperatures": [float(v) for v in args.temperatures],
        "logit_scales": [float(v) for v in args.logit_scales],
        "attack_schedule": "exposure",
        "attack_exposure_milestones": [float(v) for v in args.attack_milestones],
        "max_runs": int(args.max_runs),
    }
    write_json(experiment_dir / "manifest.json", manifest)

    summary_rows: list[dict] = []
    pair_results: list[dict] = []

    total = len(args.batch_sizes) * len(args.temperatures) * len(args.logit_scales)
    run_id = 0
    completed = 0
    for batch_size in args.batch_sizes:
        for temperature in args.temperatures:
            for logit_scale in args.logit_scales:
                if int(args.max_runs) > 0 and completed >= int(args.max_runs):
                    break
                run_id += 1
                tag = f"bs{int(batch_size):03d}_tau{temperature:g}_scale{logit_scale:g}"
                run_dir = experiment_dir / "runs" / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                run_cfg = _run_config(
                    batch_size=int(batch_size),
                    seed=int(args.seed),
                    num_clients=int(args.num_clients),
                    attack_iterations=int(args.attack_iterations),
                    attack_lr=float(args.attack_lr),
                    local_lr=float(args.local_lr),
                    min_exposure=float(args.min_exposure),
                    attack_milestones=[float(v) for v in args.attack_milestones],
                    run_dir=run_dir,
                )
                runtime = build_runtime(run_cfg)
                runtime.model_wrapper.gia_soft_temperature = float(temperature)
                runtime.model_wrapper.gia_init_logit_scale = float(logit_scale)
                engine = RunEngine(run_cfg, runtime)
                result = engine.run()

                summary_row = _summarize_run(
                    batch_size=int(batch_size),
                    temperature=float(temperature),
                    logit_scale=float(logit_scale),
                    run_result=result,
                )
                summary_row["run_id"] = int(run_id)
                summary_row["tag"] = tag
                summary_rows.append(summary_row)
                completed += 1
                pair_results.append(
                    {
                        "run_id": int(run_id),
                        "tag": tag,
                        "batch_size": int(batch_size),
                        "gia_soft_temperature": float(temperature),
                        "gia_init_logit_scale": float(logit_scale),
                        "run_dir": str(run_dir),
                        "summary": summary_row,
                    }
                )
                print(f"[{run_id}/{total}] completed {tag}", flush=True)
            if int(args.max_runs) > 0 and completed >= int(args.max_runs):
                break
        if int(args.max_runs) > 0 and completed >= int(args.max_runs):
            break

    summary_csv = experiment_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with summary_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    best_by_batch: list[dict] = []
    for batch_size in args.batch_sizes:
        rows = [row for row in summary_rows if int(row["batch_size"]) == int(batch_size)]
        if not rows:
            continue
        best_init = max(rows, key=lambda row: float("-inf") if row["initialized_tableak_acc"] is None else float(row["initialized_tableak_acc"]))
        best_trained = max(rows, key=lambda row: float("-inf") if row["trained_tableak_acc"] is None else float(row["trained_tableak_acc"]))
        best_by_batch.append(
            {
                "batch_size": int(batch_size),
                "best_initialized": deepcopy(best_init),
                "best_trained": deepcopy(best_trained),
            }
        )

    write_json(
        experiment_dir / "summary.json",
        {
            "manifest": manifest,
            "summary_rows": summary_rows,
            "best_by_batch": best_by_batch,
            "pair_results": pair_results,
        },
    )
    print(json.dumps({"experiment_dir": str(experiment_dir), "num_runs": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base import BaseConfig
from configs.dataset.dataset import DatasetConfig
from configs.fl.fedavg import FedAvgConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig
from configs.model.model import ModelConfig
from experiments.registry import EXPERIMENT_RUNNERS
from helper.helpers import read_yaml, write_json
from leakpro.utils.seed import seed_everything
from tabular_gia.runner.run import RunConfig, RunEngine, build_runtime


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("leakpro").setLevel(logging.WARNING)


def _run_single(
    *,
    base_cfg: BaseConfig,
    dataset_cfg: DatasetConfig,
    model_cfg: ModelConfig,
    fl_cfg: FedAvgConfig | FedSGDConfig,
    gia_cfg: GiaConfig,
    results_dir: Path,
    fl_only: bool,
) -> None:
    run_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = results_dir / "single_run" / base_cfg.protocol / run_id
    artifacts_dir = run_dir / "artifacts"

    write_json(
        run_dir / "resolved_config.json",
        {
            "base": base_cfg.to_dict(),
            "dataset": dataset_cfg.to_dict(),
            "model": model_cfg.to_dict(),
            "gia": {base_cfg.protocol: gia_cfg.to_dict()},
            "fl": fl_cfg.to_dict(),
        },
    )

    run_config = RunConfig(
        base_cfg=base_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        fl_cfg=fl_cfg,
        gia_cfg=gia_cfg,
        results_dir=artifacts_dir,
        fl_only=fl_only,
    )
    runtime = build_runtime(run_config)
    RunEngine(run_config, runtime).run()


def _run_experiment(
    *,
    experiment_name: str,
    sweep_cfg: dict[str, Any],
    results_dir: Path,
    fl_only: bool,
) -> None:
    if experiment_name not in EXPERIMENT_RUNNERS:
        names = ", ".join(sorted(EXPERIMENT_RUNNERS.keys()))
        raise ValueError(f"Unknown experiment '{experiment_name}'. Available: {names}")

    runner_cls = EXPERIMENT_RUNNERS[experiment_name]
    runner = runner_cls(
        sweep_cfg=sweep_cfg,
        results_dir=results_dir / "experiments",
        fl_only=fl_only,
    )
    runner.run()


def main(
    project_root: Path,
    results_dir: Path,
    experiment_name: str | None = None,
    fl_only: bool = False,
) -> None:
    base_cfg = BaseConfig()
    dataset_cfg = DatasetConfig()
    model_cfg = ModelConfig()
    if experiment_name is None:
        protocol = base_cfg.protocol

        seed_everything(base_cfg.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")

        gia_cfg = GiaConfig()

        fl_cfg: FedAvgConfig | FedSGDConfig
        if protocol == "fedavg":
            fl_cfg = FedAvgConfig()
        elif protocol == "fedsgd":
            fl_cfg = FedSGDConfig()
        else:
            raise ValueError(f"Unknown protocol '{protocol}'.")

        _run_single(
            base_cfg=base_cfg,
            dataset_cfg=dataset_cfg,
            model_cfg=model_cfg,
            fl_cfg=fl_cfg,
            gia_cfg=gia_cfg,
            results_dir=results_dir,
            fl_only=fl_only,
        )
        return

    _run_experiment(
        experiment_name=experiment_name,
        sweep_cfg=read_yaml(project_root / "configs" / "sweep.yaml"),
        results_dir=results_dir,
        fl_only=fl_only,
    )


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Run tabular GIA single runs or named experiments")
    parser.add_argument("--results-dir", default=str(project_dir / "results"))
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--fl-only", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    main(
        project_dir,
        results_dir,
        experiment_name=args.experiment,
        fl_only=args.fl_only,
    )

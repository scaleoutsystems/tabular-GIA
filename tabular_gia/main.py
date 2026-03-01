import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.registry import EXPERIMENT_RUNNERS
from helper.helpers import read_yaml, write_json
from leakpro.utils.seed import seed_everything
from tabular_gia.runner.run import RunEngine, RunSpec


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("leakpro").setLevel(logging.WARNING)


def _run_single(
    *,
    protocol: str,
    base_cfg: dict,
    dataset_cfg: dict,
    model_cfg: dict,
    fl_cfg: dict,
    gia_cfg_path: Path,
    results_dir: Path,
    fl_only: bool,
) -> None:
    logger.info("Loading gia config: base=%s", gia_cfg_path)
    gia_cfg_root = read_yaml(gia_cfg_path)
    gia_cfg = gia_cfg_root.get(protocol)
    if gia_cfg is None or gia_cfg.get("invertingconfig") is None:
        raise ValueError(f"Missing GIA config at '{protocol}.invertingconfig'.")

    run_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = results_dir / "single_run" / protocol / run_id
    artifacts_dir = run_dir / "artifacts"

    write_json(
        run_dir / "resolved_config.json",
        {
            "base": base_cfg,
            "dataset": dataset_cfg,
            "model": model_cfg,
            "gia": {protocol: gia_cfg},
            "fl": fl_cfg,
        },
    )

    spec = RunSpec(
        protocol=protocol,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        fl_cfg=fl_cfg,
        gia_cfg=gia_cfg,
        results_dir=artifacts_dir,
        fl_only=fl_only,
    )
    RunEngine(spec).run()


def _run_experiment(
    *,
    experiment_name: str,
    base_cfg: dict,
    dataset_cfg: dict,
    model_cfg: dict,
    cfg_dir: Path,
    results_dir: Path,
    fl_only: bool,
) -> None:
    if experiment_name not in EXPERIMENT_RUNNERS:
        names = ", ".join(sorted(EXPERIMENT_RUNNERS.keys()))
        raise ValueError(f"Unknown experiment '{experiment_name}'. Available: {names}")

    if experiment_name != "sweep":
        raise ValueError(f"Experiment runner not wired yet for '{experiment_name}'")

    sweep_cfg_path = cfg_dir / "sweep.yaml"
    logger.info("Loading experiment config: %s", sweep_cfg_path)
    sweep_cfg = read_yaml(sweep_cfg_path)

    runner_cls = EXPERIMENT_RUNNERS[experiment_name]
    runner = runner_cls(
        sweep_cfg=sweep_cfg,
        base_cfg=base_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        config_dir=cfg_dir,
        results_dir=results_dir / "experiments",
        fl_only=fl_only,
    )
    runner.run()


def main(
    base_cfg_path: Path,
    dataset_cfg_path: Path,
    model_cfg_path: Path,
    gia_cfg_path: Path,
    cfg_dir: Path,
    results_dir: Path,
    experiment_name: str | None = None,
    fl_only: bool = False,
) -> None:
    logger.info("Loading base config: base=%s", base_cfg_path)
    base_cfg = read_yaml(base_cfg_path)
    protocol = base_cfg.get("protocol")
    if not protocol:
        raise ValueError("Missing 'protocol' in base config.")
    seed = base_cfg.get("seed")

    seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    logger.info("Loading dataset config: base=%s", dataset_cfg_path)
    dataset_cfg = read_yaml(dataset_cfg_path)
    dataset_cfg["seed"] = seed

    logger.info("Loading model config: base=%s", model_cfg_path)
    model_cfg = read_yaml(model_cfg_path)

    fl_cfg_path = base_cfg_path.parent / "fl" / f"{protocol}.yaml"
    logger.info("Loading fl config: base=%s", fl_cfg_path)
    fl_cfg = read_yaml(fl_cfg_path)
    fl_cfg["batch_size"] = dataset_cfg.get("batch_size")

    if experiment_name is None:
        _run_single(
            protocol=protocol,
            base_cfg=base_cfg,
            dataset_cfg=dataset_cfg,
            model_cfg=model_cfg,
            fl_cfg=fl_cfg,
            gia_cfg_path=gia_cfg_path,
            results_dir=results_dir,
            fl_only=fl_only,
        )
        return

    _run_experiment(
        experiment_name=experiment_name,
        base_cfg=base_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        cfg_dir=cfg_dir,
        results_dir=results_dir,
        fl_only=fl_only,
    )


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    cfg_dir = project_dir / "configs"
    base_path = cfg_dir / "base.yaml"
    dataset_path = cfg_dir / "dataset" / "dataset.yaml"
    model_path = cfg_dir / "model" / "model.yaml"
    gia_path = cfg_dir / "gia" / "gia.yaml"

    parser = argparse.ArgumentParser(description="Run tabular GIA single runs or named experiments")
    parser.add_argument("--results-dir", default=str(project_dir / "results"))
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--fl-only", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    main(
        base_path,
        dataset_path,
        model_path,
        gia_path,
        cfg_dir,
        results_dir,
        experiment_name=args.experiment,
        fl_only=args.fl_only,
    )

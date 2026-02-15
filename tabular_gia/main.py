import argparse
from pathlib import Path
import logging

import torch

from leakpro.utils.seed import seed_everything

from helper.helpers import read_yaml
from runner.run import run, run_sweep


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.getLogger("leakpro").setLevel(logging.WARNING)


def main(
    base_cfg_path: Path,
    dataset_cfg_path: Path,
    model_cfg_path: Path,
    gia_cfg_path: Path,
    cfg_dir: Path,
    results_dir: Path,
    sweep_cfg_path: Path | None = None,
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

    if sweep_cfg_path is None:
        # run single
        logger.info("Loading gia config: base=%s", gia_cfg_path)
        gia_cfg_root = read_yaml(gia_cfg_path)
        gia_cfg = gia_cfg_root.get(protocol, {}).get("invertingconfig")
        if gia_cfg is None:
            raise ValueError(f"Missing GIA config at '{protocol}.invertingconfig'.")
        results_dir = results_dir / "single_run" / protocol
        run(
            protocol=protocol,
            dataset_cfg=dataset_cfg,
            model_cfg=model_cfg,
            fl_cfg=fl_cfg,
            gia_cfg=gia_cfg,
            results_dir=results_dir,
        )
    else:
        # run experiments
        logger.info("Loading sweep config: base=%s", sweep_cfg_path)
        sweep_cfg = read_yaml(sweep_cfg_path)
        results_dir = results_dir / "experiments"
        run_sweep(
            sweep_cfg=sweep_cfg,
            base_cfg=base_cfg,
            dataset_cfg=dataset_cfg,
            model_cfg=model_cfg,
            config_dir=cfg_dir,
            results_dir=results_dir,
        )


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    cfg_dir = project_dir / "configs"
    base_path = cfg_dir / "base.yaml"
    dataset_path = cfg_dir / "dataset" / "dataset.yaml"
    model_path = cfg_dir / "model" / "model.yaml"
    gia_path = cfg_dir / "gia" / "gia.yaml"
    sweep_path = cfg_dir / "sweep.yaml"

    parser = argparse.ArgumentParser(description="Run tabular GIA single runs or experiment sweeps")
    parser.add_argument("--results-dir", default=str(project_dir / "results"))
    parser.add_argument("--run_experiment", "--run-experiment", action="store_true", dest="run_experiment")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.run_experiment:
        main(base_path, dataset_path, model_path, gia_path, cfg_dir, results_dir, sweep_path)
    else:
        main(base_path, dataset_path, model_path, gia_path, cfg_dir, results_dir)

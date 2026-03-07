from __future__ import annotations

import itertools
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm
import torch

from helper.helpers import write_json
from helper.summary_aggregation import SeedSummaryBuilder
from leakpro.utils.seed import seed_everything
from tabular_gia.runner.run import RunConfig, RunEngine, RunResult, build_runtime
from tabular_gia.helper.results_writer import SweepResultsWriter


def _iter_grid(grid: dict[str, list[Any]]):
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    values = [_unique_values(grid[k]) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _grid_size(grid: dict[str, list[Any]]) -> int:
    if not grid:
        return 1
    size = 1
    for values in grid.values():
        size *= len(_unique_values(values))
    return size


def _unique_values(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if not any(value == seen for seen in unique):
            unique.append(value)
    return unique


def _iter_run_combos(
    *,
    protocols: list[str],
    base_grid: dict[str, list[Any]],
    dataset_grid: dict[str, list[Any]],
    model_grid: dict[str, list[Any]],
    gia_grid: dict[str, list[Any]],
    fl_grids: dict[str, dict[str, list[Any]]],
):
    for protocol in protocols:
        fl_protocol_grid = fl_grids[protocol]
        for base_override in _iter_grid(base_grid):
            for dataset_override in _iter_grid(dataset_grid):
                for model_override in _iter_grid(model_grid):
                    for gia_override in _iter_grid(gia_grid):
                        for fl_override in _iter_grid(fl_protocol_grid):
                            yield protocol, base_override, dataset_override, model_override, gia_override, fl_override


def _parse_section(name: str, section: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    default_raw = section["default"]
    grid_raw_obj = section["grid"]
    if default_raw is None:
        default_raw = {}
    if grid_raw_obj is None:
        grid_raw_obj = {}
    default = dict(default_raw)
    grid_raw = dict(grid_raw_obj)
    grid: dict[str, list[Any]] = {}
    for key, values in grid_raw.items():
        if type(values) is list:
            if len(values) == 0:
                raise ValueError(f"Sweep section '{name}.grid.{key}' must be non-empty.")
            grid[key] = values
            continue
        grid[key] = [values]
    for key, values in grid.items():
        if len(values) == 0:
            raise ValueError(f"Sweep section '{name}.grid.{key}' must be non-empty.")
    return default, grid


def _resolve_dataset_cfg(dataset_override: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    dataset_cfg_run = deepcopy(dataset_override)
    for dataset_key, value in dataset_override.items():
        if dataset_key == "dataset_path_and_meta_path":
            dataset_cfg_run["dataset_path"], dataset_cfg_run["dataset_meta_path"] = value
        else:
            dataset_cfg_run[dataset_key] = value

    for path_key in ("dataset_path", "dataset_meta_path"):
        raw_path = Path(dataset_cfg_run[path_key])
        if not raw_path.is_absolute():
            raw_path = config_dir.parent / raw_path
        dataset_cfg_run[path_key] = str(raw_path)
    return dataset_cfg_run


def _resolve_model_cfg(
    *,
    model_params: dict[str, Any],
    model_presets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    preset_name = model_params["preset"]
    if preset_name is not None:
        if preset_name not in model_presets:
            raise ValueError(f"Preset '{preset_name}' not found in model config.")
        return deepcopy(model_presets[preset_name])
    model_explicit = deepcopy(model_params)
    model_explicit.pop("preset")
    model_explicit.pop("presets")
    if not model_explicit:
        raise ValueError("Sweep model config resolved to empty explicit model params while preset=null.")
    return model_explicit


def _resolve_gia_cfg(
    *,
    gia_params: dict[str, Any],
) -> dict[str, Any]:
    protocol_cfg: dict[str, Any] = {}
    invertingconfig: dict[str, Any] = {}
    protocol_keys = {
        "attack_mode",
        "fixed_batch_k",
        "attack_schedule",
        "attack_rounds",
        "attack_num_checkpoints",
        "attack_exposure_milestones",
    }
    for key, value in gia_params.items():
        if key in protocol_keys:
            protocol_cfg[key] = value
        else:
            invertingconfig[key] = value
    if "attack_mode" not in protocol_cfg:
        raise ValueError("Sweep gia config missing required key: 'attack_mode'.")
    if "attack_schedule" not in protocol_cfg:
        raise ValueError("Sweep gia config missing required key: 'attack_schedule'.")
    if "fixed_batch_k" not in protocol_cfg:
        raise ValueError("Sweep gia config missing required key: 'fixed_batch_k'.")
    if "attack_exposure_milestones" not in protocol_cfg:
        raise ValueError("Sweep gia config missing required key: 'attack_exposure_milestones'.")
    if not invertingconfig:
        raise ValueError("Sweep gia config resolved empty invertingconfig.")
    protocol_cfg["invertingconfig"] = invertingconfig
    return protocol_cfg


def build_run_configs(
    sweep_cfg: dict[str, Any],
    *,
    config_dir: Path,
) -> list[dict[str, Any]]:
    base_section = sweep_cfg["base"]
    dataset_section = sweep_cfg["dataset"]
    model_section = sweep_cfg["model"]
    fl_section = sweep_cfg["fl"]
    gia_section = sweep_cfg["gia"]

    base_default, base_grid_full = _parse_section("base", base_section)
    dataset_default, dataset_grid = _parse_section("dataset", dataset_section)
    model_default, model_grid_full = _parse_section("model", model_section)
    gia_default, gia_grid = _parse_section("gia", gia_section)
    if "presets" not in model_default:
        raise ValueError("Sweep model.default must include 'presets'.")
    model_presets = deepcopy(model_default["presets"])
    if type(model_presets) is not dict or not model_presets:
        raise ValueError("Sweep model.default.presets must be a non-empty mapping.")

    if "protocol" in base_grid_full:
        protocols = [str(p) for p in base_grid_full["protocol"]]
    elif "protocol" in base_default:
        protocols = [str(base_default["protocol"])]
    else:
        raise ValueError("Sweep config must define protocol in base.grid.protocol or base.default.protocol.")
    if not protocols:
        raise ValueError("Sweep config resolved an empty protocol list.")

    if "seed" not in base_grid_full:
        raise ValueError("Sweep config must define base.grid.seed as a non-empty list.")
    raw_seeds = base_grid_full["seed"]
    seeds = [int(s) for s in _unique_values(raw_seeds)]
    if not seeds:
        raise ValueError("Sweep config must define at least one seed.")

    base_grid = {k: v for k, v in base_grid_full.items() if k not in {"protocol", "seed"}}

    model_params_for_mode = deepcopy(model_default)
    model_params_for_mode.update({k: v[0] for k, v in model_grid_full.items()})
    preset_for_mode = model_params_for_mode["preset"]
    if preset_for_mode is not None:
        model_presets_selected = model_grid_full["preset"] if "preset" in model_grid_full else [model_default["preset"]]
        if not model_presets_selected:
            raise ValueError("Sweep config sets model preset mode but model.preset resolved empty.")
        model_grid = {"preset": model_presets_selected}
    else:
        model_grid = {k: v for k, v in model_grid_full.items() if k != "preset"}
        if not model_grid:
            raise ValueError("Sweep must define model.preset or explicit model parameter lists when preset is null.")

    run_configs: list[dict[str, Any]] = []
    protocol_fl_sweep_defaults: dict[str, dict[str, Any]] = {}
    protocol_fl_sweep_grids: dict[str, dict[str, list[Any]]] = {}
    total_specs = 0
    common_size = _grid_size(base_grid) * _grid_size(dataset_grid) * _grid_size(model_grid) * _grid_size(gia_grid)

    for protocol in protocols:
        if protocol not in fl_section:
            raise ValueError(f"Sweep config missing fl section for protocol '{protocol}'.")
        fl_default, fl_grid = _parse_section(f"fl.{protocol}", fl_section[protocol])
        protocol_fl_sweep_defaults[protocol] = fl_default
        protocol_fl_sweep_grids[protocol] = fl_grid
        total_specs += common_size * _grid_size(fl_grid)

    with tqdm(total=total_specs, desc="Building sweep run configs", unit="config", leave=False) as bar:
        run_id = 0
        for protocol, base_override, dataset_override, model_override, gia_override, fl_override in _iter_run_combos(
            protocols=protocols,
            base_grid=base_grid,
            dataset_grid=dataset_grid,
            model_grid=model_grid,
            gia_grid=gia_grid,
            fl_grids=protocol_fl_sweep_grids,
        ):
            bar.update(1)
            run_id += 1

            base_cfg_run = deepcopy(base_default)
            base_cfg_run["protocol"] = protocol
            base_cfg_run.update(base_override)
            base_cfg_run.pop("seed", None)

            dataset_params = deepcopy(dataset_default)
            dataset_params.update(dataset_override)
            dataset_cfg_run = _resolve_dataset_cfg(dataset_params, config_dir)
            model_params = deepcopy(model_default)
            model_params.update(model_override)
            model_cfg_run = _resolve_model_cfg(
                model_params=model_params,
                model_presets=model_presets,
            )
            gia_params = deepcopy(gia_default)
            gia_params.update(gia_override)
            gia_cfg_run = _resolve_gia_cfg(
                gia_params=gia_params,
            )
            if "attack_schedule" in base_cfg_run:
                gia_cfg_run["attack_schedule"] = base_cfg_run["attack_schedule"]

            fl_cfg_run = deepcopy(protocol_fl_sweep_defaults[protocol])
            fl_cfg_run.update(fl_override)

            overrides = {
                "base": base_override,
                "dataset": dataset_override,
                "model": model_override,
                "gia": gia_override,
                "fl": fl_override,
            }
            run_configs.append(
                {
                    "run_id": run_id,
                    "seeds": seeds,
                    "overrides": overrides,
                    "base_cfg": base_cfg_run,
                    "dataset_cfg": dataset_cfg_run,
                    "model_cfg": model_cfg_run,
                    "fl_cfg": fl_cfg_run,
                    "gia_cfg": gia_cfg_run,
                }
            )

    return run_configs


class SweepExperimentRunner:
    def __init__(
        self,
        *,
        sweep_cfg: dict,
        config_dir: Path,
        results_dir: Path,
        fl_only: bool = False,
    ) -> None:
        self.sweep_cfg = sweep_cfg
        self.config_dir = config_dir
        self.results_dir = results_dir
        self.fl_only = fl_only
        self.csv_writer = SweepResultsWriter()
        self.seed_summary_builder = SeedSummaryBuilder()

    def run(self) -> None:
        experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        experiment_dir = self.results_dir / experiment_id
        run_configs = build_run_configs(
            sweep_cfg=self.sweep_cfg,
            config_dir=self.config_dir,
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        run_records: list[dict] = []
        sweep_result_rows_per_seed: list[dict] = []
        sweep_result_rows_agg: list[dict] = []
        for run_config in run_configs:
            run_id = int(run_config["run_id"])
            protocol = str(run_config["base_cfg"]["protocol"])
            seeds = [int(s) for s in run_config["seeds"]]

            run_dir = experiment_dir / protocol / f"run_{run_id:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "run_config.json",
                {
                    "base": run_config["base_cfg"],
                    "seeds": seeds,
                    "dataset": run_config["dataset_cfg"],
                    "model": run_config["model_cfg"],
                    "gia": {protocol: run_config["gia_cfg"]},
                    "fl": run_config["fl_cfg"],
                    "overrides": run_config["overrides"],
                },
            )

            run_results_for_run: list[RunResult] = []
            for run_seed in seeds:
                seed_everything(run_seed)
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.set_float32_matmul_precision("high")
                dataset_cfg_run = deepcopy(run_config["dataset_cfg"])
                dataset_cfg_run["seed"] = int(run_seed)
                seed_dir = run_dir / f"seed_{int(run_seed):04d}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                base_cfg_seed = deepcopy(run_config["base_cfg"])
                base_cfg_seed["seed"] = int(run_seed)
                write_json(
                    seed_dir / "resolved_config.json",
                    {
                        "base": base_cfg_seed,
                        "dataset": dataset_cfg_run,
                        "model": run_config["model_cfg"],
                        "gia": {protocol: run_config["gia_cfg"]},
                        "fl": run_config["fl_cfg"],
                        "overrides": run_config["overrides"],
                    },
                )

                run_seed_config = RunConfig(
                    protocol=protocol,
                    dataset_cfg=dataset_cfg_run,
                    model_cfg=deepcopy(run_config["model_cfg"]),
                    fl_cfg=deepcopy(run_config["fl_cfg"]),
                    gia_cfg=deepcopy(run_config["gia_cfg"]),
                    results_dir=seed_dir / "artifacts",
                    fl_only=self.fl_only,
                )
                runtime = build_runtime(run_seed_config)
                run_result = RunEngine(run_seed_config, runtime).run()
                run_summary = run_result.run_summary
                if run_summary is None:
                    if not self.fl_only:
                        raise ValueError(
                            f"Run summary missing for run_id={run_id}, seed={run_seed}. "
                            "This sweep mode requires run_summary output."
                        )
                    run_results_for_run.append(run_result)
                    continue

                run_results_for_run.append(run_result)
                row = {"run_id": run_id, "seed": int(run_seed)}
                row.update(run_summary)
                sweep_result_rows_per_seed.append(row)

            seed_aggregate = self.seed_summary_builder.build_seed_aggregate(run_results_for_run)
            self.csv_writer.write_seed_aggregate(run_dir, seed_aggregate)
            if seed_aggregate.run_summary is None:
                if not self.fl_only:
                    raise ValueError(
                        f"Aggregated run summary missing for run_id={run_id}. "
                        "This sweep mode requires aggregated run_summary output."
                    )
            else:
                sweep_result_rows_agg.append(
                    {"run_id": run_id, "num_seeds": int(len(run_results_for_run)), **seed_aggregate.run_summary}
                )

            run_records.append(
                {
                    "run_id": run_id,
                    "protocol": protocol,
                    "seeds": seeds,
                    "run_dir": str(run_dir),
                    "overrides": run_config["overrides"],
                }
            )

        with open(experiment_dir / "sweep_runs.json", "w", encoding="utf-8") as f:
            json.dump(run_records, f, indent=2)
        self.csv_writer.write_sweep_results(experiment_dir, sweep_result_rows_agg)
        self.csv_writer.write_sweep_results_per_seed(experiment_dir, sweep_result_rows_per_seed)

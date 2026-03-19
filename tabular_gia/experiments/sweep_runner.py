from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import RLock
from pathlib import Path
from typing import Any

from tqdm import tqdm

from configs.base import BaseConfig
from configs.dataset.dataset import DatasetConfig
from configs.fl.fedavg import FedAvgConfig
from configs.fl.fedsgd import FedSGDConfig
from configs.gia.gia import GiaConfig, InvertingConfig as GiaInvertingConfig
from configs.model.model import ModelConfig
from helper.helpers import write_json
from helper.summary_aggregation import SeedSummaryBuilder
from leakpro.utils.seed import seed_everything
from tabular_gia.helper.results_writer import SweepResultsWriter
from tabular_gia.runner.run import RunConfig, RunEngine, RunResult, build_runtime


RunEntry = tuple[int, dict[str, dict[str, Any]], RunConfig]


@dataclass(frozen=True)
class SweepSeedRunResult:
    seed: int
    run_config: RunConfig
    run_result: RunResult


@dataclass(frozen=True)
class SweepGroupResult:
    run_group_id: int
    protocol: str
    seeds: list[int]
    overrides: dict[str, dict[str, Any]]
    run_dir: Path
    seed_runs: list[SweepSeedRunResult]
    aggregated_run_summary: dict | None


@dataclass(frozen=True)
class SweepRunResults:
    experiment_dir: Path
    groups: list[SweepGroupResult]


@dataclass(frozen=True)
class GroupExecutionResult:
    run_record: dict
    sweep_result_rows_per_seed: list[dict]
    sweep_result_row_agg: dict | None
    group_result: SweepGroupResult


def _unique_values(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if not any(value == seen for seen in unique):
            unique.append(value)
    return unique


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
    default_raw = section["default"] if "default" in section else {}
    grid_raw_obj = section["grid"] if "grid" in section else {}
    if default_raw is None:
        default_raw = {}
    if grid_raw_obj is None:
        grid_raw_obj = {}

    if type(default_raw) is not dict:
        raise ValueError(f"Sweep section '{name}.default' must be a mapping.")
    if type(grid_raw_obj) is not dict:
        raise ValueError(f"Sweep section '{name}.grid' must be a mapping.")

    default = dict(default_raw)
    grid: dict[str, list[Any]] = {}
    for key, values_raw in grid_raw_obj.items():
        # Grid keys not present in defaults are invalid.
        if key not in default:
            raise ValueError(f"Sweep section '{name}.grid.{key}' is not a valid key in '{name}.default'.")

        # Support nested grid syntax, e.g.
        # grid: {invertingconfig: {at_iterations: [100, 1000]}}
        if type(values_raw) is dict:
            default_nested = default[key]
            if type(default_nested) is not dict:
                raise ValueError(
                    f"Sweep section '{name}.grid.{key}' must be a list/scalar because '{name}.default.{key}' is not a mapping."
                )

            nested_grid: dict[str, list[Any]] = {}
            for nested_key, nested_values_raw in values_raw.items():
                if nested_key not in default_nested:
                    raise ValueError(
                        f"Sweep section '{name}.grid.{key}.{nested_key}' is not a valid key in '{name}.default.{key}'."
                    )
                nested_values = nested_values_raw if type(nested_values_raw) is list else [nested_values_raw]
                if len(nested_values) == 0:
                    raise ValueError(f"Sweep section '{name}.grid.{key}.{nested_key}' must be non-empty.")
                nested_grid[nested_key] = nested_values

            if not nested_grid:
                raise ValueError(f"Sweep section '{name}.grid.{key}' must be non-empty.")

            nested_keys = list(nested_grid.keys())
            nested_values_list = [_unique_values(nested_grid[nested_key]) for nested_key in nested_keys]
            nested_values: list[dict[str, Any]] = []
            for combo in itertools.product(*nested_values_list):
                merged_nested = deepcopy(default_nested)
                merged_nested.update(dict(zip(nested_keys, combo)))
                nested_values.append(merged_nested)
            grid[key] = nested_values
            continue

        values = values_raw if type(values_raw) is list else [values_raw]
        if len(values) == 0:
            raise ValueError(f"Sweep section '{name}.grid.{key}' must be non-empty.")
        grid[key] = values

    return default, grid


def _resolve_dataset_cfg(dataset_params: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = deepcopy(dataset_params)
    if "dataset_path_and_meta_path" in dataset_cfg:
        dataset_path, dataset_meta_path = dataset_cfg.pop("dataset_path_and_meta_path")
        dataset_cfg["dataset_path"] = dataset_path
        dataset_cfg["dataset_meta_path"] = dataset_meta_path

    for path_key in ("dataset_path", "dataset_meta_path"):
        if path_key not in dataset_cfg:
            raise ValueError(f"Sweep dataset config missing required key: '{path_key}'.")
        dataset_cfg[path_key] = str(dataset_cfg[path_key])

    return dataset_cfg


def _resolve_model_cfg(
    *,
    model_params: dict[str, Any],
    model_presets: dict[str, dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    preset_name = model_params["preset"]
    if preset_name is not None:
        if preset_name not in model_presets:
            raise ValueError(f"Preset '{preset_name}' not found in model config.")
        return preset_name, deepcopy(model_presets[preset_name])

    model_explicit = deepcopy(model_params)
    model_explicit.pop("preset", None)
    model_explicit.pop("presets", None)
    if not model_explicit:
        raise ValueError("Sweep model config resolved to empty explicit model params while preset=null.")
    return None, model_explicit


def build_run_configs(
    sweep_cfg: dict[str, Any],
    *,
    experiment_dir: Path,
    fl_only: bool,
) -> list[RunEntry]:
    base_default, base_grid_full = _parse_section("base", sweep_cfg["base"])
    dataset_default, dataset_grid = _parse_section("dataset", sweep_cfg["dataset"])
    model_default, model_grid_full = _parse_section("model", sweep_cfg["model"])
    gia_default, gia_grid = _parse_section("gia", sweep_cfg["gia"])

    if "preset" not in model_default:
        model_default["preset"] = None

    if "protocol" in base_grid_full:
        protocols = [str(p) for p in base_grid_full["protocol"]]
    elif "protocol" in base_default:
        protocols = [str(base_default["protocol"])]
    else:
        raise ValueError("Sweep config must define protocol in base.grid.protocol or base.default.protocol.")
    protocols = _unique_values(protocols)
    if not protocols:
        raise ValueError("Sweep config resolved an empty protocol list.")

    if "seed" not in base_grid_full:
        raise ValueError("Sweep config must define base.grid.seed as a non-empty list.")
    seeds = [int(s) for s in _unique_values(base_grid_full["seed"])]
    if not seeds:
        raise ValueError("Sweep config must define at least one seed.")

    base_grid = {k: v for k, v in base_grid_full.items() if k not in {"protocol", "seed"}}

    preset_grid_values = model_grid_full["preset"] if "preset" in model_grid_full else None
    explicit_model_mode: bool
    if preset_grid_values is not None:
        preset_values = _unique_values(preset_grid_values)
        has_none = any(value is None for value in preset_values)
        has_non_none = any(value is not None for value in preset_values)
        if has_none and has_non_none:
            raise ValueError("Sweep model.grid.preset cannot mix null and non-null values.")
        explicit_model_mode = has_none
    else:
        explicit_model_mode = model_default["preset"] is None

    if explicit_model_mode:
        model_grid = {k: v for k, v in model_grid_full.items() if k != "preset"}
        if not model_grid:
            raise ValueError("Sweep must define explicit model parameter lists when preset is null.")
    else:
        if "presets" not in model_default:
            raise ValueError("Sweep model.default must include 'presets' in preset mode.")
        if type(model_default["presets"]) is not dict or not model_default["presets"]:
            raise ValueError("Sweep model.default.presets must be a non-empty mapping in preset mode.")
        presets = preset_grid_values if preset_grid_values is not None else [model_default["preset"]]
        if not presets:
            raise ValueError("Sweep config sets model preset mode but model.preset resolved empty.")
        model_grid = {"preset": presets}

    protocol_fl_defaults: dict[str, dict[str, Any]] = {}
    protocol_fl_grids: dict[str, dict[str, list[Any]]] = {}
    total_specs = 0
    common_size = _grid_size(base_grid) * _grid_size(dataset_grid) * _grid_size(model_grid) * _grid_size(gia_grid)

    for protocol in protocols:
        if protocol not in sweep_cfg["fl"]:
            raise ValueError(f"Sweep config missing fl section for protocol '{protocol}'.")
        fl_default, fl_grid = _parse_section(f"fl.{protocol}", sweep_cfg["fl"][protocol])
        protocol_fl_defaults[protocol] = fl_default
        protocol_fl_grids[protocol] = fl_grid
        total_specs += common_size * _grid_size(fl_grid)

    run_entries: list[RunEntry] = []
    with tqdm(total=total_specs, desc="Building sweep run configs", unit="config", leave=False) as bar:
        run_group_id = 0
        for protocol, base_override, dataset_override, model_override, gia_override, fl_override in _iter_run_combos(
            protocols=protocols,
            base_grid=base_grid,
            dataset_grid=dataset_grid,
            model_grid=model_grid,
            gia_grid=gia_grid,
            fl_grids=protocol_fl_grids,
        ):
            bar.update(1)
            run_group_id += 1

            dataset_params = deepcopy(dataset_default)
            dataset_params.update(dataset_override)
            dataset_cfg_dict = _resolve_dataset_cfg(dataset_params)

            model_params = deepcopy(model_default)
            if explicit_model_mode:
                model_params["preset"] = None
            model_params.update(model_override)
            model_preset, model_cfg_dict = _resolve_model_cfg(
                model_params=model_params,
                model_presets=model_params["presets"] if "presets" in model_params else {},
            )

            gia_params = deepcopy(gia_default)
            gia_params.update(gia_override)
            if "invertingconfig" not in gia_params:
                raise ValueError("Sweep gia config missing required key: 'invertingconfig'.")
            gia_cfg_params = deepcopy(gia_params)
            invertingconfig_params = gia_cfg_params.pop("invertingconfig")
            if type(invertingconfig_params) is not dict:
                raise ValueError("Sweep gia.invertingconfig must be a mapping.")

            fl_cfg_dict = deepcopy(protocol_fl_defaults[protocol])
            fl_cfg_dict.update(fl_override)

            overrides: dict[str, dict[str, Any]] = {
                "base": base_override,
                "dataset": dataset_override,
                "model": model_override,
                "gia": gia_override,
                "fl": fl_override,
            }

            run_dir = experiment_dir / protocol / f"run_{run_group_id:04d}"
            for seed in seeds:
                seed_dir = run_dir / f"seed_{seed:04d}"
                if "presets" in model_params and type(model_params["presets"]) is dict:
                    model_cfg = ModelConfig(
                        preset=model_preset,
                        presets=deepcopy(model_params["presets"]),
                        **deepcopy(model_cfg_dict),
                    )
                else:
                    model_cfg = ModelConfig(
                        preset=model_preset,
                        **deepcopy(model_cfg_dict),
                    )
                run_config = RunConfig(
                    base_cfg=BaseConfig(seed=seed, protocol=protocol),
                    dataset_cfg=DatasetConfig(**deepcopy(dataset_cfg_dict)),
                    model_cfg=model_cfg,
                    fl_cfg=FedAvgConfig(**deepcopy(fl_cfg_dict)) if protocol == "fedavg" else FedSGDConfig(**deepcopy(fl_cfg_dict)),
                    gia_cfg=GiaConfig(
                        **gia_cfg_params,
                        invertingconfig=GiaInvertingConfig(**invertingconfig_params),
                    ),
                    results_dir=seed_dir / "artifacts",
                    fl_only=fl_only,
                )
                run_entries.append((run_group_id, deepcopy(overrides), run_config))

    return run_entries


def _execute_run_group(
    run_group_id: int,
    entries: list[tuple[dict[str, dict[str, Any]], RunConfig]],
    fl_only: bool,
    tqdm_slot: int | None = None,
) -> GroupExecutionResult:
    if tqdm_slot is None:
        os.environ.pop("TABULAR_GIA_TQDM_SLOT", None)
    else:
        os.environ["TABULAR_GIA_TQDM_SLOT"] = str(int(tqdm_slot))
    os.environ["TABULAR_GIA_TQDM_STRIDE"] = "2"
    os.environ["TABULAR_GIA_TQDM_BASE"] = "1"

    first_overrides, first_run_config = entries[0]
    protocol = first_run_config.base_cfg.protocol
    seeds = [entry_run_config.base_cfg.seed for _, entry_run_config in entries]
    run_dir = first_run_config.results_dir.parent.parent

    write_json(
        run_dir / "run_config.json",
        {
            "base": first_run_config.base_cfg.to_dict(),
            "seeds": seeds,
            "dataset": first_run_config.dataset_cfg.to_dict(),
            "model": first_run_config.model_cfg.to_dict(),
            "gia": {protocol: first_run_config.gia_cfg.to_dict()},
            "fl": first_run_config.fl_cfg.to_dict(),
            "overrides": first_overrides,
        },
    )

    run_results_for_run: list[RunResult] = []
    seed_runs: list[SweepSeedRunResult] = []
    sweep_result_rows_per_seed: list[dict] = []

    for overrides, run_config in entries:
        run_seed = run_config.base_cfg.seed
        seed_dir = run_config.results_dir.parent
        seed_dir.mkdir(parents=True, exist_ok=True)
        run_tag = f"Run {int(run_group_id)}"
        os.environ["TABULAR_GIA_TQDM_TAG"] = run_tag

        write_json(
            seed_dir / "resolved_config.json",
            {
                "base": run_config.base_cfg.to_dict(),
                "dataset": run_config.dataset_cfg.to_dict(),
                "model": run_config.model_cfg.to_dict(),
                "gia": {protocol: run_config.gia_cfg.to_dict()},
                "fl": run_config.fl_cfg.to_dict(),
                "overrides": overrides,
            },
        )

        seed_everything(run_seed)

        runtime = build_runtime(run_config)
        run_result = RunEngine(run_config, runtime).run()
        run_results_for_run.append(run_result)
        seed_runs.append(
            SweepSeedRunResult(
                seed=run_seed,
                run_config=run_config,
                run_result=run_result,
            )
        )

        run_summary = run_result.run_summary
        if run_summary is None:
            if not fl_only:
                raise ValueError(
                    f"Run summary missing for run_id={run_group_id}, seed={run_seed}. "
                    "This sweep mode requires run_summary output."
                )
            continue

        row = {"run_id": run_group_id, "seed": run_seed}
        row.update(run_summary)
        sweep_result_rows_per_seed.append(row)

    seed_summary_builder = SeedSummaryBuilder()
    csv_writer = SweepResultsWriter()
    seed_aggregate = seed_summary_builder.build_seed_aggregate(run_results_for_run)
    csv_writer.write_seed_aggregate(run_dir, seed_aggregate)

    aggregated_run_summary: dict | None = None
    sweep_result_row_agg: dict | None = None
    if seed_aggregate.run_summary is None:
        if not fl_only:
            raise ValueError(
                f"Aggregated run summary missing for run_id={run_group_id}. "
                "This sweep mode requires aggregated run_summary output."
            )
    else:
        aggregated_run_summary = dict(seed_aggregate.run_summary)
        sweep_result_row_agg = {
            "run_id": run_group_id,
            "num_seeds": len(run_results_for_run),
            **seed_aggregate.run_summary,
        }

    run_record = {
        "run_id": run_group_id,
        "protocol": protocol,
        "seeds": seeds,
        "run_dir": str(run_dir),
        "overrides": first_overrides,
    }
    group_result = SweepGroupResult(
        run_group_id=run_group_id,
        protocol=protocol,
        seeds=list(seeds),
        overrides=deepcopy(first_overrides),
        run_dir=run_dir,
        seed_runs=seed_runs,
        aggregated_run_summary=aggregated_run_summary,
    )

    return GroupExecutionResult(
        run_record=run_record,
        sweep_result_rows_per_seed=sweep_result_rows_per_seed,
        sweep_result_row_agg=sweep_result_row_agg,
        group_result=group_result,
    )


def _execute_run_group_from_tuple(
    task: tuple[int, list[tuple[dict[str, dict[str, Any]], RunConfig]], bool, int | None]
) -> GroupExecutionResult:
    run_group_id, entries, fl_only, tqdm_slot = task
    return _execute_run_group(run_group_id, entries, fl_only, tqdm_slot)


def _init_tqdm_worker(lock: object) -> None:
    tqdm.set_lock(lock)


class SweepExperimentRunner:
    def __init__(
        self,
        *,
        sweep_cfg: dict[str, Any],
        results_dir: Path,
        fl_only: bool = False,
        max_parallel_groups: int = 1,
    ) -> None:
        self.sweep_cfg = sweep_cfg
        self.results_dir = results_dir
        self.fl_only = fl_only
        if int(max_parallel_groups) < 1:
            raise ValueError(f"max_parallel_groups must be >= 1, got {max_parallel_groups}")
        self.max_parallel_groups = int(max_parallel_groups)
        self.csv_writer = SweepResultsWriter()
        self.seed_summary_builder = SeedSummaryBuilder()

    def run(self) -> SweepRunResults:
        experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        experiment_dir = self.results_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        run_entries = build_run_configs(
            sweep_cfg=self.sweep_cfg,
            experiment_dir=experiment_dir,
            fl_only=self.fl_only,
        )

        grouped_entries: dict[int, list[tuple[dict[str, dict[str, Any]], RunConfig]]] = {}
        group_order: list[int] = []
        for run_group_id, overrides, run_config in run_entries:
            if run_group_id not in grouped_entries:
                grouped_entries[run_group_id] = []
                group_order.append(run_group_id)
            grouped_entries[run_group_id].append((overrides, run_config))

        run_records: list[dict] = []
        sweep_result_rows_per_seed: list[dict] = []
        sweep_result_rows_agg: list[dict] = []
        group_results: list[SweepGroupResult] = []

        group_tasks = [
            (
                run_group_id,
                grouped_entries[run_group_id],
                self.fl_only,
                (idx % self.max_parallel_groups) if self.max_parallel_groups > 1 else None,
            )
            for idx, run_group_id in enumerate(group_order)
        ]
        execution_results: list[GroupExecutionResult]
        if self.max_parallel_groups > 1 and len(group_tasks) > 1:
            tqdm_lock = RLock()
            with ProcessPoolExecutor(
                max_workers=self.max_parallel_groups,
                initializer=_init_tqdm_worker,
                initargs=(tqdm_lock,),
            ) as executor:
                execution_results = list(
                    tqdm(
                        executor.map(_execute_run_group_from_tuple, group_tasks),
                        total=len(group_tasks),
                        desc="Running sweep groups",
                        unit="group",
                        position=0,
                    )
                )
        else:
            execution_results = [
                _execute_run_group_from_tuple(task)
                for task in tqdm(group_tasks, total=len(group_tasks), desc="Running sweep groups", unit="group")
            ]

        for exec_result in execution_results:
            run_records.append(exec_result.run_record)
            sweep_result_rows_per_seed.extend(exec_result.sweep_result_rows_per_seed)
            if exec_result.sweep_result_row_agg is not None:
                sweep_result_rows_agg.append(exec_result.sweep_result_row_agg)
            group_results.append(exec_result.group_result)

        with open(experiment_dir / "sweep_runs.json", "w", encoding="utf-8") as f:
            json.dump(run_records, f, indent=2)
        self.csv_writer.write_sweep_results(experiment_dir, sweep_result_rows_agg)
        self.csv_writer.write_sweep_results_per_seed(experiment_dir, sweep_result_rows_per_seed)
        return SweepRunResults(
            experiment_dir=experiment_dir,
            groups=group_results,
        )

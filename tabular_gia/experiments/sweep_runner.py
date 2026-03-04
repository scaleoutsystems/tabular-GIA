from __future__ import annotations

import csv
import itertools
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from helper.helpers import read_yaml, write_json
from leakpro.utils.seed import seed_everything
from metrics.tabular_metrics import RUN_SUMMARY_CSV_FIELDS
from tabular_gia.runner.run import RunEngine, RunSpec


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
    fl_section: dict[str, dict[str, list[Any]]],
):
    for protocol in protocols:
        fl_protocol_grid = fl_section[protocol]
        for base_override in _iter_grid(base_grid):
            for dataset_override in _iter_grid(dataset_grid):
                for model_override in _iter_grid(model_grid):
                    for gia_override in _iter_grid(gia_grid):
                        for fl_override in _iter_grid(fl_protocol_grid):
                            yield protocol, base_override, dataset_override, model_override, gia_override, fl_override


def _resolve_dataset_cfg(dataset_cfg: dict[str, Any], dataset_override: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    dataset_cfg_run = deepcopy(dataset_cfg)
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
    use_model_presets: bool,
    model_override: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    if use_model_presets:
        preset_name = model_override["name"]
        presets = model_cfg["presets"]
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found in model config.")
        return deepcopy(presets[preset_name])
    return deepcopy(model_override)


def _resolve_gia_cfg(
    *,
    protocol: str,
    gia_cfg_root: dict[str, Any],
    gia_override: dict[str, Any],
) -> dict[str, Any]:
    if protocol not in gia_cfg_root:
        raise ValueError(f"Missing GIA protocol config: '{protocol}'.")
    protocol_cfg = deepcopy(gia_cfg_root[protocol])
    if "invertingconfig" not in protocol_cfg:
        raise ValueError(f"Missing GIA config at '{protocol}.invertingconfig'.")
    invertingconfig = protocol_cfg["invertingconfig"]

    if gia_override:
        invertingconfig.update(gia_override)
    protocol_cfg["invertingconfig"] = invertingconfig
    return protocol_cfg


def build_run_specs(
    sweep_cfg: dict[str, Any],
    *,
    base_cfg: dict[str, Any],
    dataset_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    config_dir: Path,
) -> list[dict[str, Any]]:
    gia_cfg_root = read_yaml(config_dir / "gia" / "gia.yaml")

    base_section = sweep_cfg["base"]
    dataset_section = sweep_cfg["dataset"]
    model_section = sweep_cfg["model"]
    fl_section = sweep_cfg["fl"]
    gia_section = sweep_cfg["gia"]

    protocols = [str(p) for p in base_section["protocol"]]
    if not protocols:
        raise ValueError("Sweep config must define base.protocol as a non-empty list.")

    if "seed" not in base_section:
        raise ValueError("Sweep config must define base.seed as a non-empty list.")
    raw_seeds = base_section["seed"]
    if not isinstance(raw_seeds, list):
        raise ValueError("Sweep config field base.seed must be a list.")
    seeds = [int(s) for s in _unique_values(raw_seeds)]
    if not seeds:
        raise ValueError("Sweep config must define at least one seed.")

    base_grid = {k: v for k, v in base_section.items() if k not in {"protocol", "seed"}}
    dataset_grid = dict(dataset_section)
    gia_grid = dict(gia_section)

    use_model_presets = bool(model_section["use_preset"])
    if use_model_presets:
        model_names = model_section["name"]
        if not model_names:
            raise ValueError("Sweep config sets model.use_preset=true but model.name is missing/empty.")
        model_grid = {"name": model_names}
    else:
        model_grid = {k: v for k, v in model_section.items() if k not in ("use_preset", "name")}
        if not model_grid:
            raise ValueError("Sweep must define model.name (with use_preset=true) or explicit model parameter lists.")

    run_specs: list[dict[str, Any]] = []
    protocol_fl_defaults: dict[str, dict[str, Any]] = {}
    total_specs = 0
    common_size = _grid_size(base_grid) * _grid_size(dataset_grid) * _grid_size(model_grid) * _grid_size(gia_grid)

    for protocol in protocols:
        fl_protocol_grid = fl_section[protocol]
        fl_cfg_path = config_dir / "fl" / f"{protocol}.yaml"
        protocol_fl_defaults[protocol] = read_yaml(fl_cfg_path)
        total_specs += common_size * _grid_size(fl_protocol_grid)

    with tqdm(total=total_specs, desc="Building sweep run specs", unit="spec", leave=False) as bar:
        run_id = 0
        for protocol, base_override, dataset_override, model_override, gia_override, fl_override in _iter_run_combos(
            protocols=protocols,
            base_grid=base_grid,
            dataset_grid=dataset_grid,
            model_grid=model_grid,
            gia_grid=gia_grid,
            fl_section=fl_section,
        ):
            bar.update(1)
            run_id += 1

            base_cfg_run = deepcopy(base_cfg)
            base_cfg_run["protocol"] = protocol
            base_cfg_run.update(base_override)
            base_cfg_run.pop("seed", None)

            dataset_cfg_run = _resolve_dataset_cfg(dataset_cfg, dataset_override, config_dir)
            model_cfg_run = _resolve_model_cfg(
                use_model_presets=use_model_presets,
                model_override=model_override,
                model_cfg=model_cfg,
            )
            gia_cfg_run = _resolve_gia_cfg(
                protocol=protocol,
                gia_cfg_root=gia_cfg_root,
                gia_override=gia_override,
            )
            if "attack_schedule" in base_cfg_run:
                gia_cfg_run["attack_schedule"] = base_cfg_run["attack_schedule"]

            fl_cfg_run = deepcopy(protocol_fl_defaults[protocol])
            fl_cfg_run.update(fl_override)

            overrides = {
                "base": base_override,
                "dataset": dataset_override,
                "model": model_override,
                "gia": gia_override,
                "fl": fl_override,
            }
            run_specs.append(
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

    return run_specs


SWEEP_RESULTS_CSV_FIELDS = (
    "run_id",
    "seed",
    *RUN_SUMMARY_CSV_FIELDS,
)


class SweepExperimentRunner:
    def __init__(
        self,
        *,
        sweep_cfg: dict,
        base_cfg: dict,
        dataset_cfg: dict,
        model_cfg: dict,
        config_dir: Path,
        results_dir: Path,
        fl_only: bool = False,
    ) -> None:
        self.sweep_cfg = sweep_cfg
        self.base_cfg = base_cfg
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.config_dir = config_dir
        self.results_dir = results_dir
        self.fl_only = fl_only

    def _write_sweep_results_csv(self, out_path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        for row in rows:
            unknown = sorted(set(row.keys()) - set(SWEEP_RESULTS_CSV_FIELDS))
            if unknown:
                raise ValueError(f"Unexpected sweep_results.csv fields: {unknown}")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(SWEEP_RESULTS_CSV_FIELDS))
            writer.writeheader()
            for row in rows:
                missing = [field for field in SWEEP_RESULTS_CSV_FIELDS if field not in row]
                if missing:
                    raise ValueError(f"sweep_results.csv row is missing required fields: {missing}")
                writer.writerow({field: row[field] for field in SWEEP_RESULTS_CSV_FIELDS})

    def run(self) -> None:
        experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        experiment_dir = self.results_dir / experiment_id
        run_specs = build_run_specs(
            sweep_cfg=self.sweep_cfg,
            base_cfg=self.base_cfg,
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            config_dir=self.config_dir,
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        run_records: list[dict] = []
        sweep_result_rows: list[dict] = []
        for spec in run_specs:
            run_id = int(spec["run_id"])
            protocol = str(spec["base_cfg"]["protocol"])
            seeds = [int(s) for s in spec["seeds"]]

            run_dir = self.results_dir / protocol / experiment_id / f"run_{run_id:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "run_config.json",
                {
                    "base": spec["base_cfg"],
                    "seeds": seeds,
                    "dataset": spec["dataset_cfg"],
                    "model": spec["model_cfg"],
                    "gia": {protocol: spec["gia_cfg"]},
                    "fl": spec["fl_cfg"],
                    "overrides": spec["overrides"],
                },
            )

            for run_seed in seeds:
                seed_everything(run_seed)
                dataset_cfg_run = deepcopy(spec["dataset_cfg"])
                dataset_cfg_run["seed"] = int(run_seed)
                seed_dir = run_dir / f"seed_{int(run_seed):04d}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                base_cfg_seed = deepcopy(spec["base_cfg"])
                base_cfg_seed["seed"] = int(run_seed)
                write_json(
                    seed_dir / "resolved_config.json",
                    {
                        "base": base_cfg_seed,
                        "dataset": dataset_cfg_run,
                        "model": spec["model_cfg"],
                        "gia": {protocol: spec["gia_cfg"]},
                        "fl": spec["fl_cfg"],
                        "overrides": spec["overrides"],
                    },
                )

                run_spec = RunSpec(
                    protocol=protocol,
                    dataset_cfg=dataset_cfg_run,
                    model_cfg=deepcopy(spec["model_cfg"]),
                    fl_cfg=deepcopy(spec["fl_cfg"]),
                    gia_cfg=deepcopy(spec["gia_cfg"]),
                    results_dir=seed_dir / "artifacts",
                    fl_only=self.fl_only,
                )
                run_summary = RunEngine(run_spec).run()
                if run_summary is None:
                    raise ValueError(
                        f"Run summary missing for run_id={run_id}, seed={run_seed}. "
                        "This sweep mode requires run_summary output."
                    )

                row = {"run_id": run_id, "seed": int(run_seed)}
                row.update(run_summary)
                sweep_result_rows.append(row)

            run_records.append(
                {
                    "run_id": run_id,
                    "protocol": protocol,
                    "seeds": seeds,
                    "run_dir": str(run_dir),
                    "overrides": spec["overrides"],
                }
            )

        with open(experiment_dir / "sweep_runs.json", "w", encoding="utf-8") as f:
            json.dump(run_records, f, indent=2)
        self._write_sweep_results_csv(experiment_dir / "sweep_results.csv", sweep_result_rows)

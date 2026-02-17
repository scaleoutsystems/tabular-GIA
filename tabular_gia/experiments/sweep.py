"""Experiment sweep utilities."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

from tqdm import tqdm

from helper.helpers import read_yaml


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
    dataset_cfg_run = copy.deepcopy(dataset_cfg)
    for dataset_key, value in dataset_override.items():
        if dataset_key == "dataset_path_and_meta_path":
            dataset_cfg_run["dataset_path"], dataset_cfg_run["dataset_meta_path"] = value
        else:
            dataset_cfg_run[dataset_key] = value

    for path_key in ("dataset_path", "dataset_meta_path"):
        dataset_cfg_run[path_key] = str(config_dir.parent / dataset_cfg_run[path_key])
    return dataset_cfg_run


def _resolve_model_cfg(
    *,
    use_model_presets: bool,
    model_override: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    if use_model_presets:
        preset_name = model_override["name"]
        presets = model_cfg.get("presets", {})
        preset_cfg = presets.get(preset_name)
        if preset_cfg is None:
            raise ValueError(f"Preset '{preset_name}' not found in model config.")
        return copy.deepcopy(preset_cfg)
    return copy.deepcopy(model_override)


def _resolve_gia_cfg(
    *,
    protocol: str,
    gia_cfg_root: dict[str, Any],
    gia_override: dict[str, Any],
) -> dict[str, Any]:
    protocol_cfg = copy.deepcopy(gia_cfg_root.get(protocol, {}))
    invertingconfig = protocol_cfg.get("invertingconfig")
    if invertingconfig is None:
        if not gia_override:
            raise ValueError(f"Missing GIA config at '{protocol}.invertingconfig'.")
        invertingconfig = {}

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
    config_dir: Path
) -> list[dict[str, Any]]:
    """Build resolved run configurations from in-memory sweep and base configs."""

    gia_cfg_root = read_yaml(config_dir / "gia" / "gia.yaml")

    base_section = sweep_cfg["base"]
    dataset_section = sweep_cfg["dataset"]
    model_section = sweep_cfg["model"]
    fl_section = sweep_cfg["fl"]
    gia_section = sweep_cfg["gia"]

    protocols = [str(p) for p in base_section["protocol"]]
    if not protocols:
        raise ValueError("Sweep config must define base.protocol as a non-empty list.")

    base_grid = {k: v for k, v in base_section.items() if k != "protocol"}
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

            base_cfg_run = copy.deepcopy(base_cfg)
            base_cfg_run["protocol"] = protocol
            base_cfg_run.update(base_override)

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

            fl_cfg_run = copy.deepcopy(protocol_fl_defaults[protocol])
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
                    "overrides": overrides,
                    "base_cfg": base_cfg_run,
                    "dataset_cfg": dataset_cfg_run,
                    "model_cfg": model_cfg_run,
                    "fl_cfg": fl_cfg_run,
                    "gia_cfg": gia_cfg_run,
                }
            )

    return run_specs

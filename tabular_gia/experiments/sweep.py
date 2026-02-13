"""Experiment sweep utilities."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

from tqdm import tqdm

from helper.helpers import read_yaml


def _iter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

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
    common_combo_lists = (
        _iter_grid(base_grid),
        _iter_grid(dataset_grid),
        _iter_grid(model_grid),
        _iter_grid(gia_grid),
    )
    protocol_combo_lists: dict[str, list[dict[str, Any]]] = {}
    protocol_fl_defaults: dict[str, dict[str, Any]] = {}
    total_specs = 0

    for protocol in protocols:
        fl_protocol_grid = fl_section[protocol]
        fl_combos = _iter_grid(fl_protocol_grid)
        protocol_combo_lists[protocol] = fl_combos
        fl_cfg_path = config_dir / "fl" / f"{protocol}.yaml"
        protocol_fl_defaults[protocol] = read_yaml(fl_cfg_path)
        total_specs += (
            len(common_combo_lists[0])
            * len(common_combo_lists[1])
            * len(common_combo_lists[2])
            * len(common_combo_lists[3])
            * len(fl_combos)
        )

    with tqdm(total=total_specs, desc="Building sweep run specs", unit="spec", leave=False) as bar:
        for protocol in protocols:
            fl_combos = protocol_combo_lists[protocol]
            for base_override, dataset_override, model_override, gia_override, fl_override in itertools.product(
                common_combo_lists[0],
                common_combo_lists[1],
                common_combo_lists[2],
                common_combo_lists[3],
                fl_combos,
            ):
                bar.update(1)

                base_cfg_run = copy.deepcopy(base_cfg)
                base_cfg_run["protocol"] = protocol
                base_cfg_run.update(base_override)
                dataset_cfg_run = copy.deepcopy(dataset_cfg)
                gia_cfg_root_run = copy.deepcopy(gia_cfg_root)

                fl_cfg_run = copy.deepcopy(protocol_fl_defaults[protocol])
                fl_cfg_run.update(fl_override)

                for dataset_key, value in dataset_override.items():
                    if dataset_key == "dataset_path_and_meta_path":
                        dataset_cfg_run["dataset_path"], dataset_cfg_run["dataset_meta_path"] = value
                    else:
                        dataset_cfg_run[dataset_key] = value

                # Resolve to concrete model kwargs so written model.yaml is uniform across runs.
                if use_model_presets:
                    preset_name = model_override["name"]
                    presets = model_cfg.get("presets", {})
                    preset_cfg = presets.get(preset_name)
                    if preset_cfg is None:
                        raise ValueError(f"Preset '{preset_name}' not found in model config.")
                    model_cfg_run = copy.deepcopy(preset_cfg)
                else:
                    model_cfg_run = copy.deepcopy(model_override)

                for path_key in ("dataset_path", "dataset_meta_path"):
                    dataset_cfg_run[path_key] = str(config_dir.parent / dataset_cfg_run[path_key])

                if gia_override:
                    protocol_cfg = gia_cfg_root_run.setdefault(protocol, {})
                    inv_cfg = protocol_cfg.setdefault("invertingconfig", {})
                    inv_cfg.update(gia_override)
                gia_cfg_run = gia_cfg_root_run.get(protocol, {}).get("invertingconfig")
                if gia_cfg_run is None:
                    raise ValueError(f"Missing GIA config at '{protocol}.invertingconfig'.")

                run_id = len(run_specs) + 1
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

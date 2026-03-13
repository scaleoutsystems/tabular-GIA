# tabular_gia/experiments/experiment_template.py
from pathlib import Path
from typing import Any

from experiments.sweep_runner import SweepExperimentRunner, SweepRunResults


def build_sweep_cfg() -> dict[str, Any]:
    base = {
        "default": {
            "protocol": "fedavg",
            "seed": 42,
        },
        "grid": {
            "protocol": ["fedsgd", "fedavg"],
            "seed": [7, 13, 42],
        },
    }

    dataset = {
        "default": {
            "dataset_path_and_meta_path": [
                "data/binary/adult/adult.csv",
                "data/binary/adult/adult.yaml",
            ],
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": False,
            "batch_size": 256,
            "train_frac": 0.70,
            "val_frac": 0.15,
            "test_frac": 0.15,
            "partition_strategy": "iid",
            "dirichlet_alpha": 0.3,
            "min_client_samples": 1,
            "dirichlet_max_attempts": 50,
        },
        "grid": {},
    }

    model = {
        "default": {
            "preset": "small",
            "arch": "mlp",
            "d_hidden": 32,
            "n_hidden_layers": 1,
            "norm": "none",
            "dropout": 0.0,
            "activation": "relu",
            "presets": {
                "small": {
                    "arch": "mlp",
                    "d_hidden": 32,
                    "n_hidden_layers": 1,
                    "norm": "layernorm",
                    "dropout": 0.0,
                    "activation": "gelu",
                },
                "medium": {
                    "arch": "mlp",
                    "d_hidden": 64,
                    "n_hidden_layers": 2,
                    "norm": "layernorm",
                    "dropout": 0.1,
                    "activation": "gelu",
                },
                "large": {
                    "arch": "mlp",
                    "d_hidden": 128,
                    "n_hidden_layers": 3,
                    "norm": "layernorm",
                    "dropout": 0.2,
                    "activation": "gelu",
                },
                "fttransformer": {"arch": "fttransformer"},
                "resnet": {"arch": "resnet"},
            },
        },
        "grid": {
            "preset": ["small", "fttransformer"],
        },
    }

    fl = {
        "fedsgd": {
            "default": {
                "local_steps": 1,
                "local_epochs": 1,
                "num_clients": 10,
                "min_exposure": 10.0,
                "optimizer": "MetaSGD",
                "lr": 0.01,
            },
            "grid": {},
        },
        "fedavg": {
            "default": {
                "local_steps": "all",
                "local_epochs": 1,
                "max_client_dataset_examples": 64,
                "num_clients": 3,
                "min_exposure": 10.0,
                "optimizer": "MetaAdam",
                "lr": 0.01,
            },
            "grid": {},
        },
    }

    gia = {
        "default": {
            "attack_mode": "round_checkpoint",
            "fixed_batch_k": 1,
            "attack_schedule": "auto",
            "auto_checkpoints": 6,
            "attack_exposure_milestones": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0],
            "invertingconfig": {
                "label_known": True,
                "attack_lr": 0.03,
                "at_iterations": 1,
                "data_extension": "GiaTabularExtension",
            },
        },
        "grid": {
            # Nested key sweep example:
            # "invertingconfig": {"at_iterations": [100, 500, 1000]},
        },
    }

    return {
        "base": base,
        "dataset": dataset,
        "model": model,
        "fl": fl,
        "gia": gia,
    }


def analyze_results(sweep_results: SweepRunResults) -> None:
    # Example:
    # for group in sweep_results.groups:
    #     print(group.run_group_id, group.protocol, group.aggregated_run_summary)
    return None


def run_experiment(
    *,
    project_root: Path,
    results_dir: Path,
    fl_only: bool = False,
) -> SweepRunResults:
    runner = SweepExperimentRunner(
        sweep_cfg=build_sweep_cfg(),
        results_dir=results_dir / "experiments",
        fl_only=fl_only,
    )
    sweep_results = runner.run()
    analyze_results(sweep_results)
    return sweep_results

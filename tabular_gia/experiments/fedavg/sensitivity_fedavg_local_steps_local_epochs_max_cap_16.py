# tabular_gia/experiments/experiment_template.py
from typing import Any

from experiments.sweep_runner import SweepExperimentRunner


def build_sweep_cfg() -> dict[str, Any]:
    base = {
        "default": {
            "protocol": "fedsgd",
            "seed": 42,
        },
        "grid": {
            "protocol": "fedavg",
            "seed": [7, 13, 42],
        },
    }

    dataset = {
        "default": {
            "dataset_path_and_meta_path": [
                "data/regression/california_housing/california_housing.csv",
                "data/regression/california_housing/california_housing.yaml",
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
        "grid": {
            "dataset_path_and_meta_path": [
                [
                    "data/binary/mimic_admission_tier3_binary/mimic_admission_tier3_binary.train.csv",
                    "data/binary/mimic_admission_tier3_binary/mimic_admission_tier3_binary.yaml",
                ],
            ],
            "batch_size": [4,8,16],
        },
    }

    model = {
        "default": {
            "preset": "small",
            "arch": "mlp",
            "d_hidden": 32,
            "n_hidden_layers": 2,
            "norm": "layernorm",
            "dropout": 0.0,
            "activation": "gelu",
            "presets": {
                "small": {
                    "arch": "mlp",
                    "d_hidden": 32,
                    "n_hidden_layers": 2,
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
            "preset": ["small", "resnet", "fttransformer"],
        },
    }

    fl = {
        "fedsgd": {
            "default": {
                "local_steps": 1,
                "local_epochs": 1,
                "num_clients": 10,
                "min_exposure": 25.0,
                "optimizer": "MetaSGD",
                "lr": 0.01,
                "vectorized_clients": True,
            },
            "grid": {},
        },
        "fedavg": {
            "default": {
                "local_steps": "all",
                "local_epochs": 1,
                "max_client_dataset_examples": 256,
                "num_clients": 3,
                "min_exposure": 25.0,
                "optimizer": "MetaSGD",
                "lr": 0.01,
                "vectorized_clients": True,
            },
            "grid": {
                "local_epochs": [1,2,5],
                "max_client_dataset_examples": 16,

            },
        },
    }

    gia = {
        "default": {
            "attack_mode": "round_checkpoint",
            "fixed_batch_k": 1,
            "attack_schedule": "auto",
            "auto_checkpoints": 5,
            "attack_exposure_milestones": [0.0, 1.0, 5.0, 10.0, 25.0],
            "vectorized_attacks": True,
            "invertingconfig": {
                "label_known": True,
                "attack_lr": 0.06,
                "at_iterations": 10000,
                "data_extension": "GiaTabularExtension",
            },
        },
        "grid": {
            # Nested key sweep example:
            # "invertingconfig": {"at_iterations": [100, 500, 1000]},
            "attack_schedule": "exposure",
            "attack_exposure_milestones": [[0.0, 4.0, 8.0, 16.0, 24.0]],
        },
    }

    return {
        "base": base,
        "dataset": dataset,
        "model": model,
        "fl": fl,
        "gia": gia,
    }


class FedAvgLocalStepsLocalEpochsBatchSizesMaxCap16Runner(SweepExperimentRunner):
    def __init__(
        self,
        sweep_cfg,
        results_dir,
        fl_only=False,
        max_parallel_groups: int = 1,
        resume_experiment_dir=None,
    ):
        # ignore passed sweep_cfg; use hardcoded experiment config
        _ = sweep_cfg
        super().__init__(
            sweep_cfg=build_sweep_cfg(),
            results_dir=results_dir,
            fl_only=fl_only,
            max_parallel_groups=max_parallel_groups,
            resume_experiment_dir=resume_experiment_dir,
        )

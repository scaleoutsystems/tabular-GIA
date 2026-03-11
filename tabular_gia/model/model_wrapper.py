from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from configs.model.model import ModelConfig


class ModelWrapper(nn.Module):
    def __init__(self, *, task: str) -> None:
        super().__init__()
        self.task = str(task)
        if self.task == "binary":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.task == "multiclass":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def snapshot_state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def restore_state(self, state_dict: dict) -> None:
        self.load_state_dict(state_dict)

    @staticmethod
    def resolve_model_cfg(model_cfg: ModelConfig) -> dict:
        cfg = deepcopy(model_cfg.to_dict())
        preset_name = cfg["preset"]
        if preset_name is not None:
            presets = cfg["presets"]
            if preset_name not in presets:
                raise ValueError(f"Model preset '{preset_name}' not found in model config.")
            return deepcopy(presets[preset_name])
        cfg.pop("preset")
        cfg.pop("presets")
        return cfg

    @classmethod
    def infer_encoding_mode(cls, model_cfg: ModelConfig) -> str:
        cfg = cls.resolve_model_cfg(model_cfg)
        arch = str(cfg["arch"]).strip().lower()
        if arch == "fttransformer":
            return "ordinal"
        return "onehot"

    @classmethod
    def from_config(cls, model_cfg: ModelConfig, feature_schema: dict) -> "ModelWrapper":
        cfg = cls.resolve_model_cfg(model_cfg)

        task = feature_schema["task"]
        if task == "multiclass":
            d_out = int(feature_schema["num_classes"])
        else:
            d_out = 1

        arch = str(cfg["arch"]).strip().lower()
        del cfg["arch"]
        if arch == "fttransformer":
            if cfg:
                raise ValueError(
                    f"FTTransformer uses fixed defaults and does not accept extra config keys: {sorted(cfg.keys())}"
                )
            return FTTransformerWrapper(
                task=task,
                d_out=d_out,
                n_num_features=int(feature_schema["n_num_features"]),
                cat_cardinalities=list(feature_schema["cat_cardinalities"]),
            )
        if arch != "mlp":
            raise ValueError(f"Unknown model arch '{arch}'.")

        from model.model import TabularMLP

        return TabularMLP(
            d_in=int(feature_schema["num_features"]),
            d_out=d_out,
            task=task,
            **cfg,
        )


class FTTransformerWrapper(ModelWrapper):
    def __init__(
        self,
        *,
        task: str,
        d_out: int,
        n_num_features: int,
        cat_cardinalities: list[int],
    ) -> None:
        super().__init__(task=task)
        from rtdl_revisiting_models import FTTransformer

        self.n_num_features = int(n_num_features)
        self.n_cat_features = int(len(cat_cardinalities))
        self.register_buffer(
            "cat_index_max",
            torch.tensor([int(c) - 1 for c in cat_cardinalities], dtype=torch.float32),
            persistent=False,
        )
        default_kwargs = FTTransformer.get_default_kwargs()
        self.backbone = FTTransformer(
            n_cont_features=self.n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_out=int(d_out),
            **default_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_num = x[:, : self.n_num_features].float()
        if self.n_cat_features > 0:
            x_cat = x[:, self.n_num_features : self.n_num_features + self.n_cat_features]
            x_cat = torch.clamp(x_cat, min=0.0) # this needs to change later and be removed and an explicit GIA forward needs to be defined for differentiable ordinals in the attack. adaptive attacker style because that is realistic under the HBC server assumption
            x_cat = torch.minimum(x_cat, self.cat_index_max.to(device=x_cat.device, dtype=x_cat.dtype))
            x_cat = x_cat.long()
        else:
            x_cat = None
        return self.backbone(x_num, x_cat)

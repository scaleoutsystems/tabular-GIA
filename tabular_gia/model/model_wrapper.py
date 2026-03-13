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
        if arch == "resnet": # defaults from rtdl_revisiting_models README.md
            encoding_mode = str(feature_schema["encoding_mode"]).strip().lower()
            assert encoding_mode=="onehot", "ResNet feature_schema['encoding_mode']=='onehot'"
            defaults = {
                "n_blocks": 2,
                "d_block": 192,
                "d_hidden": None,
                "d_hidden_multiplier": 2.0,
                "dropout1": 0.15,
                "dropout2": 0.0,
            }
            unknown = sorted(set(cfg.keys()) - set(defaults.keys()))
            if unknown:
                raise ValueError(f"Unknown ResNet config keys: {unknown}")
            resnet_kwargs = {**defaults, **cfg}
            return ResNetWrapper(
                task=task,
                d_out=d_out,
                d_in=int(feature_schema["num_features"]),
                **resnet_kwargs,
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
        self.gia_soft_temperature = 2
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

    def forward_gia(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable FTTransformer path for gradient inversion attacks."""
        x_num = x[:, : self.n_num_features].float()
        if self.n_cat_features <= 0:
            return self.backbone(x_num, None)

        x_cat = x[:, self.n_num_features : self.n_num_features + self.n_cat_features]
        x_cat = torch.clamp(x_cat, min=0.0)
        x_cat = torch.minimum(x_cat, self.cat_index_max.to(device=x_cat.device, dtype=x_cat.dtype))

        cat_module = self.backbone.cat_embeddings
        if cat_module is None:
            raise ValueError("cat_embeddings must not be None when categorical features are present")

        # Standard temperature parameterization: lower tau => sharper distribution.
        tau = max(float(self.gia_soft_temperature), 1e-8)
        cat_tokens = []
        for i, emb in enumerate(cat_module.embeddings):
            idx = torch.arange(emb.num_embeddings, device=x_cat.device, dtype=x_cat.dtype)
            logits = -(x_cat[:, i : i + 1] - idx[None, :]).pow(2) / tau
            probs = torch.softmax(logits, dim=-1)
            cat_tokens.append(probs @ emb.weight)
        cat_tokens = torch.stack(cat_tokens, dim=1)
        if cat_module.bias is not None:
            cat_tokens = cat_tokens + cat_module.bias

        tokens = [self.backbone.cls_embedding(x_cat.shape[:-1])]
        if self.backbone.cont_embeddings is not None:
            tokens.append(self.backbone.cont_embeddings(x_num))
        tokens.append(cat_tokens)
        return self.backbone.backbone(torch.cat(tokens, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FTTransformer forward with dual categorical handling.

        NOTE: Uses standard discrete categorical indices for normal inference/training.
        When x.requires_grad is True, uses a differentiable surrogate
        (soft weighting over category embeddings) for gradient-based inversion.
        This surrogate does not exactly match the model's discrete inference path.
        """
        x_num = x[:, : self.n_num_features].float()
        if self.n_cat_features <= 0:
            return self.backbone(x_num, None)
        x_cat = x[:, self.n_num_features : self.n_num_features + self.n_cat_features]
        x_cat = torch.clamp(x_cat, min=0.0)
        x_cat = torch.minimum(x_cat, self.cat_index_max.to(device=x_cat.device, dtype=x_cat.dtype))

        if x.requires_grad:
            return self.forward_gia(x)

        return self.backbone(x_num, x_cat.long())


class ResNetWrapper(ModelWrapper):
    def __init__(
        self,
        *,
        task: str,
        d_out: int,
        d_in: int,
        n_blocks: int,
        d_block: int,
        d_hidden: int | None,
        d_hidden_multiplier: float | None,
        dropout1: float,
        dropout2: float,
    ) -> None:
        super().__init__(task=task)
        from rtdl_revisiting_models import ResNet

        self.backbone = ResNet(
            d_in=int(d_in),
            d_out=int(d_out),
            n_blocks=int(n_blocks),
            d_block=int(d_block),
            d_hidden=None if d_hidden is None else int(d_hidden),
            d_hidden_multiplier=None if d_hidden_multiplier is None else float(d_hidden_multiplier),
            dropout1=float(dropout1),
            dropout2=float(dropout2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.float())

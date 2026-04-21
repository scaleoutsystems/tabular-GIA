from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from configs.model.model import ModelConfig


class ModelWrapper(nn.Module):
    def __init__(self, *, task: str, binary_pos_weight: float | None = None) -> None:
        super().__init__()
        self.task = str(task)
        if self.task == "binary":
            if binary_pos_weight is not None:
                self.criterion = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([float(binary_pos_weight)], dtype=torch.float32)
                )
            else:
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
        arch = cfg["arch"].strip().lower()
        if arch == "fttransformer":
            return "ordinal"
        return "onehot"

    @classmethod
    def from_config(cls, model_cfg: ModelConfig, feature_schema: dict) -> "ModelWrapper":
        cfg = cls.resolve_model_cfg(model_cfg)

        task = feature_schema["task"]
        binary_pos_weight = feature_schema.get("binary_pos_weight")
        if task == "multiclass":
            d_out = int(feature_schema["num_classes"])
        else:
            d_out = 1

        arch = cfg["arch"].strip().lower()
        del cfg["arch"]
        if arch == "fttransformer":
            if cfg:
                raise ValueError(
                    f"FTTransformer uses fixed defaults and does not accept extra config keys: {sorted(cfg.keys())}"
                )
            return FTTransformerWrapper(
                task=task,
                binary_pos_weight=binary_pos_weight,
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
                binary_pos_weight=binary_pos_weight,
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
            binary_pos_weight=binary_pos_weight,
            **cfg,
        )


class FTTransformerWrapper(ModelWrapper):
    def __init__(
        self,
        *,
        task: str,
        binary_pos_weight: float | None,
        d_out: int,
        n_num_features: int,
        cat_cardinalities: list[int],
    ) -> None:
        super().__init__(task=task, binary_pos_weight=binary_pos_weight)
        from rtdl_revisiting_models import FTTransformer

        self.n_num_features = int(n_num_features)
        self.n_cat_features = int(len(cat_cardinalities))
        self.gia_forward_mode = "logits"
        self.gia_soft_temperature = 1.0
        self.gia_init_logit_scale = 5.0
        self.cat_cardinalities = [int(c) for c in cat_cardinalities]
        self.register_buffer(
            "cat_index_max",
            torch.tensor([c - 1 for c in self.cat_cardinalities], dtype=torch.float32),
            persistent=False,
        )
        default_kwargs = FTTransformer.get_default_kwargs()
        small_kwargs = {
            'n_blocks': 2,
            'd_block': 128,
            'attention_n_heads': 4,
            'attention_dropout': 0.1,
            'ffn_d_hidden': None,
            'ffn_d_hidden_multiplier': 1.3333333333333333,
            'ffn_dropout': 0.1,
            'residual_dropout': 0.0,
            }
        self.backbone = FTTransformer(
            n_cont_features=self.n_num_features,
            cat_cardinalities=self.cat_cardinalities,
            d_out=int(d_out),
            **small_kwargs,
        )

    @property
    def gia_space_dim(self) -> int:
        return int(self.n_num_features + sum(self.cat_cardinalities))

    def to_gia_space(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_cat_features <= 0:
            return x.float()
        x_num = x[:, : self.n_num_features].float()
        x_cat = x[:, self.n_num_features : self.n_num_features + self.n_cat_features]
        x_cat = torch.clamp(x_cat, min=0.0)
        x_cat = torch.minimum(x_cat, self.cat_index_max.to(device=x_cat.device, dtype=x_cat.dtype))
        x_idx = x_cat.long()
        cat_blocks = []
        for i, card in enumerate(self.cat_cardinalities):
            if str(self.gia_forward_mode).strip().lower() == "probabilities":
                probs = torch.zeros((x.shape[0], card), device=x.device, dtype=x.dtype)
                probs.scatter_(1, x_idx[:, i : i + 1], 1.0)
                cat_blocks.append(probs)
            else:
                scale = float(self.gia_init_logit_scale)
                logits = torch.full(
                    (x.shape[0], card),
                    fill_value=-scale,
                    device=x.device,
                    dtype=x.dtype,
                )
                logits.scatter_(1, x_idx[:, i : i + 1], scale)
                cat_blocks.append(logits)
        return torch.cat([x_num, *cat_blocks], dim=1)

    def from_gia_space(self, x_gia: torch.Tensor) -> torch.Tensor:
        if self.n_cat_features <= 0:
            return x_gia[:, : self.n_num_features].float()
        x_num = x_gia[:, : self.n_num_features].float()
        start = self.n_num_features
        decoded = []
        for card in self.cat_cardinalities:
            logits = x_gia[:, start : start + card]
            idx = torch.argmax(logits, dim=1, keepdim=True).float()
            decoded.append(idx)
            start += card
        return torch.cat([x_num, *decoded], dim=1)

    def _build_tokens_from_cat_probs(self, x_num: torch.Tensor, cat_probs: list[torch.Tensor]) -> torch.Tensor:
        cat_module = self.backbone.cat_embeddings
        if cat_module is None:
            raise ValueError("cat_embeddings must not be None when categorical features are present")
        cat_tokens = []
        for probs, emb in zip(cat_probs, cat_module.embeddings):
            cat_tokens.append(probs @ emb.weight)
        cat_tokens = torch.stack(cat_tokens, dim=1)
        if cat_module.bias is not None:
            cat_tokens = cat_tokens + cat_module.bias

        tokens = [self.backbone.cls_embedding(x_num.shape[:-1])]
        if self.backbone.cont_embeddings is not None:
            tokens.append(self.backbone.cont_embeddings(x_num))
        tokens.append(cat_tokens)
        return self.backbone.backbone(torch.cat(tokens, dim=1))

    def forward_gia(self, x_gia: torch.Tensor) -> torch.Tensor:
        """Differentiable FTTransformer path over relaxed categorical variables."""
        x_num = x_gia[:, : self.n_num_features].float()
        if self.n_cat_features <= 0:
            return self.backbone(x_num, None)
        mode = str(self.gia_forward_mode).strip().lower()
        start = self.n_num_features
        cat_probs = []
        for card in self.cat_cardinalities:
            cat_slice = x_gia[:, start : start + card]
            if mode == "probabilities":
                probs = torch.clamp(cat_slice, min=0.0)
                denom = probs.sum(dim=1, keepdim=True)
                uniform = torch.full_like(probs, 1.0 / float(card))
                probs = torch.where(denom > 0, probs / denom.clamp_min(1e-12), uniform)
                cat_probs.append(probs)
            else:
                tau = max(float(self.gia_soft_temperature), 1e-8)
                logits = cat_slice / tau
                cat_probs.append(torch.softmax(logits, dim=1))
            start += card
        return self._build_tokens_from_cat_probs(x_num, cat_probs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FTTransformer forward.

        Uses native discrete path for standard ordinal input shape
        [n_num_features + n_cat_features]. Uses differentiable simplex/logit
        GIA path for expanded input shape [n_num_features + sum(cat_cardinalities)].
        """
        if x.shape[1] == self.gia_space_dim and self.n_cat_features > 0:
            return self.forward_gia(x)

        x_num = x[:, : self.n_num_features].float()
        if self.n_cat_features <= 0:
            return self.backbone(x_num, None)
        x_cat = x[:, self.n_num_features : self.n_num_features + self.n_cat_features]
        x_cat = torch.clamp(x_cat, min=0.0)
        x_cat = torch.minimum(x_cat, self.cat_index_max.to(device=x_cat.device, dtype=x_cat.dtype))
        return self.backbone(x_num, x_cat.long())


class ResNetWrapper(ModelWrapper):
    def __init__(
        self,
        *,
        task: str,
        binary_pos_weight: float | None,
        d_out: int,
        d_in: int,
        n_blocks: int,
        d_block: int,
        d_hidden: int | None,
        d_hidden_multiplier: float | None,
        dropout1: float,
        dropout2: float,
    ) -> None:
        super().__init__(task=task, binary_pos_weight=binary_pos_weight)
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

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


NormType = Literal["batchnorm", "layernorm", "none"]
GatingType = Literal["swiglu", "geglu"]


def _make_norm(norm_type: NormType, d: int) -> nn.Module:
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(d)
    if norm_type == "layernorm":
        return nn.LayerNorm(d)
    return nn.Identity()


class FeatureDropout(nn.Module):
    """Drop entire input features (columns) across the batch."""

    def __init__(self, p: float) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"FeatureDropout p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.rand(x.shape[-1], device=x.device) < keep_prob
        mask = mask.to(dtype=x.dtype) / keep_prob
        return x * mask.unsqueeze(0)


class RandomFourierFeatures(nn.Module):
    """Random Fourier features for numeric enrichment."""

    def __init__(self, in_features: int, num_bands: int, sigma: float = 1.0) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be > 0")
        b = torch.randn(num_bands, in_features) / sigma
        self.register_buffer("b", b, persistent=False)

    @property
    def out_features(self) -> int:
        return self.b.shape[0] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.b.t()
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PeriodicEmbeddings(nn.Module):
    """Learned periodic embeddings for numeric enrichment."""

    def __init__(self, in_features: int, num_bands: int) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be > 0")
        self.weight = nn.Parameter(torch.randn(num_bands, in_features))

    @property
    def out_features(self) -> int:
        return self.weight.shape[0] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.weight.t()
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class GatedMLPBlock(nn.Module):
    """Pre-norm gated MLP block with residual connection."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float,
        norm_type: NormType,
        gating: GatingType,
        layer_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.norm = _make_norm(norm_type, d_model)
        self.fc1 = nn.Linear(d_model, d_hidden * 2)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gating = gating
        if layer_scale is None:
            self.layer_scale = None
        else:
            self.layer_scale = nn.Parameter(torch.full((d_model,), layer_scale))

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        if self.gating == "geglu":
            return a * F.gelu(b)
        return a * F.silu(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self._gate(y)
        y = self.dropout(y)
        y = self.fc2(y)
        if self.layer_scale is not None:
            y = y * self.layer_scale
        return x + y


class ResidualGatedMLP(nn.Module):
    """Residual gated MLP backbone."""

    def __init__(
        self,
        d_model: int,
        depth: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        norm_type: NormType = "layernorm",
        gating: GatingType = "swiglu",
        layer_scale: float | None = 1e-3,
    ) -> None:
        super().__init__()
        d_hidden = int(d_model * mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                GatedMLPBlock(
                    d_model=d_model,
                    d_hidden=d_hidden,
                    dropout=dropout,
                    norm_type=norm_type,
                    gating=gating,
                    layer_scale=layer_scale,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TabularResidualGatedMLP(nn.Module):
    """Tabular model with optional numeric enrichment and residual gated MLP."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 256,
        depth: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        norm_type: NormType = "batchnorm",
        gating: GatingType = "swiglu",
        layer_scale: float | None = 1e-3,
        num_numeric: int | None = None,
        use_fourier: bool = False,
        use_periodic: bool = False,
        num_bands: int = 8,
        feature_dropout: float = 0.0,
        cat_feature_dropout: float | None = None,
    ) -> None:
        super().__init__()
        if use_fourier and use_periodic:
            raise ValueError("use_fourier and use_periodic are mutually exclusive.")
        self.num_numeric = num_numeric

        self.num_norm = None
        if num_numeric is not None and num_numeric > 0:
            self.num_norm = _make_norm(norm_type, num_numeric)

        self.num_enrichment: nn.Module | None = None
        if num_numeric is not None and num_numeric > 0:
            if use_fourier:
                self.num_enrichment = RandomFourierFeatures(num_numeric, num_bands=num_bands)
            elif use_periodic:
                self.num_enrichment = PeriodicEmbeddings(num_numeric, num_bands=num_bands)

        self.feature_dropout = FeatureDropout(feature_dropout) if feature_dropout > 0 else nn.Identity()
        if cat_feature_dropout is None:
            self.cat_feature_dropout = None
        else:
            self.cat_feature_dropout = FeatureDropout(cat_feature_dropout)

        d_in_total = d_in
        if self.num_enrichment is not None:
            d_in_total = d_in + self.num_enrichment.out_features

        self.in_proj = nn.Linear(d_in_total, d_model) if d_in_total != d_model else nn.Identity()
        self.backbone = ResidualGatedMLP(
            d_model=d_model,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            norm_type=norm_type,
            gating=gating,
            layer_scale=layer_scale,
        )
        self.head_norm = _make_norm(norm_type, d_model)
        self.head = nn.Linear(d_model, d_out)

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.num_numeric is None:
            return x, x.new_zeros((x.shape[0], 0))
        x_num = x[:, : self.num_numeric]
        x_cat = x[:, self.num_numeric :]
        return x_num, x_cat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_num, x_cat = self._split_features(x)
        if self.num_norm is not None:
            x_num = self.num_norm(x_num)
        if self.num_enrichment is not None:
            enrich = self.num_enrichment(x_num)
            x_num = torch.cat([x_num, enrich], dim=-1)

        if self.cat_feature_dropout is not None:
            x_cat = self.cat_feature_dropout(x_cat)
        x = torch.cat([x_num, x_cat], dim=-1) if x_cat.numel() else x_num
        x = self.feature_dropout(x)

        x = self.in_proj(x)
        x = self.backbone(x)
        x = self.head_norm(x)
        return self.head(x)

import torch
from torch import nn


class TabularMLP(nn.Module):
    """"""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int = 2,
        norm: str = "batchnorm",
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.hidden = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        for layer_idx in range(max(num_layers - 1, 0)):
            in_dim = d_in if layer_idx == 0 else d_hidden
            self.hidden.append(nn.Linear(in_dim, d_hidden))
            if norm == "batchnorm":
                self.norms.append(nn.BatchNorm1d(d_hidden))
            elif norm == "layernorm":
                self.norms.append(nn.LayerNorm(d_hidden))
            elif norm == "none":
                self.norms.append(nn.Identity())
            else:
                raise ValueError(f"Unknown norm: {norm}")
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())

        out_in = d_in if num_layers == 1 else d_hidden
        self.out = nn.Linear(out_in, d_out)
        for layer in list(self.hidden) + [self.out]:
            if not isinstance(layer, nn.Linear):
                continue
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm, drop in zip(self.hidden, self.norms, self.dropouts):
            x = layer(x)
            x = norm(x)
            x = self.act(x)
            x = drop(x)
        return self.out(x)

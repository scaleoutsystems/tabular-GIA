import torch
from torch import nn
from model.model_wrapper import ModelWrapper


class TabularMLP(ModelWrapper):
    """"""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        n_hidden_layers: int,
        norm: str,
        dropout: float,
        activation: str,
        task: str,
        binary_pos_weight: float | None = None,
    ) -> None:
        super().__init__(task=task, binary_pos_weight=binary_pos_weight)
        if n_hidden_layers < 0:
            raise ValueError("n_hidden_layers must be >= 0")

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

        for layer_idx in range(n_hidden_layers):
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

        out_in = d_in if n_hidden_layers == 0 else d_hidden
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

from dataclasses import asdict, dataclass, field
from typing import Any


def _default_presets() -> dict[str, dict[str, Any]]:
    return {
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
        "fttransformer": {
            "arch": "fttransformer",
        },
        "resnet": {
            "arch": "resnet",
        },
    }


@dataclass
class ModelConfig:
    preset: str | None = "fttransformer" # / None

    arch: str = "mlp"
    d_hidden: int = 32
    n_hidden_layers: int = 2
    norm: str = "none"
    dropout: float = 0.0
    activation: str = "relu"

    presets: dict[str, dict[str, Any]] = field(default_factory=_default_presets)

    def to_dict(self) -> dict:
        return asdict(self)

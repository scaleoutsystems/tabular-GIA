from dataclasses import asdict, dataclass
from typing import Literal


ProtocolName = Literal["fedavg", "fedsgd"]


@dataclass
class BaseConfig:
    seed: int = 42
    protocol: ProtocolName = "fedavg"

    def to_dict(self) -> dict:
        return asdict(self)

from dataclasses import asdict, dataclass
from typing import Literal


ProtocolName = Literal["fedsgd", "fedavg"]


@dataclass
class BaseConfig:
    seed: int = 42
    protocol: ProtocolName = "fedsgd"

    def to_dict(self) -> dict:
        return asdict(self)

from dataclasses import asdict, dataclass


@dataclass
class FedSGDConfig:
    local_steps: int = 1 # NOTE: always 1 in fedsgd. 1 single batch
    local_epochs: int = 1 # NOTE: always 1 in fedsgd. 1 single batch step iteration
    num_clients: int = 10
    min_exposure: float = 25.0 # stop when least-exposed client reaches this effective pass count
    optimizer: str = "MetaSGD" # unused: because train_nostep does no optimizer step
    lr: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

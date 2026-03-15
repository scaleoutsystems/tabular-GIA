from dataclasses import asdict, dataclass


@dataclass
class FedAvgConfig:
    local_steps: str | int = "all" # integer minibatch steps per round, or 'all' for one full local dataloader pass per round (classic FedAvg)
    local_epochs: int = 1 # meaning how many iterations over local_steps batches each communication round
    max_client_dataset_examples: int | None = 256 # optional fixed cap per client dataset for FedAvg full-pass rounds n / null
    num_clients: int = 3
    min_exposure: float = 25.0 # stop when least-exposed client reaches this effective pass count
    optimizer: str = "MetaSGD"
    lr: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

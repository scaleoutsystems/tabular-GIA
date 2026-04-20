from dataclasses import asdict, dataclass
from typing import Literal


PartitionStrategy = Literal["iid", "dirichlet"]


@dataclass
class DatasetConfig:
    # dataset paths
    dataset_path: str = "data/binary/adult/adult.csv"
    dataset_meta_path: str = "data/binary/adult/adult.yaml"

    #dataset_path: str = "data/binary/mimic_admission_tier3_binary/mimic_admission_tier3_binary.train.csv"
    #dataset_meta_path: str = "data/binary/mimic_admission_tier3_binary/mimic_admission_tier3_binary.yaml"

    #dataset_path: str = "data/multiclass/pandemic_movement_office/pandemic_movement_office.csv"
    #dataset_meta_path: str = "data/multiclass/pandemic_movement_office/pandemic_movement_office.yaml"

    #dataset_path: str = "data/regression/california_housing/california_housing.csv"
    #dataset_meta_path: str = "data/regression/california_housing/california_housing.yaml"

    # dataloader gpu speedups
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False

    # dataloader specifics
    batch_size: int = 256
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # client partitioning
    partition_strategy: PartitionStrategy = "iid" # iid | dirichlet
    dirichlet_alpha: float = 0.3
    min_client_samples: int = 1
    dirichlet_max_attempts: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

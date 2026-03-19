from dataclasses import asdict, dataclass, field


@dataclass
class InvertingConfig:
    label_known: bool = True
    attack_lr: float = 0.06
    at_iterations: int = 10000
    data_extension: str = "GiaTabularExtension"

@dataclass
class GiaConfig:
    # attack_mode options:
    # - round_checkpoint: attack real updates produced in training rounds
    # - fixed_batch: attack fixed sample-ID batches repeatedly at checkpoints
    attack_mode: str = "round_checkpoint"
    fixed_batch_k: int = 1 # only active on attack_mode: fixed_batch
    # used when attack_mode=fixed_batch (fixed replay batches per client)
    # attack_schedule options:
    # - all: attack every communication round
    # - pow2: attack rounds 1,2,4,8,...
    # - fixed: attack rounds from attack_rounds list
    # - logspace: attack attack_num_checkpoints log-spaced rounds
    # - auto: attack auto_checkpoints evenly-spaced rounds
    # - exposure: attack rounds from attack_exposure_milestones
    attack_schedule: str = "auto"

    attack_rounds: list[int] = field(default_factory=list) # NOTE: attack_mode: fixed
    attack_num_checkpoints: int = 8 # NOTE: attack_mode: logspace
    auto_checkpoints: int = 5 # NOTE: attack_mode: auto
    attack_exposure_milestones: list[float] = field(default_factory=lambda: [0.0, 1.0, 5.0, 10.0, 25.0]) # NOTE: attack_mode: exposure

    # invertingconfig
    invertingconfig: InvertingConfig = field(default_factory=InvertingConfig)

    vectorized_attacks: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

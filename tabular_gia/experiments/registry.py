from experiments.sweep_runner import SweepExperimentRunner

# fedsgd
from experiments.fedsgd.experiment_fedsgd_batch_sizes import FedSGDBatchSizesRunner
from experiments.fedsgd.experiment_fedsgd_torch_modules import FedSGDTorchModulesRunner
from experiments.fedsgd.experiment_fedsgd_attack_iterations import FedSGDAttackIterationsRunner
from tabular_gia.experiments.fedsgd.experiment_fedsgd_batch_sizes_label_unkown import FedSGDBatchSizesLabelUnkownRunner

# fedavg
from experiments.fedavg.experiment_fedavg_batch_sizes import FedAvgBatchSizesRunner


EXPERIMENT_RUNNERS = {
    "sweep": SweepExperimentRunner,

    # fedsgd
    "fedsgdbatchsizes": FedSGDBatchSizesRunner,
    "fedsgdtorchmodules": FedSGDTorchModulesRunner,
    "fedsgdattackiterations": FedSGDAttackIterationsRunner,
    "fedsgdbatchsizeslabelunkown": FedSGDBatchSizesLabelUnkownRunner,

    # fedavg
    "fedavgbatchsizes": FedAvgBatchSizesRunner,
}

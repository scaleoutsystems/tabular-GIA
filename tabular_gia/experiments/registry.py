from experiments.sweep_runner import SweepExperimentRunner

# fedsgd
from experiments.fedsgd.experiment_fedsgd_batch_sizes import FedSGDBatchSizesRunner
from experiments.fedsgd.experiment_fedsgd_torch_modules import FedSGDTorchModulesRunner
from experiments.fedsgd.experiment_fedsgd_attack_iterations import FedSGDAttackIterationsRunner
from tabular_gia.experiments.fedsgd.experiment_fedsgd_batch_sizes_label_unkown import FedSGDBatchSizesLabelUnkownRunner
from experiments.fedsgd.experiment_fedsgd_fixed_batch import FedSGDFixedBatchRunner

# fedavg
from tabular_gia.experiments.fedavg.experiment_fedavg_local_steps_local_epochs_max_cap_32 import FedAvgLocalStepsLocalEpochsBatchSizesMaxCap32Runner
from tabular_gia.experiments.fedavg.sensitivity_fedavg_local_steps_local_epochs_max_cap_16 import FedAvgLocalStepsLocalEpochsBatchSizesMaxCap16Runner
from tabular_gia.experiments.fedavg.sensitivity_fedavg_local_steps_local_epochs_max_cap_64 import FedAvgLocalStepsLocalEpochsBatchSizesMaxCap64Runner


EXPERIMENT_RUNNERS = {
    "sweep": SweepExperimentRunner,

    # fedsgd
    "fedsgdbatchsizes": FedSGDBatchSizesRunner,
    "fedsgdtorchmodules": FedSGDTorchModulesRunner,
    "fedsgdattackiterations": FedSGDAttackIterationsRunner,
    "fedsgdbatchsizeslabelunkown": FedSGDBatchSizesLabelUnkownRunner,
    "fedsgdfixedbatch": FedSGDFixedBatchRunner,

    # fedavg
    "fedavglocalstepslocalepochsmaxcap32": FedAvgLocalStepsLocalEpochsBatchSizesMaxCap32Runner,
    "fedavglocalstepslocalepochsmaxcap16": FedAvgLocalStepsLocalEpochsBatchSizesMaxCap16Runner,
    "fedavglocalstepslocalepochsmaxcap64": FedAvgLocalStepsLocalEpochsBatchSizesMaxCap64Runner,
}

from experiments.sweep_runner import SweepExperimentRunner
from experiments.experiment_fedsgd_batch_sizes import FedSGDBatchSizesRunner
from experiments.experiment_fedsgd_torch_modules import FedSGDTorchModules


EXPERIMENT_RUNNERS = {
    "sweep": SweepExperimentRunner,
    "fedsgdbatchsizes": FedSGDBatchSizesRunner,
    "fedsgdtorchmodules": FedSGDTorchModules
}


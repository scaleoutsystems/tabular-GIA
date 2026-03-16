from experiments.sweep_runner import SweepExperimentRunner
from experiments.experiment_fedsgd_batch_sizes import FedSGDBatchSizesRunner


EXPERIMENT_RUNNERS = {
    "sweep": SweepExperimentRunner,
    "fedsgdbatchsizes": FedSGDBatchSizesRunner
}


from experiments.sweep_runner import SweepExperimentRunner
from experiments.experiment_fedsgd_batch_sizes import FedSGDBatchSizesRunner
from experiments.experiment_attack_iterations import AttackIterationsRunner


EXPERIMENT_RUNNERS = {
    "sweep": SweepExperimentRunner,
    "fedsgdbatchsizes": FedSGDBatchSizesRunner,
    "attackiterations": AttackIterationsRunner,
}


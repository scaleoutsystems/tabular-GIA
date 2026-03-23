from experiments.sweep_runner import SweepExperimentRunner
from experiments.fedsgd.experiment_fedsgd_batch_sizes import FedSGDBatchSizesRunner
from experiments.fedsgd.experiment_fedsgd_torch_modules import FedSGDTorchModulesRunner
from experiments.fedsgd.experiment_attack_iterations import FedSGDAttackIterationsRunner
from experiments.fedsgd.experiment_label_knowledge import FedSGDLabelKnowledgeRunner
from experiments.fedavg.experiment_fedavg_batch_sizes import FedAvgBatchSizesRunner


EXPERIMENT_RUNNERS = {
    "sweep": SweepExperimentRunner,
    "fedsgdbatchsizes": FedSGDBatchSizesRunner,
    "fedsgdtorchmodules": FedSGDTorchModulesRunner,
    "attackiterations": FedSGDAttackIterationsRunner,
    "labelknowledge": FedSGDLabelKnowledgeRunner,
    "fedavgbatchsizes": FedAvgBatchSizesRunner,
}

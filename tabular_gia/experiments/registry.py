from experiments.sweep_runner import SweepExperimentRunner
from experiments.experiment_fedsgd_batch_sizes import FedSGDBatchSizesRunner
from experiments.experiment_attack_iterations import AttackIterationsRunner
from experiments.experiment_label_knowledge import LabelKnowledgeRunner
from experiments.experiment_label_knowledge_california import LabelKnowledgeCaliforniaRunner


EXPERIMENT_RUNNERS = {
    "sweep": SweepExperimentRunner,
    "fedsgdbatchsizes": FedSGDBatchSizesRunner,
    "attackiterations": AttackIterationsRunner,
    "labelknowledge": LabelKnowledgeRunner,
    "labelknowledgecalifornia": LabelKnowledgeCaliforniaRunner,
}


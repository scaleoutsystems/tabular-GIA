"""Experiment runners."""

from typing import Any, Dict

from tabular_gia.fl.protocol import run_federated


def run_experiment(exp_cfg: Dict[str, Any], data_splits: Dict[str, Any], model: Any) -> Dict[str, Any]:
    """Run a single experiment using the FL protocol.

    Returns a dict with at least: model, metrics, attack_logs.
    """
    return run_federated(
        model=model,
        data_splits=data_splits,
        config=exp_cfg["fl"],
        attack_fn=exp_cfg.get("attack_fn"),
    )

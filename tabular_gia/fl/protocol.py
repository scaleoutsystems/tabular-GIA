"""Federated learning protocol interface."""

from typing import Any, Callable, Dict


def run_federated(model: Any, data_splits: Dict[str, Any], config: Dict[str, Any], attack_fn: Callable | None = None) -> Dict[str, Any]:
    """Run FL training and optional GIA.

    Returns a dict with:
      - model: trained model
      - metrics: task metrics
      - attack_logs: reconstruction / risk metrics
    """
    raise NotImplementedError

"""Two-step risk/utility validation (stub)."""

from typing import Any, Dict

from tabular_gia.experiments.registry import register
from tabular_gia.experiments.runners import run_experiment


@register("risk_utility")
def run_risk_utility(exp_cfg: Dict[str, Any], data_splits: Dict[str, Any], model: Any) -> Dict[str, Any]:
    """Step A: risk screening; Step B: utility validation.

    This is a stub; implement your risk/utility logic here.
    """
    result = run_experiment(exp_cfg, data_splits, model)
    return result

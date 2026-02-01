"""Model sweep experiments (stub)."""

from typing import Any, Dict

from tabular_gia.experiments.registry import register


@register("model_sweep")
def run_model_sweep(exp_cfg: Dict[str, Any], data_splits: Dict[str, Any], model_factory: Any) -> Dict[str, Any]:
    """Run a parameter sweep over model configs.

    This is a stub; implement looping logic here.
    """
    raise NotImplementedError

"""Experiment-level metrics helpers (stub)."""

from typing import Any, Dict


def summarize_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and summarize metrics from an experiment result."""
    return result.get("metrics", {})

"""Experiment registry."""

from typing import Callable, Dict

EXPERIMENTS: Dict[str, Callable] = {}


def register(name: str):
    """Decorator to register experiments by name."""
    def _wrap(fn: Callable) -> Callable:
        EXPERIMENTS[name] = fn
        return fn
    return _wrap


def get(name: str) -> Callable:
    """Fetch a registered experiment by name."""
    return EXPERIMENTS[name]

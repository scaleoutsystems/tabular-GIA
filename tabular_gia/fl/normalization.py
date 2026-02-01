"""Normalization strategies (stub)."""

from typing import Any, Dict, Tuple


def client_normalize(x: Any) -> Tuple[Any, Any, Any]:
    """Normalize using client-local stats (stub)."""
    raise NotImplementedError


def global_normalize(x: Any, config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """Normalize using global stats (stub)."""
    raise NotImplementedError

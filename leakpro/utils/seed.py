"""Function to seed randomness for different libraries."""
import random
import os

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set the seed for different libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cfg not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG "
                "to be ':4096:8' or ':16:8'."
            )


def capture_rng_state() -> dict:
    """Capture Python/NumPy/Torch RNG state for deterministic replay."""
    return {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Restore Python/NumPy/Torch RNG state."""
    if not state:
        return
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])

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
    torch.set_float32_matmul_precision("highest")

    if torch.cuda.is_available():
        cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cfg not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG "
                "to be ':4096:8' or ':16:8'."
            )

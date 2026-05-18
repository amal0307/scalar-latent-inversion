import os
import random
import numpy as np

SEED = 42

def set_determinism(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

def sanitize_output(x: np.ndarray) -> np.ndarray:
    """Guarantees finite numeric output (Stage 1 validity requirement)."""
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1e9, neginf=-1e9)
    return x

def standardize(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x)
    sd = np.std(x)
    return (x - mu) / (sd + 1e-12)
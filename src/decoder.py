import numpy as np
from .hypotheses import pick_best_hypothesis
from .priors import apply_prior

def decode(z: np.ndarray, D: int, use_prior: bool = True) -> np.ndarray:
    X_raw = pick_best_hypothesis(z, D)
    if use_prior:
        return apply_prior(X_raw)
    return X_raw
"""
Domain prior for bank/credit tabular data.
Distributions chosen from the literature on UCI German Credit and
Bank Marketing datasets. Used only to set realistic location/scale for
each reconstructed feature. Declared in the write-up.
"""
import numpy as np

# (name, marginal_mean, marginal_std, support_clip)
BANK_FEATURE_PRIOR = [
    ("age",            42.0,  12.0, (18,   95)),
    ("income",         55000, 30000, (0,    500000)),
    ("balance",        2500,  4500,  (-5000, 80000)),
    ("loan_amount",    12000, 9000,  (0,    300000)),
    ("loan_term",      36,    18,    (6,    360)),
    ("interest_rate",  7.5,   4.0,   (0,    35)),
    ("credit_score",   680,   90,    (300,  850)),
    ("debt_ratio",     0.35,  0.20,  (0,    2.0)),
    ("n_open_accounts", 8,    5,     (0,    40)),
    ("delinquencies",  0.3,   1.0,   (0,    20)),
    ("inquiries_6mo",  1.0,   1.5,   (0,    20)),
    ("employment_yrs", 7.5,   7.0,   (0,    50)),
    ("home_owner",     0.5,   0.5,   (0,    1)),
    ("has_default",    0.1,   0.3,   (0,    1)),
    ("region_code",    5.0,   3.0,   (0,    20)),
    ("product_type",   2.0,   1.0,   (0,    8)),
]

def apply_prior(X_hat: np.ndarray) -> np.ndarray:
    """Map standardized columns onto realistic feature means/scales."""
    X_hat = np.asarray(X_hat, dtype=np.float64)
    N, D = X_hat.shape
    out = np.empty_like(X_hat)
    # Per-column standardization to (0,1) then scale by prior
    for j in range(D):
        col = X_hat[:, j]
        col = (col - col.mean()) / (col.std() + 1e-12)
        if j < len(BANK_FEATURE_PRIOR):
            _, mean, std, (lo, hi) = BANK_FEATURE_PRIOR[j]
            out[:, j] = np.clip(col * std + mean, lo, hi)
        else:
            out[:, j] = col
    return out
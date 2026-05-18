import numpy as np
from scipy import stats

def hypothesis_linear_projection(z: np.ndarray, D: int) -> np.ndarray:
    """H1: z = w·x for unknown random w. We recover the 1D projection only;
    other directions are unrecoverable. Output rank-quantile features as a
    consistent partial reconstruction."""
    z = np.asarray(z, dtype=np.float64).ravel()
    N = z.size
    ranks = stats.rankdata(z, method="average") / (N + 1)
    # Each output feature is a fixed nonlinear function of the rank
    cols = []
    for j in range(D):
        phase = 2.0 * np.pi * (j + 1) / (D + 1)
        cols.append(stats.norm.ppf(np.clip(ranks, 1e-6, 1 - 1e-6)) * np.cos(phase)
                    + (z - z.mean()) / (z.std() + 1e-12) * np.sin(phase))
    return np.stack(cols, axis=1)

def hypothesis_bitpack(z: np.ndarray, D: int) -> np.ndarray:
    """H2: D features packed into mantissa. Slice mantissa into D groups."""
    z = np.asarray(z, dtype=np.float64).ravel()
    bits = z.view(np.uint64)
    mantissa = bits & ((1 << 52) - 1)
    bits_per_feature = max(1, 52 // D)
    cols = []
    for j in range(D):
        shift = j * bits_per_feature
        mask = (1 << bits_per_feature) - 1
        slice_ = ((mantissa >> shift) & mask).astype(np.float64)
        slice_ /= float(mask)  # in [0,1]
        cols.append(slice_)
    return np.stack(cols, axis=1)

def hypothesis_mixed_radix(z: np.ndarray, D: int, base: int = 10) -> np.ndarray:
    """H3: fractional part packs D digits in mixed radix."""
    z = np.asarray(z, dtype=np.float64).ravel()
    z_scaled = (z - z.min()) / (z.max() - z.min() + 1e-12)
    cols = []
    cur = z_scaled.copy()
    for _ in range(D):
        cur = cur * base
        digit = np.floor(cur)
        cur = cur - digit
        cols.append(digit / (base - 1))
    return np.stack(cols, axis=1)

def self_consistency_score(X_hat: np.ndarray, z: np.ndarray) -> float:
    """A reconstruction is internally consistent if its first PC tracks z."""
    from sklearn.decomposition import PCA
    z = np.asarray(z, dtype=np.float64).ravel()
    if X_hat.shape[1] < 2:
        return -np.inf
    pc1 = PCA(n_components=1, random_state=42).fit_transform(X_hat).ravel()
    return float(abs(np.corrcoef(pc1, z)[0, 1]))

def pick_best_hypothesis(z: np.ndarray, D: int) -> np.ndarray:
    candidates = {
        "linear": hypothesis_linear_projection(z, D),
        "bitpack": hypothesis_bitpack(z, D),
        "mixed_radix": hypothesis_mixed_radix(z, D),
    }
    scored = [(name, X, self_consistency_score(X, z)) for name, X in candidates.items()]
    scored.sort(key=lambda r: r[2], reverse=True)
    return scored[0][1]
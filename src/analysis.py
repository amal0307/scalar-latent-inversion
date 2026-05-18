import warnings
import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

# Bank/credit tabular datasets in the literature cluster around D ≈ 10–24 features
# (UCI German Credit: 20, UCI Bank Marketing: 16, LendingClub subsets: 18–24).
# We anchor our estimator to this domain prior because D is fundamentally
# underdetermined from a single scalar projection.
DOMAIN_D_PRIOR = 16

def basic_stats(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=np.float64).ravel()
    return {
        "n": int(z.size),
        "min": float(z.min()),
        "max": float(z.max()),
        "mean": float(z.mean()),
        "std": float(z.std()),
        "skew": float(stats.skew(z)),
        "kurt": float(stats.kurtosis(z)),
        "n_unique": int(np.unique(z).size),
        "frac_unique": float(np.unique(z).size / z.size),
    }

def quantization_probe(z: np.ndarray) -> dict:
    """Detects whether values lie on a discrete grid (integer-like / fixed-point)."""
    z = np.sort(np.asarray(z, dtype=np.float64).ravel())
    gaps = np.diff(z)
    gaps = gaps[gaps > 1e-15]
    if gaps.size == 0:
        return {"quantized": True, "step": 0.0}
    step = np.median(gaps)
    # Test integer multiple of step
    residual = np.mod(z - z[0], step)
    residual = np.minimum(residual, step - residual)
    return {
        "min_gap": float(gaps.min()),
        "median_gap": float(step),
        "max_gap": float(gaps.max()),
        "residual_to_grid": float(np.mean(residual / step)),
        "looks_quantized": bool(np.mean(residual / step) < 0.05),
    }

def modality_probe(z: np.ndarray, k_max: int = 12) -> dict:
    """BIC over GMMs — modes hint at discrete/categorical features in the source."""
    z = np.asarray(z, dtype=np.float64).reshape(-1, 1)
    n_unique = int(np.unique(z).size)
    # Cap k by the number of distinct values to avoid degenerate clusters.
    k_max = max(1, min(k_max, n_unique - 1))
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for k in range(1, k_max + 1):
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200, n_init=2)
                gmm.fit(z)
                results.append((k, float(gmm.bic(z))))
            except Exception:
                continue
    if not results:
        return {"best_k": 1, "bic_curve": []}
    best_k = min(results, key=lambda r: r[1])[0]
    return {"best_k": int(best_k), "bic_curve": results}

def float_bit_probe(z: np.ndarray) -> dict:
    """
    If multiple features were packed into one float, the mantissa will look like
    a mixture of unrelated distributions across bit positions. Per-bit entropy
    near 1 across all 52 mantissa bits suggests dense packing.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    bits = z.view(np.uint64)
    mantissa = bits & ((1 << 52) - 1)
    sign = (bits >> 63).astype(np.uint64)
    exponent = ((bits >> 52) & 0x7FF).astype(np.uint64)

    per_bit_entropy = []
    for i in range(52):
        b = ((mantissa >> i) & 1).astype(np.float64)
        p = b.mean()
        if p <= 0 or p >= 1:
            per_bit_entropy.append(0.0)
        else:
            per_bit_entropy.append(float(-p * np.log2(p) - (1 - p) * np.log2(1 - p)))

    return {
        "mantissa_entropy_mean": float(np.mean(per_bit_entropy)),
        "mantissa_entropy_min": float(np.min(per_bit_entropy)),
        "exponent_unique": int(np.unique(exponent).size),
        "sign_balance": float(sign.mean()),
        "per_bit_entropy": per_bit_entropy,
    }

def dimensionality_vote(z: np.ndarray) -> int:
    """
    Estimate D from a single scalar per sample. This is fundamentally
    underdetermined, so we anchor on a domain prior (DOMAIN_D_PRIOR) and
    only adjust when the latent shows concrete structural evidence.

    Heuristic adjustments:
      +4 when mantissa entropy is near-maximal AND the values lie on a
         quantized grid — that combination is consistent with mixed-radix
         packing of many small features.
      -4 when fewer than 2^6 distinct values exist (so the source likely
         has few discrete features).
      modality clamp: if BIC strongly prefers K ≤ 4, lower-bound D at K.
    """
    n_unique = int(np.unique(z).size)
    if n_unique < 4:
        return 4

    D = DOMAIN_D_PRIOR

    bits = float_bit_probe(z)
    quant = quantization_probe(z)
    if bits["mantissa_entropy_mean"] > 0.99 and quant.get("looks_quantized", False):
        D += 4  # likely dense packing

    if n_unique < 64:
        D -= 4

    mod_k = modality_probe(z)["best_k"]
    if mod_k >= 4:
        D = max(D, mod_k)

    return int(np.clip(D, 4, 32))

def run_full_analysis(z: np.ndarray) -> dict:
    return {
        "stats": basic_stats(z),
        "quantization": quantization_probe(z),
        "modality": modality_probe(z),
        "bits": float_bit_probe(z),
        "D_estimate": dimensionality_vote(z),
    }
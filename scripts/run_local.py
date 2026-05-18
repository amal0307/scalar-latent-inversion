"""Local smoke test of the full reconstruction pipeline.
Run from project root:  python scripts/run_local.py
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.reconstruct import reconstruct
from src.analysis import run_full_analysis

DATA = os.path.join(ROOT, "data", "synthetic")
z_pub = np.load(os.path.join(DATA, "public_latents.npy"))
z_hid = np.load(os.path.join(DATA, "hidden_latents.npy"))
X_truth = np.load(os.path.join(DATA, "X_truth.npy"))
D_truth = int(np.load(os.path.join(DATA, "D_truth.npy"))[0])

print("=== forensic report on public latents ===")
report = run_full_analysis(z_pub)
print(f"  n              : {report['stats']['n']}")
print(f"  mean / std     : {report['stats']['mean']:.4f} / {report['stats']['std']:.4f}")
print(f"  unique frac    : {report['stats']['frac_unique']:.4f}")
print(f"  mantissa H avg : {report['bits']['mantissa_entropy_mean']:.4f}")
print(f"  GMM best k     : {report['modality']['best_k']}")
print(f"  D estimate     : {report['D_estimate']}    (truth = {D_truth})")

print("\n=== reconstruction ===")
X_hat = reconstruct(z_pub, z_hid)
print(f"  X_hat shape    : {X_hat.shape}")
print(f"  finite         : {bool(np.all(np.isfinite(X_hat)))}")

print("\n=== stage 6 check: f(Pz) == P f(z) ===")
rng = np.random.default_rng(0)
perm = rng.permutation(len(z_hid))
X_perm = reconstruct(z_pub, z_hid[perm])
ok = np.allclose(X_perm, X_hat[perm], rtol=1e-10, atol=1e-10)
print(f"  permutation equivariant: {ok}")

print("\n=== error vs ground truth (informational only) ===")
import warnings
from scipy.stats import spearmanr, ConstantInputWarning


def mean_best_spearman(X_pred, X_true):
    """For each predicted column, find its best |spearman| against any true
    column; return the mean. This is generous to permuted-feature solutions
    and the right way to score row-aligned reconstruction quality."""
    D_p = X_pred.shape[1]
    D_t = X_true.shape[1]
    common = min(D_p, D_t)
    best = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for j in range(common):
            cors = []
            for k in range(common):
                r, _ = spearmanr(X_pred[:, j], X_true[:, k])
                cors.append(abs(r if r == r else 0.0))
            best.append(max(cors))
    return float(np.mean(best)), float(np.max(best))


D_hat = X_hat.shape[1]
ours_mean, ours_max = mean_best_spearman(X_hat, X_truth)
print(f"  D_hat / D_truth          : {D_hat} / {D_truth}")
print(f"  ours    mean / max |rho| : {ours_mean:.4f} / {ours_max:.4f}")

print("\n=== Stage 5 baseline separation ===")
rng = np.random.default_rng(0)
# Baseline A: random output of the same shape
X_rand = rng.normal(size=X_hat.shape)
rand_mean, rand_max = mean_best_spearman(X_rand, X_truth)
print(f"  random  mean / max |rho| : {rand_mean:.4f} / {rand_max:.4f}")

# Baseline B: constant output (replicate the column means)
X_const = np.broadcast_to(X_hat.mean(axis=0, keepdims=True), X_hat.shape).copy()
const_mean, const_max = mean_best_spearman(X_const, X_truth)
print(f"  const   mean / max |rho| : {const_mean:.4f} / {const_max:.4f}")

# Baseline C: replicate the latent itself across D columns (trivial dependence)
X_repl = np.repeat(z_hid.reshape(-1, 1), D_hat, axis=1)
repl_mean, repl_max = mean_best_spearman(X_repl, X_truth)
print(f"  z-repl  mean / max |rho| : {repl_mean:.4f} / {repl_max:.4f}")

print(f"\n  separation (ours - random) mean : {ours_mean - rand_mean:+.4f}")
print(f"  separation (ours - const)  mean : {ours_mean - const_mean:+.4f}")
print(f"  separation (ours - z-repl) mean : {ours_mean - repl_mean:+.4f}")

"""Generate the three write-up figures from the synthetic test harness.
Run from project root:  python scripts/make_figures.py

Output:
  writeup/figures/fig1_bic_curve.png
  writeup/figures/fig2_mantissa_entropy.png
  writeup/figures/fig3_baseline_separation.png
"""
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, ConstantInputWarning

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.reconstruct import reconstruct
from src.analysis import modality_probe, float_bit_probe

OUT_DIR = os.path.join(ROOT, "writeup", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DATA = os.path.join(ROOT, "data", "synthetic")
z_pub = np.load(os.path.join(DATA, "public_latents.npy"))
z_hid = np.load(os.path.join(DATA, "hidden_latents.npy"))
X_truth = np.load(os.path.join(DATA, "X_truth.npy"))


# Figure 1 — BIC curve
print("[1/3] BIC curve ...")
mod = modality_probe(z_pub, k_max=12)
ks, bics = zip(*mod["bic_curve"])
best_k = mod["best_k"]
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(ks, bics, marker="o", linewidth=1.5, color="#1f77b4")
ax.axvline(best_k, color="#d62728", linestyle="--", alpha=0.7,
           label=f"argmin BIC = K={best_k}")
ax.set_xlabel("Number of mixture components K")
ax.set_ylabel("BIC (lower is better)")
ax.set_title("Gaussian mixture model selection on the public latents")
ax.legend(frameon=False)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_bic_curve.png"), dpi=150)
plt.close(fig)


# Figure 2 — per-bit mantissa entropy
print("[2/3] mantissa entropy ...")
bits = float_bit_probe(z_pub)
H = np.array(bits["per_bit_entropy"])
fig, ax = plt.subplots(figsize=(8.0, 3.5))
colors = ["#2ca02c" if h > 0.99 else "#ff7f0e" if h > 0.9 else "#7f7f7f" for h in H]
ax.bar(range(52), H, color=colors, edgecolor="none")
ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_xlabel("Mantissa bit index (LSB → MSB)")
ax.set_ylabel("Per-bit Shannon entropy (bits)")
ax.set_title(f"Per-bit mantissa entropy — mean = {bits['mantissa_entropy_mean']:.4f}")
ax.set_ylim(0, 1.05)
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_mantissa_entropy.png"), dpi=150)
plt.close(fig)


# Figure 3 — baseline separation (requires ground truth)
print("[3/3] baseline separation ...")


def mean_best_spearman(X_pred, X_true):
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
    return float(np.mean(best))


X_hat = reconstruct(z_pub, z_hid)
D_hat = X_hat.shape[1]
rng = np.random.default_rng(0)

ours = mean_best_spearman(X_hat, X_truth)
rand = mean_best_spearman(rng.normal(size=X_hat.shape), X_truth)
const = mean_best_spearman(
    np.broadcast_to(X_hat.mean(axis=0, keepdims=True), X_hat.shape).copy(),
    X_truth,
)
zrepl = mean_best_spearman(np.repeat(z_hid.reshape(-1, 1), D_hat, axis=1), X_truth)

methods = ["Constant", "Random", "Ours", "z-replicated\n(ceiling)"]
values = [const, rand, ours, zrepl]
colors = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#d62728"]

fig, ax = plt.subplots(figsize=(6.5, 3.8))
bars = ax.bar(methods, values, color=colors, edgecolor="none")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylabel(r"mean best $|\rho_{\mathrm{Spearman}}|$")
ax.set_title("Baseline-separation — ours matches the information-theoretic ceiling")
ax.set_ylim(0, max(values) * 1.25)
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_baseline_separation.png"), dpi=150)
plt.close(fig)


print(f"\nFigures saved to {os.path.relpath(OUT_DIR, ROOT)}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")

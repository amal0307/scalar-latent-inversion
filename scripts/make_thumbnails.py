"""Generate Kaggle-card-shaped and square-thumbnail images from the same
baseline-separation data. Run from project root:
    python scripts/make_thumbnails.py
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
from src.reconstruct import reconstruct  # noqa: E402

OUT = os.path.join(ROOT, "writeup", "figures")
os.makedirs(OUT, exist_ok=True)

DATA = os.path.join(ROOT, "data", "synthetic")
z_pub = np.load(os.path.join(DATA, "public_latents.npy"))
z_hid = np.load(os.path.join(DATA, "hidden_latents.npy"))
X_truth = np.load(os.path.join(DATA, "X_truth.npy"))


def mean_best_spearman(Xp, Xt):
    D = min(Xp.shape[1], Xt.shape[1])
    best = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for j in range(D):
            cors = []
            for k in range(D):
                r, _ = spearmanr(Xp[:, j], Xt[:, k])
                cors.append(abs(r if r == r else 0.0))
            best.append(max(cors))
    return float(np.mean(best))


X_hat = reconstruct(z_pub, z_hid)
D_hat = X_hat.shape[1]
rng = np.random.default_rng(0)
ours = mean_best_spearman(X_hat, X_truth)
rand = mean_best_spearman(rng.normal(size=X_hat.shape), X_truth)
const = mean_best_spearman(
    np.broadcast_to(X_hat.mean(axis=0, keepdims=True), X_hat.shape).copy(), X_truth)
zrepl = mean_best_spearman(np.repeat(z_hid.reshape(-1, 1), D_hat, axis=1), X_truth)

methods = ["z-replicated (ceiling)", "Ours", "Random", "Constant"]
values = [zrepl, ours, rand, const]
colors = ["#d62728", "#1f77b4", "#ff7f0e", "#7f7f7f"]


def _draw(ax, title_fs, label_fs, value_fs):
    bars = ax.barh(methods, values, color=colors, edgecolor="none")
    for bar, v in zip(bars, values):
        ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", ha="left", fontsize=value_fs, fontweight="bold")
    ax.set_xlabel(r"mean best $|\rho_{\mathrm{Spearman}}|$", fontsize=label_fs)
    ax.set_xlim(0, max(values) * 1.30)
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(axis='both', labelsize=label_fs)
    ax.set_title("Our reconstruction matches\nthe information-theoretic ceiling",
                 fontsize=title_fs, fontweight="bold", pad=12)
    ax.invert_yaxis()


# Card image — 16:9 landscape (Kaggle social card spec ≈ 1200x630)
fig, ax = plt.subplots(figsize=(12, 6.3))
_draw(ax, title_fs=20, label_fs=14, value_fs=14)
fig.tight_layout()
card_path = os.path.join(OUT, "card_image.png")
fig.savefig(card_path, dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Thumbnail — square 1:1 (Kaggle thumbnail ≈ 800x800)
fig, ax = plt.subplots(figsize=(8, 8))
_draw(ax, title_fs=18, label_fs=15, value_fs=16)
fig.tight_layout()
thumb_path = os.path.join(OUT, "thumbnail.png")
fig.savefig(thumb_path, dpi=100, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Card image:  {card_path}")
print(f"Thumbnail:   {thumb_path}")

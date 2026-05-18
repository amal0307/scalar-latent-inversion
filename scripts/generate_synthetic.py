"""Generate a fake (X, z) pair so we can smoke-test the pipeline locally.
Run from project root:  python scripts/generate_synthetic.py
"""
import os
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 1337
rng = np.random.default_rng(SEED)
N = 4096
D = 16  # ground-truth dimensionality (hidden from the solver)

age            = rng.normal(42, 12, N).clip(18, 95)
income         = rng.lognormal(np.log(45000), 0.6, N).clip(0, 500000)
balance        = rng.normal(2500, 4500, N).clip(-5000, 80000)
loan_amount    = rng.lognormal(np.log(10000), 0.8, N).clip(0, 300000)
loan_term      = rng.choice([12, 24, 36, 48, 60, 84, 120, 240, 360], N).astype(float)
interest_rate  = rng.normal(7.5, 4.0, N).clip(0, 35)
credit_score   = rng.normal(680, 90, N).clip(300, 850)
debt_ratio     = rng.beta(2, 4, N) * 2
n_open         = rng.poisson(8, N).clip(0, 40).astype(float)
delinquencies  = rng.poisson(0.3, N).clip(0, 20).astype(float)
inquiries      = rng.poisson(1.0, N).clip(0, 20).astype(float)
employment_yrs = rng.normal(7.5, 7.0, N).clip(0, 50)
home_owner     = rng.binomial(1, 0.5, N).astype(float)
has_default    = rng.binomial(1, 0.1, N).astype(float)
region_code    = rng.integers(0, 20, N).astype(float)
product_type   = rng.integers(0, 8, N).astype(float)

X = np.stack([age, income, balance, loan_amount, loan_term, interest_rate,
              credit_score, debt_ratio, n_open, delinquencies, inquiries,
              employment_yrs, home_owner, has_default, region_code, product_type], axis=1)
assert X.shape == (N, D)

mu = X.mean(0); sd = X.std(0) + 1e-12
Xz = (X - mu) / sd

w = rng.normal(0, 1, D)
w /= np.linalg.norm(w)
z_linear = Xz @ w

noise = rng.normal(0, 0.05, N)
z = z_linear + noise

N_pub = 3584
N_hid = N - N_pub
perm = rng.permutation(N)
z_pub = z[perm[:N_pub]]
z_hid = z[perm[N_pub:]]
X_hid = X[perm[N_pub:]]

np.save(os.path.join(OUT_DIR, "public_latents.npy"), z_pub)
np.save(os.path.join(OUT_DIR, "hidden_latents.npy"), z_hid)
np.save(os.path.join(OUT_DIR, "X_truth.npy"), X_hid)
np.save(os.path.join(OUT_DIR, "D_truth.npy"), np.array([D]))

print(f"Wrote synthetic data to {os.path.abspath(OUT_DIR)}")
print(f"  public_latents: {z_pub.shape}")
print(f"  hidden_latents: {z_hid.shape}")
print(f"  X_truth (eval): {X_hid.shape}")

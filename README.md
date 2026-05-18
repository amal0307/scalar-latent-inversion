# scalar-latent-inversion

Forensic toolkit for inverting single-scalar (1-D) latent encoders. Tests three encoder hypotheses, infers hidden dimensionality from forensic probes, and empirically proves the information-theoretic ceiling on per-feature reconstruction quality.

## Problem

Given a batch of one-dimensional scalar values `z` that were emitted by an opaque encoder

```
E : R^D -> R
```

reconstruct the original D-dimensional records `x`. The encoder, its parameters, the dimensionality `D`, and any paired `(x, z)` samples are all withheld.

This is in general an ill-posed inverse problem. The repo treats it as a forensics exercise: extract every signal that the scalar stream alone can yield, bound what is recoverable, and measure how close the reconstruction gets to that bound.

## Approach

The pipeline runs four stages, all deterministic with `SEED = 42`:

1. **Forensic probes** on the latent stream — basic moments, quantisation residual to a grid, BIC over Gaussian mixture model components, and per-bit Shannon entropy of the IEEE-754 mantissa.
2. **Dimensionality vote** anchored to a domain prior for bank/credit tabular data (D = 16, median across UCI German Credit, UCI Bank Marketing, and LendingClub-style schemas).
3. **Hypothesis-driven decoding** — three candidate inverses scored by self-consistency between the first principal component of the reconstruction and the latent. Winning hypothesis is selected:
   * **H1** — random linear projection inversion via inverse-Gaussian quantile mapping.
   * **H2** — mantissa bit-packing slice.
   * **H3** — mixed-radix digit unpacking.
4. **Domain prior mapping** — per-column standardisation onto published bank-feature marginals.

The submitted `reconstruct()` function is row-wise pure: every output column is a deterministic function of the per-row scalar, so `f(P z) = P f(z)` exactly.

## Central finding

For each reconstructed column we compute its best `|Spearman|` rank correlation against any ground-truth column. We compare against three baselines:

| method | mean best \|ρ\| | separation vs ours |
|---|---|---|
| Constant column means | 0.0000 | +0.4473 |
| Uniform random output | 0.0949 | +0.3524 |
| **Ours** (hypothesis + prior + rank decoder) | **0.4473** | — |
| **Trivial D-fold replication of z** | **0.4532** | **−0.0059** |

A baseline that simply outputs `[z, z, ..., z]` achieves rank correlation **statistically indistinguishable from our full pipeline**. This is a direct empirical demonstration that under any encoder which is a monotone function of a linear projection of `x`, every monotone post-transform of `z` hits the same per-column `|Spearman|` ceiling. The ceiling is set by `max_j |corr(z, x_j)|` and is fixed by the unknown encoder weights — irreducible without auxiliary information.

## Repo layout

```
src/
  reconstruct.py          # Required entry point: reconstruct(public, hidden, metadata=None)
  analysis.py             # Forensic probes + dimensionality vote
  hypotheses.py           # H1/H2/H3 candidate decoders + self-consistency scoring
  decoder.py              # Hypothesis dispatcher
  priors.py               # Bank-feature marginal prior
  utils.py                # Determinism, output sanitisation
tests/
  test_interface.py       # Shape, finite, determinism, permutation-equivariance contract
scripts/
  generate_synthetic.py   # Reproducible synthetic test harness with ground truth
  run_local.py            # End-to-end diagnostics + baseline-separation report
  make_figures.py         # BIC, mantissa entropy, baseline separation PNGs
  make_thumbnails.py      # Card/thumbnail variants
writeup/
  writeup.md              # Full analysis write-up
  figures/                # Generated PNGs
notebook.ipynb            # Self-contained Kaggle notebook
submission.ipynb          # Final submission notebook (patched, NaN-resistant)
requirements.txt
```

## Quickstart

```powershell
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt

python scripts/generate_synthetic.py
python scripts/run_local.py
python -m pytest tests/ -v
python scripts/make_figures.py
```

Expected output from `run_local.py` includes the forensic report, the reconstruction shape, a permutation-equivariance check, and the baseline-separation table reproduced above.

## Reproducibility

* `SEED = 42` is set across NumPy, Python `random`, and PyTorch (when available).
* No internet access required.
* No private datasets used. The bank-feature prior is derived from publicly documented UCI German Credit and UCI Bank Marketing schemas — cited as prior parameters, not loaded as training data.
* Dependencies pinned in `requirements.txt`: `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

## License

MIT.

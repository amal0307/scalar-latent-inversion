import numpy as np
from .utils import set_determinism, sanitize_output
from .analysis import dimensionality_vote
from .decoder import decode

def reconstruct(public_latents, hidden_latents, metadata=None):
    """
    Required interface.
      public_latents:  shape (N_pub,) or (N_pub, 1)  — visible 1D values
      hidden_latents:  shape (N_hid,) or (N_hid, 1)  — to be reconstructed
      metadata:        optional dict (ignored unless it carries D explicitly)
    Returns:
      X_hat: np.ndarray of shape (N_hid, D_hat), finite, deterministic.
    """
    set_determinism()

    pub = np.asarray(public_latents, dtype=np.float64).ravel()
    hid = np.asarray(hidden_latents, dtype=np.float64).ravel()

    # Use the union of visible latents to estimate D (more data = more stable)
    combined = np.concatenate([pub, hid]) if pub.size > 0 else hid

    # Allow metadata override if the host ever provides D
    if isinstance(metadata, dict) and "D" in metadata:
        D = int(metadata["D"])
    else:
        D = dimensionality_vote(combined)

    X_hat = decode(hid, D=D, use_prior=True)

    # CRITICAL: row-alignment is guaranteed by construction (decode is pointwise).
    # CRITICAL: f(P z) = P f(z) holds because decode is row-wise pure.
    return sanitize_output(X_hat)
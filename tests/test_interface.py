import numpy as np
from src.reconstruct import reconstruct

def test_shape_and_finite():
    rng = np.random.default_rng(0)
    pub = rng.normal(size=4096)
    hid = rng.normal(size=512)
    X = reconstruct(pub, hid)
    assert X.shape[0] == 512
    assert X.ndim == 2 and X.shape[1] >= 4
    assert np.all(np.isfinite(X))

def test_determinism():
    pub = np.linspace(-1, 1, 4096)
    hid = np.linspace(-2, 2, 256)
    a = reconstruct(pub, hid)
    b = reconstruct(pub, hid)
    np.testing.assert_array_equal(a, b)

def test_permutation_equivariance():
    """Stage 6: f(P z) must equal P f(z)."""
    rng = np.random.default_rng(1)
    pub = rng.normal(size=4096)
    hid = rng.normal(size=256)
    perm = rng.permutation(256)
    X = reconstruct(pub, hid)
    Xp = reconstruct(pub, hid[perm])
    np.testing.assert_allclose(Xp, X[perm], rtol=1e-10, atol=1e-10)

def test_input_dependence():
    """Different inputs must produce different outputs (Stage 5/6)."""
    pub = np.zeros(4096)
    a = reconstruct(pub, np.array([0.1, 0.2, 0.3]))
    b = reconstruct(pub, np.array([0.9, -0.5, 1.7]))
    assert not np.allclose(a, b)

if __name__ == "__main__":
    test_shape_and_finite()
    test_determinism()
    test_permutation_equivariance()
    test_input_dependence()
    print("all tests passed")
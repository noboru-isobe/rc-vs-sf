import numpy as np

from rcsf.filters import filter_bank, hankel_matrix


def test_hankel_symmetric_positive():
    Z = hankel_matrix(64)
    assert np.allclose(Z, Z.T)
    assert np.all(Z > 0)
    # PSD: all eigenvalues nonnegative (up to numerical noise)
    eigvals = np.linalg.eigvalsh(Z)
    assert eigvals.min() > -1e-12


def test_hankel_entries():
    Z = hankel_matrix(3)
    # Z[0,0] -> i=j=1: 2/(2^3-2) = 1/3
    assert np.isclose(Z[0, 0], 2.0 / 6.0)
    # i=1, j=2: 2/(3^3-3) = 1/12
    assert np.isclose(Z[0, 1], 2.0 / 24.0)


def test_eigenvalues_descending_and_decay():
    fb = filter_bank(256, 24)
    assert np.all(np.diff(fb.sigma) <= 0)
    assert np.all(fb.sigma > 0)
    # Exponential decay: sigma_20 should be many orders below sigma_1
    assert fb.sigma[20] / fb.sigma[0] < 1e-10


def test_top_h_captures_trace():
    L = 256
    fb = filter_bank(L, 24)
    trace = np.trace(hankel_matrix(L))
    assert fb.sigma.sum() / trace > 0.999999


def test_filters_orthonormal():
    fb = filter_bank(128, 12)
    gram = fb.phi.T @ fb.phi
    assert np.allclose(gram, np.eye(12), atol=1e-10)


def test_features_shape_and_value():
    fb = filter_bank(32, 4)
    rng = np.random.default_rng(0)
    hist = rng.normal(size=(32, 3))
    feats = fb.features(hist)
    assert feats.shape == (4, 3)
    expected = (fb.sigma**0.25)[:, None] * (fb.phi.T @ hist)
    assert np.allclose(feats, expected)

"""Spectral filter bank from the Hankel matrix of Hazan et al. (2017).

Used by Algorithm 1 of arXiv:2508.11990. The filters are the top-h eigenvectors
of Z_L[i, j] = 2 / ((i + j)^3 - (i + j)) (1-based indices), each weighted by
sigma^{1/4} at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class FilterBank:
    """Top-h eigenpairs of the Hankel matrix Z_L.

    phi: (L, h) eigenvectors, column i is the i-th filter (descending eigenvalue).
    sigma: (h,) eigenvalues, descending.
    sigma_quarter: (h,) sigma**0.25, the prediction-time scaling.
    """

    phi: np.ndarray
    sigma: np.ndarray
    sigma_quarter: np.ndarray

    @property
    def length(self) -> int:
        return self.phi.shape[0]

    @property
    def num_filters(self) -> int:
        return self.phi.shape[1]

    def features(self, history: np.ndarray) -> np.ndarray:
        """Scaled filter features of a history window.

        history: (L, d) array whose row s is the observation at lag s+1
        (most recent first). Returns (h, d) = diag(sigma^{1/4}) @ phi.T @ history.
        """
        return self.sigma_quarter[:, None] * (self.phi.T @ history)


def hankel_matrix(L: int) -> np.ndarray:
    """Z_L[i, j] = 2 / ((i + j)^3 - (i + j)) with 1-based i, j."""
    idx = np.arange(1, L + 1)
    s = idx[:, None] + idx[None, :]
    return 2.0 / (s**3 - s)


@lru_cache(maxsize=8)
def filter_bank(L: int, h: int) -> FilterBank:
    """Top-h eigenpairs of Z_L, cached by (L, h)."""
    if h > L:
        raise ValueError(f"h={h} exceeds filter length L={L}")
    Z = hankel_matrix(L)
    sigma, phi = eigh(Z, subset_by_index=(L - h, L - 1))
    order = np.argsort(sigma)[::-1]
    sigma = sigma[order]
    phi = phi[:, order]
    # Fix sign convention so results don't depend on LAPACK details.
    signs = np.sign(phi[np.argmax(np.abs(phi), axis=0), np.arange(h)])
    phi = phi * signs
    return FilterBank(phi=phi, sigma=sigma, sigma_quarter=sigma**0.25)

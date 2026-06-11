"""SFeDMD (paper Section 7.2.1): spectral filtering on the eDMD-lifted space.

Lift z_t = psi(y_t) with the same thin-plate RBF dictionary as eDMD (fitted on
causally standardized data), then predict with the filter-feature form of
Algorithm 1 (k filters, m = 0):

  ŷ_t = Σ_{i=1}^{k} σ_i^{1/4} N_i <φ_i, z_{t-1:t-L-1}>,  N_i ∈ R^{d_y × n}

where N is fit by ridge regression on all causal pairs (features_s, y_s). The
dictionary, standardization statistics and N are all refit every T/10 steps on
the history so far; between refits predictions use the last fitted model
(persistence before the first fit).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ..filters import filter_bank
from .base import OnlinePredictor
from .edmd import make_observables

MAX_FILTER_LEN = 1024


class SFeDMD(OnlinePredictor):
    name = "sfedmd"

    def __init__(self, h: int = 24, n_centers: int = 20, ridge: float = 1e-6,
                 refit_every: int | None = None, fit_washout: int = 200):
        self.h = h
        self.n_centers = n_centers
        self.ridge = ridge
        self.refit_every = refit_every
        # Rows excluded from the ridge fit: initial transients (e.g. Langevin
        # starts at x=0, several sigma from the stationary wells) otherwise
        # dominate the regression through their leverage.
        self.fit_washout = fit_washout

    def reset(self, d_y, d_u, T, seed=0):
        self.d_y = d_y
        self.L = min(T - 1, MAX_FILTER_LEN)
        self.bank = filter_bank(self.L, self.h)
        self.refit = self.refit_every or max(2, T // 10)
        self.history: list[np.ndarray] = []
        self.obs = None       # fitted dictionary (on standardized data)
        self.mu = None
        self.sd = None
        self.N = None         # (d_y, h*n) readout
        self.z_hist = None    # (L, n) lifted history, row 0 = z_{t-1}
        self.seed = seed
        self._max_norm = 0.0
        self._radius = np.inf

    def _lift(self, Y: np.ndarray) -> np.ndarray:
        return np.asarray(self.obs.transform((Y - self.mu) / self.sd))

    def predict(self, u_t):
        if self.N is None or self.z_hist is None:
            return self.history[-1].copy() if self.history else np.zeros(self.d_y)
        feats = self.bank.features(self.z_hist).ravel()
        y_hat = self.N @ feats
        # Projection step (Algorithm 1): clip to a robust radius (2x the 95th
        # percentile of observed norms, set at refit time) — using the max
        # would let an initial transient inflate the radius by an order of
        # magnitude and defeat the projection.
        norm = np.linalg.norm(y_hat)
        if norm > self._radius:
            y_hat *= self._radius / norm
        return y_hat

    def _causal_features(self, Z: np.ndarray) -> np.ndarray:
        """Features for predicting y_s from z_{s-1}, ..., for all s in [0, len(Z)).

        Returns (T, h*n): row s = vec(diag(sigma^{1/4}) Phi^T [z_{s-1}; ...; z_{s-L}]).
        One FFT convolution per filter.
        """
        T, n = Z.shape
        out = np.empty((T, self.h, n))
        for i in range(self.h):
            kernel = self.bank.phi[:, i][:, None]
            conv = fftconvolve(Z, kernel, axes=0)[:T]   # conv[s] = sum_k phi_i[k] z_{s-k}
            out[:, i, :] = self.bank.sigma_quarter[i] * conv
        # Shift by one: features at step s must end at z_{s-1}.
        out = np.roll(out, 1, axis=0)
        out[0] = 0.0
        return out.reshape(T, self.h * n)

    def update(self, y_t):
        y_t = np.asarray(y_t, dtype=float)
        self.history.append(y_t)
        self._max_norm = max(self._max_norm, float(np.linalg.norm(y_t)))
        t = len(self.history)
        if self.obs is not None:
            z_t = self._lift(y_t[None, :])[0]
            self.z_hist = np.roll(self.z_hist, 1, axis=0)
            self.z_hist[0] = z_t
        if t % self.refit == 0 and t > self.n_centers + 2:
            Y = np.asarray(self.history)
            self.mu, self.sd = Y.mean(0), Y.std(0)
            self.sd[self.sd < 1e-12] = 1.0
            np.random.seed(self.seed)  # center fitting uses global state
            self.obs = make_observables(self.n_centers)
            self.obs.fit((Y - self.mu) / self.sd)
            Z = self._lift(Y)
            # Rebuild the rolling lifted history from the refreshed dictionary.
            self.z_hist = np.zeros((self.L, Z.shape[1]))
            m = min(self.L, len(Z))
            self.z_hist[:m] = Z[::-1][:m]
            F = self._causal_features(Z)
            w = min(self.fit_washout, t // 2)
            F, Yw = F[w:], Y[w:]
            # Standardize feature columns before ridge: the sigma^{1/4} scaling
            # makes column scales differ by orders of magnitude, and a uniform
            # ridge would otherwise suppress the small-sigma filters entirely.
            col_sd = F.std(axis=0)
            col_sd[col_sd < 1e-12] = 1.0
            Fn = F / col_sd
            G = Fn.T @ Fn + self.ridge * len(Fn) * np.eye(Fn.shape[1])
            self.N = (np.linalg.solve(G, Fn.T @ Yw) / col_sd[:, None]).T
            norms = np.linalg.norm(Y[w:], axis=1)
            self._radius = 2.0 * float(np.percentile(norms, 95))

    def num_trainable_params(self):
        if self.N is None:        # not fitted yet: analytic count
            return self.d_y * self.h * (self.d_y + self.n_centers)
        return self.N.size

"""Online-causal eDMD baseline (paper Section 7.2.1).

PyKoopman eDMD with a dictionary of the identity plus 20 thin-plate-spline RBFs
whose centers are fitted to the data. The model is refit every T/10 steps on the
full history so far; predictions always come from the last fitted model
(persistence before the first fit), so the method stays causal.

Two numerically essential choices (validated against persistence on Lorenz):
the explicit least-squares regressor (pykoopman's default PyDMDRegressor
truncates the spectrum and is orders of magnitude worse here), and causal
standardization of the data before lifting (thin-plate r^2 log r blows up on
raw Lorenz scales).
"""

from __future__ import annotations

import numpy as np
import pykoopman as pk
from pykoopman.observables import RadialBasisFunction
from pykoopman.regression import EDMD as EDMDRegressor

from .base import OnlinePredictor


def make_observables(n_centers: int = 20):
    return RadialBasisFunction(
        rbf_type="thinplate", n_centers=n_centers, include_state=True,
    )


class EDMD(OnlinePredictor):
    name = "edmd"

    def __init__(self, n_centers: int = 20, refit_every: int | None = None):
        self.n_centers = n_centers
        self.refit_every = refit_every

    def reset(self, d_y, d_u, T, seed=0):
        self.d_y = d_y
        self.refit = self.refit_every or max(2, T // 10)
        self.history: list[np.ndarray] = []
        self.model = None
        self.mu = np.zeros(d_y)
        self.sd = np.ones(d_y)
        self.seed = seed
        self._n_params = 0

    def predict(self, u_t):
        if self.model is None or not self.history:
            return self.history[-1].copy() if self.history else np.zeros(self.d_y)
        last = (self.history[-1] - self.mu) / self.sd
        pred = np.asarray(self.model.predict(last[None, :]))[0]
        return pred * self.sd + self.mu

    def update(self, y_t):
        self.history.append(np.asarray(y_t, dtype=float))
        t = len(self.history)
        if t % self.refit == 0 and t > self.n_centers + 2:
            data = np.asarray(self.history)
            mu, sd = data.mean(0), data.std(0)
            sd[sd < 1e-12] = 1.0
            np.random.seed(self.seed)  # pykoopman center fitting uses global state
            model = pk.Koopman(observables=make_observables(self.n_centers),
                               regressor=EDMDRegressor())
            model.fit((data - mu) / sd)
            self.model, self.mu, self.sd = model, mu, sd
            n = model.A.shape[0]
            self._n_params = n * n + n * self.d_y

    def num_trainable_params(self):
        if self._n_params == 0:   # not fitted yet: analytic count
            n = self.d_y + self.n_centers
            return n * n + n * self.d_y
        return self._n_params

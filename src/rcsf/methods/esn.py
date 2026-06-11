"""Echo state network with online readout learning (RLS or LMS), via ReservoirPy.

Strict predict-then-update wiring: the reservoir node is stepped manually, the
readout's forward pass produces ŷ_t with pre-update weights (verified by test),
and readout.train(s_t, y_t) runs only after the truth is revealed.

Notes:
- reservoirpy is pinned to 0.3.x because 0.4.x effectively requires
  scipy >= 1.12 (scipy.sparse.random_array) while pykoopman 1.2.1 pins
  scipy <= 1.11.2 (see README).
- Inputs are causally standardized (running Welford stats); raw scales like
  Lorenz (±20) otherwise saturate the tanh reservoir.
- Readout features are [reservoir state, standardized input] (direct input
  connections, standard ESN practice; matters a lot on NARMA10).

Input convention (matches the information available to SF+obs): autonomous
systems get x_t = y_{t-1}; input-driven systems get x_t = [u_t, y_{t-1}].
"""

from __future__ import annotations

import numpy as np

from .base import OnlinePredictor


class ESN(OnlinePredictor):
    def __init__(self, units: int = 300, sr: float = 0.9, lr_leak: float = 1.0,
                 input_scaling: float = 0.1, readout: str = "rls",
                 forgetting: float = 0.999, lms_rate: float = 1e-3,
                 name: str | None = None):
        # forgetting < 1 matters: the causal input standardization drifts early
        # on, and RLS with forgetting=1.0 never forgets that transient.
        self.units = units
        self.sr = sr
        self.lr_leak = lr_leak
        self.input_scaling = input_scaling
        self.readout_kind = readout
        self.forgetting = forgetting
        self.lms_rate = lms_rate
        self.name = name or f"esn-{readout}"

    def reset(self, d_y, d_u, T, seed=0):
        from reservoirpy.nodes import LMS, RLS, Reservoir

        self.d_y, self.d_u = d_y, d_u
        self._last_y = np.zeros(d_y)
        # Causal running standardization of the input (Welford).
        d_in = d_y + d_u
        self._n_obs = 0
        self._mean = np.zeros(d_in)
        self._m2 = np.zeros(d_in)

        self.reservoir = Reservoir(
            units=self.units, sr=self.sr, lr=self.lr_leak,
            input_scaling=self.input_scaling, input_connectivity=1.0, seed=seed,
        )
        if self.readout_kind == "rls":
            self.readout = RLS(output_dim=d_y, forgetting=self.forgetting)
        elif self.readout_kind == "lms":
            self.readout = LMS(output_dim=d_y, alpha=self.lms_rate)
        else:
            raise ValueError(f"unknown readout: {self.readout_kind}")
        self._s_t = None

    def _make_input(self, u_t) -> np.ndarray:
        if self.d_u > 0:
            x = np.concatenate([np.asarray(u_t, dtype=float).ravel(), self._last_y])
        else:
            x = self._last_y.copy()
        self._observe_input(x)
        if self._n_obs < 2:
            return x
        sd = np.sqrt(self._m2 / (self._n_obs - 1))
        sd[sd < 1e-8] = 1.0
        return (x - self._mean) / sd

    def _observe_input(self, x: np.ndarray) -> None:
        self._n_obs += 1
        delta = x - self._mean
        self._mean += delta / self._n_obs
        self._m2 += delta * (x - self._mean)

    def predict(self, u_t):
        x = self._make_input(u_t)
        s = self.reservoir(x[None, :])              # advances reservoir state once
        # Readout features: state + direct input connection (the readout nodes
        # are plain linear maps on whatever features we pass).
        self._s_t = np.hstack([np.asarray(s), x[None, :]])
        y = self.readout(self._s_t)                 # forward with pre-update weights
        return np.asarray(y).ravel()

    def update(self, y_t):
        y_t = np.asarray(y_t, dtype=float).ravel()
        self.readout.train(self._s_t, y_t[None, :])
        self._last_y = y_t

    def num_trainable_params(self):
        return self.d_y * (self.units + self.d_y + self.d_u + 1)

"""Algorithm 1 of arXiv:2508.11990 — Observation Spectral Filtering + Regression.

Prediction (m = AR order, h = number of filters, sigma/phi from the Hankel bank):

  ŷ_t = Σ_{j=1}^{m-1} J_j u_{t-j}
      + Σ_{i=1}^{h} σ_i^{1/4} M_i <φ_i, u_{t-2:t-T}>
      + Σ_{j=1}^{m}  P_j y_{t-j}
      + Σ_{i=1}^{h} σ_i^{1/4} N_i <φ_i, y_{t-1:t-T-1}>

then clipped to a ball of radius R. All blocks are packed into one matrix W so
ŷ_t = W f_t; trained online on the UNSQUARED norm loss ||ŷ_t - y_t|| (paper's
choice) with COCOB or OGD. History convention (see README): the y-window starts
at lag 1, the u-window at lag 2, both of length L = min(T-1, 1024).

Variants:
  use_obs_filters=False, with inputs -> vanilla SF (Hazan et al. 2017 style).
  Autonomous systems (d_u=0) automatically drop the J/M blocks.
"""

from __future__ import annotations

import numpy as np
import torch

from ..filters import filter_bank
from .base import OnlinePredictor
from .optimizers import make_optimizer

MAX_FILTER_LEN = 1024


class SpectralFilter(OnlinePredictor):
    def __init__(self, h: int = 24, m: int = 1, optimizer: str = "cocob",
                 lr: float = 1e-2, use_obs_filters: bool = True,
                 use_input_filters: bool = True, clip_radius: float | None = None,
                 loss: str = "norm", name: str | None = None):
        self.h, self.m = h, m
        self.optimizer_kind = optimizer
        self.lr = lr
        self.use_obs_filters = use_obs_filters
        self.use_input_filters = use_input_filters
        self.clip_radius = clip_radius
        # "norm" is Algorithm 1 as stated; "squared" converges ~2x faster with
        # COCOB and matches the paper's empirical convergence speed (README).
        if loss not in ("norm", "squared"):
            raise ValueError(f"unknown loss: {loss}")
        self.loss_kind = loss
        self.name = name or ("sf+obs" if use_obs_filters else "sf")

    def reset(self, d_y, d_u, T, seed=0):
        torch.manual_seed(seed)
        self.d_y, self.d_u = d_y, d_u
        L = min(T - 1, MAX_FILTER_LEN)
        self.bank = filter_bank(L, self.h)
        self.L = L
        # Ring-free simple buffers: row 0 = most recent.
        self.y_hist = np.zeros((L, d_y))
        # u buffer holds one extra row so the filter window can start at lag 2.
        self.u_hist = np.zeros((L + 1, d_u)) if d_u > 0 else None
        self._pending_u = None

        feat_dim = self.m * d_y + self.h * d_y * self.use_obs_filters
        if d_u > 0:
            feat_dim += (self.m - 1) * d_u + self.h * d_u * self.use_input_filters
        if self.optimizer_kind == "rls":
            # Second-order online update (ONS-equivalent for the squared loss of
            # a linear predictor) — and the same learner as the ESN readout,
            # making the comparison "same learner, different fixed features".
            self.W_np = np.zeros((d_y, feat_dim))
            self.P = np.eye(feat_dim) * 1e6
            self.W = None
        else:
            self.W = torch.zeros(d_y, feat_dim, requires_grad=True)
            self.opt, self.sched = make_optimizer([self.W], self.optimizer_kind, self.lr)
        self._last_features = None
        self._max_norm = 0.0  # running radius estimate for the projection step

    def _features(self) -> np.ndarray:
        parts = [self.y_hist[: self.m].ravel()]                       # P: y_{t-1..t-m}
        if self.use_obs_filters:
            parts.append(self.bank.features(self.y_hist).ravel())     # N: filters on y
        if self.d_u > 0:
            if self.m > 1:
                parts.append(self.u_hist[: self.m - 1].ravel())       # J: u_{t-1..t-m+1}
            if self.use_input_filters:
                # M filters: window starting at lag 1 (u_{t-1}). The paper writes
                # u_{t-2:t-T}, but with m=1 (their experimental setting) the J
                # block is empty and a lag-2 window would drop u_{t-1} entirely,
                # making even a one-step LDS unlearnable. See README.
                parts.append(self.bank.features(self.u_hist[:-1]).ravel())
        return np.concatenate(parts)

    def predict(self, u_t):
        # u_t only matters for FUTURE predictions (in the LDS, y_t depends on
        # u_{t-1} at the latest), so it is buffered here and pushed in update().
        # At this point u_hist row 0 = u_{t-1}, matching the J block (lag 1),
        # and u_hist[1:] = u_{t-2}, ... matching the M filter window (lag 2).
        self._pending_u = u_t
        f = self._features()
        if self.optimizer_kind == "rls":
            self._last_features = f
            y_hat = self.W_np @ f
        else:
            self._last_features = torch.from_numpy(f).float()
            with torch.no_grad():
                y_hat = (self.W @ self._last_features).numpy().astype(float)
        # Projection step of Algorithm 1: clip to a ball of radius R. Default R
        # is the causal estimate 2 * max ||y_s|| observed so far.
        radius = self.clip_radius if self.clip_radius is not None else 2.0 * self._max_norm
        norm = np.linalg.norm(y_hat)
        if radius > 0.0 and norm > radius:
            y_hat *= radius / norm
        return y_hat

    def update(self, y_t):
        if self.optimizer_kind == "rls":
            f = self._last_features
            Pf = self.P @ f
            k = Pf / (1.0 + f @ Pf)
            err = np.asarray(y_t, dtype=float) - self.W_np @ f
            self.W_np += np.outer(err, k)
            self.P -= np.outer(k, Pf)
            self.P = 0.5 * (self.P + self.P.T)
        else:
            target = torch.from_numpy(np.asarray(y_t, dtype=np.float64)).float()
            y_hat = self.W @ self._last_features
            err = y_hat - target
            if self.loss_kind == "norm":
                loss = torch.linalg.vector_norm(err)
            else:
                loss = torch.sum(err**2)
            if loss.item() > 1e-12:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if self.sched is not None:
                    self.sched.step()
        self._max_norm = max(self._max_norm, float(np.linalg.norm(y_t)))
        self.y_hist = np.roll(self.y_hist, 1, axis=0)
        self.y_hist[0] = y_t
        if self.u_hist is not None and self._pending_u is not None:
            self.u_hist = np.roll(self.u_hist, 1, axis=0)
            self.u_hist[0] = self._pending_u
            self._pending_u = None

    def num_trainable_params(self):
        return self.W_np.size if self.optimizer_kind == "rls" else self.W.numel()

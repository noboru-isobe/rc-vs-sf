"""Marginally stable linear systems from arXiv:2508.11990 Section 7.1.

x_{t+1} = A x_t + B u_t + w_t,   y_t = C x_t
with scalar input/output, Gaussian B, C, x0 and i.i.d. standard Gaussian u_t.
"""

from __future__ import annotations

import numpy as np

from .base import Trajectory


def _simulate_lds(A, B, C, x0, u, noise) -> np.ndarray:
    T = len(u)
    d = len(x0)
    x = x0.copy()
    y = np.empty((T, C.shape[0]))
    for t in range(T):
        y[t] = C @ x
        x = A @ x + B @ u[t] + noise(t, d)
    return y


def gaussian_lds(seed: int, T: int = 5000, d: int = 128, noise_amp: float = 0.1,
                 noise_freq: float = 3 * np.pi / 100) -> Trajectory:
    """Random Gaussian A normalized to unit spectral norm, correlated sinusoidal noise.

    Note: the paper writes w_t = 0.1 sin(3πt)·[1..1], which is identically zero
    at integer t (yet their Figure shows a clear disturbance floor for vanilla
    SF). We read t as continuous time k·dt with dt = 0.01, i.e. ω = 3π/100 per
    step (period ≈ 67 steps) — a smooth correlated disturbance that observation
    filters can correct, matching the paper's 20x floor gap. See README.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    A /= np.linalg.norm(A, 2)
    B = rng.normal(size=(d, 1))
    C = rng.normal(size=(1, d))
    x0 = rng.normal(size=d)
    u = rng.normal(size=(T, 1))

    def noise(t, dim):
        return noise_amp * np.sin(noise_freq * t) * np.ones(dim)

    y = _simulate_lds(A, B, C, x0, u, noise)
    return Trajectory(y=y, u=u, name="gaussian_lds")


def permutation_lds(seed: int, T: int = 20000, d: int = 16) -> Trajectory:
    """Cyclic permutation A (marginally stable, highly asymmetric), no noise.

    Autonomous (u = 0): with a marginally stable A, a persistent Gaussian input
    would make the state variance grow linearly in t, which contradicts the flat
    loss plateau of the paper's Figure (Permutation LDS) — the bounded periodic
    rotation of x0 matches it. See README.
    """
    rng = np.random.default_rng(seed)
    A = np.roll(np.eye(d), 1, axis=0)
    C = rng.normal(size=(1, d))
    x0 = rng.normal(size=d)
    x = x0.copy()
    y = np.empty((T, 1))
    for t in range(T):
        y[t] = C @ x
        x = A @ x
    return Trajectory(y=y, u=None, name="permutation_lds")


def diagonal_lds(seed: int, T: int = 1024, d: int = 32) -> Trajectory:
    """Symmetric PSD diagonal A ~ diag(U[0,1]) — sanity system where vanilla
    spectral filtering (Hazan et al. 2017) provably works."""
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0, 1, size=d))
    B = rng.normal(size=(d, 1))
    C = rng.normal(size=(1, d))
    x0 = rng.normal(size=d)
    u = rng.normal(size=(T, 1))

    y = _simulate_lds(A, B, C, x0, u, lambda t, dim: 0.0)
    return Trajectory(y=y, u=u, name="diagonal_lds")

import numpy as np

from rcsf.evaluation import run_online
from rcsf.methods import ESN, Persistence
from rcsf.systems import SYSTEMS


def test_rls_node_converges_to_least_squares():
    # The library RLS node, fed random features, must recover the true linear
    # map (closed-form least-squares solution of a noiseless problem).
    from reservoirpy.nodes import RLS

    rng = np.random.default_rng(0)
    w_true = rng.normal(size=5)
    ro = RLS(output_dim=1)
    for _ in range(300):
        s = rng.normal(size=(1, 5))
        ro.train(s, np.array([[w_true @ s[0]]]))
    assert np.allclose(np.asarray(ro.Wout).ravel(), w_true, atol=1e-6)


def test_forward_uses_pre_update_weights():
    # predict-then-update contract: the forward pass before train() must not
    # see the current target.
    from reservoirpy.nodes import RLS

    rng = np.random.default_rng(1)
    ro = RLS(output_dim=1)
    s = rng.normal(size=(1, 5))
    y = np.array([[3.0]])
    pred_before = np.asarray(ro(s)).ravel()[0]
    ro.train(s, y)
    pred_after = np.asarray(ro(s)).ravel()[0]
    assert pred_before == 0.0           # zero-initialized weights
    assert abs(pred_after - 3.0) < 1e-6  # train() did update toward the target


def test_esn_rls_tracks_mackey_glass():
    # Gate: one-step NRMSE < 1e-2 after convergence (RC literature level).
    traj = SYSTEMS["mackey_glass"](seed=0, T=6000)
    res = run_online(ESN(units=300, sr=1.1, lr_leak=0.3, input_scaling=0.1),
                     traj, seed=0)
    tail = res.errors[-2000:]
    nrmse = np.sqrt(tail.mean()) / traj.y[-2000:].std()
    assert nrmse < 1e-2, f"NRMSE={nrmse:.4f}"


def test_esn_rls_narma10_range():
    # Gate: NARMA10 online one-step NRMSE in the literature range (~0.15-0.25);
    # accept < 0.35 to allow for the strictly-online (no batch ridge) setting.
    traj = SYSTEMS["narma10"](seed=0, T=4000)
    res = run_online(ESN(units=300, sr=0.9, lr_leak=1.0, input_scaling=0.01),
                     traj, seed=0)
    tail = res.errors[-1000:]
    nrmse = np.sqrt(tail.mean()) / traj.y[-1000:].std()
    assert nrmse < 0.35, f"NRMSE={nrmse:.4f}"


def test_lms_slower_than_rls():
    # Textbook behavior (and the reason FORCE uses RLS): early-phase error of
    # LMS stays well above RLS on the same stream.
    traj = SYSTEMS["mackey_glass"](seed=0, T=3000)
    rls = run_online(ESN(units=200, readout="rls"), traj, seed=0)
    lms = run_online(ESN(units=200, readout="lms", lms_rate=1e-3), traj, seed=0)
    early = slice(100, 600)
    assert rls.errors[early].mean() < lms.errors[early].mean()


def test_esn_beats_persistence_on_lorenz():
    traj = SYSTEMS["lorenz"](seed=0, T=1024)
    esn = run_online(ESN(units=300), traj, seed=0)
    pers = run_online(Persistence(), traj, seed=0)
    assert esn.errors[-200:].mean() < pers.errors[-200:].mean()

"""Forecasting a VAR under the matrix-normal DLM, by simulation.

Why not iterated expectations
-----------------------------
**The Kronecker structure does not survive multi-step forecasting.** For a
VAR(1) with ``Theta`` known,

    Var(y_{t+2} | y_t, Theta) = Sigma + B Sigma B'      B = Theta_lag'

which is not a scalar multiple of ``Sigma``. So the predictive is ``Q_1 Sigma``
at one step and is **not** ``(x) Sigma`` from two steps on.

``uv_dlm``'s ``iterated_obs_forecast`` returns a SCALAR ``q_h`` per series, and
legitimately can: its series are independent and each carries its own lags. A
VAR's whole point is that they are coupled, so the analogue would have to carry
a full ``q x q`` predictive, and the coefficient-uncertainty term that
``_regressor_var_prop`` handles as ``(phi^2 + Var phi) * v`` would become a
matrix quadratic. Simulation carries all of that for free, by drawing
``Theta`` rather than propagating its moments.

``test_h2_is_not_proportional_to_Sigma`` measures how badly a scalar-``q_h``
implementation would be wrong, and the answer is: not subtly.

The gates
---------
1. **h=1 against closed form.** One step is the only place the two routes
   overlap: marginalising the draw over ``Theta_{t+1} ~ N(a, R (x) Sigma)``
   gives exactly ``N(F'a, (1 + F'RF) Sigma)``. So the simulation must converge
   to the analytic answer there, and does.

2. **h=2 with Theta FIXED.** Setting ``C_T = 0`` makes ``W = 0`` and the draw
   deterministic in ``Theta``, so the two-step answer is exactly
   ``Sigma + B Sigma B'`` with no approximation. This is the gate that catches
   a scalar-``q_h`` regression.

3. **Feedback alignment.** ``y_{t+k}`` must enter ``F_{t+k+1}``. Without it the
   simulation silently becomes a random walk about the last observed lags.

Tolerances here are Monte Carlo, not numerical: a sample covariance from N
draws converges as 1/sqrt(N), so these compare to a few percent with a fixed
seed rather than to machine precision. That is the honest bar for a simulation
and it is stated so nobody later tightens it and wonders why it flakes.
"""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random

from DLMAX.mvdlm import mv_dlm, congruence, _build_covariance_path

Q, T, N_LAGS = 3, 30, 1
P_STRUCT = 1
P = P_STRUCT + N_LAGS * Q
SIGMA = np.array([[1.0, 0.4, 0.2], [0.4, 1.5, 0.3], [0.2, 0.3, 0.8]])


@pytest.fixture(scope="module")
def var_model():
    """A VAR(1) with an intercept: design ``[1 ; y_{t-1}]``, common to every
    equation -- which is what makes the collapse valid."""
    rng = np.random.default_rng(1)
    Y = np.cumsum(rng.normal(0, 1, (T, Q)), axis=0) + 20.0
    X = np.vstack([np.zeros((1, Q)), Y[:-1]])          # the lag-1 design
    F = jnp.concatenate([jnp.ones((1, 1)), jnp.zeros((Q, 1))])
    G = jnp.eye(P)
    C0 = jnp.eye(P) * 0.1
    delta = congruence(jnp.concatenate([jnp.array([0.99]), jnp.full(Q, 0.995)]))
    m = mv_dlm(data=Y, F=F, G=G, m0=jnp.zeros((P, Q)), C0=C0,
               V0=jnp.ones((1, Q)), disc_mtx=delta)
    m.cov = _build_covariance_path(T=T, F=F, G=G, C0=C0, delta_M=delta, X=X)
    m.scan_filter()
    return m, np.asarray(Y), np.asarray(delta), np.asarray(G)


@pytest.fixture(scope="module")
def chol_sigma():
    return jnp.asarray(np.linalg.cholesky(SIGMA))


def test_shape_and_determinism(var_model, chol_sigma):
    m, _Y, _d, _G = var_model
    kw = dict(h=4, sigma_tril=chol_sigma, n_lags=N_LAGS, n_draws=7)
    a = np.asarray(m.forecast_sample(key=random.PRNGKey(0), **kw))
    b = np.asarray(m.forecast_sample(key=random.PRNGKey(0), **kw))
    c = np.asarray(m.forecast_sample(key=random.PRNGKey(1), **kw))
    assert a.shape == (7, 4, Q)
    np.testing.assert_array_equal(a, b)          # same key, same draws
    assert not np.allclose(a, c)                 # different key, different draws
    assert np.all(np.isfinite(a))


def test_h1_converges_to_the_closed_form_predictive(var_model, chol_sigma):
    """The one place simulation and closed form overlap, so the only place the
    simulation can be checked against an exact answer with Theta random."""
    m, Y, delta, G = var_model
    N = 60_000
    d = np.asarray(m.forecast_sample(h=1, sigma_tril=chol_sigma,
                                     key=random.PRNGKey(0), n_lags=N_LAGS,
                                     n_draws=N))
    C_T = np.asarray(m.cov.C[-1])
    R1 = (G @ C_T @ G.T) / delta
    F1 = np.concatenate([np.asarray(m.F).reshape(P)[:P_STRUCT], Y[-1]])
    Q1 = 1.0 + F1 @ R1 @ F1
    want = Q1 * SIGMA
    got = np.cov(d[:, 0, :].T)
    assert np.abs(got - want).max() / np.abs(want).max() < 0.05


def test_h2_with_fixed_theta_matches_Sigma_plus_B_Sigma_Bt(var_model,
                                                           chol_sigma):
    """THE gate. With ``C_T = 0`` the draw is deterministic in Theta, so the
    two-step covariance is exactly ``Sigma + B Sigma B'`` -- no approximation,
    nothing to hide behind."""
    m, _Y, _d, _G = var_model
    m_T = m._filtered[1][-1]
    state = (m_T, jnp.zeros((P, P)))
    N = 80_000
    d = np.asarray(m.forecast_sample(h=2, sigma_tril=chol_sigma,
                                     key=random.PRNGKey(7), n_lags=N_LAGS,
                                     n_draws=N, state=state))
    B = np.asarray(m_T)[P_STRUCT:, :].T
    for step, want in ((0, SIGMA), (1, SIGMA + B @ SIGMA @ B.T)):
        got = np.cov(d[:, step, :].T)
        rel = np.abs(got - want).max() / np.abs(want).max()
        assert rel < 0.05, f"h={step+1}: {rel:.2%}"


def test_h2_is_not_proportional_to_Sigma(var_model, chol_sigma):
    """The property that rules out the cheap implementation.

    A scalar-``q_h`` model returns ``q_h Sigma``, which would run cleanly and
    look reasonable. Here the entrywise ratio of the true two-step covariance
    to Sigma spans a factor of several, so the cheap form is not a mild
    approximation -- it is a different answer.
    """
    m, _Y, _d, _G = var_model
    m_T = m._filtered[1][-1]
    d = np.asarray(m.forecast_sample(h=2, sigma_tril=chol_sigma,
                                     key=random.PRNGKey(7), n_lags=N_LAGS,
                                     n_draws=80_000,
                                     state=(m_T, jnp.zeros((P, P)))))
    got2 = np.cov(d[:, 1, :].T)
    ratio = got2 / SIGMA
    assert ratio.max() / ratio.min() > 1.5, (
        "the two-step covariance is proportional to Sigma, which would mean "
        "the Kronecker structure survived -- it must not")
    # and quantify what the cheap form would cost
    scalar_qh = (np.trace(got2) / np.trace(SIGMA)) * SIGMA
    assert np.abs(got2 - scalar_qh).max() / np.abs(got2).max() > 0.1


def test_the_simulation_feeds_its_own_draws_forward(var_model, chol_sigma):
    """``y_{t+k}`` must enter ``F_{t+k+1}``. Without the feedback the process
    stays anchored to the last observed lags and the predictive variance stops
    growing -- finite, smooth, and wrong."""
    m, _Y, _d, _G = var_model
    d = np.asarray(m.forecast_sample(h=5, sigma_tril=chol_sigma,
                                     key=random.PRNGKey(3), n_lags=N_LAGS,
                                     n_draws=20_000))
    v = [np.cov(d[:, k, :].T).trace() for k in range(5)]
    assert v[-1] > v[0] * 1.5, (
        f"predictive variance barely grew over the horizon ({v}); the draws "
        "are probably not being fed back into the design")


def test_seed_lags_are_most_recent_first(var_model, chol_sigma):
    """Row 0 is ``y_T``, matching DLMAX's format_seed_lag_yts convention. A
    reversed seed is a plausible-looking wrong answer, so the default is
    compared against the explicit form."""
    m, Y, _d, _G = var_model
    kw = dict(h=2, sigma_tril=chol_sigma, key=random.PRNGKey(5),
              n_lags=N_LAGS, n_draws=64)
    default = np.asarray(m.forecast_sample(**kw))
    explicit = np.asarray(m.forecast_sample(seed_lags=Y[-N_LAGS:][::-1], **kw))
    np.testing.assert_allclose(default, explicit, rtol=0, atol=0)


def test_a_tail_too_wide_for_the_state_is_refused(var_model, chol_sigma):
    """A VAR's design carries EVERY series' lags, so the tail is n_lags * q.
    Asking for more lags than the state can hold is a modelling error, not a
    shape to be broadcast around."""
    m, _Y, _d, _G = var_model
    with pytest.raises(ValueError, match="n_lags"):
        m.forecast_sample(h=2, sigma_tril=chol_sigma, key=random.PRNGKey(0),
                          n_lags=5, n_draws=2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly", "--tb=short"]))

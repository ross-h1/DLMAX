"""RTS smoothing and FFBS: correctness against an independent reference.

The square-root implementation in DLMAX.smoother is checked against a plain
covariance-form RTS written here from the textbook recursions — deliberately
naive (explicit C, explicit inverse) so it shares no code with the thing under
test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.scipy.linalg import block_diag

jax.config.update("jax_enable_x64", True)

from DLMAX.dlm_core import uv_dlm
from DLMAX.smoother import ffbs, rts_smooth, SmootherError
from DLMAX.ffs.devices import configure_devices


P_SEAS = 4


def _device():
    return configure_devices()[0] if callable(configure_devices) else None


def _model(q=3, delta=(0.95, 0.98), adapt=None, mult=False, var_power=1.0,
           seed=0, T=25):
    """A local-level + cyclic-seasonal uv_dlm, filtered over synthetic data."""
    M = P_SEAS
    p = 1 + M
    I = jnp.eye(M)
    G = block_diag(jnp.ones((1, 1)),
                   jnp.concatenate([I[:, [-1]], I[:, 0:M - 1]], 1))
    F = jnp.concatenate([jnp.ones(2), jnp.zeros(M - 1)])

    disc = jnp.concatenate([jnp.full(1, delta[0]), jnp.full(M, delta[1])])
    rng = np.random.default_rng(seed)
    ys = jnp.asarray(rng.normal(size=(T, q)) * 3.0 + 50.0
                     + 8.0 * np.sin(2 * np.pi * np.arange(T)[:, None] / M))

    from jax.sharding import NamedSharding, PartitionSpec as Pspec, Mesh
    mesh = Mesh(np.array(jax.devices()[:1]), ("s",))
    dev = NamedSharding(mesh, Pspec("s"))

    mult_comps = jnp.zeros(p)
    if mult:
        mult_comps = mult_comps.at[2].set(1.0)

    model = uv_dlm(
        series_ids=list(range(q)), F=F, G=G, n_regressors=0,
        m0=jnp.tile(jnp.concatenate([jnp.array([50.0]), jnp.zeros(M)]), (q, 1)),
        C0=jnp.tile(jnp.eye(p)[None] * 4.0, (q, 1, 1)),
        V0=jnp.full(q, 2.0), nu0=jnp.full(q, 1.0),
        disc_rates_norm=jnp.ones(p),          # adaptive floor unused when adapt=None
        disc_rates_damped=disc,               # the applied discount
        variance_disc=1.0, variance_power=var_power,
        mult_comps=mult_comps, device=dev, adapt=adapt,
    )
    for t in range(T):
        model.fwd_filter(ys[t], trajectory=True)
    return model, G, disc


# -----------------------------------------------------------------------------
# reference implementation — covariance form, no shared code
# -----------------------------------------------------------------------------


def _scale_free_C(traj):
    """C* = Z'Z / s — the scale-free covariance the smoother works in."""
    C = traj["Z"].swapaxes(-1, -2) @ traj["Z"]
    return C / jnp.asarray(traj["s"]).reshape(C.shape[0], C.shape[1], 1, 1)


def _reference_rts(m, C, G, disc):
    """Textbook RTS for one series. m (T,p), C (T,p,p). Returns s, S, B, H."""
    T, p = m.shape
    D = np.diag(1.0 / np.sqrt(np.asarray(disc)))
    G = np.asarray(G)
    m, C = np.asarray(m), np.asarray(C)

    B = np.zeros((T - 1, p, p))
    H = np.zeros((T - 1, p, p))
    R = np.zeros((T - 1, p, p))
    a = np.zeros((T - 1, p))
    for t in range(T - 1):
        R[t] = D @ (G @ C[t] @ G.T) @ D          # prior at t+1
        a[t] = G @ m[t]
        B[t] = C[t] @ G.T @ np.linalg.inv(R[t])
        H[t] = C[t] - B[t] @ R[t] @ B[t].T

    s = np.zeros_like(m)
    S = np.zeros_like(C)
    s[-1], S[-1] = m[-1], C[-1]
    for t in range(T - 2, -1, -1):
        s[t] = m[t] + B[t] @ (s[t + 1] - a[t])
        S[t] = C[t] - B[t] @ (R[t] - S[t + 1]) @ B[t].T
    return s, S, B, H


# -----------------------------------------------------------------------------
# smoothing
# -----------------------------------------------------------------------------


def test_rts_matches_covariance_form_reference():
    model, G, disc = _model()
    traj = model.trajectory
    out = rts_smooth(traj, G, disc)

    C = np.asarray(_scale_free_C(traj))
    m = np.asarray(traj["m"])

    for j in range(m.shape[1]):
        s_r, S_r, B_r, H_r = _reference_rts(m[:, j], C[:, j], G, disc)
        assert np.allclose(np.asarray(out["s"])[:, j], s_r, atol=1e-9), f"s, series {j}"
        assert np.allclose(np.asarray(out["S"])[:, j], S_r, atol=1e-8), f"S, series {j}"
        assert np.allclose(np.asarray(out["B"])[:, j], B_r, atol=1e-9), f"B, series {j}"
        assert np.allclose(np.asarray(out["H"])[:, j], H_r, atol=1e-8), f"H, series {j}"


def test_smoothed_endpoint_equals_filtered():
    """At t = T the smoothed and filtered posteriors coincide."""
    model, G, disc = _model()
    out = model.smooth()
    traj = model.trajectory
    assert jnp.allclose(out["s"][-1], traj["m"][-1])
    assert jnp.allclose(out["S"][-1], _scale_free_C(traj)[-1])


def test_smoothing_reduces_uncertainty():
    """Retrospective variance <= filtered variance, strictly somewhere."""
    model, G, disc = _model()
    out = model.smooth()
    C = _scale_free_C(model.trajectory)
    dS = jnp.diagonal(out["S"], axis1=-2, axis2=-1)
    dC = jnp.diagonal(C, axis1=-2, axis2=-1)
    assert bool(jnp.all(dS <= dC + 1e-8))
    assert bool(jnp.any(dS < dC - 1e-6))


def test_sqrtH_is_the_nearest_psd_root_of_H():
    """H can be genuinely indefinite under component discounting at differing
    rates (W = D P D - P is not PSD in general), so sqrtH is the PSD projection
    and the clip is reported rather than hidden."""
    model, G, disc = _model()
    out = model.smooth()
    rec = out["sqrtH"] @ out["sqrtH"].swapaxes(-1, -2)

    w, v = jnp.linalg.eigh((out["H"] + out["H"].swapaxes(-1, -2)) / 2)
    psd_proj = (v * jnp.clip(w, 0.0)[..., jnp.newaxis, :]) @ v.swapaxes(-1, -2)
    assert jnp.allclose(rec, psd_proj, atol=1e-8)
    assert float(out["psd_clip"]) >= 0.0


def test_component_discounting_can_make_H_indefinite():
    """Documents the model property, with the 2x2 counterexample."""
    P = np.array([[1.0, 1.0], [1.0, 1.0]])
    D = np.diag(1.0 / np.sqrt(np.array([0.9, 0.99])))
    W = D @ P @ D - P
    assert np.linalg.det(W) < 0                      # not PSD
    # a single shared rate is fine
    D1 = np.diag(1.0 / np.sqrt(np.array([0.9, 0.9])))
    assert np.all(np.linalg.eigvalsh(D1 @ P @ D1 - P) > -1e-12)


def test_rank_deficient_prior_uses_pseudo_inverse():
    """The standard W&H seasonal prior makes C0 — and every R — one rank short.

    seasonal_prior projects out the sum direction so the seasonal effects sum to
    zero, giving C0 rank p-1 and R condition ~1e16. B = C G' R^+ must then be a
    pseudo-inverse; a triangular solve divides by ~1e-17 and returns garbage in
    the null direction. H is insensitive to this (B R B' annihilates it) but the
    backward mean is not, so this checks B directly against pinv.
    """
    from DLMAX.dlm_core import seasonal_prior

    M = P_SEAS
    p = 1 + M
    C0 = np.asarray(block_diag(np.ones((1, 1)), seasonal_prior(jnp.ones(M))))
    assert np.linalg.matrix_rank(C0) == p - 1, "prior should be one rank short"

    I = jnp.eye(M)
    G = np.asarray(block_diag(jnp.ones((1, 1)),
                              jnp.concatenate([I[:, [-1]], I[:, 0:M - 1]], 1)))
    disc = np.concatenate([np.full(1, 0.95), np.full(M, 0.98)])
    D = np.diag(1.0 / np.sqrt(disc))

    # one prior-covariance step, exactly as the filter forms it
    R = D @ (G @ C0 @ G.T) @ D
    assert np.linalg.matrix_rank(R) == p - 1
    assert np.linalg.cond(R) > 1e12

    from DLMAX.smoother import _B_and_H, _prior_root
    # Exact root of the singular C0 (no jitter): C0 = V diag(w) V' => Z = sqrt(w) V',
    # so the null direction is exactly zero rather than ~1e-14, which would sit
    # awkwardly between _B_and_H's rcond and pinv's default.
    w, V = np.linalg.eigh(C0)
    Z = jnp.asarray((np.sqrt(np.maximum(w, 0.0)))[:, None] * V.T)
    assert np.allclose(np.asarray(Z.T @ Z), C0, atol=1e-12)

    NR = _prior_root(Z, jnp.asarray(G), 1.0 / jnp.sqrt(jnp.asarray(disc)))
    B, H = _B_and_H(Z, NR, jnp.asarray(G))

    B_ref = C0 @ G.T @ np.linalg.pinv(np.asarray(NR.T @ NR), rcond=1e-12,
                                      hermitian=True)
    assert np.allclose(np.asarray(B), B_ref, atol=1e-8), "B must match pinv"
    assert np.all(np.isfinite(np.asarray(H)))

    # and the failure mode this guards: a triangular solve against the same NR
    # blows up, which is what the previous implementation did.
    Tri = np.linalg.qr(np.asarray(NR), mode="r")
    assert np.min(np.abs(np.diag(Tri))) < 1e-8, "R really is rank-deficient here"


@pytest.mark.parametrize("var_power", [1.0, 0.5, 0.25])
def test_smoother_runs_for_any_var_power(var_power):
    """No var_power-specific code: the recursion consumes forward quantities
    that already embed var_scale."""
    model, G, disc = _model(var_power=var_power)
    traj = model.trajectory
    out = rts_smooth(traj, G, disc)
    C = np.asarray(_scale_free_C(traj))
    m = np.asarray(traj["m"])
    s_r, S_r, _, _ = _reference_rts(m[:, 0], C[:, 0], G, disc)
    assert np.allclose(np.asarray(out["s"])[:, 0], s_r, atol=1e-9)
    assert np.allclose(np.asarray(out["S"])[:, 0], S_r, atol=1e-8)


# -----------------------------------------------------------------------------
# FFBS
# -----------------------------------------------------------------------------


def test_ffbs_moments_match_the_smoother():
    """Averaged over many draws, FFBS reproduces the smoothed mean and variance."""
    model, G, disc = _model(q=1, T=15)
    sm = model.smooth()
    draws = model.backward_sample(random.PRNGKey(0), n_draws=40000)

    emp_mean = draws.mean(0)
    emp_var = draws.var(0)
    # draws carry the scale via the default right factor diag(sqrt(s_T)),
    # so they match the s_T-scaled smoothed covariance, not the scale-free one.
    sm_var = jnp.diagonal(sm["S_at_sT"], axis1=-2, axis2=-1)

    sd = jnp.sqrt(sm_var)
    assert float(jnp.max(jnp.abs(emp_mean - sm["s"]) / (sd + 1e-12))) < 0.06
    rel = jnp.abs(emp_var - sm_var) / (sm_var + 1e-12)
    assert float(jnp.max(rel)) < 0.10


def test_ffbs_terminal_draw_uses_the_filtered_posterior():
    """Theta_T ~ N(m_T, C_T) — the *posterior* at T, not the prior."""
    model, G, disc = _model(q=1, T=12)
    draws = model.backward_sample(random.PRNGKey(1), n_draws=40000)
    traj = model.trajectory
    C_T = traj["Z"][-1].swapaxes(-1, -2) @ traj["Z"][-1]
    assert jnp.allclose(draws[:, -1].mean(0), traj["m"][-1],
                        atol=4e-2 * jnp.sqrt(jnp.diagonal(C_T, axis1=-2, axis2=-1)))
    emp = jnp.cov(draws[:, -1, 0].T)
    assert jnp.allclose(emp, C_T[0], rtol=0.06, atol=1e-3)


def test_right_factor_couples_series():
    """L = chol(Sigma) induces exactly Sigma's cross-series correlation."""
    model, G, disc = _model(q=3, T=12)
    Sigma = jnp.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.25], [0.3, 0.25, 1.0]])
    L = jnp.linalg.cholesky(Sigma)
    draws = model.backward_sample(random.PRNGKey(2), n_draws=60000, right_factor=L)

    # innovations about the smoothed mean, at the terminal step, level component
    resid = draws[:, -1, :, 0] - draws[:, -1, :, 0].mean(0)
    corr = jnp.corrcoef(resid.T)
    target = Sigma / jnp.sqrt(jnp.outer(jnp.diag(Sigma), jnp.diag(Sigma)))
    assert jnp.allclose(corr, target, atol=0.02)


def test_default_right_factor_leaves_series_independent():
    model, G, disc = _model(q=3, T=12)
    draws = model.backward_sample(random.PRNGKey(3), n_draws=40000)
    resid = draws[:, -1, :, 0] - draws[:, -1, :, 0].mean(0)
    corr = jnp.corrcoef(resid.T)
    off = corr - jnp.diag(jnp.diag(corr))
    assert float(jnp.max(jnp.abs(off))) < 0.02


# -----------------------------------------------------------------------------
# guards
# -----------------------------------------------------------------------------


def test_smooth_without_trajectory_errors():
    model, G, disc = _model()
    model.clear_trajectory()
    with pytest.raises(SmootherError, match="trajectory=True"):
        model.smooth()


def test_backward_sample_rejects_multiplicative():
    model, G, disc = _model(mult=True)
    with pytest.raises(SmootherError, match="mult_comps"):
        model.backward_sample(random.PRNGKey(0))


def test_backward_sample_rejects_adaptive_discount():
    model, G, disc = _model(adapt=0.5)
    with pytest.raises(SmootherError, match="state-adaptive"):
        model.backward_sample(random.PRNGKey(0))


def test_allow_approximate_overrides_the_guard():
    model, G, disc = _model(mult=True)
    draws = model.backward_sample(random.PRNGKey(0), n_draws=2,
                                  allow_approximate=True)
    assert draws.shape[0] == 2
    assert bool(jnp.all(jnp.isfinite(draws)))


def test_smooth_flags_approximation_but_still_runs():
    model, G, disc = _model(mult=True)
    out = model.smooth()
    assert out["exact"] is False
    assert any("mult_comps" in r for r in out["approximations"])

    exact_model, _, _ = _model()
    out2 = exact_model.smooth()
    assert out2["exact"] is True
    assert out2["approximations"] == []

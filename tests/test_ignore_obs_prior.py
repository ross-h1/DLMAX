"""A missing observation must leave the posterior equal to the PRIOR.

On a skipped step (`y` is NaN) the correct update is `m_t = a_t`, `C_t = R_t`,
with the variance state untouched. Reverting the covariance to the *incoming*
`C_{t-1}` instead of advancing it to `R_t` would

  (i) dropped that step's discount forgetting  — a gap of k steps lost
      delta^-k of variance inflation, so the model left the gap overconfident;
 (ii) under non-identity G left the mean rotated and the covariance not, so
      `C[i, j]` stopped describing `m[i], m[j]`.

These tests pin both kernels to the prior.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from DLMAX.dlm_core import dlm_uv_fwd_qr_step, dlm_uv_fwd_svd_step


def _qr_step(C, m, s, nu, y, G, F, delta):
    """One QR-kernel step. Returns (m, C, s, nu)."""
    state = {"m": m, "Z": jnp.linalg.cholesky(C).T, "s": s, "nu": nu}
    new, _ = dlm_uv_fwd_qr_step(
        jnp.diag((1 - delta) / delta), jnp.array(1.0), jnp.array(1.0),
        jnp.zeros_like(F), state, {"F": F, "G": G, "y": y},
    )
    return new["m"], new["Z"].T @ new["Z"], new["s"], new["nu"]


def _svd_step(C, m, s, nu, y, G, F, delta):
    """One SVD-kernel step. Returns (m, C, s, nu). Convention C = UC diag(SC^2) UC'."""
    U, S, _ = jnp.linalg.svd(C)
    state = {"m": m, "UC": U, "SC": jnp.sqrt(S), "s": s, "nu": nu}
    new, _ = dlm_uv_fwd_svd_step(
        jnp.diag((1 - delta) / delta), jnp.array(1.0), jnp.array(1.0),
        jnp.zeros_like(F), state, {"F": F, "G": G, "y": y},
    )
    C_new = new["UC"] @ jnp.diag(new["SC"] ** 2) @ new["UC"].T
    return new["m"], C_new, new["s"], new["nu"]


KERNELS = pytest.mark.parametrize("step", [_qr_step, _svd_step], ids=["qr", "svd"])


@KERNELS
def test_missing_obs_applies_discount_forgetting(step):
    """Local level: C must advance to R = C/delta, not stay at C."""
    F, G, delta = jnp.ones(1), jnp.eye(1), jnp.full(1, 0.9)
    C = jnp.eye(1) * 4.0
    m, C_new, _, _ = step(C, jnp.array([5.0]), jnp.array(1.0), jnp.array(10.0),
                          jnp.array(jnp.nan), G, F, delta)
    assert C_new[0, 0] == pytest.approx(4.0 / 0.9, rel=1e-12)
    assert m[0] == pytest.approx(5.0, rel=1e-12)          # a = G m, G = I


@KERNELS
def test_missing_obs_rotates_covariance_with_mean(step):
    """Seasonal G: mean and covariance must both advance through G."""
    M = 4
    I = jnp.eye(M)
    G = jnp.concatenate([I[:, [-1]], I[:, 0:M - 1]], 1)   # cyclic seasonal
    F = jnp.concatenate([jnp.ones(1), jnp.zeros(M - 1)])
    delta = jnp.full(M, 1.0)                              # isolate the rotation

    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.normal(size=(M, M)))
    C = A @ A.T + M * jnp.eye(M)
    m = jnp.arange(1.0, M + 1)

    m_new, C_new, _, _ = step(C, m, jnp.array(1.0), jnp.array(10.0),
                              jnp.array(jnp.nan), G, F, delta)

    assert jnp.allclose(m_new, G @ m)
    assert jnp.allclose(C_new, G @ C @ G.T, atol=1e-10)
    # and specifically NOT frozen at C_{t-1}
    assert not jnp.allclose(C_new, C, atol=1e-6)


@KERNELS
def test_missing_obs_leaves_variance_state_untouched(step):
    """No observation => no information about the observation variance."""
    F, G, delta = jnp.ones(1), jnp.eye(1), jnp.full(1, 0.95)
    s, nu = jnp.array(2.5), jnp.array(7.0)
    _, _, s_new, nu_new = step(jnp.eye(1) * 3.0, jnp.array([1.0]), s, nu,
                               jnp.array(jnp.nan), G, F, delta)
    assert s_new == pytest.approx(float(s), rel=1e-12)
    assert nu_new == pytest.approx(float(nu), rel=1e-12)


@KERNELS
def test_gap_matches_repeated_prior_evolution(step):
    """k consecutive missing steps == k pure time-evolutions of (m, C)."""
    M = 3
    I = jnp.eye(M)
    G = jnp.concatenate([I[:, [-1]], I[:, 0:M - 1]], 1)
    F = jnp.concatenate([jnp.ones(1), jnp.zeros(M - 1)])
    delta = jnp.array([0.9, 0.95, 0.95])

    rng = np.random.default_rng(1)
    A = jnp.asarray(rng.normal(size=(M, M)))
    C = A @ A.T + M * jnp.eye(M)
    m = jnp.asarray(rng.normal(size=M))
    s, nu = jnp.array(1.3), jnp.array(4.0)

    C_k, m_k = C, m
    for _ in range(4):
        m_k, C_k, s, nu = step(C_k, m_k, s, nu, jnp.array(jnp.nan), G, F, delta)

    # analytic: m -> G^k m ;  C -> congruence-discounted G C G' , k times
    D = jnp.diag(1.0 / jnp.sqrt(delta))
    C_ref, m_ref = C, m
    for _ in range(4):
        m_ref = G @ m_ref
        C_ref = D @ (G @ C_ref @ G.T) @ D

    assert jnp.allclose(m_k, m_ref, atol=1e-10)
    assert jnp.allclose(C_k, C_ref, atol=1e-8)


@KERNELS
def test_observed_step_unchanged(step):
    """A normal (non-NaN) step is unaffected by the skipped-step branch."""
    M = 3
    I = jnp.eye(M)
    G = jnp.concatenate([I[:, [-1]], I[:, 0:M - 1]], 1)
    F = jnp.concatenate([jnp.ones(1), jnp.zeros(M - 1)])
    delta = jnp.array([0.9, 0.95, 0.95])

    rng = np.random.default_rng(2)
    A = jnp.asarray(rng.normal(size=(M, M)))
    C = A @ A.T + M * jnp.eye(M)
    m = jnp.asarray(rng.normal(size=M))

    m_new, C_new, s_new, nu_new = step(C, m, jnp.array(1.0), jnp.array(5.0),
                                       jnp.array(2.0), G, F, delta)
    # the observation moved the state and shrank uncertainty along F
    assert not jnp.allclose(m_new, G @ m)
    assert float(F @ C_new @ F) < float(F @ (G @ C @ G.T) @ F)
    assert float(nu_new) == pytest.approx(6.0)

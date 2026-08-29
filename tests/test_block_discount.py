"""Tests for the West & Harrison block (congruence) discount.

The model uses the block congruence discount ``R = B P B`` with
``B = diag(1/sqrt(delta))`` unconditionally — there is no diagonal-discount
alternative. These tests pin the block forecast/filter math: the W&H additive
origin-W growth with a congruence increment ``W = B C B - C``, the
correlation-preserving property, and the k=1 reduction.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import DLMAX.dlm_core as dc


# ---------------------------------------------------------------------------
# forecast kernel — RH trajectory (the kernel returns it directly as "C")
# ---------------------------------------------------------------------------

def _fcast_RH(delta, C, G, m, h):
    disc_factor = jnp.array((1.0 - delta) / delta)
    GH = jnp.array(np.stack([np.linalg.matrix_power(G, j + 1) for j in range(h)]))
    FH = jnp.array(np.tile(np.eye(1, G.shape[0], 0).ravel(), (h, 1)))
    Stm1 = {"m": jnp.array(m), "C": jnp.array(C),
            "s": jnp.array(1.0), "nu": jnp.array(10.0)}
    out = dc.dlm_uv_fcast_H(disc_factor, jnp.array(1.0), jnp.array(1.0),
                            jnp.zeros(G.shape[0]), Stm1, {"GH": GH, "FH": FH})
    return np.array(out["C"])


def _additive_RH(W, C, G, h):
    """W&H additive forecast: RH_j = G^{j+1} C G^{j+1}' + sum_{i=0..j} G^i W G^i'."""
    out = []
    for j in range(h):
        Gp = np.linalg.matrix_power(G, j + 1)
        R = Gp @ C @ Gp.T
        for i in range(j + 1):
            Gi = np.linalg.matrix_power(G, i)
            R = R + Gi @ W @ Gi.T
        out.append(R)
    return out


def test_fcast_block_matches_additive_block_increment():
    """Block forecast: W&H additive growth with the congruence increment
    W = B C B - C (the discount is NOT re-applied each forecast step)."""
    delta = np.array([0.95, 0.90])
    C = np.array([[2.0, 1.5], [1.5, 2.0]])
    G = np.array([[1.0, 1.0], [0.0, 1.0]])      # local linear trend
    h = 4
    RH = _fcast_RH(delta, C, G, np.array([1.0, 0.2]), h)

    b = 1.0 / np.sqrt(delta)
    W_block = (b[:, None] * b[None, :]) * C - C           # BCB - C
    for j, R_ref in enumerate(_additive_RH(W_block, C, G, h)):
        assert np.allclose(RH[j], R_ref, atol=1e-10), f"horizon {j}"


def test_block_increment_diagonal_is_per_state_inflation():
    """The congruence increment's DIAGONAL is the standard per-state variance
    inflation ((b_i^2 - 1) C_ii = (1-delta_i)/delta_i C_ii); the congruence adds
    the off-diagonal cross-covariance a per-state form would drop."""
    delta = np.array([0.95, 0.90])
    C = np.array([[2.0, 1.5], [1.5, 2.0]])
    b = 1.0 / np.sqrt(delta)
    W_block = (b[:, None] * b[None, :]) * C - C
    per_state = np.diag((1.0 - delta) / delta) * C
    assert np.allclose(np.diag(W_block), np.diag(per_state), atol=1e-12)
    assert abs(W_block[0, 1]) > 1e-6                       # off-diagonal added


def test_fcast_block_preserves_correlation_first_step():
    """The headline property: a common-delta block keeps the prior correlation
    exactly (R = P/delta, a scalar multiple of C)."""
    delta = np.array([0.9, 0.9])                 # common within-block discount
    C = np.array([[2.0, 1.2], [1.2, 1.0]])
    G = np.eye(2)
    corr0 = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
    RH_b = _fcast_RH(delta, C, G, np.zeros(2), 1)[0]
    corr_b = RH_b[0, 1] / np.sqrt(RH_b[0, 0] * RH_b[1, 1])
    assert np.isclose(corr_b, corr0, atol=1e-12)


def test_fcast_k1_additive_growth():
    """k=1 (no off-diagonals): R_h = C (1 + h (1-delta)/delta) — additive growth,
    no geometric/multiplicative inflation."""
    delta = np.array([0.93])
    C = np.array([[1.7]])
    G = np.array([[1.0]])
    RH = _fcast_RH(delta, C, G, np.array([0.5]), 5).reshape(-1)
    d = float(delta[0])
    for j in range(5):
        assert np.isclose(RH[j], 1.7 * (1.0 + (j + 1) * (1 - d) / d), atol=1e-10)


# ---------------------------------------------------------------------------
# filter kernel
# ---------------------------------------------------------------------------

def _filter_step(delta, C0, G, m0, y):
    k = G.shape[0]
    UC = jnp.eye(k)
    SC = jnp.sqrt(jnp.array(np.diag(C0)))         # C0 diagonal here
    Stm1 = {"m": jnp.array(m0), "UC": UC, "SC": SC,
            "s": jnp.array(1.0), "nu": jnp.array(10.0)}
    disc_factor = jnp.diag(jnp.array((1.0 - delta) / delta))
    Dt = {"F": jnp.array(np.eye(1, k, 0).ravel()), "G": jnp.array(G),
          "y": jnp.array(y)}
    return dc.dlm_uv_fwd_svd_step(
        disc_factor, jnp.array(1.0), jnp.array(1.0), jnp.zeros(k), Stm1, Dt
    )


def test_filter_block_runs_nan_free():
    """k=2: the block filter runs NaN-free with a finite, positive 1-step
    predictive variance."""
    delta = np.array([0.95, 0.90])
    C0 = np.array([[2.0, 0.0], [0.0, 1.0]])
    G = np.array([[1.0, 1.0], [0.0, 1.0]])
    state, model = _filter_step(delta, C0, G, np.array([1.0, 0.2]), 1.3)
    assert np.all(np.isfinite(np.array(model["q"])))
    assert np.all(np.isfinite(np.array(state["SC"])))
    assert float(model["q"]) > 0.0

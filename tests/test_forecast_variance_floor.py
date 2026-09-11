"""The h-step predictive variance cannot fall below the observation variance.

``dlm_uv_fcast_H`` returns ``q = s + F'RH F``. The state term is a quadratic
form in a covariance, so it is non-negative for any PSD ``RH`` and ``q >= s``
holds analytically. Numerically it did not: running the filter near marginal
stability -- a high discount floor over a long horizon -- leaves ``RH`` large
and ill-conditioned, and rounding can drive the term negative.

The consequence was silent rather than loud. A negative ``q`` reached
``np.sqrt`` in the Vincent SD combine, emitted *invalid value encountered in
sqrt*, went NaN, and was dropped by the ``np.isfinite`` guard there -- so that
worker simply lost its DMA weight for the cell, with nothing in the output to
say so. Observed on the ENTSO-E exhibit (hourly, h=168, annual Fourier, discounts
floored at 0.99); never on M4 or M5, which run nowhere near that operating point.

The clamp is on the state TERM, not on ``q``: it restores the analytic bound
instead of imposing an arbitrary epsilon, and it must not rescue a component
that has genuinely diverged.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from DLMAX.dlm_core import dlm_uv_fcast_H

K, H = 3, 4


def _fcast(C, s=1.0):
    """``q`` for a k-state random walk from origin covariance ``C``."""
    GH = jnp.stack([jnp.eye(K)] * H)
    DH = {"GH": GH, "FH": jnp.tile(jnp.ones(K), (H, 1))}
    state = {"m": jnp.ones(K), "C": jnp.asarray(C),
             "nu": jnp.asarray(10.0), "s": jnp.asarray(s)}
    out = dlm_uv_fcast_H(jnp.full(K, 0.01), jnp.asarray(0.99),
                         jnp.asarray(1.0), jnp.asarray(0.0), state, DH)
    return np.asarray(out["q"]), np.asarray(out["s"])


def test_psd_covariance_is_untouched():
    """The clamp is a no-op on sound arithmetic — the normal case must not move."""
    q, s = _fcast(np.eye(K) * 2.0)
    assert (q > 0).all()
    # q = s + a strictly positive state term, so it must exceed s and grow with h
    assert (q > s).all()
    assert (np.diff(q) > 0).all()


def test_indefinite_covariance_floors_at_the_observation_variance():
    """An indefinite ``RH`` yields ``q == s`` rather than a negative variance."""
    C = np.eye(K) * 1e-12
    C[0, 0] = -5.0                      # forces F'RH F < 0
    q, s = _fcast(C, s=1.0)
    assert np.isfinite(q).all(), "a rounding artefact must not produce NaN"
    assert (q > 0).all()
    np.testing.assert_allclose(q, s, rtol=1e-12)


def test_genuine_divergence_is_still_reported():
    """inf/NaN must pass through: the clamp fixes rounding, not divergence.

    If it rescued these too, a diverged worker would silently rejoin the DMA
    average with a plausible-looking variance — worse than the bug it fixes.
    """
    # np.full, not np.eye * inf: the latter puts 0 * inf = NaN off-diagonal and
    # emits a numpy warning of the test's own making.
    q_inf, _ = _fcast(np.full((K, K), np.inf))
    assert not np.isfinite(q_inf).any()
    q_nan, _ = _fcast(np.full((K, K), np.nan))
    assert np.isnan(q_nan).all()


def test_floor_holds_across_observation_scales():
    """``q >= s`` for any positive observation variance, not just s = 1."""
    C = np.eye(K) * 1e-12
    C[0, 0] = -50.0
    for s0 in (0.01, 1.0, 1000.0):
        q, s = _fcast(C, s=s0)
        assert np.isfinite(q).all()
        assert (q >= s * (1 - 1e-12)).all()

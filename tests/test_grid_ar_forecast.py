"""AR tails in the wing: the iterated-expectation forecast.

An exogenous tail can be forecast by filling ``FH`` with the known future rows.
An AR tail cannot: horizon j's row carries forecasts made at horizons below it.
``dlm_core.iterated_obs_forecast`` already solves this for the legacy path, and
documents that ``n_reg = 0`` reduces it EXACTLY to the standard ``f, q`` — so
the grid calls it rather than reimplementing it, and that reduction is a free
correctness test.

The state propagation is untouched by the state-dependent F (G is
block-diagonal and the coefficient block is the identity), so ``aH``/``RH`` come
verbatim from ``dlm_uv_fcast_H`` and only the observation step is redone.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from DLMAX.dlm_core import iterated_obs_forecast
from DLMAX.ffs.dlm_builder import AR, DLM, LocalLevel
from DLMAX.ffs.discount_grid import (
    _grid_model, _logit, forecast_origin, grid_stream_carry0,
    grid_stream_forecast, grid_stream_scan, grid_stream_static)

T, ORDER, H = 50, 2, 6
WARMUP = 8
D_LEVEL, D_AR, D_BETA = 0.95, 0.99, 0.99


def _series(seed=3):
    rng = np.random.default_rng(seed)
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = 10.0 + 0.6 * (y[t - 1] - 10.0) - 0.25 * (y[t - 2] - 10.0) \
            + rng.normal(0, 0.5)
    return y.reshape(T, 1)


def _grid_of(order):
    d = DLM(n_series=1)
    d.add_component(LocalLevel(name="level", disc_rate=D_LEVEL))
    if order:
        d.add_component(AR(name="ar", order=order, disc_rate=D_AR))
    d.set_error(disc_rate=D_BETA, power=1.0)
    comps = list(d.components)
    return [("cell0", _grid_model(comps, 1.0), tuple(comps))], d


def _static(grid, order):
    P = grid[0][1].n_blocks + 1
    vals = [_logit(D_LEVEL)] + ([_logit(D_AR)] if order else []) + [_logit(D_BETA)]
    assert len(vals) == P
    clip = {"level": (D_LEVEL, D_LEVEL), "beta": (D_BETA, D_BETA)}
    if order:
        clip["regression"] = (D_AR, D_AR)
    return grid_stream_static(grid, offset=0.0, adapt_guard=None, learn_dma=False,
                              disc_init=jnp.asarray([vals]), clip=clip)


def test_seed_lags_change_the_forecast():
    """An AR tail that is not fed forward would forecast as if the coefficients
    were multiplying zero — the silent failure this whole exercise guards."""
    y = _series()
    grid, _ = _grid_of(ORDER)
    static = _static(grid, ORDER)
    ys = jnp.asarray(y)
    carry = grid_stream_scan(static, grid_stream_carry0(grid, static, ys, WARMUP),
                             ys, jnp.zeros(T),
                             jnp.asarray(_lag_matrix(y, ORDER)))

    seed = jnp.asarray(y[-1::-1][:ORDER].reshape(1, ORDER))     # most recent first
    loc_ar, _q, _c = grid_stream_forecast(static, carry, H, None, seed)
    loc_no, _q2, _c2 = grid_stream_forecast(static, carry, H)
    assert np.isfinite(np.asarray(loc_ar)).all()
    assert not np.allclose(np.asarray(loc_ar), np.asarray(loc_no))


def _lag_matrix(y, order):
    """(T, q, order) of [y_{t-1}, ..., y_{t-order}]; zeros where unavailable."""
    T_, q = y.shape
    out = np.zeros((T_, q, order))
    for lag in range(1, order + 1):
        out[lag:, :, lag - 1] = y[:-lag]
    return out


def test_reduction_to_static_when_there_is_no_tail():
    """iterated_obs_forecast with n_reg = 0 must equal the standard forecast --
    its own documented property, and the cheapest check that the wiring is right."""
    y = _series()
    grid, _ = _grid_of(0)
    static = _static(grid, 0)
    ys = jnp.asarray(y)
    carry = grid_stream_scan(static, grid_stream_carry0(grid, static, ys, WARMUP),
                             ys, jnp.zeros(T))
    a, _, _ = grid_stream_forecast(static, carry, H)
    b, _, _ = grid_stream_forecast(static, carry, H, None,
                                   jnp.zeros((1, 0)))   # empty tail
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_grid_ar_forecast_matches_the_primitive():
    """The grid's AR forecast must equal a direct call to the legacy primitive on
    the same state — i.e. the grid is USING it, not approximating it."""
    y = _series()
    grid, _ = _grid_of(ORDER)
    gm = grid[0][1]
    static = _static(grid, ORDER)
    ys = jnp.asarray(y)
    carry = grid_stream_scan(static, grid_stream_carry0(grid, static, ys, WARMUP),
                             ys, jnp.zeros(T),
                             jnp.asarray(_lag_matrix(y, ORDER)))
    seed = np.ascontiguousarray(y[::-1][:ORDER].ravel())        # most recent first

    loc, q, _c = grid_stream_forecast(static, carry, H, None,
                                      jnp.asarray(seed.reshape(1, ORDER)))

    # the same thing by hand, straight through forecast_origin
    st = jax.tree_util.tree_map(lambda x: x[0, 0, 0], carry[0][0])
    wth = np.asarray(carry[0][5])[0, 0, 0]
    theta = 1.0 / (1.0 + np.exp(-wth))
    f_h, q_h, _nu = forecast_origin(gm, st, jnp.asarray(theta), H,
                                    None, jnp.asarray(seed))
    # the block combines over workers; with offset 0 every worker is identical,
    # so the combined mean is that single worker's
    assert np.isfinite(np.asarray(f_h)).all()     # else the compare is vacuous
    np.testing.assert_allclose(np.asarray(loc)[0], np.asarray(f_h),
                               rtol=1e-9, atol=1e-9)

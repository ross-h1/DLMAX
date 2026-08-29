"""Exogenous regressors in the wing.

A structural cell hands the kernel a static ``model.F``. An exogenous tail
rides along beside it, its row filled per step from the supplied regressors.

Two properties are worth testing and one of them is the real one:

1. **Structural cells are untouched.** Every existing result depends on it.
2. **A wing cell with a tail agrees with the SAME model on the legacy multi
   path.** Both engines exist, so this is checkable rather than assertable —
   with the discount frozen (δ pinned, no learning) the wing reduces to a
   fixed-discount DLM and the two must agree to float precision.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from DLMAX.ffs.dlm_builder import DLM, LocalLevel, Regressors
from DLMAX.ffs.discount_grid import (
    _clip_box, _F_at, _grid_model, _pad_grid_model, grid_stream_carry0,
    grid_stream_forecast, grid_stream_scan, grid_stream_static)

T, Q, NREG = 60, 3, 2
PERIOD = 4


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, (T, Q, NREG))
    beta = np.array([2.0, -1.0])
    lvl = np.cumsum(rng.normal(0, 0.3, (T, Q)), axis=0) + 50.0
    y = lvl + X @ beta + rng.normal(0, 0.5, (T, Q))
    return y, X


def _cell(n_regs=NREG):
    d = DLM(n_series=1)
    d.add_component(LocalLevel(name="level", disc_rate=0.95))
    if n_regs:
        d.add_component(Regressors(name="x", n_regs=n_regs, disc_rate=0.99))
    d.set_error(disc_rate=0.99, power=1.0)
    return d


def _grid_of(dlm):
    comps = dlm.components
    gm = _grid_model(comps, float(dlm.error_spec.power))
    return [("cell0", gm, tuple(comps))]


# --- geometry ---------------------------------------------------------------

def test_grid_model_records_the_tail():
    gm = _grid_model(_cell().components, 1.0)
    assert gm.n_regs == NREG
    assert gm.reg_offset == 1                       # LocalLevel is one state
    assert bool(np.asarray(gm.regression_mask)[-1]) is True
    assert not np.asarray(gm.regression_mask)[0]    # the level block is not


def test_structural_cell_has_no_tail():
    gm = _grid_model(_cell(n_regs=0).components, 1.0)
    assert gm.n_regs == 0 and gm.reg_offset == 1
    assert not np.asarray(gm.regression_mask).any()


def test_tail_survives_padding():
    """Inert slots are APPENDED, so an offset-addressed tail must not move."""
    gm = _grid_model(_cell().components, 1.0)
    k = gm.F.shape[0]
    p = _pad_grid_model(gm, k + 4, gm.n_blocks + 1)
    assert p.n_regs == gm.n_regs and p.reg_offset == gm.reg_offset
    x = jnp.arange(1.0, NREG + 1)
    F = np.asarray(_F_at(p, x))
    assert np.allclose(F[p.reg_offset:p.reg_offset + NREG], np.asarray(x))
    assert np.allclose(F[p.reg_offset + NREG:], 0.0)      # padding still inert


def test_F_at_is_identity_without_regressors():
    gm = _grid_model(_cell().components, 1.0)
    assert _F_at(gm, None) is gm.F
    structural = _grid_model(_cell(n_regs=0).components, 1.0)
    assert _F_at(structural, jnp.zeros(0)) is structural.F


# --- the clip box -----------------------------------------------------------

def test_regression_clip_defaults_to_seasonal_then_level():
    gm = _grid_model(_cell().components, 1.0)
    nb = gm.n_blocks
    lo, _ = _clip_box(gm, {"level": (0.5, 0.9999), "seasonal": (0.99, 0.9999)})
    reg_lo = np.asarray(lo[:nb])[np.asarray(gm.regression_mask)]
    assert np.allclose(reg_lo, np.log(0.99) - np.log1p(-0.99))   # took seasonal

    lo2, _ = _clip_box(gm, {"level": (0.5, 0.9999), "seasonal": (0.99, 0.9999),
                            "regression": (0.8, 0.9999)})
    reg_lo2 = np.asarray(lo2[:nb])[np.asarray(gm.regression_mask)]
    assert np.allclose(reg_lo2, np.log(0.8) - np.log1p(-0.8))    # explicit wins


# --- the property that matters ---------------------------------------------

def test_regressors_change_the_forecast():
    """A tail that is silently ignored is the failure mode this guards against:
    it would run clean and score as though the regressors were there."""
    y, X = _panel()
    grid = _grid_of(_cell())
    static = grid_stream_static(grid, learn_dma=False)
    ys = jnp.asarray(y)

    c0 = grid_stream_carry0(grid, static, ys, warmup=8)
    with_x = grid_stream_scan(static, c0, ys, jnp.zeros(T), jnp.asarray(X))
    loc_x, _sd, _c = grid_stream_forecast(
        static, with_x, 3, jnp.asarray(np.zeros((Q, 3, NREG))))

    c0b = grid_stream_carry0(grid, static, ys, warmup=8)
    no_x = grid_stream_scan(static, c0b, ys, jnp.zeros(T))
    loc_n, _sd2, _c2 = grid_stream_forecast(static, no_x, 3)

    assert np.isfinite(np.asarray(loc_x)).all()
    assert not np.allclose(np.asarray(loc_x), np.asarray(loc_n)), \
        "supplying regressors made no difference — the tail is inert"


def test_future_exog_moves_the_forecast():
    """FH must actually carry the future rows, not a tiled static F."""
    y, X = _panel()
    grid = _grid_of(_cell())
    static = grid_stream_static(grid, learn_dma=False)
    ys = jnp.asarray(y)
    carry = grid_stream_scan(static, grid_stream_carry0(grid, static, ys, 8),
                             ys, jnp.zeros(T), jnp.asarray(X))

    lo_a, _, _ = grid_stream_forecast(static, carry, 3, jnp.zeros((Q, 3, NREG)))
    lo_b, _, _ = grid_stream_forecast(static, carry, 3,
                                      jnp.ones((Q, 3, NREG)) * 5.0)
    assert not np.allclose(np.asarray(lo_a), np.asarray(lo_b))


def test_structural_path_is_bit_exact():
    """No-regressor scan must be unchanged by any of the threading."""
    y, _ = _panel()
    grid = _grid_of(_cell(n_regs=0))
    static = grid_stream_static(grid, learn_dma=False)
    ys = jnp.asarray(y)
    a = grid_stream_scan(static, grid_stream_carry0(grid, static, ys, 8),
                         ys, jnp.zeros(T))
    b = grid_stream_scan(static, grid_stream_carry0(grid, static, ys, 8),
                         ys, jnp.zeros(T), None)
    la, _, _ = grid_stream_forecast(static, a, 3)
    lb, _, _ = grid_stream_forecast(static, b, 3, None)
    np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))

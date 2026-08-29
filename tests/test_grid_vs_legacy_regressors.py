"""Cross-engine agreement: a wing cell with an exogenous tail vs the legacy path.

Showing the tail is not inert does NOT show it is *right* — a sign error or a
misaligned tail offset would pass such a check. This does, by
running the SAME model through both engines and demanding they agree.

The wing learns its discount by RTRL, so to compare against a fixed-discount
``uv_dlm`` the learning has to be pinned: ``offset=0`` collapses the three
wingmen onto the centre, a degenerate clip box (lo == hi) freezes every
parameter, and ``disc_init`` starts them already at the pinned values so even
step 1 uses them.

The covariance guard must be set the SAME WAY ON BOTH ENGINES, and this is the
one that is easy to get wrong: ``uv_dlm.adapt`` defaults to 0.5 and
``DLM.compile()`` never turns it off, so pinning only the grid's ``adapt_guard``
to ``None`` leaves the two paths running different discount envelopes. That
shows up as an agreeing predictive MEAN and a disagreeing predictive VARIANCE —
``f = F.(G.m)`` does not involve the covariance, while ``q`` does — which reads
like a covariance bug and is not one. Both settings are exercised below.

Both engines run the same QR kernel underneath (``uv_dlm.fwd_filter`` and the
grid both use ``dlm_uv_fwd_qr_step``), so agreement should be at float
precision, not merely close.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.dlm_builder import DLM, LocalLevel, LocalTrend, Regressors
from DLMAX.ffs.discount_grid import (
    _grid_model, _logit, grid_stream_carry0, grid_stream_scan, grid_stream_static)

T, NREG, H = 40, 2, 5
WARMUP = 8
D_LEVEL, D_REG, D_BETA = 0.95, 0.98, 0.99


def _data(seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (T, 1, NREG))
    lvl = np.cumsum(rng.normal(0, 0.2, T)) + 20.0
    y = lvl + X[:, 0, :] @ np.array([1.5, -0.7]) + rng.normal(0, 0.4, T)
    return y.reshape(T, 1), X


def _build():
    d = DLM(n_series=1)
    d.add_component(LocalLevel(name="level", disc_rate=D_LEVEL))
    d.add_component(Regressors(name="x", n_regs=NREG, disc_rate=D_REG))
    d.set_error(disc_rate=D_BETA, power=1.0)
    return d


def _pinned_static(grid, adapt_guard=None):
    """A wing frozen at the fixed discounts, so it IS a fixed-discount DLM."""
    P = grid[0][1].n_blocks + 1
    di = np.array([[_logit(D_LEVEL), _logit(D_REG), _logit(D_BETA)]])
    assert di.shape[1] == P
    return grid_stream_static(
        grid, offset=0.0, adapt_guard=adapt_guard, learn_dma=False,
        disc_init=jnp.asarray(di),
        clip={"level": (D_LEVEL, D_LEVEL), "regression": (D_REG, D_REG),
              "beta": (D_BETA, D_BETA)})


@pytest.mark.parametrize("guard", [None, 0.5])
def test_wing_with_exog_matches_the_legacy_filter(guard):
    """Both guard settings, because they fail differently.

    ``guard=None`` is the plain fixed-discount comparison. ``guard=0.5`` is the
    production configuration and additionally pins the two implementations of the
    info-envelope against each other: the grid's, inside ``_filter_step``, and
    ``dlm_core._adapt_discount_z`` on the legacy side. (Their COMPOSITION with the
    damped correction needs a damped trend to exercise —
    ``test_damped_trend_matches_the_legacy_filter``.)
    """
    y, X = _data()
    dlm = _build()
    comps = list(dlm.components)
    grid = [("cell0", _grid_model(comps, 1.0), tuple(comps))]
    static = _pinned_static(grid, adapt_guard=guard)

    ys = jnp.asarray(y)
    # warmup >= 4: the diffuse elicitation returns a NaN prior at warmup=0, and a
    # NaN-vs-NaN comparison passes vacuously under assert_allclose's equal_nan.
    carry = grid_stream_carry0(grid, static, ys, warmup=WARMUP)

    # legacy uv_dlm, started from the grid's OWN prior so the comparison isolates
    # the engine rather than the elicitation
    uv = dlm.compile(pd.DataFrame(y), warmup_steps=0, h=H)
    # uv_dlm.adapt defaults to 0.5 and compile() never clears it, so this line is
    # what makes the two engines the same model. Without it the means agree and
    # the variances do not -- see the module docstring.
    assert uv.adapt == 0.5, "uv_dlm's guard default moved; this test pins it"
    uv.adapt = guard
    st0 = carry[0][0]                                   # (nf, q, wings, ...)
    for k in ("m", "Z", "s", "nu"):
        v = np.asarray(st0[k])[0, 0, 0]                 # family 0, series 0, wing 0
        uv.dlm_state[k] = jnp.asarray(v)[None, ...]     # (1, ...)

    f_leg, q_leg = [], []
    for t in range(T):
        uv.fwd_filter(jnp.asarray(y[t]), regressors=jnp.asarray(X[t]))
        f_leg.append(float(np.ravel(uv.model["f"])[0]))
        q_leg.append(float(np.ravel(uv.model["q"])[0]))

    _c, (Fg, Qg) = grid_stream_scan(static, carry, ys, jnp.zeros(T),
                                    jnp.asarray(X), return_trace=True)
    f_grid = np.asarray(Fg)[:, 0, 0]                    # (T, q, M) -> series 0, worker 0
    q_grid = np.asarray(Qg)[:, 0, 0]

    # finiteness FIRST: assert_allclose's equal_nan default would otherwise let
    # a pair of all-NaN traces pass as perfect agreement.
    assert np.isfinite(f_grid).all() and np.isfinite(q_grid).all()
    assert np.isfinite(f_leg).all() and np.isfinite(q_leg).all()
    np.testing.assert_allclose(f_grid, np.array(f_leg), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(q_grid, np.array(q_leg), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("guard", [None, 0.5])
def test_structural_cell_matches_the_legacy_filter(guard):
    """The same comparison with NO tail at all.

    Cheap canary: it shares every mechanism with the test above except the
    regressors, so when the pair diverges together the cause is the engine, and
    when only the tailed one moves the cause is the tail. That distinction is
    what took the longest to establish the first time round."""
    y = _data()[0]
    dlm = DLM(n_series=1)
    dlm.add_component(LocalLevel(name="level", disc_rate=D_LEVEL))
    dlm.set_error(disc_rate=D_BETA, power=1.0)
    comps = list(dlm.components)
    grid = [("cell0", _grid_model(comps, 1.0), tuple(comps))]

    di = np.array([[_logit(D_LEVEL), _logit(D_BETA)]])
    static = grid_stream_static(
        grid, offset=0.0, adapt_guard=guard, learn_dma=False,
        disc_init=jnp.asarray(di),
        clip={"level": (D_LEVEL, D_LEVEL), "beta": (D_BETA, D_BETA)})

    ys = jnp.asarray(y)
    carry = grid_stream_carry0(grid, static, ys, warmup=WARMUP)

    uv = dlm.compile(pd.DataFrame(y), warmup_steps=0, h=H)
    uv.adapt = guard
    st0 = carry[0][0]
    for k in ("m", "Z", "s", "nu"):
        uv.dlm_state[k] = jnp.asarray(np.asarray(st0[k])[0, 0, 0])[None, ...]

    f_leg, q_leg = [], []
    for t in range(T):
        uv.fwd_filter(jnp.asarray(y[t]))
        f_leg.append(float(np.ravel(uv.model["f"])[0]))
        q_leg.append(float(np.ravel(uv.model["q"])[0]))

    _c, (Fg, Qg) = grid_stream_scan(static, carry, ys, jnp.zeros(T),
                                    return_trace=True)
    f_grid, q_grid = np.asarray(Fg)[:, 0, 0], np.asarray(Qg)[:, 0, 0]

    assert np.isfinite(f_grid).all() and np.isfinite(q_grid).all()
    np.testing.assert_allclose(f_grid, np.array(f_leg), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(q_grid, np.array(q_leg), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("guard", [None, 0.5])
@pytest.mark.parametrize("damping", [1.0, 0.95])
def test_damped_trend_matches_the_legacy_filter(damping, guard):
    """Cross the guard with the damped-discount correction.

    ``damping=1.0`` leaves ``damped`` all-ones, so nothing about the composition
    is tested; ``damping=0.95`` puts δ² = 0.9025 on the growth slot and makes the
    ORDER load-bearing. Both engines must apply the envelope to the raw delta and
    multiply by ``damped`` afterwards — the grid in ``_filter_step``, the legacy
    path in ``dlm_core._adapt_discount_z``. Swapping either one shows up here and
    nowhere else in this file."""
    rng = np.random.default_rng(3)
    y = (np.cumsum(np.cumsum(rng.normal(0, 0.05, T)) + 0.1) + 20.0).reshape(T, 1)

    dlm = DLM(n_series=1)
    dlm.add_component(LocalTrend(name="trend", disc_rate=D_LEVEL, damping=damping))
    dlm.set_error(disc_rate=D_BETA, power=1.0)
    comps = list(dlm.components)
    gm = _grid_model(comps, 1.0)
    grid = [("cell0", gm, tuple(comps))]

    di = np.array([[_logit(D_LEVEL)] * gm.n_blocks + [_logit(D_BETA)]])
    static = grid_stream_static(
        grid, offset=0.0, adapt_guard=guard, learn_dma=False,
        disc_init=jnp.asarray(di),
        clip={"level": (D_LEVEL, D_LEVEL), "beta": (D_BETA, D_BETA)})

    ys = jnp.asarray(y)
    carry = grid_stream_carry0(grid, static, ys, warmup=WARMUP)

    uv = dlm.compile(pd.DataFrame(y), warmup_steps=0, h=H)
    uv.adapt = guard
    # the correction itself must match before its composition can mean anything
    np.testing.assert_allclose(np.asarray(gm.damped),
                               np.asarray(uv.disc_rates_damped))
    st0 = carry[0][0]
    for k in ("m", "Z", "s", "nu"):
        uv.dlm_state[k] = jnp.asarray(np.asarray(st0[k])[0, 0, 0])[None, ...]

    f_leg, q_leg = [], []
    for t in range(T):
        uv.fwd_filter(jnp.asarray(y[t]))
        f_leg.append(float(np.ravel(uv.model["f"])[0]))
        q_leg.append(float(np.ravel(uv.model["q"])[0]))

    _c, (Fg, Qg) = grid_stream_scan(static, carry, ys, jnp.zeros(T),
                                    return_trace=True)
    f_grid, q_grid = np.asarray(Fg)[:, 0, 0], np.asarray(Qg)[:, 0, 0]

    assert np.isfinite(f_grid).all() and np.isfinite(q_grid).all()
    np.testing.assert_allclose(f_grid, np.array(f_leg), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(q_grid, np.array(q_leg), rtol=1e-9, atol=1e-9)


def test_regressor_scale_changes_the_filter():
    """Guard on the guard: the tail must actually reach the filter.

    NB permuting the regressor COLUMNS does not work as a check, which is why
    this scales instead. The predictive is exchangeable in the regressors: swap
    the columns and the learned coefficients swap with them, leaving
    x1*b1 + x2*b2 invariant (verified: identical to 3.6e-15). That is a property
    of the model, not a bug, and it means permutation can never detect a
    misaligned tail."""
    y, X = _data()
    dlm = _build()
    comps = list(dlm.components)
    grid = [("cell0", _grid_model(comps, 1.0), tuple(comps))]
    static = _pinned_static(grid)
    ys = jnp.asarray(y)
    carry = grid_stream_carry0(grid, static, ys, warmup=WARMUP)

    _c1, (F1, _) = grid_stream_scan(static, carry, ys, jnp.zeros(T),
                                    jnp.asarray(X), return_trace=True)
    _c2, (F2, _) = grid_stream_scan(static, carry, ys, jnp.zeros(T),
                                    jnp.asarray(X * 100.0), return_trace=True)
    assert np.isfinite(np.asarray(F1)).all()      # else "differs" is vacuous
    assert not np.allclose(np.asarray(F1), np.asarray(F2))

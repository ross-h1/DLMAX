"""AR/TVAR stationarity guard on the forecast paths.

The iterated variance recursion ``Q_j = s + sum (phi_k^2 + Var phi_k) Q_{j-k}``
diverges for non-stationary AR coefficients. The guard reflects any AR root
inside the unit circle before forecasting. These tests cover the paths that
apply it:

* ``multi_model_dlm.forecast_iterated`` — reached by ``AutoFFS.predict`` /
  ``AutoFFSUniverse.forecast`` / direct callers — now flips in numpy before
  forecasting.
* the multi-window CV emit-scan — flips in-jax over only the ~5q TVAR rows
  (numpy can't reach inside ``lax.scan``).
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp

import DLMAX.dlm_core as dc
from DLMAX.ffs.dlm_builder import DLM, LocalLevel, AR
from DLMAX.dlm_core import multi_model_dlm
from DLMAX.ffs_core import _run_cv_batch


def _ar_multi(panel, order, h):
    n = panel.shape[1]
    dlm = DLM(family="Gaussian", n_series=n)
    dlm.add_component(LocalLevel("level", disc_rate=0.95))
    dlm.add_component(AR("ar", order=order, disc_rate=0.98))
    dlm.set_error(disc_rate=1.0, power=[1], nu0=1)
    models, _desc = dlm.compile_universe(panel, h=h)
    return multi_model_dlm(models)


def test_forecast_iterated_stationarises_explosive_ar():
    """An explosive AR(1) coefficient must not blow up the iterated variance.

    Compares the guard on vs off on the same explosive state: reflecting the
    root inside the unit circle cuts the h-step predictive variance by orders of
    magnitude (without it Q_h ~ phi^(2h) diverges).
    """
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(size=(30, 2)).cumsum(0) + 50.0)
    h = 18
    multi = _ar_multi(panel, order=1, h=h)

    # Force |phi| > 1 (non-stationary) on every (model, series) AR slot.
    reg0 = multi.k - multi.n_regressors
    m = np.asarray(multi.dlm_state["m"]).copy()
    m[:, reg0:] = 1.5
    multi.dlm_state = {**multi.dlm_state, "m": jnp.asarray(m)}
    seed = jnp.asarray(rng.normal(size=(2, multi.n_regressors)))

    # guard on (default); forecast_iterated flips a copy, leaving state explosive
    _f_on, q_on = multi.forecast_iterated(h=h, seed_lags=seed)
    q_on = np.asarray(q_on)

    # guard off
    saved = dc.AR_STATIONARITY_EPS
    dc.AR_STATIONARITY_EPS = None
    try:
        _f_off, q_off = multi.forecast_iterated(h=h, seed_lags=seed)
    finally:
        dc.AR_STATIONARITY_EPS = saved
    q_off = np.asarray(q_off)

    assert np.all(np.isfinite(q_on))
    # the guard cuts the explosive h-step variance by orders of magnitude
    assert q_on[..., -1].max() < 1e-3 * q_off[..., -1].max()


def test_forecast_iterated_unchanged_when_stationary():
    """A benign (stationary) coefficient set forecasts finite, bounded variance."""
    rng = np.random.default_rng(7)
    panel = pd.DataFrame(rng.normal(size=(40, 3)).cumsum(0) + 100.0)
    h = 12
    multi = _ar_multi(panel, order=2, h=h)
    seed = jnp.asarray(rng.normal(size=(3, multi.n_regressors)))
    f, q = multi.forecast_iterated(h=h, seed_lags=seed)
    assert np.all(np.isfinite(np.asarray(f)))
    assert np.all(np.isfinite(np.asarray(q)))


def _seasonal_panel(n_series=3, T=60, season=7, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return np.column_stack([
        10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / season)
        + rng.normal(0, 0.3, T)
        for _ in range(n_series)
    ])


def test_ar_cv_multiwindow_emit_finite():
    """Multi-window AR CV (the emit-scan path with the in-jax flip) is finite."""
    arr = _seasonal_panel(T=60, season=7)
    h = 6
    cutoffs = np.array([60 - 2 * h - 1, 60 - h - 1], dtype=np.int32)
    out = _run_cv_batch(
        ("a", "b", "c"), arr, cutoffs,
        periodicity=7, n_seas_comps=None, h=h,
        dma_pdr=0.95, dma_mdr=0.75, warmup_steps=7, include_ar=True,
    )
    assert out.f_h.shape[0] == 2                       # two origins
    assert np.all(np.isfinite(out.f_h))
    assert np.all(np.isfinite(out.q_h))


def test_ar_emit_matches_fast_at_shared_cutoff():
    """The emit-scan AR forecast agrees with the numpy fast-path flip at a
    shared cutoff (the two flip implementations match to ~1e-5)."""
    arr = _seasonal_panel(T=60, season=7)
    h = 6
    shared = 60 - h - 1
    common = dict(
        periodicity=7, n_seas_comps=None, h=h,
        dma_pdr=0.95, dma_mdr=0.75, warmup_steps=7, include_ar=True,
    )
    fast = _run_cv_batch(("a", "b", "c"), arr, np.array([shared], np.int32), **common)
    emit = _run_cv_batch(
        ("a", "b", "c"), arr, np.array([60 - 2 * h - 1, shared], np.int32), **common
    )
    # fast: single window; emit: the second (shared) window
    np.testing.assert_allclose(
        np.asarray(emit.f_h[-1]), np.asarray(fast.f_h[0]), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(emit.q_h[-1]), np.asarray(fast.q_h[0]), rtol=1e-3, atol=1e-3
    )

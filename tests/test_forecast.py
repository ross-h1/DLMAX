"""Forecast SD column tests.

Covers:
- _t_predictive_sd for single-component (matches scalar t-variance).
- _t_predictive_sd for multi-component mixture (matches independent
  numpy reference computation).
- Schema: column appears in cross_validation, forecast, predict outputs.
"""
# NB AutoFFS now defaults to the WING grid and implements the CV arm only.
# These tests exercise the legacy static fit/predict machinery (ds/freq
# handling, forecast plumbing), so they target StaticFFS directly.

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs_core import _t_predictive_sd, FFSPredictive


def _make_pred(nm, n_series, h, *, seed=0):
    rng = np.random.default_rng(seed)
    f_h = rng.normal(size=(nm, n_series, h))
    q_h = rng.uniform(0.5, 2.0, size=(nm, n_series, h))
    nu = rng.uniform(3.0, 20.0, size=(nm, n_series))
    weights_raw = rng.uniform(size=(nm, n_series))
    weights = weights_raw / weights_raw.sum(axis=0, keepdims=True)
    return FFSPredictive(
        loc=None,
        sd=None,
        f_h=f_h,
        q_h=q_h,
        nu=nu,
        weights=weights,
    )


def test_single_component_matches_t_variance():
    """One model with weight 1: SD should equal sqrt(q · ν/(ν-2))."""
    h, n = 5, 3
    f_h = np.zeros((1, n, h))
    q_h = np.full((1, n, h), 4.0)
    nu = np.full((1, n), 10.0)
    weights = np.ones((1, n))
    pred = FFSPredictive(loc=None, sd=None, f_h=f_h, q_h=q_h, nu=nu, weights=weights)

    sd = _t_predictive_sd(pred)
    expected = np.sqrt(4.0 * 10.0 / 8.0)  # q · ν/(ν-2) with q=4, ν=10
    np.testing.assert_allclose(sd, np.full((h, n), expected), rtol=1e-12)


def test_mixture_matches_law_of_total_variance():
    """Independent numpy computation of mixture variance should agree."""
    nm, n, h = 4, 3, 6
    pred = _make_pred(nm, n, h, seed=42)

    # Compute reference via straightforward numpy
    f_h, q_h, nu, w = pred.f_h, pred.q_h, pred.nu, pred.weights
    var_m = q_h * (nu / (nu - 2.0))[..., None]  # (nm, n, h)
    mu_mix = (f_h * w[..., None]).sum(axis=0)  # (n, h)
    e_var = (var_m * w[..., None]).sum(axis=0)  # (n, h)
    var_e = ((f_h - mu_mix[None, ...]) ** 2 * w[..., None]).sum(axis=0)
    expected = np.sqrt(e_var + var_e).T  # (h, n)

    sd = _t_predictive_sd(pred)
    np.testing.assert_allclose(sd, expected, rtol=1e-12)


def test_low_nu_yields_nan():
    """ν ≤ 2 means undefined t-variance — mixture SD should be NaN."""
    nm, n, h = 2, 2, 3
    f_h = np.zeros((nm, n, h))
    q_h = np.ones((nm, n, h))
    nu = np.array([[1.5, 5.0], [10.0, 5.0]])  # series 0 has one component with nu < 2
    weights = np.full((nm, n), 0.5)
    pred = FFSPredictive(loc=None, sd=None, f_h=f_h, q_h=q_h, nu=nu, weights=weights)

    sd = _t_predictive_sd(pred)
    # Series 0: NaN (one component has ν=1.5). Series 1: finite.
    assert np.all(np.isnan(sd[:, 0])), f"expected NaN, got {sd[:, 0]}"
    assert np.all(np.isfinite(sd[:, 1])), f"expected finite, got {sd[:, 1]}"


def test_sd_is_nonnegative():
    """SD column must never be negative."""
    pred = _make_pred(nm=5, n_series=4, h=8, seed=7)
    sd = _t_predictive_sd(pred)
    finite = sd[np.isfinite(sd)]
    assert np.all(finite >= 0)


def test_predictive_bundle_sd_matches_helper(fitted_autoffs):
    """`_run_predict_batch` populates FFSPredictive.sd via the requested
    averaging scheme. The default is the Vincentised (quantile) SD; passing
    ``sd_method="moment"`` recovers the law-of-total-variance helper. Both must
    match their respective helper for every batch."""
    ffs, _ = fitted_autoffs
    from DLMAX.ffs_core import _run_predict_batch, _t_vincent_sd

    for state in ffs._batches:
        # Default: quantile / Vincent SD.
        pred_q = _run_predict_batch(state, h=4)
        np.testing.assert_allclose(
            pred_q.sd, _t_vincent_sd(pred_q).T, equal_nan=True, rtol=1e-12
        )
        # Opt-in: moment / law-of-total-variance SD.
        pred_m = _run_predict_batch(state, h=4, sd_method="moment")
        np.testing.assert_allclose(
            pred_m.sd, _t_predictive_sd(pred_m).T, equal_nan=True, rtol=1e-12
        )


# --- Schema smoke tests: column appears in output dataframes -----------


@pytest.fixture
def fitted_autoffs():
    """Small fitted AutoFFS for schema checks."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    rng = np.random.default_rng(0)
    n_periods, n_series = 36, 3
    rows = []
    for i in range(n_series):
        for t in range(n_periods):
            y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal()
            rows.append({"unique_id": f"s{i}", "ds": t, "y": float(y)})
    df = pd.DataFrame(rows)

    ffs = AutoFFS(
        season_length=12, n_seas_comps=2, alias="DLM", dma_pdr=0.99, dma_mdr=0.99
    )
    ffs.fit(df, freq=1, h_template=6)
    return ffs, df


def test_forecast_emits_sd_column(fitted_autoffs):
    ffs, df = fitted_autoffs
    out = ffs.forecast(df, h=6, freq=1, level=[80, 95])
    assert "DLM-sd" in out.columns
    assert (out["DLM-sd"] >= 0).all()


def test_cross_validation_emits_sd_column(fitted_autoffs):
    ffs, df = fitted_autoffs
    out = ffs.cross_validation(df, h=6, n_windows=2, freq=1, level=[80])
    assert "DLM-sd" in out.columns
    assert (out["DLM-sd"] >= 0).all()

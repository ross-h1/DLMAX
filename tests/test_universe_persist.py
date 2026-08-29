"""Tests for AutoFFSUniverse persistence and the warmup/diffuse prior.

Covers the directory-backed lifecycle that had no coverage before and so
hid several bugs: create -> fit -> open -> update -> add_series ->
forecast. Also pins:

  * ``warmup_steps`` is a universe-level setting, persisted in config and
    restored by ``open`` (selects the diffuse prior over legacy OLS).
  * NaN in ``y`` is permitted (pre-launch / structurally-missing periods,
    skipped by the filter via ignore_obs); only all-NaN series are
    rejected. Leading-NaN runs longer than ``2*periodicity`` must still
    yield a finite forecast (whole-window diffuse elicitation).
  * ``update`` validates temporal continuity (catches a skipped step).
  * ``AutoFFS`` and ``AutoFFSUniverse`` are bit-identical for the same
    data + warmup (shared fit path), on both the diffuse and OLS branches.
"""

import numpy as np
import pandas as pd
import pytest


def _panel(n_series=3, n_obs=60, season=7, leading_nan=None):
    """Daily long-format panel with weekly seasonality.

    ``leading_nan`` maps series index -> number of leading NaN ``y``
    values (a pre-launch period).
    """
    leading_nan = leading_nan or {}
    rng = np.arange(n_obs)
    dates = pd.date_range("2020-01-01", periods=n_obs, freq="D")
    rows = []
    for s in range(n_series):
        y = (
            10.0
            + 0.05 * rng
            + 3.0 * np.sin(2 * np.pi * rng / season)
            + np.random.default_rng(s).normal(0, 0.5, n_obs)
        )
        lead = leading_nan.get(s, 0)
        if lead:
            y[:lead] = np.nan
        rows.append(pd.DataFrame({"unique_id": f"s{s}", "ds": dates, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_warmup_steps_persisted_and_restored(tmp_path):
    """warmup_steps survives a create/fit/open round-trip."""
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "uni")
    uni = AutoFFSUniverse.create(p, season_length=7, warmup_steps=14, max_batch_size=10)
    assert uni.warmup_steps == 14
    uni.fit(_panel(), freq="D", h_template=28)

    reopened = AutoFFSUniverse.open(p)
    assert reopened.warmup_steps == 14
    assert len(reopened.list_series()) == 3


def test_open_update_forecast_lifecycle(tmp_path):
    """Reopen, update one step, add a series, forecast — all finite.

    Exercises the load paths (_load_manifest, _load_batch_state).
    """
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "uni")
    df = _panel(leading_nan={2: 20})  # s2 launches after 2*period (=14)
    AutoFFSUniverse.create(
        p, season_length=7, warmup_steps=14, max_batch_size=10,
        grid_period=None                      # legacy path: continuity check
    ).fit(df, freq="D", h_template=28)

    uni = AutoFFSUniverse.open(p)
    dates = pd.to_datetime(df["ds"].unique())
    nxt = pd.DatetimeIndex([dates.max() + pd.Timedelta(days=1)])
    upd = pd.concat(
        [pd.DataFrame({"unique_id": f"s{s}", "ds": nxt, "y": [9.0]}) for s in range(3)],
        ignore_index=True,
    )
    uni.update(upd)

    # add a late-launching product with its own leading-NaN region
    rng = np.arange(60)
    y = 10.0 + 2.0 * np.sin(2 * np.pi * rng / 7)
    y[:10] = np.nan
    uni.add_series("late", pd.DataFrame({"ds": dates + pd.Timedelta(days=1), "y": y}))

    fc = uni.forecast(h=28, level=[80, 95])
    assert fc["unique_id"].nunique() == 4
    val_cols = [c for c in fc.columns if c not in ("unique_id", "ds")]
    assert np.isfinite(fc[val_cols].to_numpy()).all()


def test_leading_nan_beyond_two_periods_is_finite(tmp_path):
    """A leading-NaN run longer than 2*periodicity still yields a finite
    forecast (whole-window diffuse elicitation)."""
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "uni")
    # 55/60 leading NaN: only 5 real obs, far past 2*period (=14)
    df = _panel(n_series=1, leading_nan={0: 55})
    AutoFFSUniverse.create(
        p, season_length=7, warmup_steps=14, max_batch_size=10,
        grid_period=None                      # legacy path: continuity check
    ).fit(df, freq="D", h_template=28)
    fc = AutoFFSUniverse.open(p).forecast(h=28)
    assert np.isfinite(fc["AutoFFS"].to_numpy()).all()


def test_all_nan_series_rejected(tmp_path):
    """A series with no real observations cannot be fit."""
    from DLMAX.ffs_core import AutoFFSUniverse

    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    bad = pd.DataFrame({"unique_id": "z", "ds": dates, "y": np.full(60, np.nan)})
    with pytest.raises(ValueError, match="no non-NaN observations"):
        AutoFFSUniverse.create(
            str(tmp_path / "uni"), season_length=7, warmup_steps=14
        ).fit(bad, freq="D")


def test_update_rejects_date_gap(tmp_path):
    """update() validates temporal continuity."""
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "uni")
    df = _panel()
    AutoFFSUniverse.create(
        p, season_length=7, warmup_steps=14, max_batch_size=10,
        grid_period=None                      # legacy path: continuity check
    ).fit(df, freq="D", h_template=28)

    uni = AutoFFSUniverse.open(p)
    dates = pd.to_datetime(df["ds"].unique())
    gap = pd.DatetimeIndex([dates.max() + pd.Timedelta(days=5)])  # skips 4 days
    bad = pd.concat(
        [pd.DataFrame({"unique_id": f"s{s}", "ds": gap, "y": [9.0]}) for s in range(3)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="expected"):
        uni.update(bad)


@pytest.mark.parametrize("warmup", [14, None])
def test_universe_matches_autoffs(tmp_path, warmup):
    """AutoFFS and AutoFFSUniverse are bit-identical for the same data,
    across the diffuse/OLS prior paths."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS, AutoFFSUniverse

    # leading NaN only on the diffuse path (OLS is not nan-safe)
    df = _panel(leading_nan={2: 20} if warmup else None)

    a = AutoFFS(season_length=7, max_batch_size=10)
    a.fit(df, freq="D", h_template=28, warmup_steps=warmup)
    fa = a.predict(h=28, level=[80, 95]).sort_values(
        ["unique_id", "ds"]
    ).reset_index(drop=True)

    p = str(tmp_path / "uni")
    u = AutoFFSUniverse.create(
        p, season_length=7, warmup_steps=warmup, max_batch_size=10,
        grid_period=None                      # match StaticFFS above
    )
    u.fit(df, freq="D", h_template=28)
    fu = u.forecast(h=28, level=[80, 95]).sort_values(
        ["unique_id", "ds"]
    ).reset_index(drop=True)

    val_cols = [c for c in fa.columns if c not in ("unique_id", "ds")]
    assert fa.shape == fu.shape
    assert (fa["unique_id"].to_numpy() == fu["unique_id"].to_numpy()).all()
    np.testing.assert_array_equal(fa[val_cols].to_numpy(), fu[val_cols].to_numpy())

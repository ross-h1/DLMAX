"""Tests for integer-ds support in AutoFFS.

Covers: fit, update, cross_validation, predict, forecast — all with
integer-typed `ds` columns (M1/M3-competition style data, where ds is
an observation index rather than a calendar date).
"""
# NB AutoFFS now defaults to the WING grid and implements the CV arm only.
# These tests exercise the legacy static fit/predict machinery (ds/freq
# handling, forecast plumbing), so they target StaticFFS directly.

import numpy as np
import pandas as pd
import pytest


def _make_integer_ds_dataframe(n_series=3, n_obs=36, freq=1):
    """Build an integer-ds DataFrame for AutoFFS."""
    rng = np.random.default_rng(42)
    rows = []
    for s in range(n_series):
        level = 100.0 + 50 * s
        slope = 0.5 - 0.2 * s
        seasonal_amp = 5.0 + 2 * s
        for t in range(n_obs):
            ds = t * freq
            seasonal = seasonal_amp * np.sin(2 * np.pi * t / 12)
            noise = rng.normal(0, 1.0)
            y = level + slope * t + seasonal + noise
            rows.append({"unique_id": f"s{s}", "ds": ds, "y": y})
    return pd.DataFrame(rows)


def test_fit_with_integer_ds():
    """fit() succeeds with integer ds; freq is stored as int."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=3, n_obs=36, freq=1)
    model = AutoFFS(season_length=12).fit(df, freq=1, h_template=12)

    assert model.is_fitted
    assert isinstance(model._freq, (int, np.integer))
    assert int(model._freq) == 1


def test_fit_then_update_integer_ds():
    """fit() then update() works with integer ds (the original M3 bug)."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df_full = _make_integer_ds_dataframe(n_series=3, n_obs=48, freq=1)
    df_train = df_full[df_full["ds"] < 36]
    df_new = df_full[df_full["ds"] >= 36]

    model = AutoFFS(season_length=12).fit(df_train, freq=1, h_template=12)
    model = model.update(df_new)

    last_ds_per_state = [
        max(state.last_ds_per_sid.values()) for state in model._batches
    ]
    assert all(int(ds) == 47 for ds in last_ds_per_state)


def test_fit_freq_inference_integer_ds():
    """When freq is not given, infer from integer-spaced ds."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=3, n_obs=36, freq=2)
    model = AutoFFS(season_length=12).fit(df, h_template=12)

    assert isinstance(model._freq, (int, np.integer))
    assert int(model._freq) == 2


def test_fit_freq_inference_uneven_integer_ds_raises():
    """Uneven integer ds raises a clear error rather than inferring wrong."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=1, n_obs=36, freq=1)
    df = df[df["ds"] != 5]

    with pytest.raises(ValueError, match="not evenly spaced"):
        AutoFFS(season_length=12).fit(df, h_template=12)


def test_update_temporal_gap_integer_ds_raises():
    """If df_new starts before last_ds + freq, update raises clearly."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df_full = _make_integer_ds_dataframe(n_series=3, n_obs=48, freq=1)
    df_train = df_full[df_full["ds"] < 36]
    df_new_bad = df_full[df_full["ds"] >= 38]

    model = AutoFFS(season_length=12).fit(df_train, freq=1, h_template=12)

    with pytest.raises(ValueError, match="first new ds"):
        model.update(df_new_bad)


def test_cross_validation_with_integer_ds_works():
    """cross_validation() works with integer ds (smoke test)."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=3, n_obs=48, freq=1)
    model = AutoFFS(season_length=12).fit(df, freq=1, h_template=12)

    cv_out = model.cross_validation(df, h=6, n_windows=2)

    assert len(cv_out) > 0
    assert {"unique_id", "ds", "cutoff", "y"}.issubset(cv_out.columns)


def test_predict_with_integer_ds_returns_int_ds():
    """predict() with integer-ds fit returns int-typed ds matching arithmetic."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=3, n_obs=36, freq=1)
    model = AutoFFS(season_length=12).fit(df, freq=1, h_template=12)

    out = model.predict(h=6)

    assert "ds" in out.columns
    assert pd.api.types.is_integer_dtype(
        out["ds"]
    ), f"Expected integer ds, got dtype {out['ds'].dtype}"
    for sid, sub in out.groupby("unique_id"):
        sub_sorted = sub.sort_values("ds")
        ds_vals = sub_sorted["ds"].values
        assert list(ds_vals) == [
            36,
            37,
            38,
            39,
            40,
            41,
        ], f"Series {sid}: got ds={list(ds_vals)}"


def test_predict_with_integer_ds_freq_2():
    """predict() with freq=2 produces ds spaced by 2."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=2, n_obs=36, freq=2)
    model = AutoFFS(season_length=12).fit(df, freq=2, h_template=12)

    out = model.predict(h=4)

    assert pd.api.types.is_integer_dtype(out["ds"])
    sub = out[out["unique_id"] == "s0"].sort_values("ds")
    assert list(sub["ds"].values) == [72, 74, 76, 78]


def test_forecast_with_integer_ds_works():
    """forecast() with integer-ds dataframe runs end-to-end."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=2, n_obs=36, freq=1)
    out = AutoFFS(season_length=12).forecast(df, h=6, freq=1)

    assert pd.api.types.is_integer_dtype(out["ds"])
    assert len(out) == 2 * 6


def test_cross_validation_integer_ds_returns_int_ds():
    """cross_validation() preserves integer-typed ds/cutoff.

    Wrapping ds/cutoff in pd.Timestamp unconditionally would mangle integer
    indices to epoch+N-nanoseconds. Integer mode must round-trip ds as
    integers, the way the predict path does.
    """
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=3, n_obs=48, freq=1)  # ds = 0..47
    cv_out = AutoFFS(season_length=12).cross_validation(df, h=6, n_windows=2)

    assert pd.api.types.is_integer_dtype(
        cv_out["ds"]
    ), f"Expected integer ds, got {cv_out['ds'].dtype}"
    assert pd.api.types.is_integer_dtype(
        cv_out["cutoff"]
    ), f"Expected integer cutoff, got {cv_out['cutoff'].dtype}"

    # step_size defaults to h (=6), so windows are non-overlapping:
    # window 0: train_end=42 -> cutoff=41, targets 42..47
    # window 1: train_end=36 -> cutoff=35, targets 36..41
    assert sorted(cv_out["cutoff"].unique().tolist()) == [35, 41]

    latest = cv_out[
        (cv_out["cutoff"] == 41) & (cv_out["unique_id"] == "s0")
    ].sort_values("ds")
    assert latest["ds"].tolist() == [42, 43, 44, 45, 46, 47]
    # forecast targets are strictly after the cutoff (last observed index)
    assert (latest["ds"] > latest["cutoff"]).all()


def test_cross_validation_integer_ds_freq_2():
    """cross_validation() with freq=2 keeps ds spaced by 2 and integer-typed."""
    from DLMAX.ffs_core import StaticFFS as AutoFFS

    df = _make_integer_ds_dataframe(n_series=2, n_obs=48, freq=2)  # ds = 0,2,..,94
    cv_out = AutoFFS(season_length=12).cross_validation(df, h=4, n_windows=1)

    assert pd.api.types.is_integer_dtype(cv_out["ds"])
    assert pd.api.types.is_integer_dtype(cv_out["cutoff"])

    sub = cv_out[cv_out["unique_id"] == "s0"].sort_values("ds")
    # train_end = 48-4 = 44 -> last observed index 43 -> ds = 86; targets 88..94
    assert sub["cutoff"].unique().tolist() == [86]
    assert sub["ds"].tolist() == [88, 90, 92, 94]

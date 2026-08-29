"""``AutoFFS`` forecasts forward, and agrees exactly with ``AutoFFSUniverse``.

``AutoFFS`` has two arms over one model: ``cross_validation`` for
rolling-origin backtests, and ``fit``/``update``/``predict`` for forecasting
forward with the state held in memory. These cover the forward arm, including
the documented quick start.

The central assertion is EQUIVALENCE with the disk-backed streaming face. The
two are the same blocks over the same rows with the same union DMA; one holds
its carry in memory, the other in HDF5. So they must agree BITWISE, and the
tests assert ``rtol=0, atol=0`` rather than a tolerance -- a tolerance here
would hide exactly the kind of drift (a differing warmup gate, a stale weight,
a calendar off-by-one) the pairing exists to catch.
"""
import numpy as np
import pandas as pd
import pytest

from DLMAX import AutoFFS, AutoFFSUniverse

H, SEASON, T = 6, 12, 48


def _panel(n=3, t=T, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=t, freq="MS")
    out = []
    for i in range(n):
        y = (10.0 + i + np.linspace(0, 5, t)
             + 2.0 * np.sin(2 * np.pi * np.arange(t) / SEASON)
             + rng.normal(0, 0.5, t))
        out.append(pd.DataFrame({"unique_id": f"s{i}", "ds": dates, "y": y}))
    return pd.concat(out, ignore_index=True)


def _universe(tmp_path, df, name="u"):
    u = AutoFFSUniverse.create(str(tmp_path / name), season_length=SEASON,
                               warmup=SEASON)
    u.fit(df)
    return u


# --------------------------------------------------------------------------
# the documented entry point must work
# --------------------------------------------------------------------------
def test_fit_predict_runs():
    df = _panel()
    out = AutoFFS(season_length=SEASON, warmup=SEASON).fit(df).predict(h=H)
    assert len(out) == 3 * H
    assert set(out.columns) >= {"unique_id", "ds", "AutoFFS", "AutoFFS-sd"}
    assert np.isfinite(out["AutoFFS"]).all()
    assert (out["AutoFFS-sd"] > 0).all()


def test_predict_levels_are_ordered_and_nested():
    df = _panel()
    out = AutoFFS(season_length=SEASON, warmup=SEASON).fit(df).predict(
        h=H, level=[80, 95])
    assert (out["AutoFFS-lo-95"] <= out["AutoFFS-lo-80"]).all()
    assert (out["AutoFFS-hi-80"] <= out["AutoFFS-hi-95"]).all()
    assert (out["AutoFFS-lo-80"] <= out["AutoFFS"]).all()
    assert (out["AutoFFS"] <= out["AutoFFS-hi-80"]).all()


def test_forecast_is_the_one_shot_form():
    """``forecast(df, h)`` is fit + predict in one call, discarding the state.

    NOT an alias of ``predict``: it takes the data, because the whole point is
    that nothing is held afterwards. It must also run THIS model -- inheriting
    the legacy one-shot implementation would forecast the static universe while
    the caller constructed a wing.
    """
    df = _panel()
    one_shot = AutoFFS(season_length=SEASON, warmup=SEASON).forecast(df, h=H)
    staged = AutoFFS(season_length=SEASON, warmup=SEASON).fit(df).predict(h=H)
    np.testing.assert_allclose(one_shot["AutoFFS"].to_numpy(),
                               staged["AutoFFS"].to_numpy(), rtol=0, atol=0)


def test_forecast_leaves_the_instance_unfitted():
    """The one-shot form works on a scratch copy, so a fitted model is reusable."""
    df = _panel()
    m = AutoFFS(season_length=SEASON, warmup=SEASON)
    m.forecast(df, h=H)
    assert m.is_fitted is False


def test_predict_does_not_move_the_state():
    df = _panel()
    m = AutoFFS(season_length=SEASON, warmup=SEASON).fit(df)
    first, second = m.predict(h=H), m.predict(h=H)
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------
# equivalence with the streaming face
# --------------------------------------------------------------------------
def test_fit_predict_matches_universe_bitwise(tmp_path):
    df = _panel()
    mem = AutoFFS(season_length=SEASON, warmup=SEASON).fit(df).predict(h=H)
    dsk = _universe(tmp_path, df).forecast(h=H)
    key = ["unique_id", "ds"]
    a = mem.sort_values(key).reset_index(drop=True)
    b = dsk.sort_values(key).reset_index(drop=True)
    assert list(a["unique_id"]) == list(b["unique_id"])
    assert list(a["ds"]) == list(b["ds"])
    for col in ("AutoFFS", "AutoFFS-sd"):
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(),
                                   rtol=0, atol=0)


def test_update_then_predict_matches_universe_bitwise(tmp_path):
    """The carry must advance identically too, not just initialise identically."""
    full = _panel(t=T + 4)
    hist = full.groupby("unique_id", group_keys=False).head(T)
    new = full.groupby("unique_id", group_keys=False).tail(4)

    mem = AutoFFS(season_length=SEASON, warmup=SEASON).fit(hist).update(new)
    dsk = _universe(tmp_path, hist)
    dsk.update(new)

    a = mem.predict(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    b = dsk.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    assert list(a["ds"]) == list(b["ds"])          # calendar advanced together
    for col in ("AutoFFS", "AutoFFS-sd"):
        np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(),
                                   rtol=0, atol=0)


def test_update_is_equivalent_to_fitting_the_whole_history(tmp_path):
    """fit(all) == fit(head) + update(tail): the filter is sequential, so a
    split ingest cannot change the answer."""
    full = _panel(t=T + 4)
    hist = full.groupby("unique_id", group_keys=False).head(T)
    new = full.groupby("unique_id", group_keys=False).tail(4)
    one = AutoFFS(season_length=SEASON, warmup=SEASON).fit(full).predict(h=H)
    two = (AutoFFS(season_length=SEASON, warmup=SEASON)
           .fit(hist).update(new).predict(h=H))
    np.testing.assert_allclose(one["AutoFFS"].to_numpy(),
                               two["AutoFFS"].to_numpy(), rtol=0, atol=0)


# --------------------------------------------------------------------------
# guards: fail loudly rather than silently forecasting nonsense
# --------------------------------------------------------------------------
def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="Call fit"):
        AutoFFS(season_length=SEASON).predict(h=H)


def test_update_before_fit_raises():
    with pytest.raises(RuntimeError, match="Call fit"):
        AutoFFS(season_length=SEASON).update(_panel())


def test_ragged_panel_points_at_the_universe():
    df = _panel()
    df = df.drop(df.index[-3:])                     # one series ends early
    with pytest.raises(ValueError, match="AutoFFSUniverse"):
        AutoFFS(season_length=SEASON, warmup=SEASON).fit(df)


def test_update_with_an_unknown_series_raises():
    df = _panel()
    m = AutoFFS(season_length=SEASON, warmup=SEASON).fit(df)
    nxt = _panel(n=4, t=T + 1).groupby("unique_id", group_keys=False).tail(1)
    with pytest.raises(ValueError, match="not fitted"):
        m.update(nxt)


def test_update_missing_a_series_raises():
    df = _panel()
    m = AutoFFS(season_length=SEASON, warmup=SEASON).fit(df)
    nxt = _panel(t=T + 1).groupby("unique_id", group_keys=False).tail(1)
    with pytest.raises(ValueError, match="missing fitted series"):
        m.update(nxt[nxt["unique_id"] != "s2"])


def test_bad_h_raises():
    m = AutoFFS(season_length=SEASON, warmup=SEASON).fit(_panel())
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            m.predict(h=bad)

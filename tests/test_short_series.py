"""Short series warn (non-blocking) instead of being rejected.

The hard minimum-observations gate (2 seasonal cycles) was replaced by a
warning so series shorter than the seasonal period are still fitted /
forecast, leaning on the diffuse prior. ``cross_validation`` keeps only a
genuine hard floor for the CV mechanics (enough observations to form one
forecast origin). Motivated by M4 weekly, where 65/359 series are shorter
than 2 * (365.25/7) ~ 105 weeks.
"""
import warnings

import numpy as np
import pandas as pd
import pytest


def _panel(lengths, season=7):
    dates = pd.date_range("2020-01-01", periods=max(lengths), freq="D")
    rows = []
    for i, n in enumerate(lengths):
        rng = np.arange(n)
        y = (10 + 0.1 * rng + 3 * np.sin(2 * np.pi * rng / season)
             + np.random.default_rng(i).normal(0, 0.3, n))
        rows.append(pd.DataFrame({"unique_id": f"s{i}", "ds": dates[:n], "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_cross_validation_warns_not_raises_on_short_series():
    """A series shorter than 2 seasonal cycles is forecast, with a warning."""
    from DLMAX.ffs_core import AutoFFS

    df = _panel([10, 40])           # s0 short (< 2*7=14), s1 long
    m = AutoFFS(season_length=7)
    with pytest.warns(UserWarning, match="under-determined"):
        cv = m.cross_validation(df, h=3, n_windows=1, warmup_steps=7)

    assert set(cv["unique_id"]) == {"s0", "s1"}
    short = cv[cv.unique_id == "s0"].dropna(subset=["AutoFFS"])
    assert len(short) == 3 and np.isfinite(short["AutoFFS"]).all()


def test_cross_validation_hard_floor_still_raises():
    """Below the CV-mechanics floor (can't form an origin) still errors."""
    from DLMAX.ffs_core import AutoFFS

    df = _panel([3])                # length 3 < floor = h + 1 = 4
    m = AutoFFS(season_length=7)
    with pytest.raises(ValueError, match="forecast origin"):
        m.cross_validation(df, h=3, n_windows=1, warmup_steps=7)


def test_universe_fit_warns_not_raises_on_short_series(tmp_path):
    """AutoFFSUniverse.fit also warns rather than rejecting short series."""
    from DLMAX.ffs_core import AutoFFSUniverse

    df = _panel([10, 40])
    # the under-determined warning is raised by the legacy fit path
    uni = AutoFFSUniverse.create(
        str(tmp_path / "u"), season_length=7, warmup_steps=7, max_batch_size=10,
        grid_period=None)
    with pytest.warns(UserWarning, match="under-determined"):
        uni.fit(df, freq="D", h_template=3)
    assert len(uni.list_series()) == 2


def test_long_series_do_not_warn():
    """No warning when every series has >= 2 seasonal cycles."""
    from DLMAX.ffs_core import AutoFFS

    df = _panel([40, 50])
    m = AutoFFS(season_length=7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # any UserWarning -> failure
        m.cross_validation(df, h=3, n_windows=1, warmup_steps=7)

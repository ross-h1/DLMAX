"""``cross_validation(output="xarray")`` — the n-dimensional CV return.

The long frame repeats keys on every row and every consumer pivots it back to
arrays before writing netCDF. The xarray form is that shape directly. These
tests pin it to the long output so the two can never disagree.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs_core import AutoFFS

pytest.importorskip("xarray")

PERIOD, H = 12, 4


def _panel(n_series=3, L=80, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2015-01-01", periods=L, freq="MS")
    return pd.concat(
        [pd.DataFrame({
            "unique_id": f"s{j}", "ds": ds,
            "y": (100.0 + 2 * j + 0.3 * np.arange(L)
                  + 8.0 * np.sin(2 * np.pi * np.arange(L) / PERIOD)
                  + rng.normal(0, 2.0, L))})
         for j in range(n_series)], ignore_index=True)


def _index_long(lng):
    """Add the (window, h) coordinates the xarray form is keyed on."""
    d = lng.sort_values(["unique_id", "cutoff", "ds"]).copy()
    d["h"] = d.groupby(["unique_id", "cutoff"], sort=False).cumcount()
    d["window"] = (d.groupby("unique_id", sort=False)["cutoff"]
                   .rank(method="dense").astype(int) - 1)
    return d


@pytest.mark.parametrize("n_windows", [1, 3])
def test_xarray_matches_long_cellwise(n_windows):
    """Every long row equals the xarray cell it maps to — including at
    n_windows > 1, where a naive per-series cumcount would mis-shape."""
    df = _panel()
    kw = dict(h=H, n_windows=n_windows)
    lng = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(df, **kw)
    xa = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        df, output="xarray", **kw)

    d = _index_long(lng)
    for var, col in (("loc", "AutoFFS"), ("sd", "AutoFFS-sd"), ("y", "y")):
        got = np.array([xa[var].sel(unique_id=r.unique_id, window=r.window,
                                    h=r.h).item() for r in d.itertuples()])
        np.testing.assert_array_equal(got, d[col].to_numpy())


def test_xarray_shape_and_coords():
    df = _panel()
    xa = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        df, h=H, n_windows=3, output="xarray")

    assert xa["loc"].dims == ("unique_id", "window", "h")
    assert xa.sizes["unique_id"] == 3
    assert xa.sizes["window"] == 3
    assert xa.sizes["h"] == H
    # cutoff is a COORD over (unique_id, window), never a dimension: length
    # groups put window i on different dates.
    assert "cutoff" not in xa.dims
    assert xa["cutoff"].dims == ("unique_id", "window")
    assert xa["ds"].dims == ("unique_id", "window", "h")
    # windows are oldest-first
    for uid in xa["unique_id"].values:
        cut = pd.to_datetime(xa["cutoff"].sel(unique_id=uid).values)
        assert list(cut) == sorted(cut)


def test_xarray_levels_become_a_dimension():
    df = _panel()
    lng = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        df, h=H, n_windows=2, level=[80, 95])
    xa = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        df, h=H, n_windows=2, level=[80, 95], output="xarray")

    assert xa["lo"].dims == ("unique_id", "window", "h", "level")
    assert list(xa["level"].values) == [80, 95]

    d = _index_long(lng)
    for L in (80, 95):
        got = np.array([xa["lo"].sel(unique_id=r.unique_id, window=r.window,
                                     h=r.h, level=L).item()
                        for r in d.itertuples()])
        np.testing.assert_array_equal(got, d[f"AutoFFS-lo-{L}"].to_numpy())


def test_long_remains_the_default():
    df = _panel()
    out = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        df, h=H, n_windows=1)
    assert isinstance(out, pd.DataFrame)


def test_bad_output_value_rejected():
    df = _panel()
    with pytest.raises(ValueError, match="output must be"):
        AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
            df, h=H, n_windows=1, output="wide")


def _mi_panel(L=60):
    idx = pd.date_range("2015-01-01", periods=L, freq="MS")
    mi = pd.MultiIndex.from_tuples(
        [("FOODS", "s0"), ("FOODS", "item_long"), ("HOBBIES", "s2")],
        names=["dept", "item"])
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(100, 5, (L, 3)), index=idx, columns=mi), mi


def test_xarray_restores_the_input_multiindex():
    """A MultiIndex in must come back out: the result is keyed the way the
    input was, not by the internal flattened id."""
    wide, mi = _mi_panel()
    xa = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        wide, h=H, n_windows=1, output="xarray")
    got = xa.indexes["unique_id"]
    assert isinstance(got, pd.MultiIndex)
    assert set(got) == set(mi)
    assert list(got.names) == list(mi.names)
    # and it selects by level, like the input frame would (xarray drops the
    # selected level, so the remaining dim is renamed -- count the rows)
    assert xa["loc"].sel(dept="FOODS").shape[0] == 2


def test_long_gains_the_multiindex_levels_additively():
    """The long form keeps unique_id (nothing breaks) and gains the levels."""
    wide, mi = _mi_panel()
    lng = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        wide, h=H, n_windows=1)
    for nm in mi.names:
        assert nm in lng.columns
    assert "unique_id" in lng.columns
    pairs = set(map(tuple, lng[["dept", "item"]].drop_duplicates().to_numpy()))
    assert pairs == set(mi)


def test_plain_index_input_is_left_alone():
    """No MultiIndex in, no extra columns or coords out."""
    wide, _ = _mi_panel()
    wide.columns = ["a", "b", "c"]
    lng = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        wide, h=H, n_windows=1)
    assert list(lng.columns) == ["unique_id", "ds", "cutoff", "y",
                                 "AutoFFS", "AutoFFS-sd"]
    xa = AutoFFS(season_length=PERIOD, warmup=PERIOD).cross_validation(
        wide, h=H, n_windows=1, output="xarray")
    assert not isinstance(xa.indexes["unique_id"], pd.MultiIndex)

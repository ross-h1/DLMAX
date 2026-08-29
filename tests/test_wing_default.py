"""The WING spec is the default for both orchestrators.

``AutoFFS(season_length=P, warmup=W)`` and
``AutoFFSUniverse.create(path, season_length=P, warmup=W)`` must run the
wing spec with nothing else configured: grid_disc_prior (3, 1, 5, 1),
grid_seasonal_prior 4, grid_learn_dma on, grid_additive_logscore off,
grid_decouple_trend off, grid_offset 1.0, dma 0.9/0.9.

If any of these fail, the two-argument call no longer reproduces the spec the
paper's results were produced under.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.discount_grid import (
    WING_DISC_PRIOR, WING_LEARN_DMA, WING_SEASONAL_PRIOR)
from DLMAX.ffs.grid_block import GridBlock
from DLMAX.ffs.static_block import StaticBlock
from DLMAX.ffs_core import AutoFFS, AutoFFSUniverse

PERIOD, WARMUP, H = 12, 12, 4


def _panel(n_series=3, L=60, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2015-01-01", periods=L, freq="MS")
    out = []
    for j in range(n_series):
        t = np.arange(L)
        y = (100.0 + 5.0 * j + 0.4 * t
             + 8.0 * np.sin(2 * np.pi * t / PERIOD)
             + rng.normal(0, 2.0, L))
        out.append(pd.DataFrame({"unique_id": f"s{j}", "ds": ds, "y": y}))
    return pd.concat(out, ignore_index=True)


def _wide(df):
    return df.pivot(index="ds", columns="unique_id", values="y")


def test_default_is_the_published_wing_spec():
    """The two-argument call equals the explicit wing construction, bit-for-bit.

    The 'explicit' side is exactly what m4_wing_run.py builds.
    """
    df = _panel()
    kw = dict(h=H, n_windows=1)

    grid = GridBlock.build(period=PERIOD, warmup=WARMUP, offset=1.0,
                           pdr=0.9, mdr=0.9, var_powers=None, n_comps=None,
                           additive_logscore=False, disc_prior=WING_DISC_PRIOR,
                           decouple_trend=False, couple_trend=False,
                           couple_sd=None, period2=None, n_comps2=None,
                           seasonal_prior=WING_SEASONAL_PRIOR, adapt_guard=0.5)
    explicit = AutoFFS(blocks=[grid], season_length=PERIOD, dma_pdr=0.9,
                       dma_mdr=0.9, learn_dma=True).cross_validation(
        df, warmup_steps=WARMUP, **kw)
    default = AutoFFS(season_length=PERIOD, warmup=WARMUP).cross_validation(df, **kw)

    k = ["unique_id", "ds"]
    a = explicit.sort_values(k).reset_index(drop=True)
    b = default.sort_values(k).reset_index(drop=True)
    for c in ("AutoFFS", "AutoFFS-sd"):
        np.testing.assert_array_equal(a[c].to_numpy(), b[c].to_numpy())


def test_warmup_constructor_equals_cross_validation_warmup_steps():
    """Both routes to the warmup are equivalent; neither is silently dropped."""
    df = _panel()
    kw = dict(h=H, n_windows=1)
    a = AutoFFS(season_length=PERIOD, warmup=WARMUP).cross_validation(df, **kw)
    b = AutoFFS(season_length=PERIOD).cross_validation(df, warmup_steps=WARMUP, **kw)
    np.testing.assert_array_equal(a["AutoFFS"].to_numpy(), b["AutoFFS"].to_numpy())


def test_wide_and_long_input_agree_on_the_default():
    df = _panel()
    kw = dict(h=H, n_windows=1)
    lng = AutoFFS(season_length=PERIOD, warmup=WARMUP).cross_validation(df, **kw)
    wid = AutoFFS(season_length=PERIOD, warmup=WARMUP).cross_validation(_wide(df), **kw)
    k = ["unique_id", "ds"]
    np.testing.assert_array_equal(
        lng.sort_values(k)["AutoFFS"].to_numpy(),
        wid.sort_values(k)["AutoFFS"].to_numpy())


def test_blocks_is_the_escape_hatch_to_static():
    """A static universe must be reachable, and must NOT be the default."""
    df = _panel()
    kw = dict(h=H, n_windows=1)
    wing = AutoFFS(season_length=PERIOD, warmup=WARMUP).cross_validation(df, **kw)
    static = AutoFFS(blocks=[StaticBlock(season_length=PERIOD, n_seas_comps=2)],
                     learn_dma=False).cross_validation(df, **kw)
    assert not np.allclose(wing["AutoFFS"].to_numpy(), static["AutoFFS"].to_numpy())


def test_use_grid_is_retired_with_an_actionable_message():
    with pytest.raises(TypeError, match="use_grid was removed"):
        AutoFFS(season_length=PERIOD, use_grid=True)


def test_autoffs_has_both_arms():
    """``AutoFFS`` forecasts forward as well as cross-validating.

    Both arms are present on one class: ``cross_validation`` for rolling-origin
    backtests, and ``fit``/``update``/``predict`` for forecasting forward. The
    forward arm agrees with ``AutoFFSUniverse`` bitwise; see
    ``tests/test_autoffs_forecast_face.py``.

    What the universe is FOR is asserted there: state that outlives the
    process, panels too large for memory, and series that arrive later. Both
    routes to it (a ragged panel, an unknown series) raise and name it.
    """
    m = AutoFFS(season_length=PERIOD, warmup=WARMUP)
    for name in ("fit", "update", "predict", "forecast", "cross_validation"):
        assert callable(getattr(m, name))
    assert m.is_fitted is False


def test_universe_default_resolves_to_the_wing_spec(tmp_path):
    """Every knob the archived M5 universe recorded, from two arguments."""
    uni = AutoFFSUniverse(str(tmp_path / "u"), season_length=7, warmup=14)
    assert uni._grid_mode is True
    assert uni.grid_period == 7
    assert uni.grid_warmup == 14
    assert uni.warmup_steps == 14
    assert uni.grid_disc_prior == WING_DISC_PRIOR
    assert uni.grid_seasonal_prior == WING_SEASONAL_PRIOR
    assert uni.grid_learn_dma is WING_LEARN_DMA
    assert uni.grid_additive_logscore is False
    assert uni.grid_decouple_trend is False
    assert uni.grid_offset == 1.0
    assert uni.dma_pdr == 0.9 and uni.dma_mdr == 0.9


def test_universe_explicit_none_period_opts_out_to_legacy(tmp_path):
    """The documented escape hatch: an EXPLICIT None turns the grid off."""
    uni = AutoFFSUniverse(str(tmp_path / "u"), season_length=7, warmup=14,
                          grid_period=None)
    assert uni._grid_mode is False
    assert uni.grid_period is None


def test_universe_wing_is_on_even_without_seasonality(tmp_path):
    """season_length=None (yearly) is still a wing, not a fallback to legacy."""
    uni = AutoFFSUniverse(str(tmp_path / "u"), warmup=4)
    assert uni._grid_mode is True
    assert uni.grid_period is None


def test_exog_with_grid_mode_fails_with_the_fix_in_the_message(tmp_path):
    """A config-route grid universe has no tail, so a provider alone is an error.

    Grid mode carries exog only through a ``block_builder`` — no config field
    can express a tail — so a provider without one is an error. BOTH remedies
    must be in the message: a builder keeps the wing, ``grid_period=None``
    routes to the legacy engine, and someone who wanted the latter should not
    have to discover it elsewhere.
    """
    uni = AutoFFSUniverse(str(tmp_path / "u"), season_length=7, warmup=14,
                          exog_provider=lambda ids, ds: np.zeros((len(ds), len(ids))))
    with pytest.raises(ValueError, match="no regression tail") as ei:
        uni.fit(_panel(), freq="MS")
    assert "block_builder" in str(ei.value)
    assert "grid_period=None" in str(ei.value)

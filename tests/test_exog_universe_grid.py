"""Exogenous regressors through AutoFFSUniverse in GRID (wing) mode.

``test_exog_universe.py`` covers the same journey on the legacy multi path.
Here the concern is the universe-level plumbing: materialising the design at
each of fit / update / forecast / add_series and handing it to the block.

The test that matters is the end-to-end one: a tail that is wired correctly must
LIFT the forecast on future dates where the indicator is on. Everything else here
guards the ways it could be wired wrongly and still look fine — in particular the
two silent no-ops, exog-without-a-tail and tail-without-exog, which produce
plausible numbers and quietly score as something other than what was asked for.

A tail reaches grid mode only through ``block_builder``: no config-tuple field
can express one, by design.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.discount_grid import _grid_model
from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Regressors
from DLMAX.ffs.grid_block import GridBlock
from DLMAX.ffs_core import AutoFFSUniverse

BASE, BETA, N_DAYS, H, WARMUP = 10.0, 5.0, 120, 14, 14


# --- module-level builders/providers: persisted by NAME, re-supplied on open() ---
def exog_grid_blocks(init_data, h, ctx):
    """One block, two var_power families, each LocalTrend + a 1-wide tail.

    Built from components rather than compiled ``Wing`` cells so it depends on
    ``init_data`` not at all — the contract ``_blocks_from_builder`` requires,
    since the same builder is called with ``init_data=None`` on every
    rebuild-to-load.
    """
    grid = []
    for vp in (1.0, 0.25):
        d = DLM(n_series=1)
        d.add_component(LocalTrend(name="trend", disc_rate=0.95, damping=0.9))
        d.add_component(Regressors(name="reg", n_regs=1, disc_rate=0.999))
        d.set_error(disc_rate=0.99, power=vp)
        comps = list(d.components)
        grid.append((f"vp{vp}", _grid_model(comps, vp), tuple(comps)))
    return [GridBlock(grid, period=None, warmup=ctx["warmup"])]


def structural_grid_blocks(init_data, h, ctx):
    """The same shape with NO tail — for the exog-without-a-tail guard."""
    grid = []
    for vp in (1.0, 0.25):
        d = DLM(n_series=1)
        d.add_component(LocalTrend(name="trend", disc_rate=0.95, damping=0.9))
        d.set_error(disc_rate=0.99, power=vp)
        comps = list(d.components)
        grid.append((f"vp{vp}", _grid_model(comps, vp), tuple(comps)))
    return [GridBlock(grid, period=None, warmup=ctx["warmup"])]


def two_exog_blocks(init_data, h, ctx):
    """Two blocks -> the multi-block path, which exog is not wired for."""
    return exog_grid_blocks(init_data, h, ctx) + structural_grid_blocks(
        init_data, h, ctx)


def _x_of(ds):
    """Known indicator: 1 on Mon/Tue, else 0 — a pure function of the date."""
    return (pd.DatetimeIndex(np.asarray(ds)).weekday < 2).astype(float)


def exog_provider(srs_ids, ds):
    """``(T, n_series, 1)`` — the indicator shared across series, like SNAP."""
    x = _x_of(ds)
    return np.broadcast_to(x[:, None, None], (len(x), len(srs_ids), 1)).copy()


def _make_df(series_ids, dates, seed=0):
    rng = np.random.default_rng(seed)
    x = _x_of(dates)
    return pd.concat(
        [pd.DataFrame({"unique_id": s, "ds": dates,
                       "y": BASE + BETA * x + rng.normal(0, 0.4, len(dates))})
         for s in series_ids], ignore_index=True)


def _dates(n=N_DAYS, start="2021-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def _fit(path, n_series=4, n_days=N_DAYS, **kw):
    dates = _dates(n_days)
    df = _make_df([f"s{i}" for i in range(n_series)], dates)
    kw.setdefault("block_builder", exog_grid_blocks)
    kw.setdefault("exog_provider", exog_provider)
    uni = AutoFFSUniverse.create(path, season_length=None, warmup=WARMUP,
                                 max_batch_size=10, **kw)
    uni.fit(df, freq="D", h_template=H)
    return uni, dates


def _lift(fc):
    """Mean forecast on indicator days minus mean on the others."""
    x = _x_of(fc["ds"])
    loc = fc["AutoFFS"].to_numpy()
    return loc[x == 1].mean(), loc[x == 0].mean()


# --- the headline: the tail is wired, end to end -----------------------------

def test_exog_drives_the_grid_forecast(tmp_path):
    """A correctly wired tail lifts the forecast on future x==1 dates.

    This is the end-to-end property, and it fails for every wrong wiring:
    unfilled tail, exog dropped in the worker, wrong axis order in the future
    design, design misaligned to the horizon dates.
    """
    uni, _ = _fit(str(tmp_path / "uni"))
    fc = uni.forecast(h=H).sort_values(["unique_id", "ds"])
    hi, lo = _lift(fc)
    assert hi - lo > 3.0, (hi, lo)          # coefficient learned (~BETA=5)
    assert abs(lo - BASE) < 2.0, lo         # x==0 level tracks BASE, not BASE+BETA


def test_forecast_axis_order_is_not_transposed(tmp_path):
    """Guard on the (h, q, r) -> (q, h, r) swap in ``_grid_predict_batch_file``.

    With a design shared across series a transpose is INVISIBLE whenever h == q,
    so this uses a batch whose series count differs from the horizon and checks
    the lift survives. A wrong axis order would either raise or smear the
    indicator across the wrong horizons.
    """
    uni, _ = _fit(str(tmp_path / "uni"), n_series=3)     # q=3 != H=14
    fc = uni.forecast(h=H).sort_values(["unique_id", "ds"])
    hi, lo = _lift(fc)
    assert hi - lo > 3.0, (hi, lo)


def test_exog_survives_update(tmp_path):
    """Fitting to N then updating equals fitting to N+k in one go.

    The update path materialises exog for the NEW rows only, so an off-by-one or
    a stale window shows up as a mismatch against the one-shot fit.
    """
    all_dates = _dates(N_DAYS)
    ids = [f"s{i}" for i in range(3)]
    df = _make_df(ids, all_dates)
    cut = N_DAYS - 10

    a = AutoFFSUniverse.create(str(tmp_path / "a"), season_length=None,
                               warmup=WARMUP, max_batch_size=10,
                               block_builder=exog_grid_blocks,
                               exog_provider=exog_provider)
    a.fit(df[df["ds"].isin(all_dates[:cut])], freq="D", h_template=H)
    a.update(df[df["ds"].isin(all_dates[cut:])])

    b = AutoFFSUniverse.create(str(tmp_path / "b"), season_length=None,
                               warmup=WARMUP, max_batch_size=10,
                               block_builder=exog_grid_blocks,
                               exog_provider=exog_provider)
    b.fit(df, freq="D", h_template=H)

    fa = a.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    fb = b.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    np.testing.assert_allclose(fa, fb, rtol=1e-9, atol=1e-9)


def test_exog_roundtrips_through_reopen(tmp_path):
    path = str(tmp_path / "uni")
    uni, _ = _fit(path)
    f1 = uni.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    uni2 = AutoFFSUniverse.open(path, block_builder=exog_grid_blocks,
                                exog_provider=exog_provider)
    f2 = uni2.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    np.testing.assert_allclose(f1, f2, rtol=1e-12, atol=1e-12)


def test_exog_add_series(tmp_path):
    """A late launcher is fit alone on its own calendar, so its design is
    materialised per series rather than from the batch's."""
    uni, dates = _fit(str(tmp_path / "uni"), n_series=3)
    hist = _make_df(["late"], dates, seed=7)
    uni.add_series("late", hist)
    fc = uni.forecast(h=H).sort_values(["unique_id", "ds"])
    assert "late" in set(fc["unique_id"])
    hi, lo = _lift(fc[fc["unique_id"] == "late"])
    assert hi - lo > 3.0, (hi, lo)          # the added series carries the tail too


# --- the guards: both silent no-ops, and the unwired multi-block path ---------

def test_exog_without_a_tail_is_rejected(tmp_path):
    """Regressors supplied to a structural block would go nowhere, silently."""
    uni = AutoFFSUniverse.create(str(tmp_path / "u"), season_length=None,
                                 warmup=WARMUP,
                                 block_builder=structural_grid_blocks,
                                 exog_provider=exog_provider)
    with pytest.raises(ValueError, match="no regression tail"):
        uni.fit(_make_df(["s0", "s1"], _dates()), freq="D", h_template=H)


def test_a_tail_without_exog_is_rejected(tmp_path):
    """The mirror image: the tail would filter against a zero row every step,
    costing states and a discount block while contributing nothing."""
    uni = AutoFFSUniverse.create(str(tmp_path / "u"), season_length=None,
                                 warmup=WARMUP, block_builder=exog_grid_blocks)
    with pytest.raises(ValueError, match="no exog_provider"):
        uni.fit(_make_df(["s0", "s1"], _dates()), freq="D", h_template=H)


def test_exog_on_the_multiblock_path_is_rejected(tmp_path):
    """Not silently ignored: which block carries the tail is unresolved."""
    uni = AutoFFSUniverse.create(str(tmp_path / "u"), season_length=None,
                                 warmup=WARMUP, block_builder=two_exog_blocks,
                                 exog_provider=exog_provider)
    with pytest.raises(ValueError, match="MULTI-block"):
        uni.fit(_make_df(["s0", "s1"], _dates()), freq="D", h_template=H)


def test_structural_grid_universe_is_unaffected(tmp_path):
    """No provider, no tail — the ordinary grid universe still fits and
    forecasts, i.e. the new gate does not fire on the common case."""
    uni = AutoFFSUniverse.create(str(tmp_path / "u"), season_length=7,
                                 warmup=WARMUP, max_batch_size=10)
    uni.fit(_make_df(["s0", "s1"], _dates()), freq="D", h_template=H)
    fc = uni.forecast(h=H)
    assert len(fc) == 2 * H and np.isfinite(fc["AutoFFS"]).all()

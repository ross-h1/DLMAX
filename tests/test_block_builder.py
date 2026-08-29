"""``AutoFFSUniverse(block_builder=...)`` — the caller supplies the blocks.

The config knobs (``grid_period``, ``grid_var_powers``, ...) can only express
what someone remembered to add to the spec tuple, and they cannot express a
hand-assembled taxonomy at all (e.g. M4 daily's period-7 SEASONAL-ONLY block).
A builder callable removes that ceiling, on the same contract as
``universe_builder``: the name is persisted, the callable is re-supplied on
``open()``.

The bar is that a builder returning the SAME block the config route builds must
give bit-identical results — through fit, forecast, reopen and update.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.discount_grid import (
    WING_DISC_PRIOR, WING_LEARN_DMA, WING_SEASONAL_PRIOR, build_grid)
from DLMAX.ffs.grid_block import GridBlock
from DLMAX.ffs_core import AutoFFSUniverse

PERIOD, WARMUP, H = 7, 14, 7
VP = [1.0, 0.25]          # the M5 wing's compound-Poisson sweep


def m5_wing_blocks(init_data, h, ctx):
    """The published M5 wing block, built by hand instead of by config."""
    return [GridBlock.build(period=ctx["period"], warmup=ctx["warmup"],
                            var_powers=VP, disc_prior=WING_DISC_PRIOR,
                            seasonal_prior=WING_SEASONAL_PRIOR,
                            learn_dma=WING_LEARN_DMA)]


def seasonal_only_blocks(init_data, h, ctx):
    """A taxonomy the spec tuple cannot express: the seasonal families only."""
    grid = [f for f in build_grid(ctx["period"], var_powers=None)
            if any(getattr(c, "name", "") == "seasonal" for c in f[2])]
    return [GridBlock(grid, period=ctx["period"], warmup=ctx["warmup"])]


def _panel(n=6, L=90, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2020-01-01", periods=L, freq="D")
    return pd.concat(
        [pd.DataFrame({"unique_id": f"s{j}", "ds": ds,
                       "y": (20.0 + j + 3.0 * np.sin(2 * np.pi * np.arange(L) / PERIOD)
                             + rng.normal(0, 1.0, L)).round(1)})
         for j in range(n)], ignore_index=True)


def _fc(uni, h=H):
    return uni.forecast(h=h).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def test_builder_matches_the_config_route_bit_for_bit(tmp_path):
    df = _panel()
    cfg_uni = AutoFFSUniverse.create(
        str(tmp_path / "cfg"), season_length=PERIOD, warmup=WARMUP,
        grid_var_powers=VP)
    cfg_uni.fit(df, freq="D", h_template=H)

    bld_uni = AutoFFSUniverse.create(
        str(tmp_path / "bld"), season_length=PERIOD, warmup=WARMUP,
        block_builder=m5_wing_blocks)
    bld_uni.fit(df, freq="D", h_template=H)

    a, b = _fc(cfg_uni), _fc(bld_uni)
    np.testing.assert_array_equal(a["AutoFFS"].to_numpy(), b["AutoFFS"].to_numpy())
    np.testing.assert_array_equal(a["AutoFFS-sd"].to_numpy(),
                                  b["AutoFFS-sd"].to_numpy())


def test_reopen_requires_the_builder_back(tmp_path):
    p = str(tmp_path / "u")
    AutoFFSUniverse.create(p, season_length=PERIOD, warmup=WARMUP,
                           block_builder=m5_wing_blocks).fit(
        _panel(), freq="D", h_template=H)

    # silently falling back to the config block would run a DIFFERENT model
    with pytest.raises(ValueError, match="re-supply block_builder"):
        AutoFFSUniverse.open(p)

    uni = AutoFFSUniverse.open(p, block_builder=m5_wing_blocks)
    assert uni.block_builder is m5_wing_blocks
    assert len(_fc(uni)) == 6 * H


def test_reopen_with_a_different_builder_warns(tmp_path):
    p = str(tmp_path / "u")
    AutoFFSUniverse.create(p, season_length=PERIOD, warmup=WARMUP,
                           block_builder=m5_wing_blocks).fit(
        _panel(), freq="D", h_template=H)
    with pytest.warns(UserWarning, match="block_builder mismatch"):
        AutoFFSUniverse.open(p, block_builder=seasonal_only_blocks)


def test_update_through_a_reopen_matches_the_config_route(tmp_path):
    """The M5 shape: fit, then reopen-and-advance one origin at a time."""
    df = _panel(L=90)
    head, tail = df[df.ds < "2020-03-25"], df[df.ds >= "2020-03-25"]

    pc, pb = str(tmp_path / "cfg"), str(tmp_path / "bld")
    AutoFFSUniverse.create(pc, season_length=PERIOD, warmup=WARMUP,
                           grid_var_powers=VP).fit(head, freq="D", h_template=H)
    AutoFFSUniverse.create(pb, season_length=PERIOD, warmup=WARMUP,
                           block_builder=m5_wing_blocks).fit(head, freq="D",
                                                             h_template=H)
    for day in sorted(tail.ds.unique()):
        rows = tail[tail.ds == day]
        AutoFFSUniverse.open(pc).update(rows)
        AutoFFSUniverse.open(pb, block_builder=m5_wing_blocks).update(rows)

    a = _fc(AutoFFSUniverse.open(pc))
    b = _fc(AutoFFSUniverse.open(pb, block_builder=m5_wing_blocks))
    np.testing.assert_array_equal(a["AutoFFS"].to_numpy(), b["AutoFFS"].to_numpy())


def test_builder_can_express_what_the_spec_tuple_cannot(tmp_path):
    """A filtered taxonomy runs end to end — no config route reaches this."""
    uni = AutoFFSUniverse.create(str(tmp_path / "u"), season_length=PERIOD,
                                 warmup=WARMUP, block_builder=seasonal_only_blocks)
    uni.fit(_panel(), freq="D", h_template=H)
    fc = _fc(uni)
    assert len(fc) == 6 * H
    assert np.isfinite(fc["AutoFFS"].to_numpy()).all()


# --- the builder needs the data: AdaptiveBlock cells are compiled from it -----

def adaptive_blocks(init_data, h, ctx):
    """An ``AdaptiveBlock`` over hand-compiled Wing cells — the case a
    ``builder(ctx)`` signature could not express at all, because
    ``DLM.compile`` will not compile without ``init_data``. Two structural
    cells (additive / multiplicative error), each with an RTRL-learned level
    discount.

    Note what the contract requires: the cells are compiled at whatever width
    ``init_data`` happens to have (1 on a rebuild), because the block's series
    axis is sized from the real data at ``scan_filter``, not from here.
    """
    from DLMAX.ffs.dlm_builder import DLM, Fourier, LocalTrend, Wing
    from DLMAX.ffs.grid_block import AdaptiveBlock

    frame = (pd.DataFrame(np.ones((max(ctx["warmup"], 2), 1)))
             if init_data is None else init_data)
    cells = []
    for power in (1.0, 0.0):
        c = DLM(family="Gaussian", n_series=frame.shape[1])
        c.add_component(LocalTrend(name="leveltrend",
                                   disc_rate=Wing(0.99, offset=1.0),
                                   damping=0.99))
        c.add_component(Fourier(name="seas", period=ctx["period"], n_comps=2,
                                disc_rate=0.999))
        c.set_error(disc_rate=0.999, power=power)
        cells.append(c.compile(init_data=frame, warmup_steps=ctx["warmup"], h=h))
    return [AdaptiveBlock.from_cells(cells, warmup=ctx["warmup"], offset=1.0)]


def test_adaptive_block_builder_runs_end_to_end(tmp_path):
    """Fit, forecast, reopen and update with a builder that MUST have the data.

    The rebuild legs (open/update/forecast) get ``init_data=None`` and have to
    come out structurally identical, or loading the persisted carry fails.
    """
    df = _panel(L=90)
    head, tail = df[df.ds < "2020-03-25"], df[df.ds >= "2020-03-25"]
    p = str(tmp_path / "adapt")

    uni = AutoFFSUniverse.create(p, season_length=PERIOD, warmup=WARMUP,
                                 block_builder=adaptive_blocks)
    uni.fit(head, freq="D", h_template=H)
    first = _fc(AutoFFSUniverse.open(p, block_builder=adaptive_blocks))
    assert len(first) == 6 * H
    assert np.isfinite(first["AutoFFS"].to_numpy()).all()
    assert np.isfinite(first["AutoFFS-sd"].to_numpy()).all()

    for day in sorted(tail.ds.unique())[:3]:
        rows = tail[tail.ds == day]
        AutoFFSUniverse.open(p, block_builder=adaptive_blocks).update(rows)

    after = _fc(AutoFFSUniverse.open(p, block_builder=adaptive_blocks))
    assert np.isfinite(after["AutoFFS"].to_numpy()).all()
    # the carry actually advanced — a silently-rebuilt-from-scratch block would
    # forecast the same thing three days running
    assert not np.array_equal(first["AutoFFS"].to_numpy(),
                              after["AutoFFS"].to_numpy())


def test_builder_gets_the_fit_window_and_none_on_rebuild(tmp_path):
    """The contract itself: data on the fit leg, None on every rebuild leg."""
    seen = []

    def recording_blocks(init_data, h, ctx):
        seen.append(None if init_data is None else init_data.shape)
        return m5_wing_blocks(init_data, h, ctx)

    p = str(tmp_path / "rec")
    uni = AutoFFSUniverse.create(p, season_length=PERIOD, warmup=WARMUP,
                                 block_builder=recording_blocks)
    assert seen == [None]                      # construction-time block count
    df = _panel(n=3, L=60)
    uni.fit(df, freq="D", h_template=H)
    assert seen[-1] == (60, 3)                 # the batch's fit window

    seen.clear()
    AutoFFSUniverse.open(p, block_builder=recording_blocks).forecast(h=H)
    assert set(seen) == {None}


def test_non_block_return_is_rejected(tmp_path):
    def bad(init_data, h, ctx):
        return ["not a block"]

    with pytest.raises(TypeError, match="no 'scan_filter'"):
        AutoFFSUniverse(str(tmp_path / "u"), season_length=PERIOD, warmup=WARMUP,
                        block_builder=bad)


# --- the spec tuple's own seasonal_prior gap ---------------------------------

def test_multiblock_spec_carries_seasonal_prior(tmp_path):
    """grid_blocks= must reach seasonal_prior, and inherit the universe's.

    It was absent from the spec tuple, so a MULTI-block universe silently got
    None while the single-block path got the WING value -- the same spec
    expressed two ways giving two different models.
    """
    from DLMAX.ffs.discount_grid import WING_SEASONAL_PRIOR

    gb = [dict(period=PERIOD, warmup=WARMUP), dict(period=PERIOD, warmup=WARMUP)]
    uni = AutoFFSUniverse(str(tmp_path / "u"), season_length=PERIOD,
                          warmup=WARMUP, grid_blocks=gb)
    assert uni._multiblock
    # inherited from the universe (which defaults to the wing spec)
    assert all(spec[12] == WING_SEASONAL_PRIOR for spec in uni._block_specs)

    # and an explicit per-block value wins
    gb2 = [dict(period=PERIOD, warmup=WARMUP, seasonal_prior=2.0),
           dict(period=PERIOD, warmup=WARMUP)]
    uni2 = AutoFFSUniverse(str(tmp_path / "u2"), season_length=PERIOD,
                           warmup=WARMUP, grid_blocks=gb2)
    assert uni2._block_specs[0][12] == 2.0
    assert uni2._block_specs[1][12] == WING_SEASONAL_PRIOR


def test_short_legacy_spec_still_rebuilds():
    """A spec persisted before the field existed is 12 long and must rebuild as
    it ran -- seasonal_prior None, not the new default."""
    from DLMAX.ffs_core import _grid_blocks_from_cfg

    legacy = (PERIOD, None, WARMUP, 1.0, 0.9, 0.9, False, False, None,
              False, False, None)              # 12 fields, no seasonal_prior
    blocks, _p, _m, _c = _grid_blocks_from_cfg(((legacy, legacy), 0.9, 0.9, None))
    assert all(b.seasonal_prior is None for b in blocks)


def test_universe_builder_opts_out_of_the_wing_default(tmp_path):
    """A custom universe_builder must NOT be silently replaced by the wing.

    grid mode consumes GridBlocks and never calls universe_builder, so
    defaulting to the wing when one is supplied would run a different model
    than the caller asked for -- and say nothing. (The ENTSOE study regressed
    exactly this way.)
    """
    def my_universe(init_data, h, ctx):      # never called here; identity matters
        raise AssertionError("should not be invoked by construction")

    uni = AutoFFSUniverse(str(tmp_path / "ub"), season_length=168,
                          warmup_steps=100, universe_builder=my_universe)
    assert uni._grid_mode is False
    assert uni.grid_period is None
    assert uni.universe_builder is my_universe

    # an EXPLICIT grid_period still wins -- the opt-out is only for the default
    grid = AutoFFSUniverse(str(tmp_path / "g"), season_length=7, warmup=14,
                           grid_period=7, universe_builder=my_universe)
    assert grid._grid_mode is True

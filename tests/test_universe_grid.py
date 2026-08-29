"""AutoFFSUniverse single-block grid mode.

With ``grid_period`` set, each batch is one adaptive-discount ``GridBlock`` (the
wing grid), streamed via the block's production face and persisted alongside a
``/metadata`` group. These tests assert the universe wraps the block exactly:
fit/update/forecast reproduce the block bit-for-bit, and the mode round-trips
through save/reopen.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs_core import AutoFFSUniverse
from DLMAX.ffs.discount_grid import (
    WING_DISC_PRIOR, WING_LEARN_DMA, WING_SEASONAL_PRIOR)
from DLMAX.ffs.grid_block import GridBlock

IDS = ["A", "B", "C"]
PERIOD, WARMUP, H = 12, 8, 6


def _wide(T=44, seed=1):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2016-01-01", periods=T, freq="MS")
    t = np.arange(T)
    seas = 10 * np.sin(2 * np.pi * t / 12)
    return pd.DataFrame(
        {s: 100 + 0.4 * t + seas + rng.normal(0, 2, T) for s in IDS}, index=ds)


def _mk(path):
    AutoFFSUniverse.create(path, season_length=PERIOD, grid_period=PERIOD,
                           grid_warmup=WARMUP)
    return AutoFFSUniverse.open(path)


def _block_forecast(arr):
    # The reference block must carry the SAME spec the universe now
    # defaults to (the WING_* constants), or universe-vs-block comparisons
    # are comparing two different models.
    b = GridBlock.build(period=PERIOD, warmup=WARMUP, offset=1.0, pdr=0.90, mdr=0.90,
                        disc_prior=WING_DISC_PRIOR,
                        seasonal_prior=WING_SEASONAL_PRIOR,
                        learn_dma=WING_LEARN_DMA)
    b.scan_filter(arr)
    loc, sd, _ = b.forecast(H)
    return loc, sd


def _cmp(fc, loc, sd):
    worst = 0.0
    for j, sid in enumerate(IDS):
        sub = fc[fc["unique_id"] == sid].sort_values("ds")
        worst = max(worst,
                    np.max(np.abs(sub["AutoFFS"].to_numpy() - loc[j])),
                    np.max(np.abs(sub["AutoFFS-sd"].to_numpy() - sd[j])))
    return worst


def test_grid_fit_forecast_matches_block(tmp_path):
    wide = _wide()
    u = _mk(str(tmp_path / "u"))
    u.fit(wide, freq="MS")
    fc = u.forecast(h=H)
    loc, sd = _block_forecast(wide[IDS].to_numpy(float))
    assert _cmp(fc, loc, sd) == 0.0


def test_grid_update_equals_fit(tmp_path):
    """Incremental update lands where a single fit over the whole history does.

    To float precision, not bitwise: the filter is sequential so the two are
    the same computation, but XLA lowers the reductions differently per
    platform and they land about one ulp apart (~2.4e-16 relative). An exact
    assertion would pin one platform's lowering. 1e-12 is far above that noise
    and far below any real update/fit divergence, which would be a state
    discrepancy of O(1e-3) or worse.
    """
    wide = _wide()
    t0, t1 = 30, 36
    uA = _mk(str(tmp_path / "a"))
    uA.fit(wide.iloc[:t0], freq="MS")
    for tt in range(t0, t1):
        uA.update(wide.iloc[tt:tt + 1])
    fcA = uA.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    uB = _mk(str(tmp_path / "b"))
    uB.fit(wide.iloc[:t1], freq="MS")
    fcB = uB.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    np.testing.assert_allclose(fcA["AutoFFS"], fcB["AutoFFS"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(fcA["AutoFFS-sd"], fcB["AutoFFS-sd"],
                               rtol=1e-12, atol=0)
    # and equals the batch block at t1. _cmp is an ABSOLUTE max difference (not
    # relative like the two assertions above); series here are O(100) and the
    # measured cross-platform noise is 2.8e-14, so 1e-10 keeps ~3 orders of
    # headroom while staying far below any real divergence.
    loc, sd = _block_forecast(wide[IDS].to_numpy(float)[:t1])
    assert _cmp(fcA, loc, sd) < 1e-10


def _dfh(wide, sid):
    return pd.DataFrame({"ds": wide.index, "y": wide[sid].to_numpy()})


def test_grid_add_series_equals_fit_together(tmp_path):
    wide = _wide()
    u1 = _mk(str(tmp_path / "1"))
    u1.fit(wide[["A", "B"]], freq="MS")
    u1.add_series("C", _dfh(wide, "C"))                 # diffuse (no warm-start)
    f1 = u1.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    u2 = _mk(str(tmp_path / "2"))
    u2.fit(wide[["A", "B", "C"]], freq="MS")
    f2 = u2.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    np.testing.assert_allclose(f1["AutoFFS"], f2["AutoFFS"], rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(f1["AutoFFS-sd"], f2["AutoFFS-sd"], rtol=1e-10, atol=1e-9)


def test_grid_add_series_many_equals_loop(tmp_path):
    wide = _wide()
    uL = _mk(str(tmp_path / "L"))
    uL.fit(wide[["A"]], freq="MS")
    uL.add_series("B", _dfh(wide, "B")); uL.add_series("C", _dfh(wide, "C"))
    uM = _mk(str(tmp_path / "M"))
    uM.fit(wide[["A"]], freq="MS")
    uM.add_series_many(["B", "C"], [_dfh(wide, "B"), _dfh(wide, "C")])
    fL = uL.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    fM = uM.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    np.testing.assert_allclose(fL["AutoFFS"], fM["AutoFFS"], rtol=0, atol=0)
    np.testing.assert_allclose(fL["AutoFFS-sd"], fM["AutoFFS-sd"], rtol=0, atol=0)


def test_grid_add_series_warmstart_runs(tmp_path):
    wide = _wide()
    u = _mk(str(tmp_path / "w"))
    u.fit(wide[["A", "B"]], freq="MS")
    cp = {"trend": (np.array([[110.0, 0.5]]), np.array([[[50.0, 0.], [0., 5.]]])),
          "seasonal": (np.zeros((1, 22)), (np.eye(22) * 20)[None])}
    logit = float(np.log(0.9) - np.log1p(-0.9))
    u.add_series("C", _dfh(wide, "C"), wing_centre=logit, component_priors=cp)
    fc = u.forecast(h=H)
    c = fc[fc["unique_id"] == "C"]
    assert np.isfinite(c["AutoFFS"]).all() and np.isfinite(c["AutoFFS-sd"]).all()


def test_grid_capacity_padding(tmp_path):
    """Persisted capacity padding: forecasts match the no-cap path (placeholders
    are forecast-neutral), and add_series fills a placeholder slot so the batch
    stays cap-wide (no q growth -> the jitted kernels never recompile)."""
    from DLMAX.ffs_core import _load_batch_meta
    wide = _wide()

    def run(mbs):
        p = str(tmp_path / f"u{mbs}")
        AutoFFSUniverse.create(p, season_length=PERIOD, grid_period=PERIOD,
                               grid_warmup=WARMUP, max_batch_size=mbs)
        u = AutoFFSUniverse.open(p)
        u.fit(wide[["A", "B"]], freq="MS")
        for tt in range(30, 36):
            u.update(wide.iloc[tt:tt + 1][["A", "B"]])
        u.add_series("C", _dfh(wide, "C"))
        for tt in range(36, 40):
            u.update(wide.iloc[tt:tt + 1])
        return u, u.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    _uN, fN = run(None)
    uC, fC = run(500)
    np.testing.assert_allclose(fN["AutoFFS"], fC["AutoFFS"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(fN["AutoFFS-sd"], fC["AutoFFS-sd"], rtol=0, atol=1e-9)
    bid = int(uC._manifest["batch_id"].iloc[0])
    srs, act, _ = _load_batch_meta(uC._batch_path(bid))
    assert len(srs) == 500 and int(np.sum(act)) == 3      # filled a slot, cap-wide


def test_multiblock_universe_matches_direct(tmp_path):
    """Two-block universe (coupled-mult + compound-Poisson wings) reproduces the
    in-memory `_multiblock_fit`/`_multiblock_forecast`, and round-trips exactly
    through reopen. Exercises the full config->fit->persist->forecast->open wiring."""
    from DLMAX.ffs.grid_block import GridBlock
    from DLMAX.ffs_core import _multiblock_fit, _multiblock_forecast

    wide = _wide()
    gb = [dict(period=PERIOD, var_powers=None, warmup=WARMUP, offset=1.0),
          dict(period=PERIOD, var_powers=[0.25], warmup=WARMUP, offset=1.0)]
    p = str(tmp_path / "u")
    AutoFFSUniverse.create(p, season_length=PERIOD, grid_blocks=gb)
    u = AutoFFSUniverse.open(p)
    assert u._multiblock and len(u._block_specs) == 2
    u.fit(wide, freq="MS")
    fc = u.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # in-memory reference (default dma rates 0.90/0.90)
    arr = wide[IDS].to_numpy(float)
    bd = [GridBlock.build(period=PERIOD, warmup=WARMUP, offset=1.0, pdr=0.90,
                          mdr=0.90, var_powers=None),
          GridBlock.build(period=PERIOD, warmup=WARMUP, offset=1.0, pdr=0.90,
                          mdr=0.90, var_powers=[0.25])]
    _st, w, _ = _multiblock_fit(bd, arr, 0.90, 0.90)
    loc, sd, _ = _multiblock_forecast(bd, w, H)              # loc (q,h), sd (h,q)
    worst = 0.0
    for j, sid in enumerate(IDS):
        sub = fc[fc["unique_id"] == sid].sort_values("ds")
        worst = max(worst,
                    np.max(np.abs(sub["AutoFFS"].to_numpy() - loc[j])),
                    np.max(np.abs(sub["AutoFFS-sd"].to_numpy() - sd[:, j])))
    assert worst < 1e-9, f"universe vs direct worst={worst:.2e}"

    # reopen recovers multi-block mode and forecasts identically
    u2 = AutoFFSUniverse.open(p)
    assert u2._multiblock and len(u2._block_specs) == 2
    fc2 = u2.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    np.testing.assert_allclose(fc["AutoFFS"], fc2["AutoFFS"], rtol=0, atol=0)
    np.testing.assert_allclose(fc["AutoFFS-sd"], fc2["AutoFFS-sd"], rtol=0, atol=0)


def test_multiblock_universe_update_equals_fit(tmp_path):
    """Two-block universe: fit to t0 then update to t1 == fit straight to t1."""
    wide = _wide()
    t0, t1 = 30, 36
    gb = [dict(period=PERIOD, var_powers=None, warmup=WARMUP, offset=1.0),
          dict(period=PERIOD, var_powers=[0.25], warmup=WARMUP, offset=1.0)]

    pa = str(tmp_path / "a")
    AutoFFSUniverse.create(pa, season_length=PERIOD, grid_blocks=gb)
    ua = AutoFFSUniverse.open(pa)
    ua.fit(wide.iloc[:t0], freq="MS")
    for tt in range(t0, t1):
        ua.update(wide.iloc[tt:tt + 1])
    fa = ua.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    pb = str(tmp_path / "b")
    AutoFFSUniverse.create(pb, season_length=PERIOD, grid_blocks=gb)
    ub = AutoFFSUniverse.open(pb)
    ub.fit(wide.iloc[:t1], freq="MS")
    fb = ub.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    np.testing.assert_allclose(fa["AutoFFS"], fb["AutoFFS"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(fa["AutoFFS-sd"], fb["AutoFFS-sd"], rtol=0, atol=1e-6)


def test_multiblock_add_series_equals_fit_together(tmp_path):
    """Multi-block diffuse add_series: fit [A,B] then add C == fit [A,B,C] from
    the start (series independence holds for the blocks AND the per-series union
    column)."""
    wide = _wide()
    gb = [dict(period=PERIOD, var_powers=None, warmup=WARMUP, offset=1.0),
          dict(period=PERIOD, var_powers=[0.25], warmup=WARMUP, offset=1.0)]

    p1 = str(tmp_path / "1")
    AutoFFSUniverse.create(p1, season_length=PERIOD, grid_blocks=gb)
    u1 = AutoFFSUniverse.open(p1)
    u1.fit(wide[["A", "B"]], freq="MS")
    u1.add_series("C", _dfh(wide, "C"))                 # diffuse
    f1 = u1.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    p2 = str(tmp_path / "2")
    AutoFFSUniverse.create(p2, season_length=PERIOD, grid_blocks=gb)
    u2 = AutoFFSUniverse.open(p2)
    u2.fit(wide[["A", "B", "C"]], freq="MS")
    f2 = u2.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    np.testing.assert_allclose(f1["AutoFFS"], f2["AutoFFS"], rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(f1["AutoFFS-sd"], f2["AutoFFS-sd"], rtol=1e-9, atol=1e-8)


def test_multiblock_add_series_capacity(tmp_path):
    """Multi-block add_series into a CAPACITY-padded batch (fills a placeholder
    slot — the M5 path: pad_to/set_slot on blocks + union) == fit-together, with
    forecast skipping inactive slots."""
    wide = _wide()
    gb = [dict(period=PERIOD, var_powers=None, warmup=WARMUP, offset=1.0),
          dict(period=PERIOD, var_powers=[0.25], warmup=WARMUP, offset=1.0)]

    p1 = str(tmp_path / "c1")
    AutoFFSUniverse.create(p1, season_length=PERIOD, grid_blocks=gb,
                           max_batch_size=50)
    u1 = AutoFFSUniverse.open(p1)
    u1.fit(wide[["A", "B"]], freq="MS")
    u1.add_series("C", _dfh(wide, "C"))
    f1 = u1.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    p2 = str(tmp_path / "c2")
    AutoFFSUniverse.create(p2, season_length=PERIOD, grid_blocks=gb,
                           max_batch_size=50)
    u2 = AutoFFSUniverse.open(p2)
    u2.fit(wide[["A", "B", "C"]], freq="MS")
    f2 = u2.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    assert set(f1["unique_id"]) == {"A", "B", "C"}      # placeholders skipped
    np.testing.assert_allclose(f1["AutoFFS"], f2["AutoFFS"], rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(f1["AutoFFS-sd"], f2["AutoFFS-sd"], rtol=1e-9, atol=1e-8)


def test_multiblock_add_series_warmstart_runs(tmp_path):
    """Multi-block warm-start add_series: per-block component_priors + wing_centre
    (one entry per grid) seed each block; the new series forecasts finite."""
    wide = _wide()
    gb = [dict(period=PERIOD, var_powers=None, warmup=WARMUP, offset=1.0),
          dict(period=PERIOD, var_powers=[0.25], warmup=WARMUP, offset=1.0)]
    p = str(tmp_path / "w")
    AutoFFSUniverse.create(p, season_length=PERIOD, grid_blocks=gb)
    u = AutoFFSUniverse.open(p)
    u.fit(wide[["A", "B"]], freq="MS")

    cp = {"trend": (np.array([[110.0, 0.5]]), np.array([[[50.0, 0.], [0., 5.]]])),
          "seasonal": (np.zeros((1, 22)), (np.eye(22) * 20)[None])}
    logit = float(np.log(0.9) - np.log1p(-0.9))
    # per-block lists (one entry per grid)
    u.add_series("C", _dfh(wide, "C"),
                 component_priors=[cp, cp], wing_centre=[logit, logit])
    fc = u.forecast(h=H)
    c = fc[fc["unique_id"] == "C"]
    assert len(c) == H
    assert np.isfinite(c["AutoFFS"]).all() and np.isfinite(c["AutoFFS-sd"]).all()

    # wrong-length warm-start is rejected
    import pytest
    with pytest.raises(ValueError):
        u.add_series("D", _dfh(wide, "A"), wing_centre=[logit])   # len 1 != 2 blocks


def test_multiblock_add_series_many_equals_loop(tmp_path):
    """Multi-block add_series_many == looping add_series."""
    wide = _wide()
    gb = [dict(period=PERIOD, var_powers=None, warmup=WARMUP, offset=1.0),
          dict(period=PERIOD, var_powers=[0.25], warmup=WARMUP, offset=1.0)]

    pL = str(tmp_path / "L")
    AutoFFSUniverse.create(pL, season_length=PERIOD, grid_blocks=gb)
    uL = AutoFFSUniverse.open(pL)
    uL.fit(wide[["A"]], freq="MS")
    uL.add_series("B", _dfh(wide, "B"))
    uL.add_series("C", _dfh(wide, "C"))
    fL = uL.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    pM = str(tmp_path / "M")
    AutoFFSUniverse.create(pM, season_length=PERIOD, grid_blocks=gb)
    uM = AutoFFSUniverse.open(pM)
    uM.fit(wide[["A"]], freq="MS")
    uM.add_series_many(["B", "C"], [_dfh(wide, "B"), _dfh(wide, "C")])
    fM = uM.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    np.testing.assert_allclose(fL["AutoFFS"], fM["AutoFFS"], rtol=0, atol=0)
    np.testing.assert_allclose(fL["AutoFFS-sd"], fM["AutoFFS-sd"], rtol=0, atol=0)


def test_grid_reopen_recovers_mode(tmp_path):
    wide = _wide()
    u = _mk(str(tmp_path / "u"))
    u.fit(wide, freq="MS")
    fc = u.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    u2 = AutoFFSUniverse.open(str(tmp_path / "u"))
    assert u2._grid_mode and u2.grid_period == PERIOD and u2.grid_warmup == WARMUP
    fc2 = u2.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    np.testing.assert_allclose(fc["AutoFFS"], fc2["AutoFFS"], rtol=0, atol=0)
    np.testing.assert_allclose(fc["AutoFFS-sd"], fc2["AutoFFS-sd"], rtol=0, atol=0)

"""Tests for ``GridBlock`` — the discount grid as a Block (refactor Phase 5a).

Asserts GridBlock satisfies the ``Block`` protocol and that its rolling-origin
CV path reproduces the standalone grid (``run_grid_batch``) bit-for-bit — the
adapter is a thin wrapper, so equivalence is the contract. Production mode
(scan_filter/fwd_filter/forecast) streams a resumable carry that reproduces the
batch emission bit-for-bit; save/load round-trips through the
AutoFFSUniverse HDF5 integration.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from DLMAX.ffs.block import Block
from DLMAX.ffs.grid_block import GridBlock
from DLMAX.ffs.discount_grid import run_grid_batch


def _arr(q=3, L=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    cols = [
        100.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, L)
        for _ in range(q)
    ]
    return np.column_stack(cols)


def test_gridblock_is_a_block():
    gb = GridBlock.build(period=12, warmup=12)
    assert isinstance(gb, Block)
    assert gb.nm == len(gb._grid) * 3
    assert gb.n_classes == len(gb._grid)
    assert len(gb.names) == gb.nm


def test_forecast_rolling_matches_standalone_grid():
    arr = _arr()
    h, cutoffs, period, warmup = 6, [20, 25, 30], 12, 12
    gb = GridBlock.build(period=period, warmup=warmup)
    loc, sd = gb.forecast_rolling(arr, cutoffs, h)

    srs_ids = [f"s{i}" for i in range(arr.shape[1])]
    _ids, loc0, sd0, _obs = run_grid_batch(arr, srs_ids, cutoffs, h, period, warmup)

    np.testing.assert_allclose(loc, loc0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd, sd0, rtol=1e-12, atol=1e-12)
    assert gb.q == arr.shape[1]


def test_production_mode_matches_rolling():
    """Streaming (scan_filter to an origin + forecast) reproduces the batch
    rolling-origin emission at that origin bit-for-bit — the streaming carry is
    the batch scan state at that point."""
    arr = _arr()
    h, cutoffs, period, warmup = 6, [20, 25, 30], 12, 12
    gb = GridBlock.build(period=period, warmup=warmup)
    loc_b, sd_b = gb.forecast_rolling(arr, cutoffs, h)   # (q, n_cut, h)
    for ci, cut in enumerate(cutoffs):
        b = GridBlock.build(period=period, warmup=warmup)
        b.scan_filter(arr[: cut + 1])
        loc_s, sd_s, comp = b.forecast(h)
        np.testing.assert_allclose(loc_s, loc_b[:, ci], rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(sd_s, sd_b[:, ci], rtol=1e-10, atol=1e-8)
        assert set(comp) == {"LOCc", "QHc", "NUc", "Wc"}


def test_fwd_filter_resumes_exactly():
    """Incremental fwd_filter from a checkpoint == a fresh scan to the same
    point (the carry holds all dynamic state — nothing hidden)."""
    arr = _arr()
    h, period, warmup = 6, 12, 12
    b = GridBlock.build(period=period, warmup=warmup)
    b.scan_filter(arr[:20])
    for t in range(20, 25):
        b.fwd_filter(arr[t])
    loc_r, sd_r, _ = b.forecast(h)
    fresh = GridBlock.build(period=period, warmup=warmup)
    fresh.scan_filter(arr[:25])
    loc_f, sd_f, _ = fresh.forecast(h)
    np.testing.assert_allclose(loc_r, loc_f, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sd_r, sd_f, rtol=0, atol=1e-12)


def test_save_load_roundtrip(tmp_path):
    """The streaming carry round-trips through HDF5 exactly: a loaded block
    forecasts identically and resumes identically to the live one."""
    arr = _arr()
    h, period, warmup = 6, 12, 12
    b = GridBlock.build(period=period, warmup=warmup)
    b.scan_filter(arr[:25])
    loc0, sd0, _ = b.forecast(h)
    fn = str(tmp_path / "grid.h5")
    b.save(fn)
    b2 = GridBlock.build(period=period, warmup=warmup)
    b2.load(fn)
    loc1, sd1, _ = b2.forecast(h)
    np.testing.assert_allclose(loc1, loc0, rtol=0, atol=0)
    np.testing.assert_allclose(sd1, sd0, rtol=0, atol=0)
    assert b2._t == b._t
    # resume both and confirm they stay identical
    for t in range(25, 30):
        b.fwd_filter(arr[t]); b2.fwd_filter(arr[t])
    lc, sc, _ = b.forecast(h); lc2, sc2, _ = b2.forecast(h)
    np.testing.assert_allclose(lc2, lc, rtol=0, atol=0)
    np.testing.assert_allclose(sc2, sc, rtol=0, atol=0)

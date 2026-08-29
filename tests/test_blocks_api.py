"""The ``AutoFFS(blocks=[...])`` constructor — the block-list API (Phase 4).

Construct blocks yourself and feed a list to ``AutoFFS``; it drives them and
combines their per-model predictives with one union DMA. A single ``StaticBlock``
reproduces Legacy (via the union path, ~float precision); ``[StaticBlock,
GridBlock]`` is the grid mode expressed as an explicit block list.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd

from DLMAX.ffs_core import AutoFFS, StaticFFS
from DLMAX.ffs.static_block import StaticBlock
from DLMAX.ffs.grid_block import GridBlock


def _long_df(n_series=3, L=48, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    frames = []
    for i in range(n_series):
        level, slope, amp = 100.0 + 20 * i, 0.3, 8.0 + 2 * i
        y = level + slope * t + amp * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.0, L)
        frames.append(pd.DataFrame({"unique_id": f"s{i}", "ds": t, "y": y}))
    return pd.concat(frames, ignore_index=True)


def test_blocks_api_derives_season_length():
    m = AutoFFS(blocks=[StaticBlock(season_length=12, n_seas_comps=2, warmup=4)])
    assert m.season_length == 12
    # explicit override wins
    m2 = AutoFFS(season_length=7, blocks=[StaticBlock(season_length=12, n_seas_comps=2)])
    assert m2.season_length == 7


def test_blocks_static_reproduces_legacy():
    df = _long_df()
    kw = dict(h=4, n_windows=2, freq=1)
    legacy = StaticFFS(season_length=12, n_seas_comps=2).cross_validation(
        df, warmup_steps=4, **kw)
    blk = AutoFFS(
        blocks=[StaticBlock(season_length=12, n_seas_comps=2, warmup=4)]
    ).cross_validation(df, **kw)  # block carries warmup

    # single static block routes through the union path -> ~1e-9, not bit-exact
    pd.testing.assert_frame_equal(
        blk.reset_index(drop=True), legacy.reset_index(drop=True),
        check_exact=False, rtol=1e-7, atol=1e-7,
    )


def test_blocks_static_plus_grid_runs():
    df = _long_df(n_series=3, L=48)
    kw = dict(h=4, n_windows=2, freq=1)
    legacy = StaticFFS(season_length=12, n_seas_comps=2).cross_validation(
        df, warmup_steps=4, **kw)
    blk = AutoFFS(blocks=[
        StaticBlock(season_length=12, n_seas_comps=2, warmup=4),
        GridBlock.build(period=12, warmup=4),
    ]).cross_validation(df, **kw)

    assert list(blk.columns) == list(legacy.columns)
    assert len(blk) == len(legacy)
    assert np.all(np.isfinite(blk["AutoFFS"].values))
    assert np.all(np.isfinite(blk["AutoFFS-sd"].values))
    assert not np.allclose(blk["AutoFFS"].values, legacy["AutoFFS"].values)

"""Tests for ``StaticBlock`` and the canonical CV routing (refactor Phase 5b-i).

The key milestone: canonical ``AutoFFS.cross_validation`` routes per-batch
trajectory production through a ``StaticBlock`` and must be **bit-null-diff** vs
``StaticFFS.cross_validation`` for the single static block — proving the
block seam is transparent before the grid is added (5b-ii).
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.block import Block
from DLMAX.ffs.grid_block import GridBlock
from DLMAX.ffs.static_block import StaticBlock
from DLMAX.ffs_core import AutoFFS, StaticFFS


def _long_df(n_series=3, L=48, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    frames = []
    for i in range(n_series):
        level, slope, amp = 100.0 + 20 * i, 0.3, 8.0 + 2 * i
        y = level + slope * t + amp * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.0, L)
        frames.append(pd.DataFrame({"unique_id": f"s{i}", "ds": t, "y": y}))
    return pd.concat(frames, ignore_index=True)


def test_static_block_is_a_block():
    sb = StaticBlock(season_length=12, n_seas_comps=2)
    assert isinstance(sb, Block)


def _assert_cv_identical(**cv_kwargs):
    df = _long_df(seed=cv_kwargs.pop("_seed", 0))
    legacy = StaticFFS(season_length=12, n_seas_comps=2).cross_validation(df, **cv_kwargs)
    # AutoFFS defaults to the WING grid now, so the static universe is
    # requested through the blocks interface; learn_dma=False gives the
    # fixed DMA replay the legacy in-scan combine performs.
    canon = AutoFFS(
        blocks=[StaticBlock(season_length=12, n_seas_comps=2)],
        learn_dma=False,
    ).cross_validation(df, **cv_kwargs)
    pd.testing.assert_frame_equal(
        canon.reset_index(drop=True),
        legacy.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_canonical_cv_null_diff_vs_legacy():
    _assert_cv_identical(h=4, n_windows=3, freq=1)


def test_canonical_cv_null_diff_with_levels():
    _assert_cv_identical(h=4, n_windows=2, freq=1, level=[80, 95], _seed=1)


def test_production_methods_stubbed():
    sb = StaticBlock(season_length=12, n_seas_comps=2)
    for name in ("scan_filter", "fwd_filter", "forecast"):
        with pytest.raises(NotImplementedError):
            getattr(sb, name)(None)


def test_canonical_cv_grid_mode_runs_and_differs():
    # [StaticBlock, GridBlock] driven through the union DMA. It
    # must produce a well-formed, finite CV frame that DIFFERS from Legacy
    # (the grid genuinely changes the combination — the point of the refactor).
    df = _long_df(n_series=3, L=48)
    kw = dict(h=4, n_windows=2, freq=1)
    legacy = StaticFFS(season_length=12, n_seas_comps=2).cross_validation(df, **kw)
    grid = AutoFFS(
        blocks=[StaticBlock(season_length=12, n_seas_comps=2),
                GridBlock.build(period=12, warmup=24)],
        season_length=12,
    ).cross_validation(df, **kw)

    assert list(grid.columns) == list(legacy.columns)
    assert len(grid) == len(legacy)
    assert np.all(np.isfinite(grid["AutoFFS"].values))
    assert np.all(np.isfinite(grid["AutoFFS-sd"].values))
    # the grid moves the numbers vs static-only
    assert not np.allclose(grid["AutoFFS"].values, legacy["AutoFFS"].values)

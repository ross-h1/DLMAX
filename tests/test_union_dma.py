"""Union-DMA-over-blocks machinery (block refactor Phase 5b-ii).

The critical validation: ``_union_combine_cv`` running a post-pass union
Allocator over a *single* block's captured one-step trace reproduces the
legacy *in-scan* combine (loc/sd) to ~float precision — the proof that the
union path degenerates correctly, so adding a second block (the grid) is the
only thing that moves the numbers. Plus a two-block smoke test.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from DLMAX.ffs.static_block import StaticBlock
from DLMAX.ffs_core import (
    FFSPredictive,
    _combined_predictive_sd,
    _union_combine_cv,
    _union_dma_weights,
)


def _arr(q=3, T=48, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    cols = [
        100.0 + 20 * i + 0.3 * t + (8.0 + 2 * i) * np.sin(2 * np.pi * t / 12)
        + rng.normal(0, 1.0, T)
        for i in range(q)
    ]
    return np.column_stack(cols)


def test_union_single_block_reproduces_inscan_combine():
    arr = _arr()
    srs_ids = [f"s{i}" for i in range(arr.shape[1])]
    cutoffs = np.array([30, 40], dtype=np.int32)
    h = 6
    sb = StaticBlock(season_length=12, n_seas_comps=2)  # dma_pdr=mdr=0.90
    traj = sb.forecast_rolling(srs_ids, arr, cutoffs, h, capture_trace=True)

    union = _union_combine_cv(
        [traj], arr, dma_pdr=0.90, dma_mdr=0.90, level=None, sd_method="quantile"
    )

    for gi in range(len(cutoffs)):
        # in-scan combine at this cutoff (from the trajectory's own DMA weights)
        pred = FFSPredictive(
            loc=None, sd=None,
            f_h=traj.f_h[gi], q_h=traj.q_h[gi], nu=traj.nu[gi],
            weights=traj.weights[gi],
        )
        loc_inscan = (traj.f_h[gi] * traj.weights[gi][..., None]).sum(axis=0)
        sd_inscan = _combined_predictive_sd(pred, "quantile")

        loc_union, sd_union, _ = union[gi]
        np.testing.assert_allclose(loc_union, loc_inscan, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(sd_union, sd_inscan, rtol=1e-9, atol=1e-9)


def test_union_two_blocks_finite_and_normalised():
    arr = _arr(seed=1)
    srs_ids = [f"s{i}" for i in range(arr.shape[1])]
    cutoffs = np.array([30, 40], dtype=np.int32)
    h = 6
    a = StaticBlock(season_length=12, n_seas_comps=2)
    b = StaticBlock(season_length=12, n_seas_comps=1)
    ta = a.forecast_rolling(srs_ids, arr, cutoffs, h, capture_trace=True)
    tb = b.forecast_rolling(srs_ids, arr, cutoffs, h, capture_trace=True)

    # union weights over the concatenated model axis sum to 1 per (step, series)
    F1 = np.concatenate([ta.f1_full, tb.f1_full], axis=1)
    Q1 = np.concatenate([ta.q1_full, tb.q1_full], axis=1)
    from DLMAX.ffs_core import _block_diag_mi

    mi = _block_diag_mi([ta.model_indicator, tb.model_indicator])
    W = _union_dma_weights(F1, Q1, arr, mi, 0.90, 0.90)
    assert W.shape == (arr.shape[0], a.nm + b.nm, arr.shape[1])
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-8, atol=1e-8)

    union = _union_combine_cv(
        [ta, tb], arr, dma_pdr=0.90, dma_mdr=0.90, level=None, sd_method="quantile"
    )
    for loc, sd, _ in union:
        assert np.all(np.isfinite(loc))
        assert np.all(np.isfinite(sd))
        assert loc.shape == (arr.shape[1], h)


def test_union_static_plus_grid_finite():
    from DLMAX.ffs.grid_block import GridBlock

    arr = _arr(q=3, T=48)
    srs_ids = [f"s{i}" for i in range(arr.shape[1])]
    cutoffs = np.array([24, 30], dtype=np.int32)  # < L-h = 42
    h = 6
    static = StaticBlock(season_length=12, n_seas_comps=2)
    grid = GridBlock.build(period=12, warmup=12)

    st = static.forecast_rolling(srs_ids, arr, cutoffs, h, capture_trace=True)  # T=48
    gt = grid.cv_trajectory(srs_ids, arr, cutoffs, h)                           # T=42

    union = _union_combine_cv(
        [st, gt], arr, dma_pdr=0.90, dma_mdr=0.90, level=None, sd_method="quantile"
    )
    assert len(union) == len(cutoffs)
    for loc, sd, _ in union:
        assert loc.shape == (arr.shape[1], h)
        assert np.all(np.isfinite(loc))
        assert np.all(np.isfinite(sd))

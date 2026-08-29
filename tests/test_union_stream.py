"""Phase 1 (multi-block AutoFFSUniverse): the streaming union DMA carry.

`_union_allocator` builds a live `Allocator` over the block-diagonal union of
per-block indicators, holding a persistable `AllocatorState` advanced one origin
at a time. This must reproduce the from-t=0 replay `_union_dma_weights` (the
existing CV post-pass driver) to float precision — same allocator, incremental
carry vs. full scan. That equivalence is what lets the persisted universe carry
+ resume the union weights instead of replaying.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from DLMAX.ffs_core import (ForecastBundle, _block_diag_mi, _union_dma_weights,
                            _union_allocator)


def test_streaming_union_allocator_matches_replay():
    rng = np.random.default_rng(0)
    T, n_series = 40, 6
    mi1 = np.array([[1, 0], [1, 0], [0, 1]], bool)     # block A: 3 workers, 2 classes
    mi2 = np.array([[1], [1], [1]], bool)              # block B: 3 workers, 1 class
    union_mi = _block_diag_mi([mi1, mi2])              # (6, 3)
    M = int(union_mi.shape[0])
    F1 = rng.normal(0.0, 1.0, (T, M, n_series))
    Q1 = rng.uniform(0.4, 2.0, (T, M, n_series))
    obs = rng.normal(0.0, 1.0, (T, n_series))
    pdr, mdr = 0.99, 0.95

    # replay from t=0 (the existing union driver)
    W_replay = _union_dma_weights(F1, Q1, obs, union_mi, pdr, mdr)   # (T, M, n_series)

    # live allocator, advanced one origin at a time (the streaming carry)
    dma, umi = _union_allocator([mi1, mi2], n_series, pdr, mdr)
    assert np.array_equal(umi, union_mi)
    step = dma.prepared_step()
    state = dma.state
    W_stream = np.zeros((T, M, n_series))
    for t in range(T):
        fc = ForecastBundle(jnp.asarray(F1[t])[..., None], jnp.asarray(Q1[t])[..., None])
        state, w = step(state, fc, jnp.asarray(obs[t]))
        W_stream[t] = np.asarray(w)[..., 0]

    mad = float(np.max(np.abs(W_replay - W_stream)))
    assert mad < 1e-11, f"streaming union != replay: max|Δ|={mad:.2e}"


def test_block_diag_indicator_shapes():
    mi1 = np.ones((6, 2), bool)
    mi2 = np.ones((3, 1), bool)
    u = _block_diag_mi([mi1, mi2])
    assert u.shape == (9, 3)
    assert u[:6, :2].all() and u[6:, 2:].all()
    assert not u[:6, 2:].any() and not u[6:, :2].any()


def test_scan_filter_trace_matches_cv_trajectory():
    """`scan_filter(return_trace=True)` emits the same per-worker one-step trace
    as `cv_trajectory`'s f1_full/q1_full (incl. the warmup origins) — so the union
    carry built by driving over the streaming trace reproduces the CV union
    replay over the fit window, not just the post-warmup tail."""
    from DLMAX.ffs.grid_block import GridBlock

    rng = np.random.default_rng(1)
    L, q, h, period, warmup = 44, 4, 6, 12, 8
    t = np.arange(L)
    arr = (100 + 0.4 * t[:, None] + 10 * np.sin(2 * np.pi * t[:, None] / period)
           + rng.normal(0, 2, (L, q)))
    cutoffs = np.array([22, 30])

    ba = GridBlock.build(period=period, warmup=warmup, offset=1.0, pdr=0.9, mdr=0.9)
    traj = ba.cv_trajectory(list(range(q)), arr, cutoffs, h)      # f1_full (L-h, M, q)

    bb = GridBlock.build(period=period, warmup=warmup, offset=1.0, pdr=0.9, mdr=0.9)
    _b, (F, Q) = bb.scan_filter(arr[:L - h], return_trace=True)   # F (L-h, q, M)
    f1_stream = np.transpose(F, (0, 2, 1))                        # -> (T, M, q)
    q1_stream = np.transpose(Q, (0, 2, 1))

    assert np.allclose(traj.f1_full, f1_stream, atol=1e-8, rtol=0), \
        f"f1 max|Δ|={np.max(np.abs(traj.f1_full - f1_stream)):.2e}"
    assert np.allclose(traj.q1_full, q1_stream, atol=1e-8, rtol=0), \
        f"q1 max|Δ|={np.max(np.abs(traj.q1_full - q1_stream)):.2e}"


def test_multiblock_fit_forecast_matches_cv_union():
    """The streaming multi-block fit+forecast (`_multiblock_fit` builds the union
    carry over the window, `_multiblock_forecast` combines under its weights)
    reproduces the CV post-pass replay (`_union_combine_cv`) — for TWO grids of
    different period, at a mid-window origin."""
    from DLMAX.ffs.grid_block import GridBlock
    from DLMAX.ffs_core import (_multiblock_fit, _multiblock_forecast,
                                _union_combine_cv)

    rng = np.random.default_rng(2)
    L, q, h, pA, pB, warmup = 48, 4, 6, 12, 6, 8
    t = np.arange(L)
    arr = (100 + 0.3 * t[:, None] + 8 * np.sin(2 * np.pi * t[:, None] / pA)
           + rng.normal(0, 2, (L, q)))
    t0, pdr, mdr = 30, 0.9, 0.9

    def mk(p):
        return GridBlock.build(period=p, warmup=warmup, offset=1.0, pdr=0.9, mdr=0.9)

    # reference: per-block CV trajectory + union replay combine at cutoff t0
    tA = mk(pA).cv_trajectory(list(range(q)), arr, np.array([t0]), h)
    tB = mk(pB).cv_trajectory(list(range(q)), arr, np.array([t0]), h)
    ref_loc, ref_sd, _ = _union_combine_cv([tA, tB], arr, pdr, mdr,
                                           level=None, sd_method="quantile")[0]

    # streaming: fit two fresh blocks + union to t0, then combine-forecast
    blocks = [mk(pA), mk(pB)]
    _state, weights, _umi = _multiblock_fit(blocks, arr[:t0 + 1], pdr, mdr)
    loc, sd, _ = _multiblock_forecast(blocks, weights, h,
                                      level=None, sd_method="quantile")

    assert np.allclose(loc, ref_loc, atol=1e-7, rtol=0), \
        f"loc max|Δ|={np.max(np.abs(loc - ref_loc)):.2e}"
    assert np.allclose(sd, ref_sd, atol=1e-7, rtol=0), \
        f"sd max|Δ|={np.max(np.abs(sd - ref_sd)):.2e}"


def test_multiblock_workers_roundtrip(tmp_path):
    """Persistence + workers: (a) fit_batch_file->save->load->predict reproduces
    the in-memory `_multiblock_fit`/`_multiblock_forecast` exactly; (b) fit to t0
    then update to t1 == fit straight to t1 (streaming update self-consistency)."""
    from DLMAX.ffs.grid_block import GridBlock
    from DLMAX.ffs_core import (_multiblock_fit_batch_file,
                                _multiblock_predict_batch_file,
                                _multiblock_update_batch_file,
                                _multiblock_fit, _multiblock_forecast)

    rng = np.random.default_rng(3)
    L, q, h, pA, pB, warmup = 46, 3, 6, 12, 6, 8
    t = np.arange(L)
    arr = (100 + 0.3 * t[:, None] + 8 * np.sin(2 * np.pi * t[:, None] / pA)
           + rng.normal(0, 2, (L, q)))
    specA = (pA, None, warmup, 1.0, 0.9, 0.9)      # coupled add+mult
    specB = (pB, [0.25], warmup, 1.0, 0.9, 0.9)    # additive compound-Poisson
    cfg = ((specA, specB), 0.9, 0.9, None)         # capacity None
    t0, t1 = 24, 34
    srs = [f"s{j}" for j in range(q)]

    p1 = str(tmp_path / "b1.h5")
    _multiblock_fit_batch_file(p1, arr[:t1], srs, [t1 - 1] * q, True, cfg)
    loc1, sd1, *_ = _multiblock_predict_batch_file(p1, h, cfg)

    # (a) persistence round-trip == in-memory direct
    bd = [GridBlock.build(period=pA, warmup=warmup, offset=1.0, pdr=0.9, mdr=0.9,
                          var_powers=None),
          GridBlock.build(period=pB, warmup=warmup, offset=1.0, pdr=0.9, mdr=0.9,
                          var_powers=[0.25])]
    _st, w, _ = _multiblock_fit(bd, arr[:t1], 0.9, 0.9)
    locd, sdd, _ = _multiblock_forecast(bd, w, h)
    np.testing.assert_allclose(loc1, locd, atol=1e-9, rtol=0)
    np.testing.assert_allclose(sd1, sdd.T, atol=1e-9, rtol=0)   # worker returns (q, h)

    # (b) fit(t0) + update(t0->t1) == fit(t1)
    p2 = str(tmp_path / "b2.h5")
    _multiblock_fit_batch_file(p2, arr[:t0], srs, [t0 - 1] * q, True, cfg)
    _multiblock_update_batch_file(p2, arr[t0:t1], [t1 - 1] * q, True, cfg)
    loc2, sd2, *_ = _multiblock_predict_batch_file(p2, h, cfg)
    np.testing.assert_allclose(loc1, loc2, atol=1e-6, rtol=0)
    np.testing.assert_allclose(sd1, sd2, atol=1e-6, rtol=0)

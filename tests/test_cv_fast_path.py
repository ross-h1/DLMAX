"""n_windows==1 fast path in cross_validation matches the emit-scan bitwise.

Single-origin CV doesn't need the (T, nm, q, h) forecast trajectory that the
emit-scan stacks — it filters to the lone cutoff and forecasts once. This is
the memory fix for long-T / large-h frequencies (e.g. M4 hourly, where the
stacked trajectory hit ~60-100 GB per batch). The forecast at the cutoff must
be identical to the emit path; ``_run_cv_batch`` selects the fast path when
``capture_trace`` is False and there is a single cutoff, so we drive both
paths directly and compare.
"""
import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs_core import _run_cv_batch, AutoFFS


def _panel_array(n_series=3, T=40, season=7):
    rng = np.arange(T)
    return np.column_stack([
        10 + 0.1 * rng + 3 * np.sin(2 * np.pi * rng / season)
        + np.random.default_rng(s).normal(0, 0.3, T)
        for s in range(n_series)
    ])


@pytest.mark.parametrize("periodicity,h", [(7, 6), (12, 4)])
def test_single_origin_fast_path_matches_emit_bitwise(periodicity, h):
    """Fast path (capture_trace=False, one cutoff) == emit-scan, to float precision.

    The claim is that the two paths are the SAME COMPUTATION. Float arithmetic
    realises that to within rounding rather than bitwise: XLA lowers the
    reductions differently per platform, and the two land about one ulp apart
    (~2.2e-16 relative). An exact assertion would pin one platform's lowering
    rather than the property. 1e-12 is four orders above that noise and far
    below any real divergence between these paths.
    """
    arr = _panel_array(n_series=3, T=40, season=periodicity)
    cutoff = np.array([arr.shape[0] - h - 1], dtype=np.int32)  # single origin
    common = dict(
        periodicity=periodicity, n_seas_comps=None, h=h,
        dma_pdr=0.95, dma_mdr=0.75, warmup_steps=periodicity,
    )
    srs = ("a", "b", "c")
    fast = _run_cv_batch(srs, arr, cutoff, capture_trace=False, **common)
    emit = _run_cv_batch(srs, arr, cutoff, capture_trace=True, **common)

    for field in ("f_h", "q_h", "nu", "weights"):
        np.testing.assert_allclose(
            np.asarray(getattr(fast, field)), np.asarray(getattr(emit, field)),
            rtol=1e-12, atol=0,
            err_msg=f"{field} differs between fast and emit paths",
        )
    # integer index: exact equality is the right assertion here
    np.testing.assert_array_equal(fast.cutoff_t_idx, emit.cutoff_t_idx)


def _panel_df(n_series=3, T=40, season=7):
    dates = pd.date_range("2020-01-01", periods=T, freq="D")
    rng = np.arange(T)
    rows = []
    for s in range(n_series):
        y = (10 + 0.1 * rng + 3 * np.sin(2 * np.pi * rng / season)
             + np.random.default_rng(s).normal(0, 0.3, T))
        rows.append(pd.DataFrame({"unique_id": f"s{s}", "ds": dates, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_cross_validation_n_windows_1_end_to_end():
    """Public cross_validation with n_windows=1 (fast path) runs and forecasts."""
    df = _panel_df()
    cv = AutoFFS(season_length=7).cross_validation(df, h=6, n_windows=1, warmup_steps=7)
    assert set(cv["unique_id"]) == {"s0", "s1", "s2"}
    assert len(cv) == 3 * 6  # 3 series x h
    assert np.isfinite(cv["AutoFFS"].to_numpy()).all()


def test_cross_validation_n_windows_2_still_uses_emit_path():
    """n_windows>1 keeps the emit-scan path and produces two origins/series."""
    df = _panel_df(T=60)
    cv = AutoFFS(season_length=7).cross_validation(df, h=6, n_windows=2, warmup_steps=7)
    assert cv.groupby("unique_id")["cutoff"].nunique().eq(2).all()
    assert np.isfinite(cv["AutoFFS"].to_numpy()).all()

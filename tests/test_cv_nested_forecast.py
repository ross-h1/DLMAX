"""CV nested-fq forecast kernel == flat kernel (equivalence).

The cross-validation h-step predictive (:func:`DLMAX.ffs_core._hstep_predictive`)
was switched from the FLAT forecast vmap (``_jvdlm_uv_fcast_H``, replicated
``(nm*q, ...)`` operands, full-dict output) to the NESTED-fq vmap
(``_jvdlm_uv_fcast_H_nested_fq``, model-only operands shared across series,
``(f, q)``-only output) for a ~34% speed / lower-memory win. The flat body is
retained as ``_hstep_predictive_flat`` behind the module flag
``DLMAX.ffs_core._CV_USE_NESTED`` so the bit-exact M1M reference path still
exists.

This module drives ``_run_cv_batch`` (the shared CV forecast entry point) with
the flag False (flat reference) and True (nested) for BOTH the n_windows==1
fast path AND the n_windows==2 emit-scan, and asserts the returned trajectories
match to rtol=atol=1e-12. The ~1e-14 batch-ordering float noise the two kernels
differ by is well inside that bound; the tolerance is NOT loosened.
"""
import numpy as np
import pandas as pd
import pytest

import DLMAX.ffs_core as fc
from DLMAX.ffs_core import _run_cv_batch, AutoFFS


def _panel_array(n_series=3, T=40, season=7):
    rng = np.arange(T)
    return np.column_stack([
        10 + 0.1 * rng + 3 * np.sin(2 * np.pi * rng / season)
        + np.random.default_rng(s).normal(0, 0.3, T)
        for s in range(n_series)
    ])


def _panel_df(n_series=3, T=40, season=7):
    dates = pd.date_range("2020-01-01", periods=T, freq="D")
    rng = np.arange(T)
    rows = []
    for s in range(n_series):
        y = (10 + 0.1 * rng + 3 * np.sin(2 * np.pi * rng / season)
             + np.random.default_rng(s).normal(0, 0.3, T))
        rows.append(pd.DataFrame({"unique_id": f"s{s}", "ds": dates, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _run_with_flag(flag, srs, arr, cutoff, common, capture_trace):
    """Run _run_cv_batch under the given _CV_USE_NESTED setting, restoring it."""
    saved = fc._CV_USE_NESTED
    try:
        fc._CV_USE_NESTED = flag
        return _run_cv_batch(srs, arr, cutoff, capture_trace=capture_trace, **common)
    finally:
        fc._CV_USE_NESTED = saved


def _assert_traj_close(ref, new, ctx):
    for field in ("f_h", "q_h", "nu", "weights"):
        a = np.asarray(getattr(ref, field))
        b = np.asarray(getattr(new, field))
        assert a.shape == b.shape, f"{ctx}: {field} shape {a.shape} vs {b.shape}"
        assert np.isfinite(b).all(), f"{ctx}: nested {field} not all finite"
        np.testing.assert_allclose(
            b, a, rtol=1e-12, atol=1e-12,
            err_msg=f"{ctx}: nested {field} != flat {field}",
        )
    np.testing.assert_array_equal(
        np.asarray(new.cutoff_t_idx), np.asarray(ref.cutoff_t_idx),
        err_msg=f"{ctx}: cutoff_t_idx differs",
    )


@pytest.mark.parametrize("periodicity,h", [(7, 6), (12, 4)])
def test_fast_path_nested_matches_flat(periodicity, h):
    """n_windows==1 fast path: nested kernel == flat kernel to 1e-12."""
    arr = _panel_array(n_series=3, T=40, season=periodicity)
    cutoff = np.array([arr.shape[0] - h - 1], dtype=np.int32)  # single origin
    common = dict(
        periodicity=periodicity, n_seas_comps=None, h=h,
        dma_pdr=0.95, dma_mdr=0.75, warmup_steps=periodicity,
    )
    srs = ("a", "b", "c")
    ref = _run_with_flag(False, srs, arr, cutoff, common, capture_trace=False)
    new = _run_with_flag(True, srs, arr, cutoff, common, capture_trace=False)
    _assert_traj_close(ref, new, ctx=f"fast-path p={periodicity} h={h}")


@pytest.mark.parametrize("periodicity,h", [(7, 6), (12, 4)])
def test_emit_scan_nested_matches_flat(periodicity, h):
    """n_windows==2 emit-scan: nested kernel == flat kernel to 1e-12.

    Two cutoffs force the emit-scan (lax.scan + lax.cond) path; capture_trace
    also forces it. Both cutoffs are compared.
    """
    T = 60
    arr = _panel_array(n_series=3, T=T, season=periodicity)
    step = 1
    cutoffs = np.array(
        [T - h - 1 - step, T - h - 1], dtype=np.int32
    )  # two origins -> emit scan
    common = dict(
        periodicity=periodicity, n_seas_comps=None, h=h,
        dma_pdr=0.95, dma_mdr=0.75, warmup_steps=periodicity,
    )
    srs = ("a", "b", "c")
    ref = _run_with_flag(False, srs, arr, cutoffs, common, capture_trace=False)
    new = _run_with_flag(True, srs, arr, cutoffs, common, capture_trace=False)
    assert np.asarray(new.f_h).shape[0] == 2  # two origins emitted
    _assert_traj_close(ref, new, ctx=f"emit-scan p={periodicity} h={h}")


def test_cross_validation_public_api_nested_matches_flat():
    """End-to-end cross_validation forecasts agree for n_windows 1 and 2."""
    for n_windows, T in [(1, 40), (2, 60)]:
        df = _panel_df(T=T)
        saved = fc._CV_USE_NESTED
        try:
            fc._CV_USE_NESTED = False
            cv_ref = AutoFFS(season_length=7).cross_validation(
                df, h=6, n_windows=n_windows, warmup_steps=7
            )
            fc._CV_USE_NESTED = True
            cv_new = AutoFFS(season_length=7).cross_validation(
                df, h=6, n_windows=n_windows, warmup_steps=7
            )
        finally:
            fc._CV_USE_NESTED = saved

        # Align rows on the join keys before comparing the forecast column.
        keys = ["unique_id", "ds", "cutoff"]
        merged = cv_ref.merge(cv_new, on=keys, suffixes=("_ref", "_new"))
        assert len(merged) == len(cv_ref) == len(cv_new)
        a = merged["AutoFFS_ref"].to_numpy()
        b = merged["AutoFFS_new"].to_numpy()
        assert np.isfinite(b).all(), f"n_windows={n_windows}: nested not finite"
        np.testing.assert_allclose(
            b, a, rtol=1e-12, atol=1e-12,
            err_msg=f"n_windows={n_windows}: nested CV forecast != flat",
        )

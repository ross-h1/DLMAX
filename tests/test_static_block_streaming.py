"""``StaticBlock`` streams, so one vocabulary works whichever engine is underneath.

``StaticBlock`` used to raise on ``scan_filter``/``fwd_filter``/``forecast``: it
was built as the CV seam, and streaming a static universe already existed
elsewhere (the legacy multi path behind ``AutoFFSUniverse(grid_period=None)``).
That left ``AutoFFS(blocks=[StaticBlock(...)])`` constructible but unusable —
the ``Block`` protocol checks names, not behaviour — and forced anyone wanting a
streamed static universe onto the disk-backed orchestrator for a block
capability rather than for persistence.

The one difference from CV that remains is inherent: ``_build_multi_and_dma``
elicits its diffuse prior from the whole array it is given, which under CV is the
entire batch. A stream cannot do that without look-ahead. Given the same window
the two agree to float precision, which is what ``test_matches_cv_given_the_same_prior_window``
pins.
"""
import numpy as np
import pandas as pd
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from DLMAX.ffs_core import AutoFFS
from DLMAX.ffs.static_block import StaticBlock

PERIOD, T, Q, H = 12, 60, 3, 4

# CV and streaming are the same arithmetic once the prior window matches; they
# land ~1e-15 apart rather than bitwise because the batch path reduces through a
# fused scan and the streaming path through a Python loop over the same step.
CV_STREAM_RTOL = 1e-11


def _arr(n=T, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.column_stack([
        100 + 5 * j + 0.4 * t + 8 * np.sin(2 * np.pi * t / PERIOD)
        + rng.normal(0, 2.0, n) for j in range(Q)])


def _wide(arr):
    return pd.DataFrame({f"s{j}": arr[:, j] for j in range(arr.shape[1])},
                        index=pd.date_range("2015-01-01", periods=len(arr),
                                            freq="MS"))


def _blk(**kw):
    kw.setdefault("h_template", H)
    kw.setdefault("warmup", PERIOD)
    return StaticBlock(season_length=PERIOD, n_seas_comps=2, **kw)


def test_scan_then_forecast_returns_gridblock_layout():
    """The orchestrator consumes either block type without knowing which."""
    arr = _arr()
    b = _blk()
    b.scan_filter(arr)
    loc, sd, comp = b.forecast(H)
    loc, sd = np.asarray(loc), np.asarray(sd)
    assert loc.shape == (Q, H)          # (q, h), as GridBlock.forecast
    assert sd.shape == (H, Q)           # (h, q)
    assert np.isfinite(loc).all() and (sd > 0).all()
    assert set(comp) == {"Wc", "LOCc", "QHc", "NUc"}
    assert comp["Wc"].shape == (Q, b.nm)            # (q, M)
    assert comp["LOCc"].shape == (Q, b.nm, H)       # (q, M, h)
    assert comp["NUc"].shape == (Q, b.nm)


def test_step_equals_scan_bitwise():
    """``scan_filter`` IS a loop over ``fwd_filter``, so this holds by construction.

    Unlike the grid, whose scan is a fused ``lax.scan`` and therefore reduces in
    a different order from its step (see test_fwd_filter_face's STEP_SCAN_RTOL).
    """
    arr = _arr()
    a = _blk(); a.scan_filter(arr)
    b = _blk(); b.scan_filter(arr[:PERIOD])
    for i in range(PERIOD, len(arr)):
        b.fwd_filter(arr[i])
    la, _, _ = a.forecast(H)
    lb, _, _ = b.forecast(H)
    np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))


def test_matches_cv_given_the_same_prior_window():
    """Stream and CV differ ONLY in where the prior comes from.

    The load-bearing test. Hand the streaming face the window CV builds from and
    the two agree to float precision, which says the filter, the warmup handling
    and the h-step predictive are the same computation — and localises the
    remaining difference entirely to prior elicitation.
    """
    arr = _arr()
    srs = [f"s{j}" for j in range(Q)]
    cut = T - H - 1

    cv = _blk().forecast_rolling(srs, arr, np.array([cut], dtype=np.int32), H,
                                 warmup_steps=PERIOD, capture_trace=True)
    f_cv, q_cv = np.asarray(cv.f_h)[0], np.asarray(cv.q_h)[0]     # (M, q, h)

    sb = _blk()
    sb._ensure_stream(arr[:cut + 1], init_window=arr)   # CV's window, exactly
    for i in range(cut + 1):
        sb.fwd_filter(arr[i])
    _loc, _sd, comp = sb.forecast(H)
    f_st = np.moveaxis(np.asarray(comp["LOCc"]), 0, 1)
    q_st = np.moveaxis(np.asarray(comp["QHc"]), 0, 1)

    np.testing.assert_allclose(f_st, f_cv, rtol=CV_STREAM_RTOL, atol=0)
    np.testing.assert_allclose(q_st, q_cv, rtol=CV_STREAM_RTOL, atol=0)


def test_streaming_prior_uses_only_the_warmup_window():
    """No look-ahead: chunking the stream must not change the model.

    If the prior were elicited from whatever the first call happened to carry,
    ``scan_filter(all)`` and ``scan_filter(warmup) + fwd_filter(rest)`` would
    build different universes — which is exactly what they did before.
    """
    arr = _arr()
    a = _blk(); a.scan_filter(arr)                       # first call: everything
    b = _blk(); b.scan_filter(arr[:PERIOD])              # first call: the window
    for i in range(PERIOD, len(arr)):
        b.fwd_filter(arr[i])
    la, _, _ = a.forecast(H)
    lb, _, _ = b.forecast(H)
    np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))


def test_warmup_is_applied_per_step():
    """Inside the window the discount is suppressed, so the state must differ.

    ``multi.fwd_filter`` did not expose ``warmup_flag`` at all, so a step-by-step
    caller silently filtered the warmup window untreated.
    """
    arr = _arr()
    treated = _blk(warmup=PERIOD)
    untreated = _blk(warmup=0)
    treated.scan_filter(arr[:PERIOD + 1])
    untreated.scan_filter(arr[:PERIOD + 1])
    lt, _, _ = treated.forecast(H)
    lu, _, _ = untreated.forecast(H)
    assert not np.allclose(np.asarray(lt), np.asarray(lu))


def test_through_autoffs_end_to_end():
    """The point of the exercise: the same calls as a grid block."""
    arr = _arr()
    wide = _wide(arr)
    m = AutoFFS(blocks=[_blk()], learn_dma=False).fit(wide)
    out = m.forecast(h=H)
    assert len(out) == Q * H
    assert np.isfinite(out["AutoFFS"].to_numpy()).all()
    bundle = m.fwd_filter(arr[-1])
    assert bundle.loc.shape == (Q,)
    assert np.isfinite(bundle.loc).all() and (bundle.var > 0).all()


def test_forecast_beyond_h_template_says_why():
    """The packed universe fixes its h-step template when built, unlike the grid."""
    arr = _arr()
    b = _blk(h_template=2)
    b.scan_filter(arr)
    with pytest.raises(ValueError, match="h_template"):
        b.forecast(H)


def test_regressors_are_still_refused():
    """A static block has no tail; a supplied design would go nowhere."""
    arr = _arr()
    b = _blk(); b.scan_filter(arr)
    with pytest.raises(ValueError, match="no regression"):
        b.fwd_filter(arr[0], regressors=np.zeros((Q, 1)))


def test_persistence_still_raises_and_names_the_alternative():
    """Streaming in memory is implemented; persistence deliberately is not."""
    b = _blk()
    for fn in (b.save, b.load):
        with pytest.raises(NotImplementedError, match="grid_period=None"):
            fn("ignored.h5")

"""``fwd_filter`` is the step face; ``update`` is the scan face.

``uv_dlm`` and ``multi_model_dlm`` both pair ``fwd_filter(yt)`` — advance one
observation, return the one-step-ahead predictive — with ``scan_filter(ys)``,
which advances a block and returns nothing useful. The orchestrators only had
the scan half, spelled ``update(df)``. These tests pin the step half.

The distinction is NOT step granularity: ``update`` with a single-timestamp
frame already advances one step. It is the RETURN — the one-step-ahead
predictive is computed on every step regardless (it drives the DMA weights) and
``update`` discards it.
"""
import numpy as np
import pandas as pd
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from DLMAX.ffs_core import AutoFFS
from DLMAX.ffs.grid_block import GridBlock

PERIOD, T, Q, K, H = 12, 60, 3, 5, 4

# Single-block ``update`` runs the block's ``scan_filter`` while ``fwd_filter``
# runs its step face. Those are the SAME arithmetic in a different reduction
# order, so they land about one ulp apart (~2e-16 relative) rather than bitwise
# — the same property, and the same reasoning, as
# ``test_cv_fast_path.test_single_origin_fast_path_matches_emit_bitwise``. An
# exact assertion there would pin one platform's XLA lowering, not the property.
# The MULTI-block path shares ``_multiblock_step`` between the two faces, so it
# IS bitwise; that is asserted separately below.
STEP_SCAN_RTOL = 1e-12


def _panel(n=T, seed=0):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2015-01-01", periods=n, freq="MS")
    t = np.arange(n)
    return pd.concat([
        pd.DataFrame({"unique_id": f"s{j}", "ds": ds,
                      "y": 100 + 5 * j + 0.4 * t
                           + 8 * np.sin(2 * np.pi * t / PERIOD)
                           + rng.normal(0, 2.0, n)})
        for j in range(Q)], ignore_index=True)


def _split(df):
    cut = sorted(df["ds"].unique())[T]
    hist, new = df[df.ds < cut], df[df.ds >= cut]
    wide = new.pivot(index="ds", columns="unique_id", values="y")[
        [f"s{j}" for j in range(Q)]].to_numpy()
    return hist, new, wide


def _single():
    return AutoFFS(season_length=PERIOD, warmup=PERIOD)


def _multi():
    return AutoFFS(blocks=[GridBlock.build(period=PERIOD, warmup=PERIOD, n_comps=2),
                           GridBlock.build(period=PERIOD, warmup=PERIOD, n_comps=1)],
                   learn_dma=False)


@pytest.mark.parametrize("mk,bitwise", [(_single, False), (_multi, True)])
def test_step_equals_scan(mk, bitwise):
    """K ``fwd_filter`` calls leave the same state as one K-row ``update``."""
    df = _panel(T + K)
    hist, new, wide = _split(df)
    step, scan = mk().fit(hist), mk().fit(hist)
    for i in range(K):
        step.fwd_filter(wide[i])
    scan.update(new)

    key = ["unique_id", "ds"]
    a = step.forecast(h=H).sort_values(key).reset_index(drop=True)
    b = scan.forecast(h=H).sort_values(key).reset_index(drop=True)
    for col in ("AutoFFS", "AutoFFS-sd"):
        if bitwise:
            # multi-block: both faces go through _multiblock_step, so there is
            # no reduction-order difference left to explain a gap.
            np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy())
        else:
            np.testing.assert_allclose(a[col].to_numpy(), b[col].to_numpy(),
                                       rtol=STEP_SCAN_RTOL, atol=0)


@pytest.mark.parametrize("mk", [_single, _multi])
def test_returns_the_one_step_ahead_predictive(mk):
    """The bundle is ``predict(1)`` from the carry BEFORE ``yt`` — exactly.

    Both go through ``_predictive_arrays``, so this is an identity, not an
    approximation. It is what makes the returned value scoreable against the
    observation being fed in.
    """
    df = _panel(T + 1)
    hist, _new, wide = _split(df)
    m = mk().fit(hist)
    want = m.forecast(h=1).sort_values("unique_id")["AutoFFS"].to_numpy()
    got = m.fwd_filter(wide[0])
    np.testing.assert_array_equal(got.loc, want)
    assert got.loc.shape == (Q,) and got.var.shape == (Q,)
    assert (got.var > 0).all()


@pytest.mark.parametrize("mk", [_single, _multi])
def test_advances_the_calendar(mk):
    """Consuming an observation moves the forecast origin on one period."""
    df = _panel(T + K)
    hist, new, wide = _split(df)
    step, scan = mk().fit(hist), mk().fit(hist)
    for i in range(K):
        step.fwd_filter(wide[i])
    scan.update(new)
    key = ["unique_id", "ds"]
    a = step.forecast(h=H).sort_values(key).reset_index(drop=True)
    b = scan.forecast(h=H).sort_values(key).reset_index(drop=True)
    assert (a["ds"].to_numpy() == b["ds"].to_numpy()).all()


def test_trace_is_the_per_worker_one_step():
    """``return_trace`` gives ``(F, Q)`` ``(M, q)`` — CV's ``f1_full``/``q1_full``.

    The combined bundle must be consistent with the trace it came from: the
    combined location lies within the spread of the per-worker locations, since
    it is a weighted average of them.
    """
    df = _panel(T + 1)
    hist, _new, wide = _split(df)
    m = _single().fit(hist)
    bundle, (F, Qq) = m.fwd_filter(wide[0], return_trace=True)
    assert F.shape == Qq.shape and F.shape[1] == Q          # (M, q)
    assert F.shape[0] == m._fit_blocks[0].nm
    assert np.isfinite(F).all() and (Qq > 0).all()
    assert (bundle.loc >= F.min(axis=0) - 1e-9).all()
    assert (bundle.loc <= F.max(axis=0) + 1e-9).all()


def test_rejects_a_wrong_length_observation():
    df = _panel(T + 1)
    hist, _new, _wide = _split(df)
    m = _single().fit(hist)
    with pytest.raises(ValueError, match="one observation per fitted series"):
        m.fwd_filter(np.zeros(Q + 1))


def test_does_not_require_a_prior_fit():
    """``fwd_filter`` self-initialises: no ``fit`` call is required.

    It used to raise "Call fit(...) first", which was true of the
    implementation but wrong as a contract -- ``uv_dlm.fwd_filter`` steps from
    construction, and this face is meant to match it. What AutoFFS needs that
    ``uv_dlm`` does not is a warmup window to elicit its prior from, so it
    buffers one; see the self-initialising section below.
    """
    m = AutoFFS(season_length=PERIOD, warmup=PERIOD)
    bundle = m.fwd_filter(np.zeros(Q))          # must not raise
    assert np.isnan(bundle.loc).all()
    assert m.is_fitted is False                 # still no prior after one row


# --------------------------------------------------------------------------
# Self-initialising: stepping from construction, with no fit() call
#
# AutoFFS derives its diffuse prior from data (``grid_init`` over each block's
# warmup window), so unlike ``uv_dlm`` -- handed m0/C0 -- it cannot step from
# nothing. ``fwd_filter`` therefore buffers the opening observations and elicits
# the prior once there are enough, which must be the SAME computation ``fit``
# does over those rows.
# --------------------------------------------------------------------------

def _blocks4():
    """Four blocks with DIFFERENT warmups — the mixed case that matters."""
    return [GridBlock.build(period=PERIOD, warmup=w, n_comps=c)
            for w, c in ((PERIOD, 2), (PERIOD, 1), (2 * PERIOD, 2), (2 * PERIOD, 1))]


def _mk4():
    return AutoFFS(blocks=_blocks4(), learn_dma=False)


@pytest.mark.parametrize("mk", [_single, _mk4])
def test_self_init_equals_fit_then_step(mk):
    """Streaming from construction == fit(window) + step, bitwise.

    This is the load-bearing assertion. Prior elicitation, per-block learning
    suppression and the union DMA's warmup gate are all keyed on each block's
    OWN warmup; if any were mishandled by the buffered path, these two routes
    would diverge.
    """
    df = _panel(T)
    wide = df.pivot(index="ds", columns="unique_id", values="y")[
        [f"s{j}" for j in range(Q)]]
    arr = wide.to_numpy()

    a = mk()
    need = a._warmup_target()
    for i in range(len(arr)):
        a.fwd_filter(arr[i])

    b = mk().fit(wide.iloc[:need])
    for i in range(need, len(arr)):
        b.fwd_filter(arr[i])

    key = ["unique_id", "ds"]
    fa = a.forecast(h=H).sort_values(key).reset_index(drop=True)
    fb = b.forecast(h=H).sort_values(key).reset_index(drop=True)
    for col in ("AutoFFS", "AutoFFS-sd"):
        np.testing.assert_array_equal(fa[col].to_numpy(), fb[col].to_numpy())


def test_warmup_steps_return_nan_then_real_predictives():
    """NaN while no prior exists — not a plausible-looking number."""
    df = _panel(T)
    wide = df.pivot(index="ds", columns="unique_id", values="y")[
        [f"s{j}" for j in range(Q)]]
    arr = wide.to_numpy()
    m = _single()
    need = m._warmup_target()

    for i in range(need):
        b = m.fwd_filter(arr[i])
        assert np.isnan(b.loc).all() and np.isnan(b.var).all()
        assert b.loc.shape == (Q,)
    after = m.fwd_filter(arr[need])
    assert np.isfinite(after.loc).all()
    assert (after.var > 0).all()


def test_warmup_target_is_the_max_over_blocks():
    """Buffer to the LONGEST warmup, so every block gets its full window."""
    m = _mk4()
    assert m._warmup_target() == 2 * PERIOD
    assert _single()._warmup_target() == PERIOD


def test_trace_during_warmup_is_absent():
    """No per-worker trace before the workers have a prior."""
    df = _panel(T)
    arr = df.pivot(index="ds", columns="unique_id", values="y")[
        [f"s{j}" for j in range(Q)]].to_numpy()
    bundle, trace = _single().fwd_filter(arr[0], return_trace=True)
    assert np.isnan(bundle.loc).all()
    assert trace == (None, None)


def test_width_change_during_warmup_is_rejected():
    m = _single()
    m.fwd_filter(np.zeros(Q))
    with pytest.raises(ValueError, match="changed width during warmup"):
        m.fwd_filter(np.zeros(Q + 1))


def test_self_init_names_series_positionally():
    """No frame was given, so ids are positional and the calendar counts steps."""
    df = _panel(T)
    arr = df.pivot(index="ds", columns="unique_id", values="y")[
        [f"s{j}" for j in range(Q)]].to_numpy()
    m = _single()
    for i in range(m._warmup_target() + 2):
        m.fwd_filter(arr[i])
    out = m.forecast(h=2)
    assert set(out["unique_id"]) == {f"s{j}" for j in range(Q)}
    assert np.issubdtype(np.asarray(out["ds"]).dtype, np.integer)


# --------------------------------------------------------------------------
# AutoFFSUniverse: the same face over the disk-backed carry
# --------------------------------------------------------------------------

def _uni(tmp_path, tag, hist):
    from DLMAX.ffs_core import AutoFFSUniverse
    u = AutoFFSUniverse.create(str(tmp_path / tag), season_length=PERIOD,
                               warmup=PERIOD)
    return u.fit(hist, freq="MS", h_template=H)


def test_universe_step_equals_scan(tmp_path):
    """``fwd_filter`` + ``flush`` leaves the same state on disk as ``update``."""
    df = _panel(T + K)
    hist, new, wide = _split(df)
    step = _uni(tmp_path, "step", hist)
    for i in range(K):
        step.fwd_filter(wide[i], persist=False)
    step.flush()
    scan = _uni(tmp_path, "scan", hist)
    scan.update(new)

    key = ["unique_id", "ds"]
    a = step.forecast(h=H).sort_values(key).reset_index(drop=True)
    b = scan.forecast(h=H).sort_values(key).reset_index(drop=True)
    assert (a["ds"].to_numpy() == b["ds"].to_numpy()).all()
    np.testing.assert_allclose(a["AutoFFS"].to_numpy(), b["AutoFFS"].to_numpy(),
                               rtol=STEP_SCAN_RTOL, atol=0)


def test_universe_persist_mode_does_not_change_numbers(tmp_path):
    """``persist`` is a durability choice, not a numerical one — bitwise equal.

    Saving per step versus deferring to ``flush`` must be indistinguishable in
    the result, or the fast path would be quietly running a different model.
    """
    df = _panel(T + K)
    hist, _new, wide = _split(df)
    eager = _uni(tmp_path, "eager", hist)
    for i in range(K):
        eager.fwd_filter(wide[i], persist=True)
    lazy = _uni(tmp_path, "lazy", hist)
    for i in range(K):
        lazy.fwd_filter(wide[i], persist=False)
    lazy.flush()

    key = ["unique_id", "ds"]
    a = eager.forecast(h=H).sort_values(key).reset_index(drop=True)
    b = lazy.forecast(h=H).sort_values(key).reset_index(drop=True)
    np.testing.assert_array_equal(a["AutoFFS"].to_numpy(), b["AutoFFS"].to_numpy())


def test_universe_agrees_with_autoffs_bitwise(tmp_path):
    """The two orchestrators' step faces are the same computation.

    Mirrors ``tests/test_autoffs_forecast_face.py``, which pins the same
    property for fit/update/predict.
    """
    df = _panel(T + K)
    hist, _new, wide = _split(df)
    mem = AutoFFS(season_length=PERIOD, warmup=PERIOD).fit(hist)
    uni = _uni(tmp_path, "uni", hist)
    for i in range(K):
        bm = mem.fwd_filter(wide[i])
        bu = uni.fwd_filter(wide[i], persist=False)
        np.testing.assert_array_equal(np.asarray(bm.loc), np.asarray(bu.loc))
        np.testing.assert_array_equal(np.asarray(bm.var), np.asarray(bu.var))
    uni.flush()
    key = ["unique_id", "ds"]
    a = mem.forecast(h=H).sort_values(key).reset_index(drop=True)
    b = uni.forecast(h=H).sort_values(key).reset_index(drop=True)
    np.testing.assert_array_equal(a["AutoFFS"].to_numpy(), b["AutoFFS"].to_numpy())


def test_universe_rejects_a_wrong_length_observation(tmp_path):
    df = _panel(T + 1)
    hist, _new, _wide = _split(df)
    u = _uni(tmp_path, "bad", hist)
    with pytest.raises(ValueError, match="one observation per ACTIVE series"):
        u.fwd_filter(np.zeros(Q + 1))


def test_flush_is_a_noop_when_nothing_pending(tmp_path):
    df = _panel(T + 1)
    hist, _new, _wide = _split(df)
    u = _uni(tmp_path, "noop", hist)
    assert u.flush() is u
    u.flush()

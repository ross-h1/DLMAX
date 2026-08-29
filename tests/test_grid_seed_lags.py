"""Seed-lag persistence: an AR tail that survives chunking and save/reopen.

``forecast(seed_lags=)`` drives the iterated forecast, but the lags must be
carried by the block itself: if the caller has to supply both the filter
design and the forecast seed, a reopened block knows neither. Two consequences,
both silent:

* **chunking blanked the tail.** A design built from ``ys`` alone zero-fills its
  first ``n_regs`` rows. That is right for the first chunk and wrong for every
  one after it, so a universe that fits then updates lost the tail at each
  origin — and still produced perfectly plausible numbers.
* **a reopened block could not forecast at all**, since nothing outside it knows
  the y stream.

The fix is one buffer of trailing observations, held on the block and persisted
beside the carry. It is DATA, not filter state, so it stays out of the wing carry
pytree — which is also why blocks saved before it still load.

The load-bearing test here is ``test_chunked_filtering_equals_one_shot``, with the
guard immediately after it showing the naive alternative really does differ.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.discount_grid import _grid_model, lag_design
from DLMAX.ffs.dlm_builder import AR, DLM, LocalLevel, Regressors
from DLMAX.ffs.grid_block import GridBlock
from DLMAX.ffs_core import AutoFFSUniverse

T, ORDER, Q, H, WARMUP = 60, 2, 2, 6, 8


def _ar_series(T_=T, q=Q, seed=3):
    rng = np.random.default_rng(seed)
    y = np.zeros((T_, q))
    for t in range(2, T_):
        y[t] = (10.0 + 0.6 * (y[t - 1] - 10.0) - 0.25 * (y[t - 2] - 10.0)
                + rng.normal(0, 0.5, q))
    return y


def _comps(kind, order=ORDER):
    d = DLM(n_series=1)
    d.add_component(LocalLevel(name="level", disc_rate=0.95))
    if kind == "ar":
        d.add_component(AR(name="ar", order=order, disc_rate=0.99))
    elif kind == "exog":
        d.add_component(Regressors(name="x", n_regs=order, disc_rate=0.99))
    d.set_error(disc_rate=0.99, power=1.0)
    return list(d.components)


def _block(kind="ar", order=ORDER):
    c = _comps(kind, order)
    return GridBlock([("c0", _grid_model(c, 1.0), tuple(c))], period=None,
                     warmup=WARMUP)


# --- the design itself -------------------------------------------------------

def test_lag_design_is_the_documented_lag_matrix():
    """``out[t, s, j] = y_{t-1-j}``, zero where the lag predates the data."""
    y = _ar_series(8, 2)
    got = np.asarray(lag_design(jnp.asarray(y), ORDER))
    assert got.shape == (8, 2, ORDER)
    for t in range(8):
        for j in range(ORDER):
            want = y[t - 1 - j] if t - 1 - j >= 0 else np.zeros(2)
            np.testing.assert_allclose(got[t, :, j], want)


def test_lag_design_history_makes_chunks_continuous():
    """``lag_design`` over a whole run == the two chunks, the second seeded.

    This identity is the whole reason the block carries a buffer; if it did not
    hold, nothing downstream could.
    """
    y = _ar_series(20, 2)
    whole = np.asarray(lag_design(jnp.asarray(y), ORDER))
    first = np.asarray(lag_design(jnp.asarray(y[:12]), ORDER))
    second = np.asarray(lag_design(jnp.asarray(y[12:]), ORDER,
                                   history=y[12 - ORDER:12]))
    np.testing.assert_allclose(whole, np.concatenate([first, second], axis=0))


# --- the buffer --------------------------------------------------------------

def test_the_block_holds_the_trailing_observations():
    y = _ar_series()
    b = _block()
    assert b.n_regs == ORDER and b.is_autoregressive
    assert b._lags is None                       # nothing ingested yet
    b.scan_filter(y)
    np.testing.assert_allclose(np.asarray(b._lags), y[-ORDER:])       # time order
    # the forecast seed is the same thing, most-recent-first, per series
    np.testing.assert_allclose(b.seed_lags, y[-ORDER:][::-1].T)


def test_a_structural_block_holds_nothing():
    b = _block(kind="none")
    b.scan_filter(_ar_series())
    assert b.n_regs == 0 and not b.is_autoregressive
    assert b._lags is None and b.seed_lags is None


# --- the headline ------------------------------------------------------------

def test_chunked_filtering_equals_one_shot():
    """Fit-then-update must equal fitting the whole run at once."""
    y = _ar_series()
    one = _block()
    one.scan_filter(y)
    chunked = _block()
    chunked.scan_filter(y[:40])
    chunked.scan_filter(y[40:])
    np.testing.assert_allclose(np.asarray(chunked._lags), np.asarray(one._lags))
    a, _sd, _c = one.forecast(H)
    b, _sd2, _c2 = chunked.forecast(H)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


def test_the_naive_per_chunk_design_really_does_differ():
    """Guard on the test above: without the seeding it would pass vacuously.

    Rebuilding each chunk's design from that chunk alone — the only thing a
    caller could do before the buffer existed — blanks the tail for ``n_regs``
    steps at every origin. If that produced the same answer there would be
    nothing here worth fixing.
    """
    y = _ar_series()
    one = _block()
    one.scan_filter(y)
    naive = _block()
    naive.scan_filter(y[:40], regressors=lag_design(jnp.asarray(y[:40]), ORDER))
    naive.scan_filter(y[40:], regressors=lag_design(jnp.asarray(y[40:]), ORDER))
    a, _s, _c = one.forecast(H)
    b, _s2, _c2 = naive.forecast(H)
    assert not np.allclose(np.asarray(a), np.asarray(b))


def test_an_explicit_design_still_wins():
    """The caller can drive the tail by hand; auto-build is only the default."""
    y = _ar_series()
    auto = _block()
    auto.scan_filter(y)
    manual = _block()
    manual.scan_filter(y, regressors=lag_design(jnp.asarray(y), ORDER))
    a, _s, _c = auto.forecast(H)
    b, _s2, _c2 = manual.forecast(H)
    # identical here because a single chunk needs no history -- which is exactly
    # why the chunked test above is the one that matters
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


# --- forecasting -------------------------------------------------------------

def test_an_ar_block_seeds_its_own_forecast():
    y = _ar_series()
    b = _block()
    b.scan_filter(y)
    auto, _s, _c = b.forecast(H)
    explicit, _s2, _c2 = b.forecast(H, seed_lags=b.seed_lags)
    np.testing.assert_array_equal(np.asarray(auto), np.asarray(explicit))
    # and it is not the same as forecasting with the tail left at zero
    zeroed, _s3, _c3 = b.forecast(H, seed_lags=np.zeros((Q, ORDER)))
    assert not np.allclose(np.asarray(auto), np.asarray(zeroed))


def test_an_exogenous_block_neither_holds_nor_seeds_lags():
    """Defaulting the seed for an EXOGENOUS tail would feed the model's own
    forecasts back as though they were the caller's regressors.

    An exogenous tail takes its regressors from the caller at every step, future
    included, so the block keeps no buffer at all — there is nothing it could
    self-seed WITH, which is a stronger guarantee than declining to.
    """
    y = _ar_series()
    X = np.asarray(lag_design(jnp.asarray(y), ORDER))     # any design will do
    b = _block(kind="exog")
    assert b.n_regs == ORDER and not b.is_autoregressive
    b.scan_filter(y, regressors=X)
    assert b._lags is None and b.seed_lags is None
    loc, _s, _c = b.forecast(H)
    assert np.isfinite(np.asarray(loc)).all()


def test_zero_seed_lags_are_not_the_same_as_no_seed_lags():
    """A trap worth pinning: ``seed_lags=zeros`` is NOT "the tail is off".

    ``None`` takes the ordinary ``FH`` path, where the tail's F entries stay zero
    for every horizon. Zeros instead ENTER the iterated forecast, whose whole
    job is to feed each horizon's predictive mean forward as the next lag — so
    only horizon 1 sees a zero lag and the rest see the model's own forecasts.
    The two differ, and a caller reaching for zeros to disable the tail would get
    the opposite of what they wanted.
    """
    from DLMAX.ffs.discount_grid import grid_stream_forecast
    y = _ar_series()
    b = _block()
    b.scan_filter(y)
    off, _q, _c = grid_stream_forecast(b._ensure_static(), b._carry, H, None, None)
    zeros, _s2, _c2 = b.forecast(H, seed_lags=np.zeros((Q, ORDER)))
    assert np.isfinite(np.asarray(off)).all()
    assert not np.allclose(np.asarray(off), np.asarray(zeros))


# --- persistence -------------------------------------------------------------

def test_lags_survive_save_and_load(tmp_path):
    y = _ar_series()
    b = _block()
    b.scan_filter(y)
    before, _s, _c = b.forecast(H)
    p = str(tmp_path / "blk.h5")
    b.save(p)

    b2 = _block()
    b2.load(p)
    np.testing.assert_allclose(np.asarray(b2._lags), np.asarray(b._lags))
    after, _s2, _c2 = b2.forecast(H)
    np.testing.assert_allclose(np.asarray(before), np.asarray(after),
                               rtol=1e-12, atol=1e-12)


def test_a_block_saved_without_lags_still_loads(tmp_path):
    """Backwards compatibility: the buffer is its own dataset, so a file written
    before it existed loads with no lags rather than failing."""
    import h5py
    y = _ar_series()
    b = _block()
    b.scan_filter(y)
    p = str(tmp_path / "old.h5")
    b.save(p)
    with h5py.File(p, "a") as f:                 # simulate a pre-buffer file
        del f["grid_state"]["lags"]

    b2 = _block()
    b2.load(p)
    assert b2._lags is None and b2.seed_lags is None
    loc, _s, _c = b2.forecast(H)                 # falls back to the zero tail
    assert np.isfinite(np.asarray(loc)).all()


def test_padding_and_appending_carry_the_lags():
    y = _ar_series()
    b = _block()
    b.scan_filter(y)
    b.pad_to(5)
    assert np.asarray(b._lags).shape == (ORDER, 5)
    np.testing.assert_allclose(np.asarray(b._lags)[:, :Q], y[-ORDER:])

    one = _block()
    one.scan_filter(y[:, :1])
    two = _block()
    two.scan_filter(y[:, 1:])
    one.append_series(two)
    assert np.asarray(one._lags).shape == (ORDER, 2)
    np.testing.assert_allclose(np.asarray(one._lags), y[-ORDER:])


# --- through the universe ----------------------------------------------------

def ar_blocks(init_data, h, ctx):
    """An AR grid block — needs no provider, since the tail is the y stream."""
    c = _comps("ar")
    return [GridBlock([("c0", _grid_model(c, 1.0), tuple(c))], period=None,
                      warmup=ctx["warmup"])]


def _panel(n_days=T, ids=("s0", "s1")):
    ds = pd.date_range("2021-01-01", periods=n_days, freq="D")
    y = _ar_series(n_days, len(ids))
    return ds, pd.concat(
        [pd.DataFrame({"unique_id": s, "ds": ds, "y": y[:, j]})
         for j, s in enumerate(ids)], ignore_index=True)


def test_ar_universe_runs_without_a_provider(tmp_path):
    """The gate must accept an AR tail with no exog_provider — it supplies its
    own design — and the round trip must survive a reopen, which is what the
    persisted buffer buys."""
    ds, df = _panel()
    p = str(tmp_path / "uni")
    uni = AutoFFSUniverse.create(p, season_length=None, warmup=WARMUP,
                                 max_batch_size=10, block_builder=ar_blocks)
    uni.fit(df, freq="D", h_template=H)
    f1 = uni.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    assert np.isfinite(f1).all()

    uni2 = AutoFFSUniverse.open(p, block_builder=ar_blocks)
    f2 = uni2.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    np.testing.assert_allclose(f1, f2, rtol=1e-12, atol=1e-12)


def test_ar_universe_update_equals_one_shot_fit(tmp_path):
    """The chunking property, end to end: this is what silently broke before."""
    ds, df = _panel()
    cut = ds[T - 10]

    a = AutoFFSUniverse.create(str(tmp_path / "a"), season_length=None,
                               warmup=WARMUP, max_batch_size=10,
                               block_builder=ar_blocks)
    a.fit(df[df["ds"] < cut], freq="D", h_template=H)
    a.update(df[df["ds"] >= cut])

    b = AutoFFSUniverse.create(str(tmp_path / "b"), season_length=None,
                               warmup=WARMUP, max_batch_size=10,
                               block_builder=ar_blocks)
    b.fit(df, freq="D", h_template=H)

    fa = a.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    fb = b.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    np.testing.assert_allclose(fa, fb, rtol=1e-9, atol=1e-9)


def test_ar_tail_with_a_provider_is_rejected(tmp_path):
    """Two sources for the same tail slots."""
    _ds, df = _panel()
    uni = AutoFFSUniverse.create(
        str(tmp_path / "u"), season_length=None, warmup=WARMUP,
        block_builder=ar_blocks,
        exog_provider=lambda ids, ds: np.zeros((len(ds), len(ids), ORDER)))
    with pytest.raises(ValueError, match="AUTOREGRESSIVE"):
        uni.fit(df, freq="D", h_template=H)

"""append-on-add add_series() matches singleton-add + defragment().

The new ``add_series`` fits each series exactly as before, then concatenates
its post-fit state into the current "open" batch (the last non-full batch),
rebuilding that one batch in place — O(open batch) rather than the
O(universe) ``defragment``. The result must be identical (to float
precision) to the legacy path of ``_add_series_singleton`` followed by
``defragment``, because both use the same per-array concat logic on the same
post-fit per-model state.
"""
import numpy as np
import pandas as pd


def _legacy_create(*args, **kwargs):
    """Create a universe on the LEGACY multi-model path.

    AutoFFSUniverse now defaults to the wing grid, which persists batches in the
    block format and carries no exog / defragment / add-series-append face.
    These tests exercise those legacy behaviours, so they opt out explicitly via
    the documented grid_period=None escape hatch.
    """
    from DLMAX.ffs_core import AutoFFSUniverse
    kwargs.setdefault("grid_period", None)
    return AutoFFSUniverse.create(*args, **kwargs)



def _panel(ids, n=40, season=7):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.arange(n)
    rows = []
    for i, sid in enumerate(ids):
        y = (10 + 0.1 * rng + 3 * np.sin(2 * np.pi * rng / season)
             + np.random.default_rng(i).normal(0, 0.3, n))
        rows.append(pd.DataFrame({"unique_id": sid, "ds": dates, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _series_df(sid, n=40, season=7, seed=99):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.arange(n)
    y = (8 + 2 * np.sin(2 * np.pi * rng / season)
         + np.random.default_rng(seed).normal(0, 0.3, n))
    return pd.DataFrame({"ds": dates, "y": y})


ADDS = [(f"a{i}", 10 + i) for i in range(5)]  # five new series


def _fc(uni):
    return (uni.forecast(h=12, level=[80, 95])
            .sort_values(["unique_id", "ds"]).reset_index(drop=True))


def test_add_series_append_matches_singleton_plus_defragment(tmp_path):
    from DLMAX.ffs_core import AutoFFSUniverse

    base_ids = ["s0", "s1", "s2"]

    # Reference: legacy singleton add then defragment.
    pref = str(tmp_path / "ref")
    ref = _legacy_create(
        pref, season_length=7, warmup_steps=14, max_batch_size=4)
    ref.fit(_panel(base_ids), freq="D", h_template=12)
    for sid, seed in ADDS:
        ref._add_series_singleton(sid, _series_df(sid, seed=seed))
    ref.defragment()
    fc_ref = _fc(ref)

    # New: append-on-add, no defragment.
    pnew = str(tmp_path / "new")
    new = _legacy_create(
        pnew, season_length=7, warmup_steps=14, max_batch_size=4)
    new.fit(_panel(base_ids), freq="D", h_template=12)
    for sid, seed in ADDS:
        new.add_series(sid, _series_df(sid, seed=seed))
    fc_new = _fc(new)

    # Identical membership and last_ds.
    assert set(new.list_series()) == set(ref.list_series())
    assert set(new.list_series()) == set(base_ids + [a for a, _ in ADDS])
    for sid in new.list_series():
        assert (new._manifest.loc[sid, "last_ds"]
                == ref._manifest.loc[sid, "last_ds"])

    # Identical forecasts to float precision (same basis as test_defragment:
    # nested-vmap forecast tiles differently by batch size, ~1e-15 noise only).
    assert set(fc_new["unique_id"]) == set(fc_ref["unique_id"])
    vcols = [c for c in fc_ref.columns if c not in ("unique_id", "ds")]
    np.testing.assert_allclose(
        fc_new[vcols].to_numpy(), fc_ref[vcols].to_numpy(),
        rtol=1e-12, atol=1e-12)


def test_add_series_append_then_update_streams_correctly(tmp_path):
    """After append-on-add the (rebuilt) open batch keeps streaming."""
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "u")
    uni = _legacy_create(
        p, season_length=7, warmup_steps=14, max_batch_size=4)
    uni.fit(_panel(["s0", "s1", "s2"]), freq="D", h_template=12)
    for sid, seed in ADDS:
        uni.add_series(sid, _series_df(sid, seed=seed))

    all_ids = ["s0", "s1", "s2"] + [a for a, _ in ADDS]
    nxt = pd.DatetimeIndex([pd.Timestamp("2020-01-01") + pd.Timedelta(days=40)])
    upd = pd.concat(
        [pd.DataFrame({"unique_id": s, "ds": nxt, "y": [9.0]})
         for s in all_ids], ignore_index=True)
    uni.update(upd)
    fc = uni.forecast(h=12, level=[80, 95])
    assert fc["unique_id"].nunique() == len(all_ids)
    vcols = [c for c in fc.columns if c not in ("unique_id", "ds")]
    assert np.isfinite(fc[vcols].to_numpy()).all()
    # 12 horizons per series.
    assert len(fc) == 12 * len(all_ids)


def test_add_series_append_respects_max_batch_size(tmp_path):
    """No live batch exceeds max_batch_size; crossing the boundary opens one."""
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "u")
    mbs = 4
    uni = _legacy_create(
        p, season_length=7, warmup_steps=14, max_batch_size=mbs)
    uni.fit(_panel(["s0", "s1", "s2"]), freq="D", h_template=12)

    def n_live_batches():
        man = uni._manifest
        return man.loc[man["active"], "batch_id"].nunique()

    def max_batch_fill():
        man = uni._manifest
        return int(man.loc[man["active"]].groupby("batch_id").size().max())

    # fit -> 1 batch of 3 (under the cap).
    assert n_live_batches() == 1
    assert max_batch_fill() == 3

    # Add a0 -> open batch had 3 < 4, append -> still 1 batch of 4 (full).
    uni.add_series("a0", _series_df("a0", seed=10))
    assert n_live_batches() == 1
    assert max_batch_fill() == 4

    # Add a1 -> open batch full -> new batch spawned -> 2 batches.
    uni.add_series("a1", _series_df("a1", seed=11))
    assert n_live_batches() == 2
    assert max_batch_fill() == mbs

    # A few more; the invariant max-fill <= mbs must hold throughout.
    for i in range(2, 6):
        uni.add_series(f"a{i}", _series_df(f"a{i}", seed=10 + i))
        assert max_batch_fill() <= mbs
    # 3 + 6 = 9 series over ceil(9/4)=3 batches.
    assert len(uni.list_series()) == 9
    assert n_live_batches() == 3

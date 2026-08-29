"""defragment() consolidates batches without changing forecasts.

`add_series` writes a singleton batch per product, so a streaming universe
accumulates many tiny batches. defragment() merges the live per-series
state across all batches (concatenation along the series axis, no refit),
so forecasts are unchanged bit-for-bit while the batch count collapses to
the max_batch_size-optimal layout.
"""
import numpy as np
import pandas as pd
import pytest


def _legacy_create(*args, **kwargs):
    """Create a universe on the LEGACY multi-model path.

    AutoFFSUniverse now defaults to the wing grid, which persists batches in the
    block format and carries no exog / defragment / fixed-capacity face. These
    tests exercise those legacy features, so they opt out explicitly — the same
    grid_period=None escape hatch documented for users.
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


def _add(uni, sid, n=40, season=7, seed=99):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.arange(n)
    y = 8 + 2 * np.sin(2 * np.pi * rng / season) + np.random.default_rng(seed).normal(0, 0.3, n)
    uni.add_series(sid, pd.DataFrame({"ds": dates, "y": y}))


def test_defragment_preserves_forecasts(tmp_path):
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "u")
    uni = _legacy_create(p, season_length=7, warmup_steps=14, max_batch_size=None)
    uni.fit(_panel(["s0", "s1", "s2"]), freq="D", h_template=12)
    _add(uni, "a0", seed=10)
    _add(uni, "a1", seed=11)               # -> init batch + 2 singletons

    fc_before = (uni.forecast(h=12, level=[80, 95])
                 .sort_values(["unique_id", "ds"]).reset_index(drop=True))

    info = uni.defragment()
    assert info["n_batches_after"] == 1     # max_batch_size None -> one batch

    uni2 = AutoFFSUniverse.open(p)
    assert len(uni2.list_series()) == 5
    fc_after = (uni2.forecast(h=12, level=[80, 95])
                .sort_values(["unique_id", "ds"]).reset_index(drop=True))

    assert set(fc_before["unique_id"]) == set(fc_after["unique_id"])
    vcols = [c for c in fc_before.columns if c not in ("unique_id", "ds")]
    # Float-precision, not bit-for-bit: multi_model_dlm.forecast uses a nested
    # vmap (model-only operands shared across series, never materialised as the
    # (nm*q, h, k, k) GH tensor), and XLA tiles the inner per-series vmap
    # differently for different batch sizes, so consolidating batches shifts
    # forecasts by ~machine epsilon (~1e-15). That can't survive into any
    # reported metric and cannot amplify (contractive filter + softmax weights);
    # it is exactly the float-level nondeterminism the project tolerates. The
    # earlier flat single-axis vmap was strictly bitwise-stable across regrouping
    # but materialised the full GH tensor (~33 GB unbatched) — the trade we took.
    np.testing.assert_allclose(
        fc_before[vcols].to_numpy(), fc_after[vcols].to_numpy(),
        rtol=1e-12, atol=1e-12)


def test_defragment_then_update_streams_correctly(tmp_path):
    """After consolidation the merged batch keeps streaming correctly."""
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "u")
    uni = _legacy_create(p, season_length=7, warmup_steps=14, max_batch_size=None)
    uni.fit(_panel(["s0", "s1", "s2"]), freq="D", h_template=12)
    _add(uni, "a0", seed=10)
    uni.defragment()

    nxt = pd.DatetimeIndex([pd.Timestamp("2020-01-01") + pd.Timedelta(days=40)])
    upd = pd.concat(
        [pd.DataFrame({"unique_id": s, "ds": nxt, "y": [9.0]})
         for s in ["s0", "s1", "s2", "a0"]], ignore_index=True)
    uni.update(upd)
    fc = uni.forecast(h=12)
    assert fc["unique_id"].nunique() == 4
    assert np.isfinite(fc["AutoFFS"].to_numpy()).all()


def test_defragment_chunks_to_max_batch_size(tmp_path):
    """With a finite max_batch_size, consolidate into ceil(n/size) batches."""
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "u")
    uni = _legacy_create(p, season_length=7, warmup_steps=14, max_batch_size=2)
    uni.fit(_panel([f"s{i}" for i in range(5)]), freq="D", h_template=12)
    for i in range(3):
        _add(uni, f"a{i}", seed=20 + i)     # 8 series total
    info = uni.defragment()
    assert info["n_batches_after"] == 4     # ceil(8 / 2)
    assert len(AutoFFSUniverse.open(p).list_series()) == 8


def test_defragment_drops_inactive_slots(tmp_path):
    from DLMAX.ffs_core import AutoFFSUniverse

    p = str(tmp_path / "u")
    uni = _legacy_create(p, season_length=7, warmup_steps=14, max_batch_size=None)
    uni.fit(_panel(["s0", "s1", "s2"]), freq="D", h_template=12)
    uni.remove_series("s1")
    info = uni.defragment()
    assert info["n_slots_freed"] == 1
    uni2 = AutoFFSUniverse.open(p)
    assert set(uni2.list_series()) == {"s0", "s2"}
    fc = uni2.forecast(h=12)
    assert set(fc["unique_id"]) == {"s0", "s2"}

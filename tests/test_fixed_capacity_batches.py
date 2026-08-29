"""Fixed-capacity (padded) batches: every batch keeps a constant slot count
(= max_batch_size) across add_series, so the jitted filter/forecast compile
once instead of recompiling on each add (which leaked host RAM on long rolls).
Padding slots are inactive, never leak into output, and forecasts stay finite.
"""
import numpy as np
import pandas as pd


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


def _sdf(sid, seed):
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    rng = np.arange(40)
    y = 8 + 2 * np.sin(2 * np.pi * rng / 7) + np.random.default_rng(seed).normal(0, 0.3, 40)
    return pd.DataFrame({"ds": dates, "y": y})


def _slot_counts(uni):
    from DLMAX.ffs_core import _load_batch_state
    man = uni._manifest
    bids = sorted(set(man.loc[man["active"], "batch_id"].astype(int)))
    return [len(_load_batch_state(uni._batch_path(b))[0].srs_ids) for b in bids]


def test_batches_stay_constant_capacity(tmp_path):
    from DLMAX.ffs_core import AutoFFSUniverse
    mbs = 4
    uni = _legacy_create(str(tmp_path / "u"), season_length=7,
                                 warmup_steps=14, max_batch_size=mbs)
    uni.fit(_panel(["s0", "s1", "s2"]), freq="D", h_template=12)
    # fit of 3 -> one batch padded to capacity (4 slots, 3 active)
    assert _slot_counts(uni) == [mbs]
    assert int(uni._manifest["active"].sum()) == 3

    for i in range(6):
        uni.add_series(f"a{i}", _sdf(f"a{i}", 10 + i))
        # EVERY live batch always has exactly mbs slots -> constant shape
        assert all(c == mbs for c in _slot_counts(uni)), _slot_counts(uni)

    assert len(uni.list_series()) == 9
    # 9 active over ceil(9/4)=3 batches, each padded to 4 slots
    assert _slot_counts(uni) == [mbs, mbs, mbs]


def test_padding_never_leaks_and_forecasts_finite(tmp_path):
    from DLMAX.ffs_core import AutoFFSUniverse
    uni = _legacy_create(str(tmp_path / "u"), season_length=7,
                                 warmup_steps=14, max_batch_size=3)
    uni.fit(_panel(["s0", "s1"]), freq="D", h_template=12)  # 2 active in a 3-slot batch
    for i in range(3):
        uni.add_series(f"a{i}", _sdf(f"a{i}", 20 + i))
    fc = uni.forecast(h=12, level=[80, 95])
    # no synthetic padding id leaks into output
    assert not any(str(u).startswith("__pad_") for u in fc["unique_id"].unique())
    assert set(fc["unique_id"]) == {"s0", "s1", "a0", "a1", "a2"}
    vcols = [c for c in fc.columns if c not in ("unique_id", "ds")]
    assert np.isfinite(fc[vcols].to_numpy()).all()


def test_padded_universe_round_trips(tmp_path):
    from DLMAX.ffs_core import AutoFFSUniverse
    p = str(tmp_path / "u")
    uni = _legacy_create(p, season_length=7, warmup_steps=14, max_batch_size=3)
    uni.fit(_panel(["s0", "s1"]), freq="D", h_template=12)
    uni.add_series("a0", _sdf("a0", 30))
    fc1 = uni.forecast(h=12).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    reopened = AutoFFSUniverse.open(p)
    assert _slot_counts(reopened) == _slot_counts(uni)        # padding survives persist
    fc2 = reopened.forecast(h=12).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    vcols = [c for c in fc1.columns if c not in ("unique_id", "ds")]
    np.testing.assert_allclose(fc1[vcols].to_numpy(), fc2[vcols].to_numpy(),
                               rtol=1e-12, atol=1e-12)

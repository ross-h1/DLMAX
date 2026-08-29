"""``add_series_many`` is bit-identical to k sequential ``add_series`` calls.

The batched append fits each new series as a singleton (exactly as
``add_series``), then does ONE ``_gather_live`` + ``_build_batch`` + save per
affected batch instead of one per series. Because the gather/build is pure
series-axis slice/concatenate/reshape (no recomputation), gathering all the
singletons at once equals gathering them incrementally — so the resulting
universe (batch layout, membership, and forecasts) matches the sequential path
to float precision.

Covers: the plain case, a capacity split spanning multiple batches, per-series
priors, the exogenous-regressor universe (the M5 SNAP path), and the trivial
k0/k1 edges.
"""
import numpy as np
import pandas as pd

from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Regressors
from DLMAX.ffs_core import AutoFFSUniverse, _load_batch_state


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



# --------------------------------------------------------------------------
# plain (no-exog) panel
# --------------------------------------------------------------------------
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


def _fc(uni):
    return (uni.forecast(h=12, level=[80, 95])
            .sort_values(["unique_id", "ds"]).reset_index(drop=True))


def _assert_same(ref, new, expect_ids):
    assert set(new.list_series()) == set(ref.list_series()) == set(expect_ids)
    for sid in new.list_series():
        assert (new._manifest.loc[sid, "last_ds"]
                == ref._manifest.loc[sid, "last_ds"])
    fc_ref, fc_new = _fc(ref), _fc(new)
    assert set(fc_new["unique_id"]) == set(fc_ref["unique_id"])
    vcols = [c for c in fc_ref.columns if c not in ("unique_id", "ds")]
    np.testing.assert_allclose(
        fc_new[vcols].to_numpy(), fc_ref[vcols].to_numpy(),
        rtol=1e-12, atol=1e-12)


ADDS = [(f"a{i}", 10 + i) for i in range(5)]   # five new series


def test_add_series_many_matches_sequential(tmp_path):
    """Batched add of 5 series (cap 4 -> spans 2 batches) == 5 sequential adds."""
    base = ["s0", "s1", "s2"]

    ref = _legacy_create(
        str(tmp_path / "ref"), season_length=7, warmup_steps=14, max_batch_size=4)
    ref.fit(_panel(base), freq="D", h_template=12)
    for sid, seed in ADDS:
        ref.add_series(sid, _series_df(sid, seed=seed))

    new = _legacy_create(
        str(tmp_path / "new"), season_length=7, warmup_steps=14, max_batch_size=4)
    new.fit(_panel(base), freq="D", h_template=12)
    new.add_series_many([sid for sid, _ in ADDS],
                        [_series_df(sid, seed=seed) for sid, seed in ADDS])

    _assert_same(ref, new, base + [a for a, _ in ADDS])
    # Same batch layout: ceil((3+5)/4) = 2 live batches, none over cap.
    man = new._manifest
    assert man.loc[man["active"], "batch_id"].nunique() == 2
    assert int(man.loc[man["active"]].groupby("batch_id").size().max()) == 4


def test_add_series_many_no_open_batch(tmp_path):
    """When the base batch is already full, the batched add opens fresh batches
    only (no open batch to reuse) — still matches sequential."""
    base = ["s0", "s1", "s2", "s3"]      # == cap, so the base batch is full

    ref = _legacy_create(
        str(tmp_path / "ref"), season_length=7, warmup_steps=14, max_batch_size=4)
    ref.fit(_panel(base), freq="D", h_template=12)
    for sid, seed in ADDS:
        ref.add_series(sid, _series_df(sid, seed=seed))

    new = _legacy_create(
        str(tmp_path / "new"), season_length=7, warmup_steps=14, max_batch_size=4)
    new.fit(_panel(base), freq="D", h_template=12)
    new.add_series_many([sid for sid, _ in ADDS],
                        [_series_df(sid, seed=seed) for sid, seed in ADDS])

    _assert_same(ref, new, base + [a for a, _ in ADDS])


def test_add_series_many_unbounded_capacity(tmp_path):
    """max_batch_size=None -> single consolidated batch; batched == sequential."""
    base = ["s0", "s1", "s2"]

    ref = _legacy_create(
        str(tmp_path / "ref"), season_length=7, warmup_steps=14, max_batch_size=None)
    ref.fit(_panel(base), freq="D", h_template=12)
    for sid, seed in ADDS:
        ref.add_series(sid, _series_df(sid, seed=seed))

    new = _legacy_create(
        str(tmp_path / "new"), season_length=7, warmup_steps=14, max_batch_size=None)
    new.fit(_panel(base), freq="D", h_template=12)
    new.add_series_many([sid for sid, _ in ADDS],
                        [_series_df(sid, seed=seed) for sid, seed in ADDS])

    _assert_same(ref, new, base + [a for a, _ in ADDS])
    assert new._manifest.loc[new._manifest["active"], "batch_id"].nunique() == 1


def test_add_series_many_per_series_error_nu0(tmp_path):
    """Per-series priors are dispatched to the right series: a list of distinct
    error_nu0 values must give the same result as the matching sequential adds."""
    base = ["s0", "s1", "s2"]
    nu0s = [0.0, 1.0, 5.0, None, 2.0]        # mix incl. a None (diffuse default)

    ref = _legacy_create(
        str(tmp_path / "ref"), season_length=7, warmup_steps=14, max_batch_size=4)
    ref.fit(_panel(base), freq="D", h_template=12)
    for (sid, seed), e0 in zip(ADDS, nu0s):
        ref.add_series(sid, _series_df(sid, seed=seed), error_nu0=e0)

    new = _legacy_create(
        str(tmp_path / "new"), season_length=7, warmup_steps=14, max_batch_size=4)
    new.fit(_panel(base), freq="D", h_template=12)
    new.add_series_many([sid for sid, _ in ADDS],
                        [_series_df(sid, seed=seed) for sid, seed in ADDS],
                        error_nu0=nu0s)

    _assert_same(ref, new, base + [a for a, _ in ADDS])


def test_add_series_many_then_update_streams(tmp_path):
    """After a batched add, the universe keeps streaming correctly."""
    base = ["s0", "s1", "s2"]
    uni = _legacy_create(
        str(tmp_path / "u"), season_length=7, warmup_steps=14, max_batch_size=4)
    uni.fit(_panel(base), freq="D", h_template=12)
    uni.add_series_many([sid for sid, _ in ADDS],
                        [_series_df(sid, seed=seed) for sid, seed in ADDS])

    all_ids = base + [a for a, _ in ADDS]
    nxt = pd.DatetimeIndex([pd.Timestamp("2020-01-01") + pd.Timedelta(days=40)])
    upd = pd.concat([pd.DataFrame({"unique_id": s, "ds": nxt, "y": [9.0]})
                     for s in all_ids], ignore_index=True)
    uni.update(upd)
    fc = uni.forecast(h=12, level=[80, 95])
    assert fc["unique_id"].nunique() == len(all_ids)
    vcols = [c for c in fc.columns if c not in ("unique_id", "ds")]
    assert np.isfinite(fc[vcols].to_numpy()).all()


def test_add_series_many_k1_and_k0(tmp_path):
    """k==1 delegates to add_series (identical); k==0 is a no-op."""
    base = ["s0", "s1", "s2"]
    ref = _legacy_create(
        str(tmp_path / "ref"), season_length=7, warmup_steps=14, max_batch_size=4)
    ref.fit(_panel(base), freq="D", h_template=12)
    ref.add_series("a0", _series_df("a0", seed=10))

    new = _legacy_create(
        str(tmp_path / "new"), season_length=7, warmup_steps=14, max_batch_size=4)
    new.fit(_panel(base), freq="D", h_template=12)
    new.add_series_many([], [])                       # no-op
    assert set(new.list_series()) == set(base)
    new.add_series_many(["a0"], [_series_df("a0", seed=10)])   # k==1
    _assert_same(ref, new, base + ["a0"])


def test_add_series_many_rejects_bad_input(tmp_path):
    import pytest
    uni = _legacy_create(
        str(tmp_path / "u"), season_length=7, warmup_steps=14, max_batch_size=4)
    uni.fit(_panel(["s0", "s1", "s2"]), freq="D", h_template=12)
    with pytest.raises(ValueError, match="same length"):
        uni.add_series_many(["a0", "a1"], [_series_df("a0")])
    with pytest.raises(ValueError, match="duplicated"):
        uni.add_series_many(["a0", "a0"],
                            [_series_df("a0"), _series_df("a0")])
    with pytest.raises(ValueError, match="already in universe"):
        uni.add_series_many(["s0", "a1"],
                            [_series_df("s0"), _series_df("a1")])


# --------------------------------------------------------------------------
# exogenous-regressor universe (the M5 SNAP path)
# --------------------------------------------------------------------------
def _exog_builder(init_data, h, ctx):
    n = init_data.shape[1]
    dlm = DLM(family="Gaussian", n_series=n)
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.95, damping=0.9))
    dlm.add_component(
        Regressors(name="reg", n_regs=1, disc_rate=0.999, damping=1.0, x_scale=0.5))
    dlm.set_error(disc_rate=1.0, power=[1.0, 0.5], nu0=1)
    models, desc = dlm.compile_universe(init_data, h=h, warmup_steps=ctx.warmup_steps)
    desc = desc.copy()
    desc["Class"] = ["v" + str(p) for p in desc["error.power"]]
    return models, desc.set_index("key")


def _x_of(ds):
    return (pd.DatetimeIndex(np.asarray(ds)).weekday < 2).astype(float)


def _exog_provider(srs_ids, ds):
    x = _x_of(ds)
    return np.broadcast_to(x[:, None, None], (len(x), len(srs_ids), 1)).copy()


def _exog_df(ids, dates, seed=0):
    rng = np.random.default_rng(seed)
    x = _x_of(dates)
    frames = [pd.DataFrame({"unique_id": s, "ds": dates,
                            "y": 10.0 + 5.0 * x + rng.normal(0, 0.4, len(dates))})
              for s in ids]
    return pd.concat(frames, ignore_index=True)


def _make_exog_uni(path):
    dates = pd.date_range("2021-01-01", periods=120, freq="D")
    uni = _legacy_create(
        path, season_length=None, max_batch_size=6, warmup_steps=14,
        universe_builder=_exog_builder, exog_provider=_exog_provider)
    uni.fit(_exog_df([f"s{i}" for i in range(4)], dates), freq="D")
    return uni, dates


def test_add_series_many_exog_matches_sequential(tmp_path):
    """SNAP path: batched add through an exogenous-regressor universe matches
    sequential, and the new series keep the regression tail."""
    refu, dates = _make_exog_uni(str(tmp_path / "ref"))
    newu, _ = _make_exog_uni(str(tmp_path / "new"))

    new_ids = ["n0", "n1", "n2", "n3", "n4"]    # 4 base + 5 new, cap 6 -> 2 batches
    hist = {s: _exog_df([s], dates, seed=100 + i)[["ds", "y"]]
            for i, s in enumerate(new_ids)}

    for s in new_ids:
        refu.add_series(s, hist[s])
    newu.add_series_many(new_ids, [hist[s] for s in new_ids])

    # regression tail preserved on a batch holding new series
    bid = int(newu._manifest.loc["n4", "batch_id"])
    st, _ = _load_batch_state(newu._batch_path(bid))
    assert st.multi.exog_regressors is True and st.multi.n_regressors == 1
    assert np.asarray(st.multi.reg_mask).all()

    # bit-identical forecasts (h-step, exog re-supplied on both)
    fr = refu.forecast(h=14).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    fn = newu.forecast(h=14).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    assert set(fn["unique_id"]) == set(fr["unique_id"])
    np.testing.assert_allclose(fn["AutoFFS"].to_numpy(), fr["AutoFFS"].to_numpy(),
                               rtol=1e-11, atol=1e-11)

"""Multi-block ``add_series_many`` is equivalent to k sequential ``add_series``.

``add_series_many`` had a batched fast path only on the SINGLE-block grid
branch; a multi-block universe fell back to looping ``add_series``, which
reloads and rewrites the whole batch per series. At production batch sizes that
dominates — a 4000-slot two-block M5 batch is ~281 MB and ~3.5 s to rewrite,
against ~2.4 s to fit the series — so an origin adding 34 launchers paid the
rewrite 34 times.

``_multiblock_add_series_many`` fits each newcomer exactly as ``add_series``
does, then writes each affected batch ONCE. The slot assignment order is
unchanged, so the resulting universe matches the sequential path.

Also covers the union-allocator bug the batching exposed: ``add_series`` fitted
the newcomer's union carry WITHOUT forwarding ``union_learn_dma`` /
``union_dma_prior``, so with SGDDMA on, a fixed ``AllocatorState`` came back
against a stored SGDDMA carry and ``_union_set_slot``'s ``tree_map`` raised
"Expected tuple, got AllocatorState". The ``union_learn_dma=True`` cases below
fail outright without that fix.
"""
import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs_core import AutoFFSUniverse

PERIOD, WARMUP, H = 7, 14, 6
N = 60


def _panel(ids, n=N, season=PERIOD):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.arange(n)
    rows = []
    for i, sid in enumerate(ids):
        y = (10 + 0.1 * rng + 3 * np.sin(2 * np.pi * rng / season)
             + np.random.default_rng(i).normal(0, 0.3, n))
        rows.append(pd.DataFrame({"unique_id": sid, "ds": dates, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _history(sid, seed):
    """One newcomer's own history, in the (ds, y) shape add_series takes."""
    dates = pd.date_range("2020-01-01", periods=N, freq="D")
    rng = np.arange(N)
    y = (8 + 0.15 * rng + 2 * np.sin(2 * np.pi * rng / PERIOD)
         + np.random.default_rng(seed).normal(0, 0.3, N))
    return pd.DataFrame({"ds": dates, "y": y})


def _blocks():
    """Two blocks: the classic A/M wing and a compound-Poisson one."""
    return [dict(period=PERIOD, var_powers=None, warmup=WARMUP, offset=1.0),
            dict(period=PERIOD, var_powers=[0.25], warmup=WARMUP, offset=1.0)]


def _build(path, seed_ids, *, learn_dma, cap):
    u = AutoFFSUniverse.create(path, season_length=PERIOD, grid_blocks=_blocks(),
                               union_learn_dma=learn_dma, max_batch_size=cap)
    u.fit(_panel(seed_ids), freq="D", h_template=H)
    assert u._multiblock
    return u


def _fc(u):
    return (u.forecast(h=H).sort_values(["unique_id", "ds"])
            .reset_index(drop=True))


def _run_pair(tmp_path, *, learn_dma, cap, n_new=4):
    """Same universe, same newcomers, batched vs sequential. Returns both
    forecast frames."""
    seed_ids = [f"s{i}" for i in range(5)]
    new_ids = [f"n{i}" for i in range(n_new)]
    hists = [_history(s, 100 + i) for i, s in enumerate(new_ids)]

    ub = _build(str(tmp_path / "batched"), seed_ids, learn_dma=learn_dma, cap=cap)
    ub.add_series_many(new_ids, hists)

    us = _build(str(tmp_path / "seq"), seed_ids, learn_dma=learn_dma, cap=cap)
    for uid, h in zip(new_ids, hists):
        us.add_series(uid, h)

    return _fc(ub), _fc(us), ub, us


@pytest.mark.parametrize("learn_dma", [False, True])
def test_batched_equals_sequential(tmp_path, learn_dma):
    """Batched adds land the same forecasts as one-at-a-time adds."""
    fb, fs, ub, us = _run_pair(tmp_path, learn_dma=learn_dma, cap=500)
    assert list(fb["unique_id"]) == list(fs["unique_id"])
    alias = ub.alias
    np.testing.assert_allclose(fb[alias].to_numpy(), fs[alias].to_numpy(),
                               rtol=0, atol=0)
    np.testing.assert_allclose(fb[f"{alias}-sd"].to_numpy(),
                               fs[f"{alias}-sd"].to_numpy(), rtol=0, atol=0)


@pytest.mark.parametrize("learn_dma", [False, True])
def test_batched_equals_sequential_capacity_split(tmp_path, learn_dma):
    """Newcomers overflowing the open batch spill into a fresh one identically.

    cap=6 with 5 seeded series leaves ONE free slot, so of 4 newcomers the
    first fills it and the other 3 open a new batch — exercising both the
    fill-open-batch and the fresh-batch branches in one go.
    """
    fb, fs, ub, us = _run_pair(tmp_path, learn_dma=learn_dma, cap=6)
    assert ub._manifest["batch_id"].nunique() == 2
    assert (ub._manifest["batch_id"].value_counts().sort_index().tolist()
            == us._manifest["batch_id"].value_counts().sort_index().tolist())
    alias = ub.alias
    np.testing.assert_allclose(fb[alias].to_numpy(), fs[alias].to_numpy(),
                               rtol=0, atol=0)


def test_batched_with_per_series_priors(tmp_path):
    """Per-block, per-series component_priors / wing_centre reach the right
    newcomer — the M5 sibling warm-start shape."""
    seed_ids = [f"s{i}" for i in range(5)]
    new_ids = ["n0", "n1"]
    hists = [_history(s, 200 + i) for i, s in enumerate(new_ids)]
    # per series -> per block (2 blocks); None means "diffuse for that block"
    wc = [[0.2, None], [None, 0.4]]

    ub = _build(str(tmp_path / "b"), seed_ids, learn_dma=True, cap=500)
    ub.add_series_many(new_ids, hists, wing_centre=wc)

    us = _build(str(tmp_path / "s"), seed_ids, learn_dma=True, cap=500)
    for uid, h, w in zip(new_ids, hists, wc):
        us.add_series(uid, h, wing_centre=w)

    alias = ub.alias
    np.testing.assert_allclose(_fc(ub)[alias].to_numpy(),
                               _fc(us)[alias].to_numpy(), rtol=0, atol=0)


def test_k0_and_k1_edges(tmp_path):
    """k=0 is a no-op; k=1 delegates to add_series and matches it."""
    seed_ids = [f"s{i}" for i in range(5)]
    u0 = _build(str(tmp_path / "k0"), seed_ids, learn_dma=True, cap=500)
    before = _fc(u0)[u0.alias].to_numpy().copy()
    u0.add_series_many([], [])
    np.testing.assert_allclose(_fc(u0)[u0.alias].to_numpy(), before,
                               rtol=0, atol=0)

    h1 = _history("n0", 300)
    u1 = _build(str(tmp_path / "k1a"), seed_ids, learn_dma=True, cap=500)
    u1.add_series_many(["n0"], [h1])
    u2 = _build(str(tmp_path / "k1b"), seed_ids, learn_dma=True, cap=500)
    u2.add_series("n0", h1)
    np.testing.assert_allclose(_fc(u1)[u1.alias].to_numpy(),
                               _fc(u2)[u2.alias].to_numpy(), rtol=0, atol=0)


def test_update_and_reopen_after_batched_add(tmp_path):
    """The batched-add carry survives an update + reopen, still matching the
    sequential path — i.e. what was written is a valid resumable batch."""
    fb, fs, ub, us = _run_pair(tmp_path, learn_dma=True, cap=500)
    step = _panel([f"s{i}" for i in range(5)] + [f"n{i}" for i in range(4)],
                  n=N + 1).groupby("unique_id").tail(1)
    for u in (ub, us):
        u.update(step)
    alias = ub.alias
    np.testing.assert_allclose(_fc(ub)[alias].to_numpy(),
                               _fc(us)[alias].to_numpy(), rtol=0, atol=0)

    ur = AutoFFSUniverse.open(str(tmp_path / "batched"))
    np.testing.assert_allclose(_fc(ur)[alias].to_numpy(),
                               _fc(ub)[alias].to_numpy(), rtol=0, atol=0)

"""End-to-end exogenous regressors through the AutoFFSUniverse streaming API.

The low-level exog machinery (``uv_dlm`` / ``multi_model_dlm.scan_filter`` /
``forecast_exog`` / the ``Regressors`` component) is covered by
``test_multi_regression.py``. This file exercises the *streaming wrapper* that
plumbs a caller-supplied regressor design through ``fit`` / ``update`` /
``forecast`` / ``add_series`` and across a save/reopen cycle — the path the M5
SNAP run depends on.

A custom ``universe_builder`` emits an exogenous regression tail (LocalTrend +
``Regressors``); a matching ``exog_provider`` supplies a known 0/1 indicator
keyed on the calendar date. The synthetic series are ``y = base + beta*x +
noise``, so a correct pipeline must (a) learn ``beta`` into the regression
coefficient and (b) lift the forecast on future days where ``x == 1``.
"""

import numpy as np
import pandas as pd

from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Regressors
from DLMAX.ffs_core import AutoFFSUniverse, _load_batch_state


def _legacy_create(*args, **kwargs):
    """Create a universe on the LEGACY multi-model path.

    AutoFFSUniverse now defaults to the wing grid, which persists batches in the
    block format and carries no exog / defragment / fixed-capacity face. These
    tests exercise those legacy features, so they opt out explicitly — the same
    grid_period=None escape hatch documented for users.
    """
    kwargs.setdefault("grid_period", None)
    return AutoFFSUniverse.create(*args, **kwargs)



BASE = 10.0
BETA = 5.0
N_DAYS = 120
H = 14


# --- top-level builder + provider (re-suppliable to open(); must be picklable) ---
def exog_builder(init_data, h, ctx):
    """LocalTrend + a single exogenous regressor, swept over var_power to give a
    2-model DMA universe. Class = var_power so the between-class layer is real."""
    n = init_data.shape[1]
    dlm = DLM(family="Gaussian", n_series=n)
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.95, damping=0.9))
    dlm.add_component(
        Regressors(name="reg", n_regs=1, disc_rate=0.999, damping=1.0, x_scale=0.5)
    )
    dlm.set_error(disc_rate=1.0, power=[1.0, 0.5], nu0=1)
    models, desc = dlm.compile_universe(
        init_data, h=h, warmup_steps=ctx.warmup_steps
    )
    desc = desc.copy()
    desc["Class"] = ["v" + str(p) for p in desc["error.power"]]
    desc = desc.set_index("key")
    return models, desc


def _x_of(ds):
    """Known indicator: 1 on Mon/Tue, else 0 — a pure function of the date."""
    return (pd.DatetimeIndex(np.asarray(ds)).weekday < 2).astype(float)


def exog_provider(srs_ids, ds):
    """(T, n_series, 1) design — the indicator broadcast across series (shared,
    like SNAP-by-state). Agnostic to whether ``ds`` is past or future."""
    x = _x_of(ds)
    return np.broadcast_to(x[:, None, None], (len(x), len(srs_ids), 1)).copy()


def _make_df(series_ids, dates, seed=0):
    rng = np.random.default_rng(seed)
    x = _x_of(dates)
    frames = []
    for s in series_ids:
        y = BASE + BETA * x + rng.normal(0, 0.4, len(dates))
        frames.append(pd.DataFrame({"unique_id": s, "ds": dates, "y": y}))
    return pd.concat(frames, ignore_index=True)


def _fit_universe(path):
    dates = pd.date_range("2021-01-01", periods=N_DAYS, freq="D")
    df = _make_df([f"s{i}" for i in range(4)], dates)
    uni = _legacy_create(
        path,
        season_length=None,
        max_batch_size=10,
        warmup_steps=14,
        universe_builder=exog_builder,
        exog_provider=exog_provider,
    )
    uni.fit(df, freq="D")
    return uni, dates


def test_exog_universe_fit_sets_flag(tmp_path):
    uni, _ = _fit_universe(str(tmp_path / "uni"))
    bid = int(uni._manifest["batch_id"].iloc[0])
    st, _ = _load_batch_state(uni._batch_path(bid))
    assert st.multi.n_regressors == 1
    assert st.multi.exog_regressors is True
    # reg_mask covers every (model, series) slot and is all-True (every model
    # carries the exogenous tail).
    assert np.asarray(st.multi.reg_mask).all()


def test_exog_drives_forecast(tmp_path):
    """The forecast must sit ~BETA higher on future days where x == 1."""
    uni, dates = _fit_universe(str(tmp_path / "uni"))
    fc = uni.forecast(h=H).sort_values(["unique_id", "ds"])
    fx = _x_of(fc["ds"])
    loc = fc["AutoFFS"].to_numpy()
    hi = loc[fx == 1].mean()
    lo = loc[fx == 0].mean()
    # Coefficient learned (~BETA=5); generous slack for discounting / noise.
    assert hi - lo > 3.0, (hi, lo)
    # Sanity: the low (x==0) level tracks BASE, not BASE+BETA.
    assert abs(lo - BASE) < 2.0, lo


def test_exog_roundtrips_through_reopen(tmp_path):
    path = str(tmp_path / "uni")
    uni, _ = _fit_universe(path)
    fc1 = uni.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()

    uni2 = AutoFFSUniverse.open(
        path, universe_builder=exog_builder, exog_provider=exog_provider
    )
    bid = int(uni2._manifest["batch_id"].iloc[0])
    st, _ = _load_batch_state(uni2._batch_path(bid))
    assert st.multi.exog_regressors is True
    assert st.multi.n_regressors == 1
    # Forecast after a clean reopen reproduces the pre-save forecast.
    fc2 = uni2.forecast(h=H).sort_values(["unique_id", "ds"])["AutoFFS"].to_numpy()
    np.testing.assert_allclose(fc1, fc2, rtol=1e-9, atol=1e-9)


def test_exog_update_and_add_series(tmp_path):
    path = str(tmp_path / "uni")
    uni, dates = _fit_universe(path)

    # Stream one more day for all existing series.
    next_day = dates[-1] + pd.Timedelta(days=1)
    upd = _make_df([f"s{i}" for i in range(4)], pd.DatetimeIndex([next_day]), seed=1)
    uni.update(upd)

    # Add a brand-new series (append-on-add path: must keep the exog tail).
    new_dates = pd.date_range("2021-01-01", periods=N_DAYS + 1, freq="D")
    new_hist = _make_df(["s_new"], new_dates, seed=2)[["ds", "y"]]
    uni.add_series("s_new", new_hist)

    bid = int(uni._manifest.loc["s_new", "batch_id"])
    st, _ = _load_batch_state(uni._batch_path(bid))
    assert st.multi.exog_regressors is True
    assert st.multi.n_regressors == 1
    assert np.asarray(st.multi.reg_mask).all()

    fc = uni.forecast(h=H)
    assert "s_new" in set(fc["unique_id"])
    # The new series still responds to the exogenous driver.
    sub = fc[fc["unique_id"] == "s_new"].sort_values("ds")
    fx = _x_of(sub["ds"])
    loc = sub["AutoFFS"].to_numpy()
    assert loc[fx == 1].mean() - loc[fx == 0].mean() > 2.0


def test_exog_provider_not_handed_pad_slots(tmp_path):
    """A uid-KEYED provider (like SNAP's unique_id->column lookup) must never be
    handed capacity-padding slot ids. With max_batch_size > n_series the batch is
    padded to capacity, so the loaded state's srs_ids include ``__pad_N``;
    _materialise_exog must filter those before calling the provider (else
    KeyError on '__pad_0'). Regression for the full-M5 padding bug."""
    dates = pd.date_range("2021-01-01", periods=N_DAYS, freq="D")
    ids = [f"s{i}" for i in range(5)]
    col_of = {u: i for i, u in enumerate(ids)}
    seen = set()

    def keyed_provider(srs_ids, ds):
        seen.update(map(str, srs_ids))
        # mimic SNAP: positional lookup that KeyErrors on an unknown (pad) id
        _ = np.array([col_of[str(u)] for u in srs_ids])
        x = _x_of(ds)
        return np.broadcast_to(x[:, None, None], (len(x), len(srs_ids), 1)).copy()

    uni = _legacy_create(
        str(tmp_path / "u"), season_length=None, max_batch_size=8,
        warmup_steps=14, universe_builder=exog_builder, exog_provider=keyed_provider)
    uni.fit(_make_df(ids, dates), freq="D")     # 5 series, cap 8 -> padded to 8
    # one update + a forecast both go through the padded state's srs_ids
    nd = dates[-1] + pd.Timedelta(days=1)
    uni.update(_make_df(ids, pd.DatetimeIndex([nd]), seed=3))
    fc = uni.forecast(h=H)
    assert len(fc) == len(ids) * H
    assert not any(s.startswith("__pad") for s in seen), \
        f"provider was handed pad slots: {[s for s in seen if s.startswith('__pad')]}"


def test_exog_universe_requires_provider_on_forecast(tmp_path):
    """Reopening an exog universe without re-supplying the provider must fail
    loudly at forecast (rather than silently dropping the regressor)."""
    path = str(tmp_path / "uni")
    _fit_universe(path)
    uni = AutoFFSUniverse.open(path, universe_builder=exog_builder)  # no provider
    import pytest

    with pytest.raises(ValueError, match="exog"):
        uni.forecast(h=H)

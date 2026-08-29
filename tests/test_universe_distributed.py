"""Distributed per-period universe step == in-process, bit-for-bit.

``AutoFFSUniverse.update`` / ``.forecast`` distribute their per-batch work across
Dask workers (path-based: each worker loads/saves the batch file on shared
storage; only KB-scale payloads cross the wire). This must give *identical*
results to the in-process path — the whole point of the directory-backed design.

We fit once in-process, clone the universe directory, then drive one clone
in-process and the other through a real (local) Dask cluster, and assert the
forecasts match exactly. The builder/exog_provider are only ever used on the
head (update/forecast load state on the worker and exog is materialised
head-side), so defining them in this test module is fine — workers run only
ffs_core internals.
"""
import shutil

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Regressors
from DLMAX.ffs_core import AutoFFSUniverse


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


N_DAYS, H, BASE, BETA = 120, 14, 10.0, 5.0


def exog_builder(init_data, h, ctx):
    n = init_data.shape[1]
    dlm = DLM(family="Gaussian", n_series=n)
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.95, damping=0.9))
    dlm.add_component(Regressors(name="reg", n_regs=1, disc_rate=0.999,
                                 damping=1.0, x_scale=0.5))
    dlm.set_error(disc_rate=1.0, power=[1.0, 0.5], nu0=1)
    models, desc = dlm.compile_universe(init_data, h=h, warmup_steps=ctx.warmup_steps)
    desc = desc.copy()
    desc["Class"] = ["v" + str(p) for p in desc["error.power"]]
    return models, desc.set_index("key")


def _x_of(ds):
    return (pd.DatetimeIndex(np.asarray(ds)).weekday < 2).astype(float)


def exog_provider(srs_ids, ds):
    x = _x_of(ds)
    return np.broadcast_to(x[:, None, None], (len(x), len(srs_ids), 1)).copy()


def _make_df(series_ids, dates, seed=0):
    rng = np.random.default_rng(seed)
    x = _x_of(dates)
    frames = [
        pd.DataFrame({"unique_id": s, "ds": dates,
                      "y": BASE + BETA * x + rng.normal(0, 0.4, len(dates))})
        for s in series_ids
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def dask_client():
    distributed = pytest.importorskip("dask.distributed")
    from DLMAX.ffs_core import AutoFFS

    cluster = distributed.LocalCluster(
        n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
    )
    client = distributed.Client(cluster)
    AutoFFS.setup_workers(client, compute="cpu")   # JAX x64 on every worker
    yield client
    client.close()
    cluster.close()


def _fit(path):
    dates = pd.date_range("2021-01-01", periods=N_DAYS, freq="D")
    uni = _legacy_create(
        path, season_length=None, max_batch_size=3, warmup_steps=14,
        universe_builder=exog_builder, exog_provider=exog_provider,
    )
    uni.fit(_make_df([f"s{i}" for i in range(8)], dates), freq="D")


def _step_and_forecast(path, dask_client=None):
    dates = pd.date_range("2021-01-01", periods=N_DAYS, freq="D")
    uni = AutoFFSUniverse.open(
        path, universe_builder=exog_builder, exog_provider=exog_provider,
        dask_client=dask_client,
    )
    next_day = dates[-1] + pd.Timedelta(days=1)
    uni.update(_make_df([f"s{i}" for i in range(8)],
                        pd.DatetimeIndex([next_day]), seed=7))
    return uni.forecast(h=H).sort_values(["unique_id", "ds"]).reset_index(drop=True)


def test_distributed_update_forecast_matches_in_process(tmp_path, dask_client):
    base = str(tmp_path / "base")
    _fit(base)
    ip_dir, dist_dir = str(tmp_path / "ip"), str(tmp_path / "dist")
    shutil.copytree(base, ip_dir)
    shutil.copytree(base, dist_dir)

    fc_ip = _step_and_forecast(ip_dir, dask_client=None)
    fc_dist = _step_and_forecast(dist_dir, dask_client=dask_client)

    assert list(fc_ip["unique_id"]) == list(fc_dist["unique_id"])
    np.testing.assert_array_equal(fc_ip["ds"].to_numpy(), fc_dist["ds"].to_numpy())
    np.testing.assert_allclose(
        fc_ip["AutoFFS"].to_numpy(), fc_dist["AutoFFS"].to_numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(
        fc_ip["AutoFFS-sd"].to_numpy(), fc_dist["AutoFFS-sd"].to_numpy(), rtol=0, atol=0)

"""Regression guard: multi_model_dlm.forecast (nested vmap, model-only operands
shared across series) == _forecast_flat (flat vmap, operands replicated x q).

forecast() was changed to a nested vmap so it never materialises the
(nm*q, h, k, k) GH tensor (the forecast-stage memory blow-up and ~half the
streamed bytes on this bandwidth-bound step). It must stay numerically
identical to the reference flat path retained as _forecast_flat. If this
fails, the two have diverged — fix before trusting any multi-step predictive
output.
"""
import numpy as np
import pandas as pd

from DLMAX.ffs.dlm_builder import DLM, LocalTrend
from DLMAX.dlm_core import multi_model_dlm


def _build_multi(n_series=5, h=12):
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(size=(40, n_series)).cumsum(axis=0) + 100.0)
    dlm = DLM(family="Gaussian", n_series=n_series)
    # Damped trend exercises disc_rates AND disc_rates_damped (-> G varies).
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.95, damping=0.9))
    # Swept power -> nm > 1, including the log-scale (power=0) path used by M5.
    dlm.set_error(disc_rate=1.0, power=[0, 1], nu0=1)
    models, _ = dlm.compile_universe(panel, h=h)
    return multi_model_dlm(models), h


def test_nested_forecast_matches_flat():
    multi, h = _build_multi()
    # exercise both vmap axes
    assert multi.nm > 1 and multi.q > 1 and multi.GH is not None

    f_new, q_new = (np.asarray(x) for x in multi.forecast(h))
    f_ref, q_ref = (np.asarray(x) for x in multi._forecast_flat(h))

    assert f_new.shape == f_ref.shape == (multi.nm, multi.q, h)
    np.testing.assert_allclose(f_new, f_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(q_new, q_ref, rtol=1e-9, atol=1e-9)

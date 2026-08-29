"""Regression guard: uv_dlm.forecast and multi_model_dlm.forecast agree.

The forecast-stage system-noise covariance W must be computed identically in
both paths:  W = ((1 - δ·δ_damped) / (δ·δ_damped)) · C, applied as a diagonal.
A past bug in ``uv_dlm.forecast`` used ``W = C * disc_rates`` (wrong formula,
missing the damped term, and a column-scale rather than a diagonal), which
inflated multi-step predictive variances. This test builds the same single
model both ways and asserts the h-step forecast means and variances agree to
numerical precision.

If this fails, the two forecast implementations have diverged again — fix the
W computation before trusting any multi-step predictive variance.
"""

import numpy as np
import pandas as pd

from DLMAX.ffs.dlm_builder import DLM, LocalTrend
from DLMAX.dlm_core import multi_model_dlm


def test_uv_vs_multi_forecast_W():
    rng = np.random.default_rng(0)
    n_series = 3
    h = 18
    # A damped-trend model exercises both disc_rates and disc_rates_damped.
    panel = pd.DataFrame(rng.normal(size=(36, n_series)).cumsum(axis=0) + 100.0)

    dlm = DLM(family="Gaussian", n_series=n_series)
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.95, damping=0.9))
    # power as a 1-element list -> a single-model "universe" (compile_universe
    # requires at least one swept dim; the dict form is what multi_model_dlm wants).
    dlm.set_error(disc_rate=1.0, power=[1], nu0=1)

    models, _desc = dlm.compile_universe(panel, h=h)
    assert len(models) == 1, f"expected a single model, got {len(models)}"

    u = next(iter(models.values()))
    multi = multi_model_dlm(models)  # n_models == 1, same initial state as `u`

    # multi path: (nm, q, h) -> (n_series, h)
    f_multi, q_multi = multi.forecast(h=h)
    f_multi = np.asarray(f_multi)[0]
    q_multi = np.asarray(q_multi)[0]

    # uv path: fc_model fields are (h, n_series[, +pad]) -> (n_series, h)
    u.forecast(h)
    f_uv = np.asarray(u.fc_model["f"])[:, :n_series].T
    q_uv = np.asarray(u.fc_model["q"])[:, :n_series].T

    np.testing.assert_allclose(f_uv, f_multi, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(q_uv, q_multi, rtol=1e-9, atol=1e-9)

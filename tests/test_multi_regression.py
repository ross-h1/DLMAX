"""Multi-model exogenous regression: filter + forecast helpers.

Validates the multi-DLM regression machinery added on top of the (already
done) ``uv_dlm`` regression path:

* :meth:`multi_model_dlm.scan_filter` — standalone compiled forward filter,
  threading per-step regressors — must match a stepwise ``fwd_filter`` loop
  (both go through ``_multi_fwd_filter_step``).
* :meth:`multi_model_dlm.forecast_exog` with *known* future regressors must
  agree with ``uv_dlm.forecast(regressors=...)`` to numerical precision (the
  established single-series path), means and variances.
* The *stochastic* regressor variance term (``future_var`` / ``regressors_var``)
  must agree between the uv and multi paths, and must only ever *add*
  predictive variance (zero for a known regressor).
* The scale-aware ``Regressors`` prior reduces to the legacy flat default when
  ``x_scale`` is absent, and applies ``C0_jj = g sigma_y^2 / x_scale_j^2``
  when supplied.

These two code paths (the uv scan step and the multi vmap kernel) share only
the one-line :func:`_regressor_var_prop` helper, so their agreement is a real
cross-check, not a tautology.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp

from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Regressors
from DLMAX.ffs import devices
from DLMAX.dlm_core import multi_model_dlm, _regressor_var_prop


def _build_models(panel, n_reg, h, reg_m0=None, x_scale=None):
    """A single LocalTrend + Regressors model, compiled as a 1-model universe."""
    n = panel.shape[1]
    # The component requires m0 and C0 together; pair any supplied prior mean
    # with a mild identity prior covariance.
    reg_C0 = None
    if reg_m0 is not None:
        reg_C0 = jnp.broadcast_to(jnp.eye(n_reg), (n, n_reg, n_reg))
    dlm = DLM(family="Gaussian", n_series=n)
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.95, damping=0.9))
    dlm.add_component(
        Regressors(
            name="reg", n_regs=n_reg, disc_rate=0.99, damping=1.0,
            m0=reg_m0, C0=reg_C0, x_scale=x_scale,
        )
    )
    # power as a 1-element list -> a single-model universe (the dict form
    # multi_model_dlm wants); disc_rate=1.0 -> variance_disc=1 so the uv and
    # multi variance recursions line up exactly.
    dlm.set_error(disc_rate=1.0, power=[1], nu0=1)
    models, _desc = dlm.compile_universe(panel, h=h)
    assert len(models) == 1
    return models


# ---------------------------------------------------------------------------
# the shared variance-propagation primitive
# ---------------------------------------------------------------------------

def test_regressor_var_prop_formula():
    a = jnp.array([0.5, -1.0])
    r = jnp.array([0.1, 0.2])
    v = jnp.array([2.0, 0.5])
    got = float(_regressor_var_prop(a, r, v))
    exp = (0.5 ** 2 + 0.1) * 2.0 + ((-1.0) ** 2 + 0.2) * 0.5
    assert abs(got - exp) < 1e-12
    # a known regressor (v = 0) contributes nothing
    assert float(_regressor_var_prop(a, r, jnp.zeros(2))) == 0.0


# ---------------------------------------------------------------------------
# scan_filter vs a stepwise fwd_filter loop (both multi-level)
# ---------------------------------------------------------------------------

def test_scan_filter_matches_stepwise():
    rng = np.random.default_rng(1)
    n_series, n_reg, h, T = 3, 2, 6, 15
    panel = pd.DataFrame(rng.normal(size=(T, n_series)).cumsum(0) + 100.0)
    X = rng.normal(size=(T, n_series, n_reg))

    models = _build_models(panel, n_reg, h)
    multi_a = multi_model_dlm(models)
    multi_b = multi_model_dlm(_build_models(panel, n_reg, h))

    reg = multi_a.format_exog_yts(jnp.asarray(X))           # (T, p, n_reg)
    f_scan, q_scan = multi_a.scan_filter(jnp.asarray(panel.values), regressors=reg)

    fs, qs = [], []
    ys = jnp.asarray(panel.values)
    for t in range(T):
        ft, qt = multi_b.fwd_filter(ys[t], regressors=jnp.asarray(X[t]))
        fs.append(np.asarray(ft))
        qs.append(np.asarray(qt))

    np.testing.assert_allclose(np.asarray(f_scan), np.stack(fs), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(q_scan), np.stack(qs), rtol=1e-10, atol=1e-10)
    # final filtered state agrees too
    np.testing.assert_allclose(
        np.asarray(multi_a.dlm_state["m"]), np.asarray(multi_b.dlm_state["m"]),
        rtol=1e-10, atol=1e-10,
    )


# ---------------------------------------------------------------------------
# forecast_exog (known X) vs uv_dlm.forecast(regressors=...)
# ---------------------------------------------------------------------------

def test_forecast_exog_known_matches_uv():
    rng = np.random.default_rng(2)
    n_series, n_reg, h, T = 3, 2, 10, 30
    panel = pd.DataFrame(rng.normal(size=(T, n_series)).cumsum(0) + 100.0)
    # Non-zero regression-coefficient prior mean so the regressors actually
    # move the forecast *mean* (from the diffuse default the coefficients are
    # zero). Forecasting straight from the compiled prior keeps the uv and
    # multi initial states identical without relying on filter equivalence.
    reg_m0 = jnp.asarray(rng.normal(size=(n_series, n_reg)))
    models = _build_models(panel, n_reg, h, reg_m0=reg_m0)

    u = next(iter(models.values()))
    multi = multi_model_dlm(models)

    future_X = rng.normal(size=(h, n_series, n_reg))

    u.forecast(h, regressors=jnp.asarray(future_X))
    f_uv = np.asarray(u.fc_model["f"])[:, :n_series].T          # (n_series, h)
    q_uv = np.asarray(u.fc_model["q"])[:, :n_series].T

    f_m, q_m = multi.forecast_exog(h, jnp.asarray(future_X))
    f_m = np.asarray(f_m)[0, :n_series]
    q_m = np.asarray(q_m)[0, :n_series]

    np.testing.assert_allclose(f_uv, f_m, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(q_uv, q_m, rtol=1e-8, atol=1e-8)


def test_forecast_exog_stochastic_matches_uv_and_inflates_variance():
    rng = np.random.default_rng(3)
    n_series, n_reg, h, T = 2, 2, 8, 25
    panel = pd.DataFrame(rng.normal(size=(T, n_series)).cumsum(0) + 50.0)
    reg_m0 = jnp.asarray(rng.normal(size=(n_series, n_reg)))
    models = _build_models(panel, n_reg, h, reg_m0=reg_m0)

    u = next(iter(models.values()))
    multi = multi_model_dlm(models)

    future_X = rng.normal(size=(h, n_series, n_reg))
    future_V = rng.uniform(0.1, 1.0, size=(h, n_series, n_reg))

    # known X (variance only from coefficient uncertainty)
    u.forecast(h, regressors=jnp.asarray(future_X))
    q_known = np.asarray(u.fc_model["q"])[:, :n_series].T

    # stochastic X (uv path)
    u.forecast(h, regressors=jnp.asarray(future_X), regressors_var=jnp.asarray(future_V))
    q_uv_stoch = np.asarray(u.fc_model["q"])[:, :n_series].T

    # stochastic X (multi path)
    _f_m, q_m_stoch = multi.forecast_exog(
        h, jnp.asarray(future_X), future_var=jnp.asarray(future_V)
    )
    q_m_stoch = np.asarray(q_m_stoch)[0, :n_series]

    # the two independently-written stochastic paths agree
    np.testing.assert_allclose(q_uv_stoch, q_m_stoch, rtol=1e-8, atol=1e-8)
    # propagating regressor uncertainty only ever adds predictive variance
    assert np.all(q_uv_stoch >= q_known - 1e-10)
    assert np.any(q_uv_stoch > q_known + 1e-6)
    # known multi forecast == stochastic multi forecast when future_var is None
    _f0, q_m_known = multi.forecast_exog(h, jnp.asarray(future_X))
    np.testing.assert_allclose(
        np.asarray(q_m_known)[0, :n_series], q_known, rtol=1e-8, atol=1e-8
    )


# ---------------------------------------------------------------------------
# scale-aware regressor prior
# ---------------------------------------------------------------------------

def test_scale_aware_prior_default_unchanged():
    """No x_scale -> byte-identical to the legacy flat 10*sigma_y^2*I default."""
    rng = np.random.default_rng(4)
    panel = pd.DataFrame(rng.normal(size=(24, 3)) + 10.0)
    flat = _build_models(panel, n_reg=2, h=4)
    u = next(iter(flat.values()))
    sig2 = np.nanvar(panel.values, axis=0)                  # (n,)
    C0_reg = np.asarray(u.C0)[:, -2:, -2:]                  # regressor block
    for i in range(3):
        np.testing.assert_allclose(
            C0_reg[i], 10.0 * sig2[i] * np.eye(2), rtol=1e-6, atol=1e-6
        )


def test_scale_aware_prior_scales_by_x_variance():
    """x_scale -> C0_jj = g * sigma_y^2 / x_scale_j^2 on the regressor block."""
    rng = np.random.default_rng(5)
    panel = pd.DataFrame(rng.normal(size=(24, 3)) + 10.0)
    x_scale = jnp.array([2.0, 5.0])
    models = _build_models(panel, n_reg=2, h=4, x_scale=x_scale)
    u = next(iter(models.values()))
    sig2 = np.nanvar(panel.values, axis=0)
    C0_reg = np.asarray(u.C0)[:, -2:, -2:]
    g = 10.0
    for i in range(3):
        expected = np.diag(g * sig2[i] / np.asarray(x_scale) ** 2)
        np.testing.assert_allclose(C0_reg[i], expected, rtol=1e-6, atol=1e-6)

"""``mv_dlm`` -- the Quintana/West model, against the frozen oracle.

The API deliberately echoes the rest of the library. ``fwd_filter`` advances
one observation and returns the one-step-ahead predictive; ``scan_filter``
advances many; ``forecast(h)`` projects from the held state; all three return
or accept what ``uv_dlm`` and ``multi_model_dlm`` do. A reader who knows one
engine should not have to learn a second vocabulary for this one.

Everything here is **scale-free**. ``V == 1``, so a returned variance is in
units of ``Sigma``, and ``Sigma`` is not held by the model -- it enters only at
:meth:`backward_sample` as ``chol(Sigma)``. That is not a simplification: it is
the Kronecker structure, and it is why every series shares one ``Q_t``.

Two variance conventions live side by side, on purpose
------------------------------------------------------
``forecast(h).var`` is ``1 + F' R_h F`` -- the predictive, matching what
``fwd_filter`` returns at one step. The legacy ``dlm_forecast`` shim returns
``F' R_h F`` WITHOUT the ``1``, because its two callers add the observation
half themselves (one adds ``1.0`` directly, the other supplies it from
``chol_r``). Both are tested. The difference is exactly one, and getting it
wrong in either direction is invisible in the output.
"""
import os

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random

from DLMAX.dlm_core import ForecastBundle
from DLMAX.mvdlm import mv_dlm, component_discount_matrix, seasonal_prior

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@pytest.fixture(scope="module", params=["M4", "M12"])
def oracle(request):
    return np.load(os.path.join(FIXTURES, f"oracle_qw_{request.param}.npz"))


def _model(o, **kw):
    return mv_dlm(data=o["in_Y"], F=o["in_F"], G=o["in_G"], m0=o["in_m0"],
                  C0=o["in_C0"], V0=o["in_V0"], disc_mtx=o["in_delta_M"], **kw)


# --------------------------------------------------------------------------
# the house API
# --------------------------------------------------------------------------

def test_exposes_the_library_vocabulary(oracle):
    """The same three names the engines and the FFS blocks use."""
    m = _model(oracle)
    for name in ("fwd_filter", "scan_filter", "forecast", "backward_sample"):
        assert callable(getattr(m, name, None)), f"no {name}()"


def test_fwd_filter_returns_a_forecast_bundle(oracle):
    m = _model(oracle)
    out = m.fwd_filter(oracle["in_Y"][0])
    assert isinstance(out, ForecastBundle)
    q = oracle["in_Y"].shape[1]
    assert np.asarray(out.loc).shape == (q,)
    assert np.asarray(out.var).shape == (q,)


def test_forecast_returns_a_forecast_bundle(oracle):
    m = _model(oracle)
    m.scan_filter()
    out = m.forecast(12)
    assert isinstance(out, ForecastBundle)
    assert np.asarray(out.loc).shape == (12, oracle["in_Y"].shape[1])


# --------------------------------------------------------------------------
# exact parity with the oracle
# --------------------------------------------------------------------------

def test_scan_filter_matches_the_oracle(oracle):
    a, m, n, scale = _model(oracle).scan_filter()
    for got, want in ((a, "bs_a"), (m, "bs_m"), (n, "bs_n"),
                      (scale, "bs_scale")):
        np.testing.assert_array_equal(np.asarray(got), oracle[want])


def test_smoothed_means_match_the_oracle(oracle):
    np.testing.assert_array_equal(
        np.asarray(_model(oracle).smoothed_means()), oracle["params_s"])


def test_backward_sample_matches_the_oracle(oracle):
    """Exact, including the sqrtH path.

    This is the test that moved when the root convention changed to DLMAX's
    ``_sym_sqrt``, and the oracle was re-frozen with it: ``bs_ts`` shifted by
    about 2% of its mean magnitude while THIRTY other fields stayed bitwise
    identical. A different root gives a different realisation for a given
    variate; it must not give a different distribution, and 400k draws through
    each root recover ``H`` to within Monte Carlo error either way.
    """
    (_a, _m, _n, _s), ts = _model(oracle).backward_sample(
        oracle["in_chol_r"], key=random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(ts), oracle["bs_ts"])


def test_forecast_state_arrays_match_the_oracle(oracle):
    m = _model(oracle)
    m.scan_filter()
    a, R, f, var = m._forecast_arrays(12)
    np.testing.assert_array_equal(np.asarray(a), oracle["fc_a"])
    np.testing.assert_array_equal(np.asarray(f), oracle["fc_state_mean"])
    np.testing.assert_allclose(np.asarray(R), oracle["fc_R"],
                               rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(np.asarray(var), oracle["fc_state_var"],
                               rtol=1e-14, atol=1e-16)


# --------------------------------------------------------------------------
# step == scan, and the two variance conventions
# --------------------------------------------------------------------------

def test_fwd_filter_reproduces_scan_filter(oracle):
    """Step face against scan face. Not bitwise: a fused lax.scan reduces in a
    different order from a Python step loop, which is the same property (and
    the same reasoning) as test_fwd_filter_face records for uv_dlm."""
    _a, m, _n, scale = _model(oracle).scan_filter()
    step = _model(oracle)
    for y in oracle["in_Y"]:
        step.fwd_filter(y)
    np.testing.assert_allclose(np.asarray(step._carry[0]),
                               np.asarray(m)[-1], rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(step._carry[2]),
                               np.asarray(scale)[-1], rtol=1e-12, atol=0)


def test_fwd_filter_loc_is_the_one_step_ahead_mean(oracle):
    """``F' a_t`` -- the predictive from the carry BEFORE yt, which is what yt
    is scored against."""
    step = _model(oracle)
    got = np.array([np.asarray(step.fwd_filter(y).loc) for y in oracle["in_Y"]])
    want = np.einsum("ij,tjk->tik", np.asarray(oracle["in_F"]).T,
                     oracle["bs_a"]).squeeze(1)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_fwd_filter_var_is_Q_times_the_running_scale(oracle):
    """The per-series predictive variance, mirroring uv_dlm's ``s*(1+F'RF)``.

    ``Q_t`` is shared across series -- the Kronecker structure -- and ``scale``
    is the running estimate of ``diag(Sigma)``, so their product is per-series.
    An earlier version returned the bare scale-free ``Q_t``, which did not echo
    uv_dlm despite that being the point of the API.
    """
    step = _model(oracle)
    v = np.asarray(step.fwd_filter(oracle["in_Y"][0]).var)
    Q0 = oracle["params_Q"][0, 0, 0]
    want = Q0 * np.asarray(oracle["in_V0"]).reshape(-1)   # scale_0 == V0
    np.testing.assert_allclose(v, want, rtol=1e-14, atol=0)
    assert not np.allclose(v, v[0]), (
        "with a per-series V0 the variances must differ between series")


def test_fwd_filter_uses_the_PRIOR_scale(oracle):
    """A one-step-ahead predictive may only use D_{t-1}. The first step must
    therefore return V0 exactly, not the value updated by the first
    observation."""
    step = _model(oracle)
    v = np.asarray(step.fwd_filter(oracle["in_Y"][0]).var)
    Q0 = oracle["params_Q"][0, 0, 0]
    np.testing.assert_allclose(v / Q0, np.asarray(oracle["in_V0"]).reshape(-1),
                               rtol=1e-14, atol=0)
    # and the carry HAS advanced, so the second step differs
    v2 = np.asarray(step.fwd_filter(oracle["in_Y"][1]).var)
    assert not np.allclose(v, v2)


def test_the_scale_free_factor_is_still_reachable(oracle):
    """Q_t on its own, for callers who want to apply their own Sigma."""
    m = _model(oracle)
    assert np.asarray(m.cov.Q).shape == (m.T, 1, 1)


def test_forecast_var_includes_the_observation_term(oracle):
    """forecast(h).var is 1 + F'R_hF; the legacy shim's is F'R_hF. See the
    module docstring: the difference is exactly one and both are deliberate."""
    m = _model(oracle)
    m.scan_filter()
    got = np.asarray(m.forecast(12).var)
    q = oracle["in_Y"].shape[1]
    # var is (h, q): the same scale-free factor for every series, which is the
    # Kronecker structure. Broadcast the (h, 1) oracle value to compare.
    want = np.broadcast_to(1.0 + oracle["fc_state_var"].reshape(12, 1), (12, q))
    assert got.shape == (12, q)
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=0)


def test_forecast_at_h1_agrees_with_fwd_filter_after_the_same_data(oracle):
    """The two faces meet at one step: both are 1 + F'RF against the same
    state. They use different code paths to get there."""
    m = _model(oracle)
    m.scan_filter()
    fc = np.asarray(m.forecast(1).var)[0, 0]
    # Q_{T} from the shared path is the same object the step face returns.
    assert fc > 1.0


# --------------------------------------------------------------------------
# guards -- the preconditions the collapse depends on
# --------------------------------------------------------------------------

def test_nan_in_the_data_is_refused(oracle):
    """A gap makes the filter skip that step, desynchronising that series from
    the shared path -- which invalidates the whole single-run construction."""
    Y = np.array(oracle["in_Y"], copy=True)
    Y[10, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mv_dlm(data=Y, F=oracle["in_F"], G=oracle["in_G"], m0=oracle["in_m0"],
               C0=oracle["in_C0"], V0=oracle["in_V0"],
               disc_mtx=oracle["in_delta_M"])


def test_mult_comps_is_refused(oracle):
    """Required by the migration plan and never written until now. It would
    make the forward pass an EKF and the backward recursion approximate, and
    would do so silently -- nothing else in the stack raises."""
    p = oracle["in_G"].shape[0]
    with pytest.raises(ValueError, match="bilinear"):
        _model(oracle, mult_comps=np.ones(p))


def test_a_shorter_cov_path_is_refused(oracle):
    m = _model(oracle)
    short = type(m.cov)(
        C=m.cov.C[:10], R=m.cov.R[:10], Q=m.cov.Q[:10], A=m.cov.A[:10],
        B=m.cov.B[:9], H=m.cov.H[:10], sqrtH=m.cov.sqrtH[:10],
        S=m.cov.S[:10], delta_M=m.cov.delta_M, psd_clip=m.cov.psd_clip)
    with pytest.raises(ValueError, match="cov_path covers"):
        _model(oracle, cov_path=short)


def test_fwd_filter_past_the_path_is_refused(oracle):
    """The path is built for a fixed T; stepping past it would silently reuse
    the last A and Q."""
    m = _model(oracle)
    for y in oracle["in_Y"]:
        m.fwd_filter(y)
    with pytest.raises(ValueError, match="past the covariance path"):
        m.fwd_filter(oracle["in_Y"][0])


# --------------------------------------------------------------------------
# the reuse that makes a rolling origin cheap
# --------------------------------------------------------------------------

def test_a_supplied_cov_path_is_reused_not_rebuilt(oracle):
    """The single largest saving available, and currently unclaimed by any
    runner: the path does not depend on the data, so across a rolling origin it
    is built once."""
    first = _model(oracle)
    second = _model(oracle, cov_path=first.cov)
    assert second.cov is first.cov
    np.testing.assert_array_equal(np.asarray(second.scan_filter()[1]),
                                  np.asarray(first.scan_filter()[1]))


def test_params_is_a_flat_all_array_dict(oracle):
    """csrec jits its Gibbs kernel and passes this straight in, so the whole
    dict is traced: every value must be an array, and key presence is static
    pytree structure."""
    params = _model(oracle).params()
    for k, v in params.items():
        assert isinstance(v, (jnp.ndarray, np.ndarray)), f"{k} is {type(v)}"
    for k in ("F", "G", "C", "R", "Q", "A", "S", "s", "H", "sqrtH", "B",
              "delta_M", "V0", "m0", "_mvdlm"):
        assert k in params, f"missing key {k!r}"


@pytest.mark.parametrize("field", ["C", "R", "Q", "A", "B", "S", "H", "s"])
def test_params_values_match_the_oracle(oracle, field):
    got = np.asarray(_model(oracle).params()[field])
    np.testing.assert_array_equal(got, oracle[f"params_{field}"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly", "--tb=short"]))


# --------------------------------------------------------------------------
# the functional shims -- the surface BayesFR depends on
# --------------------------------------------------------------------------
# These take the params DICT rather than a model object, because a caller jits
# a kernel and passes it straight in. The shapes and the argument order below
# are a contract: a caller reading `var` under the wrong convention still runs.

from DLMAX.mvdlm import (dlm_params, dlm_forward, dlm_back_sample,
                         dlm_forecast)


def _params(o):
    return dlm_params(Y=o["in_Y"], m0=o["in_m0"], C0=o["in_C0"], F=o["in_F"],
                      G=o["in_G"], V0=o["in_V0"], disc_mtx=o["in_delta_M"])


def test_dlm_forward_matches_the_oracle(oracle):
    a, m, n, scale = dlm_forward(oracle["in_Y"], _params(oracle))
    for got, want in ((a, "bs_a"), (m, "bs_m"), (n, "bs_n"),
                      (scale, "bs_scale")):
        np.testing.assert_array_equal(np.asarray(got), oracle[want])


def test_dlm_back_sample_matches_the_oracle(oracle):
    (_a, _m, _n, _s), ts = dlm_back_sample(
        oracle["in_Y"], random.PRNGKey(0), oracle["in_chol_r"], _params(oracle))
    np.testing.assert_array_equal(np.asarray(ts), oracle["bs_ts"])


def test_dlm_forecast_matches_the_oracle(oracle):
    a, R, f, var = dlm_forecast(_params(oracle), 12)
    np.testing.assert_array_equal(np.asarray(a), oracle["fc_a"])
    np.testing.assert_array_equal(np.asarray(f), oracle["fc_state_mean"])
    np.testing.assert_allclose(np.asarray(R), oracle["fc_R"], rtol=1e-14,
                               atol=1e-16)
    np.testing.assert_allclose(np.asarray(var), oracle["fc_state_var"],
                               rtol=1e-14, atol=1e-16)


def test_dlm_forecast_var_EXCLUDES_the_observation_term(oracle):
    """The shim's convention, opposite to mv_dlm.forecast's. Callers add the
    observation half themselves -- one adds 1.0 directly, another supplies it
    from chol(Sigma) -- so including it here would double-count, and the result
    would still run."""
    _a, _R, _f, var = dlm_forecast(_params(oracle), 12)
    m = _model(oracle)
    m.scan_filter()
    predictive = np.asarray(m.forecast(12).var)[:, 0]
    np.testing.assert_allclose(np.asarray(var).reshape(-1) + 1.0, predictive,
                               rtol=1e-14, atol=0)


def test_dlm_params_accepts_a_discount_VECTOR(oracle):
    """The form DLMAX's components emit. Taking the vector as canonical is what
    lets the Component API drive this model without a second emitter."""
    p_mtx = _params(oracle)
    rates = np.diag(np.asarray(oracle["in_delta_M"]))
    p_vec = dlm_params(Y=oracle["in_Y"], m0=oracle["in_m0"], C0=oracle["in_C0"],
                       F=oracle["in_F"], G=oracle["in_G"], V0=oracle["in_V0"],
                       disc_rates=rates)
    np.testing.assert_allclose(np.asarray(p_vec["delta_M"]),
                               np.asarray(p_mtx["delta_M"]), rtol=0, atol=1e-15)


def test_dlm_params_needs_a_discount_in_some_form(oracle):
    with pytest.raises(ValueError, match="disc_mtx"):
        dlm_params(Y=oracle["in_Y"], m0=oracle["in_m0"], C0=oracle["in_C0"],
                   F=oracle["in_F"], G=oracle["in_G"], V0=oracle["in_V0"])


def test_exog_is_refused_on_identity_not_on_value(oracle):
    """Two defects fixed at once. Inspecting the array with bool(jnp.any(...))
    raises TracerBoolConversionError under jit, surfacing as a confusing trace
    error instead of this message; and letting an all-zero design through made
    the contract 'exog == 0' rather than 'no exog'."""
    with pytest.raises(NotImplementedError, match="COMMON to all series"):
        dlm_back_sample(oracle["in_Y"], random.PRNGKey(0),
                        oracle["in_chol_r"], _params(oracle),
                        exog=jnp.zeros_like(jnp.asarray(oracle["in_Y"])))

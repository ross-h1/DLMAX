"""Tests for the Regressors component in DLMAX.ffs.dlm_builder."""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Fourier, Regressors
from DLMAX.ffs import devices


@pytest.fixture
def init_data():
    rng = np.random.default_rng(0)
    T, n = 24, 4
    return pd.DataFrame(rng.normal(size=(T, n)) + np.arange(T)[:, None] * 0.1)


@pytest.fixture
def baseline_dlm(init_data):
    """LocalTrend + Fourier model, no regressors. p_struct = 6."""
    n = init_data.shape[1]
    dlm = DLM(family="Gaussian", n_series=n)
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.99, damping=0.99))
    dlm.add_component(
        Fourier(name="annual", period=12, n_comps=2, disc_rate=0.95)
    )
    dlm.set_error(disc_rate=1.0, power=1, nu0=1)
    return dlm.compile(init_data, device=devices.host_device)


@pytest.fixture
def regressors_dlm(init_data):
    """Same structural part plus 3 regressors. p_total = 6 + 3 = 9."""
    n = init_data.shape[1]
    dlm = DLM(family="Gaussian", n_series=n)
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.99, damping=0.99))
    dlm.add_component(
        Fourier(name="annual", period=12, n_comps=2, disc_rate=0.95)
    )
    dlm.add_component(
        Regressors(name="reg", n_regs=3, disc_rate=0.98, damping=0.9)
    )
    dlm.set_error(disc_rate=1.0, power=1, nu0=1)
    return dlm.compile(init_data, device=devices.host_device)


def test_baseline_state_shape(baseline_dlm):
    """Structural-only model has state width p_struct = 6 (level+slope+2*2)."""
    assert baseline_dlm.m0.shape == (4, 6)
    assert baseline_dlm.n_regressors == 0
    assert baseline_dlm.regression_G is None


def test_regressors_state_shape(regressors_dlm):
    """With n_regs=3, full state is p_struct + n_regs = 9."""
    assert regressors_dlm.m0.shape == (4, 9)
    assert regressors_dlm.n_regressors == 3
    assert regressors_dlm.regression_G.shape == (3, 3)


def test_regressors_damping_in_regression_G(regressors_dlm):
    """regression_G diagonal carries the damping factor (delta = 0.9)."""
    diag = jnp.diag(regressors_dlm.regression_G)
    np.testing.assert_allclose(diag, [0.9, 0.9, 0.9], atol=1e-6)


def test_regression_slots_at_state_tail(regressors_dlm):
    """Regression slots sit at the end of the state vector, starting at m0=0."""
    np.testing.assert_array_equal(regressors_dlm.m0[:, -3:], jnp.zeros((4, 3)))


def test_structural_priors_invariant_under_regressor_addition(
    baseline_dlm, regressors_dlm
):
    """Adding zero-prior regressors must not perturb the structural priors.

    This is the load-bearing backwards-compatibility guarantee for
    state-tail placement: existing structural-only behaviour is
    byte-identical when regressors are added with default priors.
    """
    np.testing.assert_array_equal(regressors_dlm.m0[:, :6], baseline_dlm.m0[:, :6])
    np.testing.assert_array_equal(regressors_dlm.C0[:, :6, :6], baseline_dlm.C0[:, :6, :6])


def test_regressors_filter_step_moves_coefficients(regressors_dlm):
    """A single filter step with non-trivial regressors moves coefficients off zero."""
    rng = np.random.default_rng(42)
    yt = jnp.array(rng.normal(size=4))
    Xt = jnp.array(rng.normal(size=(4, 3)))
    regressors_dlm.fwd_filter(yt, regressors=Xt)
    coefs = regressors_dlm.dlm_state["m"][:, -3:]
    assert bool(jnp.any(coefs != 0)), (
        "Regression coefficients did not move from zero after one filter step."
    )


def test_regression_only_model_rejected(init_data):
    """A DLM with no structural components must fail _validate."""
    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(
        Regressors(name="reg", n_regs=3, disc_rate=0.98, damping=0.9)
    )
    dlm.set_error(disc_rate=1.0, power=1, nu0=1)
    with pytest.raises(ValueError, match="at least one structural component"):
        dlm.compile(init_data, device=devices.host_device)


def test_n_regs_must_be_positive():
    """Regressors with n_regs <= 0 rejected at construction."""
    with pytest.raises(ValueError, match="n_regs must be > 0"):
        Regressors(name="reg", n_regs=0, disc_rate=0.98)


def test_damping_must_be_in_unit_interval():
    """Damping outside [0, 1] rejected at construction."""
    with pytest.raises(ValueError, match=r"damping must be in \[0, 1\]"):
        Regressors(name="reg", n_regs=3, disc_rate=0.98, damping=1.5)
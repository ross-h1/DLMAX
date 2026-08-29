"""Tests for the ``Predictive`` scaffolding (block refactor Phase 4).

Pins the family math (Gaussian / Student-t / lognormal) and — the key contract —
that ``combine(form="vincent")`` over Student-t components reproduces the current
FFS combine (``_t_vincent_sd`` SD + ``_t_quantile_average`` intervals + weighted
mean) to float precision, so wiring blocks onto ``Predictive`` stays null-diff.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from scipy.stats import norm, t as t_dist

from DLMAX.ffs.predictive import (
    Predictive,
    GaussianPredictive,
    StudentTPredictive,
    LogNormalPredictive,
    combine,
)
from DLMAX.ffs_core import FFSPredictive, _t_vincent_sd, _t_quantile_average


def _stack(nm=5, S=3, h=4, seed=0):
    rng = np.random.default_rng(seed)
    f_h = rng.normal(100.0, 5.0, (nm, S, h))
    q_h = rng.uniform(1.0, 9.0, (nm, S, h))           # scale^2
    nu = rng.uniform(5.0, 40.0, (nm, S))              # nu > 2 (finite variance)
    w = rng.uniform(0.1, 1.0, (nm, S))
    w = w / w.sum(axis=0, keepdims=True)              # DMA weights sum to 1 over models
    return f_h, q_h, nu, w


# --- family math ------------------------------------------------------------

def test_families_are_predictives():
    assert isinstance(GaussianPredictive(0.0, 1.0), Predictive)
    assert isinstance(StudentTPredictive(0.0, 1.0, 10.0), Predictive)
    assert isinstance(LogNormalPredictive(0.0, 1.0), Predictive)


def test_gaussian_math():
    g = GaussianPredictive(np.array([2.0, -1.0]), np.array([4.0, 9.0]))
    assert np.allclose(g.mean, [2.0, -1.0])
    assert np.allclose(g.sd, [2.0, 3.0])
    assert np.allclose(g.quantile(0.975), [2.0, -1.0] + np.array([2.0, 3.0]) * norm.ppf(0.975))
    assert np.allclose(g.log_score(np.array([2.0, -1.0])), norm.logpdf([2.0, -1.0], [2.0, -1.0], [2.0, 3.0]))


def test_studentt_math():
    st = StudentTPredictive(loc=1.0, scale2=4.0, nu=10.0)
    assert np.isclose(st.mean, 1.0)
    assert np.isclose(st.var, 4.0 * 10.0 / 8.0)                      # scale^2 * nu/(nu-2)
    assert np.isclose(st.sd, np.sqrt(4.0 * 10.0 / 8.0))
    assert np.isclose(st.quantile(0.975), 1.0 + 2.0 * t_dist.ppf(0.975, 10.0))
    assert np.isclose(st.log_score(1.3), t_dist.logpdf(1.3, 10.0, loc=1.0, scale=2.0))


def test_lognormal_math_and_jacobian():
    ln = LogNormalPredictive(mu_log=0.5, sigma2_log=0.25)
    assert np.isclose(ln.mean, np.exp(0.5 + 0.125))
    assert np.isclose(ln.var, (np.exp(0.25) - 1.0) * np.exp(1.0 + 0.25))
    assert np.isclose(ln.quantile(0.5), np.exp(0.5))                 # median = exp(mu)
    y = 2.0
    # observation-scale density = N(log y; mu, sigma) / y  (Jacobian -log y)
    assert np.isclose(ln.log_score(y), norm.logpdf(np.log(y), 0.5, 0.5) - np.log(y))


# --- the null-diff contract -------------------------------------------------

def test_vincent_combine_reproduces_ffs_studentt():
    f_h, q_h, nu, w = _stack()
    st = StudentTPredictive(loc=f_h, scale2=q_h, nu=nu)
    comb = combine([st], [w], form="vincent")

    ref = FFSPredictive(loc=None, sd=None, f_h=f_h, q_h=q_h, nu=nu, weights=w)

    # weighted mean
    mean_ref = (f_h * w[..., None]).sum(axis=0)                      # (S, h)
    np.testing.assert_allclose(comb.mean, mean_ref, rtol=1e-12, atol=1e-12)

    # Vincent SD (ffs returns (h, S); combine gives (S, h))
    sd_ref = _t_vincent_sd(ref).T                                   # (S, h)
    np.testing.assert_allclose(comb.sd, sd_ref, rtol=1e-9, atol=1e-9)

    # quantile-averaged 95% interval
    lo_ref, hi_ref = _t_quantile_average(ref, [95])[95]            # each (h, S)
    lo, hi = comb.interval(95)                                      # each (S, h)
    np.testing.assert_allclose(lo, lo_ref.T, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(hi, hi_ref.T, rtol=1e-9, atol=1e-9)


def test_heterogeneous_combine_finite():
    # a Student-t block + a lognormal block, combined on the observation scale
    f_h, q_h, nu, w = _stack(nm=4)
    st = StudentTPredictive(loc=f_h, scale2=q_h, nu=nu)
    # lognormal block: 2 models, matched shapes on the same (S, h) grid
    S, h = f_h.shape[1], f_h.shape[2]
    rng = np.random.default_rng(1)
    mu = rng.normal(4.6, 0.1, (2, S, h))          # ~ log(100)
    s2 = rng.uniform(0.01, 0.05, (2, S, h))
    ln = LogNormalPredictive(mu_log=mu, sigma2_log=s2)
    w_ln = np.full((2, S), 0.5)
    # renormalise the union weights across the 6 models
    w_all = np.concatenate([w * 0.6, w_ln * 0.4 / w_ln.sum(0, keepdims=True)], axis=0)
    w_all = w_all / w_all.sum(0, keepdims=True)
    comb = combine([st, ln], [w_all[:4], w_all[4:]], form="vincent")
    assert comb.mean.shape == (S, h)
    assert np.all(np.isfinite(comb.mean))
    assert np.all(np.isfinite(comb.sd))
    lo, hi = comb.interval(80)
    assert np.all(hi > lo)


def test_mixture_form_runs():
    f_h, q_h, nu, w = _stack(nm=4)
    st = StudentTPredictive(loc=f_h, scale2=q_h, nu=nu)
    mix = combine([st], [w], form="mixture")
    ls = mix.log_score(f_h.mean(axis=0))   # score at the ensemble-mean obs (S, h)
    assert np.all(np.isfinite(ls))
    assert np.all(np.isfinite(mix.mean))

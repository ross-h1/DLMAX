"""Regression test for the PowerLawUpdate zero-denominator fix.

Previously, when a batch of likelihoods was all-zero (or near-zero)
in some configuration, the line ``pSgM = liks / (liks @ mi @ mi.T)``
divided by zero and propagated NaN through every model weight.

This was triggered on real M3 Monthly data when one of the model
structure probabilities went to zero during the filter — once it hit
zero, denominators in subsequent updates became zero and the entire
allocator state went NaN. The fix masks zero denominators with 1.0;
this preserves whatever ``pSgM`` would otherwise be and lets the
filter recover.
"""

import jax.numpy as jnp
import numpy as np
import pytest


def test_powerlaw_zero_denom_no_nan():
    """When a column of `liks` is identically zero, weights stay finite."""
    from DLMAX.dlm_core import PowerLawUpdate

    n_models = 4
    n_series = 3

    n_classes = 2

    # Construct a likelihoods array where every series-by-model
    # likelihood is zero. This forces denom = liks @ mi @ mi.T to
    # zero everywhere, which without the safe divide yields NaN throughout.
    scores = jnp.zeros((n_models, n_series, 1))
    # Model indicator: rows are models, columns are model classes.
    mi = jnp.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=jnp.float64)

    # Uniform prior on models per series.
    pset_prior = jnp.ones((n_models, n_series, 1)) / n_models
    # Uniform prior on classes per series.
    mset_prior = jnp.ones((n_classes, n_series, 1)) / n_classes

    pset_post, mset_post = PowerLawUpdate(
        dma_pdr=0.99,
        dma_mdr=0.99,
        c=1e-3,
        scores=scores,
        pset_prior=pset_prior,
        mset_prior=mset_prior,
        mi=mi,
    )

    assert jnp.all(
        jnp.isfinite(pset_post)
    ), "pset_post contains NaN/inf; safe-divide fix did not engage."
    assert jnp.all(
        jnp.isfinite(mset_post)
    ), "mset_post contains NaN/inf; safe-divide fix did not engage."


def test_powerlaw_normal_denom_unchanged():
    """When likelihoods are positive, weights are non-trivial and
    finite (sanity check that the safe-divide didn't break the
    happy path)."""
    from DLMAX.dlm_core import PowerLawUpdate

    n_models = 4
    n_series = 3

    n_classes = 2

    rng = np.random.default_rng(42)
    scores = jnp.asarray(rng.uniform(0.1, 1.0, (n_models, n_series, 1)))
    mi = jnp.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=jnp.float64)

    pset_prior = jnp.ones((n_models, n_series, 1)) / n_models
    mset_prior = jnp.ones((n_classes, n_series, 1)) / n_classes

    pset_post, mset_post = PowerLawUpdate(
        dma_pdr=0.99,
        dma_mdr=0.99,
        c=1e-3,
        scores=scores,
        pset_prior=pset_prior,
        mset_prior=mset_prior,
        mi=mi,
    )

    assert jnp.all(jnp.isfinite(pset_post))
    assert jnp.all(jnp.isfinite(mset_post))
    # Posterior is not just the prior verbatim.
    assert not jnp.allclose(pset_post, pset_prior)

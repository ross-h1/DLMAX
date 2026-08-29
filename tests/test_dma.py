"""Tests for the DMA-side helpers in DLMAX.dma."""

import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from DLMAX.dma import as_forecast_bundle
from DLMAX.dlm_core import ForecastBundle

from DLMAX.dma import make_model_indicator


@pytest.fixture
def desc_with_key():
    return pd.DataFrame(
        {
            "key": ["c0", "c1", "c2", "c3"],
            "trend.disc_rate": [0.99, 0.99, 0.95, 0.95],
            "error.power": [0.0, 1.0, 0.0, 1.0],
        }
    )


def test_default_single_class(desc_with_key):
    mi = make_model_indicator(desc_with_key)
    assert mi.shape == (4, 1)
    assert (mi.values == 1).all()
    assert mi.values.dtype == bool
    assert list(mi.index) == ["c0", "c1", "c2", "c3"]
    assert list(mi.columns) == ["all"]


def test_by_column(desc_with_key):
    mi = make_model_indicator(desc_with_key, by="error.power")
    assert mi.shape == (4, 2)
    # Two cells per power level.
    assert mi.sum(axis=0).tolist() == [2, 2]
    # Row order preserved.
    assert list(mi.index) == ["c0", "c1", "c2", "c3"]


def test_by_callable(desc_with_key):
    def class_fn(row):
        return "add" if row["error.power"] == 1.0 else "log"

    mi = make_model_indicator(desc_with_key, by=class_fn)
    assert mi.shape == (4, 2)
    assert set(mi.columns) == {"add", "log"}


def test_unknown_column_raises(desc_with_key):
    with pytest.raises(KeyError, match="not in descriptor"):
        make_model_indicator(desc_with_key, by="nonexistent")


def test_invalid_by_type_raises(desc_with_key):
    with pytest.raises(TypeError, match="must be None, str, or callable"):
        make_model_indicator(desc_with_key, by=42)


def test_no_key_column_falls_back_to_index():
    desc = pd.DataFrame({"x": [0, 1, 2]})
    mi = make_model_indicator(desc)
    assert list(mi.index) == [0, 1, 2]


# In test_legacy_equivalence, drop the cast and add a dtype assertion:
def test_legacy_equivalence():
    """by='Class' should match pd.get_dummies(model_desc['Class']) including dtype."""
    desc = pd.DataFrame(
        {
            "key": ["k0", "k1", "k2", "k3"],
            "Class": ["A", "A", "M", "M"],
        }
    )
    mi = make_model_indicator(desc, by="Class")
    legacy = pd.get_dummies(desc["Class"])
    np.testing.assert_array_equal(mi.values, legacy.values)
    assert list(mi.columns) == list(legacy.columns)
    assert mi.values.dtype == legacy.values.dtype  # both bool

def test_as_forecast_bundle_from_fwd_filter_shape():
    """Flat (nm * q,) input reshapes to (nm, q, 1)."""
    nm, q = 4, 3
    f_flat = jnp.arange(nm * q, dtype=jnp.float64)
    q_flat = jnp.ones(nm * q, dtype=jnp.float64) * 4.0  # variance = 4 -> sd = 2

    bundle = as_forecast_bundle(f_flat, q_flat, n_models=nm, n_series=q, h=1)

    assert isinstance(bundle, ForecastBundle)
    assert bundle.loc.shape == (nm, q, 1)
    assert bundle.sd.shape == (nm, q, 1)
    # sqrt applied:
    assert jnp.all(bundle.sd == 2.0)


def test_as_forecast_bundle_from_forecast_h_shape():
    """3D (nm, q, h) input passes through with sqrt(q)."""
    nm, q, h = 4, 3, 6
    f_3d = jnp.zeros((nm, q, h), dtype=jnp.float64)
    q_3d = jnp.ones((nm, q, h), dtype=jnp.float64) * 9.0  # variance = 9 -> sd = 3

    bundle = as_forecast_bundle(f_3d, q_3d, n_models=nm, n_series=q, h=h)

    assert bundle.loc.shape == (nm, q, h)
    assert bundle.sd.shape == (nm, q, h)
    assert jnp.all(bundle.sd == 3.0)


def test_as_forecast_bundle_logscore_compatible():
    """End-to-end: bundle plumbs into LogScore without shape errors."""
    from DLMAX.dlm_core import LogScore

    nm, q = 2, 3
    rng = np.random.default_rng(0)
    f_flat = jnp.asarray(rng.normal(size=nm * q))
    q_flat = jnp.asarray(rng.uniform(0.5, 2.0, size=nm * q))

    bundle = as_forecast_bundle(f_flat, q_flat, n_models=nm, n_series=q, h=1)
    obs = jnp.asarray(rng.normal(size=q))

    scores = LogScore(bundle, obs)
    # LogScore broadcasts obs[:, newaxis] against forecast.loc/sd.
    # Result shape should be (nm, q, h) — same as bundle.loc.
    assert scores.shape == (nm, q, 1)
    assert jnp.all(jnp.isfinite(scores))
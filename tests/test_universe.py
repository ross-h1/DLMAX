"""Tests for the ``Universe`` construction seam (refactor Phase 2).

``_build_universe`` wraps the existing single-``multi_model_dlm`` construction
into a :class:`~DLMAX.ffs.universe.Universe`. These tests assert the wrapper is
equivalent to the legacy ``_build_multi_and_dma`` tuple — a non-behavioural
change — and that the container behaves as specified.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from DLMAX.ffs.block import Block
from DLMAX.ffs.universe import Universe
from DLMAX.ffs_core import _build_universe, _build_multi_and_dma


@pytest.fixture(scope="module")
def fit_array():
    rng = np.random.default_rng(0)
    T, n = 36, 3
    t = np.arange(T)
    cols = []
    for level, slope, amp in [(100.0, 0.5, 10.0), (50.0, -0.2, 5.0), (200.0, 1.0, 20.0)]:
        cols.append(level + slope * t + amp * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, T))
    return np.column_stack(cols)


def test_build_universe_returns_universe(fit_array):
    u = _build_universe(fit_array, 12, 2, 18, 0.9, 0.9, 0)
    assert isinstance(u, Universe)
    assert len(u.blocks) == 1
    assert isinstance(u.blocks[0], Block)
    assert u.multi is u.blocks[0]
    assert u.dma is not None
    assert u.model_indicator is not None


def test_universe_equivalent_to_legacy_tuple(fit_array):
    u = _build_universe(fit_array, 12, 2, 18, 0.9, 0.9, 0)
    multi, dma, mind = _build_multi_and_dma(fit_array, 12, 2, 18, 0.9, 0.9, 0)
    assert u.multi.nm == multi.nm
    assert u.multi.q == multi.q
    np.testing.assert_array_equal(
        np.asarray(u.model_indicator.values), np.asarray(mind.values)
    )
    assert type(u.dma) is type(dma)


def test_universe_multi_rejects_multiblock():
    u = Universe(blocks=[object(), object()], dma=None)
    with pytest.raises(ValueError):
        _ = u.multi

"""Conformance tests for the ``Block`` protocol (refactor Phase 1).

Asserts that :class:`~DLMAX.dlm_core.multi_model_dlm` — the default block engine
— satisfies :class:`~DLMAX.ffs.block.Block` unmodified, and that the protocol
is genuinely structural (rejects objects missing the required members). This is
a non-behavioural change: the protocol only describes the face
``multi_model_dlm`` already presents.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.dlm_core import multi_model_dlm
from DLMAX.ffs.block import Block
from DLMAX.ffs.factories import make_ffs_universe, _legacy_constraint


@pytest.fixture(scope="module")
def init_data():
    rng = np.random.default_rng(42)
    n_periods = 36
    t = np.arange(n_periods)
    series = {}
    for i, (level, slope, amp) in enumerate(
        [(100.0, 0.5, 10.0), (50.0, -0.2, 5.0), (200.0, 1.0, 20.0)]
    ):
        seasonal = amp * np.sin(2 * np.pi * t / 12)
        noise = rng.normal(0, 1.0, size=n_periods)
        series[f"s{i}"] = level + slope * t + seasonal + noise
    return pd.DataFrame(series, index=pd.RangeIndex(n_periods))


@pytest.fixture(scope="module")
def multi(init_data):
    """A small packed structural universe — the canonical default block."""
    dlm = make_ffs_universe(
        periodicity=12,
        n_seas_comps=2,
        n_series=init_data.shape[1],
        mult_models="A",
    )
    models, _desc = dlm.compile_universe(
        init_data, h=18, constraint=_legacy_constraint
    )
    return multi_model_dlm(models)


def test_block_is_runtime_checkable():
    assert getattr(Block, "_is_runtime_protocol", False)


def test_multi_model_dlm_is_a_block(multi):
    assert isinstance(multi, Block)


def test_multi_model_dlm_exposes_block_members(multi):
    # shape accessors and the core face are all present
    assert isinstance(multi.q, int) and multi.q == 3
    assert isinstance(multi.nm, int) and multi.nm >= 1
    for name in ("fwd_filter", "scan_filter", "forecast", "save", "load"):
        assert callable(getattr(multi, name))


def test_protocol_rejects_non_conforming_objects():
    class Incomplete:
        q = 1
        nm = 1

        def forecast(self, h):
            return None

        # missing fwd_filter / scan_filter / save / load

    assert not isinstance(Incomplete(), Block)
    assert not isinstance(object(), Block)
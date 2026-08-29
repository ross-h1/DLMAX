"""ComponentBlock — a block built from the component DSL (block refactor).

Closes the "same face at every level" loop: components -> DLM -> block ->
AutoFFS(blocks=[...]), with no universe_builder plumbing. A swept spec is the
block's DMA'd model set; a scalar spec is a single-model block; and a DSL block
composes with the standard StaticBlock in one list.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd

from DLMAX.ffs_core import AutoFFS
from DLMAX.ffs.component_block import ComponentBlock
from DLMAX.ffs.static_block import StaticBlock
from DLMAX.ffs.dlm_builder import DLM, LocalTrend


def _wide(n=3, L=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    return pd.DataFrame(
        {f"s{i}": 100.0 + 2 * t + rng.normal(0, 1, L) for i in range(n)},
        index=np.arange(L),
    )


def test_component_block_sweep_runs():
    wide = _wide()
    dlm = (DLM(family="Gaussian")
           .add_component(LocalTrend(name="trend", disc_rate=[0.9, 0.95, 0.99]))
           .set_error())
    cv = AutoFFS(blocks=[ComponentBlock(dlm, warmup=4)]).cross_validation(
        wide, h=4, n_windows=1, freq=1)
    assert {"unique_id", "ds", "AutoFFS", "AutoFFS-sd"} <= set(cv.columns)
    assert len(cv) == 3 * 4
    assert np.all(np.isfinite(cv["AutoFFS"].values))
    assert np.all(np.isfinite(cv["AutoFFS-sd"].values))


def test_component_block_single_model_runs():
    wide = _wide(seed=1)
    dlm = (DLM(family="Gaussian")
           .add_component(LocalTrend(name="trend", disc_rate=0.95))
           .set_error())
    cv = AutoFFS(blocks=[ComponentBlock(dlm, warmup=4)]).cross_validation(
        wide, h=4, n_windows=1, freq=1)
    assert np.all(np.isfinite(cv["AutoFFS"].values))


def test_component_block_composes_with_static_grid():
    wide = _wide(seed=2)
    dlm = (DLM(family="Gaussian")
           .add_component(LocalTrend(name="trend", disc_rate=[0.9, 0.99]))
           .set_error())
    cv = AutoFFS(blocks=[
        ComponentBlock(dlm, warmup=4),
        StaticBlock(season_length=None, n_seas_comps=None, warmup=4),
    ]).cross_validation(wide, h=4, n_windows=1, freq=1)
    assert np.all(np.isfinite(cv["AutoFFS"].values))
    assert len(cv) == 3 * 4

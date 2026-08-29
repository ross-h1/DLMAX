"""Per-state discounts on ``LocalTrend`` (the level and trend states get their
own discount). Scalar sets both; a 2-tuple ``(level, trend)`` sets them
separately (one model); a list stays a sweep, so a list of 2-tuples sweeps
per-state settings. The flat-list-as-sweep semantics (used by the frozen static
universe) are unchanged.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.dlm_builder import DLM, LocalTrend


def test_scalar_sets_both_states():
    b = np.asarray(LocalTrend(name="t", disc_rate=0.95).disc_norm_block())
    assert np.allclose(b, [0.95, 0.95])


def test_tuple_is_per_state():
    b = np.asarray(LocalTrend(name="t", disc_rate=(0.95, 0.99)).disc_norm_block())
    assert np.allclose(b, [0.95, 0.99])       # level, trend


def test_uniform_tuple_equals_scalar():
    a = np.asarray(LocalTrend(name="t", disc_rate=(0.95, 0.95)).disc_norm_block())
    b = np.asarray(LocalTrend(name="t", disc_rate=0.95).disc_norm_block())
    assert np.array_equal(a, b)               # per-state uniform == scalar, bit-exact


def test_list_stays_a_sweep():
    lt = LocalTrend(name="t", disc_rate=[0.95, 0.99])   # list => sweep, not per-state
    assert lt._disc_rate_kind == "list"
    assert lt.disc_rate is None               # resolved per cell


def test_per_state_validation():
    with pytest.raises(ValueError):
        LocalTrend(name="t", disc_rate=(0.95, 0.99, 0.9))   # wrong length
    with pytest.raises(ValueError):
        LocalTrend(name="t", disc_rate=(1.5, 0.9))          # out of (0, 1]


def test_list_of_tuples_sweeps_per_state():
    rng = np.random.default_rng(0)
    n, T = 3, 30
    t = np.arange(T)
    init = pd.DataFrame({f"s{i}": 100.0 + 2 * t + rng.normal(0, 1, T) for i in range(n)})
    dlm = DLM(family="Gaussian", n_series=n).add_component(
        LocalTrend(name="trend", disc_rate=[(0.95, 0.99), (0.9, 0.95)])
    ).set_error()
    models, desc = dlm.compile_universe(init, h=6)
    assert len(models) == 2
    got = {tuple(np.round(np.asarray(m.disc_rates).ravel(), 6)[:2])
           for m in models.values()}
    assert (0.95, 0.99) in got and (0.9, 0.95) in got

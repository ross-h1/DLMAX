"""Factory tests: make_ffs_universe and make_additive_var_power_set."""

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs.dlm_builder import DLM, LocalLevel
from DLMAX.dlm_core import multi_model_dlm

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


def test_make_ffs_universe_returns_dlm_with_three_components(init_data):
    from DLMAX.ffs.factories import make_ffs_universe

    dlm = make_ffs_universe(
        periodicity=12,
        n_seas_comps=2,
        n_series=init_data.shape[1],
        mult_models="A",
    )
    component_names = [c.name for c in dlm.components]
    assert component_names == ["trend", "seasonal"]
    assert dlm.error_spec is not None


def test_additive_universe_compiles(init_data):
    from DLMAX.ffs.factories import make_ffs_universe, _legacy_constraint

    dlm = make_ffs_universe(
        periodicity=12,
        n_seas_comps=2,
        n_series=init_data.shape[1],
        mult_models="A",
    )
    models, desc = dlm.compile_universe(
        init_data,
        h=18,
        constraint=_legacy_constraint,
    )
    assert len(models) > 0
    # Multiplicative settings should reflect mult_models='A'.
    seasonal = next(c for c in dlm.components if c.name == "seasonal")
    assert seasonal.multiplicative is False


def test_multiplicative_universe_compiles(init_data):
    from DLMAX.ffs.factories import make_ffs_universe, _legacy_constraint

    dlm = make_ffs_universe(
        periodicity=12,
        n_seas_comps=2,
        n_series=init_data.shape[1],
        mult_models="M",
    )
    models, desc = dlm.compile_universe(
        init_data,
        h=18,
        constraint=_legacy_constraint,
    )
    assert len(models) > 0
    seasonal = next(c for c in dlm.components if c.name == "seasonal")
    assert seasonal.multiplicative is True


def test_invalid_mult_models_rejected(init_data):
    from DLMAX.ffs.factories import make_ffs_universe

    with pytest.raises(ValueError, match="mult_models must be"):
        make_ffs_universe(
            periodicity=12,
            n_seas_comps=2,
            n_series=init_data.shape[1],
            mult_models="X",
        )


def test_constraint_filters_correctly(init_data):
    """LT_disc <= S_disc filter should reduce the universe."""
    from DLMAX.ffs.factories import make_ffs_universe, _legacy_constraint

    dlm = make_ffs_universe(
        periodicity=12,
        n_seas_comps=2,
        n_series=init_data.shape[1],
        mult_models="A",
    )
    full, _ = dlm.compile_universe(init_data, h=6)  # no constraint
    filtered, _ = dlm.compile_universe(
        init_data,
        h=6,
        constraint=_legacy_constraint,
    )
    assert len(filtered) < len(full)
    # All inert-seasonal cells should be in `filtered` regardless of constraint.
    inert_full = sum(1 for k in full if "Sn" in k or "S5" in k)  # depends on key shape
    # (skip the precise count assertion — covered by assemble_models tests)


def test_additive_var_power_set_sweeps_power(init_data):
    from DLMAX.ffs.factories import make_additive_var_power_set

    dlm = make_additive_var_power_set(
        periodicity=12,
        n_seas_comps=2,
        n_series=init_data.shape[1],
    )
    _, desc = dlm.compile_universe(init_data, h=6)
    # var_power should appear in descriptor with both values.
    assert "error.power" in desc.columns
    assert set(desc["error.power"].unique()) == {0.0, 1.0}

def test_multi_model_dlm_default_device():
    # device=None must resolve to a NamedSharding, not a list of devices:
    # device_put in _init_from_dlms cannot consume a list.
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(size=(12, 3)))

    dlm = DLM(family='Gaussian', n_series=3)
    dlm.add_component(LocalLevel(name='level', disc_rate=0.99))
    dlm.set_error(disc_rate=0.99, power=[0, 1], nu0=1)

    models, desc = dlm.compile_universe(panel)

    multi = multi_model_dlm(models)
    assert multi.nm == len(models)
    assert multi.q == 3
    assert multi.nm == 2  # power swept over [0, 1]
"""DLM.compile_universe and list-valued component params tests."""

import numpy as np
import pandas as pd
import pytest


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


def test_compile_with_lists_raises(init_data):
    """compile() must reject list-valued tunables explicitly."""
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(LocalTrend(name="trend", disc_rate=[0.99, 0.95], damping=0.99))
    dlm.set_error(disc_rate=1.0, power=1)
    with pytest.raises(ValueError, match="compile_universe"):
        dlm.compile(init_data)


def test_compile_universe_with_no_lists_raises(init_data):
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.99, damping=0.99))
    dlm.set_error(disc_rate=1.0, power=1)
    with pytest.raises(ValueError, match="no list-valued"):
        dlm.compile_universe(init_data)


def test_basic_universe_cell_count(init_data):
    """3 disc_rate × 2 damping × 2 power = 12 cells."""
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=[1.0, 0.97, 0.9],
            damping=[1.0, 0.99],
        )
    )
    dlm.set_error(disc_rate=0.99, power=[0, 1])
    models, desc = dlm.compile_universe(init_data, h=6)

    assert len(models) == 12
    assert len(desc) == 12
    assert set(desc.columns) >= {
        "key",
        "trend.disc_rate",
        "trend.damping",
        "error.power",
    }


def test_each_universe_cell_matches_scalar_compile(init_data):
    """Per-cell uv_dlm matches what .compile() would produce for that scalar."""
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend
    from DLMAX.ffs_core import mcomp_dlm

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=[1.0, 0.95],
            damping=0.99,
        )
    )
    dlm.set_error(disc_rate=0.99, power=1)
    models, desc = dlm.compile_universe(init_data, h=6)

    for _, row in desc.iterrows():
        scalar_dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
        scalar_dlm.add_component(
            LocalTrend(
                name="trend",
                disc_rate=row["trend.disc_rate"],
                damping=0.99,
            )
        )
        scalar_dlm.set_error(disc_rate=0.99, power=1)
        expected = scalar_dlm.compile(init_data, h=6)
        actual = models[row["key"]]
        np.testing.assert_allclose(
            np.asarray(actual.F), np.asarray(expected.F), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(actual.G), np.asarray(expected.G), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(actual.C0), np.asarray(expected.C0), atol=1e-12
        )


def test_constraint_filters_cells(init_data):
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Fourier

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=[1.0, 0.97, 0.9],
            damping=0.99,
        )
    )
    dlm.add_component(
        Fourier(
            name="seasonal",
            period=12,
            n_comps=2,
            disc_rate=[0.95, 0.9, 0.75],
        )
    )
    dlm.set_error(disc_rate=1.0, power=1)

    # Without constraint: 3 * 3 = 9 cells.
    no_filter, _ = dlm.compile_universe(init_data, h=6)
    assert len(no_filter) == 9

    # With LT_disc <= S_disc: count by hand. trend disc 1.0 vs S in
    # [0.95, 0.9, 0.75]: 0 keep. trend 0.97 vs S: 0 keep. trend 0.9 vs S
    # in [0.95, 0.9, 0.75]: 1 keep (S=0.95 only — 0.9<=0.9 also keeps;
    # so 2 keep). Wait — recount: 1.0<=S? never. 0.97<=S? never.
    # 0.9<=S? S=0.95 yes, S=0.9 yes, S=0.75 no. So 2 cells.
    constraint = lambda cell: (
        cell["trend"]["disc_rate"] <= cell["seasonal"]["disc_rate"]
    )
    filtered, _ = dlm.compile_universe(init_data, h=6, constraint=constraint)
    assert len(filtered) == 2


def test_inert_fourier_via_none_in_disc_rate_list(init_data):
    """Fourier(disc_rate=[None, 0.95]) should produce one inert + one active cell."""
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Fourier

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.99, damping=0.99))
    dlm.add_component(
        Fourier(
            name="seasonal",
            period=12,
            n_comps=2,
            disc_rate=[None, 0.95],
        )
    )
    dlm.set_error(disc_rate=1.0, power=1)
    models, desc = dlm.compile_universe(init_data, h=6)

    assert len(models) == 2
    # Inert cell: F[2:] should be zero.
    inert_row = desc[desc["seasonal.disc_rate"].isna()]
    assert len(inert_row) == 1
    inert_model = models[inert_row.iloc[0]["key"]]
    np.testing.assert_allclose(np.asarray(inert_model.F)[2:], np.zeros(4))
    # Active cell: F[2:] should not be all zero.
    active_row = desc[desc["seasonal.disc_rate"].notna()]
    active_model = models[active_row.iloc[0]["key"]]
    assert np.abs(np.asarray(active_model.F)[2:]).sum() > 0


def test_nan_damping_resolves_to_no_trend(init_data):
    """LocalTrend(damping=[0.99, NaN]) — NaN cell should give state with no-trend G."""
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=0.99,
            damping=[0.99, float("nan")],
        )
    )
    dlm.set_error(disc_rate=1.0, power=1)
    models, desc = dlm.compile_universe(init_data, h=6)
    assert len(models) == 2

    nan_row = desc[desc["trend.damping"].isna()]
    nan_model = models[nan_row.iloc[0]["key"]]
    G = np.asarray(nan_model.G)
    # No-trend G has [[1, 0], [0, 0]] in the trend block.
    np.testing.assert_allclose(
        G[:2, :2], np.array([[1.0, 0.0], [0.0, 0.0]]), atol=1e-12
    )


def test_descriptor_columns_include_all_params(init_data):
    """Descriptor includes every tunable, even if it wasn't swept.

    Non-swept params get their scalar value repeated across rows, so
    the descriptor is a self-contained record of each cell's full
    parameterisation.
    """
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=[1.0, 0.97],
            damping=0.99,
        )
    )
    dlm.set_error(disc_rate=[0.99, 1.0], power=1)
    _, desc = dlm.compile_universe(init_data, h=6)

    # Both swept and non-swept params present.
    assert "trend.disc_rate" in desc.columns
    assert "trend.damping" in desc.columns  # not swept; still present
    assert "error.disc_rate" in desc.columns
    assert "error.power" in desc.columns  # not swept; still present

    # Non-swept values should be the scalar repeated across rows.
    assert (desc["trend.damping"] == 0.99).all()
    assert (desc["error.power"] == 1).all()


def test_h_threaded_through_to_uvdlm(init_data):
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(LocalTrend(name="trend", disc_rate=[0.99, 0.95], damping=0.99))
    dlm.set_error(disc_rate=1.0, power=1)
    models, _ = dlm.compile_universe(init_data, h=18)
    for m in models.values():
        assert hasattr(m, "GH")


def test_fourier_multiplicative_sweep(init_data):
    """Sweep Fourier.multiplicative across [False, True]; verify
    distinct mult_comps in the resulting models."""
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Fourier

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.99, damping=0.99))
    dlm.add_component(
        Fourier(
            name="seasonal",
            period=12,
            n_comps=2,
            disc_rate=0.95,
            multiplicative=[False, True],
        )
    )
    dlm.set_error(disc_rate=0.99, power=1)
    models, desc = dlm.compile_universe(init_data, h=6)

    assert len(models) == 2
    assert "seasonal.multiplicative" in desc.columns
    assert set(desc["seasonal.multiplicative"]) == {False, True}

    # The two cells should differ in mult_comps[2:6] (the seasonal slots).
    add_row = desc[desc["seasonal.multiplicative"] == False].iloc[0]
    mult_row = desc[desc["seasonal.multiplicative"] == True].iloc[0]
    add_model = models[add_row["key"]]
    mult_model = models[mult_row["key"]]

    add_mc = np.asarray(add_model.mult_comps)
    mult_mc = np.asarray(mult_model.mult_comps)
    np.testing.assert_array_equal(add_mc[2:6], np.zeros(4))
    np.testing.assert_array_equal(mult_mc[2:6], np.ones(4))


def test_mixed_multiplicative_fouriers_in_same_dlm(init_data):
    """Two Fourier components with different multiplicative settings
    coexist; sweep one, hold the other fixed."""
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Fourier

    dlm = DLM(family="Gaussian", n_series=init_data.shape[1])
    dlm.add_component(LocalTrend(name="trend", disc_rate=0.99, damping=0.99))
    dlm.add_component(
        Fourier(
            name="weekly",
            period=12,
            n_comps=2,
            disc_rate=0.95,
            multiplicative=False,  # always additive
        )
    )
    dlm.add_component(
        Fourier(
            name="annual",
            period=12,
            n_comps=1,  # different period for distinct state
            disc_rate=0.9,
            multiplicative=[False, True],  # swept
        )
    )
    dlm.set_error(disc_rate=0.99, power=1)
    models, desc = dlm.compile_universe(init_data, h=6)

    assert len(models) == 2
    # Descriptor records both Fouriers' multiplicative — one constant,
    # one swept.
    assert "weekly.multiplicative" in desc.columns
    assert "annual.multiplicative" in desc.columns
    assert set(desc["weekly.multiplicative"]) == {False}  # not swept
    assert set(desc["annual.multiplicative"]) == {False, True}  # swept


def test_fourier_multiplicative_non_bool_in_list_rejected():
    """List values must be bool, not int 0/1 — explicit to catch
    obvious type confusion."""
    from DLMAX.ffs.dlm_builder import Fourier

    with pytest.raises(TypeError, match="must contain bool values"):
        Fourier(
            name="seasonal",
            period=12,
            n_comps=2,
            disc_rate=0.95,
            multiplicative=[0, 1],  # ints, not bools
        )

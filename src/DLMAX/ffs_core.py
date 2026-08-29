"""Forecast factory: assemble grids of DLM specifications (level / trend /
seasonality / variance combinations) and run Dynamic Model Averaging across
them, producing per-series filtered forecasts and final h-step forecasts.

Device setup is done by ``configure_devices`` below. It is called once on
import with sensible CPU defaults so the module is safe to import. To switch
to GPU, call ``configure_devices('gpu', device_id=0)`` explicitly before
creating models. The CLI is only active when this file is run directly.
"""

import os
import warnings
import h5py

import argparse
from typing import Optional, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm
from jax import config, device_put, devices, make_mesh
from jax.tree_util import Partial
from jax.scipy.linalg import block_diag
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
from jax.lax import scan, cond

from DLMAX.dlm_core import (
    fourier_FG,
    j_DMA_model_update,
    multi_model_dlm,
    uv_dlm,
    vdiag,
    Allocator,
    ForecastBundle,
    AllocatorState,
    allocator_step,
    LogScore,
    IdentityAggregator,
    PowerLawUpdate,
    init_alloc_state,
    _multi_fwd_filter_step,
    _format_yt,
    axis0dot,
    _jvdlm_uv_fcast_H,
    _jvdlm_uv_fcast_H_nested_fq,
    iterated_obs_forecast,
    _flip_ar_np,
    _flip_to_stationary,
    AR_STATIONARITY_EPS,
)
from jax import vmap

# Module flag selecting the CV forecast kernel. When True (default) the CV
# h-step predictive uses the nested-fq vmap (model-only operands shared across
# series, returns only (f, q)) — ~34% faster and lower-memory than the flat
# kernel. When False it uses the verbatim flat path (``_hstep_predictive_flat``),
# retained as the bit-exact M1M regression reference. The two differ only by
# ~1e-14 batch-ordering float noise. ``_run_cv_batch`` branches on this flag to
# build the matching ``DH`` form (compact vs replicated) and pick the kernel,
# threading the choice into both the n_windows==1 fast path and the emit-scan.
_CV_USE_NESTED = True

from DLMAX.dma import make_model_indicator

# -----------------------------------------------------------------------------
# Device / sharding configuration
# -----------------------------------------------------------------------------
# The actual implementation lives in ``DLMAX.ffs.devices``. We import the
# module itself (rather than its contents) so that internal references
# resolve via attribute lookup and stay live across reconfiguration.
# ``configure_devices`` is re-exported for backwards compatibility.

from DLMAX.ffs import devices
from DLMAX.ffs.devices import configure_devices  # noqa: F401  (re-export)
from DLMAX.ffs.universe import Universe
from DLMAX.ffs.static_block import StaticBlock


# -----------------------------------------------------------------------------
# Discount-rate tables
# -----------------------------------------------------------------------------
# These tables encode the discount-rate grid used by ``assemble_models``.
# Every model class shares the same per-level options; the table is kept in
# this wider form so a class can be given its own grid without restructuring.

_LT_VALUES = [0.9999, 0.99, 0.95, 0.9, 0.75]
_N_CLASSES = 8

# Error-monitoring (adaptive-discount) policy. alpha is the EWMA weight on the
# |standardised one-step error| that drives each model's online forgetting; held
# constant here (the FFS-side policy knob — override via ffs_core.ADAPT_ALPHA).
ADAPT_ALPHA = 0.3
# Standard scalar monitor sensitivity (SD multiple) for the static-grid error
# monitor. This is the AutoFFS / AutoFFSUniverse DEFAULT (monitor ON — the
# finalised house spec, applied across all fit/CV/universe routes). Pass
# ``None``/``False`` for the bit-exact legacy (monitor-off) path. Injection
# magnitudes live in dlm_builder (MONITOR_INJECT_LT/SEAS).
MONITOR_TAU = 3.0

LT_dr = pd.DataFrame(
    {str(c): _LT_VALUES for c in range(_N_CLASSES)},
    index=[str(i) for i in range(len(_LT_VALUES))],
)

S_dr = LT_dr.loc[["0", "1", "2"], ["4", "5", "6", "7"]].copy()
S_dr.loc["n"] = None

var_dr = pd.Series({"0": 1.0, "1": 0.99})
trend = pd.Series({"0": 0.99, "1": 0.95, "2": 0.75})
trend.loc["n"] = None


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------


def df(x, index=None, columns=None):
    """Convenience: wrap a (possibly singleton-squeezable) array in a DataFrame."""
    mdf = pd.DataFrame(x.squeeze(), dtype=np.float64)
    if index is not None:
        mdf.index = index
    if columns is not None:
        mdf.columns = columns
    return mdf


def partition_by_srs_len(data):
    """Group time series by their non-null length."""
    ts_lengths = pd.Series({i: len(data[i].dropna()) for i in data.columns})
    lens = ts_lengths.groupby(ts_lengths).count()
    return ts_lengths, lens


# Backward-compatible alias with the original (misspelled) name
partion_by_srs_len = partition_by_srs_len


# -----------------------------------------------------------------------------
# Initial-state estimation
# -----------------------------------------------------------------------------


def initial_state(idata, M, bounded=True):
    """Initial state priors via OLS on level/trend, with optional seasonal
    regression using period ``M``."""

    n_obs = min(len(idata), 10 if M is None else M, 10)

    if M is not None and M != 1:
        # approximation for non-integer M (e.g. weekly)
        M = int(np.round(M, 0))
        # regress detrended data on seasonal factors
        detrend = (
            idata
            - idata.rolling(int(np.round(2 * M, 0)), min_periods=0, center=True).mean()
        )
        X = pd.get_dummies(idata.index % M).values
        iXX = np.linalg.pinv(X.T @ X)
        coeffs = iXX @ X.T @ detrend.values
        errs = (idata - detrend.values - X @ coeffs).values
        s2 = (errs**2).sum(0) / (len(idata) - M)
        m_seas = coeffs
        C_seas = s2[np.newaxis, :] * np.diag(iXX)[:, np.newaxis]
        deseas = (idata - X @ coeffs).values[:n_obs]
    else:
        m_seas = None
        C_seas = None
        deseas = idata.values[:n_obs]

    tvals = np.arange(n_obs)
    X = np.concatenate([np.ones((n_obs, 1)), tvals[:, None]], axis=1)
    iXX = np.linalg.pinv(X.T @ X)
    coeffs = iXX @ X.T @ deseas[:n_obs]
    s2 = ((deseas - X @ coeffs) ** 2).sum(0) / (n_obs - 2)
    if bounded:
        coeffs = np.asarray(
            [
                np.where(coeffs[0] < 0, idata.iloc[0].values / 2, coeffs[0]),
                coeffs[1],
            ]
        )
    m_level_trend = coeffs
    C_level_trend = s2[np.newaxis, :] * np.diag(iXX)[:, np.newaxis]

    return m_level_trend.T, m_seas, C_level_trend.T, C_seas, s2


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def mcomp_dlm(
    disc_rates,
    trend_damping,
    variance_disc,
    periodicity,
    n_seas_comps,
    var_power,
    init_data,
    seasonality=None,
    h=None,
    device=None,
    warmup_steps=None,
):
    """Build a DLM via the component-based DLM builder.

    Functionally equivalent to :func:`Mcomp_DLM` for the parameter
    combinations exercised by :func:`assemble_models`. Differences:

    * Uses :class:`~DLMAX.ffs.dlm_builder.LocalTrend` for the trend
      component, with ``damping=0.0`` reproducing the legacy
      "no trend" semantics (``trend_damping=NaN``). State size stays
      consistent across the model set whether or not the model has
      a trend.
    * Uses :class:`~DLMAX.ffs.dlm_builder.Fourier` for the seasonal
      component, with ``multiplicative=True`` for ``seasonality='mult'``.
      When ``periodicity`` is set but ``seasonality=None``, an inert
      Fourier (``inert=True``) preserves state shape across the model
      set.
    * Supports ``warmup_steps>0`` via :meth:`DLM.compile`.

    See :func:`Mcomp_DLM` for the parameter documentation.
    """
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Fourier

    if periodicity is None and seasonality is not None:
        raise ValueError("Must specify periodicity if seasonality is set")

    n = init_data.shape[1]

    # Translate legacy NaN-trend convention to damping=0 in the builder.
    damping = (
        0.0
        if (isinstance(trend_damping, float) and trend_damping != trend_damping)
        else float(trend_damping)
    )

    dlm = DLM(family="Gaussian", n_series=n)
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=float(disc_rates[0]),
            damping=damping,
        )
    )

    if seasonality is not None:
        # Active Fourier: contributes to observations.
        dlm.add_component(
            Fourier(
                name="seasonal",
                period=int(periodicity),
                disc_rate=float(disc_rates[1]),
                n_comps=n_seas_comps,
                multiplicative=(seasonality == "mult"),
            )
        )
    elif periodicity is not None:
        # Inert Fourier: F=0, zero priors. Reproduces the legacy
        # "periodicity set, seasonality None" branch in Mcomp_DLM,
        # which exists for state-shape consistency across the model set.
        dlm.add_component(
            Fourier(
                name="seasonal",
                period=int(periodicity),
                disc_rate=1.0,
                n_comps=n_seas_comps,
                inert=True,
            )
        )

    dlm.set_error(
        disc_rate=float(variance_disc),
        power=float(var_power),
        nu0=1.0,
    )

    return dlm.compile(
        init_data,
        device=device,
        warmup_steps=warmup_steps,
        h=h,
    )


def _ternary_np(cond, a, b):
    """Explicit scalar ternary: the original code used np.where as a ternary,
    which returns a 0-d array rather than a Python scalar."""
    return a if cond else b


def _register_model(model_desc, key, mclass, LT, D, S, V, is_multiplicative):
    """Record the descriptor row for a model with key ``key``."""
    model_desc.loc[key, "Class"] = float(mclass)
    model_desc.loc[key, "LD"] = LT_dr.loc[LT, mclass]
    model_desc.loc[key, "T"] = 0.0 if D == "n" else 1.0
    model_desc.loc[key, "Td"] = 0.0 if D == "n" else np.float64(trend[D])
    model_desc.loc[key, "S"] = 0.0 if S == "n" else 1.0
    model_desc.loc[key, "SD"] = 0.0 if S == "n" else S_dr.loc[S, mclass]
    model_desc.loc[key, "M"] = 1.0 if is_multiplicative else 0.0
    model_desc.loc[key, "V"] = var_dr[V]


#: Discount rates for the TVAR class (one model each); see _build_tvar_models.
#: Tracks the shared LT grid so the TVAR class is on the same discount values as
#: the structural classes (between-class comparison isn't confounded by the grid).
_TVAR_DISC_RATES = tuple(_LT_VALUES)
#: mset Class id for the TVAR group (structural classes are 0..7).
_TVAR_CLASS = 8
#: AR order for the no-seasonal (yearly) TVAR — k_struct there is only 2, so the
#: default k-1 = AR(1) is too restrictive. AR(4) -> TVAR state 5 (= quarterly).
_AR_ORDER_NO_SEASONAL = 4


def _build_tvar_models(init_data, h, order, periodicity, warmup_steps):
    """Build the TVAR class: ``LocalLevel + AR(order)`` at each of the five
    fast/slow discount rates, compiled and ready to pack alongside the
    structural universe (same total state dim ``1 + order``).

    ``order`` is chosen by the caller as the structural state size minus one so
    the TVAR state fills the structural ``k`` exactly (no padding). The AR uses
    the seasonal Minnesota prior (anchoring lag-1 and lag-``periodicity``) when
    the period fits within the order, else the plain lag-1 anchor.

    Returns ``(models, desc)`` with ``desc`` in the legacy 8-column schema and
    ``Class == _TVAR_CLASS``.
    """
    from DLMAX.ffs.dlm_builder import DLM, LocalLevel, AR

    n_series = init_data.shape[1]
    period = periodicity if (periodicity and 1 < int(periodicity) <= order) else None

    models = {}
    rows = []
    for i, dr in enumerate(_TVAR_DISC_RATES):
        dlm = DLM(family="Gaussian", n_series=n_series)
        dlm.add_component(LocalLevel("level", disc_rate=dr))
        dlm.add_component(AR("ar", order=order, disc_rate=dr, period=period))
        dlm.set_error(disc_rate=0.99, power=1.0, nu0=1.0)
        key = f"R{i}"  # TVAR model key
        models[key] = dlm.compile(init_data, h=h, warmup_steps=warmup_steps)
        rows.append(
            {
                "Class": _TVAR_CLASS,
                "LD": str(i),
                "T": "n",
                "Td": float("nan"),
                "S": "n",
                "SD": float("nan"),
                "M": 0,
                "V": "0",
            }
        )
    return models, pd.DataFrame(rows, index=list(models.keys()))


#: AR-coefficient discount grid for the combined AR(1) forms (fixed AR + slowly
#: time-varying). Small on purpose — the structural trend carries the dynamics.
_AR_DISC_RATES = (1.0, 0.9)
#: AR(1) coefficient prior anchor (lag-1 mean): 1.0 = pure RW; <1 (e.g. 0.95)
#: = sub-integrated (stationary prior, bounded long-h variance). Env override
#: (FFS_AR_ANCHOR) for A/B; the chosen default is hard-coded once decided.
_AR_ANCHOR = float(os.environ.get("FFS_AR_ANCHOR", "1.0"))
#: Trend-damping grid for the combined AR(1) forms. 0.0 = level-only (trend slot
#: zeroed). Module-level so an experiment can override it (e.g. prepend 1.0 for an
#: undamped trend); default unchanged so the default AR universe is bit-identical.
_AR_DAMPING = (0.99, 0.95, 0.0)
#: mset Class ids for the combined AR(1) forms (must not collide with the
#: structural 0..7). level+AR1 and level+trend+AR1 are distinct DMA classes.
_AR_CLASS_LEVEL = 8
_AR_CLASS_TREND = 9


def _build_ar_combined_models(init_data, h, warmup_steps):
    """Combined AR(1) structural forms: ``LocalTrend + AR(1)`` in ONE model.

    Replaces the legacy ``LocalLevel + AR(order)`` *switching* class
    (:func:`_build_tvar_models`). The AR(1) sits ON TOP of the level (+ optional
    damped trend) so it mops up residual autocorrelation rather than the DMA
    switching to a pure-AR class. AR(1) suffices (vs the old AR(4) no-seasonal
    order) precisely because the trend now carries the momentum.

    Standard grids so the AR forms sit on the same discount values as the
    structural classes: ``_LT_VALUES`` x damping (``[0.99,0.95,None]``,
    None=level-only) x ``_AR_DISC_RATES`` x ``var_dr``. Additive error only
    (multiplicative variance on an AR block is ill-posed); Minnesota RW
    coefficient prior (the annual A/B verdict: RW anchor beats a diffuse zero
    anchor). No seasonal block (AR is the no-seasonal alternative to the seasonal
    structural forms -- "AR(1) XOR seasonality").

    Returns ``(models, desc)`` in the legacy 8-column schema with two mset
    classes (``_AR_CLASS_LEVEL`` = level+AR1, ``_AR_CLASS_TREND`` =
    level+trend+AR1). State dim (level[+trend] + AR(1)) is reconciled with the
    structural ``k`` by ``multi_model_dlm``'s mixed-k padding.
    """
    from DLMAX.ffs.dlm_builder import DLM, LocalTrend, AR

    n_series = init_data.shape[1]
    # damping 0.0 = level-only (the trend slot is zeroed); >0 = damped trend;
    # 1.0 = undamped. Grid is the overridable module constant _AR_DAMPING
    # (default [0.99, 0.95, 0.0]; Td=0.75 omitted).
    damping = list(_AR_DAMPING)
    dlm = DLM(family="Gaussian", n_series=n_series)
    dlm.add_component(LocalTrend("lt", disc_rate=list(_LT_VALUES), damping=damping))
    dlm.add_component(AR("ar", order=1, disc_rate=list(_AR_DISC_RATES), anchor=_AR_ANCHOR))
    dlm.set_error(disc_rate=[var_dr["0"], var_dr["1"]], power=1.0, nu0=1.0)
    models, desc = dlm.compile_universe(init_data, h=h, warmup_steps=warmup_steps)

    lt_idx = {v: str(i) for i, v in enumerate(_LT_VALUES)}
    v_idx = {float(var_dr["0"]): "0", float(var_dr["1"]): "1"}
    rows = []
    for _, r in desc.iterrows():
        td = r["lt.damping"]
        is_trend = (not pd.isna(td)) and float(td) != 0.0    # td==0 -> level-only
        rows.append({
            "Class": _AR_CLASS_TREND if is_trend else _AR_CLASS_LEVEL,
            "LD": lt_idx.get(r["lt.disc_rate"], "?"),
            "T": "t" if is_trend else "n",
            "Td": float(td) if is_trend else float("nan"),
            "S": "n",
            "SD": float("nan"),
            "M": 0,
            "V": v_idx.get(float(r["error.disc_rate"]), "?"),
        })
    models = {f"AR_{k}": v for k, v in models.items()}    # prefix: no key collision
    return models, pd.DataFrame(rows, index=list(models.keys()))


def assemble_models(
    init_data,
    h,
    periodicity=12,
    n_seas_comps=None,
    mult_models=True,
    warmup_steps=None,
    include_ar=False,
    component_priors=None,
    error_nu0=None,
    monitor_tau=MONITOR_TAU,
):
    """Build a dict of DLM instances keyed by a compact code, plus a
    DataFrame describing each model.

    Now a thin shim over :func:`make_ffs_universe` and
    :meth:`DLM.compile_universe`. Cell keys and ``model_desc`` columns
    preserve the legacy format for backward compatibility.

    ``monitor_tau`` (a scalar SD multiple, e.g. 3.0) turns the signed-error
    variance-injection monitor ON for every model in the static grid; ``None``
    (default) keeps the legacy static path (bit-exact).
    """
    from DLMAX.ffs.factories import (
        make_ffs_universe,
        _legacy_constraint,
        _trend_seasonal_light_damping_only,
        _make_legacy_keyfn_for_universe,
    )

    n_series = init_data.shape[1]

    # Grid constraints (the empirical default): when both trend and seasonal
    # are active, restrict to Td=0.99 — composed with the legacy LT<=S filter
    # (both must pass).
    def _constraint(cell):
        return _legacy_constraint(cell) and _trend_seasonal_light_damping_only(cell)

    def build_branch(prefix):
        dlm = make_ffs_universe(
            periodicity=periodicity,
            n_seas_comps=n_seas_comps,
            n_series=n_series,
            mult_models=prefix,
            component_priors=component_priors,
            error_nu0=error_nu0,
            monitor_tau=monitor_tau,
        )
        meta = dlm.__ffs_universe_meta__
        keyfn = _make_legacy_keyfn_for_universe(
            prefix=meta["prefix"],
            lt_keys=meta["lt_disc_keys"],
            s_keys=meta["seasonal_disc_keys"],
            d_keys=meta["trend_damping_keys"],
            v_keys=meta["var_disc_keys"],
        )
        models, desc = dlm.compile_universe(
            init_data,
            h=h,
            constraint=_constraint,
            warmup_steps=warmup_steps,
            keyfn=keyfn,
        )
        # Translate descriptor to legacy 8-column schema.
        legacy_rows = []
        for _, row in desc.iterrows():
            key = row["key"]
            # Parse the legacy key shape: "{A|M}L{LT}T{D}S{S}E{V}".
            # The four index strings are taken from the original keyfn
            # mapping; round-trip them out of the key.
            assert key.startswith(prefix + "L"), key
            rest = key[len(prefix) + 1 :]
            lt_part, _, after_t = rest.partition("T")
            d_part, _, after_d = after_t.partition("S")
            s_part, _, v_part = after_d.partition("E")
            mclass = _legacy_mclass(
                prefix=prefix,
                d_index=d_part,
                s_index=s_part,
            )
            legacy_rows.append(
                {
                    "Class": mclass,
                    "LD": lt_part,
                    "T": d_part,  # legacy schema names
                    "Td": (
                        float("nan") if d_part == "n" else float(row["trend.damping"])
                    ),
                    "S": s_part,
                    "SD": (
                        float("nan")
                        if s_part == "n"
                        else float(row["seasonal.disc_rate"])
                    ),
                    "M": int(prefix == "M"),
                    "V": v_part,
                }
            )
        return models, pd.DataFrame(
            legacy_rows, index=[r["LD"] for r in legacy_rows]
        )  # placeholder index

    A_models, A_desc = build_branch("A")
    if mult_models:
        M_models, M_desc = build_branch("M")
        all_models = {**A_models, **M_models}
        model_desc = pd.concat([A_desc, M_desc], axis=0)
    else:
        all_models = A_models
        model_desc = A_desc

    # Re-key the descriptor by model key to match legacy.
    model_desc.index = list(all_models.keys())
    model_desc = model_desc[["Class", "LD", "T", "Td", "S", "SD", "M", "V"]]

    # Combined AR(1) forms (mset classes 8/9): LocalTrend + AR(1) carried in the
    # SAME model, so the AR mops up residual autocorrelation on top of the
    # structural trend (vs the legacy LocalLevel+AR(order) switching class). AR(1)
    # suffices because the trend carries the momentum the old AR(4) was patching;
    # additive-only error; Minnesota RW coefficient prior; no seasonal block (AR
    # is the no-seasonal alternative to the seasonal structural forms). Validated
    # to recover the annual ETS-wins series, deploy-safe (FFS scripts/annual_ltar).
    if include_ar:
        ar_models, ar_desc = _build_ar_combined_models(
            init_data, h, warmup_steps
        )
        all_models = {**all_models, **ar_models}
        model_desc = pd.concat([model_desc, ar_desc], axis=0)

    return all_models, model_desc


def _legacy_mclass(*, prefix: str, d_index: str, s_index: str) -> int:
    """Reproduce the legacy mclass integer mapping.

    Additive (prefix='A'):
      S=n, D=n -> 0; S=n, D=set -> 2; S=set, D=n -> 4; S=set, D=set -> 6
    Multiplicative (prefix='M'):
      S=n, D=n -> 1; S=n, D=set -> 3; S=set, D=n -> 5; S=set, D=set -> 7
    """
    s_set = s_index != "n"
    d_set = d_index != "n"
    base = 0
    if s_set and d_set:
        base = 6
    elif s_set:
        base = 4
    elif d_set:
        base = 2
    if prefix == "M":
        base += 1
    return base


def assemble_models_adaptive(
    init_data,
    h,
    periodicity=12,
    n_seas_comps=None,
    mult_models=True,
    warmup_steps=None,
    tau_values=None,
    var_disc_values=None,
    component_priors=None,
    error_nu0=None,
):
    """Adaptive (error-monitoring) sibling of :func:`assemble_models`.

    Builds the tau-universe via :func:`make_ffs_universe_adaptive` and returns
    ``(models, model_desc)`` in the same shape ``assemble_models`` does, so the
    downstream multi/DMA assembly (:func:`_build_multi_and_dma`) is unchanged.

    Each compiled ``uv_dlm`` carries ``monitor=tau``; stacking gives a positive
    ``multi.monitor`` that :func:`_adapt_tau` detects -> the scan runs the
    error-driven discount. The structural model-set ``Class`` is the same
    :func:`_legacy_mclass` mapping (the tau-models are members WITHIN a class —
    DMA selects sensitivity within each structural form). The standard universe
    is always used; there is no AR/TVAR class on this path.
    """
    from DLMAX.ffs.factories import (
        make_ffs_universe_adaptive,
        _adaptive_constraint,
        _make_adaptive_keyfn_for_universe,
    )

    n_series = init_data.shape[1]

    def build_branch(prefix):
        dlm = make_ffs_universe_adaptive(
            periodicity=periodicity,
            n_seas_comps=n_seas_comps,
            n_series=n_series,
            mult_models=prefix,
            tau_values=tau_values,
            var_disc_values=var_disc_values,
            component_priors=component_priors,
            error_nu0=error_nu0,
        )
        meta = dlm.__ffs_universe_meta__
        keyfn = _make_adaptive_keyfn_for_universe(
            prefix=meta["prefix"],
            tau_keys=meta["tau_keys"],
            s_keys=meta["seasonal_disc_keys"],
            d_keys=meta["trend_damping_keys"],
            v_keys=meta["var_disc_keys"],
        )
        models, desc = dlm.compile_universe(
            init_data,
            h=h,
            constraint=_adaptive_constraint,
            warmup_steps=warmup_steps,
            keyfn=keyfn,
        )
        # Translate to the legacy 8-column descriptor. The key shape is
        # "{A|M}K{tau}T{D}S{S}E{V}"; the LD column carries the tau index (the
        # forgetting axis on this path). Class drives the DMA model-indicator.
        legacy_rows = []
        for _, row in desc.iterrows():
            key = row["key"]
            assert key.startswith(prefix + "K"), key
            rest = key[len(prefix) + 1 :]
            k_part, _, after_k = rest.partition("T")
            d_part, _, after_d = after_k.partition("S")
            s_part, _, v_part = after_d.partition("E")
            mclass = _legacy_mclass(prefix=prefix, d_index=d_part, s_index=s_part)
            legacy_rows.append(
                {
                    "Class": mclass,
                    "LD": k_part,  # tau index (forgetting axis on this path)
                    "T": d_part,
                    "Td": (
                        float("nan") if d_part == "n" else float(row["trend.damping"])
                    ),
                    "S": s_part,
                    "SD": (
                        float("nan")
                        if s_part == "n"
                        else float(row["seasonal.disc_rate"])
                    ),
                    "M": int(prefix == "M"),
                    "V": v_part,
                }
            )
        return models, pd.DataFrame(legacy_rows, index=[r["LD"] for r in legacy_rows])

    A_models, A_desc = build_branch("A")
    if mult_models:
        M_models, M_desc = build_branch("M")
        all_models = {**A_models, **M_models}
        model_desc = pd.concat([A_desc, M_desc], axis=0)
    else:
        all_models = A_models
        model_desc = A_desc

    model_desc.index = list(all_models.keys())
    model_desc = model_desc[["Class", "LD", "T", "Td", "S", "SD", "M", "V"]]
    return all_models, model_desc


# -----------------------------------------------------------------------------
# Main driver: filter + DMA over a set of series
# -----------------------------------------------------------------------------


def calc_srs(
    periodicity,
    n_seas_comps,
    h,
    data,
    srs,
    dma_pdr=0.90,
    dma_mdr=None,
):
    """Run the full multi-model DLM + DMA pipeline for a set of series.

    JIT-compiled fast path. For the pre-JIT reference implementation
    preserved for cross-checking, see :func:`calc_srs_orig`.

    Returns
    -------
    obs : np.ndarray
        Held-out test observations (the last ``h`` rows of each series).
    loc : jnp.ndarray
        DMA-combined h-step forecast location.
    sd : jnp.ndarray
        DMA-combined h-step forecast standard deviation.
    """

    if dma_mdr is None:
        dma_mdr = dma_pdr

    all_data = data[srs].dropna()
    fit_data = all_data.iloc[:-h]
    test_data = all_data.iloc[-h:]
    if periodicity is not None:
        init_data = all_data.iloc[: int(np.ceil(2 * periodicity))]
    else:
        init_data = all_data.iloc[0:10]

    _, n = init_data.shape

    ATS, model_desc = assemble_models(init_data, h, periodicity, n_seas_comps)
    models = model_desc["Class"]

    multi = multi_model_dlm(ATS, devices.dlm_compute)

    T = len(fit_data)
    m_list = list(ATS.keys())
    m_aug = m_list + ["DMA"]
    del ATS

    nm = len(m_aug)
    idx = jnp.arange(len(m_list))

    # model_set_loc = jnp.zeros((n, T, nm), device=devices.allocation_compute)
    # model_set_sd = jnp.zeros((n, T, nm), device=devices.allocation_compute)
    # MP = jnp.zeros((n, T, nm), device=devices.allocation_compute)

    model_indicator = make_model_indicator(model_desc, by="Class")
    nmset = model_indicator.shape[1]
    k = len(m_list)

    # pset_prior = jnp.ones((k, n), device=devices.allocation_compute) / (
    #    model_indicator.sum().loc[models].values[:, jnp.newaxis]
    # )
    # mset_prior = jnp.ones((nmset, n)) / nmset

    # all_mod_probs = jnp.zeros((n, T, nmset), device=devices.allocation_compute)

    c = 1e-3

    dma = Allocator(
        scoring_rule=LogScore,
        update_rule=Partial(PowerLawUpdate, dma_pdr=dma_pdr, dma_mdr=dma_mdr, c=c),
        model_indicator=model_indicator.values,
        device=devices.allocation_compute,
    )

    dma.init(
        n_models=k, n_series=n, n_classes=nmset, model_indicator=model_indicator.values
    )

    dma_step_fn = dma.prepared_step()

    multi_params = (
        multi.F,
        multi.G,
        multi.nm,
        multi.k,
        multi.q,
        multi.p,
        multi.disc_rates,
        multi.disc_rates_damped,
        multi.monitor_inject,
        multi.variance_disc,
        multi.variance_power,
        multi.mult_comps,
    )

    _tau = _adapt_tau(multi)

    def scan_step(carry, y_vals):
        ys_t, y_t = y_vals
        dlm_state, alloc_state = carry
        new_dlm_state, model, f, q = _multi_fwd_filter_step(
            dlm_state, multi_params, ys_t, tau=_tau, alpha=ADAPT_ALPHA
        )
        fc_bundle = ForecastBundle(f[..., None], q[..., None])
        new_alloc_state, weights = dma_step_fn(alloc_state, fc_bundle, y_t)
        new_dlm_state["s"] = new_dlm_state["s"].squeeze()
        new_dlm_state["nu"] = new_dlm_state["nu"].squeeze()
        return (new_dlm_state, new_alloc_state), (weights, model)

    # for t in tqdm.trange(T):
    # filter all models for this timestep
    #    y = jnp.asarray(fit_data.iloc[t].values)
    #    f, q = multi.fwd_filter(y)

    #    f = device_put(f, devices.allocation_compute)
    #    q = device_put(q, devices.allocation_compute)

    #    fc_bundle = ForecastBundle(jnp.atleast_3d(f), jnp.atleast_3d(q))
    #    weights = dma.update(fc_bundle, device_put(y,devices.allocation_compute))

    # build y values for batched updating
    yts = multi.format_yts(fit_data.values)

    init_carry = (multi.dlm_state, dma.state)
    # pass two set of obs, one for dlms on devices.dlm_compute, appropriately formated,
    # and one on alloc_compute
    # with jax.disable_jit():
    (final_dlm_state, final_alloc_state), (all_weights, all_models) = scan(
        scan_step,
        init_carry,
        (yts, device_put(fit_data.values, devices.allocation_compute)),
    )

    multi.dlm_state = final_dlm_state
    # mulit.model
    dma.state = final_alloc_state

    # final h-step forecasts
    f_h, q_h = multi.forecast(h=h)
    f_h = device_put(f_h, devices.allocation_compute)
    q_h = device_put(q_h, devices.allocation_compute)

    fc_bundle = ForecastBundle(f_h, q_h)
    fc_final = dma.combine(all_weights[-1], fc_bundle)

    return test_data.reset_index(drop=True).values, fc_final.loc, fc_final.sd


# ---------------------------------------------------------------------------
# Pre-JIT DMA loop, kept as a readable reference implementation to
# cross-check `calc_srs` against.
# ---------------------------------------------------------------------------


def calc_srs_orig(
    periodicity,
    n_seas_comps,
    h,
    data,
    srs,
    dma_pdr=0.90,
    dma_mdr=None,
):
    """[Legacy] Original non-JIT DMA loop, preserved for cross-checking.

    This is the pre-JIT reference implementation. New code should use
    :func:`calc_srs` (or :class:`AutoFFS` for stateful workflows). This
    function is retained as a readable reference to cross-check ``calc_srs``
    against.

    Returns
    -------
    (obs, loc, sd, model_set_loc, model_set_sd, m_list, MP, all_mod_probs,
     model_desc)
    """
    if dma_mdr is None:
        dma_mdr = dma_pdr

    all_data = data[srs].dropna()
    fit_data = all_data.iloc[:-h]
    test_data = all_data.iloc[-h:]
    if periodicity is not None:
        init_data = all_data.iloc[: int(np.ceil(2 * periodicity))]
    else:
        init_data = all_data.iloc[0:10]

    _, n = init_data.shape

    ATS, model_desc = assemble_models(init_data, h, periodicity, n_seas_comps)
    models = model_desc["Class"]

    multi = multi_model_dlm(ATS, devices.dlm_compute)

    T = len(fit_data)
    m_list = list(ATS.keys())
    m_aug = m_list + ["DMA"]
    del ATS

    nm = len(m_aug)
    idx = jnp.arange(len(m_list))

    model_set_loc = jnp.zeros((n, T, nm), device=devices.allocation_compute)
    model_set_sd = jnp.zeros((n, T, nm), device=devices.allocation_compute)
    MP = jnp.zeros((n, T, nm), device=devices.allocation_compute)

    model_indicator = make_model_indicator(model_desc, by="Class")
    nmset = model_indicator.shape[1]
    k = len(m_list)

    pset_prior = jnp.ones((k, n), device=devices.allocation_compute) / (
        model_indicator.sum().loc[models].values[:, jnp.newaxis]
    )
    mset_prior = jnp.ones((nmset, n)) / nmset

    all_mod_probs = jnp.zeros((n, T, nmset), device=devices.allocation_compute)

    c = 1e-3

    dma = Allocator(
        scoring_rule=LogScore,
        update_rule=Partial(PowerLawUpdate, dma_pdr=dma_pdr, dma_mdr=dma_mdr, c=c),
        model_indicator=model_indicator.values,
        device=devices.allocation_compute,
    )

    dma.init(
        n_models=k, n_series=n, n_classes=nmset, model_indicator=model_indicator.values
    )

    for t in tqdm.trange(T):
        # filter all models for this timestep

        y = jnp.asarray(fit_data.iloc[t].values)
        f, q = multi.fwd_filter(y)

        model_set_loc = model_set_loc.at[:, t, idx].set(
            device_put(f.T, device=devices.allocation_compute)
        )
        model_set_sd = model_set_sd.at[:, t, idx].set(
            device_put(jnp.sqrt(q).T, device=devices.allocation_compute)
        )

        # DMA update
        pset_post, mset_post = j_DMA_model_update(
            device_put(y, devices.allocation_compute),
            model_set_loc[:, t, :-1],
            model_set_sd[:, t, :-1],
            pset_prior,
            mset_prior,
            c,
            dma_pdr,
            dma_mdr,
            device_put(model_indicator.values, devices.allocation_compute),
        )

        wts = pset_post * (mset_post.T @ model_indicator.values.T).T

        MP = MP.at[:, t, :-1].set(wts.T)
        all_mod_probs = all_mod_probs.at[:, t, :].set(mset_post.T)

        f = device_put(f, devices.allocation_compute)
        q = device_put(q, devices.allocation_compute)

        fc_bundle = ForecastBundle(jnp.atleast_3d(f), jnp.atleast_3d(q))
        weights = dma.update(fc_bundle, device_put(y, devices.allocation_compute))

        # redundant
        fc_final = dma.combine(weights, fc_bundle)

        fd = (f * wts).sum(0)

        # DMA location & SD via quantile averaging
        model_set_loc = model_set_loc.at[:, t, -1].set(fd)
        model_set_sd = model_set_sd.at[:, t, -1].set(
            (((f + 1.96 * jnp.sqrt(q)) * wts).sum(0) - fd) / 1.96
        )

        pset_prior = pset_post
        mset_prior = mset_post

    # final h-step forecasts
    f_h, q_h = multi.forecast(h=h)
    f_h = device_put(f_h, devices.allocation_compute)
    q_h = device_put(q_h, devices.allocation_compute)
    wts_h = wts[..., jnp.newaxis]
    loc = (f_h * wts_h).sum(0)
    sd = (((f_h + 1.96 * jnp.sqrt(q_h)) * wts_h).sum(0) - loc) / 1.96

    fc_bundle = ForecastBundle(f_h, q_h)
    fc_final = dma.combine(weights, fc_bundle)

    return (
        test_data.reset_index(drop=True).values,
        loc,
        sd,
        model_set_loc,
        model_set_sd,
        m_list,
        MP,
        all_mod_probs,
        model_desc,
    )


# =====================================================================
# Predictive bundle (used by predict / forecast and by cv internally)
# =====================================================================


class FFSPredictive(NamedTuple):
    """Predictive output for a batch. Numpy arrays throughout.

    Attributes
    ----------
    loc : np.ndarray, shape (n_series, h)
        DMA-combined posterior predictive mean.
    sd : np.ndarray or None, shape (n_series, h)
        DMA-combined predictive SD. By default this is the quantile-averaged
        (Vincentised) SD — see :func:`_t_vincent_sd` /
        :func:`_combined_predictive_sd`; the law-of-total-variance SD
        (:func:`_t_predictive_sd`) is used only when ``sd_method="moment"``.
        ``None`` when the bundle was constructed without computing SD (e.g.
        inside ``cross_validation``, where the helper is called directly at
        emission time).
    f_h : np.ndarray, shape (nm, n_series, h)
    q_h : np.ndarray, shape (nm, n_series, h)
        Per-model predictive mean and scale-squared.
    nu : np.ndarray, shape (nm, n_series)
        Per-model degrees of freedom.
    weights : np.ndarray, shape (nm, n_series)
        Final DMA weights.
    """

    loc: np.ndarray
    sd: np.ndarray
    f_h: np.ndarray
    q_h: np.ndarray
    nu: np.ndarray
    weights: np.ndarray


# =====================================================================
# Per-batch persistent state
# =====================================================================


_PAD_PREFIX = "__pad_"  # synthetic unique_id prefix for inactive capacity-pad slots


class _BatchState:
    """Persistent per-batch container.

    Holds the live ``multi_model_dlm`` and ``Allocator`` for one batch of
    series, plus the most recent DMA weights (for quick predict without
    re-running the filter) and the timestamp of the last ingested
    observation per series. Picklable.
    """

    __slots__ = (
        "srs_ids",
        "multi",
        "dma",
        "model_indicator",
        "latest_weights",
        "last_ds_per_sid",
        "T_ingested",
        "lag_tail",
    )

    def __init__(
        self,
        srs_ids,
        multi,
        dma,
        model_indicator,
        latest_weights,
        last_ds_per_sid,
        T_ingested,
        lag_tail=None,
    ):
        self.srs_ids = tuple(srs_ids)
        self.multi = multi
        self.dma = dma
        self.model_indicator = model_indicator
        self.latest_weights = latest_weights  # (nm, q, 1)
        self.last_ds_per_sid = dict(last_ds_per_sid)
        self.T_ingested = int(T_ingested)
        # The last n_regressors observations (oldest-first, shape
        # (n_reg, n_series)) for an AR/TVAR universe — seeds forecast lags and
        # the cross-batch filter continuation. None for a structural universe.
        self.lag_tail = None if lag_tail is None else np.asarray(lag_tail)

    def __repr__(self):
        return (
            f"_BatchState(n_series={len(self.srs_ids)}, "
            f"T_ingested={self.T_ingested}, "
            f"nm={getattr(self.multi, 'nm', '?')})"
        )


# =====================================================================
# Prediction-interval helper
# =====================================================================


def _nan_safe_w(weights, finite):
    """DMA weights renormalised over the components finite for each (series,
    horizon). ``weights`` is ``(nm, n_series)``; ``finite`` the per-component
    finiteness mask ``(nm, n_series, h)``. Returns ``(nm, n_series, h)`` weights
    summing to 1 over the finite components, so a diverged component (non-finite
    forecast but finite DMA weight) is dropped from the mixture instead of
    poisoning it; PowerLawUpdate leaves its weight at the prior, so power
    discounting reweights it back gradually. Bit-identical to ``weights[..., None]``
    wherever every component is finite (DMA weights already sum to 1 there).
    """
    w = weights[..., None]                                # (nm, n_series, 1)
    any_bad = ~np.all(finite, axis=0, keepdims=True)      # (1, n_series, h)
    wb = np.where(finite, w, 0.0)
    wsum = wb.sum(axis=0, keepdims=True)
    wb = wb / np.where(wsum > 0, wsum, 1.0)
    return np.where(any_bad, wb, w)                       # all-finite cols -> original (bit-exact)


def _t_quantile_average(predictive, levels):
    r"""Student-t quantile averaging weighted by final DMA probabilities.

    For each model :math:`m`, the h-step predictive is
    :math:`y \sim T(f_{m,h}, q_{m,h}, \nu_m)`. Bounds at level L are
    averaged with the final model weights.

    Returns
    -------
    dict
        ``{L: (lo, hi)}``, each shape ``(h, n_series)``.
    """
    from scipy.stats import t as t_dist

    nu = predictive.nu  # (nm, n_series)
    weights = predictive.weights  # (nm, n_series)
    f_h = predictive.f_h  # (nm, n_series, h)
    sd_h = np.sqrt(predictive.q_h)  # (nm, n_series, h)
    finite = np.isfinite(f_h) & np.isfinite(sd_h)  # drop diverged components
    w = _nan_safe_w(weights, finite)  # (nm, n_series, h) renormalised over survivors

    out = {}
    for L in levels:
        p_upper = 0.5 + L / 200.0
        z = t_dist.ppf(p_upper, nu[..., None])  # (nm, n_series, 1)
        upper_pm = np.where(finite, f_h + z * sd_h, 0.0)
        lower_pm = np.where(finite, f_h - z * sd_h, 0.0)
        upper = (upper_pm * w).sum(axis=0)  # (n_series, h)
        lower = (lower_pm * w).sum(axis=0)
        out[int(L)] = (lower.T, upper.T)  # (h, n_series)
    return out


def _t_predictive_sd(predictive):
    r"""SD of the DMA mixture predictive distribution.

    For each component :math:`m`, the h-step predictive is
    :math:`y \sim T(f_{m,h}, \sqrt{q_{m,h}}, \nu_m)`, with variance
    :math:`q_{m,h}\,\nu_m / (\nu_m - 2)` for :math:`\nu_m > 2`. The
    mixture variance follows the law of total variance:

    .. math::
        \mathrm{Var}[Y] = \sum_m w_m \frac{q_{m,h} \nu_m}{\nu_m - 2}
                       + \sum_m w_m (f_{m,h} - \mu_\mathrm{mix})^2,
        \quad \mu_\mathrm{mix} = \sum_m w_m f_{m,h}.

    Returns
    -------
    np.ndarray, shape ``(h, n_series)``
        Predictive SD. Cells where any active component has
        :math:`\nu_m \le 2` are returned as ``NaN`` (t-variance
        undefined).

    Notes
    -----
    This is the correct *second moment* of the mixture, but **not** the
    default combined SD. Because a mixture of Gaussians with different
    locations/scales is leptokurtic, ``loc ± 1.96·sd_LoTV`` over-covers; the
    default :func:`_t_vincent_sd` (quantile/Vincent averaging) reflects the
    mixture's central interval and is sharper and better calibrated
    (Lichtendahl, Grushka-Cockayne & Winkler 2013). Request this via
    ``sd_method="moment"`` only when the mixture variance itself is wanted.
    """
    nu = np.asarray(predictive.nu)  # (nm, n_series)
    weights = np.asarray(predictive.weights)  # (nm, n_series)
    f_h = np.asarray(predictive.f_h)  # (nm, n_series, h)
    q_h = np.asarray(predictive.q_h)  # (nm, n_series, h)

    # Drop diverged components (non-finite raw forecast) and renormalise — keyed
    # on f_h/q_h, NOT var_m, so the intentional nu<=2 NaN below is preserved.
    finite = np.isfinite(f_h) & np.isfinite(q_h)  # (nm, n_series, h)
    w = _nan_safe_w(weights, finite)  # (nm, n_series, h)
    f_safe = np.where(finite, f_h, 0.0)
    mu_mix = (f_safe * w).sum(axis=0)  # (n_series, h)

    # Per-component t-variance: scale^2 * nu/(nu-2). Undefined for nu<=2.
    with np.errstate(divide="ignore", invalid="ignore"):
        inflation = np.where(nu > 2.0, nu / (nu - 2.0), np.nan)
    var_m = q_h * inflation[..., None]  # (nm, n_series, h); NaN for nu<=2 (intended)
    var_m = np.where(finite, var_m, 0.0)  # zero only diverged; nu<=2 NaN flows through

    # Law of total variance.
    e_var = (var_m * w).sum(axis=0)  # (n_series, h)
    var_e = ((f_safe - mu_mix[None, ...]) ** 2 * w).sum(axis=0)
    var_mix = e_var + var_e

    sd = np.sqrt(var_mix)  # (n_series, h)
    return sd.T  # (h, n_series) — match _t_quantile_average


def _t_vincent_sd(predictive):
    r"""Quantile-averaged (Vincentised) SD of the DMA combined predictive.

    This is the **default** combined SD. Each component's Student-t predictive
    :math:`T(f_{m,h}, \sqrt{q_{m,h}}, \nu_m)` is replaced by a *variance-matched
    Gaussian* with standard deviation
    :math:`\sigma_{m,h} = \sqrt{q_{m,h}\,\nu_m/(\nu_m-2)}`, and the component
    SDs (equivalently the symmetric intervals) are **averaged across components**
    with the final DMA weights — a quantile average — rather than the
    *variances*:

    .. math::
        \mathrm{sd}_\mathrm{comb}
            = \sum_m w_m \sqrt{q_{m,h}\,\nu_m/(\nu_m-2)}.

    Averaging quantiles rather than probabilities yields a sharper, better
    calibrated combined forecast (Lichtendahl, Grushka-Cockayne & Winkler,
    2013, "Is It Better to Average Probabilities or Quantiles?",
    *Management Science* 59(7):1594-1611). The law-of-total-variance SD
    (:func:`_t_predictive_sd`) is the correct *second moment* of the mixture,
    but a mixture of Gaussians with different locations/scales is leptokurtic,
    so :math:`\pm 1.96\,\mathrm{sd}_\mathrm{LoTV}` around the combined mean
    over-covers; the Vincentised SD reflects the mixture's central interval.

    The result is a single, level-independent SD: the combined predictive is
    treated as a Gaussian :math:`N(\mathrm{loc}, \mathrm{sd}_\mathrm{comb})`
    downstream (so ``loc ± 1.96·sd`` is the 95% interval and the Gaussian
    log-score uses this SD).

    For :math:`\nu_m \le 2` (infinite t-variance — only on very short series)
    the component SD falls back to the 95% t-quantile-implied Gaussian SD
    :math:`\sqrt{q_{m,h}}\;t^{-1}_{\nu_m}(0.975)/1.96`, which is finite for any
    :math:`\nu_m>0`. The variance-matched factor grows steeply for
    :math:`\nu_m \in (2, 3]`; this is the intended heavy-tail widening and is
    irrelevant when :math:`\nu_m` is large (the usual case).

    Returns
    -------
    np.ndarray, shape ``(h, n_series)`` — matches :func:`_t_predictive_sd`.
    """
    from scipy.stats import t as t_dist

    nu = np.asarray(predictive.nu)  # (nm, n_series)
    weights = np.asarray(predictive.weights)  # (nm, n_series)
    q_h = np.asarray(predictive.q_h)  # (nm, n_series, h)

    scale = np.sqrt(q_h)  # per-component Student-t scale (nm, n_series, h)
    z95 = 1.959963984540054  # norm.isf(0.025)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_factor = np.sqrt(nu / (nu - 2.0))  # variance-matched (nu > 2)
    quant_factor = t_dist.ppf(0.975, nu) / z95  # finite for any nu > 0
    factor = np.where(nu > 2.0, var_factor, quant_factor)  # (nm, n_series)

    sd_m = scale * factor[..., None]  # variance-matched Gaussian sd (nm, n_series, h)
    finite = np.isfinite(sd_m)        # drop diverged (non-finite) components
    w = _nan_safe_w(weights, finite)  # (nm, n_series, h) renormalised over survivors
    sd = (np.where(finite, sd_m, 0.0) * w).sum(axis=0)  # quantile (Vincent) average (n_series, h)
    return sd.T  # (h, n_series)


def _combined_predictive_sd(predictive, method="quantile"):
    """Combined DMA predictive SD under the requested averaging scheme.

    ``method="quantile"`` (default) is the Vincentised SD
    (:func:`_t_vincent_sd`) — averages component SDs/intervals; sharper and
    better calibrated (Lichtendahl, Grushka-Cockayne & Winkler 2013).
    ``method="moment"`` is the law-of-total-variance mixture SD
    (:func:`_t_predictive_sd`) — the correct second moment, but over-disperses
    the central interval. ``"quantile"`` is the default for every DMA-averaged
    output; request ``"moment"`` only when the mixture variance itself is
    wanted.
    """
    if method == "quantile":
        return _t_vincent_sd(predictive)
    if method == "moment":
        return _t_predictive_sd(predictive)
    raise ValueError(f"sd_method must be 'quantile' or 'moment', got {method!r}")


# =====================================================================
# Filter scan step (no emission) — used by fit and update
# =====================================================================


# (ADAPT_ALPHA / MONITOR_TAU are defined near the top with the discount-grid
# constants, so functions defined above this point can use MONITOR_TAU as a
# default. See that block for the policy notes.)


def _adapt_tau(multi):
    """Return the per-(model,series) tau vector for the adaptive-discount scan,
    or None for a legacy (static) universe.

    Discriminator is the monitor VALUE, not its dtype: ``multi_model_dlm``
    coerces a legacy ``monitor=False`` to a float-zero array, so dtype alone
    can't tell legacy from adaptive. An adaptive universe
    (make_ffs_universe_adaptive) stamps strictly-positive tau (>= 1) into the
    monitor slot, which stacks to a (p, 1) positive array; legacy stacks to
    all-zero. Any positive entry => adaptive (tau=0 is not a meaningful
    sensitivity). Adaptive -> reshape to (p,) to match dlm_state["S"]; legacy
    -> None, so ``_multi_fwd_filter_step`` takes the static branch (bit-exact).
    """
    mon = getattr(multi, "monitor", None)
    if mon is None:
        return None
    mon = jnp.asarray(mon)
    if bool(jnp.any(mon > 0)):
        return mon.reshape(-1)
    return None


def _disc_factor_from_tau(multi, dlm_state, tau):
    """Per-(model,series,state) ``disc_factor = (1-disc)/disc`` for the h-step
    forecast's system noise ``W = C * disc_factor``, given a PRECOMPUTED tau
    (None => static). Safe inside a traced scan (no host sync): ``tau`` must be a
    concrete None/array decided outside the trace.

    Static: ``disc = disc_rates * disc_rates_damped`` (byte-identical to before).
    Adaptive: ``disc`` is the model's CURRENT error-monitoring discount at the
    forecast origin — the same envelope the filter applied (mirrors
    dlm_core.adapt_discount_err), evaluated at the origin's EWMA ``S`` and frozen
    over the h steps (future S is unknowable). Stops a calm model forecasting
    with its d_min floor (~100x too much variance/step).
    """
    # Always the static grid: the signed-error monitor's injection is a one-off
    # applied during filtering (already in C at the origin), so the h-step
    # forecast projects forward with the plain grid forgetting only. `tau` is
    # accepted for signature compatibility; it does not affect the forecast.
    del tau
    disc = multi.disc_rates * multi.disc_rates_damped
    return (1.0 - disc) / disc


def _forecast_disc_factor(multi, dlm_state):
    """Eager convenience: resolve tau via _adapt_tau then _disc_factor_from_tau.
    Use only OUTSIDE a trace (e.g. the n_windows==1 fast path)."""
    return _disc_factor_from_tau(multi, dlm_state, _adapt_tau(multi))


def _build_filter_scan_step(multi_params, dma_step_fn, has_regressors=False,
                            reg_mask=None, tau=None, alpha=ADAPT_ALPHA):
    """Closure-based scan step. Closes over static params for jit-friendliness.

    The scan input is ``(ys_t, y_t, warmup_flag_t)``, or, when
    ``has_regressors`` is True, ``(ys_t, y_t, warmup_flag_t, reg_t)`` where
    ``reg_t`` is the ``(p, n_reg)`` AR-lag slice for the step (see
    ``multi_model_dlm.format_lag_yts``). ``warmup_flag_t`` is a scalar 0.0/1.0
    that, when 1.0, zeroes the system-noise contribution W for that step (see
    ``_multi_fwd_filter_step``). Callers that don't use warmup should pass an
    array of zeros.

    ``reg_mask`` (``(p,)`` bool, closed over) gates the lag-fill to the TVAR
    models in a mixed universe so structural models' F is left untouched.

    ``has_regressors=False`` (the structural universe) keeps the original
    three-input scan untouched — bit-exact with the M1M reference.
    """

    if not has_regressors:

        def scan_step(carry, y_vals):
            ys_t, y_t, warmup_flag_t = y_vals
            dlm_state, alloc_state = carry
            new_dlm_state, _model, f, q = _multi_fwd_filter_step(
                dlm_state, multi_params, ys_t, warmup_flag_t, tau=tau, alpha=alpha
            )
            fc_bundle = ForecastBundle(f[..., None], q[..., None])
            new_alloc_state, weights = dma_step_fn(alloc_state, fc_bundle, y_t)
            new_dlm_state["s"] = new_dlm_state["s"].squeeze()
            new_dlm_state["nu"] = new_dlm_state["nu"].squeeze()
            return (new_dlm_state, new_alloc_state), weights

        return scan_step

    def scan_step_reg(carry, y_vals):
        ys_t, y_t, warmup_flag_t, reg_t = y_vals
        dlm_state, alloc_state = carry
        new_dlm_state, _model, f, q = _multi_fwd_filter_step(
            dlm_state, multi_params, ys_t, warmup_flag_t,
            regressors=reg_t, reg_mask=reg_mask, tau=tau, alpha=alpha,
        )
        fc_bundle = ForecastBundle(f[..., None], q[..., None])
        new_alloc_state, weights = dma_step_fn(alloc_state, fc_bundle, y_t)
        new_dlm_state["s"] = new_dlm_state["s"].squeeze()
        new_dlm_state["nu"] = new_dlm_state["nu"].squeeze()
        return (new_dlm_state, new_alloc_state), weights

    return scan_step_reg


# =====================================================================
# Filter scan step with per-step h-step forecast emission — used by cv
# =====================================================================


def _hstep_predictive_flat(dlm_state, weights, nm, q, h, disc_factor,
                           variance_disc, variance_power, mult_comps, DH):
    """Flat-kernel h-step predictive ``(f_h, q_h, nu, weights)``.

    Verbatim original implementation, retained as the bit-exact M1M reference.
    Calls the FLAT ``_jvdlm_uv_fcast_H`` over the replicated ``(nm*q, ...)``
    operands; ``DH`` here is the REPLICATED form (``FH`` ``(p, h, k)``, ``GH``
    ``(p, h, k, k)``) built by :func:`_run_cv_batch` when ``_CV_USE_NESTED`` is
    False. Used by both the fast path and the emit-scan when that flag is off.

    Shapes: ``f_h, q_h`` → ``(nm, q, h)``; ``nu, weights`` → ``(nm, q)``.
    """
    C_post = axis0dot(
        dlm_state["Z"].swapaxes(-1, -2), dlm_state["Z"],  # C = Z' Z
    )
    init_for_fcast = {
        "m": dlm_state["m"],
        "s": dlm_state["s"],
        "nu": dlm_state["nu"],
        "C": C_post,
    }
    # disc_factor is the per-state (1-δ)/δ vector; dlm_uv_fcast_H builds the
    # system noise (block congruence W = B C B - C) internally.
    fc = _jvdlm_uv_fcast_H(
        disc_factor, variance_disc, variance_power, mult_comps, init_for_fcast, DH
    )
    f_h_t = fc["f"].reshape(nm, q, h)
    q_h_t = fc["q"].reshape(nm, q, h)
    nu_t = dlm_state["nu"].reshape(nm, q)
    weights_t = weights.squeeze(-1)
    return f_h_t, q_h_t, nu_t, weights_t


def _hstep_predictive(dlm_state, weights, nm, q, h, disc_factor,
                      variance_disc, variance_power, mult_comps, DH):
    """h-step predictive ``(f_h, q_h, nu, weights)`` from a filtered state.

    Shared by the emit-scan (:func:`_build_emit_scan_step`) and the
    ``n_windows == 1`` fast path in :func:`_run_cv_batch`, so both compute
    the cutoff forecast through identical code.

    Nested-fq path: reshapes the flat ``(nm*q, ...)`` operands it receives to
    ``(nm, q, ...)``, compacts the model-only constants (``variance_disc``,
    ``variance_power``, ``mult_comps``) to ``(nm, ...)`` by taking the q=0 slice
    (valid because :func:`_stack_and_broadcast` lays out p model-major, so the q
    axis is a pure broadcast), and calls the nested ``_jvdlm_uv_fcast_H_nested_fq``
    kernel which shares those operands across series and returns only ``(f, q)``
    already shaped ``(nm, q, h)``. ``DH`` here is the COMPACT form (``FH``
    ``(nm, h, k)``, ``GH`` ``(nm, h, k, k)``) built by :func:`_run_cv_batch` when
    ``_CV_USE_NESTED`` is True. Mirrors ``multi_model_dlm.forecast``. Numerically
    identical to :func:`_hstep_predictive_flat` to ~1e-14.

    Shapes: ``f_h, q_h`` → ``(nm, q, h)``; ``nu, weights`` → ``(nm, q)``.
    """
    C_post = axis0dot(
        dlm_state["Z"].swapaxes(-1, -2), dlm_state["Z"],  # C = Z' Z
    )
    init_for_fcast = {
        "m": dlm_state["m"],
        "s": dlm_state["s"],
        "nu": dlm_state["nu"],
        "C": C_post,
    }

    def _to_mq(a):                       # (p, ...) -> (nm, q, ...)
        return a.reshape((nm, q) + a.shape[1:])

    init_mq = {key: _to_mq(val) for key, val in init_for_fcast.items()}

    # Model-only operands kept compact (nm, ...) and shared across the q series
    # by the inner vmap (in_axes=None). The q axis is a pure broadcast (p is
    # model-major), so [:, 0] picks the model value. disc_factor (the per-state
    # (1-δ)/δ vector) is series-invariant too, so it joins them; the kernel
    # builds the system noise internally.
    disc_factor_m = _to_mq(disc_factor)[:, 0]          # (nm, k)
    variance_disc_m = _to_mq(variance_disc)[:, 0]      # (nm, 1)
    variance_power_m = _to_mq(variance_power)[:, 0]    # (nm, 1)
    mult_comps_m = _to_mq(mult_comps)[:, 0]            # (nm, k)

    f_h_t, q_h_t = _jvdlm_uv_fcast_H_nested_fq(
        disc_factor_m, variance_disc_m, variance_power_m, mult_comps_m, init_mq, DH
    )
    nu_t = dlm_state["nu"].reshape(nm, q)
    weights_t = weights.squeeze(-1)
    return f_h_t, q_h_t, nu_t, weights_t


def _hstep_predictive_ar(dlm_state, weights, nm, q, h, disc_factor,
                         variance_disc, variance_power, mult_comps, DH,
                         Fp, n_reg, reg_mask, seed_lags):
    """Iterated-expectations h-step predictive for an AR/TVAR universe — the
    AR analogue of :func:`_hstep_predictive_flat`.

    Reuses the ordinary state propagation (``aH = G^j m``, ``RH`` via the flat
    ``_jvdlm_uv_fcast_H`` — ``DH`` is the REPLICATED ``(p, h, k)`` / ``(p, h, k,
    k)`` form) and then runs :func:`iterated_obs_forecast` per ``(model,
    series)``, feeding each horizon's predictive mean forward as the next lag.
    ``reg_mask`` (``(p,)``) gates the AR iteration to the TVAR models so the
    structural models packed alongside get the standard forecast. ``seed_lags``
    (``(p, n_reg)``) are the cutoff lags ``[y_t, y_{t-1}, ...]`` per series.

    Shapes match :func:`_hstep_predictive`: ``f_h, q_h`` → ``(nm, q, h)``;
    ``nu, weights`` → ``(nm, q)``.
    """
    C_post = axis0dot(
        dlm_state["Z"].swapaxes(-1, -2), dlm_state["Z"],  # C = Z' Z
    )
    init = {
        "m": dlm_state["m"],
        "s": dlm_state["s"],
        "nu": dlm_state["nu"],
        "C": C_post,
    }
    # disc_factor is the per-state (1-δ)/δ vector; dlm_uv_fcast_H builds the
    # system noise (block congruence W = B C B - C) internally.
    prop = _jvdlm_uv_fcast_H(
        disc_factor, variance_disc, variance_power, mult_comps, init, DH
    )
    aH, RH = prop["m"], prop["C"]                 # (p, h, k), (p, h, k, k)

    s0 = jnp.ravel(dlm_state["s"])
    vd = jnp.ravel(variance_disc)
    vp = jnp.ravel(variance_power)

    def _one(a, R, s, d, pw, m, F, sl, ir):
        return iterated_obs_forecast(a, R, s, d, pw, m, F, n_reg, sl, ir)

    f_h_t, q_h_t = vmap(_one)(aH, RH, s0, vd, vp, mult_comps, Fp, seed_lags, reg_mask)
    nu_t = dlm_state["nu"].reshape(nm, q)
    weights_t = weights.squeeze(-1)
    return f_h_t.reshape(nm, q, h), q_h_t.reshape(nm, q, h), nu_t, weights_t


def _build_emit_scan_step(
    multi,
    multi_params,
    dma_step_fn,
    DH,
    h,
    capture_trace=False,
    has_regressors=False,
    reg_mask=None,
    n_reg=0,
    Fp=None,
    exog_mode=False,
    reg_offset=0,
):
    """Scan step that conditionally emits the h-step predictive.

    Each step receives ``(ys_t, y_t, emit_flag_t, warmup_flag_t)`` as the scan
    input, or, for an AR/TVAR universe (``has_regressors``),
    ``(ys_t, y_t, emit_flag_t, warmup_flag_t, reg_t, seed_t)`` where ``reg_t`` is
    the ``(p, n_reg)`` filter-lag slice and ``seed_t`` the ``(p, n_reg)`` cutoff
    seed lags ``[y_t, ...]`` for the iterated forecast.
    ``warmup_flag_t`` is 0/1; 1 zeroes W for that step.
    The filter and DMA updates always run. The forecast emission is
    gated by ``emit_flag_t``: when ``True`` the h-step predictive runs and the
    result is emitted; when ``False``, ``lax.cond`` short-circuits to a
    zero-shaped output of the same pytree shape and the forecast compute is
    skipped entirely.

    Emits ``(f_h, q_h, nu, weights, emit_flag)`` per step:

      * ``f_h``, ``q_h`` shape ``(nm, q, h)``
      * ``nu``, ``weights`` shape ``(nm, q)``
      * ``emit_flag`` is the input flag, returned for post-scan gathering.

    ``has_regressors=False`` keeps the original structural scan untouched
    (bit-exact with the M1M reference).
    """

    disc_factor = (1 - multi.disc_rates * multi.disc_rates_damped) / (
        multi.disc_rates * multi.disc_rates_damped
    )

    # Adaptive-discount tau (None for a legacy static universe -> bit-exact).
    _tau = _adapt_tau(multi)

    # Output shapes for the cond branches
    fc_shape = (multi.nm, multi.q, h)
    nu_shape = (multi.nm, multi.q)
    # Pick up the dtype from the model so that float32/float64 setups
    # both work — _emit_true and _emit_false must agree under lax.cond.
    fp_dtype = multi.F.dtype

    def _emit_false(args):
        # Same pytree shape as _emit_true, all zeros — the cond requires
        # matching output structure between branches.
        return (
            jnp.zeros(fc_shape, dtype=fp_dtype),
            jnp.zeros(fc_shape, dtype=fp_dtype),
            jnp.zeros(nu_shape, dtype=fp_dtype),
            jnp.zeros(nu_shape, dtype=fp_dtype),
        )

    def _pack(carry, new_alloc_state, new_dlm_state, emitted, emit_flag_t, weights, f, q):
        f_h_t, q_h_t, nu_t, weights_t = emitted
        if capture_trace:
            weight_full_t = weights.squeeze(-1)  # (nm, n_series)
            return (
                (new_dlm_state, new_alloc_state),
                (f_h_t, q_h_t, nu_t, weights_t, emit_flag_t, weight_full_t, f, q),
            )
        return (
            (new_dlm_state, new_alloc_state),
            (f_h_t, q_h_t, nu_t, weights_t, emit_flag_t),
        )

    if has_regressors and exog_mode:
        # EXOGENOUS tail. Unlike AR, the horizon design is KNOWN, so there is no
        # iterated forecast: the standard flat predictive runs against an ``FH``
        # whose regressor columns carry this origin's future rows. That is the
        # only difference from the structural branch, and it is why exog needs a
        # per-cutoff ``FH`` rather than the batch-wide one (the analogue of the
        # AR path's per-cutoff ``seed_yts``).
        _reg0 = reg_offset

        def _emit_true(args):
            new_dlm_state, weights, xh_t = args          # xh_t: (p, h, n_reg)
            df = (disc_factor if _tau is None
                  else _disc_factor_from_tau(multi, new_dlm_state, _tau))
            DH_t = {"FH": DH["FH"].at[:, :, _reg0:_reg0 + n_reg].set(xh_t),
                    "GH": DH["GH"]}
            return _hstep_predictive_flat(
                new_dlm_state, weights, multi.nm, multi.q, h, df,
                multi.variance_disc, multi.variance_power, multi.mult_comps, DH_t,
            )

        def scan_step(carry, scan_in):
            ys_t, y_t, emit_flag_t, warmup_flag_t, reg_t, xh_t = scan_in
            dlm_state, alloc_state = carry
            new_dlm_state, _model, f, q = _multi_fwd_filter_step(
                dlm_state, multi_params, ys_t, warmup_flag_t,
                regressors=reg_t, tau=_tau, alpha=ADAPT_ALPHA,
            )
            fc_bundle = ForecastBundle(f[..., None], q[..., None])
            new_alloc_state, weights = dma_step_fn(alloc_state, fc_bundle, y_t)
            # same carry-shape restoration the AR branch does: the regression
            # filter step returns s/nu with a trailing axis the scan carry lacks
            new_dlm_state["s"] = new_dlm_state["s"].squeeze()
            new_dlm_state["nu"] = new_dlm_state["nu"].squeeze()
            emitted = cond(emit_flag_t, _emit_true, _emit_false,
                           (new_dlm_state, weights, xh_t))
            return _pack(carry, new_alloc_state, new_dlm_state, emitted,
                         emit_flag_t, weights, f, q)

        return scan_step

    if not has_regressors:
        # Honour the module flag so the emit-scan uses the same kernel as the
        # n_windows==1 fast path. ``DH`` is built to match (compact vs
        # replicated) by ``_run_cv_batch``.
        _predictive_fn = _hstep_predictive if _CV_USE_NESTED else _hstep_predictive_flat

        def _emit_true(args):
            new_dlm_state, weights = args
            # Adaptive: forecast at the origin's current discount (from S), not
            # the static d_min. _tau is concrete (decided at build time).
            df = (disc_factor if _tau is None
                  else _disc_factor_from_tau(multi, new_dlm_state, _tau))
            return _predictive_fn(
                new_dlm_state, weights, multi.nm, multi.q, h, df,
                multi.variance_disc, multi.variance_power, multi.mult_comps, DH,
            )

        def scan_step(carry, scan_in):
            ys_t, y_t, emit_flag_t, warmup_flag_t = scan_in
            dlm_state, alloc_state = carry

            new_dlm_state, _model, f, q = _multi_fwd_filter_step(
                dlm_state, multi_params, ys_t, warmup_flag_t, tau=_tau, alpha=ADAPT_ALPHA
            )
            fc_bundle = ForecastBundle(f[..., None], q[..., None])
            new_alloc_state, weights = dma_step_fn(alloc_state, fc_bundle, y_t)
            new_dlm_state["s"] = new_dlm_state["s"].squeeze()
            new_dlm_state["nu"] = new_dlm_state["nu"].squeeze()

            emitted = cond(
                emit_flag_t, _emit_true, _emit_false, (new_dlm_state, weights)
            )
            return _pack(carry, new_alloc_state, new_dlm_state, emitted,
                         emit_flag_t, weights, f, q)

        return scan_step

    # AR/TVAR universe: filter feeds the lags (masked to the TVAR models) and the
    # emit runs the iterated-expectations forecast with the cutoff seed lags. DH
    # here is the replicated (p, h, ...) form for _hstep_predictive_ar.
    #
    # In-jax stationarity flip for the multi-window emit-scan. The n_windows==1
    # fast path flips in numpy over the TVAR rows before forecasting; here we are
    # inside a lax.scan, so numpy can't reach the per-cutoff state. Instead flip
    # in-jax over ONLY the static ~5q TVAR rows (reg_mask is a concrete constant)
    # — the batched companion-eig stays ~300 MB, where jnp.eig over all p rows
    # block-diagonalises into an O((p*n_reg)^2) dense matrix and OOMs at M4 scale.
    # _flip_to_stationary matches the numpy _flip_ar_np to ~1e-5, so the emit path
    # and the fast path agree to that tolerance. Runs only on emit steps (inside
    # the lax.cond true branch). G_ar = I, so the flipped block propagates
    # unchanged through aH = G^j m.
    _do_flip = (
        reg_mask is not None and AR_STATIONARITY_EPS is not None and n_reg > 0
    )
    _tvar_idx = (
        jnp.asarray(np.nonzero(np.asarray(reg_mask).astype(bool))[0])
        if _do_flip else None
    )
    _reg0 = multi.k - n_reg

    def _stationarise_ar(dlm_state):
        if _tvar_idx is None or _tvar_idx.shape[0] == 0:
            return dlm_state
        sub = dlm_state["m"][_tvar_idx, _reg0:]               # (n_tvar, n_reg)
        flipped = vmap(
            lambda phi: _flip_to_stationary(phi, AR_STATIONARITY_EPS)
        )(sub)
        return {
            **dlm_state,
            "m": dlm_state["m"].at[_tvar_idx, _reg0:].set(flipped),
        }

    def _emit_true_ar(args):
        new_dlm_state, weights, seed_t = args
        new_dlm_state = _stationarise_ar(new_dlm_state)
        return _hstep_predictive_ar(
            new_dlm_state, weights, multi.nm, multi.q, h, disc_factor,
            multi.variance_disc, multi.variance_power, multi.mult_comps, DH,
            Fp, n_reg, reg_mask, seed_t,
        )

    def scan_step_ar(carry, scan_in):
        ys_t, y_t, emit_flag_t, warmup_flag_t, reg_t, seed_t = scan_in
        dlm_state, alloc_state = carry

        new_dlm_state, _model, f, q = _multi_fwd_filter_step(
            dlm_state, multi_params, ys_t, warmup_flag_t,
            regressors=reg_t, reg_mask=reg_mask, tau=_tau, alpha=ADAPT_ALPHA,
        )
        fc_bundle = ForecastBundle(f[..., None], q[..., None])
        new_alloc_state, weights = dma_step_fn(alloc_state, fc_bundle, y_t)
        new_dlm_state["s"] = new_dlm_state["s"].squeeze()
        new_dlm_state["nu"] = new_dlm_state["nu"].squeeze()

        emitted = cond(
            emit_flag_t, _emit_true_ar, _emit_false,
            (new_dlm_state, weights, seed_t),
        )
        return _pack(carry, new_alloc_state, new_dlm_state, emitted,
                     emit_flag_t, weights, f, q)

    return scan_step_ar


# =====================================================================
# Per-batch worker functions (top-level for pickling / Dask)
# =====================================================================


class UniverseContext(NamedTuple):
    """Read-only context handed to a custom ``universe_builder`` (the FFS
    front-end function that defines a non-standard model universe). Carries the
    AutoFFS config knobs the builder may use; the builder returns
    ``(models, model_desc)``."""
    season_length: Optional[int]
    n_seas_comps: Optional[int]
    warmup_steps: Optional[int]
    monitor_tau: Optional[float]
    error_nu0: Optional[float]
    component_priors: Optional[dict]


def _build_multi_and_dma(
    fit_array, periodicity, n_seas_comps, h, dma_pdr, dma_mdr, warmup_steps,
    include_ar=False, component_priors=None,
    weight_override=None, error_nu0=None, adaptive=False, tau_values=None,
    var_disc_values=None, monitor_tau=None, universe_builder=None,
):
    """Construct a fresh ``multi_model_dlm`` and ``Allocator`` for a batch.

    ``component_priors`` (dict name -> (m0, C0)) seeds the structural components'
    informative prior; ``weight_override`` (dict with ``pset``/``mset`` arrays)
    replaces the uniform initial DMA weights. Both default to None -> the
    diffuse-prior / uniform-weight behaviour is unchanged.

    Returns
    -------
    multi, dma, model_indicator
    """
    fit_data = pd.DataFrame(np.asarray(fit_array))

    if warmup_steps is not None and warmup_steps > 0:
        # Diffuse-prior path: elicit from the whole fit window with
        # nan-aware moments. The short legacy windows below assume real
        # data at the series start; a long leading-NaN / structural-zero
        # run (e.g. a late-launching M5 product) would otherwise fall
        # entirely inside the window and yield an all-NaN prior.
        init_data = fit_data
    elif periodicity is not None:
        init_data = fit_data.iloc[: int(np.ceil(2 * periodicity))]
    else:
        init_data = fit_data.iloc[: min(10, len(fit_data))]

    _, n = init_data.shape

    if universe_builder is not None:
        # User-supplied universe (defined in the FFS front end). It OWNS the
        # model set; it receives the config knobs via UniverseContext and must
        # return (models, model_desc) where model_desc is indexed in models
        # order and carries a 'Class' column (the DMA between-class layer).
        ctx = UniverseContext(
            season_length=periodicity, n_seas_comps=n_seas_comps,
            warmup_steps=warmup_steps,
            monitor_tau=monitor_tau, error_nu0=error_nu0,
            component_priors=component_priors,
        )
        ATS, model_desc = universe_builder(init_data, h, ctx)
    elif adaptive:
        # Error-monitoring (tau) universe; no AR/TVAR class.
        ATS, model_desc = assemble_models_adaptive(
            init_data, h, periodicity, n_seas_comps, warmup_steps=warmup_steps,
            tau_values=tau_values, var_disc_values=var_disc_values,
            component_priors=component_priors, error_nu0=error_nu0,
        )
    else:
        ATS, model_desc = assemble_models(
            init_data, h, periodicity, n_seas_comps, warmup_steps=warmup_steps,
            include_ar=include_ar,
            component_priors=component_priors, error_nu0=error_nu0,
            monitor_tau=monitor_tau,
        )
    multi = multi_model_dlm(ATS, devices.dlm_compute)

    dma, model_indicator = _dma_for_multi(
        len(ATS), n, model_desc, dma_pdr, dma_mdr, weight_override=weight_override
    )
    return multi, dma, model_indicator


def _dma_for_multi(k, n_series, model_desc, dma_pdr, dma_mdr, weight_override=None):
    """Build the 2-level DMA allocator for a packed universe.

    Factored from :func:`_build_multi_and_dma` so a pre-built ``multi`` (e.g.
    ``StaticBlock.from_multi``, where the user assembled the universe from
    ``uv_dlm`` cells) gets the identical allocator. ``model_desc`` must carry a
    ``'Class'`` column (the between-class DMA layer); ``k`` is the model count
    (``multi.nm``), ``n_series`` the batch width (``multi.q``).
    """
    model_indicator = make_model_indicator(model_desc, by="Class")
    nmset = model_indicator.shape[1]
    dma = Allocator(
        scoring_rule=LogScore,
        update_rule=Partial(PowerLawUpdate, dma_pdr=dma_pdr, dma_mdr=dma_mdr, c=1e-3),
        model_indicator=model_indicator.values,
        device=devices.allocation_compute,
    )
    dma.init(
        n_models=k,
        n_series=n_series,
        n_classes=nmset,
        model_indicator=model_indicator.values,
    )
    if weight_override is not None:
        # Replace the uniform initial DMA weights with an informative prior
        # (e.g. hierarchy-averaged parent weights). The subsequent filter then
        # updates from this prior rather than from uniform.
        dma.state = _apply_weight_override(dma.state, weight_override)
    return dma, model_indicator


def _build_universe(*args, **kwargs):
    """Orchestrator-facing construction seam: build the batch's :class:`Universe`
    (a list of blocks + the DMA allocator).

Wraps the single ``multi_model_dlm`` produced by
    :func:`_build_multi_and_dma`, which is the model-construction factory.
    Arguments forward verbatim to it.
    """
    multi, dma, model_indicator = _build_multi_and_dma(*args, **kwargs)
    return Universe(blocks=[multi], dma=dma, model_indicator=model_indicator)


def _run_static_block_cv(block, srs_ids, arr, cutoff_t_idx, h, warmup_steps,
                         capture_trace):
    """Canonical CV worker: produce a batch's CV trajectory through a
    :class:`~DLMAX.ffs.static_block.StaticBlock`. Delegates to ``_run_cv_batch``
    — bit-identical to the legacy path for a single static block; the seam
    through which a second block (the grid) and the union DMA are combined in
    5b-ii."""
    return block.forecast_rolling(
        srs_ids, arr, cutoff_t_idx, h, warmup_steps, capture_trace
    )


def _apply_weight_override(alloc_state, weight_override):
    """Return ``alloc_state`` with its ``pset``/``mset`` replaced by the supplied
    informative prior. ``weight_override`` keys (any subset): ``pset`` shaped
    ``(n_models, n_series, h)``, ``mset`` shaped ``(n_classes, n_series, h)`` —
    matching :func:`init_alloc_state`. Missing keys keep the uniform init."""
    pset = weight_override.get("pset")
    mset = weight_override.get("mset")
    return alloc_state._replace(
        pset=jnp.asarray(pset) if pset is not None else alloc_state.pset,
        mset=jnp.asarray(mset) if mset is not None else alloc_state.mset,
    )


def _multi_params_tuple(multi):
    return (
        multi.F,
        multi.G,
        multi.nm,
        multi.k,
        multi.q,
        multi.p,
        multi.disc_rates,
        multi.disc_rates_damped,
        multi.monitor_inject,
        multi.variance_disc,
        multi.variance_power,
        multi.mult_comps,
    )


def _scan_filter(multi, dma, fit_array, warmup_steps: int = 0, lag_history=None,
                 exog_array=None):
    """Run the no-emission filter scan over ``fit_array``.

    When the universe carries a regression tail (``multi.n_regressors > 0``) the
    per-step regressor slices are threaded as a fourth scan input, sourced two
    ways:

    * **Exogenous** (``multi.exog_regressors``) — the caller supplies the design
      matrix ``exog_array`` (shape ``(T, n_series, n_regressors)``, same time/
      series layout as ``fit_array``), formatted by
      :meth:`multi_model_dlm.format_exog_yts`. Used for SNAP / promotion drivers.
    * **Autoregressive** — the lag slices are derived from ``fit_array`` itself
      via :meth:`multi_model_dlm.format_lag_yts`; ``lag_history`` (shape
      ``(n_regressors, n_series)``) seeds the first steps' lags when continuing
      across batches (the ``update`` path).

    A structural universe (``n_regressors == 0``) uses the original three-input
    scan unchanged — bit-exact with the M1M reference.

    Returns
    -------
    final_dlm_state, final_alloc_state, all_weights
    """
    has_reg = getattr(multi, "n_regressors", 0) > 0
    multi_params = _multi_params_tuple(multi)
    dma_step_fn = dma.prepared_step()
    scan_step = _build_filter_scan_step(
        multi_params, dma_step_fn, has_reg, getattr(multi, "reg_mask", None),
        tau=_adapt_tau(multi), alpha=ADAPT_ALPHA,
    )
    T = fit_array.shape[0]

    ws = warmup_steps if warmup_steps is not None else 0
    warmup_flags = jnp.where(
        jnp.arange(T) < ws,
        jnp.array(1.0),
        jnp.array(0.0),
    )

    yts = multi.format_yts(fit_array)
    init_carry = (multi.dlm_state, dma.state)
    xs = [
        yts,
        device_put(fit_array, devices.allocation_compute),
        device_put(warmup_flags, devices.allocation_compute),
    ]
    if has_reg and getattr(multi, "exog_regressors", False):
        if exog_array is None:
            raise ValueError(
                "exog_regressors universe requires exog_array in _scan_filter; "
                "the streaming driver must supply the regressor design matrix."
            )
        xs.append(multi.format_exog_yts(exog_array))
    elif has_reg:
        xs.append(multi.format_lag_yts(fit_array, history=lag_history))
    (final_dlm_state, final_alloc_state), all_weights = scan(
        scan_step,
        init_carry,
        tuple(xs),
    )
    return final_dlm_state, final_alloc_state, all_weights


def _warn_under_seasonal(short_index, min_len, season_length):
    """Non-blocking warning that some series are too short to determine the
    seasonal component (fewer than 2 seasonal cycles).

    Short series are accepted: the filter runs and leans on the diffuse prior,
    so the seasonal component is simply under-determined.
    """
    if len(short_index):
        warnings.warn(
            f"{len(short_index)} series have fewer than {min_len} observations "
            f"(< 2 cycles of season_length={season_length}); the seasonal "
            f"component is under-determined and these forecasts lean on the "
            f"diffuse prior. Examples: {list(short_index[:5])}.",
            stacklevel=3,
        )


def _lag_tail_from(prev_tail, new_array, n_reg):
    """The trailing ``n_reg`` observations after ingesting ``new_array``, as
    (n_reg, n_series) oldest-first — seeds AR forecast lags and the next batch's
    filter continuation. Returns None when there is no regression tail."""
    if n_reg == 0:
        return None
    new_array = np.asarray(new_array)
    if prev_tail is not None:
        new_array = np.concatenate([np.asarray(prev_tail), new_array], axis=0)
    return new_array[-n_reg:]


def _run_fit_batch(
    srs_ids,
    fit_array,
    last_ds_per_sid,
    periodicity,
    n_seas_comps,
    h_template,
    dma_pdr,
    dma_mdr,
    warmup_steps: int = 0,
    include_ar: bool = False,
    adaptive: bool = False,
    tau_values=None,
    var_disc_values=None,
    monitor_tau=MONITOR_TAU,
    universe_builder=None,
    component_priors=None,
    weight_override=None,
    error_nu0=None,
    exog_array=None,
):
    """Construct a fresh batch state and run the filter to the end of fit_array.

    ``h_template`` is only used to size the model templates inside
    ``assemble_models`` (which builds them with an h-ahead view); it does
    not constrain prediction horizons later.

    ``component_priors`` / ``weight_override`` seed an informative state prior /
    DMA-weight prior (used by ``add_series`` to warm-start a new series); both
    default to None -> diffuse prior + uniform weights.

    ``exog_array`` (shape ``(T, n_series, n_regressors)``, aligned to
    ``fit_array``) supplies the exogenous-regressor design matrix when the
    universe carries an exog tail; ignored for structural / AR universes.
    """
    multi, dma, model_indicator = _build_multi_and_dma(
        fit_array, periodicity, n_seas_comps, h_template, dma_pdr, dma_mdr,
        warmup_steps, include_ar=include_ar,
        component_priors=component_priors, weight_override=weight_override,
        error_nu0=error_nu0, adaptive=adaptive, tau_values=tau_values,
        var_disc_values=var_disc_values, monitor_tau=monitor_tau,
        universe_builder=universe_builder,
    )

    final_dlm_state, final_alloc_state, all_weights = _scan_filter(
        multi, dma, fit_array, warmup_steps, exog_array=exog_array
    )
    multi.dlm_state = final_dlm_state
    dma.state = final_alloc_state
    latest_weights = all_weights[-1]  # (nm, q, 1)

    return _BatchState(
        srs_ids=srs_ids,
        multi=multi,
        dma=dma,
        model_indicator=model_indicator,
        latest_weights=np.asarray(latest_weights),
        last_ds_per_sid=last_ds_per_sid,
        T_ingested=int(fit_array.shape[0]),
        lag_tail=_lag_tail_from(None, fit_array, multi.n_regressors),
    )


def _run_update_batch(state, new_array, last_ds_per_sid_new, warmup_steps: int = 0,
                      exog_array=None):
    """Continue the filter from ``state`` using ``new_array`` only.
    ``new_array`` shape ``(T_new, n_series)``. Series order must match
    ``state.srs_ids``. Returns an updated ``_BatchState``.

    ``exog_array`` (shape ``(T_new, n_series, n_regressors)``, aligned to
    ``new_array``) supplies the exogenous-regressor values for the new steps
    when the universe carries an exog tail. Unlike the AR ``lag_tail``, exog
    values are external (not derived from past y), so no cross-batch lag
    continuation is needed.
    """
    multi = state.multi
    dma = state.dma

    final_dlm_state, final_alloc_state, all_weights = _scan_filter(
        multi, dma, new_array, warmup_steps, lag_history=state.lag_tail,
        exog_array=exog_array,
    )
    multi.dlm_state = final_dlm_state
    dma.state = final_alloc_state
    latest_weights = all_weights[-1]

    merged_last_ds = dict(state.last_ds_per_sid)
    merged_last_ds.update(last_ds_per_sid_new)

    return _BatchState(
        srs_ids=state.srs_ids,
        multi=multi,
        dma=dma,
        model_indicator=state.model_indicator,
        latest_weights=np.asarray(latest_weights),
        last_ds_per_sid=merged_last_ds,
        T_ingested=state.T_ingested + int(new_array.shape[0]),
        lag_tail=_lag_tail_from(
            state.lag_tail, new_array, getattr(multi, "n_regressors", 0)
        ),
    )


def _run_predict_batch(state, h, sd_method="quantile", exog_future=None):
    """h-step forecast from the current ``_BatchState``.

    ``exog_future`` (shape ``(h, n_series, n_regressors)``, series in
    ``state.srs_ids`` order) supplies the known future regressor values over the
    horizon when the universe carries an exogenous tail; required in that case,
    ignored otherwise.
    """
    multi = state.multi
    dma = state.dma

    if getattr(multi, "exog_regressors", False):
        # Exogenous universe (e.g. SNAP): the future regressor values are known
        # and supplied by the caller; deterministic regressors (future_var=None).
        if exog_future is None:
            raise ValueError(
                "exog_regressors universe requires exog_future in "
                "_run_predict_batch (known future regressor design over the "
                "horizon)."
            )
        f_h, q_h = multi.forecast_exog(h=h, future_mean=exog_future)
    elif getattr(multi, "n_regressors", 0) > 0:
        # AR/TVAR universe: iterated-expectations forecast, with the AR lags
        # seeded from the trailing observations (most-recent-first per series).
        n_reg = multi.n_regressors
        seed_lags = np.asarray(state.lag_tail)[::-1].T  # (n_series, n_reg)
        f_h, q_h = multi.forecast_iterated(h=h, seed_lags=seed_lags)
    else:
        f_h, q_h = multi.forecast(h=h)
    f_h = device_put(f_h, devices.allocation_compute)
    q_h = device_put(q_h, devices.allocation_compute)

    final_weights = jnp.asarray(state.latest_weights)
    fc_bundle = ForecastBundle(f_h, q_h)
    fc_final = dma.combine(final_weights, fc_bundle)  # used only for .loc

    final_nu = multi.dlm_state["nu"].reshape(multi.nm, multi.q)
    weights_squeezed = final_weights.squeeze(-1)

    f_h_np = np.asarray(f_h)
    q_h_np = np.asarray(q_h)
    nu_np = np.asarray(final_nu)
    weights_np = np.asarray(weights_squeezed)

    # Combined predictive SD (quantile/Vincent by default; see
    # _combined_predictive_sd), transposed to loc's (n_series, h) convention.
    sd_combined = _combined_predictive_sd(
        FFSPredictive(
            loc=None,
            sd=None,
            f_h=f_h_np,
            q_h=q_h_np,
            nu=nu_np,
            weights=weights_np,
        ),
        method=sd_method,
    ).T  # (n_series, h)

    return FFSPredictive(
        loc=np.asarray(fc_final.loc),
        sd=sd_combined,
        f_h=f_h_np,
        q_h=q_h_np,
        nu=nu_np,
        weights=weights_np,
    )


# =====================================================================
# Path-based batch workers (distributed per-period universe step)
# =====================================================================
# These load the (large) batch state from SHARED storage on the worker itself
# and write it back in place, so only KB-scale payloads cross the wire — the
# right shape for distributing one streaming period across the cluster
# (AutoFFSUniverse.update / .forecast). With dask_client=None the dispatcher
# runs them in-process, with identical results.


def _update_batch_file(batch_path, new_arr, new_last_ds, exog_array,
                       dma_pdr, dma_mdr, warmup_steps=0):
    """Load batch state from ``batch_path``, advance it by ``new_arr``, save it
    back in place. Returns ``{sid: last_ds}`` for the active slots (light, for
    the head's manifest). ``new_arr`` / ``exog_array`` are series-ordered to the
    batch's ``srs_ids`` (built head-side from cheap metadata)."""
    state, active_arr = _load_batch_state(batch_path)
    _attach_dma_step_fn(state.dma, dma_pdr, dma_mdr)
    new_state = _run_update_batch(
        state, new_arr, new_last_ds, warmup_steps, exog_array=exog_array
    )
    _save_batch_state_with_active(new_state, batch_path, active_arr)
    return {sid: new_last_ds[sid]
            for sid, a in zip(state.srs_ids, active_arr) if a}


def _predict_batch_file(batch_path, h, sd_method, exog_future, dma_pdr, dma_mdr):
    """Load batch state from ``batch_path`` and forecast ``h`` steps. Returns
    ``(pred, srs_ids, active, mdl_keys, model_indicator)`` — everything the head
    needs to assemble the combined or per-model output without having loaded the
    state itself."""
    state, active_arr = _load_batch_state(batch_path)
    _attach_dma_step_fn(state.dma, dma_pdr, dma_mdr)
    pred = _run_predict_batch(state, h, sd_method, exog_future=exog_future)
    return (pred, state.srs_ids, np.asarray(active_arr),
            list(state.multi.mdl_keys), state.model_indicator)


# --- grid-mode path-based batch workers (AutoFFSUniverse grid mode) -----------
# Module-level + picklable, so the streaming fit/update/forecast fan out to Dask
# workers when a client is set, and loop in-process (bit-identical) when not.
# `cfg` is the picklable grid config: (period, var_powers, warmup, offset, pdr,
# mdr, capacity).
def _grid_block_from_cfg(cfg, init_data=None):
    from DLMAX.ffs.grid_block import GridBlock
    period, var_powers, warmup, offset, pdr, mdr, capacity = cfg[:7]
    # A block_builder supersedes the config knobs entirely: the caller owns the
    # taxonomy. Appended at index 13 so every existing position is untouched and
    # a legacy tuple still rebuilds byte-identically. ``init_data`` is the fit
    # window when there is one, None on a rebuild-to-load (see
    # ``_blocks_from_builder``).
    builder = cfg[13] if len(cfg) > 13 else None
    if builder is not None:
        blocks = _blocks_from_builder(builder, cfg[14] if len(cfg) > 14 else {},
                                      init_data)
        if len(blocks) != 1:
            raise ValueError(
                f"block_builder returned {len(blocks)} blocks for the "
                f"single-block path; expected 1.")
        blocks[0].capacity = capacity
        return blocks[0]
    # Optional MAP-prior / objective knobs (appended; absent -> defaults, so a
    # legacy 7-tuple rebuilds byte-identically).
    disc_prior = cfg[7] if len(cfg) > 7 else None
    additive_logscore = bool(cfg[8]) if len(cfg) > 8 else False
    decouple_trend = bool(cfg[9]) if len(cfg) > 9 else False
    learn_dma = bool(cfg[10]) if len(cfg) > 10 else False
    dma_prior = cfg[11] if len(cfg) > 11 else None
    seasonal_prior = cfg[12] if len(cfg) > 12 else None
    b = GridBlock.build(period=period, warmup=warmup, var_powers=var_powers,
                        offset=offset, pdr=pdr, mdr=mdr,
                        disc_prior=disc_prior, additive_logscore=additive_logscore,
                        decouple_trend=decouple_trend, learn_dma=learn_dma,
                        dma_prior=dma_prior, seasonal_prior=seasonal_prior)
    b.capacity = capacity
    return b


def _save_grid_batch_file(block, path, srs_ids, active, last_ds_arr, is_int):
    """Persist a grid batch: block carry (/grid_state) + /metadata in the format
    ``_load_batch_meta`` reads."""
    block.save(path)  # writes /grid_state (mode 'w')
    with h5py.File(path, "a") as f:
        if "metadata" in f:
            del f["metadata"]
        g = f.create_group("metadata")
        g.create_dataset("schema_version", data=_BATCH_SCHEMA_VERSION)
        g.create_dataset("srs_ids", data=[str(s).encode() for s in srs_ids])
        g.create_dataset("last_ds", data=np.asarray(last_ds_arr))
        g.create_dataset("last_ds_is_int", data=int(is_int))
        g.create_dataset("active", data=np.asarray(active, dtype=bool))


def _pad_grid_meta(srs_ids, active, last_ds_arr, cap):
    """Pad a grid batch's metadata to ``cap`` with inactive placeholder slots
    (``__pad_i``), mirroring the block's ``pad_to``. No-op if cap is None or the
    batch is already at/over cap."""
    q = len(srs_ids)
    active = np.asarray(active, dtype=bool)
    last_ds_arr = np.asarray(last_ds_arr)
    if cap is None or q >= cap:
        return list(srs_ids), active, last_ds_arr
    pad = cap - q
    srs = list(srs_ids) + [f"{_PAD_PREFIX}{i}" for i in range(pad)]
    active = np.concatenate([active, np.zeros(pad, bool)])
    last_ds_arr = np.concatenate([last_ds_arr, np.full(pad, last_ds_arr[0])])
    return srs, active, last_ds_arr


def _grid_fit_batch_file(batch_path, arr, srs_ids, last_ds_arr, is_int, cfg,
                         exog_array=None):
    """Build a fresh grid block, fit it over ``arr`` (T, q), pad to capacity with
    placeholder slots, and save it + cap-wide metadata.

    ``exog_array`` ``(T, q, n_regs)`` is the regression tail's design over the
    same rows as ``arr``; ``None`` for a structural block. Appended last so a
    legacy call site is unaffected."""
    cap = cfg[6]                                        # max_batch_size
    a = np.asarray(arr, dtype=float)
    block = _grid_block_from_cfg(cfg, init_data=a)
    block.scan_filter(a, regressors=exog_array)
    if cap is not None:
        block.pad_to(cap)
    srs, active, last_ds = _pad_grid_meta(list(srs_ids), [True] * len(srs_ids),
                                          last_ds_arr, cap)
    _save_grid_batch_file(block, batch_path, srs, active, last_ds, is_int)
    return None


def _grid_update_batch_file(batch_path, new_arr, new_last_ds_arr, is_int, cfg,
                            exog_array=None):
    """Load a grid block, advance it over ``new_arr`` (T_new, q), save it back.

    ``exog_array`` ``(T_new, q, n_regs)`` covers the NEW rows only, aligned to
    ``new_arr`` — unlike the AR seed lags, exogenous rows are known per step and
    need no carry.

    Resume-safe idempotence: if this batch's active slots are already at the
    target ``last_ds`` (a re-run after a mid-update kill), skip — no double
    advance. So re-running an origin's update advances only the batches that
    hadn't been saved yet."""
    srs_ids, active_arr, last_ds = _load_batch_meta(batch_path)
    active = np.asarray(active_arr, dtype=bool)
    cur = np.array([int(last_ds[s]) if is_int else pd.Timestamp(last_ds[s]).value
                    for s in srs_ids])
    tgt = np.asarray(new_last_ds_arr)
    if len(cur) == len(tgt) and np.all(cur[active] >= tgt[active]):
        return None                # already at/past this origin — never re-advance
    block = _grid_block_from_cfg(cfg)
    block.load(batch_path)
    block.scan_filter(np.asarray(new_arr, dtype=float), regressors=exog_array)
    _save_grid_batch_file(block, batch_path, srs_ids, active, new_last_ds_arr, is_int)
    return None


def _grid_predict_batch_file(batch_path, h, cfg, exog_future=None):
    """Load a grid block and forecast ``h``. Returns ``(loc, sd, srs_ids, active,
    last_ds)`` for the head to assemble.

    ``exog_future`` is the head's ``(h, q, n_regs)`` design — TIME-major, as
    ``_materialise_exog`` returns it and as the legacy predict path consumes it.
    ``GridBlock.forecast`` wants it SERIES-major ``(q, h, n_regs)``, so the swap
    happens here rather than at the call site: one place, next to the consumer,
    and the head keeps a single exog layout everywhere.
    """
    srs_ids, active_arr, last_ds = _load_batch_meta(batch_path)
    block = _grid_block_from_cfg(cfg)
    block.load(batch_path)
    xh = (None if exog_future is None
          else np.swapaxes(np.asarray(exog_future, dtype=float), 0, 1))
    loc, sd, _comp = block.forecast(h, exog_future=xh)
    return (np.asarray(loc), np.asarray(sd), srs_ids, np.asarray(active_arr), last_ds)


# --- multi-block grid-mode workers (>=2 blocks: Universe(blocks=[...]) + union DMA)
# The single-block workers above stay byte-identical; these activate only when the
# universe config carries >=2 block specs. `cfg` = (block_specs, union_pdr,
# union_mdr, capacity) with block_specs a tuple of (period, var_powers, warmup,
# offset, pdr, mdr). Fit/forecast reuse the gated `_multiblock_fit`/
# `_multiblock_forecast` core; persistence writes each block to /blocks/<i> plus a
# /union_state group (the AllocatorState carry + combining weights).
def _blocks_from_builder(builder, ctx, init_data=None):
    """Call a user ``block_builder`` and validate what it returns.

    Contract: ``builder(init_data, h, ctx) -> list[Block]`` — the same shape as
    ``universe_builder``, so the two facilities read alike.

    ``init_data`` is a ``(T, q)`` DataFrame of the window the block is about to
    be FITTED on (the batch's fit array, or one new series' history under
    ``add_series``), and ``None`` when the block is being rebuilt only to
    receive a persisted carry — update, forecast, reopen. It is passed because
    ``AdaptiveBlock`` builders need it: their cells come from
    ``DLM.compile(init_data=...)``, which will not compile without one.

    THE RULE THAT MAKES THAT SAFE: the block a builder returns must depend on
    ``init_data``'s VALUES and WIDTH not at all — only on ``ctx`` and ``h``. A
    grid's taxonomy is data-independent by construction (``AdaptiveBlock``
    keeps only the cells' ``GridModel`` + components, both functions of the
    component spec), the series axis is sized from the real data at
    ``scan_filter``, and the diffuse prior is elicited there too. Anything a
    builder derived from the data would therefore be discarded on the fit path
    and unavailable on the rebuild path — where the block must come out
    structurally identical or it cannot take the carry back.

    ``h`` is the universe's forecast-horizon template (also in ``ctx``).

    The callable itself is never persisted (only ``block_builder_name``); it is
    shipped to workers by reference, so it must be importable there — the same
    constraint ``universe_builder`` carries.
    """
    ctx = dict(ctx)
    if init_data is not None and not isinstance(init_data, pd.DataFrame):
        init_data = pd.DataFrame(np.asarray(init_data, dtype=float))
    blocks = builder(init_data, ctx.get("h"), ctx)
    if isinstance(blocks, (list, tuple)):
        blocks = list(blocks)
    else:
        blocks = [blocks]
    if not blocks:
        raise ValueError("block_builder returned no blocks.")
    for b in blocks:
        for face in ("scan_filter", "forecast"):
            if not hasattr(b, face):
                raise TypeError(
                    f"block_builder returned {type(b).__name__}, which has no "
                    f"'{face}' — the streaming path needs a production block "
                    f"face (e.g. GridBlock).")
    return blocks


def _grid_blocks_from_cfg(cfg, init_data=None):
    from DLMAX.ffs.grid_block import GridBlock
    # (block_specs, union_pdr, union_mdr, capacity[, union_learn_dma, union_dma_prior])
    block_specs, union_pdr, union_mdr, capacity = cfg[:4]
    builder = cfg[6] if len(cfg) > 6 else None
    if builder is not None:
        blocks = _blocks_from_builder(builder, cfg[7] if len(cfg) > 7 else {},
                                      init_data)
        for b in blocks:
            b.capacity = capacity
        return blocks, union_pdr, union_mdr, capacity
    blocks = []
    for spec in block_specs:
        period, var_powers, warmup, offset, pdr, mdr = spec[:6]
        seasonal_mult = bool(spec[6]) if len(spec) > 6 else False   # 6-tuple = additive
        additive_logscore = bool(spec[7]) if len(spec) > 7 else False
        disc_prior = spec[8] if len(spec) > 8 else None
        decouple_trend = bool(spec[9]) if len(spec) > 9 else False
        learn_dma = bool(spec[10]) if len(spec) > 10 else False
        dma_prior = spec[11] if len(spec) > 11 else None
        # Appended last: a spec persisted before this existed is 12 long and
        # rebuilds with None, i.e. byte-identically to how it ran.
        seasonal_prior = spec[12] if len(spec) > 12 else None
        b = GridBlock.build(period=period, warmup=warmup, var_powers=var_powers,
                            offset=offset, pdr=pdr, mdr=mdr,
                            seasonal_mult=seasonal_mult,
                            additive_logscore=additive_logscore,
                            disc_prior=disc_prior, decouple_trend=decouple_trend,
                            learn_dma=learn_dma, dma_prior=dma_prior,
                            seasonal_prior=seasonal_prior)
        b.capacity = capacity
        blocks.append(b)
    return blocks, union_pdr, union_mdr, capacity


def _pad_union_state(union_state, weights, q, cap):
    """Pad the union carry + combining weights series dim (``q`` -> ``cap``) with
    placeholder slot-0 replicas, mirroring ``GridBlock.pad_to``. Handles both the
    fixed ``AllocatorState`` (series on a middle axis) and the SGDDMA carry (series
    on the leading axis of every leaf -> a uniform axis-0 tree op)."""
    if cap is None or q >= cap:
        return union_state, np.asarray(weights)
    pad = cap - q
    w_pad = np.asarray(jnp.concatenate(
        [jnp.asarray(weights),
         jnp.repeat(jnp.asarray(weights)[:, :1], pad, axis=1)], axis=1))  # (M, cap)
    if not isinstance(union_state, AllocatorState):     # SGDDMA carry — axis-0 series
        new_state = jax.tree_util.tree_map(
            lambda x: jnp.concatenate(
                [jnp.asarray(x), jnp.repeat(jnp.asarray(x)[:1], pad, axis=0)], axis=0),
            union_state)
        return new_state, w_pad

    def rep(x, ax):
        r = jnp.repeat(jnp.take(x, jnp.array([0]), axis=ax), pad, axis=ax)
        return jnp.concatenate([x, r], axis=ax)

    new_state = union_state._replace(
        pset=rep(union_state.pset, 1),           # (nm, n_series, h)
        mset=rep(union_state.mset, 1),           # (nc, n_series, h)
        forecast_history=rep(union_state.forecast_history, 2))  # (h, nm, n_series, 2)
    return new_state, w_pad


def _union_set_slot(union_state, weights, idx, new_state, new_weights):
    """Fill series slot ``idx`` of the union state + weights with a freshly-fit
    single-series union carry (``new_state`` n_series=1). The AllocatorState
    analogue of ``GridBlock.set_slot`` — add_series filling a placeholder."""
    if not isinstance(union_state, AllocatorState):     # SGDDMA carry — axis-0 series
        us = jax.tree_util.tree_map(
            lambda x, n: jnp.asarray(x).at[idx].set(jnp.asarray(n)[0]),
            union_state, new_state)
    else:
        us = union_state._replace(
            pset=union_state.pset.at[:, idx].set(new_state.pset[:, 0]),
            mset=union_state.mset.at[:, idx].set(new_state.mset[:, 0]),
            forecast_history=union_state.forecast_history.at[:, :, idx].set(
                new_state.forecast_history[:, :, 0]))
    w = np.asarray(weights).copy()
    w[:, idx] = np.asarray(new_weights)[:, 0]
    return us, w


def _union_append(union_state, weights, new_state, new_weights):
    """Concatenate a single-series union carry onto the batch's union state +
    weights (no-capacity add_series — grows the series axis)."""
    if not isinstance(union_state, AllocatorState):     # SGDDMA carry — axis-0 series
        us = jax.tree_util.tree_map(
            lambda x, n: jnp.concatenate([jnp.asarray(x), jnp.asarray(n)], axis=0),
            union_state, new_state)
    else:
        us = union_state._replace(
            pset=jnp.concatenate([union_state.pset, new_state.pset], axis=1),
            mset=jnp.concatenate([union_state.mset, new_state.mset], axis=1),
            forecast_history=jnp.concatenate(
                [union_state.forecast_history, new_state.forecast_history], axis=2))
    w = np.concatenate([np.asarray(weights), np.asarray(new_weights)], axis=1)
    return us, w


def _save_multiblock_batch_file(blocks, union_state, weights, path, srs_ids,
                                active, last_ds_arr, is_int):
    """Persist a multi-block batch: each block -> /blocks/<i>, the union
    AllocatorState + weights -> /union_state, plus /metadata (as _load_batch_meta
    reads)."""
    import pickle
    import jax
    for i, b in enumerate(blocks):
        b.save(path, group=f"blocks/{i}", mode=("w" if i == 0 else "a"))
    with h5py.File(path, "a") as f:
        for grp in ("union_state", "metadata"):
            if grp in f:
                del f[grp]
        gu = f.create_group("union_state")
        leaves, treedef = jax.tree_util.tree_flatten(union_state)
        gu.attrs["treedef"] = np.void(pickle.dumps(treedef))
        gu.attrs["n_leaves"] = len(leaves)
        gu.attrs["n_blocks"] = len(blocks)
        for j, lf in enumerate(leaves):
            gu.create_dataset(f"leaf_{j}", data=np.asarray(lf))
        gu.create_dataset("weights", data=np.asarray(weights))
        g = f.create_group("metadata")
        g.create_dataset("schema_version", data=_BATCH_SCHEMA_VERSION)
        g.create_dataset("srs_ids", data=[str(s).encode() for s in srs_ids])
        g.create_dataset("last_ds", data=np.asarray(last_ds_arr))
        g.create_dataset("last_ds_is_int", data=int(is_int))
        g.create_dataset("active", data=np.asarray(active, dtype=bool))


def _load_multiblock_batch(path, cfg):
    """Rebuild blocks (from ``cfg``) + load their carries + the union state/weights."""
    import pickle
    import jax
    blocks, union_pdr, union_mdr, capacity = _grid_blocks_from_cfg(cfg)
    for i, b in enumerate(blocks):
        b.load(path, group=f"blocks/{i}")
    with h5py.File(path, "r") as f:
        gu = f["union_state"]
        treedef = pickle.loads(gu.attrs["treedef"].tobytes())
        leaves = [jnp.asarray(gu[f"leaf_{j}"][()])
                  for j in range(int(gu.attrs["n_leaves"]))]
        weights = np.asarray(gu["weights"][()])
    union_state = jax.tree_util.tree_unflatten(treedef, leaves)
    return blocks, union_state, weights, capacity


def _multiblock_fit_batch_file(batch_path, arr, srs_ids, last_ds_arr, is_int, cfg,
                               exog_array=None):
    """Fit fresh blocks + build the union carry over ``arr`` (T, q), pad to
    capacity (blocks + union state), and save."""
    if exog_array is not None:
        raise ValueError(
            "exogenous regressors are not wired on the grid MULTI-block path "
            "(see AutoFFSUniverse._check_exog_supported). The parameter exists "
            "only so the head can dispatch single- and multi-block workers "
            "through one call site; reaching here means that gate was bypassed.")
    a = np.asarray(arr, dtype=float)
    blocks, union_pdr, union_mdr, capacity = _grid_blocks_from_cfg(cfg, init_data=a)
    learn_dma = bool(cfg[4]) if len(cfg) > 4 else False
    dma_prior = cfg[5] if len(cfg) > 5 else None
    q = a.shape[1]
    union_state, weights, _umi = _multiblock_fit(blocks, a, union_pdr, union_mdr,
                                                 learn_dma=learn_dma, dma_prior=dma_prior)
    if capacity is not None:
        for b in blocks:
            b.pad_to(capacity)
        union_state, weights = _pad_union_state(union_state, weights, q, capacity)
    srs, active, last_ds = _pad_grid_meta(list(srs_ids), [True] * len(srs_ids),
                                          last_ds_arr, capacity)
    _save_multiblock_batch_file(blocks, union_state, weights, batch_path, srs,
                                active, last_ds, is_int)
    return None


def _multiblock_predict_batch_file(batch_path, h, cfg, exog_future=None):
    """Load a multi-block batch and union-combine its h-step forecast."""
    if exog_future is not None:
        raise ValueError(
            "exogenous regressors are not wired on the grid MULTI-block path "
            "(see AutoFFSUniverse._check_exog_supported). The parameter exists "
            "only so the head can dispatch single- and multi-block workers "
            "through one call site; reaching here means that gate was bypassed.")
    srs_ids, active_arr, last_ds = _load_batch_meta(batch_path)
    blocks, _state, weights, _cap = _load_multiblock_batch(batch_path, cfg)
    loc, sd, _ = _multiblock_forecast(blocks, weights, h)
    # match the single-block worker's (q, h) sd layout (universe indexes sd[j,k]).
    return (np.asarray(loc), np.asarray(sd).T, srs_ids, np.asarray(active_arr), last_ds)


def _multiblock_update_batch_file(batch_path, new_arr, new_last_ds_arr, is_int, cfg,
                                  exog_array=None):
    """Load a multi-block batch, advance blocks (`fwd_filter`) + the union carry
    (one-step via `forecast(1)`) over ``new_arr`` (T_new, q_cap), save back.
    Idempotent resume guard as the single-block worker."""
    if exog_array is not None:
        raise ValueError(
            "exogenous regressors are not wired on the grid MULTI-block path "
            "(see AutoFFSUniverse._check_exog_supported). The parameter exists "
            "only so the head can dispatch single- and multi-block workers "
            "through one call site; reaching here means that gate was bypassed.")
    srs_ids, active_arr, last_ds = _load_batch_meta(batch_path)
    active = np.asarray(active_arr, dtype=bool)
    cur = np.array([int(last_ds[s]) if is_int else pd.Timestamp(last_ds[s]).value
                    for s in srs_ids])
    tgt = np.asarray(new_last_ds_arr)
    if len(cur) == len(tgt) and np.all(cur[active] >= tgt[active]):
        return None
    blocks, union_state, weights, capacity = _load_multiblock_batch(batch_path, cfg)
    union_pdr, union_mdr = cfg[1], cfg[2]
    learn_dma = bool(cfg[4]) if len(cfg) > 4 else False
    dma_prior = cfg[5] if len(cfg) > 5 else None
    # advance each block AND step the union on the per-worker one-step trace the
    # blocks just produced (identical to the scan the fit path drives the union
    # over). Shared with AutoFFS.update -- see _multiblock_advance.
    state, weights = _multiblock_advance(
        blocks, union_state, new_arr, union_pdr, union_mdr,
        learn_dma=learn_dma, dma_prior=dma_prior)
    _save_multiblock_batch_file(blocks, state, weights, batch_path, srs_ids,
                                active, new_last_ds_arr, is_int)
    return None


# =====================================================================
# CV worker: single filter pass with per-step trajectory emission
# =====================================================================


class _CVTrajectory(NamedTuple):
    """Per-batch h-step forecasts at the requested cutoffs.

    Only ``n_cutoffs`` slices are stored, not the full ``T``-length
    trajectory. The forecast emission inside the scan is gated by
    ``lax.cond``, so non-cutoff timesteps incur no forecast compute.

    Shapes:
      * f_h, q_h     : (n_cutoffs, nm, n_series, h)
      * nu, weights  : (n_cutoffs, nm, n_series)
      * cutoff_t_idx : (n_cutoffs,) — the scan-time indices ``t`` at
        which each forecast was emitted (0-indexed into fit_array).
    """

    srs_ids: tuple
    f_h: np.ndarray
    q_h: np.ndarray
    nu: np.ndarray
    weights: np.ndarray
    cutoff_t_idx: np.ndarray
    # Ungated per-step DMA trace over the full T-length filter pass (including
    # the warmup region), for exact post-hoc Method-1 reproduction. None unless
    # captured. Shapes: (T, nm, n_series). weight_full is the combined DMA
    # weight; f1_full / q1_full are the one-step (h=1) predictive mean and
    # variance that drove each DMA update.
    weight_full: Optional[np.ndarray] = None
    f1_full: Optional[np.ndarray] = None
    q1_full: Optional[np.ndarray] = None
    # The batch's model->class indicator (nm, n_classes) boolean, for the
    # union DMA across blocks. None if not captured.
    model_indicator: Optional[np.ndarray] = None


def _run_cv_batch(
    srs_ids,
    fit_array,
    cutoff_t_idx,
    periodicity,
    n_seas_comps,
    h,
    dma_pdr,
    dma_mdr,
    warmup_steps: int = 0,
    include_ar: bool = False,
    adaptive: bool = False,
    tau_values=None,
    var_disc_values=None,
    capture_trace: bool = False,
    monitor_tau=MONITOR_TAU,
    universe_builder=None,
    exog_array=None,
    _prebuilt=None,
):
    """Run the filter once and emit h-step forecasts only at the cutoffs.

    Parameters
    ----------
    srs_ids : tuple of unique_ids in batch order.
    fit_array : np.ndarray, shape ``(T, n_series)``.
    cutoff_t_idx : np.ndarray of int
        Sorted ascending. Each entry ``t`` requests a forecast emission
        immediately after the filter has consumed observation ``t``.
        For window ``i`` with ``train_end_i = T - h - i*step_size``,
        the corresponding cutoff index is ``train_end_i - 1``.
    periodicity, n_seas_comps, h, dma_pdr, dma_mdr : pass-through.

    Returns
    -------
    _CVTrajectory containing one slice per cutoff (in ascending t order).

    ``_prebuilt``, when given, is a ``(multi, dma, model_indicator)`` triple to
    run the CV over directly, bypassing :func:`_build_multi_and_dma`. Used by
    ``StaticBlock.from_multi`` (user-assembled universe); the CV body below is
    identical either way.
    """
    if _prebuilt is not None:
        multi, dma, model_indicator = _prebuilt
    else:
        multi, dma, model_indicator = _build_multi_and_dma(
            fit_array,
            periodicity,
            n_seas_comps,
            h,
            dma_pdr,
            dma_mdr,
            warmup_steps,
            include_ar=include_ar,
            adaptive=adaptive,
            tau_values=tau_values,
            var_disc_values=var_disc_values,
            monitor_tau=monitor_tau,
            universe_builder=universe_builder,
        )

    multi_params = _multi_params_tuple(multi)
    dma_step_fn = dma.prepared_step()
    has_reg = multi.n_regressors > 0

    # Tail and data must AGREE -- the same contract GridBlock._rolling_designs
    # and AutoFFSUniverse._check_exog_supported enforce. Both halves of the
    # mismatch are silent if unchecked, which is why each is an error.
    if exog_array is not None and not has_reg:
        raise ValueError(
            "exog was supplied but this universe has no regression tail "
            f"(n_regressors=0): the design would go nowhere.")
    # An exogenous universe is one whose tail is NOT autoregressive; the flag is
    # derived per-universe in multi_model_dlm from the components themselves.
    is_exog = has_reg and bool(getattr(multi, "exog_regressors", False))
    if exog_array is not None and is_exog is False and has_reg:
        raise ValueError(
            "exog was supplied but this universe's tail is AUTOREGRESSIVE: an AR "
            "tail derives its regressors from the y stream, so a supplied design "
            "would be ignored. Drop exog, or build the universe with Regressors "
            "instead of AR.")
    if is_exog and exog_array is None:
        raise ValueError(
            f"this universe has an EXOGENOUS regression tail "
            f"(n_regressors={multi.n_regressors}) but no `exog` was supplied, so "
            "the tail would filter against a zero F row at every step and "
            "contribute nothing while occupying state. Pass exog=... to "
            "cross_validation.")
    if has_reg and not is_exog and not include_ar:
        raise ValueError(
            f"this universe has an AR tail (n_regressors={multi.n_regressors}) but "
            "include_ar=False, so the tail would contribute nothing. Set "
            "include_ar=True.")
    n_reg = multi.n_regressors if has_reg else 0
    reg_offset_ = (multi.k - n_reg) if has_reg else 0   # tail is contiguous, last
    exog_fut = None
    if is_exog:
        # Future design at every step: rows t+1 .. t+h, per (model, series).
        # The exogenous analogue of the AR path's seed_yts, and the reason exog
        # needs a per-cutoff FH -- unlike AR, the horizon rows are KNOWN, so the
        # forecast is the ordinary flat predictive against a patched FH rather
        # than an iterated one.
        xa = np.asarray(exog_array, dtype=np.float64)        # (T, q, n_reg)
        Tt, qq, rr = xa.shape
        pad = np.zeros((h, qq, rr), dtype=np.float64)
        ext = np.concatenate([xa, pad], axis=0)              # (T+h, q, n_reg)
        fut = np.stack([ext[t + 1: t + 1 + h] for t in range(Tt)])   # (T,h,q,r)
        fut = np.transpose(fut, (0, 2, 1, 3))                        # (T,q,h,r)
        exog_fut = jnp.asarray(
            np.repeat(fut[:, None], multi.nm, axis=1).reshape(Tt, multi.nm * qq, h, rr))

    # Precompute DH (FH, GH). The form depends on the chosen kernel:
    #   nested (_CV_USE_NESTED): compact, model-only — FH (nm, h, k),
    #       GH (nm, h, k, k) — shared across series by the inner vmap. Mirrors
    #       multi_model_dlm.forecast.
    #   flat: replicated across q — FH (p, h, k), GH (p, h, k, k).
    # The AR/TVAR path always needs the flat form (the iterated forecast
    # propagates per-(model, series) state), so force it when has_reg.
    if _CV_USE_NESTED and not has_reg:
        DH = {
            "FH": multi.F[:, jnp.newaxis, :] * jnp.ones((1, h, 1)),  # (nm, h, k)
            "GH": multi.GH[:, 0, :h, ...],                          # (nm, h, k, k)
        }
        _predictive_fn = _hstep_predictive
    else:
        Ft = multi.F.reshape(multi.nm, 1, multi.k)[jnp.newaxis, ...] * jnp.ones(
            (h, 1, multi.q, 1)
        )
        DH = {
            "FH": jnp.moveaxis(Ft, 0, 2).reshape(multi.p, h, multi.k),
            "GH": (multi.GH[:, :, :h, ...] * jnp.ones((1, multi.q, 1, 1, 1))).reshape(
                multi.p, h, multi.k, multi.k
            ),
        }
        _predictive_fn = _hstep_predictive_flat

    # AR/TVAR operands: per-(model, series) structural F, the filter lags, and
    # the cutoff seed lags. reg_mask gates the AR iteration to the TVAR models.
    if has_reg and not is_exog:
        Fp = (
            multi.F.reshape(multi.nm, 1, multi.k) * jnp.ones((1, multi.q, 1))
        ).reshape(multi.p, multi.k)
        reg_mask = multi.reg_mask
        lag_yts = multi.format_lag_yts(fit_array)        # (T, p, n_reg) filter lags
        seed_yts = multi.format_seed_lag_yts(fit_array)  # (T, p, n_reg) cutoff seeds
        disc_factor_ar = (1 - multi.disc_rates * multi.disc_rates_damped) / (
            multi.disc_rates * multi.disc_rates_damped
        )

        def _predict_ar(dlm_state, weights, seed_t):
            return _hstep_predictive_ar(
                dlm_state, weights, multi.nm, multi.q, h, disc_factor_ar,
                multi.variance_disc, multi.variance_power, multi.mult_comps, DH,
                Fp, n_reg, reg_mask, seed_t,
            )

    T = fit_array.shape[0]
    cutoff_t_idx = np.sort(np.asarray(cutoff_t_idx, dtype=np.int32))

    # Fast path for single-origin CV (n_windows == 1). The lone cutoff is the
    # last training step, so instead of stacking a (T, nm, q, h) forecast
    # trajectory and gathering one row, filter up to the cutoff and forecast
    # once. Same per-step filter (_scan_filter uses the identical step) and
    # same forecast fn (_hstep_predictive), so the result matches the emit
    # path. Disabled when capture_trace is set (the MComp pipeline needs the
    # full per-step trace, which only the emit-scan produces).
    if not capture_trace and len(cutoff_t_idx) == 1:
        cut = int(cutoff_t_idx[0])
        final_dlm, _final_alloc, all_weights = _scan_filter(
            multi, dma, fit_array[: cut + 1], warmup_steps,
            exog_array=(None if exog_array is None
                        else np.asarray(exog_array)[: cut + 1]),
        )
        # Forecast discount: adaptive uses the origin's monitor S (so calm models
        # don't forecast with their d_min floor); static is byte-identical.
        disc_factor = _forecast_disc_factor(multi, final_dlm)
        if is_exog:
            # Known horizon design -> ordinary flat predictive against an FH
            # whose regressor columns carry this origin's future rows.
            DH_cut = {"FH": DH["FH"].at[:, :, reg_offset_:reg_offset_ + n_reg].set(
                          exog_fut[cut]),
                      "GH": DH["GH"]}
            f_h_t, q_h_t, nu_t, weights_t = _hstep_predictive_flat(
                final_dlm, all_weights[-1], multi.nm, multi.q, h, disc_factor,
                multi.variance_disc, multi.variance_power, multi.mult_comps, DH_cut,
            )
        elif has_reg:
            # Stationarise the AR coefficients before forecasting: reflect any
            # non-stationary roots inside the unit circle so the iterated variance
            # recursion cannot diverge (the near-unit-root M4Q tail). Done in
            # numpy over only the TVAR rows (reg_mask) — a few thousand 4x4 root
            # solves, no block-diagonal — rather than jnp.eig per row inside the
            # forecast vmap (which OOMs across all p rows on the cluster jaxlib).
            # G_ar = I, so flipping the filtered AR block propagates unchanged.
            if AR_STATIONARITY_EPS is not None:
                m_np = np.asarray(final_dlm["m"]).copy()      # (p, k)
                rm = np.asarray(reg_mask).astype(bool)         # (p,)
                reg0 = multi.k - n_reg
                sub = m_np[rm]
                sub[:, reg0:] = _flip_ar_np(sub[:, reg0:], AR_STATIONARITY_EPS)
                m_np[rm] = sub
                final_dlm = {**final_dlm, "m": jnp.asarray(m_np)}
            # Seed lags at the cutoff: [y_cut, y_{cut-1}, ...] per (model, series).
            seed_cut = seed_yts[cut]
            f_h_t, q_h_t, nu_t, weights_t = _predict_ar(
                final_dlm, all_weights[-1], seed_cut
            )
        else:
            f_h_t, q_h_t, nu_t, weights_t = _predictive_fn(
                final_dlm, all_weights[-1], multi.nm, multi.q, h, disc_factor,
                multi.variance_disc, multi.variance_power, multi.mult_comps, DH,
            )
        return _CVTrajectory(
            srs_ids=tuple(srs_ids),
            f_h=np.asarray(f_h_t)[None],   # (1, nm, q, h)
            q_h=np.asarray(q_h_t)[None],
            nu=np.asarray(nu_t)[None],     # (1, nm, q)
            weights=np.asarray(weights_t)[None],
            cutoff_t_idx=cutoff_t_idx,
            model_indicator=np.asarray(model_indicator.values),
        )

    # Build emission flag: True at every t in cutoff_t_idx
    emit_flags = np.zeros(T, dtype=bool)
    emit_flags[cutoff_t_idx] = True

    emit_flags_jax = device_put(jnp.asarray(emit_flags), devices.allocation_compute)

    # Warmup flags: 1.0 for the first `warmup_steps` time-steps, 0.0 thereafter.
    warmup_flags = jnp.where(
        jnp.arange(T) < warmup_steps,
        jnp.array(1.0),
        jnp.array(0.0),
    )
    warmup_flags = device_put(warmup_flags, devices.allocation_compute)

    scan_step = _build_emit_scan_step(
        multi, multi_params, dma_step_fn, DH, h, capture_trace=capture_trace,
        has_regressors=has_reg,
        reg_mask=(multi.reg_mask if has_reg and not is_exog else None),
        n_reg=n_reg,
        Fp=(Fp if has_reg and not is_exog else None),
        exog_mode=is_exog,
        reg_offset=(reg_offset_ if has_reg else 0),
    )

    yts = multi.format_yts(fit_array)
    init_carry = (multi.dlm_state, dma.state)
    scan_in = (
        yts,
        device_put(fit_array, devices.allocation_compute),
        emit_flags_jax,
        warmup_flags,
    )
    if is_exog:
        # filter rows + this origin's future rows, the exogenous counterpart of
        # (lag_yts, seed_yts)
        scan_in = scan_in + (multi.format_exog_yts(exog_array), exog_fut)
    elif has_reg:
        scan_in = scan_in + (lag_yts, seed_yts)
    if capture_trace:
        (_, _), (
            f_h_traj,
            q_h_traj,
            nu_traj,
            weights_traj,
            emit_traj,
            weight_full_traj,
            f1_traj,
            q1_traj,
        ) = scan(scan_step, init_carry, scan_in)
    else:
        (_, _), (
            f_h_traj,
            q_h_traj,
            nu_traj,
            weights_traj,
            emit_traj,
        ) = scan(scan_step, init_carry, scan_in)
        weight_full_traj = f1_traj = q1_traj = None
    # Shapes: f_h_traj (T, nm, q, h); only rows where emit_traj is True
    # carry meaningful values, the rest are zeros from the cond branch.

    # Gather emitted rows. Use the python-side cutoff_t_idx directly so
    # the output order is deterministic and ascending in t.
    f_h_out = np.asarray(f_h_traj)[cutoff_t_idx]
    q_h_out = np.asarray(q_h_traj)[cutoff_t_idx]
    nu_out = np.asarray(nu_traj)[cutoff_t_idx]
    weights_out = np.asarray(weights_traj)[cutoff_t_idx]

    return _CVTrajectory(
        srs_ids=tuple(srs_ids),
        f_h=f_h_out,
        q_h=q_h_out,
        nu=nu_out,
        weights=weights_out,
        cutoff_t_idx=cutoff_t_idx,
        # Per-step DMA trace, populated only when capture_trace=True (MComp).
        weight_full=np.asarray(weight_full_traj) if capture_trace else None,  # (T, nm, n_series), float64
        f1_full=np.asarray(f1_traj) if capture_trace else None,
        q1_full=np.asarray(q1_traj) if capture_trace else None,
        model_indicator=np.asarray(model_indicator.values),
    )


# =====================================================================
# Union DMA over blocks
# =====================================================================
# The orchestrator combines the universe's blocks with ONE existing 2-level
# Allocator over the UNION of their models (within-class pdm + between-class
# mdm) — not a new "over-blocks" level. Because the blocks run their filters
# independently, the union DMA is a post-pass over each block's captured
# one-step trace: the Allocator driven over (f1, q1, obs) from t=0 reproduces
# the in-scan weights to ~1e-15 (verified on M1M; see FFS
# legacy/horizon_weight/M1M_weight_comparison), so a single-block union
# reproduces the legacy in-scan combine.


def _block_diag_mi(mis):
    """Block-diagonal union of per-block model->class indicators into one
    ``(M_total, C_total)`` boolean indicator (each block's classes are its own,
    so the classes concatenate; models compete flat at the between-class level)."""
    Ms = [int(m.shape[0]) for m in mis]
    Cs = [int(m.shape[1]) for m in mis]
    out = np.zeros((sum(Ms), sum(Cs)), dtype=bool)
    r = c = 0
    for m in mis:
        mr, mc = int(m.shape[0]), int(m.shape[1])
        out[r:r + mr, c:c + mc] = np.asarray(m).astype(bool)
        r += mr
        c += mc
    return out


def _union_dma_weights(F1, Q1, obs, union_mi, dma_pdr, dma_mdr, dma_c=1e-3,
                       learn_dma=False, dma_prior=None, warmup=0):
    """Post-pass union DMA weight trajectory. Drives one Allocator over the
    concatenated per-model one-step trace, per series, from t=0.

    ``F1``/``Q1`` ``(T, M, n_series)`` one-step mean/variance; ``obs``
    ``(T, n_series)``; ``union_mi`` ``(M, C)`` boolean. Returns the combined
    (pset x mset) weights ``(T, M, n_series)`` summing to 1 over the M models
    each step. Same ``allocator_step`` / ``PowerLawUpdate`` the in-scan DMA uses.

    ``learn_dma``: the union forgetting rates ``(pdr, mdr)`` are LEARNED online per
    series by RTRL/MAP (SGDDMA) instead of fixed — each combiner self-tunes on its
    own combined objective (so the union learns its OWN rate, not a block's). Starts
    from ``(dma_pdr, dma_mdr)`` and reproduces the fixed replay when frozen.
    """
    F1 = jnp.asarray(F1)
    Q1 = jnp.asarray(Q1)
    obs = jnp.asarray(obs)
    mi = jnp.asarray(union_mi)
    M, C = int(mi.shape[0]), int(mi.shape[1])

    if learn_dma:
        from .ffs.discount_grid import _sgd_dma_traj, DMA_PRIOR
        dp = dma_prior if dma_prior is not None else DMA_PRIOR

        def per_series(F, Q, y):  # F, Q: (T, M); y: (T,)
            Wt, _disc = _sgd_dma_traj(F, Q, y, mi, c=dma_c, dma_prior=dp,
                                      warmup=warmup, pdr0=dma_pdr, mdr0=dma_mdr)
            return Wt

        W = vmap(per_series, in_axes=(2, 2, 1), out_axes=2)(F1, Q1, obs)
        return np.asarray(W)

    upd = Partial(PowerLawUpdate, dma_pdr=dma_pdr, dma_mdr=dma_mdr, c=dma_c)

    def per_series(F, Q, y):  # F, Q: (T, M); y: (T,)
        alloc0 = init_alloc_state(M, 1, C, mi, 1)

        def step(alloc, xs):
            f, q, yy = xs
            fc = ForecastBundle(f[:, None, None], q[:, None, None])
            alloc_n, w = allocator_step(
                alloc, fc, yy[None], LogScore, IdentityAggregator, upd, mi
            )
            return alloc_n, w[:, 0, 0]

        _, Wt = scan(step, alloc0, (F, Q, y))
        return Wt  # (T, M)

    W = vmap(per_series, in_axes=(2, 2, 1), out_axes=2)(F1, Q1, obs)  # (T, M, n_series)
    return np.asarray(W)


def _union_allocator(block_mis, n_series, dma_pdr, dma_mdr, dma_c=1e-3):
    """Build a live ``Allocator`` over the block-diagonal union of per-block
    ``model->class`` indicators.

    Mirrors :func:`_dma_for_multi` (same ``LogScore`` / ``PowerLawUpdate`` /
    ``IdentityAggregator`` config) but from a RAW union indicator — the union has
    no ``model_desc``, its classes are just the blocks' classes concatenated
    (:func:`_block_diag_mi`). Unlike :func:`_union_dma_weights` (which replays the
    allocator from t=0), this holds a persistable ``AllocatorState`` carry
    (``dma.state``) advanced one origin at a time by ``dma.prepared_step()`` — the
    streaming analogue the sequential-update persisted universe needs. Stepping it
    over a ``(f1, q1, obs)`` sequence reproduces the replay to float precision
    (same ``allocator_step``); see ``test_union_stream``.

    Returns ``(dma, union_mi)`` where ``union_mi`` ``(M_total, C_total)`` bool is
    the concatenation order the caller must feed one-step / h-step predictives in.
    """
    union_mi = _block_diag_mi(block_mis)
    M, C = int(union_mi.shape[0]), int(union_mi.shape[1])
    dma = Allocator(
        scoring_rule=LogScore,
        update_rule=Partial(PowerLawUpdate, dma_pdr=dma_pdr, dma_mdr=dma_mdr,
                            c=dma_c),
        model_indicator=union_mi,
        device=devices.allocation_compute,
    )
    dma.init(n_models=M, n_series=n_series, n_classes=C, model_indicator=union_mi)
    return dma, union_mi


def _union_predictive_combine(W, f_h, q_h, nu, level, sd_method):
    """Combine per-model Student-t h-step forecasts under union weights ``W``
    ``(M, n_series)`` through the Predictive ``vincent`` seam. ``f_h``/``q_h``/
    ``nu`` ``(M, n_series, h)``. Returns ``(loc (n_series, h), sd (h, n_series),
    bounds|None)``. Shared by the CV post-pass (:func:`_union_combine_cv`) and the
    streaming carry so both emit byte-identical forecasts from the same inputs."""
    from DLMAX.ffs.predictive import StudentTPredictive
    from DLMAX.ffs.predictive import combine as _predictive_combine
    comp = StudentTPredictive(loc=f_h, scale2=q_h, nu=nu)
    cp = _predictive_combine([comp], [W], form="vincent")
    loc = cp.mean                                                # (n_series, h)
    if sd_method == "moment":
        sd = _t_predictive_sd(FFSPredictive(
            loc=None, sd=None, f_h=f_h, q_h=q_h, nu=nu, weights=W))
    else:
        sd = cp.sd.T                                             # (h, n_series)
    bounds = None
    if level:
        bounds = {int(L): tuple(np.asarray(x).T for x in cp.interval(L))
                  for L in level}
    return loc, sd, bounds


def _multiblock_fit(blocks, arr, union_pdr, union_mdr, dma_c=1e-3,
                    wing_centres=None, component_priors=None, error_nu0=None,
                    learn_dma=False, dma_prior=None):
    """Fit each block over ``arr`` ``(T, q)`` and build the union DMA carry.

    Each block's `scan_filter(return_trace=True)` advances its production carry
    AND yields its per-worker one-step trace; a live `_union_allocator` is then
    stepped over the blocks' concatenated trace (a single `scan`) to the union
    `AllocatorState` at ``arr``'s end. This reproduces the CV replay
    (`_union_combine_cv`) over the same window incl. warmup. Returns
    ``(union_state, weights (M, q), union_mi)``; ``weights`` are the post-fit
    combining weights (also recoverable from ``union_state`` on resume).

    ``wing_centres``/``component_priors`` (per-block lists, or None) warm-start
    the blocks from siblings for ``add_series`` — each block is a distinct grid,
    so its prior is its own."""
    a = np.asarray(arr, dtype=float)
    T, q = a.shape
    Fs, Qs = [], []
    for i, b in enumerate(blocks):
        wc = wing_centres[i] if wing_centres is not None else None
        cp = component_priors[i] if component_priors is not None else None
        _b, (F, Q) = b.scan_filter(a, return_trace=True, wing_centre=wc,
                                   component_priors=cp, error_nu0=error_nu0)  # (T,q,M_b)
        Fs.append(F)
        Qs.append(Q)
    Ft = jnp.asarray(np.transpose(np.concatenate(Fs, axis=2), (0, 2, 1)))  # (T, M, q)
    Qt = jnp.asarray(np.transpose(np.concatenate(Qs, axis=2), (0, 2, 1)))
    yt = jnp.asarray(a)                                        # (T, q)
    dma, union_mi = _union_allocator([b.model_indicator for b in blocks], q,
                                     union_pdr, union_mdr, dma_c)
    if learn_dma:      # SGDDMA: the union learns its OWN forgetting rate per series
        from .ffs.discount_grid import (sgd_union_carry0, sgd_union_step_q,
                                        DMA_PRIOR)
        dp = dma_prior if dma_prior is not None else DMA_PRIOR
        M = int(union_mi.shape[0])
        carry0 = sgd_union_carry0(M, union_mi, q, union_pdr, union_mdr)

        # Gate the rate learning over the WARMUP window, exactly as the
        # block-internal SGDDMA does (``discount_grid._sgd_dma_traj`` builds the
        # same ``arange(T) < warmup`` flag). Without it the union learns its
        # forgetting rates from diffuse-prior predictives across the warmup — the
        # one structural difference between combining N blocks at the union and
        # combining one block's families internally, and worth ~3e-4 median
        # relative on M5 forecasts, decaying over the following origins.
        # The window is the blocks' own: they are what the union combines, and a
        # union that started learning before its inputs had settled would be
        # fitting noise. Disagreeing blocks take the MAX (learn only once every
        # input has settled).
        warmup = max([int(getattr(b, "warmup", 0) or 0) for b in blocks],
                     default=0)
        warm_t = (jnp.arange(T) < warmup).astype(yt.dtype)

        def ustep(carry, xs):
            f, qv, y, w = xs
            return sgd_union_step_q(carry, f, qv, y, w, mi=union_mi, c=dma_c,
                                    dma_prior=dp)

        state, ws = scan(ustep, carry0, (Ft, Qt, yt, warm_t))  # ws (T, M, q)
        return state, np.asarray(ws[-1]), union_mi            # weights (M, q)

    step = dma.prepared_step()

    def ustep(state, xs):
        f, qv, y = xs
        return step(state, ForecastBundle(f[..., None], qv[..., None]), y)

    state, ws = scan(ustep, dma.state, (Ft, Qt, yt))          # ws (T, M, q, 1)
    return state, np.asarray(ws[-1, :, :, 0]), union_mi       # weights (M, q)


def _future_ds_at(freq, last_ds, h):
    """The ``h`` dates strictly after ``last_ds`` at ``freq``, polymorphic over
    int (M1/M3-style) and datetime frequencies.

    Module-level so the disk-backed universe and the in-memory ``AutoFFS``
    forward path share ONE definition of "the dates a forecast lands on" --
    two copies would be free to drift, and a calendar disagreement between the
    two faces would be invisible until someone compared their outputs.
    """
    if isinstance(freq, (int, np.integer)):
        step = int(freq)
        return np.array([int(last_ds) + step * (i + 1) for i in range(h)])
    offset = pd.tseries.frequencies.to_offset(freq)
    base = pd.Timestamp(last_ds)
    return np.array([(base + (i + 1) * offset).to_datetime64() for i in range(h)])


def _multiblock_advance(blocks, union_state, arr, union_pdr, union_mdr,
                        dma_c=1e-3, learn_dma=False, dma_prior=None):
    """Advance blocks AND the union carry over ``arr`` ``(T_new, q)``.

    The per-step counterpart of :func:`_multiblock_fit`: each block takes the
    row through ``fwd_filter`` while the union allocator is stepped on the
    per-worker one-step trace the blocks just produced. Returns
    ``(union_state, weights (M, q))``.

    Extracted from ``_multiblock_update_batch_file`` so the file-backed update
    and the in-memory ``AutoFFS.update`` run the SAME advance -- the guarantee
    that the two faces stay bit-identical is only worth anything if they are
    literally the same code.
    """
    a = np.asarray(arr, dtype=float)
    dma, union_mi = _union_allocator([b.model_indicator for b in blocks],
                                     a.shape[1], union_pdr, union_mdr, dma_c)
    if learn_dma:
        from .ffs.discount_grid import sgd_union_step_q, DMA_PRIOR
        dp = dma_prior if dma_prior is not None else DMA_PRIOR
    else:
        step = dma.prepared_step()
    state, weights = union_state, None
    for t in range(a.shape[0]):
        f1s, q1s = [], []
        for b in blocks:
            _b, (F, Q) = b.fwd_filter(a[t], return_trace=True)      # F (q, M_b)
            f1s.append(F)
            q1s.append(Q)
        f1 = jnp.asarray(np.concatenate(f1s, axis=1).T)             # (M, q)
        q1 = jnp.asarray(np.concatenate(q1s, axis=1).T)
        if learn_dma:
            state, w = sgd_union_step_q(state, f1, q1, jnp.asarray(a[t]), 0.0,
                                        mi=union_mi, c=dma_c, dma_prior=dp)
            weights = np.asarray(w)
        else:
            state, w = step(state, ForecastBundle(f1[..., None], q1[..., None]),
                            jnp.asarray(a[t]))
            weights = np.asarray(w[:, :, 0])
    return state, weights


def _multiblock_forecast(blocks, weights, h, level=None, sd_method="quantile"):
    """Combine the blocks' h-step per-worker predictives under the union
    ``weights`` ``(M, q)`` through :func:`_union_predictive_combine`. Returns
    ``(loc (q, h), sd (h, q), bounds|None)`` — the same layout the grid batch
    predict worker emits."""
    LOC, QH, NU = [], [], []
    for b in blocks:
        _loc, _sd, comp = b.forecast(h)
        LOC.append(np.asarray(comp["LOCc"]))                  # (q, M_b, h)
        QH.append(np.asarray(comp["QHc"]))
        NU.append(np.asarray(comp["NUc"]))                    # (q, M_b)
    f_h = np.transpose(np.concatenate(LOC, axis=1), (1, 0, 2))   # (M, q, h)
    q_h = np.transpose(np.concatenate(QH, axis=1), (1, 0, 2))
    nu = np.transpose(np.concatenate(NU, axis=1), (1, 0))        # (M, q)
    return _union_predictive_combine(np.asarray(weights), f_h, q_h, nu,
                                     level, sd_method)


def _union_combine_cv(trajectories, obs, dma_pdr, dma_mdr, level, sd_method,
                      dma_c=1e-3, learn_dma=False, dma_prior=None, warmup=0):
    """Combine one-or-more blocks' CV trajectories via a single union DMA.

    Each trajectory must carry the ``capture_trace`` one-step arrays
    (``f1_full``/``q1_full`` ``(T, nm, n_series)``) and ``model_indicator``.
    ``obs`` is the batch's ``(T, n_series)`` observations. Concatenates the
    blocks' per-model predictives along the model axis, runs the union DMA over
    the concatenated one-step trace, and combines at each cutoff. For a single
    block this reproduces the legacy in-scan combine to ~float precision.

    Returns a list (one entry per cutoff, ascending t) of
    ``(loc (n_series, h), sd (h, n_series), bounds)`` where ``bounds`` is the
    ``_t_quantile_average`` dict or ``None``.
    """
    cutoffs = np.asarray(trajectories[0].cutoff_t_idx)
    # Blocks may trace over different horizons (a static block filters the full
    # T=L; the grid trains on T=L-h). Align the one-step trace + obs to the
    # shortest — the cutoffs are all < that length (they are training origins),
    # and the DMA weights at a cutoff are causal, so trimming the tail past the
    # last cutoff leaves them unchanged.
    T_min = min(int(np.asarray(t.f1_full).shape[0]) for t in trajectories)
    if cutoffs.max() >= T_min:
        raise ValueError(
            f"cutoff {int(cutoffs.max())} is outside the aligned trace length "
            f"{T_min}; cutoffs must be training origins strictly inside every "
            f"block's one-step trace."
        )
    F1 = np.concatenate([np.asarray(t.f1_full)[:T_min] for t in trajectories], axis=1)
    Q1 = np.concatenate([np.asarray(t.q1_full)[:T_min] for t in trajectories], axis=1)
    union_mi = _block_diag_mi([t.model_indicator for t in trajectories])
    W = _union_dma_weights(F1, Q1, np.asarray(obs)[:T_min], union_mi, dma_pdr, dma_mdr,
                           dma_c, learn_dma=learn_dma, dma_prior=dma_prior, warmup=warmup)
    Wc = W[cutoffs]  # (n_cut, M_total, n_series)

    f_h = np.concatenate([np.asarray(t.f_h) for t in trajectories], axis=1)
    q_h = np.concatenate([np.asarray(t.q_h) for t in trajectories], axis=1)
    nu = np.concatenate([np.asarray(t.nu) for t in trajectories], axis=1)

    # Combine through the Predictive seam. The blocks' per-model forecasts are
    # Student-t; combine(form="vincent") reproduces _t_vincent_sd /
    # _t_quantile_average, and is the exact seam a lognormal / DGLM block plugs
    # into (its family-specific Predictive is just another component). The
    # "moment" SD (law-of-total-variance) stays on _t_predictive_sd.
    out = []
    for gi in range(len(cutoffs)):
        out.append(_union_predictive_combine(
            Wc[gi], f_h[gi], q_h[gi], nu[gi], level, sd_method))
    return out


# =====================================================================
# Public class
# =====================================================================


class StaticFFS:
    """Forward Filtering Sequential DMA forecaster.

    Two complementary APIs are exposed:

    Persistent state (recommended for sequential / streaming use):

    >>> model = AutoFFS(season_length=12).fit(df_history)
    >>> fc1 = model.predict(h=12, level=[80, 95])
    >>> # later, when new data arrives:
    >>> model.update(df_new_observations)
    >>> fc2 = model.predict(h=12, level=[80, 95])

    Each datapoint is ingested by the filter exactly once across this
    lifecycle.

    Stateless one-shot:

    >>> fc = AutoFFS(season_length=12).forecast(df, h=12, level=[80, 95])

    Diagnostic backtest:

    >>> cv = AutoFFS(season_length=12).cross_validation(
    ...     df, h=12, n_windows=5, level=[95]
    ... )

    ``cross_validation`` runs a single filter pass over ``df`` with
    per-step h-step forecasts emitted, then slices at the requested
    cutoffs. Cost is one full filter run regardless of ``n_windows``.

    Parameters
    ----------
    season_length, n_seas_comps, dma_pdr, dma_mdr, max_batch_size,
    dask_client, alias :
        See attributes / earlier API.

    Notes
    -----
    The same ``max_batch_size`` and Dask dispatch architecture applies to
    both APIs. ``update`` and ``predict`` operate on the per-batch states
    saved by ``fit`` / previous ``update`` calls, dispatching one task
    per batch. Calling ``setup_workers(client, ...)`` once per session
    configures JAX on every worker.

    State produced by ``fit`` / ``update`` is held in ``self._batches``
    and is picklable: ``joblib.dump(model, path)`` / ``joblib.load`` work
    via JAX's numpy round-trip.
    """

    def __init__(
        self,
        season_length: Optional[int] = None,
        n_seas_comps: Optional[int] = None,
        dma_pdr: float = 0.90,
        dma_mdr: Optional[float] = None,
        max_batch_size: Optional[int] = None,
        dask_client=None,
        alias: str = "AutoFFS",
        sd_method: str = "quantile",
        include_ar: bool = False,
        adaptive: bool = False,
        tau_values=None,
        var_disc_values=None,
        monitor_tau=MONITOR_TAU,
        universe_builder=None,
    ):
        if not (0.0 < dma_pdr <= 1.0):
            raise ValueError("dma_pdr must lie in (0, 1].")
        if dma_mdr is not None and not (0.0 < dma_mdr <= 1.0):
            raise ValueError("dma_mdr must lie in (0, 1].")
        if season_length is not None and season_length < 2:
            raise ValueError("season_length must be >= 2 when set.")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer or None.")
        if sd_method not in ("quantile", "moment"):
            raise ValueError("sd_method must be 'quantile' or 'moment'.")

        self.season_length = season_length
        self.n_seas_comps = n_seas_comps
        self.dma_pdr = dma_pdr
        self.dma_mdr = dma_pdr if dma_mdr is None else dma_mdr
        self.max_batch_size = max_batch_size
        self.dask_client = dask_client
        self.alias = alias
        # Add the TVAR class (LocalLevel + AR(k-1)) as a 9th mset class.
        self.include_ar = include_ar
        # Error-monitoring (adaptive-discount) universe: replaces the static
        # LT-discount grid with a tau (sensitivity) grid; forgetting becomes
        # online + error-driven. ``tau_values`` (monitor SD-multiple grid) and
        # ``var_disc_values`` (static variance-discount grid) parameterise THIS
        # path only — they are IGNORED on the standard (non-adaptive) universe.
        # include_ar is also ignored here (no AR/TVAR class).
        self.adaptive = adaptive
        self.tau_values = tau_values
        self.var_disc_values = var_disc_values
        # Static-grid signed-error monitor (see _build_multi_and_dma) and a
        # user-supplied custom universe builder (FFS front end). Both default
        # off/None -> standard FFS universe, unchanged.
        self.monitor_tau = monitor_tau
        self.universe_builder = universe_builder
        # Combined-predictive SD scheme: "quantile" (Vincent, default) or
        # "moment" (law-of-total-variance). See _combined_predictive_sd.
        self.sd_method = sd_method

        # Persistent state (populated by fit / update)
        self._batches = None  # list[_BatchState]
        self._freq = None  # str
        self._series_index = None  # list of unique_ids in original order
        self._fit_h_template = None  # h used at fit time for model templates

    def __repr__(self):
        fit_str = (
            f"fitted on {sum(b.T_ingested for b in self._batches)} obs "
            f"across {len(self._batches)} batch(es)"
            if self._batches is not None
            else "unfitted"
        )
        return (
            f"{type(self).__name__}(season_length={self.season_length}, "
            f"max_batch_size={self.max_batch_size}, alias={self.alias!r}; "
            f"{fit_str})"
        )

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has been called (per-batch state is held)."""
        return self._batches is not None

    # ------------------------------------------------------------------
    # Worker setup
    # ------------------------------------------------------------------

    @staticmethod
    def setup_workers(client, compute: Optional[str] = None, device_id: int = 0):
        """Configure JAX on every Dask worker. Call once before fit/forecast.

        With the default ``compute=None``, each worker independently
        auto-detects its own device topology (GPU if visible, else CPU).
        Pass ``compute='cpu'`` to force all workers to CPU regardless of
        local GPU availability.
        """
        return client.run(configure_devices, compute, device_id)

    # ------------------------------------------------------------------
    # Shared input handling
    # ------------------------------------------------------------------

    def _prepare_input(self, df, freq):
        required = {"unique_id", "ds", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        # NaN in `y` is permitted: the filter treats a NaN observation as
        # a no-op (``ignore_obs = isnan(error)`` in dlm_uv_fwd_svd_step),
        # carrying the prior forward. This is the honest representation of
        # pre-launch / structurally-missing periods (e.g. M5 intermittent
        # series). Diffuse-prior elicitation uses nan-aware statistics; the
        # legacy OLS elicitation (warmup_steps=0) is NOT nan-safe, so pass
        # warmup_steps>0 when the input contains NaN. A series that is
        # *entirely* NaN cannot be fit and is rejected below.

        df = df.loc[:, ["unique_id", "ds", "y"]].copy()
        # Detect index type. Integer-typed `ds` is preserved as-is; everything
        # else is cast to datetime. `freq` is then either an int (integer-mode)
        # or a pandas BaseOffset (datetime-mode).
        ds_is_integer = pd.api.types.is_integer_dtype(df["ds"])
        if not ds_is_integer:
            df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        dup_mask = df.duplicated(subset=["unique_id", "ds"])
        if dup_mask.any():
            offending = df.loc[dup_mask, "unique_id"].unique()[:5]
            raise ValueError(
                f"Duplicate (unique_id, ds) rows in input. Examples: "
                f"{list(offending)}."
            )

        groups = df.groupby("unique_id", sort=False)
        per_series = groups.agg(
            length=("y", "size"),
            n_real=("y", "count"),
            last_ds=("ds", "max"),
        )
        all_nan = per_series.index[per_series["n_real"] == 0]
        if len(all_nan):
            raise ValueError(
                f"{len(all_nan)} series have no non-NaN observations and "
                f"cannot be fit. Examples: {list(all_nan[:5])}."
            )

        if freq is None:
            first_id = per_series.index[0]
            first_ds = groups.get_group(first_id)["ds"]
            if len(first_ds) < 3:
                raise ValueError(
                    "Cannot infer freq from fewer than 3 observations; "
                    "pass `freq=` explicitly."
                )
            if ds_is_integer:
                diffs = np.diff(first_ds.to_numpy())
                if not np.all(diffs == diffs[0]):
                    raise ValueError(
                        "Could not infer integer freq: ds is not evenly "
                        "spaced. Pass `freq=` explicitly as an int."
                    )
                freq = int(diffs[0])
            else:
                freq = pd.infer_freq(first_ds)
                if freq is None:
                    raise ValueError(
                        "Could not infer freq; pass `freq=` explicitly "
                        "(e.g. 'MS', 'D', 'W')."
                    )

        sid_ds = {sid: g["ds"].to_numpy() for sid, g in groups}
        sid_y = {sid: g["y"].to_numpy() for sid, g in groups}

        return per_series, sid_ds, sid_y, freq

    def _min_filter_length(self):
        return (
            int(np.ceil(2 * self.season_length))
            if self.season_length is not None
            else 10
        )

    @staticmethod
    def _normalise_level(level):
        if level is None:
            return None
        if isinstance(level, int):
            level = [level]
        for L in level:
            if not (0 < int(L) < 100):
                raise ValueError(
                    f"level entries must be integers in (0, 100); got {L}."
                )
        return [int(L) for L in level]

    def _prepare_exog(self, exog, sid_ds):
        """Long exogenous frame -> ``{sid: (T_sid, n_reg) float64}``.

        The cross-validation analogue of ``AutoFFSUniverse._materialise_exog``:
        that one CALLS a provider because a streaming universe meets its dates
        one origin at a time, whereas a backtest already has the whole design in
        hand, so it is passed as a frame and aligned here once.

        ``exog`` is long ``(unique_id, ds, <regressor columns...>)``; every
        remaining column is a regressor, in column order. It must cover exactly
        the ``ds`` each series carries in ``df``. A silently short or misaligned
        design would filter part of the panel against a zero ``F`` row, so any
        mismatch raises.
        """
        if exog is None:
            return None
        if not {"unique_id", "ds"}.issubset(exog.columns):
            raise ValueError("exog must be long with 'unique_id' and 'ds' columns "
                             f"plus one column per regressor; got {list(exog.columns)}")
        reg_cols = [c for c in exog.columns if c not in ("unique_id", "ds")]
        if not reg_cols:
            raise ValueError("exog has no regressor columns beyond unique_id/ds.")
        out = {}
        for sid, want in sid_ds.items():
            g = exog[exog["unique_id"] == sid]
            if g.empty:
                raise ValueError(f"exog has no rows for series {sid!r}.")
            g = g.set_index("ds").reindex(pd.Index(want))
            if g[reg_cols].isna().to_numpy().any():
                missing = int(g[reg_cols].isna().any(axis=1).sum())
                raise ValueError(
                    f"exog for series {sid!r} does not cover every ds in df: "
                    f"{missing} of {len(want)} rows missing or NaN. The design "
                    "must be complete over the series' own calendar.")
            out[sid] = g[reg_cols].to_numpy(dtype=np.float64)
        return out

    def _iter_batches(self, per_series, sid_y):
        """Yield (srs_ids, fit_array) chunked by max_batch_size within each
        length-group."""
        for length, lg_meta in per_series.groupby("length", sort=False):
            srs_ids_all = lg_meta.index.tolist()
            chunk = (
                len(srs_ids_all) if self.max_batch_size is None else self.max_batch_size
            )
            for k in range(0, len(srs_ids_all), chunk):
                srs_ids = srs_ids_all[k : k + chunk]
                arr = np.column_stack([sid_y[sid] for sid in srs_ids])
                yield srs_ids, arr

    # ------------------------------------------------------------------
    # Job dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, fn, args_list):
        """Dispatch a list of homogeneous jobs.

        Each entry of ``args_list`` is a tuple of positional args for
        ``fn``. Returns a list of results in the same order. With a Dask
        client, all jobs are submitted up front and gathered together.
        """
        if self.dask_client is None:
            return [fn(*args) for args in args_list]

        futures = [self.dask_client.submit(fn, *args, pure=False) for args in args_list]
        return self.dask_client.gather(futures)

    # ------------------------------------------------------------------
    # Persistent API: fit / predict / update
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        freq: Optional[str] = None,
        h_template: int = 18,
        warmup_steps: Optional[int] = None,
    ):
        """Ingest a long-format DataFrame and persist per-batch state.

        Parameters
        ----------
        df : DataFrame with columns (unique_id, ds, y).
        freq : pandas offset alias; inferred if None.
        h_template : int, default 18
            Horizon used to size internal model templates via
            ``assemble_models``. Does not constrain prediction horizons
            later — ``predict(h=h_new)`` works for any ``h_new``. Default
            18 covers most monthly use cases.

        Returns
        -------
        self
        """
        per_series, sid_ds, sid_y, freq = self._prepare_input(df, freq)

        self.warmup_steps = warmup_steps
        min_len = self._min_filter_length()
        _warn_under_seasonal(
            per_series.index[per_series["length"] < min_len],
            min_len, self.season_length)

        batches_input = list(self._iter_batches(per_series, sid_y))
        args_list = [
            (
                tuple(srs_ids),
                arr,
                {sid: sid_ds[sid][-1] for sid in srs_ids},
                self.season_length,
                self.n_seas_comps,
                h_template,
                self.dma_pdr,
                self.dma_mdr,
                self.warmup_steps,
                self.include_ar,
                self.adaptive,
                self.tau_values,
                self.var_disc_values,
                self.monitor_tau,
                self.universe_builder,
            )
            for srs_ids, arr in batches_input
        ]
        self._batches = self._dispatch(_run_fit_batch, args_list)
        self._freq = freq
        self._series_index = per_series.index.tolist()
        self._fit_h_template = h_template
        return self

    def predict(
        self,
        h: int,
        level: Optional[list] = None,
    ) -> pd.DataFrame:
        """h-step forecast from the held state. Does not modify state."""
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before predict(...).")
        if not isinstance(h, int) or h <= 0:
            raise ValueError("h must be a positive integer.")

        level = self._normalise_level(level)

        args_list = [(state, h, self.sd_method) for state in self._batches]
        preds = self._dispatch(_run_predict_batch, args_list)
        return self._assemble_forecast_output(preds, h, level)

    def update(self, df_new: pd.DataFrame):
        """Extend the held state with new observations.

        ``df_new`` must contain rows for **all** previously fitted series
        (no new series, none missing) and the first new ``ds`` per series
        must equal ``last_ds + freq``. New observations are appended to
        the filter; cost is one filter pass over the new rows only.

        Returns
        -------
        self
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit(...) before update(...).")
        per_series, sid_ds, sid_y, _ = self._prepare_input(df_new, self._freq)

        # Validate completeness
        existing = set(self._series_index)
        provided = set(per_series.index.tolist())
        if provided != existing:
            extra = sorted(provided - existing)[:5]
            missing = sorted(existing - provided)[:5]
            raise ValueError(
                f"update(df_new) must contain rows for all previously "
                f"fitted series (and no new ones). Extra: {extra}, "
                f"Missing: {missing}."
            )

        # Validate temporal continuity: per-series first new ds must
        # equal previous last_ds + freq. Check is per-batch. The arithmetic
        # is polymorphic: integer ds + integer freq, or Timestamp + offset.
        if isinstance(self._freq, (int, np.integer)):
            step = int(self._freq)
            for state in self._batches:
                for sid in state.srs_ids:
                    expected = int(state.last_ds_per_sid[sid]) + step
                    actual = int(sid_ds[sid][0])
                    if actual != expected:
                        raise ValueError(
                            f"Series {sid!r}: first new ds is {actual}, "
                            f"expected {expected} (= last_ds + freq)."
                        )
        else:
            offset = pd.tseries.frequencies.to_offset(self._freq)
            for state in self._batches:
                for sid in state.srs_ids:
                    expected = pd.Timestamp(state.last_ds_per_sid[sid]) + offset
                    actual = pd.Timestamp(sid_ds[sid][0])
                    if actual != expected:
                        raise ValueError(
                            f"Series {sid!r}: first new ds is {actual}, "
                            f"expected {expected} (= last_ds + freq)."
                        )

        # Validate all new sub-series have the same length (so the batch
        # arrays remain rectangular).
        new_lengths = {sid: len(sid_y[sid]) for sid in self._series_index}
        for state in self._batches:
            lens = {new_lengths[sid] for sid in state.srs_ids}
            if len(lens) > 1:
                raise ValueError(
                    f"Series within a batch must receive the same number "
                    f"of new observations; got {sorted(lens)} in a batch "
                    f"of {len(state.srs_ids)} series."
                )

        # Build update jobs per batch
        args_list = []
        for state in self._batches:
            new_arr = np.column_stack([sid_y[sid] for sid in state.srs_ids])
            new_last_ds = {sid: sid_ds[sid][-1] for sid in state.srs_ids}
            args_list.append((state, new_arr, new_last_ds))

        self._batches = self._dispatch(_run_update_batch, args_list)
        return self

    # ------------------------------------------------------------------
    # Stateless one-shot
    # ------------------------------------------------------------------

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: Optional[str] = None,
        level: Optional[list] = None,
        warmup_steps: Optional[int] = None,
    ) -> pd.DataFrame:
        """One-shot fit + predict. Does not modify ``self._batches``.

        A convenience for the one-shot case, where the state is not needed
        afterwards: internally it runs the filter once and predicts.
        """

        scratch = StaticFFS(
            season_length=self.season_length,
            n_seas_comps=self.n_seas_comps,
            dma_pdr=self.dma_pdr,
            dma_mdr=self.dma_mdr,
            max_batch_size=self.max_batch_size,
            dask_client=self.dask_client,
            alias=self.alias,
            sd_method=self.sd_method,
        )
        scratch.fit(df, freq=freq, h_template=h, warmup_steps=warmup_steps)
        return scratch.predict(h, level=level)

    # ------------------------------------------------------------------
    # Diagnostic: cross-validation (single filter pass)
    # ------------------------------------------------------------------

    def _cv_trajectories(self, args_list):
        """Produce the per-batch CV ``_CVTrajectory`` list from the dispatch
        arg tuples. Overridable hook: ``AutoFFS``
        routes this through the block seam (see ``AutoFFS._cv_trajectories``)."""
        return self._dispatch(_run_cv_batch, args_list)

    def _cv_combine(self, batch_item, arr, level):
        """Combine one batch's CV result into per-cutoff ``(loc, sd, bounds)``.

        Overridable hook: Legacy combines a single ``_CVTrajectory`` with its own
        in-scan DMA weights; ``AutoFFS`` combines a list
        of blocks' trajectories with the union DMA (see ``AutoFFS._cv_combine``).
        ``arr`` is the batch's ``(T, n_series)`` observations (unused here).
        Returns ``(combined_list, cutoff_t_idx)``.
        """
        traj = batch_item
        combined = []
        for gather_idx in range(len(traj.cutoff_t_idx)):
            pred = FFSPredictive(
                loc=None, sd=None,
                f_h=traj.f_h[gather_idx],     # (nm, n_series, h)
                q_h=traj.q_h[gather_idx],
                nu=traj.nu[gather_idx],       # (nm, n_series)
                weights=traj.weights[gather_idx],
            )
            loc_combined = (pred.f_h * pred.weights[..., None]).sum(axis=0)  # (n_series, h)
            bounds = _t_quantile_average(pred, level) if level else None
            sd_arr = _combined_predictive_sd(pred, self.sd_method)          # (h, n_series)
            combined.append((loc_combined, sd_arr, bounds))
        return combined, np.asarray(traj.cutoff_t_idx)

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int = 1,
        step_size: Optional[int] = None,
        freq: Optional[str] = None,
        level: Optional[list] = None,
        warmup_steps: Optional[int] = None,
        output: str = "long",
        exog: Optional[pd.DataFrame] = None,
    ):
        """Rolling-origin backtest in a single filter pass.

    ``exog`` supplies an EXOGENOUS regression design, long
    ``(unique_id, ds, <regressor columns>)``, covering the same dates as
    ``df``. It is the backtest counterpart of ``AutoFFSUniverse``'s
    ``exog_provider``: a blocks universe whose cells carry a ``Regressors``
    component needs it, and one whose cells carry an ``AR`` component must NOT
    be given it (an AR tail derives its own design from the y stream).

        ``output`` selects the return shape:

        * ``"long"`` (default) — the long frame
          ``(unique_id, ds, cutoff, y, <alias>, <alias>-sd[, -lo-L, -hi-L])``.
        * ``"xarray"`` — an ``xarray.Dataset`` over ``(unique_id, window, h)``
          with variables ``loc``/``sd``/``y`` (plus ``lo``/``hi`` over ``level``),
          ``cutoff`` and ``ds`` as coordinates. See :func:`_cv_to_xarray`. This
          is the shape callers pivot back to anyway before writing netCDF; the
          long form is expected to become the option rather than the default.

        Window ``i`` (counting from the most recent at ``i = 0``) trains
        on indices ``0 .. T - h - i*step_size - 1`` and forecasts indices
        ``T - h - i*step_size .. T - i*step_size - 1``. Default
        ``step_size = h`` gives non-overlapping windows.

        Implementation: one forward pass over the full series. The
        h-step forecast emission is gated by ``lax.cond`` so it only
        runs at the requested cutoff timesteps — non-cutoff steps incur
        the filter update only. Cost is one filter pass plus
        ``n_windows`` cheap forecast emissions, which for typical
        ``n_windows`` is a small fraction of the cost of ``n_windows``
        independent re-fits.

        Independent of ``fit`` / ``predict`` state. ``self._batches`` is
        not read or written.
        """
        if output not in ("long", "xarray"):
            raise ValueError("output must be 'long' or 'xarray'.")
        if not isinstance(h, int) or h <= 0:
            raise ValueError("h must be a positive integer.")
        if not isinstance(n_windows, int) or n_windows <= 0:
            raise ValueError("n_windows must be a positive integer.")
        if step_size is None:
            step_size = h
        if not isinstance(step_size, int) or step_size <= 0:
            raise ValueError("step_size must be a positive integer.")
        level = self._normalise_level(level)

        per_series, sid_ds, sid_y, freq = self._prepare_input(df, freq)

        min_len = self._min_filter_length()
        # Hard floor: enough observations to form at least one forecast
        # origin (the oldest window's train_end must be >= 1).
        floor = h + (n_windows - 1) * step_size + 1
        too_short = per_series[per_series["length"] < floor]
        if len(too_short):
            raise ValueError(
                f"{len(too_short)} series shorter than {floor} observations, "
                f"the minimum to form a forecast origin for n_windows="
                f"{n_windows}, h={h}, step_size={step_size}. Examples: "
                f"{too_short.index.tolist()[:5]}."
            )
        # Non-blocking: seasonality is under-determined below 2 cycles.
        _warn_under_seasonal(
            per_series.index[per_series["length"] < min_len],
            min_len, self.season_length)

        # Compute cutoff indices per batch: for window i (0 = most
        # recent), train ends at T - h - i*step_size, so the scan-time
        # index for emission is train_end - 1. We can't share a single
        # cutoff list across batches because each length-group has its
        # own T, so cutoffs are computed per batch below.
        window_indices = list(range(n_windows))

        args_list = []
        batch_arrs = []  # list[(srs_ids, arr)] — retained for the combine hook
        sid_x = self._prepare_exog(exog, sid_ds)
        for srs_ids, arr in self._iter_batches(per_series, sid_y):
            T_full = arr.shape[0]
            window_t_pairs = [
                (i, T_full - h - i * step_size - 1) for i in window_indices
            ]
            cutoff_t_idx = np.array(
                sorted(t for _, t in window_t_pairs),
                dtype=np.int32,
            )
            args_list.append(
                (
                    tuple(srs_ids),
                    arr,
                    cutoff_t_idx,
                    self.season_length,
                    self.n_seas_comps,
                    h,
                    self.dma_pdr,
                    self.dma_mdr,
                    warmup_steps if warmup_steps is not None else 0,
                    self.include_ar,
                    self.adaptive,
                    self.tau_values,
                    self.var_disc_values,
                    False,                # capture_trace: summary CV needs no per-step trace
                    self.monitor_tau,     # honour the configured monitor (was: default)
                    self.universe_builder,  # honour a custom universe (was: ignored)
                    # (T, q, n_reg) exogenous design for THIS batch, series in
                    # srs_ids order -- the same column_stack the fit array uses.
                    (None if sid_x is None
                     else np.stack([sid_x[sid] for sid in srs_ids], axis=1)),
                )
            )
            batch_arrs.append((tuple(srs_ids), arr))

        trajectories = self._cv_trajectories(args_list)

        # Map traj rows back to output. Cutoffs in traj.cutoff_t_idx
        # are ascending in t (oldest first); we just iterate over them
        # directly. The output `cutoff` column distinguishes windows,
        # so we don't need a window_i variable in the row.
        rows: list = []
        # Preserve integer-typed ds/cutoff in integer mode; only datetime
        # mode goes through pd.Timestamp. Mirrors _assemble_forecast_output.
        is_int_mode = isinstance(freq, (int, np.integer))
        for batch_item, (srs_ids, arr) in zip(trajectories, batch_arrs):
            combined, cutoff_t_idx = self._cv_combine(batch_item, arr, level)
            for gather_idx, t_idx in enumerate(cutoff_t_idx):
                loc_combined, sd_arr, bounds = combined[gather_idx]

                # train_end = t_idx + 1 (filter consumed obs 0..t_idx).
                train_end = int(t_idx) + 1

                for j, sid in enumerate(srs_ids):
                    ds_array = sid_ds[sid]
                    y_array = sid_y[sid]
                    last_obs_ds = ds_array[train_end - 1]
                    cutoff_ds = (
                        int(last_obs_ds) if is_int_mode else pd.Timestamp(last_obs_ds)
                    )
                    future_ds = ds_array[train_end : train_end + h]
                    actual_y = y_array[train_end : train_end + h]
                    forecast_loc = loc_combined[j]  # (h,)

                    for k in range(h):
                        row = {
                            "unique_id": sid,
                            "ds": (
                                int(future_ds[k])
                                if is_int_mode
                                else pd.Timestamp(future_ds[k])
                            ),
                            "cutoff": cutoff_ds,
                            "y": float(actual_y[k]),
                            self.alias: float(forecast_loc[k]),
                            f"{self.alias}-sd": float(sd_arr[k, j]),
                        }
                        if level:
                            for L in level:
                                lo, hi = bounds[L]
                                row[f"{self.alias}-lo-{L}"] = float(lo[k, j])
                                row[f"{self.alias}-hi-{L}"] = float(hi[k, j])
                        rows.append(row)

        out_df = pd.DataFrame(rows)
        if len(out_df):
            out_df = out_df.sort_values(["unique_id", "cutoff", "ds"]).reset_index(
                drop=True
            )
        series_index = getattr(self, "_input_series_index", None)
        if output == "xarray":
            return _restore_series_index(
                _cv_to_xarray(out_df, self.alias, level), series_index)
        return _restore_series_index(out_df, series_index)

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _assemble_forecast_output(self, preds, h, level):
        """Build long-format output from a list of FFSPredictive bundles.

        Uses ``self._batches`` to know which series belongs to which
        prediction (and to look up its last_ds), so this can only be
        called when the model is fitted. ``preds`` must be ordered
        consistently with ``self._batches``.
        """
        out_loc: dict = {}
        out_sd: dict = {}
        out_intervals: dict = {}
        for state, pred in zip(self._batches, preds):
            for j, sid in enumerate(state.srs_ids):
                out_loc[sid] = pred.loc[j]  # (h,)
                out_sd[sid] = pred.sd[j]  # (h,) — LoTV mixture SD
            if level:
                bounds = _t_quantile_average(pred, level)
                for j, sid in enumerate(state.srs_ids):
                    out_intervals[sid] = {
                        L: (bounds[L][0][:, j], bounds[L][1][:, j]) for L in level
                    }

        # Find each sid's last_ds via the batch state it belongs to
        last_ds_lookup: dict = {}
        for state in self._batches:
            last_ds_lookup.update(state.last_ds_per_sid)

        is_int_mode = isinstance(self._freq, (int, np.integer))
        if not is_int_mode:
            offset = pd.tseries.frequencies.to_offset(self._freq)

        rows: list = []
        for sid in self._series_index:
            last_ds = last_ds_lookup[sid]
            if is_int_mode:
                step = int(self._freq)
                future_ds = np.arange(
                    int(last_ds) + step,
                    int(last_ds) + step * (h + 1),
                    step,
                )
            else:
                future_ds = pd.date_range(
                    start=pd.Timestamp(last_ds) + offset,
                    periods=h,
                    freq=self._freq,
                )
            mu = out_loc[sid]
            sd = out_sd[sid]
            for k in range(h):
                row = {
                    "unique_id": sid,
                    "ds": future_ds[k],
                    self.alias: float(mu[k]),
                    f"{self.alias}-sd": float(sd[k]),
                }
                if level:
                    for L in level:
                        lo, hi = out_intervals[sid][L]
                        row[f"{self.alias}-lo-{L}"] = float(lo[k])
                        row[f"{self.alias}-hi-{L}"] = float(hi[k])
                rows.append(row)
        return pd.DataFrame(rows)


"""
Persistence layer for AutoFFS
=============================

Append after the existing ``AutoFFS`` class in ``ffs_core.py``.

Adds:

  * ``_BATCH_SCHEMA_VERSION`` — a module-level constant. Bump when the
    on-disk batch format changes incompatibly.
  * ``_save_batch_state(state, fname)`` / ``_load_batch_state(fname,
    construct_kwargs)`` — primitives. Mirror the existing
    ``multi_model_dlm.save`` / ``load`` layout, with two extra groups
    for the Allocator state and metadata.
  * Single-file save/load on ``AutoFFS``:
        ``model.save(path)`` / ``AutoFFS.load(path)``.
    All batches are written into one H5 file. Suitable for laptop /
    notebook workflows.
  * ``AutoFFSUniverse`` — directory-based persistent operation.
    Supports incremental ``update``, ``add_series``, ``remove_series``,
    and ``defragment``. Suitable for production / cluster workflows.

Imports needed at the top of ``ffs_core.py``:

    import os
    import h5py

(``h5py`` is already a transitive dependency via ``multi_model_dlm.save``.)


Dev note for API documentation
------------------------------
"Inactive" series are not the same as missing data:

  * Missing data is a per-timestep property of a series the filter knows
    how to handle (skip the update, prior carries forward, posterior
    variance grows). The series stays alive in the universe.
  * Inactive is a per-series flag in the universe manifest, set by
    ``remove_series``. The series's slot in the batch state still exists
    physically (we haven't rebuilt the batch yet), but the universe's
    bookkeeping treats the series as gone — it is excluded from
    ``forecast`` output and not required as input to ``update``.

Inactive slots are reclaimed by ``defragment``, which rebuilds affected
batches to drop the dead columns. Users can re-add a previously
removed series with ``add_series`` (providing fresh history) — old
state is not preserved across removal.
"""

# --- imports needed at top of ffs_core.py ---
# import os
# import h5py


_BATCH_SCHEMA_VERSION = 2


# =====================================================================
# Batch save / load primitives
# =====================================================================


def _save_batch_state(state, fname, active=None):
    """Write a single ``_BatchState`` to an HDF5 file.

    Layout::

        /dlm_state/<param>      — multi.dlm_state arrays
        /dims/                  — multi dimensions and mdl_keys
        /Other/                 — multi F, G, GH, disc_rates, ... monitor
        /allocator/
            pset                — Allocator state
            mset
            forecast_history
            model_indicator
            model_indicator_cols
            latest_weights      — final DMA weights, (nm, q, 1)
        /metadata/
            schema_version      — int
            srs_ids             — variable-length string array
            last_ds             — int64 ns timestamps, parallel to srs_ids
            T_ingested          — int
            active              — bool array, parallel to srs_ids

    Parameters
    ----------
    state : _BatchState
    fname : str
    active : np.ndarray of bool or None
        Per-slot active mask. ``None`` marks all slots active. Used by
        ``AutoFFSUniverse`` to persist the active/inactive status of
        slots in a batch.
    """
    multi = state.multi
    dma = state.dma

    if active is None:
        active = np.ones(len(state.srs_ids), dtype=bool)
    else:
        active = np.asarray(active, dtype=bool)
        if active.shape != (len(state.srs_ids),):
            raise ValueError(
                f"active mask shape {active.shape} doesn't match "
                f"n_series {len(state.srs_ids)}."
            )

    with h5py.File(fname, "w") as f:
        # -------- multi (mirrors multi_model_dlm.save) --------
        g = f.create_group("dlm_state")
        for param, arr in multi.dlm_state.items():
            g.create_dataset(param, data=np.asarray(arr))

        g = f.create_group("dims")
        for param, val in {
            "q": multi.q,
            "nm": multi.nm,
            "p": multi.p,
            "k": multi.k,
            "npad": multi.npad,
            "mdl_keys": multi.mdl_keys,
        }.items():
            g.create_dataset(param, data=val)

        g = f.create_group("Other")
        for param in (
            "F",
            "G",
            "GH",
            "disc_rates",
            "disc_rates_damped",
            "variance_disc",
            "variance_power",
            "mult_comps",
            "monitor",
            "monitor_inject",
        ):
            g.create_dataset(param, data=np.asarray(getattr(multi, param)))

        # Regression-tail descriptors. Without these a reloaded regression
        # universe loses its tail (n_regressors defaults to 0) and silently
        # forecasts as structural — the M5 SNAP run reloads on every origin, so
        # this round-trip is mandatory. Legacy structural batches omit them and
        # restore as 0 / empty / False (no tail; bit-exact structural path).
        g.create_dataset("n_regressors", data=int(getattr(multi, "n_regressors", 0)))
        g.create_dataset(
            "reg_mask",
            data=np.asarray(
                getattr(multi, "reg_mask", np.zeros(multi.p, dtype=bool))
            ),
        )
        g.create_dataset(
            "exog_regressors", data=int(getattr(multi, "exog_regressors", False))
        )

        # -------- allocator --------
        g = f.create_group("allocator")
        g.create_dataset("pset", data=np.asarray(dma.state.pset))
        g.create_dataset("mset", data=np.asarray(dma.state.mset))
        g.create_dataset(
            "forecast_history",
            data=np.asarray(dma.state.forecast_history),
        )
        g.create_dataset(
            "model_indicator",
            data=np.asarray(state.model_indicator.values),
        )
        # Coerce column labels to str — pd.get_dummies on a numeric
        # input (e.g. model_desc['Class'] as float64) yields float
        # labels (0.0, 1.0, ...) that can't be .encode()'d directly.
        g.create_dataset(
            "model_indicator_cols",
            data=[str(c).encode("utf-8") for c in state.model_indicator.columns],
        )
        g.create_dataset(
            "latest_weights",
            data=np.asarray(state.latest_weights),
        )

        # -------- metadata --------
        g = f.create_group("metadata")
        g.create_dataset("schema_version", data=_BATCH_SCHEMA_VERSION)
        g.create_dataset(
            "srs_ids",
            data=[str(s).encode("utf-8") for s in state.srs_ids],
        )
        # last_ds: int64 always, but interpretation depends on freq type.
        # If freq is int (M1/M3-style), the int64 is the literal index;
        # if freq is a pandas alias (datetime), the int64 is nanoseconds
        # since epoch (Timestamp.value).
        last_ds_arr = np.array(
            [
                (
                    int(state.last_ds_per_sid[s])
                    if isinstance(state.last_ds_per_sid[s], (int, np.integer))
                    else pd.Timestamp(state.last_ds_per_sid[s]).value
                )
                for s in state.srs_ids
            ],
            dtype=np.int64,
        )
        g.create_dataset("last_ds", data=last_ds_arr)
        g.create_dataset(
            "last_ds_is_int",
            data=int(
                isinstance(state.last_ds_per_sid[state.srs_ids[0]], (int, np.integer))
            ),
        )
        g.create_dataset("T_ingested", data=int(state.T_ingested))
        g.create_dataset("active", data=active)


def _save_batch_state_with_active(state, fname, active_array):
    """Convenience: save with an explicit active mask.

    Kept as a thin wrapper around ``_save_batch_state`` for call sites
    that want to be explicit about persisting an active state.
    """
    _save_batch_state(state, fname, active=active_array)


def _load_batch_state(fname):
    """Reconstruct a ``_BatchState`` from an HDF5 file.

    Returns
    -------
    state : _BatchState
    active : np.ndarray of bool
        Parallel to ``state.srs_ids``. The universe layer uses this to
        decide which slots to expose in forecasts; the single-file
        ``AutoFFS.load`` ignores it (assumes all active).
    """
    with h5py.File(fname, "r") as f:
        # Schema check

        # Schema check: v2 changed the on-disk semantics of /last_ds
        # to be polymorphic (int64 means int directly when _freq is
        # int, ns when _freq is a frequency alias). v1 files always
        # interpreted it as ns and so are not safe to load under v2.
        on_disk_version = int(f["metadata/schema_version"][()])
        if on_disk_version != _BATCH_SCHEMA_VERSION:
            raise RuntimeError(
                f"Batch file at {fname!r} has schema_version="
                f"{on_disk_version}; this code expects "
                f"{_BATCH_SCHEMA_VERSION}. Re-fit and re-save the "
                "model with this version of DLMAX."
            )

        # sv = int(f["metadata/schema_version"][()])
        # if sv != _BATCH_SCHEMA_VERSION:
        #    raise ValueError(
        #        f"Batch file {fname} has schema_version={sv}, expected "
        #        f"{_BATCH_SCHEMA_VERSION}. The on-disk format has "
        #        f"changed incompatibly; refit and resave is required."
        #    )

        # Build a multi shell. We need the dims and templates before we
        # can populate dlm_state. multi_model_dlm.__init__ requires an
        # ATS dict, but we don't have one — we have the assembled
        # tensors. Construct an empty multi via __new__ and populate.
        multi = multi_model_dlm.__new__(multi_model_dlm)
        multi.dlm_compute = devices.dlm_compute

        g = f["dims"]
        multi.q = int(g["q"][()])
        multi.nm = int(g["nm"][()])
        multi.p = int(g["p"][()])
        multi.k = int(g["k"][()])
        multi.npad = int(g["npad"][()])
        multi.mdl_keys = [
            s.decode() if isinstance(s, bytes) else str(s) for s in g["mdl_keys"][()]
        ]

        g = f["Other"]
        for param in (
            "F",
            "G",
            "GH",
            "disc_rates",
            "disc_rates_damped",
            "variance_disc",
            "variance_power",
            "mult_comps",
            "monitor",
        ):
            setattr(
                multi,
                param,
                device_put(jnp.asarray(g[param]), devices.dlm_compute),
            )
        # monitor_inject (per-state revised discount, used by _multi_params_tuple
        # on the update/scan path) is optional in the stored format: default to
        # ones (no injection) when absent, so a universe saved without it opens.
        if "monitor_inject" in g:
            multi.monitor_inject = device_put(
                jnp.asarray(g["monitor_inject"]), devices.dlm_compute)
        else:
            multi.monitor_inject = jnp.ones_like(multi.disc_rates)

        # Regression-tail descriptors (see _save_batch_state). Legacy structural
        # batches lack them -> no tail (n_regressors=0, structural path unchanged).
        if "n_regressors" in g:
            multi.n_regressors = int(g["n_regressors"][()])
            multi.reg_mask = device_put(
                jnp.asarray(g["reg_mask"]), devices.dlm_compute)
            multi.exog_regressors = bool(int(g["exog_regressors"][()]))
        else:
            multi.n_regressors = 0
            multi.reg_mask = device_put(
                jnp.zeros(multi.p, dtype=bool), devices.dlm_compute)
            multi.exog_regressors = False

        multi.dlm_state = {
            param: device_put(jnp.asarray(f["dlm_state"][param]), devices.dlm_compute)
            for param in f["dlm_state"].keys()
        }

        # Allocator
        g = f["allocator"]
        mi_values = np.asarray(g["model_indicator"][()])
        mi_cols = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in g["model_indicator_cols"][()]
        ]
        model_indicator = pd.DataFrame(mi_values, columns=mi_cols)
        latest_weights = np.asarray(g["latest_weights"][()])
        pset = jnp.asarray(g["pset"][()])
        mset = jnp.asarray(g["mset"][()])
        forecast_history = jnp.asarray(g["forecast_history"][()])

        # Reconstruct the Allocator. We bypass __init__ and patch all
        # attributes that __init__ sets: device, scoring_rule,
        # update_rule, horizon_aggregator, mi, state. scoring_rule,
        # update_rule, and horizon_aggregator are bound later in
        # _attach_dma_step_fn (they depend on dma_pdr / dma_mdr, which
        # the loader caller supplies).
        dma = Allocator.__new__(Allocator)
        dma.state = AllocatorState(
            pset=pset,
            mset=mset,
            forecast_history=forecast_history,
        )
        dma.mi = device_put(jnp.asarray(mi_values), devices.allocation_compute)
        dma.device = devices.allocation_compute

        # Metadata
        g = f["metadata"]
        srs_ids = tuple(
            s.decode() if isinstance(s, bytes) else str(s) for s in g["srs_ids"][()]
        )
        last_ds_arr = np.asarray(g["last_ds"][()])
        # Default to nanosecond interpretation if the flag is missing
        # (forward compatibility — a v1 file that somehow gets here
        # would still parse correctly, although the schema check above
        # should already have rejected it).
        if "last_ds_is_int" in g:
            last_ds_is_int = bool(int(g["last_ds_is_int"][()]))
        else:
            last_ds_is_int = False
        if last_ds_is_int:
            last_ds_per_sid = {sid: int(ts) for sid, ts in zip(srs_ids, last_ds_arr)}
        else:
            last_ds_per_sid = {
                sid: pd.Timestamp(int(ts), unit="ns")
                for sid, ts in zip(srs_ids, last_ds_arr)
            }
        T_ingested = int(g["T_ingested"][()])
        active = np.asarray(g["active"][()], dtype=bool)

    state = _BatchState(
        srs_ids=srs_ids,
        multi=multi,
        dma=dma,
        model_indicator=model_indicator,
        latest_weights=latest_weights,
        last_ds_per_sid=last_ds_per_sid,
        T_ingested=T_ingested,
    )
    return state, active


def _load_batch_meta(fname):
    """Cheap read of a batch file's ``/metadata`` group only.

    Returns ``(srs_ids, active, last_ds_per_sid)`` without touching the (large)
    ``dlm_state`` / ``Other`` groups — KB instead of tens of MB per batch. Used
    by ``AutoFFSUniverse.update`` / ``forecast`` on the head to build the small
    per-batch payloads for a *distributed* (path-based) per-period step: workers
    load the full state from shared storage themselves, so the head only needs
    the series order, active mask and last-ds per slot.
    """
    with h5py.File(fname, "r") as f:
        on_disk_version = int(f["metadata/schema_version"][()])
        if on_disk_version != _BATCH_SCHEMA_VERSION:
            raise RuntimeError(
                f"Batch file at {fname!r} has schema_version="
                f"{on_disk_version}; this code expects {_BATCH_SCHEMA_VERSION}."
            )
        g = f["metadata"]
        srs_ids = tuple(
            s.decode() if isinstance(s, bytes) else str(s) for s in g["srs_ids"][()]
        )
        last_ds_arr = np.asarray(g["last_ds"][()])
        last_ds_is_int = (
            bool(int(g["last_ds_is_int"][()])) if "last_ds_is_int" in g else False
        )
        if last_ds_is_int:
            last_ds_per_sid = {sid: int(ts) for sid, ts in zip(srs_ids, last_ds_arr)}
        else:
            last_ds_per_sid = {
                sid: pd.Timestamp(int(ts), unit="ns")
                for sid, ts in zip(srs_ids, last_ds_arr)
            }
        active = np.asarray(g["active"][()], dtype=bool)
    return srs_ids, active, last_ds_per_sid


def _attach_dma_step_fn(dma, dma_pdr, dma_mdr):
    """Reattach the closures that ``Allocator.__init__`` would set.

    ``Allocator.__new__`` skips ``__init__``, so after a load we need
    to re-bind ``scoring_rule``, ``update_rule``, and
    ``horizon_aggregator`` from the supplied ``dma_pdr`` / ``dma_mdr``.
    These three are not persisted because they're closures over
    user-supplied params, not state.
    """
    dma.scoring_rule = LogScore
    dma.update_rule = Partial(
        PowerLawUpdate,
        dma_pdr=dma_pdr,
        dma_mdr=dma_mdr,
        c=1e-3,
    )
    dma.horizon_aggregator = IdentityAggregator
    return dma


# =====================================================================
# Single-file save / load on AutoFFS
# =====================================================================


def _autoffs_save(self, path):
    """Save the fitted state to a single HDF5 file.

    All batches are written under ``/batches/<i>/``. Suitable for the
    typical laptop/notebook workflow where you want one self-contained
    file per model. Use :class:`AutoFFSUniverse` for cluster-scale
    streaming workloads.

    Parameters
    ----------
    path : str
        Output filename. Overwritten if it exists.
    """
    if not self.is_fitted:
        raise RuntimeError("Cannot save unfitted model. Call fit(...) first.")

    # Save each batch to a temporary in-memory dict, then merge into one file.
    # h5py supports nested groups directly so we just write into the same file.
    with h5py.File(path, "w") as f:
        # Top-level config
        cfg = f.create_group("config")
        for k, v in {
            "season_length": (
                self.season_length if self.season_length is not None else -1
            ),
            "n_seas_comps": (
                self.n_seas_comps if self.n_seas_comps is not None else -1
            ),
            "dma_pdr": float(self.dma_pdr),
            "dma_mdr": float(self.dma_mdr),
            "max_batch_size": (
                self.max_batch_size if self.max_batch_size is not None else -1
            ),
            "alias": self.alias,
            "freq": self._freq,
            "fit_h_template": int(self._fit_h_template or 18),
            "warmup_steps": self.warmup_steps,
        }.items():
            cfg.create_dataset(k, data=v)

        # Series order — must be preserved for deterministic output
        cfg.create_dataset(
            "series_index",
            data=[str(s).encode("utf-8") for s in self._series_index],
        )

        cfg.create_dataset("schema_version", data=_BATCH_SCHEMA_VERSION)
        cfg.create_dataset("n_batches", data=len(self._batches))

    # Write each batch as a separate file then merge — easier than
    # interleaving group writes. For very large multi-batch fits this
    # is slightly wasteful; the universe layer avoids it.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpd:
        for i, state in enumerate(self._batches):
            tmp_fname = os.path.join(tmpd, f"batch_{i}.h5")
            _save_batch_state(state, tmp_fname)
            with h5py.File(tmp_fname, "r") as src, h5py.File(path, "a") as dst:
                src.copy("/", dst, name=f"batches/{i}")


@classmethod
def _autoffs_load(cls, path):
    """Load a fitted ``AutoFFS`` from a single HDF5 file.

    Parameters
    ----------
    path : str

    Returns
    -------
    AutoFFS
        A fitted instance, ready for ``predict``, ``update``, etc.
    """
    with h5py.File(path, "r") as f:
        sv = int(f["config/schema_version"][()])
        if sv != _BATCH_SCHEMA_VERSION:
            raise ValueError(
                f"File {path} has schema_version={sv}, expected "
                f"{_BATCH_SCHEMA_VERSION}."
            )
        cfg = f["config"]
        season_length = int(cfg["season_length"][()])
        if season_length == -1:
            season_length = None
        n_seas_comps = int(cfg["n_seas_comps"][()])
        if n_seas_comps == -1:
            n_seas_comps = None
        max_batch_size = int(cfg["max_batch_size"][()])
        if max_batch_size == -1:
            max_batch_size = None

        dma_pdr = float(cfg["dma_pdr"][()])
        dma_mdr = float(cfg["dma_mdr"][()])
        alias = cfg["alias"][()]
        if isinstance(alias, bytes):
            alias = alias.decode()
        freq = cfg["freq"][()]
        if isinstance(freq, bytes):
            freq = freq.decode()
        fit_h_template = int(cfg["fit_h_template"][()])
        warmup_steps = int(cfg["warmup_steps"][()])

        series_index = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in cfg["series_index"][()]
        ]
        n_batches = int(cfg["n_batches"][()])

    model = cls(
        season_length=season_length,
        n_seas_comps=n_seas_comps,
        dma_pdr=dma_pdr,
        dma_mdr=dma_mdr,
        max_batch_size=max_batch_size,
        alias=alias,
    )

    # Load each batch by extracting it from the merged file. Easiest
    # path: copy each batch group out to a temp file and reuse the
    # batch loader.
    import tempfile

    batches = []
    with tempfile.TemporaryDirectory() as tmpd:
        for i in range(n_batches):
            tmp_fname = os.path.join(tmpd, f"batch_{i}.h5")
            with h5py.File(path, "r") as src, h5py.File(tmp_fname, "w") as dst:
                # Copy children of /batches/{i} to the root of dst.
                # Cannot copy the group itself to "/" because the
                # destination root group always exists.
                src_grp = src[f"batches/{i}"]
                for name in src_grp.keys():
                    src.copy(f"batches/{i}/{name}", dst, name=name)
            state, _active = _load_batch_state(tmp_fname)
            _attach_dma_step_fn(state.dma, dma_pdr, dma_mdr)
            batches.append(state)

    model._batches = batches
    model._freq = freq
    model._series_index = series_index
    model._fit_h_template = fit_h_template
    return model


# Bind the methods to StaticFFS at module load time (inherited by AutoFFS)
StaticFFS.save = _autoffs_save
StaticFFS.load = _autoffs_load


# =====================================================================
# AutoFFSUniverse: directory-based persistent operation
# =====================================================================


class StaticFFSUniverse:
    """Directory-backed persistent AutoFFS instance.

    A universe owns a directory of batch files plus a small manifest
    that tracks per-series state. Operations (``fit``, ``update``,
    ``forecast``, ``add_series``, ``remove_series``) read and write
    batch files incrementally, so:

      * ``add_series`` writes one new file; existing batches untouched.
      * ``remove_series`` flips a manifest flag; no batch files written.
      * ``update`` writes only batches that received new observations.

    This makes the universe friendly to streaming workloads at large
    scale. For small / one-shot workloads prefer
    :meth:`AutoFFS.save` / :meth:`AutoFFS.load`.

    Directory layout::

        path/
            manifest.h5             — series_id -> (batch_id, slot, ...)
            config.h5               — model hyperparameters, freq
            batches/
                batch_0.h5
                batch_1.h5
                ...

    Parameters
    ----------
    path : str
        Directory path. Created on ``create_universe``; expected to
        exist on ``open_universe``.

    Examples
    --------
    Create a fresh universe and fit history:

    >>> uni = AutoFFSUniverse.create(
    ...     "path/to/uni", season_length=12, max_batch_size=100,
    ... )
    >>> uni.fit(df_history)
    >>> uni.update(df_jan_2026)
    >>> fc = uni.forecast(h=12, level=[80, 95])

    Reopen later:

    >>> uni = AutoFFSUniverse.open("path/to/uni")
    >>> uni.update(df_feb_2026)
    >>> fc = uni.forecast(h=12)

    Add or remove series:

    >>> uni.add_series("new_product_X", df_history_X)
    >>> uni.remove_series("discontinued_product_Y")
    >>> uni.defragment()  # reclaim slots from removed series

    Notes
    -----
    Single-writer assumption. The universe is not safe for concurrent
    writes from multiple processes; lock externally if needed.
    """

    _MANIFEST_VERSION = 1

    def __init__(
        self,
        path: str,
        season_length: Optional[int] = None,
        n_seas_comps: Optional[int] = None,
        dma_pdr: float = 0.90,
        dma_mdr: Optional[float] = None,
        max_batch_size: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        alias: str = "AutoFFS",
        dask_client=None,
        sd_method: str = "quantile",
        adaptive: bool = False,
        tau_values=None,
        var_disc_values=None,
        monitor_tau=MONITOR_TAU,
        universe_builder=None,
        exog_provider=None,
        pad_batches: bool = True,
    ):
        if sd_method not in ("quantile", "moment"):
            raise ValueError("sd_method must be 'quantile' or 'moment'.")
        # Capacity padding trades memory for compile count. It pays where a
        # universe has many short series and frequent add_series churn, since
        # fixing the shape avoids a per-add XLA recompilation. It is a poor
        # trade for a few LONG series with a big state and a long horizon: cost
        # scales with capacity x nm x k^2 x h, so padding e.g. 27 hourly series
        # to a capacity of 500 inflates the series axis ~18x for no benefit --
        # a fixed panel never adds series, so nothing recompiles. Set False
        # there. See also: just set max_batch_size to the panel size.
        self.pad_batches = bool(pad_batches)
        self.path = path
        self.sd_method = sd_method
        self.season_length = season_length
        self.n_seas_comps = n_seas_comps
        self.dma_pdr = dma_pdr
        self.dma_mdr = dma_pdr if dma_mdr is None else dma_mdr
        self.max_batch_size = max_batch_size
        # Fit-time prior policy: warmup_steps > 0 selects the diffuse
        # prior (m0=mean, C0=10*var, V0=var, floored) over the legacy
        # OLS elicitation, and flags the first warmup_steps observations
        # as settling. Used by fit() and add_series(); update() ignores
        # it (it continues already-settled state).
        self.warmup_steps = warmup_steps
        # Error-monitoring (adaptive-discount) universe. Forwarded to the fit /
        # add_series path so the built multi carries monitor=tau; persisted so a
        # reopened universe (M5 update/forecast) keeps adapting. include_ar is
        # ignored on this path.
        self.adaptive = adaptive
        self.tau_values = tau_values
        self.var_disc_values = var_disc_values
        # Static-grid signed-error monitor (M4-final approach): a scalar SD
        # multiple (e.g. 3.0) baked into every batch's universe at build via
        # assemble_models(monitor_tau=...). Persisted so a reopened universe's
        # add_series builds new series with the same monitor; the streaming
        # filter detects the baked monitor via _adapt_tau (no per-step arg).
        self.monitor_tau = monitor_tau
        # Custom universe builder (FFS front end): a callable
        # (init_data, h, UniverseContext) -> (models, model_desc). None -> the
        # standard FFS universe. A CALLABLE CAN'T BE PERSISTED, so config stores
        # only its name (for a reopen safety check) and the builder must be
        # re-supplied to open().
        self.universe_builder = universe_builder
        # Exogenous-regressor provider (FFS front end): a callable
        # ``(srs_ids, ds) -> ndarray (T, n_series, n_regressors)`` giving the
        # known regressor design (e.g. SNAP indicators) for the supplied series
        # and calendar dates ``ds`` (length T). Called in-process by fit /
        # update / forecast / add_series to materialise each batch's design
        # matrix (eagerly, so the dispatched arrays stay picklable). LIKE
        # ``universe_builder`` it cannot be persisted — re-supply it to
        # ``open()``. None -> structural / AR universe (no exog). Only meaningful
        # when the universe_builder emits an exogenous regression tail.
        self.exog_provider = exog_provider
        self.alias = alias
        self.dask_client = dask_client
        self._fit_h_template = 18
        self._freq: Optional[Union[str, int]] = None
        # Manifest is a DataFrame indexed by unique_id with columns
        # batch_id, last_ds, active. Loaded lazily.
        self._manifest: Optional[pd.DataFrame] = None
        self._next_batch_id: int = 0

    # ------------------------------------------------------------------
    # Construction / opening
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, path: str, **kwargs) -> "StaticFFSUniverse":
        """Create a new empty universe at ``path``.

        ``kwargs`` are passed to ``__init__`` (``season_length``,
        ``dma_pdr``, etc.).
        """
        if os.path.exists(path) and os.listdir(path):
            raise FileExistsError(
                f"Universe path {path} already exists and is non-empty. "
                f"Use open() instead, or remove the directory."
            )
        os.makedirs(os.path.join(path, "batches"), exist_ok=True)
        uni = cls(path=path, **kwargs)
        uni._manifest = pd.DataFrame(
            columns=["batch_id", "last_ds", "active"],
        )
        uni._save_config()
        uni._save_manifest()
        return uni

    @classmethod
    def open(cls, path: str, dask_client=None, universe_builder=None,
             exog_provider=None) -> "StaticFFSUniverse":
        """Open an existing universe.

        ``universe_builder`` must be re-supplied if the universe was built with
        a custom builder (a callable can't be persisted) — it is needed for
        ``add_series`` / ``defragment`` to rebuild batches with the same model
        set. A persisted name is checked against the supplied builder.

        ``exog_provider`` must likewise be re-supplied for an exogenous-regressor
        universe (e.g. SNAP); it is the ``(srs_ids, ds) -> (T, n_series, n_reg)``
        callable that supplies the regressor design at update / forecast time.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"No universe directory at {path}.")
        config_path = os.path.join(path, "config.h5")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.h5 in {path}.")

        with h5py.File(config_path, "r") as f:
            cfg = f["config"]
            season_length = int(cfg["season_length"][()])
            if season_length == -1:
                season_length = None
            n_seas_comps = int(cfg["n_seas_comps"][()])
            if n_seas_comps == -1:
                n_seas_comps = None
            max_batch_size = int(cfg["max_batch_size"][()])
            if max_batch_size == -1:
                max_batch_size = None
            dma_pdr = float(cfg["dma_pdr"][()])
            dma_mdr = float(cfg["dma_mdr"][()])
            alias = cfg["alias"][()]
            if isinstance(alias, bytes):
                alias = alias.decode()
            fit_h_template = int(cfg["fit_h_template"][()])
            # Backward-compatible: universes written before warmup_steps
            # existed default to 0 (legacy OLS elicitation).
            warmup_steps = (
                int(cfg["warmup_steps"][()]) if "warmup_steps" in cfg else 0
            )
            # Adaptive (error-monitoring) universe — default False for universes
            # written before this existed.
            adaptive = bool(int(cfg["adaptive"][()])) if "adaptive" in cfg else False
            # Static-grid monitor — default None (off) for pre-monitor universes.
            monitor_tau = float(cfg["monitor_tau"][()]) if "monitor_tau" in cfg else None
            if monitor_tau is not None and monitor_tau < 0:
                monitor_tau = None
            # Custom-universe builder name (callable not persisted) — checked
            # below against the re-supplied ``universe_builder``.
            pad_batches = (bool(int(cfg["pad_batches"][()]))
                           if "pad_batches" in cfg else True)
            ub_name = cfg["universe_builder_name"][()] if "universe_builder_name" in cfg else ""
            ub_name = ub_name.decode() if isinstance(ub_name, bytes) else str(ub_name)
            tau_values = (
                list(np.asarray(cfg["tau_values"][()]))
                if "tau_values" in cfg and len(cfg["tau_values"][()]) > 0 else None
            )
            var_disc_values = (
                list(np.asarray(cfg["var_disc_values"][()]))
                if "var_disc_values" in cfg and len(cfg["var_disc_values"][()]) > 0 else None
            )
            freq_raw = cfg["freq"][()]
            freq = freq_raw.decode() if isinstance(freq_raw, bytes) else str(freq_raw)
            if freq == "":
                freq = None
            elif "freq_is_int" in cfg and bool(int(cfg["freq_is_int"][()])):
                freq = int(freq)

        uni = cls(
            path=path,
            season_length=season_length,
            n_seas_comps=n_seas_comps,
            dma_pdr=dma_pdr,
            dma_mdr=dma_mdr,
            max_batch_size=max_batch_size,
            warmup_steps=warmup_steps,
            alias=alias,
            dask_client=dask_client,
            adaptive=adaptive,
            tau_values=tau_values,
            var_disc_values=var_disc_values,
            monitor_tau=monitor_tau,
            universe_builder=universe_builder,
            exog_provider=exog_provider,
            pad_batches=pad_batches,
        )
        # Re-supply check: a custom-built universe needs its builder back.
        if ub_name:
            if universe_builder is None:
                raise ValueError(
                    f"Universe at {path} was built with a custom universe_builder "
                    f"'{ub_name}'; re-supply universe_builder=... to open() (a "
                    f"callable cannot be persisted, only its name).")
            supplied = f"{universe_builder.__module__}:{universe_builder.__qualname__}"
            if supplied != ub_name:
                warnings.warn(
                    f"universe_builder mismatch at {path}: built with '{ub_name}', "
                    f"opened with '{supplied}'. New/merged batches will use the "
                    f"supplied builder.")
        elif universe_builder is not None:
            warnings.warn(
                f"universe_builder supplied to open({path}) but the universe was "
                f"built with the standard FFS universe; ignoring is not possible — "
                f"new batches would diverge. Check this is intended.")
        uni._fit_h_template = fit_h_template
        uni._freq = freq
        uni._load_manifest()
        return uni

    # ------------------------------------------------------------------
    # Manifest / config persistence
    # ------------------------------------------------------------------

    def _config_path(self):
        return os.path.join(self.path, "config.h5")

    def _manifest_path(self):
        return os.path.join(self.path, "manifest.h5")

    def _batch_path(self, batch_id):
        return os.path.join(self.path, "batches", f"batch_{batch_id}.h5")

    def _save_config(self):
        with h5py.File(self._config_path(), "w") as f:
            cfg = f.create_group("config")
            for k, v in {
                "schema_version": _BATCH_SCHEMA_VERSION,
                "season_length": (
                    self.season_length if self.season_length is not None else -1
                ),
                "n_seas_comps": (
                    self.n_seas_comps if self.n_seas_comps is not None else -1
                ),
                "dma_pdr": float(self.dma_pdr),
                "dma_mdr": float(self.dma_mdr),
                "max_batch_size": (
                    self.max_batch_size if self.max_batch_size is not None else -1
                ),
                "alias": self.alias,
                "freq": (str(self._freq) if self._freq is not None else ""),
                "freq_is_int": int(isinstance(self._freq, (int, np.integer))),
                "fit_h_template": int(self._fit_h_template),
                "warmup_steps": (
                    int(self.warmup_steps) if self.warmup_steps is not None else 0
                ),
                "adaptive": int(self.adaptive),
                # static-grid monitor SD multiple (-1.0 -> None / off)
                "monitor_tau": (
                    float(self.monitor_tau) if self.monitor_tau is not None else -1.0
                ),
                # custom universe builder NAME only (callable can't be persisted;
                # re-supplied to open()). "" -> standard FFS universe.
                "pad_batches": int(self.pad_batches),
                "universe_builder_name": (
                    f"{self.universe_builder.__module__}:{self.universe_builder.__qualname__}"
                    if self.universe_builder is not None else ""
                ),
            }.items():
                cfg.create_dataset(k, data=v)
            # Variable-length adaptive grids (empty -> None on restore).
            cfg.create_dataset(
                "tau_values",
                data=(np.asarray(self.tau_values, dtype=float)
                      if self.tau_values is not None else np.empty(0)))
            cfg.create_dataset(
                "var_disc_values",
                data=(np.asarray(self.var_disc_values, dtype=float)
                      if self.var_disc_values is not None else np.empty(0)))

    def _save_manifest(self):
        df = self._manifest
        with h5py.File(self._manifest_path(), "w") as f:
            f.create_dataset("manifest_version", data=self._MANIFEST_VERSION)
            f.create_dataset("next_batch_id", data=self._next_batch_id)
            g = f.create_group("manifest")
            g.create_dataset(
                "unique_id",
                data=[str(s).encode("utf-8") for s in df.index],
            )
            g.create_dataset(
                "batch_id",
                data=np.asarray(df["batch_id"].values, dtype=np.int32),
            )
            # last_ds as int64 ns timestamps; -1 = no observations yet
            # Same int-vs-datetime polymorphism as in _save_batch_state.
            ts_ns = np.array(
                [
                    (
                        -1
                        if (
                            pd.isna(t)
                            if not isinstance(t, (int, np.integer))
                            else False
                        )
                        else (
                            int(t)
                            if isinstance(t, (int, np.integer))
                            else pd.Timestamp(t).value
                        )
                    )
                    for t in df["last_ds"].values
                ],
                dtype=np.int64,
            )
            g.create_dataset("last_ds", data=ts_ns)
            g.create_dataset(
                "active",
                data=np.asarray(df["active"].values, dtype=bool),
            )

    def _load_manifest(self):
        with h5py.File(self._manifest_path(), "r") as f:
            mv = int(f["manifest_version"][()])
            if mv != self._MANIFEST_VERSION:
                raise ValueError(
                    f"Manifest version {mv} != expected " f"{self._MANIFEST_VERSION}."
                )
            self._next_batch_id = int(f["next_batch_id"][()])
            g = f["manifest"]
            uids = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in g["unique_id"][()]
            ]
            batch_ids = np.asarray(g["batch_id"][()])
            last_ds_ns = np.asarray(g["last_ds"][()])
            active = np.asarray(g["active"][()], dtype=bool)

        last_ds = [
            (
                (pd.NaT if not isinstance(self._freq, (int, np.integer)) else -1)
                if v == -1
                else (
                    int(v)
                    if isinstance(self._freq, (int, np.integer))
                    else pd.Timestamp(int(v), unit="ns")
                )
            )
            for v in last_ds_ns
        ]
        self._manifest = pd.DataFrame(
            {
                "batch_id": batch_ids,
                "last_ds": last_ds,
                "active": active,
            },
            index=pd.Index(uids, name="unique_id"),
        )

    # ------------------------------------------------------------------
    # Manifest queries
    # ------------------------------------------------------------------

    def list_series(self, include_inactive: bool = False) -> list:
        """Return list of unique_ids in the universe."""
        if include_inactive:
            return self._manifest.index.tolist()
        return self._manifest[self._manifest["active"]].index.tolist()

    def is_active(self, unique_id) -> bool:
        """Whether ``unique_id`` is currently active (not removed) in the universe.

        Raises
        ------
        KeyError
            If ``unique_id`` is not part of the universe.
        """
        if unique_id not in self._manifest.index:
            raise KeyError(f"{unique_id!r} not in universe.")
        return bool(self._manifest.loc[unique_id, "active"])

    def n_batches(self) -> int:
        """Number of length-grouped batches the universe is partitioned into."""
        return self._next_batch_id

    def __repr__(self):
        n_active = (
            int(self._manifest["active"].sum()) if self._manifest is not None else 0
        )
        n_total = len(self._manifest) if self._manifest is not None else 0
        return (
            f"{type(self).__name__}(path={self.path!r}, "
            f"series={n_active}/{n_total} active, "
            f"batches={self._next_batch_id})"
        )

    # ------------------------------------------------------------------
    # Helpers shared with AutoFFS
    # ------------------------------------------------------------------

    def _min_filter_length(self):
        return (
            int(np.ceil(2 * self.season_length))
            if self.season_length is not None
            else 10
        )

    def _prepare_input(self, df, freq):
        # Reuse AutoFFS._prepare_input by constructing a stub. Cheaper
        # than copying the body here.
        stub = StaticFFS(
            season_length=self.season_length,
            n_seas_comps=self.n_seas_comps,
            dma_pdr=self.dma_pdr,
            dma_mdr=self.dma_mdr,
            max_batch_size=self.max_batch_size,
            alias=self.alias,
        )
        return stub._prepare_input(df, freq)

    def _new_batch_id(self):
        bid = self._next_batch_id
        self._next_batch_id += 1
        return bid

    def _materialise_exog(self, srs_ids, ds):
        """Build the exogenous-regressor design for ``srs_ids`` over calendar
        dates ``ds`` (length T) by calling ``self.exog_provider``.

        Returns ``(T, n_series, n_reg)`` float64, or ``None`` when no provider
        is configured (structural / AR universe). Assumes a shared calendar
        across the batch (rectangular panel) — the M5 streaming regime. ``ds``
        may be past dates (fit / update) or future horizon dates (forecast); the
        provider is agnostic to which, mapping (series, date) -> regressor value.
        """
        if self.exog_provider is None:
            return None
        srs_ids = list(srs_ids)
        ds = np.asarray(ds)
        # Capacity-padding placeholder slots (``_PAD_PREFIX``) are synthetic, not
        # real series, and are unknown to the exog provider; their forecasts are
        # discarded (inactive). Query the provider for real series only and
        # zero-fill the padding columns, keeping the array aligned to srs_ids.
        real_mask = np.array(
            [not str(u).startswith(_PAD_PREFIX) for u in srs_ids], dtype=bool
        )
        real_ids = [u for u, m in zip(srs_ids, real_mask) if m]
        arr_real = np.asarray(self.exog_provider(real_ids, ds), dtype=np.float64)
        if (arr_real.ndim != 3 or arr_real.shape[0] != len(ds)
                or arr_real.shape[1] != len(real_ids)):
            raise ValueError(
                f"exog_provider returned shape {arr_real.shape}; expected "
                f"(T={len(ds)}, n_series={len(real_ids)}, n_reg)."
            )
        if real_mask.all():
            return arr_real
        out = np.zeros((len(ds), len(srs_ids), arr_real.shape[-1]), dtype=np.float64)
        out[:, real_mask, :] = arr_real
        return out

    def _future_ds(self, last_ds, h):
        """The ``h`` calendar dates strictly after ``last_ds`` at the universe
        frequency, polymorphic over int (M1/M3-style) and datetime freqs. Used
        to ask the exog_provider for the known future regressor design."""
        return _future_ds_at(self._freq, last_ds, h)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, freq: Optional[str] = None, h_template: int = 18):
        """Fit the universe from a long-format DataFrame.

        Replaces any existing state. New series are partitioned into
        batches by length and chunked by ``max_batch_size``, then fit
        in parallel (sequentially or via Dask).

        Returns
        -------
        self
        """
        if len(self._manifest):
            raise RuntimeError(
                "Universe already contains series. To refit, create a "
                "new universe at a different path, or remove all "
                "existing batches manually."
            )

        per_series, sid_ds, sid_y, freq = self._prepare_input(df, freq)

        min_len = self._min_filter_length()
        _warn_under_seasonal(
            per_series.index[per_series["length"] < min_len],
            min_len, self.season_length)

        # Build batches
        stub = StaticFFS(
            season_length=self.season_length,
            n_seas_comps=self.n_seas_comps,
            dma_pdr=self.dma_pdr,
            dma_mdr=self.dma_mdr,
            max_batch_size=self.max_batch_size,
            alias=self.alias,
            dask_client=self.dask_client,
        )
        batches_input = list(stub._iter_batches(per_series, sid_y))

        # Assign batch IDs up front so files match manifest entries
        batch_ids = [self._new_batch_id() for _ in batches_input]

        args_list = [
            (
                tuple(srs_ids),
                arr,
                {sid: sid_ds[sid][-1] for sid in srs_ids},
                self.season_length,
                self.n_seas_comps,
                h_template,
                self.dma_pdr,
                self.dma_mdr,
                self.warmup_steps or 0,
                False,                 # include_ar (not used by the universe path)
                self.adaptive,
                self.tau_values,
                self.var_disc_values,
                self.monitor_tau,
                self.universe_builder,
                None,                  # component_priors
                None,                  # weight_override
                None,                  # error_nu0
                # exog_array: rows aligned to arr (shared batch calendar);
                # ds taken from the first series (rectangular panel). None when
                # no exog_provider is configured.
                self._materialise_exog(srs_ids, sid_ds[srs_ids[0]]),
            )
            for srs_ids, arr in batches_input
        ]
        states = stub._dispatch(_run_fit_batch, args_list)

        # Persist each batch + build manifest
        manifest_rows = []
        for bid, state in zip(batch_ids, states):
            for sid in state.srs_ids:          # real series only (before padding)
                manifest_rows.append(
                    {
                        "unique_id": sid,
                        "batch_id": bid,
                        "last_ds": state.last_ds_per_sid[sid],
                        "active": True,
                    }
                )
            padded, active = self._pad_state_to_capacity(state)
            _save_batch_state(padded, self._batch_path(bid), active=active)
        self._manifest = pd.DataFrame(manifest_rows).set_index("unique_id")
        self._freq = freq
        self._fit_h_template = h_template
        self._save_config()
        self._save_manifest()
        return self

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, df_new: pd.DataFrame):
        """Extend the universe with new observations.

        ``df_new`` must contain rows for at least one active series.
        Series omitted from ``df_new`` are left at their current state.
        Inactive series in ``df_new`` are silently skipped.

        Each affected batch is loaded, updated, and rewritten. Batches
        with no affected active series are not touched.

        Returns
        -------
        self
        """
        per_series, sid_ds, sid_y, _ = self._prepare_input(df_new, self._freq)

        # Filter to active series only
        active_mask = self._manifest["active"]
        active_ids = set(self._manifest[active_mask].index)
        provided = set(per_series.index)
        unknown = provided - set(self._manifest.index)
        if unknown:
            raise ValueError(
                f"update(df_new) contains {len(unknown)} unknown "
                f"unique_ids. Use add_series() for new series. "
                f"Examples: {sorted(unknown)[:5]}."
            )
        provided_active = provided & active_ids
        if not provided_active:
            raise ValueError("df_new contains no active series; nothing to update.")

        # Group provided active series by their batch_id
        affected_batches: dict = {}  # batch_id -> list of unique_ids
        for sid in provided_active:
            bid = int(self._manifest.loc[sid, "batch_id"])
            affected_batches.setdefault(bid, []).append(sid)

        # Per-period distributed step. The head reads only each affected batch's
        # cheap METADATA (series order, active mask, last-ds), builds the small
        # per-batch new-day payload (new_arr / new_last_ds / exog), and dispatches
        # the heavy load+filter+save to workers via _update_batch_file (which
        # reads and rewrites the batch on shared storage). Only KB-scale payloads
        # cross the wire. With dask_client=None the dispatch runs in-process,
        # with identical results.
        stub = StaticFFS(
            season_length=self.season_length, n_seas_comps=self.n_seas_comps,
            dma_pdr=self.dma_pdr, dma_mdr=self.dma_mdr,
            max_batch_size=self.max_batch_size, alias=self.alias,
            dask_client=self.dask_client,
        )
        args_list = []
        manifest_sets = []          # parallel: per-batch {sid: new last_ds} to write
        for bid, sids_in_update in affected_batches.items():
            srs_ids_b, active_arr, last_ds_b = _load_batch_meta(self._batch_path(bid))

            # All active slots must be present in df_new (partial updates are too
            # error-prone); inactive slots get a rectangular placeholder.
            active_in_batch = [s for s, a in zip(srs_ids_b, active_arr) if a]
            missing = set(active_in_batch) - set(sids_in_update)
            if missing:
                raise ValueError(
                    f"Batch {bid} has {len(missing)} active series "
                    f"missing from df_new. Within a batch, all active "
                    f"series must be updated together (same number of "
                    f"new observations). Missing: {sorted(missing)[:5]}."
                )

            # Validate temporal continuity. Polymorphic int/datetime.
            if isinstance(self._freq, (int, np.integer)):
                step = int(self._freq)
                for sid in active_in_batch:
                    expected = int(last_ds_b[sid]) + step
                    actual = int(sid_ds[sid][0])
                    if actual != expected:
                        raise ValueError(
                            f"Series {sid!r} in batch {bid}: first new ds "
                            f"is {actual}, expected {expected}."
                        )
            else:
                offset = pd.tseries.frequencies.to_offset(self._freq)
                for sid in active_in_batch:
                    expected = pd.Timestamp(last_ds_b[sid]) + offset
                    actual = pd.Timestamp(sid_ds[sid][0])
                    if actual != expected:
                        raise ValueError(
                            f"Series {sid!r} in batch {bid}: first new ds "
                            f"is {actual}, expected {expected}."
                        )

            new_lengths = {len(sid_y[sid]) for sid in active_in_batch}
            if len(new_lengths) > 1:
                raise ValueError(
                    f"Batch {bid}: new observations per series must "
                    f"have the same length; got {sorted(new_lengths)}."
                )
            T_new = new_lengths.pop()

            # New array in srs_ids order; inactive slots get a zero placeholder
            # (their result is discarded but the batch must stay rectangular).
            new_arr_cols = []
            new_last_ds = {}
            for sid, a in zip(srs_ids_b, active_arr):
                if a:
                    new_arr_cols.append(sid_y[sid])
                    new_last_ds[sid] = sid_ds[sid][-1]
                else:
                    new_arr_cols.append(np.zeros(T_new, dtype=np.float64))
                    if isinstance(self._freq, (int, np.integer)):
                        new_last_ds[sid] = int(last_ds_b[sid]) + T_new * int(self._freq)
                    else:
                        offset = pd.tseries.frequencies.to_offset(self._freq)
                        new_last_ds[sid] = pd.Timestamp(last_ds_b[sid]) + T_new * offset
            new_arr = np.column_stack(new_arr_cols)

            # Exogenous design for the new days, srs_ids-ordered (shared calendar
            # across the batch -> ds from the first active series).
            exog_new = self._materialise_exog(
                srs_ids_b, sid_ds[active_in_batch[0]]
            )
            args_list.append(
                (self._batch_path(bid), new_arr, new_last_ds, exog_new,
                 self.dma_pdr, self.dma_mdr, 0)
            )
            manifest_sets.append({sid: sid_ds[sid][-1] for sid in active_in_batch})

        # Distribute the heavy load+filter+save across workers (in-process if no
        # client). _update_batch_file returns the active {sid: last_ds} per batch.
        stub._dispatch(_update_batch_file, args_list)
        for mset in manifest_sets:
            for sid, lds in mset.items():
                self._manifest.loc[sid, "last_ds"] = lds

        self._save_manifest()
        return self

    # ------------------------------------------------------------------
    # forecast
    # ------------------------------------------------------------------

    def forecast(
        self,
        h: int,
        level: Optional[list] = None,
        return_components: bool = False,
    ) -> pd.DataFrame:
        """h-step forecast for all active series in the universe.

        ``return_components=True`` returns the raw PER-MODEL predictive instead of
        the DMA-combined DataFrame: a dict with ``model_keys`` / ``model_class``
        (length nm, shared across batches), ``series_ids`` (S active), and arrays
        ``loc``/``sd`` ``(nm, S, h)``, ``nu``/``weights`` ``(nm, S)`` — the
        per-(model, series, horizon) Student-t predictive. Diagnostic only (the
        nm axis is large); not for full-panel production."""
        if not isinstance(h, int) or h <= 0:
            raise ValueError("h must be a positive integer.")
        stub = StaticFFS(
            season_length=self.season_length,
            n_seas_comps=self.n_seas_comps,
            dma_pdr=self.dma_pdr,
            dma_mdr=self.dma_mdr,
            max_batch_size=self.max_batch_size,
            alias=self.alias,
            dask_client=self.dask_client,
        )
        level = stub._normalise_level(level)

        # Path-based distributed forecast: the head reads each batch's cheap
        # metadata (series order, active mask, last-ds) to build the exog future
        # design, then dispatches the heavy load+forecast to workers via
        # _predict_batch_file (state loaded on the worker from shared storage, not
        # shipped from the head). With dask_client=None this runs in-process.
        # (Inactive-only series carry batch_id -1 after defragment.)
        active_man = self._manifest[self._manifest["active"]]
        batch_ids = sorted(set(active_man["batch_id"].astype(int)))
        metas = []          # parallel to args_list: (srs_ids, active, last_ds)
        args_list = []
        for bid in batch_ids:
            srs_ids_b, active_arr, last_ds_b = _load_batch_meta(self._batch_path(bid))
            exog_future = None
            if self.exog_provider is not None:
                act_sid = next(s for s, a in zip(srs_ids_b, active_arr) if a)
                future_ds = self._future_ds(last_ds_b[act_sid], h)
                exog_future = self._materialise_exog(srs_ids_b, future_ds)
            metas.append((srs_ids_b, active_arr, last_ds_b))
            args_list.append(
                (self._batch_path(bid), h, self.sd_method, exog_future,
                 self.dma_pdr, self.dma_mdr)
            )
        # results[i] = (pred, srs_ids, active, mdl_keys, model_indicator)
        results = stub._dispatch(_predict_batch_file, args_list)

        if return_components:
            # Per-model predictive (no DMA combination), active series only.
            # model_keys/model_class are shared across batches (same universe).
            model_keys = model_class = None
            loc_l, sd_l, nu_l, w_l, sids = [], [], [], [], []
            for (srs_ids_b, active_arr, _lds), (pred, _s, _a, mdl_keys, mi) in zip(
                metas, results
            ):
                am = np.asarray(active_arr, dtype=bool)
                loc_l.append(np.asarray(pred.f_h)[:, am, :])          # (nm, S, h)
                sd_l.append(np.sqrt(np.asarray(pred.q_h)[:, am, :]))  # (nm, S, h)
                nu_l.append(np.asarray(pred.nu)[:, am])               # (nm, S)
                w_l.append(np.asarray(pred.weights)[:, am])           # (nm, S)
                sids.extend([s for s, a in zip(srs_ids_b, am) if a])
                if model_keys is None:
                    model_keys = list(mdl_keys)
                    cols = list(mi.columns)
                    model_class = [cols[j] for j in np.asarray(mi.values).argmax(axis=1)]
            return {
                "model_keys": model_keys,
                "model_class": model_class,
                "series_ids": sids,
                "loc": np.concatenate(loc_l, axis=1),     # (nm, S, h)
                "sd": np.concatenate(sd_l, axis=1),       # (nm, S, h)
                "nu": np.concatenate(nu_l, axis=1),       # (nm, S)
                "weights": np.concatenate(w_l, axis=1),   # (nm, S)
            }

        # Assemble long-format output, skipping inactive slots
        is_int_mode = isinstance(self._freq, (int, np.integer))
        if not is_int_mode:
            offset = pd.tseries.frequencies.to_offset(self._freq)
        rows = []
        for (srs_ids_b, active_arr, last_ds_b), (pred, _s, _a, _mk, _mi) in zip(
            metas, results
        ):
            if level:
                bounds = _t_quantile_average(pred, level)
            for j, sid in enumerate(srs_ids_b):
                if not active_arr[j]:
                    continue
                last_ds = last_ds_b[sid]
                if is_int_mode:
                    step = int(self._freq)
                    future_ds = np.arange(
                        int(last_ds) + step,
                        int(last_ds) + step * (h + 1),
                        step,
                    )
                else:
                    future_ds = pd.date_range(
                        start=pd.Timestamp(last_ds) + offset,
                        periods=h,
                        freq=self._freq,
                    )
                mu = pred.loc[j]
                sd = pred.sd[j]  # (h,) — LoTV mixture SD
                for k in range(h):
                    row = {
                        "unique_id": sid,
                        "ds": future_ds[k],
                        self.alias: float(mu[k]),
                        f"{self.alias}-sd": float(sd[k]),
                    }
                    if level:
                        for L in level:
                            lo, hi = bounds[L]
                            row[f"{self.alias}-lo-{L}"] = float(lo[k, j])
                            row[f"{self.alias}-hi-{L}"] = float(hi[k, j])
                    rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # add_series
    # ------------------------------------------------------------------

    def _fit_singleton(self, unique_id, df_history,
                       component_priors=None, weight_override=None, error_nu0=None):
        """Validate, prepare and fit ``unique_id`` into a singleton batch.

        Returns ``(state, last_ds)`` for the fitted one-series
        :class:`_BatchState`. Does not touch the manifest or write any
        file. Shared by :meth:`add_series` and
        :meth:`_add_series_singleton`.

        ``component_priors`` / ``weight_override`` (default None) seed an
        informative state / DMA-weight prior for the new series instead of the
        diffuse prior + uniform weights.
        """
        if unique_id in self._manifest.index:
            raise ValueError(
                f"unique_id {unique_id!r} already in universe. To "
                f"replace, remove_series() then add_series()."
            )
        # Build a one-series long-format frame and re-use _prepare_input
        df = df_history.copy()
        df["unique_id"] = unique_id
        df = df[["unique_id", "ds", "y"]]
        per_series, sid_ds, sid_y, _ = self._prepare_input(df, self._freq)

        min_len = self._min_filter_length()
        if per_series.iloc[0]["length"] < min_len:
            _warn_under_seasonal(
                per_series.index[:1], min_len, self.season_length)

        arr = sid_y[unique_id][:, None]  # (T, 1)
        last_ds = sid_ds[unique_id][-1]
        # Exogenous regressor design over this series' own history dates.
        exog_array = self._materialise_exog((unique_id,), sid_ds[unique_id])
        state = _run_fit_batch(
            (unique_id,),
            arr,
            {unique_id: last_ds},
            self.season_length,
            self.n_seas_comps,
            self._fit_h_template,
            self.dma_pdr,
            self.dma_mdr,
            self.warmup_steps or 0,
            component_priors=component_priors,
            weight_override=weight_override,
            error_nu0=error_nu0,
            adaptive=self.adaptive,
            tau_values=self.tau_values,
            var_disc_values=self.var_disc_values,
            monitor_tau=self.monitor_tau,
            universe_builder=self.universe_builder,
            exog_array=exog_array,
        )
        return state, last_ds

    def _add_series_singleton(self, unique_id, df_history: pd.DataFrame):
        """Add a new series in its OWN singleton batch (legacy behaviour).

        Identical to the pre-append ``add_series``: existing batches are
        untouched and the fitted series is written as a fresh singleton
        batch, to be consolidated later via :meth:`defragment`. Retained
        as the equivalence reference for :meth:`add_series` and as a
        fallback. See :meth:`add_series` for the append-on-add path.
        """
        state, last_ds = self._fit_singleton(unique_id, df_history)
        bid = self._new_batch_id()
        _save_batch_state(state, self._batch_path(bid))

        new_row = pd.DataFrame(
            {
                "batch_id": [bid],
                "last_ds": [last_ds],
                "active": [True],
            },
            index=pd.Index([unique_id], name="unique_id"),
        )
        self._manifest = pd.concat([self._manifest, new_row])
        self._save_manifest()
        return self

    def _open_batch_id(self, unique_id=None):
        """Return the id of the current "open" batch, or ``None``.

        The open batch is the live (active) batch whose current active
        series count is ``< max_batch_size`` — i.e. it has room for one
        more series. ``add_series`` appends into it instead of spawning a
        singleton.

        * If ``max_batch_size is None`` the single consolidated batch is
          always open (capacity is unbounded), so the sole live batch id
          is returned (or ``None`` if the universe is empty).
        * Otherwise the live batches are scanned and the first one with
          fewer than ``max_batch_size`` active series is returned. If
          every live batch is full (or there are none), ``None`` is
          returned and the caller starts a fresh batch.
        """
        man = self._manifest
        if man is None or not len(man):
            return None
        active = man[man["active"]]
        if not len(active):
            return None
        # Count active series per live batch.
        counts = active.groupby("batch_id").size()
        if self.max_batch_size is None:
            # Unbounded capacity: the (single) live batch is always open.
            # Pick the live batch with the most series — under append-on-add
            # there is exactly one, but be robust to a legacy singleton swarm.
            return int(counts.idxmax())
        for bid, n in counts.items():
            if int(n) < self.max_batch_size:
                return int(bid)
        return None

    def add_series(self, unique_id, df_history: pd.DataFrame,
                   component_priors=None, weight_override=None, error_nu0=None):
        """Add a new series to the universe (append-on-add).

        ``df_history`` must contain columns ``ds``, ``y`` (a single-
        series DataFrame; ``unique_id`` is supplied separately and need
        not be a column).

        ``component_priors`` (dict ``name -> (m0, C0)``) and ``weight_override``
        (dict with ``pset``/``mset``) seed an informative hierarchical prior /
        DMA-weight prior for the new series. Both default to None, giving the
        diffuse-prior + uniform-weight behaviour unchanged.

        The new series is fit exactly as before (its warmup is applied
        in ``_run_fit_batch``); its post-fit state is then **concatenated
        into the current "open" batch** — the live batch with room for
        one more series — which is rebuilt in place. This is
        O(open batch ≤ ``max_batch_size``) rather than O(universe): no
        ``defragment`` is needed to keep the streaming layout compact.
        When the open batch fills to ``max_batch_size`` the next add
        starts a fresh batch.

        A series shorter than ``min_len`` (2 seasonal cycles) is not
        rejected: it is added with a non-blocking warning and its seasonal
        component leans on the diffuse prior.

        Raises
        ------
        ValueError
            If ``unique_id`` already exists (active or inactive) in the
            universe.
        """
        state, last_ds = self._fit_singleton(
            unique_id, df_history,
            component_priors=component_priors, weight_override=weight_override,
            error_nu0=error_nu0,
        )

        open_bid = self._open_batch_id(unique_id)
        if open_bid is None:
            # No room (or empty universe): persist as a fresh batch.
            bid = self._new_batch_id()
            padded, active = self._pad_state_to_capacity(state)
            _save_batch_state(padded, self._batch_path(bid), active=active)
        else:
            # Append into the open batch: gather its live slots plus the
            # new singleton, rebuild that one batch, reuse its id (so the
            # existing series' manifest rows are unchanged).
            bid = open_bid
            cur, cur_act = _load_batch_state(self._batch_path(bid))
            _attach_dma_step_fn(cur.dma, self.dma_pdr, self.dma_mdr)
            new_act = np.array([True])
            merged, srs, last_ds_map, shared, max_T, _ = self._gather_live(
                [(cur, cur_act), (state, new_act)]
            )
            last_ds_map[unique_id] = last_ds
            st = self._build_batch(
                merged, srs, last_ds_map, shared, max_T, slice(0, len(srs))
            )
            st, active = self._pad_state_to_capacity(st)
            _save_batch_state(st, self._batch_path(bid), active=active)

        new_row = pd.DataFrame(
            {
                "batch_id": [bid],
                "last_ds": [last_ds],
                "active": [True],
            },
            index=pd.Index([unique_id], name="unique_id"),
        )
        self._manifest = pd.concat([self._manifest, new_row])
        self._save_manifest()
        return self

    def add_series_many(self, unique_ids, df_histories,
                        component_priors=None, weight_override=None,
                        error_nu0=None):
        """Add several new series in one batched append (semi-vectorised).

        Equivalent to calling :meth:`add_series` once per series, but does a
        SINGLE open-batch rebuild + save per affected batch instead of one per
        series. A ``k``-series origin therefore costs O(1) batch writes, not
        O(k) — the per-series ``_fit_singleton`` (cheap) still runs in a loop,
        but the expensive ``_gather_live`` + ``_build_batch`` + save (which
        re-writes a whole ``max_batch_size`` batch over shared storage) happens
        once. This removes the O(n_add) open-batch-rewrite cost that dominates
        heavy-add streaming origins.

        Parameters mirror :meth:`add_series` but are PER-SERIES sequences of
        equal length ``k``:

        unique_ids : sequence of ``k`` ids (all new, all distinct).
        df_histories : sequence of ``k`` single-series frames (columns
            ``ds``, ``y``).
        component_priors / weight_override / error_nu0 : ``None`` (diffuse for
            all) or a length-``k`` list whose ``i``-th entry is the prior for
            ``unique_ids[i]`` (each entry itself ``None`` or the per-series
            object :meth:`add_series` accepts).

        The result is **bit-identical** to the equivalent sequential
        ``add_series`` calls: each series is fit independently, and
        ``_gather_live`` / ``_build_batch`` are pure series-axis
        slice/concatenate/reshape (no recomputation), so gathering all at once
        equals gathering incrementally. Assumes the new series share a common
        history span (as the streaming loop adds them at one origin); their
        ``T_ingested`` is taken as the max across the batched set, matching the
        sequential outcome when those spans are equal.

        Raises
        ------
        ValueError
            If a ``unique_id`` already exists, the inputs disagree in length,
            or an id is duplicated within the call.
        """
        unique_ids = list(unique_ids)
        df_histories = list(df_histories)
        k = len(unique_ids)
        if k != len(df_histories):
            raise ValueError(
                f"unique_ids ({k}) and df_histories ({len(df_histories)}) "
                f"must have the same length.")
        if k == 0:
            return self

        def _pick(seq, i):
            return seq[i] if seq is not None else None

        if k == 1:
            # Single add: the scalar path is identical and avoids the gather.
            return self.add_series(
                unique_ids[0], df_histories[0],
                component_priors=_pick(component_priors, 0),
                weight_override=_pick(weight_override, 0),
                error_nu0=_pick(error_nu0, 0),
            )
        for seq, name in ((component_priors, "component_priors"),
                          (weight_override, "weight_override"),
                          (error_nu0, "error_nu0")):
            if seq is not None and len(seq) != k:
                raise ValueError(
                    f"{name} has length {len(seq)}, expected {k}.")

        seen: set = set()
        for uid in unique_ids:
            if uid in self._manifest.index:
                raise ValueError(
                    f"unique_id {uid!r} already in universe. To replace, "
                    f"remove_series() then add_series().")
            if uid in seen:
                raise ValueError(f"unique_id {uid!r} duplicated in add_series_many.")
            seen.add(uid)

        # 1. Fit each new series as a singleton (per-series priors). Cheap; the
        #    expensive batch rewrite is deferred to one pass below.
        singletons = []                 # list of (state, active_mask)
        new_last_ds: dict = {}
        for i, uid in enumerate(unique_ids):
            state, last_ds = self._fit_singleton(
                uid, df_histories[i],
                component_priors=_pick(component_priors, i),
                weight_override=_pick(weight_override, i),
                error_nu0=_pick(error_nu0, i),
            )
            singletons.append((state, np.array([True])))
            new_last_ds[uid] = last_ds

        # 2. Gather [current open batch (if any)] + all new singletons, then
        #    split into max_batch_size chunks (defragment's pattern). The first
        #    chunk reuses the open batch id so existing series stay put; the
        #    rest get fresh ids.
        open_bid = self._open_batch_id()
        states_acts = []
        if open_bid is not None:
            cur, cur_act = _load_batch_state(self._batch_path(open_bid))
            _attach_dma_step_fn(cur.dma, self.dma_pdr, self.dma_mdr)
            states_acts.append((cur, cur_act))
        states_acts.extend(singletons)

        merged, srs, last_ds, shared, max_T, _ = self._gather_live(states_acts)
        last_ds.update(new_last_ds)
        N = len(srs)

        # 3. Write the chunks. cap=max_batch_size (None -> one batch).
        cap = N if self.max_batch_size is None else self.max_batch_size
        assign: dict = {}
        reuse = open_bid is not None
        for c0 in range(0, N, cap):
            sl = slice(c0, min(c0 + cap, N))
            if reuse:
                bid = open_bid
                reuse = False
            else:
                bid = self._new_batch_id()
            st = self._build_batch(merged, srs, last_ds, shared, max_T, sl)
            for s in st.srs_ids:
                assign[s] = bid
            st, active = self._pad_state_to_capacity(st)
            _save_batch_state(st, self._batch_path(bid), active=active)

        # 4. Manifest: append rows for the new series; existing series in the
        #    reused open batch keep their id (no-op reassignment is skipped).
        new_rows = pd.DataFrame(
            {
                "batch_id": [assign[uid] for uid in unique_ids],
                "last_ds": [new_last_ds[uid] for uid in unique_ids],
                "active": [True] * k,
            },
            index=pd.Index(unique_ids, name="unique_id"),
        )
        self._manifest = pd.concat([self._manifest, new_rows])
        self._save_manifest()
        return self

    # ------------------------------------------------------------------
    # remove_series
    # ------------------------------------------------------------------

    def remove_series(self, unique_id):
        """Mark a series inactive. Cheap (manifest write only)."""
        if unique_id not in self._manifest.index:
            raise KeyError(f"{unique_id!r} not in universe.")
        if not self._manifest.loc[unique_id, "active"]:
            return self  # already inactive, no-op
        self._manifest.loc[unique_id, "active"] = False

        # Also flip the active flag inside the relevant batch file so
        # update() will treat the slot as inactive next time
        bid = int(self._manifest.loc[unique_id, "batch_id"])
        state, active_arr = _load_batch_state(self._batch_path(bid))
        _attach_dma_step_fn(state.dma, self.dma_pdr, self.dma_mdr)
        slot = state.srs_ids.index(unique_id)
        active_arr[slot] = False
        _save_batch_state_with_active(state, self._batch_path(bid), active_arr)

        self._save_manifest()
        return self

    # ------------------------------------------------------------------
    # defragment
    # ------------------------------------------------------------------

    # Per-(model, series) state, stored flattened model-major as (nm*q, ...).
    _BCAST_ATTRS = (
        "disc_rates", "disc_rates_damped", "variance_disc",
        "variance_power", "mult_comps", "monitor", "monitor_inject",
    )

    def _gather_live(self, states_acts):
        """Merge the live per-series state across a set of batches.

        ``states_acts`` is a list of ``(_BatchState, active_mask)`` pairs
        (``active_mask`` a bool array parallel to ``state.srs_ids``). All
        batches must share the same model structure (``nm``, ``k``, model
        set); per-(model, series) arrays — ``dlm_state``, the broadcast
        discount/variance/mult attributes, and the allocator
        ``pset``/``mset``/``forecast_history``/``latest_weights`` — are
        sliced to live slots and concatenated along the series axis.
        ``F``/``G``/``GH`` and the ``model_indicator`` are per-model and
        shared.

        Returns
        -------
        merged : dict
            ``("dlm", key)`` / ``("bc", key)`` -> (nm, N, ...) arrays, plus
            ``"pset"`` (nm, N, ...), ``"mset"`` (nm, N, ...),
            ``"fh"`` (h, nm, N, ...) and ``"lw"`` (nm, N, 1).
        srs : list
            Ordered live series ids (length N).
        last_ds : dict
            ``sid -> last ds`` for every live series.
        shared : dict
            Model-level objects shared by every output batch
            (``nm``, ``k``, ``npad``, ``mdl_keys``, ``F``, ``G``, ``GH``,
            ``model_indicator``, ``dlm_keys``).
        max_T : int
            Max ``T_ingested`` across the gathered batches.
        n_slots_freed : int
            Total inactive slots dropped.
        """
        acc: dict = {}          # ("dlm"/"bc", key) -> list of (nm, q_i, ...) arrays
        pset_l, mset_l, fh_l, lw_l = [], [], [], []
        srs: list = []
        last_ds: dict = {}
        nm = k = npad = mdl_keys = F = G = GH = model_indicator = dlm_keys = None
        max_T = 0
        n_slots_freed = 0

        for state, act in states_acts:
            act = np.asarray(act, dtype=bool)
            m = state.multi
            if nm is None:
                nm, k, npad = m.nm, m.k, m.npad
                mdl_keys = m.mdl_keys
                F, G = np.asarray(m.F), np.asarray(m.G)
                GH = np.asarray(m.GH) if m.GH is not None else None
                model_indicator = state.model_indicator
                dlm_keys = list(m.dlm_state.keys())
                # Regression-tail descriptors (shared across the merge). reg_mask
                # is per-(model, series); collapse to the per-MODEL flag (nm,) so
                # _build_batch can rebuild it for any merged series count.
                n_regressors = int(getattr(m, "n_regressors", 0))
                exog_regressors = bool(getattr(m, "exog_regressors", False))
                if n_regressors > 0 and getattr(m, "reg_mask", None) is not None:
                    reg_is_model = np.asarray(m.reg_mask).reshape(nm, m.q)[:, 0]
                else:
                    reg_is_model = None
            q = m.q
            max_T = max(max_T, int(state.T_ingested))
            n_slots_freed += int((~act).sum())
            for key in dlm_keys:
                arr = np.asarray(m.dlm_state[key])
                acc.setdefault(("dlm", key), []).append(
                    arr.reshape(nm, q, *arr.shape[1:])[:, act, ...])
            for key in self._BCAST_ATTRS:
                arr = np.asarray(getattr(m, key))
                acc.setdefault(("bc", key), []).append(
                    arr.reshape(nm, q, *arr.shape[1:])[:, act, ...])
            pset_l.append(np.asarray(state.dma.state.pset)[:, act, ...])
            mset_l.append(np.asarray(state.dma.state.mset)[:, act, ...])
            fh_l.append(np.asarray(state.dma.state.forecast_history)[:, :, act, ...])
            lw_l.append(np.asarray(state.latest_weights)[:, act, ...])
            sids = [s for s, a in zip(state.srs_ids, act) if a]
            srs.extend(sids)
            for s in sids:
                last_ds[s] = state.last_ds_per_sid[s]

        merged = {kk: np.concatenate(v, axis=1) for kk, v in acc.items()}  # (nm, N, ...)
        merged["pset"] = np.concatenate(pset_l, axis=1)
        merged["mset"] = np.concatenate(mset_l, axis=1)
        merged["fh"] = np.concatenate(fh_l, axis=2)
        merged["lw"] = np.concatenate(lw_l, axis=1)
        shared = {
            "nm": nm, "k": k, "npad": npad, "mdl_keys": mdl_keys,
            "F": F, "G": G, "GH": GH,
            "model_indicator": model_indicator, "dlm_keys": dlm_keys,
            "n_regressors": n_regressors, "exog_regressors": exog_regressors,
            "reg_is_model": reg_is_model,
        }
        return merged, srs, last_ds, shared, max_T, n_slots_freed

    def _pad_state_to_capacity(self, state, active=None):
        """Pad a batch's series axis to ``max_batch_size`` with INACTIVE
        placeholder slots (replicating slot 0's per-(model,series) state) so
        every batch shares one shape and the jitted filter/forecast compile
        once — eliminating the per-add XLA recompilation (and its unbounded
        host-RAM cache growth) on long streaming rolls.

        Returns ``(state, active)`` with ``state`` mutated in place. No-op when
        ``max_batch_size is None`` or the batch is already at capacity, so the
        padded slots never affect results (they are masked out by ``active``
        everywhere: update fills them with a placeholder and discards the
        result; forecast skips them; the DMA layer reduces over models, never
        across series).
        """
        cap = self.max_batch_size if getattr(self, "pad_batches", True) else None
        n = len(state.srs_ids)
        if active is None:
            active = np.ones(n, dtype=bool)
        else:
            active = np.asarray(active, dtype=bool)
        if cap is None or n >= cap:
            return state, active
        pad = cap - n
        m = state.multi
        nm = m.nm

        def pad_flat(arr):                       # (nm*n, ...) model-major -> (nm*cap, ...)
            arr = np.asarray(arr)
            a = arr.reshape(nm, n, *arr.shape[1:])
            rep = np.repeat(a[:, :1, ...], pad, axis=1)
            return np.concatenate([a, rep], axis=1).reshape(nm * cap, *arr.shape[1:])

        def pad_ax(arr, ax):                     # replicate index 0 along series axis
            arr = np.asarray(arr)
            rep = np.repeat(np.take(arr, [0], axis=ax), pad, axis=ax)
            return np.concatenate([arr, rep], axis=ax)

        m.dlm_state = {k: pad_flat(v) for k, v in m.dlm_state.items()}
        for key in self._BCAST_ATTRS:
            setattr(m, key, pad_flat(getattr(m, key)))
        # reg_mask is per-(model, series) but NOT in _BCAST_ATTRS; rebuild it for
        # the padded series count from the per-model flag (constant across series).
        if getattr(m, "n_regressors", 0) > 0 and getattr(m, "reg_mask", None) is not None:
            rm = np.asarray(m.reg_mask).reshape(nm, n)[:, 0]
            m.reg_mask = device_put(
                jnp.asarray(np.repeat(rm, cap)), devices.dlm_compute)
        m.q = cap
        m.p = nm * cap

        ds = state.dma.state
        state.dma.state = ds._replace(
            pset=pad_ax(ds.pset, 1), mset=pad_ax(ds.mset, 1),
            forecast_history=pad_ax(ds.forecast_history, 2),
        )
        state.latest_weights = pad_ax(state.latest_weights, 1)

        ref_ds = state.last_ds_per_sid[state.srs_ids[0]]
        pad_ids = [f"{_PAD_PREFIX}{i}" for i in range(pad)]
        state.srs_ids = tuple(list(state.srs_ids) + pad_ids)
        for pid in pad_ids:
            state.last_ds_per_sid[pid] = ref_ds
        if state.lag_tail is not None:           # AR/regressor tail: pad its series axis if present
            lt = np.asarray(state.lag_tail)
            axes = [ax for ax, s in enumerate(lt.shape) if s == n]
            if len(axes) == 1:
                state.lag_tail = pad_ax(lt, axes[0])

        active = np.concatenate([active, np.zeros(pad, dtype=bool)])
        return state, active

    def _build_batch(self, merged, srs, last_ds, shared, max_T, sl):
        """Build one ``_BatchState`` from gathered live state for slice ``sl``.

        ``merged``/``srs``/``last_ds``/``shared``/``max_T`` come from
        :meth:`_gather_live`. ``sl`` selects a contiguous run of series
        (typically one ``max_batch_size`` chunk, or the whole gathered set
        when appending one series).
        """
        nm = shared["nm"]
        k = shared["k"]
        npad = shared["npad"]
        mdl_keys = shared["mdl_keys"]
        F, G, GH = shared["F"], shared["G"], shared["GH"]
        model_indicator = shared["model_indicator"]
        dlm_keys = shared["dlm_keys"]

        qn = sl.stop - sl.start

        multi = multi_model_dlm.__new__(multi_model_dlm)
        multi.dlm_compute = devices.dlm_compute
        multi.nm, multi.k, multi.npad, multi.mdl_keys = nm, k, npad, mdl_keys
        multi.q, multi.p = qn, nm * qn
        multi.F, multi.G, multi.GH = F, G, GH
        # Regression-tail descriptors (carried across merge / defragment so the
        # rebuilt batch keeps filtering/forecasting with regressors). reg_mask is
        # rebuilt for this slice's series count from the per-model flag.
        multi.n_regressors = int(shared.get("n_regressors", 0))
        multi.exog_regressors = bool(shared.get("exog_regressors", False))
        reg_is_model = shared.get("reg_is_model")
        if multi.n_regressors > 0 and reg_is_model is not None:
            multi.reg_mask = device_put(
                jnp.asarray(np.repeat(reg_is_model, qn)), devices.dlm_compute)
        else:
            multi.reg_mask = device_put(
                jnp.zeros(nm * qn, dtype=bool), devices.dlm_compute)
        for key in self._BCAST_ATTRS:
            a = merged[("bc", key)][:, sl, ...]
            setattr(multi, key, a.reshape(nm * qn, *a.shape[2:]))
        multi.dlm_state = {}
        for key in dlm_keys:
            a = merged[("dlm", key)][:, sl, ...]
            multi.dlm_state[key] = a.reshape(nm * qn, *a.shape[2:])

        dma = Allocator.__new__(Allocator)
        dma.state = AllocatorState(
            pset=merged["pset"][:, sl, ...], mset=merged["mset"][:, sl, ...],
            forecast_history=merged["fh"][:, :, sl, ...])
        dma.mi = np.asarray(model_indicator.values)
        dma.device = devices.allocation_compute

        sids = srs[sl]
        return _BatchState(
            srs_ids=tuple(sids), multi=multi, dma=dma,
            model_indicator=model_indicator,
            latest_weights=merged["lw"][:, sl, ...],
            last_ds_per_sid={s: last_ds[s] for s in sids},
            T_ingested=max_T,
        )

    def defragment(self, inactive_fraction_threshold: float = 0.25):
        """Consolidate active series into ``max_batch_size`` batches.

        Merges the live per-series state across all current batches into a
        minimal batch set (one batch when ``max_batch_size`` is None),
        dropping inactive slots. No refit and no retained history: the
        filtered state is concatenated along the series axis, so forecasts
        are unchanged **bit-for-bit**. This keeps streaming workloads
        efficient once ``add_series`` has spawned many singleton batches.

        All batches share the same model structure (``nm``, ``k``, model
        set), so per-(model, series) arrays — ``dlm_state`` (m/UC/SC/s/nu),
        the broadcast discount/variance/mult attributes, and the allocator
        ``pset``/``mset``/``forecast_history``/``latest_weights`` — merge by
        concatenation along the series axis. ``F``/``G``/``GH`` and the
        ``model_indicator`` are per-model and shared.

        ``inactive_fraction_threshold`` is retained for API compatibility;
        consolidation always rebuilds to the optimal layout.

        Returns
        -------
        dict  ``{n_batches_before, n_batches_after, n_slots_freed}``.
        """
        man = self._manifest
        if man is None or not len(man):
            return {"n_batches_before": 0, "n_batches_after": 0, "n_slots_freed": 0}

        n_active = int(man["active"].sum())
        bids_before = sorted({int(b) for b in man.loc[man["active"], "batch_id"]})
        target = 1 if self.max_batch_size is None else int(
            np.ceil(n_active / self.max_batch_size))

        # Already optimal (few batches) and nothing inactive to reclaim -> no-op.
        if 0 < len(bids_before) <= target:
            inactive = 0
            for bid in bids_before:
                _, act = _load_batch_state(self._batch_path(bid))
                inactive += int((~act).sum())
            if inactive == 0:
                return {"n_batches_before": len(bids_before),
                        "n_batches_after": len(bids_before), "n_slots_freed": 0}

        # --- gather live slots across all active batches ---
        states_acts = []
        for bid in bids_before:
            state, act = _load_batch_state(self._batch_path(bid))
            states_acts.append((state, act))
        merged, srs, last_ds, shared, max_T, n_slots_freed = self._gather_live(
            states_acts)
        N = len(srs)

        # --- write consolidated batches with fresh ids, update manifest ---
        chunk = N if self.max_batch_size is None else self.max_batch_size
        new_assign: dict = {}
        new_bids: list = []
        for c0 in range(0, N, chunk):
            sl = slice(c0, min(c0 + chunk, N))
            bid = self._new_batch_id()
            new_bids.append(bid)

            st = self._build_batch(merged, srs, last_ds, shared, max_T, sl)
            for s in st.srs_ids:           # real series only (before padding)
                new_assign[s] = bid
            st, active = self._pad_state_to_capacity(st)
            _save_batch_state(st, self._batch_path(bid), active=active)

        for s, bid in new_assign.items():
            self._manifest.loc[s, "batch_id"] = bid
        # Inactive series no longer have a live batch.
        self._manifest.loc[~self._manifest["active"], "batch_id"] = -1
        for bid in bids_before:
            p = self._batch_path(bid)
            if os.path.exists(p):
                os.remove(p)
        self._save_manifest()

        return {"n_batches_before": len(bids_before),
                "n_batches_after": len(new_bids),
                "n_slots_freed": n_slots_freed}


# =====================================================================
# Block-based orchestrators
# =====================================================================
# AutoFFS and AutoFFSUniverse drive a list of Blocks combined by a top-level
# DMA. The pre-block implementations remain available as StaticFFS and
# StaticFFSUniverse, both for reproducing earlier results and as the
# reference the block path is checked against.


def _season_length_from_blocks(blocks):
    """The orchestration season length for a block list: the longest seasonal
    period across the blocks (``StaticBlock.season_length`` / ``GridBlock.period``),
    used for the min-length gate and under-seasonal warning. ``None`` if no block
    carries a period."""
    periods = []
    for b in blocks:
        p = getattr(b, "season_length", None)
        if p is None:
            p = getattr(b, "period", None)
        if p is not None:
            periods.append(int(p))
    return max(periods) if periods else None


def _flatten_series_id(tup):
    """The single id the filter keys a series by, for one MultiIndex column."""
    return "_".join(str(p) for p in tup)


def _restore_series_index(obj, series_index, dim="unique_id"):
    """Put the caller's original column index back on a CV result.

    Wide input with MultiIndex columns is flattened to scalar ids internally
    (the filter keys series by a single label), but the *output* should come
    back keyed the way it went in. ``series_index`` is the original
    ``df.columns``; a plain Index needs nothing done, a MultiIndex is restored
    level-for-level, names included.
    """
    if series_index is None or not isinstance(series_index, pd.MultiIndex):
        return obj
    lookup = {_flatten_series_id(t): t for t in series_index}

    if isinstance(obj, pd.DataFrame):
        if not len(obj):
            return obj
        tups = [lookup.get(u) for u in obj[dim]]
        if any(t is None for t in tups):
            return obj
        # additive: unique_id stays, the levels arrive alongside it
        for i, nm in enumerate(series_index.names):
            col = nm if nm is not None else f"level_{i}"
            obj.insert(i, col, [t[i] for t in tups])
        return obj

    import xarray as xr  # only reached from the xarray path, already imported

    ids = list(obj[dim].values)
    if any(u not in lookup for u in ids):
        return obj
    mi = pd.MultiIndex.from_tuples([lookup[u] for u in ids],
                                   names=list(series_index.names))
    return (obj.drop_vars(dim)
            .assign_coords(xr.Coordinates.from_pandas_multiindex(mi, dim)))


def _cv_to_xarray(df, alias, level):
    """Long CV output -> ``xarray.Dataset`` over ``(unique_id, window, h)``.

    The CV result is naturally n-dimensional; the long frame repeats the keys on
    every row and every consumer pivots it straight back. Variables are ``loc``,
    ``sd`` and ``y``, plus ``lo``/``hi`` over an extra ``level`` dimension when
    intervals were requested.

    ``cutoff`` is a **coordinate, not a dimension**: length-grouped batching
    gives each length-group its own ``T``, so window *i* falls on a different
    date per group. It is therefore 2-D ``(unique_id, window)``, and ``ds`` is
    3-D ``(unique_id, window, h)``. Padding these onto a shared cutoff axis
    would fabricate origins that do not exist for the shorter series.

    ``window`` is ordered oldest-first, matching the ascending ``cutoff``.
    """
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise ModuleNotFoundError(
            "output='xarray' needs xarray installed: pip install 'DLMAX[xarray]' "
            "(or pip install xarray)."
        ) from exc

    d = df.sort_values(["unique_id", "cutoff", "ds"]).copy()
    # h counts WITHIN a (series, origin) — not across origins, which is what a
    # bare groupby("unique_id").cumcount() would do once n_windows > 1.
    d["h"] = d.groupby(["unique_id", "cutoff"], sort=False).cumcount()
    d["window"] = (d.groupby("unique_id", sort=False)["cutoff"]
                   .rank(method="dense").astype(int) - 1)

    idx = d.set_index(["unique_id", "window", "h"])
    data = {"loc": idx[alias].to_xarray(),
            "sd": idx[f"{alias}-sd"].to_xarray(),
            "y": idx["y"].to_xarray()}
    if level:
        los = [idx[f"{alias}-lo-{L}"].to_xarray() for L in level]
        his = [idx[f"{alias}-hi-{L}"].to_xarray() for L in level]
        lvl = xr.DataArray(list(level), dims="level", name="level")
        dims = ("unique_id", "window", "h", "level")   # level last, as documented
        data["lo"] = xr.concat(los, dim=lvl).transpose(*dims)
        data["hi"] = xr.concat(his, dim=lvl).transpose(*dims)
    out = xr.Dataset(data)
    # cutoff varies over (unique_id, window) only; ds over all three.
    out = out.assign_coords(
        cutoff=idx["cutoff"].to_xarray().isel(h=0, drop=True),
        ds=idx["ds"].to_xarray(),
    )
    out.attrs["alias"] = alias
    return out


def _wide_to_long(df):
    """Convert a wide / tidy frame — index = time (``ds``), columns = series ids,
    values = ``y`` — to the long ``(unique_id, ds, y)`` form the filter consumes.

    NaN cells (pre-launch / post-end / structurally-missing) are dropped, so each
    series keeps its own observed span: ragged / unbalanced panels are simply
    trailing (or leading) NaN in the wide grid. Equivalent to a manual
    ``melt().dropna()``, so wide input yields the same result as the matching
    long input.

    **MultiIndex columns** (e.g. ``(dept, item)``) are flattened to a single
    ``unique_id`` by joining the levels with ``"_"`` — ``("FOODS", "s0")``
    becomes ``"FOODS_s0"``. Without this, ``reset_index`` has to label the index
    column ``("ds", "")`` to match the column depth and the melt fails looking
    for a plain ``"ds"``. Flatten the columns yourself beforehand if you want
    different ids.
    """
    if df.columns.nlevels > 1:
        flat = [_flatten_series_id(tup) for tup in df.columns]
        dupes = {c for c in flat if flat.count(c) > 1}
        if dupes:
            raise ValueError(
                "flattening the MultiIndex columns produced duplicate series "
                f"ids: {sorted(dupes)[:5]}. Rename the columns to a unique "
                "single level before passing the frame."
            )
        df = df.copy()
        df.columns = flat
    return (
        df.rename_axis("ds").reset_index()
        .melt(id_vars="ds", var_name="unique_id", value_name="y")
        .dropna(subset=["y"])
    )


class AutoFFS(StaticFFS):
    """Block-based AutoFFS — **the wing grid by default**.

    ::

        AutoFFS(season_length=12, warmup=12).cross_validation(df, h=18, n_windows=1)

    runs the wing spec with nothing else configured: a single
    :class:`~DLMAX.ffs.grid_block.GridBlock` (the online adaptive-discount grid)
    combined by the union DMA, with ``disc_prior=WING_DISC_PRIOR``,
    ``seasonal_prior=WING_SEASONAL_PRIOR`` and ``learn_dma=WING_LEARN_DMA``.
    Those constants (see ``DLMAX.ffs.discount_grid``) are what the M4 and M5
    exhibits were produced under, so the two-argument call reproduces them.

    ``blocks=[...]`` is the escape hatch for any model set that is *not* the
    default wing — construct the blocks yourself and feed a list::

        AutoFFS(blocks=[StaticBlock(season_length=12, n_seas_comps=2)])  # static
        AutoFFS(blocks=[StaticBlock(...), GridBlock.build(...)])         # both

    The orchestrator drives them and combines their per-model predictives with
    one union DMA; ``season_length`` is derived from the blocks if not given.

    Two arms, the same model:

    * :meth:`cross_validation` — rolling-origin backtest, what the paper's
      exhibits run.
    * :meth:`fit` / :meth:`update` / :meth:`predict` — forecast forward, state
      held in memory. Equivalent to :class:`AutoFFSUniverse` (asserted bitwise
      in ``tests/test_autoffs_forecast_face.py``); use the universe instead when
      the state must outlive the process, the panel does not fit in memory, or
      series arrive later (``add_series``).

    The two arms place the DMA rate learning differently — CV learns it at the
    union over the blocks, the forward arm learns it on the block, because with
    a single block there is no union to learn at. Same spec, and the reason
    :meth:`fit` does not simply reuse the CV block (see ``_forward_blocks``).

    The forward arm needs one shared calendar across series; ragged panels
    belong in :class:`AutoFFSUniverse`, which batches them.
    """

    def __init__(self, season_length=None, *args, blocks=None, warmup=None,
                 learn_dma=None, dma_prior=None, var_powers=None, n_comps=None,
                 grid_period=None, grid_warmup=None, use_grid=None, **kwargs):
        from .ffs.discount_grid import WING_LEARN_DMA
        if use_grid is not None:
            raise TypeError(
                "use_grid was removed: the default IS the wing grid. For the "
                "legacy static universe pass blocks=[StaticBlock(...)]; for "
                "static + grid together pass both blocks in the list."
            )
        self._blocks = list(blocks) if blocks is not None else None
        # Whether we are running the DEFAULT wing (no caller-supplied blocks).
        # Recorded now because ``_resolve_blocks`` populates ``_blocks`` later.
        self._wing_default = self._blocks is None
        if self._blocks is not None and season_length is None:
            season_length = _season_length_from_blocks(self._blocks)
        super().__init__(season_length, *args, **kwargs)
        # Construction args for the DEFAULT wing block; ignored when the caller
        # supplies ``blocks`` (a user-built block carries its own settings).
        # ``grid_period``/``grid_warmup`` are accepted as aliases of
        # ``season_length``/``warmup`` for the block's own geometry.
        self.warmup = warmup if warmup is not None else grid_warmup
        self.grid_period = grid_period
        self.var_powers = var_powers
        self.n_comps = n_comps
        # SGDDMA: learn the UNION forgetting rates online per series (each combiner
        # self-tunes on its own combined objective) instead of the fixed dma_pdr/mdr.
        # Part of the WING spec, so on for the default wing — but OFF for a
        # caller-supplied block list, which keeps the blocks API's fixed replay
        # (and so its bit-null-diff vs StaticFFS) exactly as it was.
        self.learn_dma = ((WING_LEARN_DMA if self._wing_default else False)
                          if learn_dma is None else bool(learn_dma))
        self.dma_prior = dma_prior
        # Forward-forecasting carry (fit/update/predict). Distinct from the
        # Legacy `_batches`, which this path does not use.
        self._fit_blocks = None
        self._fwd_blocks = None
        self._fit_srs_ids = None
        self._fit_last_ds = None
        self._union_state = None
        self._union_weights = None
        self._freq = None

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has been called (block carry is held)."""
        return self._fit_blocks is not None

    def _future_ds(self, last_ds, h):
        """The ``h`` dates strictly after ``last_ds`` at the fitted frequency."""
        return _future_ds_at(self._freq, last_ds, h)

    def _prepare_input(self, df, freq):
        """Accept **wide / tidy** input (index = time, columns = series ids) in
        addition to the long ``(unique_id, ds, y)`` form. Wide is auto-detected
        (no ``unique_id``/``ds``/``y`` columns) and converted; long is passed
        straight through to the Legacy parser, so both give identical results."""
        if not {"unique_id", "ds", "y"}.issubset(set(df.columns)):
            # keep the caller's column index so the RESULT can be keyed the way
            # the input was (see _restore_series_index)
            self._input_series_index = df.columns
            df = _wide_to_long(df)
        else:
            self._input_series_index = None
        return super()._prepare_input(df, freq)

    def _static_block(self):
        """The legacy static universe as a block, built from the config args.

        Not used by any default path (the default is the wing); a
        convenience for ``blocks=[m._static_block()]``.
        """
        return StaticBlock(
            season_length=self.season_length,
            n_seas_comps=self.n_seas_comps,
            dma_pdr=self.dma_pdr,
            dma_mdr=self.dma_mdr,
            include_ar=self.include_ar,
            adaptive=self.adaptive,
            tau_values=self.tau_values,
            var_disc_values=self.var_disc_values,
            monitor_tau=self.monitor_tau,
            universe_builder=self.universe_builder,
        )

    def _default_wing_block(self, warmup_hint=None):
        """The wing spec as one ``GridBlock``.

        Geometry comes from ``season_length`` / ``warmup`` (or their
        ``grid_period`` / ``grid_warmup`` aliases, or — failing both — the
        ``warmup_steps`` passed to ``cross_validation``, then
        ``_min_filter_length()``). Everything else is the WING_* spec plus
        library defaults: offset 1.0, adapt_guard 0.5, dampings DAMPINGS.
        """
        from DLMAX.ffs.grid_block import GridBlock  # lazy: avoid eager grid stack
        from .ffs.discount_grid import WING_DISC_PRIOR, WING_SEASONAL_PRIOR

        period = (self.grid_period if self.grid_period is not None
                  else self.season_length)
        warmup = self.warmup
        if warmup is None:
            warmup = (warmup_hint if warmup_hint
                      else self._min_filter_length())
        return GridBlock.build(
            period=period, warmup=int(warmup),
            var_powers=self.var_powers, n_comps=self.n_comps,
            pdr=self.dma_pdr, mdr=self.dma_mdr,
            disc_prior=WING_DISC_PRIOR, seasonal_prior=WING_SEASONAL_PRIOR,
        )

    def _resolve_blocks(self, warmup_hint=None):
        """The block list to drive: the caller's ``blocks``, else the default
        wing block (built once and cached)."""
        if self._blocks is None:
            self._blocks = [self._default_wing_block(warmup_hint)]
        return self._blocks

    def _cv_trajectories(self, args_list):
        """Route CV trajectory production through the blocks via their uniform
        ``cv_trajectory`` face — per batch, a list of block trajectories for the
        union DMA. args layout: 0=srs_ids, 1=arr, 2=cutoff_t_idx, 5=h, 8=warmup.

        Note each block carries its OWN warmup (set at construction); the
        ``warmup_steps`` argument only seeds the default wing block when
        ``warmup`` was not given to the constructor.
        """
        blocks = self._resolve_blocks(args_list[0][8] if args_list else None)
        out = []
        for a in args_list:
            srs_ids, arr, cut, hh = a[0], a[1], a[2], a[5]
            xa = a[16] if len(a) > 16 else None          # (T, q, n_reg) or None
            out.append([b.cv_trajectory(srs_ids, arr, cut, hh, regressors=xa)
                        for b in blocks])
        return out

    # ------------------------------------------------------------------
    # Forward forecasting: fit -> (update) -> predict, held in memory
    # ------------------------------------------------------------------
    # These are the IN-MEMORY counterpart of AutoFFSUniverse's grid mode. The
    # universe does the same three operations against a directory (manifest,
    # per-batch HDF5, a dispatcher); here the blocks and the union carry are
    # just attributes. Same blocks, same union DMA, same combine seam, so the
    # two agree exactly -- see tests/test_autoffs_forecast_face.py, which
    # asserts rtol=0, atol=0 against AutoFFSUniverse on the same data.
    #
    # Use the universe when the state must outlive the process or the panel
    # does not fit in memory; use these when it is one panel in one session
    # (and for the documented quick start, which is what this is for).

    def _fit_state_check(self, what):
        if self._fit_blocks is None:
            raise RuntimeError(f"Call fit(...) before {what}(...).")

    def _forward_blocks(self, warmup_hint=None):
        """The block list for the FORWARD path — deliberately not
        ``_resolve_blocks``.

        The two arms place the SGDDMA differently, and for one block that is a
        real numerical difference (~2e-3 relative), not a rounding one:

        * CV combines whatever blocks it is given through the union allocator
          and lets the UNION learn its forgetting rates (``_cv_combine`` passes
          ``learn_dma=self.learn_dma``), so ``_default_wing_block`` leaves the
          block itself on fixed replay.
        * Forward with a single block has no union to learn at — the block's own
          hierarchical DMA *is* the combine. So the rate learning has to sit on
          the BLOCK, which is exactly what ``AutoFFSUniverse`` does
          (``grid_learn_dma = WING_LEARN_DMA``, threaded into
          ``_grid_block_from_cfg``).

        Same spec either way; only the layer that owns the learning moves. Built
        separately rather than by changing ``_default_wing_block``, because that
        one feeds the published M4/M5 cross-validation results.

        A caller-supplied ``blocks=`` list is used as given: those blocks carry
        their own settings, and >=2 of them are combined at the union, where the
        universe's multi-block path learns the rate too.
        """
        if self._fwd_blocks is None:
            if not self._wing_default:
                self._fwd_blocks = self._resolve_blocks(warmup_hint)
            else:
                from DLMAX.ffs.grid_block import GridBlock
                from .ffs.discount_grid import (WING_DISC_PRIOR,
                                                WING_SEASONAL_PRIOR)
                period = (self.grid_period if self.grid_period is not None
                          else self.season_length)
                warmup = self.warmup
                if warmup is None:
                    warmup = (warmup_hint if warmup_hint
                              else self._min_filter_length())
                self._fwd_blocks = [GridBlock.build(
                    period=period, warmup=int(warmup),
                    var_powers=self.var_powers, n_comps=self.n_comps,
                    pdr=self.dma_pdr, mdr=self.dma_mdr,
                    disc_prior=WING_DISC_PRIOR,
                    seasonal_prior=WING_SEASONAL_PRIOR,
                    learn_dma=self.learn_dma, dma_prior=self.dma_prior)]
        return self._fwd_blocks

    def fit(self, df, freq=None):
        """Filter the blocks over ``df``'s history and hold the carry.

        ``df`` is long ``(unique_id, ds, y)`` or wide, as elsewhere. Every
        series must share a calendar (the grid is a single vmapped panel);
        ragged panels belong in :class:`AutoFFSUniverse`, which batches them.
        """
        per_series, sid_ds, sid_y, freq = self._prepare_input(df, freq)
        srs_ids = list(per_series.index)
        lengths = {len(sid_y[s]) for s in srs_ids}
        if len(lengths) > 1:
            raise ValueError(
                "AutoFFS.fit needs one shared calendar: series lengths differ "
                f"({sorted(lengths)[:5]}...). Use AutoFFSUniverse, which groups "
                "unequal-length series into batches.")
        arr = np.column_stack([np.asarray(sid_y[s], dtype=float) for s in srs_ids])
        blocks = self._forward_blocks(arr.shape[0])
        if len(blocks) == 1:
            # single block: its OWN hierarchical combine, byte-identical to the
            # universe's single-block worker (_grid_fit_batch_file).
            blocks[0].scan_filter(arr)
            self._union_weights = None
        else:
            _st, w, _mi = _multiblock_fit(
                blocks, arr, self.dma_pdr, self.dma_mdr,
                learn_dma=self.learn_dma, dma_prior=self.dma_prior)
            self._union_state, self._union_weights = _st, w
        self._fit_blocks = blocks
        self._fit_srs_ids = srs_ids
        self._fit_last_ds = {s: sid_ds[s][-1] for s in srs_ids}
        self._freq = freq
        return self

    def update(self, df_new):
        """Advance the held carry over new observations, one filter pass.

        Every fitted series must appear, with equal numbers of new rows -- the
        panel advances together.
        """
        self._fit_state_check("update")
        per_series, sid_ds, sid_y, _ = self._prepare_input(df_new, self._freq)
        provided = set(per_series.index)
        known = set(self._fit_srs_ids)
        if provided - known:
            raise ValueError(
                "update(df_new) contains series that were not fitted: "
                f"{sorted(provided - known)[:5]}. AutoFFS has no add_series; "
                "use AutoFFSUniverse for a panel that gains series.")
        if known - provided:
            raise ValueError(
                "update(df_new) is missing fitted series "
                f"{sorted(known - provided)[:5]}: the panel advances together.")
        lengths = {len(sid_y[s]) for s in self._fit_srs_ids}
        if len(lengths) > 1:
            raise ValueError(
                f"update(df_new) rows per series must be equal; got {sorted(lengths)}.")
        new_arr = np.column_stack(
            [np.asarray(sid_y[s], dtype=float) for s in self._fit_srs_ids])
        if len(self._fit_blocks) == 1:
            self._fit_blocks[0].scan_filter(new_arr)
        else:
            self._union_state, self._union_weights = _multiblock_advance(
                self._fit_blocks, self._union_state, new_arr,
                self.dma_pdr, self.dma_mdr,
                learn_dma=self.learn_dma, dma_prior=self.dma_prior)
        for s in self._fit_srs_ids:
            self._fit_last_ds[s] = sid_ds[s][-1]
        return self

    def predict(self, h, level=None):
        """``h``-step forecast from the held carry. Does not modify state."""
        self._fit_state_check("predict")
        if not isinstance(h, int) or h <= 0:
            raise ValueError("h must be a positive integer.")
        level = self._normalise_level(level)
        blocks = self._fit_blocks
        if self._union_weights is None:
            # Single block: loc/sd come from the block's OWN combine, which is
            # what AutoFFSUniverse reports (_grid_predict_batch_file) — so the
            # two faces agree bitwise. The union seam is used only to get the
            # interval bounds, which block.forecast() does not return; it
            # re-derives loc/sd on the way and lands ~1e-14 from the block's own
            # (same formula, different reduction order), so its loc/sd are
            # deliberately discarded rather than reported.
            loc, sd, comp = blocks[0].forecast(h)
            bounds = None
            if level:
                W = np.transpose(np.asarray(comp["Wc"]), (1, 0))        # (M, q)
                f_h = np.transpose(np.asarray(comp["LOCc"]), (1, 0, 2))  # (M,q,h)
                q_h = np.transpose(np.asarray(comp["QHc"]), (1, 0, 2))
                nu = np.transpose(np.asarray(comp["NUc"]), (1, 0))      # (M, q)
                _l, _s, bounds = _union_predictive_combine(
                    W, f_h, q_h, nu, level, self.sd_method)
        else:
            loc, sd, bounds = _multiblock_forecast(
                blocks, self._union_weights, h, level, self.sd_method)
        loc = np.asarray(loc)                                           # (q, h)
        sd = np.asarray(sd)
        if sd.shape != loc.shape:
            sd = sd.T
        rows = []
        for j, sid in enumerate(self._fit_srs_ids):
            fut = self._future_ds(self._fit_last_ds[sid], h)
            for k in range(h):
                r = {"unique_id": sid, "ds": fut[k],
                     self.alias: float(loc[j, k]),
                     f"{self.alias}-sd": float(sd[j, k])}
                if level:
                    for L in level:
                        lo, hi = bounds[int(L)]
                        r[f"{self.alias}-lo-{L}"] = float(np.asarray(lo)[k, j])
                        r[f"{self.alias}-hi-{L}"] = float(np.asarray(hi)[k, j])
                rows.append(r)
        return pd.DataFrame(rows)

    def forecast(self, df, h, freq=None, level=None):
        """One-shot fit + predict, discarding the state afterwards.

        The same signature as the legacy one-shot face, but it runs THIS
        model -- the wing by default -- rather than the legacy static
        universe. Inheriting the legacy implementation would silently forecast
        a different model set than the one the caller constructed.

        ``self`` is left untouched, so a fitted instance can be reused: the
        work happens on a scratch copy carrying the same configuration. Use
        :meth:`fit` + :meth:`predict` when the state IS wanted afterwards.
        """
        scratch = AutoFFS(
            season_length=self.season_length,
            blocks=self._blocks if not self._wing_default else None,
            warmup=self.warmup,
            learn_dma=self.learn_dma,
            dma_prior=self.dma_prior,
            var_powers=self.var_powers,
            n_comps=self.n_comps,
            grid_period=self.grid_period,
            n_seas_comps=self.n_seas_comps,
            dma_pdr=self.dma_pdr,
            dma_mdr=self.dma_mdr,
            max_batch_size=self.max_batch_size,
            dask_client=self.dask_client,
            alias=self.alias,
            sd_method=self.sd_method,
        )
        return scratch.fit(df, freq=freq).predict(h, level=level)

    def _cv_combine(self, batch_item, arr, level):
        """Combine a batch's block trajectories with the single union DMA.
        For one block this reproduces that block's own in-scan hierarchical
        combine to ~float precision (the documented single-block property)."""
        combined = _union_combine_cv(
            batch_item, arr, self.dma_pdr, self.dma_mdr, level, self.sd_method,
            learn_dma=self.learn_dma, dma_prior=self.dma_prior,
            # union DMA replays from t=0 (like the fixed path); the 1/t prior
            # anchors the early steps, so no separate learning warmup is needed.
            warmup=0,
        )
        return combined, np.asarray(batch_item[0].cutoff_t_idx)


_UNSET = object()   # "caller said nothing" — distinct from an explicit None


class AutoFFSUniverse(StaticFFSUniverse):
    """Persistent streaming AutoFFS — **the wing grid by default**.

    The streaming counterpart of :class:`AutoFFS`, with the same defaults::

        AutoFFSUniverse.create(path, season_length=7, warmup=14)

    runs the wing spec: each batch is ONE adaptive-discount
    ``GridBlock``, streamed via the block's production face
    (``scan_filter``/``fwd_filter``/``forecast``) and persisted via the block's
    ``save``/``load`` + a compatible ``/metadata`` group. ``grid_disc_prior``,
    ``grid_seasonal_prior`` and ``grid_learn_dma`` default to the WING_* spec
    (see ``DLMAX.ffs.discount_grid``), so this reproduces the published M5 run
    given its ``grid_var_powers=[1.0, 0.25]``.

    ``grid_period`` defaults to ``season_length``. Pass ``grid_period=None``
    **explicitly** to opt out of the grid entirely and get the legacy
    single-multi path, byte-identical to :class:`StaticFFSUniverse` — that
    is this class's equivalent of ``AutoFFS(blocks=[StaticBlock(...)])``, since
    the streaming path has no StaticBlock face.
    """

    def __init__(self, path, *args, warmup=None, block_builder=None,
                 grid_period=_UNSET, grid_warmup=None,
                 grid_offset: float = 1.0, grid_var_powers=None,
                 grid_disc_prior=_UNSET, grid_additive_logscore: bool = False,
                 grid_decouple_trend: bool = False,
                 grid_learn_dma=_UNSET, grid_dma_prior=None,
                 grid_seasonal_prior=_UNSET,
                 grid_blocks=None, union_pdr=None, union_mdr=None,
                 union_learn_dma: bool = False, union_dma_prior=None, **kwargs):
        from .ffs.discount_grid import (
            WING_DISC_PRIOR, WING_SEASONAL_PRIOR, WING_LEARN_DMA)
        # ``warmup`` is the name shared with AutoFFS; it seeds the
        # legacy ``warmup_steps`` (prior policy) and the grid's own warmup.
        if warmup is not None:
            kwargs.setdefault("warmup_steps", warmup)
        super().__init__(path, *args, **kwargs)
        # Wing by default: an unspecified grid_period means "grid at
        # season_length"; an EXPLICIT None means "no grid, legacy path".
        #
        # A universe_builder ALSO opts out: it defines a custom model set on the
        # legacy multi path, which grid mode does not consume -- defaulting to
        # the wing there would silently ignore the caller's universe and run a
        # different model (it did: ENTSOE regressed this way).
        _wing_default = (grid_period is _UNSET
                         and getattr(self, "universe_builder", None) is None)
        if grid_period is _UNSET:
            # resolve the sentinel either way: the wing takes season_length, the
            # universe_builder path takes None (no grid)
            grid_period = self.season_length if _wing_default else None
        self.grid_period = grid_period
        self.grid_offset = grid_offset
        # Error-variance variants of the grid: None -> classic {A, M}; a list
        # (e.g. [1.0, 0.25]) -> additive-Fourier compound-Poisson sweep (M5).
        self.grid_var_powers = list(grid_var_powers) if grid_var_powers is not None else None
        # Single-block MAP-prior / objective knobs. The discount prior defaults
        # to the WING spec; pass an explicit None for the un-regularised grid.
        if grid_disc_prior is _UNSET:
            grid_disc_prior = WING_DISC_PRIOR
        self.grid_disc_prior = (tuple(float(x) for x in grid_disc_prior)
                                if grid_disc_prior is not None else None)
        self.grid_additive_logscore = bool(grid_additive_logscore)
        # Decouple the polynomial-trend discount: level (wing-searched) vs growth
        # (RTRL-learned) get SEPARATE δ blocks (under the same prior). False ->
        # fused block, byte-identical to the coupled grid.
        self.grid_decouple_trend = bool(grid_decouple_trend)
        # SGDDMA: learn the grid's family (hierarchical) forgetting rates online per
        # series (N(3,1) logit prior) instead of fixed. Part of the wing spec, so
        # on by default; pass False for the byte-identical fixed replay.
        self.grid_learn_dma = (WING_LEARN_DMA if grid_learn_dma is _UNSET
                               else bool(grid_learn_dma))
        self.grid_dma_prior = (tuple(float(x) for x in grid_dma_prior)
                               if grid_dma_prior is not None else None)
        # Block-specific seasonal discount prior mean (N(x,1) on Fourier blocks);
        # defaults to the WING spec. An explicit None makes the seasonal blocks
        # use grid_disc_prior like everything else.
        if grid_seasonal_prior is _UNSET:
            grid_seasonal_prior = WING_SEASONAL_PRIOR
        self.grid_seasonal_prior = (float(grid_seasonal_prior)
                                    if grid_seasonal_prior is not None else None)
        self.grid_warmup = (int(grid_warmup) if grid_warmup is not None
                            else int(self.warmup_steps or 0))
        # Multi-block: >=2 GridBlocks combined by ONE top union DMA
        # (Universe(blocks=[...])). ``grid_blocks`` is a list of dicts
        # {period, var_powers, warmup, offset, pdr, mdr}; the single-block
        # ``grid_period`` path is untouched (byte-identical).
        self._block_specs = self._norm_block_specs(grid_blocks)
        self._multiblock = (self._block_specs is not None
                            and len(self._block_specs) >= 2)
        self._union_pdr = float(union_pdr) if union_pdr is not None else self.dma_pdr
        self._union_mdr = float(union_mdr) if union_mdr is not None else self.dma_mdr
        # SGDDMA on the top union combiner (multi-block): learns its OWN forgetting
        # rate online per series (separate from each block's family DMA).
        self._union_learn_dma = bool(union_learn_dma)
        self._union_dma_prior = (tuple(float(x) for x in union_dma_prior)
                                 if union_dma_prior is not None else None)
        # A block_builder is the general escape hatch: the caller returns the
        # block list themselves, so arbitrary taxonomies (a filtered grid, a
        # hand-assembled family set, an AdaptiveBlock over hand-compiled Wing
        # cells) work — things no config tuple can express. It supersedes
        # grid_period / grid_blocks. Signature ``(init_data, h, ctx)``, as
        # universe_builder; see ``_blocks_from_builder``. Called once here so an
        # error surfaces at construction, and so the block COUNT is known (>=2
        # routes to the multi-block union path) — with init_data None, the same
        # as every rebuild-to-load call.
        self.block_builder = block_builder
        if block_builder is not None:
            self._multiblock = len(_blocks_from_builder(
                block_builder, self._block_ctx())) >= 2
        # Grid mode is on by default (``_wing_default``) even when the period is
        # None — a non-seasonal wing is a legitimate grid (yearly M4). It goes
        # off only when the caller passes grid_period=None EXPLICITLY.
        self._grid_mode = (_wing_default or grid_period is not None
                           or self._multiblock or block_builder is not None)

    def _grid_tail(self):
        """``(n_regs, is_autoregressive)`` of the grid block, cached.

        Building the block is how the tail is known — it comes from the
        ``block_builder``, not from the config tuple, since no spec field can
        express a tail. Rebuild-to-load (``init_data=None``), so this is the same
        cheap construction ``open()`` already does.
        """
        if getattr(self, "_tail_cache", None) is None:
            b = self._grid_block()
            self._tail_cache = (int(b.n_regs), bool(b.is_autoregressive))
        return self._tail_cache

    def _check_exog_supported(self):
        """Exogenous regressors and the block's tail must AGREE.

        The gate is not "which engine" but "does this universe actually wire
        the data to a tail". Both halves of the mismatch are SILENT if
        unchecked, which is why each is an error:

        * exog supplied, no tail — the design matrix goes nowhere and the
          universe scores as a structural one that happened to cost more;
        * tail, no exog supplied — the tail filters against a zero ``F`` row every
          step, contributing nothing while occupying state and a discount block.

        An autoregressive tail, ``AR(disc_rate=Wing(...))``, has the same
        trap, and is worth failing loudly on for the same reason.

        Checked at first use, not in ``__init__`` — ``open()`` constructs with the
        (wing) defaults before restoring the persisted config, so a legacy exog
        universe would trip a constructor-time guard on reopen.
        """
        if not self._grid_mode:
            return
        has_exog = getattr(self, "exog_provider", None) is not None
        if self._multiblock:
            if has_exog:
                raise ValueError(
                    "exog_provider is not yet supported on the grid MULTI-block "
                    "path: which block carries the tail, and how one design "
                    "matrix aligns across blocks of different structure, is "
                    "unresolved. Use a single-block grid universe, or "
                    "grid_period=None for the legacy multi-model path."
                )
            # No tail check here: ``_grid_block`` is single-block by construction
            # and would raise on a multi-block builder, and a multi-block tail is
            # unreachable anyway while the branch above stands.
            return
        n_regs, is_ar = self._grid_tail()
        if n_regs and is_ar:
            # An AR tail takes its regressors from the y stream the block has
            # already filtered, so it needs no provider — and must not be given
            # one, which would mean two sources for the same slots.
            if has_exog:
                raise ValueError(
                    "the grid block's tail is AUTOREGRESSIVE (an AR component), "
                    "which builds its own design from the series' own lags, but "
                    "an exog_provider is also set. Those are two sources for the "
                    "same tail slots. Drop the provider for an AR tail, or use "
                    "Regressors instead of AR for a caller-supplied design."
                )
            return
        if has_exog and n_regs == 0:
            raise ValueError(
                "exog_provider is set but the grid block has no regression tail "
                "(n_regs=0), so the regressors would be silently discarded. Two "
                "fixes, and which one you want depends on the engine: pass a "
                "block_builder whose cells carry a Regressors component to keep "
                "the wing (the config tuple cannot express a tail, so a builder "
                "is the only route); or pass grid_period=None explicitly for the "
                "legacy multi-model path, which has carried exogenous regressors "
                "all along."
            )
        if not has_exog and n_regs:
            raise ValueError(
                f"the grid block has an EXOGENOUS regression tail (n_regs="
                f"{n_regs}) but no exog_provider is set, so the tail would filter "
                "against a zero row at every step and contribute nothing. Supply "
                "exog_provider; or use AR instead of Regressors for a tail that "
                "builds its own design from the series' own lags; or drop the "
                "component."
            )

    def _norm_block_specs(self, grid_blocks):
        """Normalise ``grid_blocks`` dicts to picklable tuples: ``(period,
        var_powers, warmup, offset, pdr, mdr, seasonal_mult, additive_logscore,
        disc_prior, decouple_trend, learn_dma, dma_prior, seasonal_prior)``.
        Block-level pdr/mdr default to the universe DMA rates; warmup/offset to
        the grid defaults; ``seasonal_prior`` to the universe's (so a multi-block
        universe inherits the WING spec like the single-block path does).

        Fields are only ever APPENDED — the reader guards on ``len(spec)``, so a
        shorter tuple persisted by an older version still rebuilds as it ran.
        For anything this cannot express (a filtered taxonomy, ``n_comps``,
        ``period2``) use ``block_builder``."""
        if not grid_blocks:
            return None
        specs = []
        for b in grid_blocks:
            vp = b.get("var_powers", None)
            specs.append((int(b["period"]),
                          list(vp) if vp is not None else None,
                          int(b.get("warmup", self.grid_warmup)),
                          float(b.get("offset", self.grid_offset)),
                          float(b.get("pdr", self.dma_pdr)),
                          float(b.get("mdr", self.dma_mdr)),
                          bool(b.get("seasonal_mult", False)),
                          bool(b.get("additive_logscore", False)),
                          b.get("disc_prior", None),
                          bool(b.get("decouple_trend", False)),
                          bool(b.get("learn_dma", False)),
                          b.get("dma_prior", None),
                          b.get("seasonal_prior", self.grid_seasonal_prior)))
        return specs

    def _prepare_input(self, df, freq):
        """Accept **wide / tidy** input (index = time, columns = series ids) in
        addition to long ``(unique_id, ds, y)`` — same rule as :meth:`AutoFFS.
        _prepare_input`, so the streaming and batch orchestrators take the same
        forms. Wide is auto-detected and melted (NaN cells dropped, so each series
        keeps its own span); long passes straight through unchanged."""
        if not {"unique_id", "ds", "y"}.issubset(set(df.columns)):
            # keep the caller's column index so the RESULT can be keyed the way
            # the input was (see _restore_series_index)
            self._input_series_index = df.columns
            df = _wide_to_long(df)
        else:
            self._input_series_index = None
        return super()._prepare_input(df, freq)

    # -- grid-mode config persistence -----------------------------------------
    def _save_config(self):
        super()._save_config()
        with h5py.File(self._config_path(), "a") as f:
            cfg = f["config"]
            for k, v in {
                "grid_period": (int(self.grid_period)
                                if self.grid_period is not None else -1),
                "grid_warmup": int(self.grid_warmup),
                "grid_offset": float(self.grid_offset),
                "grid_mode": int(self._grid_mode),
            }.items():
                if k in cfg:
                    del cfg[k]
                cfg.create_dataset(k, data=v)
            # grid_var_powers: variable-length (empty -> None on restore).
            if "grid_var_powers" in cfg:
                del cfg["grid_var_powers"]
            cfg.create_dataset(
                "grid_var_powers",
                data=(np.asarray(self.grid_var_powers, dtype=float)
                      if self.grid_var_powers is not None else np.empty(0)))
            # single-block MAP-prior / objective knobs (empty prior array -> None)
            if "grid_disc_prior" in cfg:
                del cfg["grid_disc_prior"]
            cfg.create_dataset(
                "grid_disc_prior",
                data=(np.asarray(self.grid_disc_prior, dtype=float)
                      if self.grid_disc_prior is not None else np.empty(0)))
            if "grid_additive_logscore" in cfg:
                del cfg["grid_additive_logscore"]
            cfg.create_dataset("grid_additive_logscore",
                               data=int(self.grid_additive_logscore))
            if "grid_decouple_trend" in cfg:
                del cfg["grid_decouple_trend"]
            cfg.create_dataset("grid_decouple_trend",
                               data=int(self.grid_decouple_trend))
            if "grid_learn_dma" in cfg:
                del cfg["grid_learn_dma"]
            cfg.create_dataset("grid_learn_dma", data=int(self.grid_learn_dma))
            if "grid_dma_prior" in cfg:
                del cfg["grid_dma_prior"]
            cfg.create_dataset(
                "grid_dma_prior",
                data=(np.asarray(self.grid_dma_prior, dtype=float)
                      if self.grid_dma_prior is not None else np.empty(0)))
            if "grid_seasonal_prior" in cfg:
                del cfg["grid_seasonal_prior"]
            cfg.create_dataset(
                "grid_seasonal_prior",
                data=(np.asarray([self.grid_seasonal_prior], dtype=float)
                      if self.grid_seasonal_prior is not None else np.empty(0)))
            # multi-block: block specs (pickled attr) + union DMA rates.
            for k in ("multiblock", "union_pdr", "union_mdr"):
                if k in cfg:
                    del cfg[k]
            # The callable cannot be persisted; the NAME is, and open() checks
            # the re-supplied builder against it (same contract as
            # universe_builder). Without this a reopened universe would silently
            # fall back to the config-built block -- a DIFFERENT model.
            if "block_builder_name" in cfg:
                del cfg["block_builder_name"]
            cfg.create_dataset("block_builder_name", data=(
                f"{self.block_builder.__module__}:{self.block_builder.__qualname__}"
                if self.block_builder is not None else ""))
            cfg.create_dataset("multiblock", data=int(self._multiblock))
            cfg.create_dataset("union_pdr", data=float(self._union_pdr))
            cfg.create_dataset("union_mdr", data=float(self._union_mdr))
            for k in ("union_learn_dma", "union_dma_prior"):
                if k in cfg:
                    del cfg[k]
            cfg.create_dataset("union_learn_dma", data=int(self._union_learn_dma))
            cfg.create_dataset(
                "union_dma_prior",
                data=(np.asarray(self._union_dma_prior, dtype=float)
                      if self._union_dma_prior is not None else np.empty(0)))
            if "block_specs" in cfg.attrs:
                del cfg.attrs["block_specs"]
            if self._multiblock:
                import pickle
                cfg.attrs["block_specs"] = np.void(pickle.dumps(self._block_specs))

    @classmethod
    def open(cls, path, dask_client=None, universe_builder=None,
             exog_provider=None, block_builder=None):
        uni = super().open(path, dask_client=dask_client,
                           universe_builder=universe_builder,
                           exog_provider=exog_provider)
        with h5py.File(uni._config_path(), "r") as f:
            cfg = f["config"]
            bb_name = cfg["block_builder_name"][()] if "block_builder_name" in cfg else ""
            bb_name = bb_name.decode() if isinstance(bb_name, bytes) else str(bb_name)
            if bb_name:
                if block_builder is None:
                    raise ValueError(
                        f"Universe at {path} was built with a custom "
                        f"block_builder '{bb_name}'; re-supply block_builder=... "
                        f"to open() (a callable cannot be persisted).")
                supplied = (f"{block_builder.__module__}:"
                            f"{block_builder.__qualname__}")
                if supplied != bb_name:
                    warnings.warn(
                        f"block_builder mismatch at {path}: built with "
                        f"'{bb_name}', opened with '{supplied}'.")
            uni.block_builder = block_builder
            if "grid_mode" in cfg and int(cfg["grid_mode"][()]):
                gp = int(cfg["grid_period"][()])
                uni.grid_period = None if gp == -1 else gp
                uni.grid_warmup = int(cfg["grid_warmup"][()])
                uni.grid_offset = float(cfg["grid_offset"][()])
                uni._grid_mode = True
                if "grid_var_powers" in cfg and len(cfg["grid_var_powers"][()]) > 0:
                    uni.grid_var_powers = list(np.asarray(cfg["grid_var_powers"][()]))
                # NB the stored value is authoritative, INCLUDING when it is
                # empty (= None). Falling through to the constructor default
                # would silently re-prior a universe that ran without one.
                if "grid_disc_prior" in cfg:
                    _dp = np.asarray(cfg["grid_disc_prior"][()])
                    uni.grid_disc_prior = (tuple(float(x) for x in _dp)
                                           if len(_dp) > 0 else None)
                if "grid_additive_logscore" in cfg:
                    uni.grid_additive_logscore = bool(int(cfg["grid_additive_logscore"][()]))
                if "grid_decouple_trend" in cfg:
                    uni.grid_decouple_trend = bool(int(cfg["grid_decouple_trend"][()]))
                if "grid_learn_dma" in cfg:
                    uni.grid_learn_dma = bool(int(cfg["grid_learn_dma"][()]))
                if "grid_dma_prior" in cfg:
                    _mp = np.asarray(cfg["grid_dma_prior"][()])
                    uni.grid_dma_prior = (tuple(float(x) for x in _mp)
                                          if len(_mp) > 0 else None)
                if "grid_seasonal_prior" in cfg:
                    _sp = np.asarray(cfg["grid_seasonal_prior"][()])
                    uni.grid_seasonal_prior = (float(_sp[0]) if len(_sp) > 0
                                               else None)
                if "multiblock" in cfg and int(cfg["multiblock"][()]):
                    import pickle
                    uni._block_specs = pickle.loads(cfg.attrs["block_specs"].tobytes())
                    uni._multiblock = True
                    uni._union_pdr = float(cfg["union_pdr"][()])
                    uni._union_mdr = float(cfg["union_mdr"][()])
                    if "union_learn_dma" in cfg:
                        uni._union_learn_dma = bool(int(cfg["union_learn_dma"][()]))
                    if "union_dma_prior" in cfg and len(cfg["union_dma_prior"][()]) > 0:
                        uni._union_dma_prior = tuple(
                            float(x) for x in np.asarray(cfg["union_dma_prior"][()]))
            else:
                # Persisted as a NON-grid universe (or written before grid mode
                # existed). The constructor now defaults to the wing, so the
                # saved state has to switch it back off explicitly — otherwise
                # reopening a static universe would silently run a wing.
                uni._grid_mode = False
                uni._multiblock = False
                uni.grid_period = None
        return uni

    # -- grid-mode batch build / persist --------------------------------------
    def _block_ctx(self):
        """Geometry handed to a ``block_builder``. ``pdr``/``mdr`` are the
        universe's configured DMA rates: the config route feeds them to
        ``GridBlock.build`` for you, so a builder has to be told them or it
        silently runs different rates than the universe was asked for."""
        return {"period": self.grid_period, "warmup": self.grid_warmup,
                "h": self._fit_h_template,
                "pdr": self.dma_pdr, "mdr": self.dma_mdr}

    def _grid_block(self, init_data=None):
        if self.block_builder is not None:
            blocks = _blocks_from_builder(self.block_builder, self._block_ctx(),
                                          init_data)
            if len(blocks) != 1:
                raise ValueError(
                    f"block_builder returned {len(blocks)} blocks but this "
                    f"universe is on the single-block path.")
            return blocks[0]
        from DLMAX.ffs.grid_block import GridBlock
        return GridBlock.build(period=self.grid_period, warmup=self.grid_warmup,
                               var_powers=self.grid_var_powers, offset=self.grid_offset,
                               pdr=self.dma_pdr, mdr=self.dma_mdr,
                               disc_prior=self.grid_disc_prior,
                               additive_logscore=self.grid_additive_logscore,
                               decouple_trend=self.grid_decouple_trend,
                               learn_dma=self.grid_learn_dma,
                               dma_prior=self.grid_dma_prior,
                               seasonal_prior=self.grid_seasonal_prior)

    def _grid_cfg(self):
        """Picklable grid config for the path-based workers. Single-block: the
        7-tuple. Multi-block: ``(block_specs, union_pdr, union_mdr, capacity)``."""
        if self._multiblock:
            return (tuple(self._block_specs or ()), self._union_pdr,
                    self._union_mdr, self.max_batch_size, self._union_learn_dma,
                    self._union_dma_prior, self.block_builder, self._block_ctx())
        return (self.grid_period, self.grid_var_powers, self.grid_warmup,
                self.grid_offset, self.dma_pdr, self.dma_mdr, self.max_batch_size,
                self.grid_disc_prior, self.grid_additive_logscore,
                self.grid_decouple_trend, self.grid_learn_dma, self.grid_dma_prior,
                self.grid_seasonal_prior, self.block_builder, self._block_ctx())

    def _grid_stub(self):
        """An StaticFFS carrying the dask_client, for ``_dispatch`` +
        ``_iter_batches`` (in-process when dask_client is None)."""
        return StaticFFS(
            season_length=self.season_length, n_seas_comps=self.n_seas_comps,
            dma_pdr=self.dma_pdr, dma_mdr=self.dma_mdr,
            max_batch_size=self.max_batch_size, alias=self.alias,
            dask_client=self.dask_client)

    def _last_ds_arr(self, srs_ids, last_ds_map):
        is_int = isinstance(self._freq, (int, np.integer))
        arr = np.array([int(last_ds_map[s]) if is_int
                        else pd.Timestamp(last_ds_map[s]).value for s in srs_ids])
        return arr, is_int

    def _save_grid_batch(self, block, path, srs_ids, active, last_ds_map):
        """Persist a grid batch head-side (add_series). Delegates to the same
        ``_save_grid_batch_file`` the workers use."""
        last_ds_arr, is_int = self._last_ds_arr(srs_ids, last_ds_map)
        _save_grid_batch_file(block, path, srs_ids, active, last_ds_arr, is_int)

    # -- grid-mode fit / forecast (single AdaptiveBlock per batch) -------------
    def fit(self, df, freq=None, h_template: int = 18):
        self._check_exog_supported()
        if not self._grid_mode:
            return super().fit(df, freq, h_template)
        if len(self._manifest):
            raise RuntimeError(
                "Universe already contains series. Create a new universe to refit.")
        per_series, sid_ds, sid_y, freq = self._prepare_input(df, freq)
        self._freq = freq
        self._fit_h_template = h_template
        stub = self._grid_stub()
        cfg = self._grid_cfg()
        batches_input = list(stub._iter_batches(per_series, sid_y))
        batch_ids = [self._new_batch_id() for _ in batches_input]
        args_list, manifest_rows = [], []
        for bid, (srs_ids, arr) in zip(batch_ids, batches_input):
            last_ds_map = {sid: sid_ds[sid][-1] for sid in srs_ids}
            last_ds_arr, is_int = self._last_ds_arr(srs_ids, last_ds_map)
            # exog rows aligned to arr over the batch's shared calendar (ds from
            # the first series — a rectangular panel, as the legacy path assumes).
            # None when no provider is configured.
            exog = self._materialise_exog(srs_ids, sid_ds[srs_ids[0]])
            args_list.append((self._batch_path(bid), np.asarray(arr, dtype=float),
                              tuple(srs_ids), last_ds_arr, is_int, cfg, exog))
            for sid in srs_ids:
                manifest_rows.append({"unique_id": sid, "batch_id": bid,
                                      "last_ds": last_ds_map[sid], "active": True})
        worker = (_multiblock_fit_batch_file if self._multiblock
                  else _grid_fit_batch_file)
        stub._dispatch(worker, args_list)                   # in-process or dask
        self._manifest = pd.DataFrame(manifest_rows).set_index("unique_id")
        self._save_config()
        self._save_manifest()
        return self

    def update(self, df_new):
        """Extend the grid universe one (or more) origins. Per affected batch,
        the block carry is loaded, advanced over the new rows (``scan_filter``
        continues the held carry — resumable exactly), and re-saved."""
        self._check_exog_supported()
        if not self._grid_mode:
            return super().update(df_new)
        per_series, sid_ds, sid_y, _ = self._prepare_input(df_new, self._freq)
        active_ids = set(self._manifest[self._manifest["active"]].index)
        provided = set(per_series.index)
        unknown = provided - set(self._manifest.index)
        if unknown:
            raise ValueError(
                f"update(df_new) contains unknown unique_ids "
                f"(use add_series): {sorted(unknown)[:5]}.")
        provided_active = provided & active_ids
        if not provided_active:
            raise ValueError("df_new contains no active series.")
        affected: dict = {}
        for sid in provided_active:
            affected.setdefault(int(self._manifest.loc[sid, "batch_id"]), []).append(sid)
        stub = self._grid_stub()
        cfg = self._grid_cfg()
        args_list, manifest_sets = [], []
        for bid, sids_upd in affected.items():
            path = self._batch_path(bid)
            srs_ids_b, active_arr, last_ds_b = _load_batch_meta(path)
            active_in_batch = [s for s, a in zip(srs_ids_b, active_arr) if a]
            missing = set(active_in_batch) - set(sids_upd)
            if missing:
                raise ValueError(
                    f"Batch {bid}: all active series must update together; "
                    f"missing {sorted(missing)[:5]}.")
            new_lengths = {len(sid_y[sid]) for sid in active_in_batch}
            if len(new_lengths) > 1:
                raise ValueError(
                    f"Batch {bid}: new observations per series must have equal "
                    f"length; got {sorted(new_lengths)}.")
            T_new = new_lengths.pop()
            # grid batches are all-active (no remove/pad), so every slot is real.
            cols, new_last_ds = [], {}
            for sid, a in zip(srs_ids_b, active_arr):
                if a:
                    cols.append(np.asarray(sid_y[sid], dtype=float))
                    new_last_ds[sid] = sid_ds[sid][-1]
                else:                                          # (unreached in grid mode)
                    cols.append(cols[0].copy() if cols else np.zeros(T_new))
                    new_last_ds[sid] = last_ds_b[sid]
            new_arr = np.column_stack(cols)                    # (T_new, q)
            new_last_ds_arr, is_int = self._last_ds_arr(srs_ids_b, new_last_ds)
            # exog over the NEW rows only, aligned to new_arr. The batch shares a
            # calendar, so any active series' ds serves; take the first, as fit
            # does. None when no provider is configured.
            exog_new = self._materialise_exog(srs_ids_b, sid_ds[active_in_batch[0]])
            args_list.append((path, new_arr, new_last_ds_arr, is_int, cfg, exog_new))
            manifest_sets.append({sid: sid_ds[sid][-1] for sid in active_in_batch})
        worker = (_multiblock_update_batch_file if self._multiblock
                  else _grid_update_batch_file)
        stub._dispatch(worker, args_list)                       # in-process or dask
        for mset in manifest_sets:
            for sid, lds in mset.items():
                self._manifest.loc[sid, "last_ds"] = lds
        self._save_manifest()
        return self

    def forecast(self, h, level=None, return_components: bool = False):
        self._check_exog_supported()
        if not self._grid_mode:
            return super().forecast(h, level, return_components)
        if not isinstance(h, int) or h <= 0:
            raise ValueError("h must be a positive integer.")
        if return_components:
            raise NotImplementedError(
                "grid-mode forecast(return_components=True) is not yet wired.")
        active_man = self._manifest[self._manifest["active"]]
        batch_ids = sorted(set(active_man["batch_id"].astype(int)))
        stub = self._grid_stub()
        cfg = self._grid_cfg()
        args_list = []
        for bid in batch_ids:
            exog_future = None
            if self.exog_provider is not None:
                # the known future design over the h dates after this batch's
                # origin, exactly as the legacy predict path builds it
                srs_ids_b, active_arr, last_ds_b = _load_batch_meta(self._batch_path(bid))
                act_sid = next(s for s, a in zip(srs_ids_b, active_arr) if a)
                exog_future = self._materialise_exog(
                    srs_ids_b, self._future_ds(last_ds_b[act_sid], h))
            args_list.append((self._batch_path(bid), h, cfg, exog_future))
        worker = (_multiblock_predict_batch_file if self._multiblock
                  else _grid_predict_batch_file)
        results = stub._dispatch(worker, args_list)             # in-process or dask
        is_int_mode = isinstance(self._freq, (int, np.integer))
        if not is_int_mode:
            off = pd.tseries.frequencies.to_offset(self._freq)
        rows = []
        for loc, sd, srs_ids_b, active_arr, last_ds_b in results:
            for j, sid in enumerate(srs_ids_b):
                if not active_arr[j]:
                    continue
                last_ds = last_ds_b[sid]
                if is_int_mode:
                    step = int(self._freq)
                    future_ds = np.arange(int(last_ds) + step,
                                          int(last_ds) + step * (h + 1), step)
                else:
                    future_ds = pd.date_range(
                        start=pd.Timestamp(last_ds) + off, periods=h, freq=self._freq)
                for k in range(h):
                    rows.append({
                        "unique_id": sid, "ds": future_ds[k],
                        self.alias: float(loc[j, k]),
                        f"{self.alias}-sd": float(sd[j, k]),
                    })
        return pd.DataFrame(rows)

    # -- grid-mode add_series (late launchers) --------------------------------
    def add_series(self, unique_id, df_history, component_priors=None,
                   wing_centre=None, weight_override=None, error_nu0=None):
        """Add a late-launching series to a grid universe (append-on-add).

        The new series is fit ALONE as a warm-started single-series ``GridBlock``
        (its own history to the current origin), then concatenated into the open
        batch's block via :meth:`GridBlock.append_series` — valid because series
        are independent in the wing grid. Warm-start: ``component_priors``
        (dict ``name -> (m0, C0)``) seeds the DLM state from siblings;
        ``wing_centre`` (logit-space scalar) seeds the wing centre at the
        analogous sibling discount. ``weight_override`` (DMA-weight prior) is not
        yet applied in grid mode."""
        if not self._grid_mode:
            return super().add_series(unique_id, df_history,
                                      component_priors=component_priors,
                                      weight_override=weight_override,
                                      error_nu0=error_nu0)
        if weight_override is not None:
            raise NotImplementedError(
                "grid-mode add_series does not yet apply weight_override.")
        if unique_id in self._manifest.index:
            raise ValueError(f"unique_id {unique_id!r} already in universe.")
        if self._multiblock:
            return self._multiblock_add_series(unique_id, df_history,
                                               component_priors, wing_centre,
                                               error_nu0)
        self._check_exog_supported()
        dfh = df_history.sort_values("ds")
        y_new = dfh["y"].to_numpy(float)[:, None]          # (T, 1)
        # the late launcher is fit ALONE, so its exog design is its own history's
        # rows for one series: (T, 1, n_regs).
        exog_new = self._materialise_exog([unique_id], dfh["ds"].to_numpy())
        newb = self._grid_block(init_data=y_new)
        newb.scan_filter(y_new, regressors=exog_new, wing_centre=wing_centre,
                         component_priors=component_priors, error_nu0=error_nu0)
        last_ds = dfh["ds"].iloc[-1]
        cap = self.max_batch_size

        def _ldv(ds, is_int):
            return int(ds) if is_int else pd.Timestamp(ds).value
        open_bid = self._open_batch_id(unique_id)
        if open_bid is None:
            # fresh batch: the new series alone, padded to capacity
            bid = self._new_batch_id()
            if cap is not None:
                newb.pad_to(cap)
            last_ds_arr, is_int = self._last_ds_arr([unique_id], {unique_id: last_ds})
            srs, active, ld = _pad_grid_meta([unique_id], [True], last_ds_arr, cap)
            _save_grid_batch_file(newb, self._batch_path(bid), srs, active, ld, is_int)
        else:
            bid = open_bid
            path = self._batch_path(bid)
            srs_ids_b, active_arr, last_ds_b = _load_batch_meta(path)
            active_arr = np.asarray(active_arr, bool)
            srs = list(srs_ids_b)
            last_ds_arr, is_int = self._last_ds_arr(srs, last_ds_b)
            openb = self._grid_block()
            openb.load(path)
            free = np.where(~active_arr)[0]
            if cap is not None and len(free):
                slot = int(free[0])                        # fill a placeholder slot
                openb.set_slot(slot, newb)
                srs[slot] = unique_id
                active_arr[slot] = True
                last_ds_arr[slot] = _ldv(last_ds, is_int)
            else:
                openb.append_series(newb)                  # no capacity: grow q
                srs.append(unique_id)
                active_arr = np.concatenate([active_arr, [True]])
                last_ds_arr = np.concatenate([last_ds_arr, [_ldv(last_ds, is_int)]])
            _save_grid_batch_file(openb, path, srs, active_arr, last_ds_arr, is_int)
        new_row = pd.DataFrame(
            {"batch_id": [bid], "last_ds": [last_ds], "active": [True]},
            index=pd.Index([unique_id], name="unique_id"))
        self._manifest = pd.concat([self._manifest, new_row])
        self._save_manifest()
        return self

    def _multiblock_add_series(self, unique_id, df_history, component_priors,
                               wing_centre, error_nu0):
        """Multi-block ``add_series``: fit fresh blocks + a union carry on the new
        series alone, then fill a placeholder slot (or append) in the open batch.
        The union ``set_slot``/``append`` is the ``AllocatorState`` analogue of
        ``GridBlock.set_slot``/``append_series``; valid by series independence
        (each series' blocks + union column are independent). ``component_priors``
        / ``wing_centre`` are per-block lists (or None) — each grid's own sibling
        warm-start; ``None`` -> diffuse."""
        nb = len(self._block_specs)
        if component_priors is not None and len(component_priors) != nb:
            raise ValueError(
                f"multi-block component_priors must be a per-block list of {nb}.")
        if wing_centre is not None and len(wing_centre) != nb:
            raise ValueError(
                f"multi-block wing_centre must be a per-block list of {nb}.")
        dfh = df_history.sort_values("ds")
        y = dfh["y"].to_numpy(float)[:, None]              # (T, 1)
        last_ds = dfh["ds"].iloc[-1]
        cap = self.max_batch_size
        cfg = self._grid_cfg()
        newb, _up, _um, _c = _grid_blocks_from_cfg(cfg, init_data=y)
        # The union carry must be built under the SAME allocator the universe
        # runs: with union_learn_dma the stored carry is the SGDDMA pytree, and
        # fitting the newcomer without it returns a fixed AllocatorState that
        # _union_set_slot's tree_map cannot merge ("Expected tuple, got
        # AllocatorState").
        new_state, new_w, _umi = _multiblock_fit(
            newb, y, self._union_pdr, self._union_mdr,
            wing_centres=wing_centre, component_priors=component_priors,
            error_nu0=error_nu0, learn_dma=self._union_learn_dma,
            dma_prior=self._union_dma_prior)
        is_int = isinstance(self._freq, (int, np.integer))

        def _ldv(ds):
            return int(ds) if is_int else pd.Timestamp(ds).value

        open_bid = self._open_batch_id(unique_id)
        if open_bid is None:
            bid = self._new_batch_id()
            if cap is not None:
                for b in newb:
                    b.pad_to(cap)
                new_state, new_w = _pad_union_state(new_state, new_w, 1, cap)
            last_ds_arr, is_int2 = self._last_ds_arr([unique_id],
                                                     {unique_id: last_ds})
            srs, active, ld = _pad_grid_meta([unique_id], [True], last_ds_arr, cap)
            _save_multiblock_batch_file(newb, new_state, new_w,
                                        self._batch_path(bid), srs, active, ld,
                                        is_int2)
        else:
            bid = open_bid
            path = self._batch_path(bid)
            srs_ids_b, active_arr, last_ds_b = _load_batch_meta(path)
            active_arr = np.asarray(active_arr, bool)
            srs = list(srs_ids_b)
            last_ds_arr, is_int2 = self._last_ds_arr(srs, last_ds_b)
            openb, ustate, uw, _cap = _load_multiblock_batch(path, cfg)
            free = np.where(~active_arr)[0]
            if cap is not None and len(free):
                slot = int(free[0])
                for bi, b in enumerate(openb):
                    b.set_slot(slot, newb[bi])
                ustate, uw = _union_set_slot(ustate, uw, slot, new_state, new_w)
                srs[slot] = unique_id
                active_arr[slot] = True
                last_ds_arr[slot] = _ldv(last_ds)
            else:
                for bi, b in enumerate(openb):
                    b.append_series(newb[bi])
                ustate, uw = _union_append(ustate, uw, new_state, new_w)
                srs.append(unique_id)
                active_arr = np.concatenate([active_arr, [True]])
                last_ds_arr = np.concatenate([last_ds_arr, [_ldv(last_ds)]])
            _save_multiblock_batch_file(openb, ustate, uw, path, srs, active_arr,
                                        last_ds_arr, is_int2)
        new_row = pd.DataFrame(
            {"batch_id": [bid], "last_ds": [last_ds], "active": [True]},
            index=pd.Index([unique_id], name="unique_id"))
        self._manifest = pd.concat([self._manifest, new_row])
        self._save_manifest()
        return self

    def _multiblock_add_series_many(self, unique_ids, df_histories,
                                    component_priors, wing_centre, error_nu0):
        """Batched multi-block ``add_series``: fit every newcomer, then write
        each affected batch ONCE.

        The multi-block analogue of the single-block batched path below.
        Looping :meth:`add_series` reloads and rewrites the WHOLE batch per
        series, which dominates at production batch sizes — a 4000-slot
        two-block M5 batch is ~281 MB and ~3.5 s to rewrite over NFS, against
        ~2.4 s to fit the series — so a 34-launcher origin pays that 34 times.
        One save per affected batch amortises it over the origin's launchers.

        Equivalent to looping :meth:`add_series`: same per-series fits, same
        slot assignment order, same resulting carry. ``k == 1`` delegates to
        the single-series path.
        """
        unique_ids, df_histories = list(unique_ids), list(df_histories)
        k = len(unique_ids)
        if k == 0:
            return self

        def _pick(seq, i):
            return seq[i] if seq is not None else None
        if k == 1:
            return self.add_series(unique_ids[0], df_histories[0],
                                   component_priors=_pick(component_priors, 0),
                                   wing_centre=_pick(wing_centre, 0),
                                   error_nu0=_pick(error_nu0, 0))
        for uid in unique_ids:
            if uid in self._manifest.index:
                raise ValueError(f"unique_id {uid!r} already in universe.")
        self._check_exog_supported()
        cfg = self._grid_cfg()
        cap = self.max_batch_size
        is_int = isinstance(self._freq, (int, np.integer))

        def _ldv(ds):
            return int(ds) if is_int else pd.Timestamp(ds).value

        # 1. Fit each newcomer alone, on its OWN calendar: its blocks + the
        #    union carry over the same window (exactly _multiblock_add_series).
        fitted = []                     # (uid, blocks, ustate, uweights, last_ds)
        for i, uid in enumerate(unique_ids):
            dfh = df_histories[i].sort_values("ds")
            y_i = dfh["y"].to_numpy(float)[:, None]           # (T, 1)
            nb_i, _up, _um, _c = _grid_blocks_from_cfg(cfg, init_data=y_i)
            st_i, w_i, _umi = _multiblock_fit(
                nb_i, y_i, self._union_pdr, self._union_mdr,
                wing_centres=_pick(wing_centre, i),
                component_priors=_pick(component_priors, i),
                error_nu0=_pick(error_nu0, i),
                learn_dma=self._union_learn_dma,
                dma_prior=self._union_dma_prior)
            fitted.append((uid, nb_i, st_i, w_i, dfh["ds"].iloc[-1]))

        assign = {}
        remaining = list(fitted)

        # 2. Fill the open batch (ONE save). Capacity mode fills placeholder
        #    slots; no-cap appends (grows q).
        open_bid = self._open_batch_id()
        if open_bid is not None and remaining:
            path = self._batch_path(open_bid)
            srs_b, active_b, ldm = _load_batch_meta(path)
            srs = list(srs_b)
            active = np.asarray(active_b, bool)
            last_ds_arr, is_int_b = self._last_ds_arr(srs, ldm)
            openb, ustate, uw, _cap = _load_multiblock_batch(path, cfg)
            if cap is not None:
                free = list(np.where(~active)[0])
                while remaining and free:
                    slot = free.pop(0)
                    uid, nbk, st, w, ld = remaining.pop(0)
                    for bi, b in enumerate(openb):
                        b.set_slot(slot, nbk[bi])
                    ustate, uw = _union_set_slot(ustate, uw, slot, st, w)
                    srs[slot] = uid
                    active[slot] = True
                    last_ds_arr[slot] = _ldv(ld)
                    assign[uid] = open_bid
            else:
                for uid, nbk, st, w, ld in remaining:
                    for bi, b in enumerate(openb):
                        b.append_series(nbk[bi])
                    ustate, uw = _union_append(ustate, uw, st, w)
                    srs.append(uid)
                    active = np.append(active, True)
                    last_ds_arr = np.append(last_ds_arr, _ldv(ld))
                    assign[uid] = open_bid
                remaining = []
            _save_multiblock_batch_file(openb, ustate, uw, path, srs, active,
                                        last_ds_arr, is_int_b)

        # 3. Anything left -> fresh batches, padded to cap (ONE save each).
        while remaining:
            chunk = remaining[:cap] if cap is not None else remaining
            remaining = remaining[len(chunk):]
            bid = self._new_batch_id()
            uid0, blocks, ustate, uw, ld0 = chunk[0]
            uids, lds = [uid0], [ld0]
            for uid, nbk, st, w, ld in chunk[1:]:
                for bi, b in enumerate(blocks):
                    b.append_series(nbk[bi])
                ustate, uw = _union_append(ustate, uw, st, w)
                uids.append(uid)
                lds.append(ld)
            if cap is not None:
                for b in blocks:
                    b.pad_to(cap)
                ustate, uw = _pad_union_state(ustate, uw, len(uids), cap)
            last_ds_arr, is_int_b = self._last_ds_arr(
                uids, {u: l for u, l in zip(uids, lds)})
            srs, active, ld_arr = _pad_grid_meta(uids, [True] * len(uids),
                                                 last_ds_arr, cap)
            _save_multiblock_batch_file(blocks, ustate, uw, self._batch_path(bid),
                                        srs, active, ld_arr, is_int_b)
            for u in uids:
                assign[u] = bid

        last_ds_map = {uid: ld for uid, _b, _s, _w, ld in fitted}
        new_rows = pd.DataFrame(
            {"batch_id": [assign[u] for u in unique_ids],
             "last_ds": [last_ds_map[u] for u in unique_ids],
             "active": [True] * k},
            index=pd.Index(unique_ids, name="unique_id"))
        self._manifest = pd.concat([self._manifest, new_rows])
        self._save_manifest()
        return self

    def add_series_many(self, unique_ids, df_histories, component_priors=None,
                        wing_centre=None, weight_override=None, error_nu0=None):
        """Add several late launchers in one pass (one batch write per affected
        batch). Mirrors :meth:`add_series`; each series is fit independently and
        appended, so the result equals looping :meth:`add_series`."""
        if not self._grid_mode:
            return super().add_series_many(unique_ids, df_histories,
                                           component_priors=component_priors,
                                           weight_override=weight_override,
                                           error_nu0=error_nu0)
        if weight_override is not None:
            raise NotImplementedError(
                "grid-mode add_series_many does not yet apply weight_override.")
        if self._multiblock:
            return self._multiblock_add_series_many(
                unique_ids, df_histories, component_priors, wing_centre,
                error_nu0)
        unique_ids, df_histories = list(unique_ids), list(df_histories)
        k = len(unique_ids)
        if k == 0:
            return self

        def _pick(seq, i):
            return seq[i] if seq is not None else None
        if k == 1:
            return self.add_series(unique_ids[0], df_histories[0],
                                   component_priors=_pick(component_priors, 0),
                                   wing_centre=_pick(wing_centre, 0),
                                   error_nu0=_pick(error_nu0, 0))
        for uid in unique_ids:
            if uid in self._manifest.index:
                raise ValueError(f"unique_id {uid!r} already in universe.")
        self._check_exog_supported()
        fitted = []                                   # (uid, block, last_ds)
        for i, uid in enumerate(unique_ids):
            dfh = df_histories[i].sort_values("ds")
            y_i = dfh["y"].to_numpy(float)[:, None]        # (T, 1)
            b = self._grid_block(init_data=y_i)
            # each launcher is fit alone on its OWN calendar, so the design is
            # materialised per series rather than once for the group.
            b.scan_filter(y_i,
                          regressors=self._materialise_exog(
                              [uid], dfh["ds"].to_numpy()),
                          wing_centre=_pick(wing_centre, i),
                          component_priors=_pick(component_priors, i),
                          error_nu0=_pick(error_nu0, i))
            fitted.append((uid, b, dfh["ds"].iloc[-1]))
        cap = self.max_batch_size

        def _ldv(ds, is_int):
            return int(ds) if is_int else pd.Timestamp(ds).value
        assign = {}
        remaining = list(fitted)

        # 1. Fill the open batch (one save). Capacity mode fills placeholder slots;
        #    no-cap appends (grows q).
        open_bid = self._open_batch_id()
        if open_bid is not None:
            path = self._batch_path(open_bid)
            srs, active, ldm = _load_batch_meta(path)
            srs = list(srs)
            active = np.asarray(active, bool)
            last_ds_arr, is_int = self._last_ds_arr(srs, ldm)
            block = self._grid_block()
            block.load(path)
            if cap is not None:
                free = list(np.where(~active)[0])
                while remaining and free:
                    slot = free.pop(0)
                    uid, nb, ld = remaining.pop(0)
                    block.set_slot(slot, nb)
                    srs[slot] = uid; active[slot] = True
                    last_ds_arr[slot] = _ldv(ld, is_int)
                    assign[uid] = open_bid
            else:
                for uid, nb, ld in remaining:
                    block.append_series(nb)
                    srs.append(uid); active = np.append(active, True)
                    last_ds_arr = np.append(last_ds_arr, _ldv(ld, is_int))
                    assign[uid] = open_bid
                remaining = []
            _save_grid_batch_file(block, path, srs, active, last_ds_arr, is_int)

        # 2. Remaining singletons -> fresh batches, padded to cap (one save each).
        while remaining:
            chunk = remaining[:cap] if cap is not None else remaining
            remaining = remaining[len(chunk):]
            bid = self._new_batch_id()
            uid0, block, ld0 = chunk[0]
            uids, lds = [uid0], [ld0]
            for uid, nb, ld in chunk[1:]:
                block.append_series(nb)
                uids.append(uid); lds.append(ld)
            if cap is not None:
                block.pad_to(cap)
            last_ds_arr, is_int = self._last_ds_arr(
                uids, {u: l for u, l in zip(uids, lds)})
            srs, active, ld_arr = _pad_grid_meta(uids, [True] * len(uids),
                                                 last_ds_arr, cap)
            _save_grid_batch_file(block, self._batch_path(bid), srs, active, ld_arr, is_int)
            for u in uids:
                assign[u] = bid

        last_ds_map = {uid: ld for uid, _b, ld in fitted}
        new_rows = pd.DataFrame(
            {"batch_id": [assign[u] for u in unique_ids],
             "last_ds": [last_ds_map[u] for u in unique_ids],
             "active": [True] * k},
            index=pd.Index(unique_ids, name="unique_id"))
        self._manifest = pd.concat([self._manifest, new_rows])
        self._save_manifest()
        return self


# -----------------------------------------------------------------------------
# CLI (only when this file is executed as a script, not on import)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify device from CMD")
    parser.add_argument(
        "--compute",
        required=False,
        default="cpu",
        help="cpu or gpu",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--compute",
        default=None,
        choices=["cpu", "gpu"],
        help="Target compute platform; auto-detects if not specified.",
    )
    args = parser.parse_args()
    configure_devices(args.compute, args.device_id)

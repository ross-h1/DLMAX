"""Factory functions for FFS model universes.

Provides pre-configured ``DLM`` instances with list-valued tunables giving the
standard discount-rate grids, built on ``DLM.compile_universe`` and the
``sweep`` primitive.
"""

from typing import Optional

import numpy as np
import pandas as pd

from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Fourier
from DLMAX.ffs.sweep import SweepDim


# ---------------------------------------------------------------------------
# Discount-rate grids (mirrors ffs_core's module-level tables)
# ---------------------------------------------------------------------------
# Imported into the factory rather than redefined so the legacy grid
# stays the single source of truth. Future cleanup can move these here
# and have ffs_core import from this module instead.

from DLMAX.ffs_core import (
    LT_dr,  # level-trend discount table
    S_dr,  # seasonal discount table
    var_dr,  # variance discount table
    trend as _trend_damping_table,  # trend damping table (NaN-aware)
    _LT_VALUES,
)

#: Structural trend-damping grid for ``make_ffs_universe`` (Td=0.75 omitted by
#: default — the empirical default). Module-level so an experiment can override
#: it (e.g. prepend 1.0 for an undamped trend) without editing the function; the
#: default value is unchanged so the default universe is bit-identical.
_STRUCT_DAMPING_GRID = [0.99, 0.95, None]
#: damping value -> legacy key char. Keyed by VALUE (not grid index), so existing
#: keys are stable when the grid is extended. '1.0' (undamped, 'u') is included
#: for override use but absent from the default grid.
_STRUCT_DAMPING_KEY = {1.0: "u", 0.99: "0", 0.95: "1", 0.75: "2", None: "n"}


# ---------------------------------------------------------------------------
# Legacy key formatter
# ---------------------------------------------------------------------------


def _legacy_keyfn(prefix: str, lt_index, s_index, d_index, v_index):
    """Build a legacy "AL{LT}T{D}S{S}E{V}" / "ML..." key.

    Each *_index is a string from the corresponding grid table's index.
    """
    return f"{prefix}L{lt_index}T{d_index}S{s_index}E{v_index}"


def _make_legacy_keyfn_for_universe(prefix: str, lt_keys, s_keys, d_keys, v_keys):
    """Return a `keyfn` matching `compile_universe`'s callable contract.

    The factory's swept dims are (in this order):
      trend.disc_rate    (LT axis)
      trend.damping      (D axis; includes NaN for "no trend")
      seasonal.disc_rate (S axis; includes None for inert)
      error.disc_rate    (V axis)
    """

    def keyfn(cell, dims, indices):
        # indices is parallel to dims; use the index strings from each
        # grid table to reconstruct the legacy short codes.
        idx_by_param = {(d.component, d.param): i for d, i in zip(dims, indices)}
        lt_i = idx_by_param[("trend", "disc_rate")]
        d_i = idx_by_param[("trend", "damping")]
        v_i = idx_by_param[("error", "disc_rate")]
        # Seasonal dim is absent when periodicity is None (no Fourier) —
        # default to the inert key "n".
        s_i = idx_by_param.get(("seasonal", "disc_rate"))
        s_key = s_keys[s_i] if s_i is not None else "n"
        return _legacy_keyfn(
            prefix,
            lt_keys[lt_i],
            s_key,
            d_keys[d_i],
            v_keys[v_i],
        )

    return keyfn


# ---------------------------------------------------------------------------
# Constraint: legacy LT_disc <= S_disc filter
# ---------------------------------------------------------------------------


def _legacy_constraint(cell):
    """Reproduce the legacy `LT_dr <= S_dr` filter for active-seasonal cells.

    Legacy `assemble_models`:
        if S == "n":  always include
        elif LT_dr.loc[LT, mclass] <= S_dr.loc[S, mclass]: include
        else: skip

    With list-swept disc_rates, the cell carries the *value* (not the
    index), and `seasonal.disc_rate is None` signals the inert case.
    """
    s_value = cell.get("seasonal", {}).get("disc_rate")
    if s_value is None:                       # inert OR no Fourier component
        return True
    return cell["trend"]["disc_rate"] <= s_value


def _trend_seasonal_light_damping_only(cell):
    """Grid constraint: when both trend and seasonal are active, restrict to
    Td=0.99 (only the lightest damping survives). Cells without an active
    trend or without an active seasonal are unaffected."""
    import math

    td = cell.get("trend", {}).get("damping")
    has_trend = td is not None and not (isinstance(td, float) and math.isnan(td))
    has_seasonal = cell.get("seasonal", {}).get("disc_rate") is not None
    if has_trend and has_seasonal:
        return td >= 0.99
    return True


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def make_ffs_universe(
    *,
    periodicity: Optional[int],
    n_seas_comps: Optional[int],
    n_series: int,
    mult_models: str = "A",
    component_priors: Optional[dict] = None,
    error_nu0: Optional[float] = None,
    monitor_tau: Optional[float] = None,
) -> DLM:
    """Pre-configured DLM with list-valued tunables matching the legacy grid.

    Parameters
    ----------
    periodicity : int
        Seasonal period passed to the Fourier component.
    n_seas_comps : int or None
        Number of Fourier harmonic pairs. ``None`` for full-rank
        Fourier (state_dim = period - 1).
    n_series : int
        Number of series the resulting models will be fit to.
    mult_models : {'A', 'M'}, default 'A'
        ``'A'`` builds the additive universe (Fourier multiplicative=False,
        var_power=1). ``'M'`` builds the multiplicative universe
        (Fourier multiplicative=True, var_power=0).

    Returns
    -------
    DLM
        Ready for ``.compile_universe(init_data, h=...)``. The default
        constraint ``_legacy_constraint`` (LT_disc ≤ S_disc) is *not*
        baked in — pass it explicitly to compile_universe to reproduce
        the legacy grid filtering. Pass ``constraint=None`` for the
        full Cartesian product.

    Notes
    -----
    The seasonal disc_rate list contains ``None`` as its last element
    to encode the inert seasonal case (``S == "n"`` in legacy keys).
    """
    if mult_models not in ("A", "M"):
        raise ValueError(f"mult_models must be 'A' or 'M', got {mult_models!r}.")

    multiplicative = mult_models == "M"
    var_power = 0.0 if multiplicative else 1.0

    # Discount-rate grids. The list orders here pin the index→value
    # mapping that the legacy keyfn relies on.
    lt_disc_values = list(_LT_VALUES)  # [0.9999, 0.99, 0.95, 0.9, 0.75]
    seasonal_disc_values = list(_LT_VALUES[:3]) + [None]  # [0.9999, 0.99, 0.95, None]
    var_disc_values = [0.99, 1.0]

    # Trend-damping grid: module-level _STRUCT_DAMPING_GRID (overridable; default
    # [0.99, 0.95, None], Td=0.75 omitted). The damping->key map is keyed by value
    # so trend_damping_keys maps correctly for any grid (incl. an undamped 1.0).
    _DAMPING_KEY = _STRUCT_DAMPING_KEY
    damping_grid = list(_STRUCT_DAMPING_GRID)
    trend_damping_values = [v if v is not None else float("nan") for v in damping_grid]
    trend_damping_keys = [_DAMPING_KEY[v] for v in damping_grid]

    # Optional per-component informative priors (m0, C0), keyed by component
    # name. None (the default) leaves the component to elicit/diffuse as before,
    # so the structural path is unchanged. Used by AutoFFSUniverse.add_series to
    # warm-start a new series from a hierarchical prior.
    cp = component_priors or {}
    trend_m0, trend_C0 = cp.get("trend", (None, None))
    seas_m0, seas_C0 = cp.get("seasonal", (None, None))

    dlm = DLM(family="Gaussian", n_series=n_series)
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=lt_disc_values,
            damping=trend_damping_values,
            m0=trend_m0,
            C0=trend_C0,
        )
    )
    # Seasonality only when a periodicity is given. periodicity=None (yearly)
    # mirrors the legacy mcomp_dlm no-Fourier branch: no seasonal component, no
    # seasonal disc sweep -> universe is level-only + (damped-)trend classes.
    if periodicity is not None:
        dlm.add_component(
            Fourier(
                name="seasonal",
                period=int(periodicity),
                n_comps=n_seas_comps,
                disc_rate=seasonal_disc_values,
                multiplicative=multiplicative,
                m0=seas_m0,
                C0=seas_C0,
            )
        )
    dlm.set_error(
        disc_rate=var_disc_values,
        power=var_power,
        nu0=1.0 if error_nu0 is None else float(error_nu0),
    )

    # Signed-error monitor: a scalar tau (SD multiple, e.g. 3.0) turns the one-off
    # variance-injection intervention ON for EVERY model in the static grid (not a
    # sweep axis — the whole universe carries the same monitor). None -> off
    # (legacy static path, bit-exact). See dlm_core._multi_fwd_filter_step.
    if monitor_tau is not None:
        dlm.set_monitor(float(monitor_tau))

    # NOTE (future work): multiplicative-trend universes (MMM, MMN) would be
    # added via a `multiplicative` flag on LocalTrend (mirroring
    # Fourier.multiplicative), not as a separate factory. Out of scope here.

    # Stash the metadata assemble_models needs for legacy descriptor
    # translation. trend_damping_keys reflects the damping grid order so the
    # keyfn index->key mapping stays correct.
    dlm.__ffs_universe_meta__ = {
        "prefix": mult_models,
        "lt_disc_keys": list(LT_dr.index),  # ['0', '1', '2', '3', '4']
        "trend_damping_keys": trend_damping_keys,  # grid order; omits '2' (Td=0.75)
        "seasonal_disc_keys": list(S_dr.index),  # ['0', '1', '2', 'n']
        "var_disc_keys": list(var_dr.index),  # ['0', '1']
        "multiplicative": multiplicative,
        "var_power": var_power,
        "has_seasonal": periodicity is not None,
    }
    return dlm


# ---------------------------------------------------------------------------
# Adaptive (error-monitoring) universe
# ---------------------------------------------------------------------------
#
# The parallel sibling of make_ffs_universe. It replaces the legacy LT-discount
# grid (the [1, .97, .9, .75, .5] forgetting axis) with a *tau* grid: the
# error-monitoring sensitivity (the reused `monitor` slot). Every tau-model is
# identical in calm (S <= tau: discount = damping-only, no extra forgetting) and
# they diverge only on a shock, where each forgets towards its per-component
# d_min (level/trend 0.5, seasonal 0.8). DMA then selects sensitivity online.
#
# Variance stays a STATIC axis (legacy [0.99, 1.0]), NOT a d_min: the dlm_core
# adaptive path drives only the state-discount matrix; variance_disc passes
# through unchanged. This is deliberate ("let level lead variance" — making the
# observation variance adaptive too creates a feedback loop where var widening
# shrinks the standardised error z and damps the very signal that should drive
# forgetting). A true adaptive var d_min is a v2 dlm_core extension.
#
# Based on the standard universe (damping grid [0.99, 0.95, None]; light-damping
# constraint when trend and seasonal are both active). The legacy LT<=S
# constraint is dropped — meaningless once the LT axis is a scalar d_min.

# Defaults (locked design, see adaptive-discount-monitor memory).
_TAU_VALUES = [1.0, 1.5, 2.0, 3.0, 5.0]   # monitor sensitivity (fast -> slow)
_DMIN_LEVEL_TREND = 0.5                    # level/trend shock floor
_DMIN_SEASONAL = 0.8                       # seasonal shock floor (shape persists)
# STATIC variance discount — held IDENTICAL to the legacy production grid
# (var_dr = {0.99, 1.0}) so the adaptive-vs-legacy comparison isolates the
# state-forgetting mechanism and is not confounded by a variance change.
_VAR_DISC_VALUES = [0.99, 1.0]


def _adaptive_constraint(cell):
    """Grid constraint for the adaptive universe: light-damping only.

    No LT<=S filter (LT is a scalar d_min, not a swept discount). When both
    trend and seasonal are active, restrict to the lightest damping (Td>=0.99),
    exactly as the standard grid does.
    """
    return _trend_seasonal_light_damping_only(cell)


def _make_adaptive_keyfn_for_universe(prefix, tau_keys, s_keys, d_keys, v_keys):
    """Return a `keyfn` for the adaptive universe: "{prefix}K{tau}T{D}S{S}E{V}".

    The 'K' (kappa/monitor) axis replaces the legacy 'L' (LT-discount) axis.
    Swept dims are trend.damping (D), seasonal.disc_rate (S, may be absent),
    error.disc_rate (V) and monitor.tau (K).
    """

    def keyfn(cell, dims, indices):
        idx_by_param = {(d.component, d.param): i for d, i in zip(dims, indices)}
        k_i = idx_by_param[("monitor", "tau")]
        d_i = idx_by_param[("trend", "damping")]
        v_i = idx_by_param[("error", "disc_rate")]
        s_i = idx_by_param.get(("seasonal", "disc_rate"))
        s_key = s_keys[s_i] if s_i is not None else "n"
        return f"{prefix}K{tau_keys[k_i]}T{d_keys[d_i]}S{s_key}E{v_keys[v_i]}"

    return keyfn


def make_ffs_universe_adaptive(
    *,
    periodicity: Optional[int],
    n_seas_comps: Optional[int],
    n_series: int,
    mult_models: str = "A",
    tau_values: Optional[list] = None,
    dmin_level_trend: float = _DMIN_LEVEL_TREND,
    dmin_seasonal: float = _DMIN_SEASONAL,
    var_disc_values: Optional[list] = None,
    component_priors: Optional[dict] = None,
    error_nu0: Optional[float] = None,
) -> DLM:
    """Adaptive (error-monitoring) sibling of :func:`make_ffs_universe`.

    Builds a DLM whose forgetting is online and error-driven
    instead of a fixed per-model discount. The swept axes are:

    - ``monitor.tau``    — sensitivity grid (``tau_values``), the new universe
      axis replacing the legacy LT-discount grid.
    - ``trend.damping``  — damping grid [0.99, 0.95, None]; in the
      adaptive path this sets the *calm-regime* discount (the discount when
      S <= tau), with tau governing how fast a shock pulls it to d_min.
    - ``seasonal.disc_rate`` — [dmin_seasonal, None]: seasonal active (d_min)
      vs inert. Only present when ``periodicity`` is given.
    - ``error.disc_rate`` — STATIC variance discount grid (``var_disc_values``,
      defaults to the legacy [0.99, 1.0] so the comparison is unconfounded).
      Not adaptive — see module note.

    Level/trend states take the scalar ``dmin_level_trend`` as their shock
    floor (read as d_min by the adaptive path). The returned DLM has
    ``set_monitor(tau_values)`` applied, so ``compile_universe`` sweeps tau and
    stamps ``monitor=tau`` on each compiled ``uv_dlm`` (which stacks into
    ``multi.monitor`` for the FFS scan to read as ``tau``).

    Returns the DLM ready for ``.compile_universe(...)``. The Cut-2 light-
    damping constraint is *not* baked in — pass ``_adaptive_constraint`` to
    compile_universe (the adaptive shim does this).
    """
    if mult_models not in ("A", "M"):
        raise ValueError(f"mult_models must be 'A' or 'M', got {mult_models!r}.")

    multiplicative = mult_models == "M"
    var_power = 0.0 if multiplicative else 1.0

    tau_values = list(_TAU_VALUES if tau_values is None else tau_values)
    # tau must be strictly positive: the scan discriminates adaptive-vs-static on
    # a positive monitor value (legacy monitor=False stacks to 0), so tau<=0
    # would be silently misread as "no monitoring".
    if any(t <= 0 for t in tau_values):
        raise ValueError(f"tau_values must be strictly positive, got {tau_values}.")
    var_disc_values = list(_VAR_DISC_VALUES if var_disc_values is None else var_disc_values)

    # Seasonal on/off axis: active d_min vs inert (None). Mirrors the legacy
    # inert-seasonal encoding (None -> "n" key).
    seasonal_disc_values = [float(dmin_seasonal), None]

    # Damping grid
    _DAMPING_KEY = {0.99: "0", 0.95: "1", None: "n"}
    damping_grid = [0.99, 0.95, None]
    trend_damping_values = [v if v is not None else float("nan") for v in damping_grid]
    trend_damping_keys = [_DAMPING_KEY[v] for v in damping_grid]

    cp = component_priors or {}
    trend_m0, trend_C0 = cp.get("trend", (None, None))
    seas_m0, seas_C0 = cp.get("seasonal", (None, None))

    dlm = DLM(family="Gaussian", n_series=n_series)
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=float(dmin_level_trend),   # scalar d_min (not swept)
            damping=trend_damping_values,        # swept (calm discount)
            m0=trend_m0,
            C0=trend_C0,
        )
    )
    if periodicity is not None:
        dlm.add_component(
            Fourier(
                name="seasonal",
                period=int(periodicity),
                n_comps=n_seas_comps,
                disc_rate=seasonal_disc_values,  # swept: d_min vs inert
                multiplicative=multiplicative,
                m0=seas_m0,
                C0=seas_C0,
            )
        )
    dlm.set_error(
        disc_rate=var_disc_values,               # swept STATIC variance discount
        power=var_power,
        nu0=1.0 if error_nu0 is None else float(error_nu0),
    )
    # The tau (monitor) sweep axis — the heart of the adaptive universe.
    dlm.set_monitor(tau_values)

    # Tau key strings: index order pins the keyfn mapping (e.g. ['0','1',...]).
    tau_keys = [str(i) for i in range(len(tau_values))]
    var_disc_keys = [str(i) for i in range(len(var_disc_values))]
    seasonal_disc_keys = ["0", "n"]              # active d_min, inert

    dlm.__ffs_universe_meta__ = {
        "adaptive": True,
        "prefix": mult_models,
        "tau_values": tau_values,
        "tau_keys": tau_keys,
        "dmin_level_trend": float(dmin_level_trend),
        "dmin_seasonal": float(dmin_seasonal),
        "trend_damping_keys": trend_damping_keys,
        "seasonal_disc_keys": seasonal_disc_keys,
        "var_disc_values": var_disc_values,
        "var_disc_keys": var_disc_keys,
        "multiplicative": multiplicative,
        "var_power": var_power,
        "has_seasonal": periodicity is not None,
    }
    return dlm


def make_additive_var_power_set(
    *,
    periodicity: int,
    n_seas_comps: Optional[int],
    n_series: int,
) -> DLM:
    """Additive-only universe with both var_power values swept.

    For DMA universes that don't include multiplicative seasonality but
    do want both linear and log-scale variance evolutions. Fourier is
    always additive; ``var_power=[0, 1]`` is swept.

    Returns
    -------
    DLM
        Ready for ``.compile_universe``. Cell keys for this factory
        use the structured default format (``trend.D0|...|error.D0.P0``)
        rather than a legacy-compatible short form, since the legacy
        grid never exercised additive×var_power=0.
    """
    lt_disc_values = list(_LT_VALUES)
    seasonal_disc_values = list(_LT_VALUES[:3]) + [None]  # [1, 0.97, 0.9, None]
    trend_damping_values = [
        v if v is not None else float("nan") for v in [0.99, 0.95, 0.75, None]
    ]
    var_disc_values = [0.99, 1.0]
    var_power_values = [0.0, 1.0]

    dlm = DLM(family="Gaussian", n_series=n_series)
    dlm.add_component(
        LocalTrend(
            name="trend",
            disc_rate=lt_disc_values,
            damping=trend_damping_values,
        )
    )
    dlm.add_component(
        Fourier(
            name="seasonal",
            period=int(periodicity),
            n_comps=n_seas_comps,
            disc_rate=seasonal_disc_values,
            multiplicative=False,
        )
    )
    dlm.set_error(
        disc_rate=var_disc_values,
        power=var_power_values,
        nu0=1.0,
    )
    return dlm

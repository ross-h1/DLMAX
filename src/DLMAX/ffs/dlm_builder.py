"""Component-based DLM construction.

A user-facing builder API for constructing :class:`~DLMAX.dlm_core.uv_dlm`
instances by composing typed components (level, trend, seasonal). The
builder produces output identical to :func:`~DLMAX.ffs_core.Mcomp_DLM`
for the parameter combinations the latter supports, and exposes a
larger configuration space (e.g. multiplicative trend, level-only
models) by removing the bespoke conditionals in ``Mcomp_DLM``.

Usage
-----
>>> from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Fourier
>>> dlm = DLM(family='Gaussian', n_series=3)
>>> dlm.add_component(LocalTrend(name='trend', disc_rate=0.99, damping=0.95))
>>> dlm.add_component(Fourier(name='annual', period=12, n_comps=2,
...                           disc_rate=0.95))
>>> dlm.set_error(disc_rate=1.0, power=1, nu0=1)
>>> uvdlm = dlm.compile(init_data)

Notes
-----
Priors are elicited either from ``initial_state`` or from a diffuse prior
settled over ``warmup_steps``; the builder API accommodates both, and
:meth:`DLM.compile` selects on which was supplied.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.linalg import block_diag

from DLMAX.dlm_core import fourier_FG, uv_dlm, vdiag
from DLMAX.ffs import devices
from DLMAX.ffs_core import initial_state

# Error-monitor one-off variance-injection discounts (the revised grid rate fed
# into the guard when the signed-error monitor triggers). Level/trend get a hard
# 0.5 kick; seasonal a gentler 0.8. See Component.monitor_inject_block and
# dlm_core._multi_fwd_filter_step (intervention branch).
MONITOR_INJECT_LT = 0.5
MONITOR_INJECT_SEAS = 0.8


@dataclass(frozen=True)
class Adapt:
    """Learn this discount online (RTRL) from a starting value.

    A single point θ moved by gradient descent on the streaming log score —
    the *continuous* (SGD) axis of the discount DSL, as opposed to a fixed
    scalar or a swept ``[list]`` (the discrete / DMA axis).

    >>> LocalTrend(disc_rate=Adapt(0.95))   # learn the level/trend discount
    """
    init: float = 0.95


@dataclass(frozen=True)
class Wing:
    """Learn a discount CENTRE online with wingmen straddling it.

    ``wings`` members sit at ``centre ± offset`` (logit space); the centre is
    moved by RTRL from the wingmen's pooled scores, and the wingmen are also
    DMA members. The both-axes cell (one ``GridModel`` cell): discrete
    (wingmen → DMA) and continuous (centre → RTRL) at once.

    >>> LocalTrend(disc_rate=Wing(0.95, offset=1.0))
    """
    init: float = 0.95
    offset: float = 1.0
    wings: int = 3


def _as_list_or_scalar(value, name: str):
    """Classify a tunable param as a learned spec, sweep-list, or scalar.

    Returns ``("adapt", spec)`` / ``("wing", spec)`` for the learned-discount
    sentinels, ``("list", value)`` for a Python list (a sweep dim), else
    ``("scalar", value)``. Tuples / ndarrays / generators are treated as
    scalars by design: only a list marks a sweep dimension, so an array-valued
    parameter stays a single value.
    """
    if isinstance(value, Adapt):
        return ("adapt", value)
    if isinstance(value, Wing):
        return ("wing", value)
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(
                f"{name}: empty list is not a valid sweep dim. Pass a "
                "scalar, or a list with at least one element."
            )
        return ("list", value)
    return ("scalar", value)


def _spec_base(kind, value):
    """The concrete value a *compiled* model uses for a param spec: the ``init``
    for a learned (``adapt``/``wing``) spec, the value itself for a scalar, and
    ``None`` for a sweep list (resolved per-cell by ``compile_universe``)."""
    if kind in ("adapt", "wing"):
        return value.init
    if kind == "scalar":
        return value
    return None


def _adapt_geometry(components, var_power, error_disc):
    """θ-geometry for an ``Adapt`` model — the discount-block layout.

    A component whose ``Adapt`` init is a **per-state tuple** (length == its
    ``state_dim``) gets one learned block *per state* (e.g. level and trend
    discounts learned separately); a scalar ``Adapt`` gets a single **fused**
    block (one shared discount); a fixed component gets a fused block that is not
    learned. Returns ``(GridModel, theta0 (P,), learn (P,) bool)`` with
    ``θ = [δ_block_0 … δ_block_{n_blocks-1}, β]``.
    """
    import numpy as _np
    from jax.scipy.linalg import block_diag as _bd
    from .discount_grid import GridModel

    F = jnp.concatenate([c.F_block() for c in components])
    G = _bd(*[c.G_block() for c in components])
    mc = jnp.concatenate([c.mult_block() for c in components])

    s2b, theta0, learn, b = [], [], [], 0
    for c in components:
        sd = c.state_dim
        kind = getattr(c, "_disc_rate_kind", "scalar")
        dr = c.disc_rate
        per_state = (kind == "adapt"
                     and isinstance(dr, (tuple, list, np.ndarray))
                     and len(_np.atleast_1d(dr)) == sd and sd > 1)
        if per_state:                       # one learned block per state
            for x in _np.atleast_1d(dr):
                s2b.append(b); theta0.append(float(x)); learn.append(True); b += 1
        else:                               # one fused block for the component
            s2b += [b] * sd
            theta0.append(float(_np.mean(_np.asarray(dr))))
            learn.append(kind == "adapt")
            b += 1
    theta0.append(float(error_disc)); learn.append(False)      # β, not learned
    gm = GridModel(F=F, G=G, mc=mc, var_power=var_power,
                   state_to_block=jnp.asarray(s2b, dtype=int), n_blocks=b)
    return gm, jnp.asarray(theta0), _np.asarray(learn, dtype=bool)


def _component_param_names(component) -> list:
    """Return the tunable-param names a component declares.

    Each component class declares ``_param_names``; this helper just
    indirects through the class so the descriptor builder doesn't
    need to know the class hierarchy.
    """
    return list(getattr(type(component), "_param_names", ()))


# ---------------------------------------------------------------------------
# Prior context
# ---------------------------------------------------------------------------


class PriorContext:
    """Bundle of data-derived inputs needed for component prior elicitation.

    Constructed once per :meth:`DLM.compile` call. Components query it
    rather than receiving raw data, which (a) means ``initial_state``
    runs only once and (b) handles the cross-component dependency
    needed by multiplicative seasonal components (which read the level
    prior).
    """

    def __init__(
        self,
        init_data: pd.DataFrame,
        components: List["Component"],
        diffuse_only: bool = False,
    ):
        self.init_data = init_data
        self.n_series = init_data.shape[1]

        # Diffuse-prior summary stats — always computed; cheap.
        Y = jnp.asarray(init_data.values)  # (T, n)
        # NaN-aware moments: pre-launch / missing periods are encoded as
        # NaN (skipped by the filter via ignore_obs) and must not poison
        # the prior. Callers are responsible for ensuring each series has
        # at least one non-NaN observation (enforced in _prepare_input).
        self._diffuse_mu = jnp.nanmean(Y, axis=0)  # (n,)
        self._diffuse_sigma2 = jnp.maximum(jnp.nanvar(Y, axis=0), 1e-6)  # (n,); floor avoids V0=C0=0 on flat warmup
        self._diffuse_sigma2_d = jnp.nanvar(jnp.diff(Y, axis=0), axis=0)
        self._diffuse_cv2 = self._diffuse_sigma2 / jnp.maximum(
            self._diffuse_mu**2, 1e-12
        )

        # Legacy elicitation — only run if the diffuse path is not being used.
        # ``initial_state`` is bounded/regression-based and only well-tested
        # for Mcomp_DLM-style data; skipping it for diffuse-only callers
        # avoids spurious failures on data shapes it wasn't designed for.
        if diffuse_only:
            self.primary_period = None
            self.m0_lt = None
            self.m0_seas = None
            self.C0_lt = None
            self.C0_seas = None
            self.s2 = None
        else:
            seas_components = [c for c in components if isinstance(c, Fourier)]
            primary_period = (
                max(c.period for c in seas_components) if seas_components else None
            )
            m0_lt, m0_seas, C0_lt, C0_seas, s2 = initial_state(
                init_data, primary_period, bounded=True
            )
            self.primary_period = primary_period
            self.m0_lt = m0_lt
            self.m0_seas = m0_seas
            self.C0_lt = C0_lt
            self.C0_seas = C0_seas
            self.s2 = s2

        # Cache the level mean for multiplicative-component access.
        # In the diffuse case, use _diffuse_mu directly.
        if any(isinstance(c, (LocalLevel, LocalTrend)) for c in components):
            if diffuse_only:
                self._level_mean = self._diffuse_mu
            else:
                self._level_mean = jnp.asarray(self.m0_lt[:, 0])
        else:
            self._level_mean = None

    def _level_mean_or_raise(self) -> jnp.ndarray:
        if self._level_mean is None:
            raise ValueError(
                "Multiplicative components require a LocalLevel or "
                "LocalTrend component to be present."
            )
        return self._level_mean


# ---------------------------------------------------------------------------
# Component base class
# ---------------------------------------------------------------------------


class Component(ABC):
    """Abstract base class for DLM components.

    Each subclass declares its contribution to the model's structural
    matrices (F, G, discount vectors, mult mask) and how to elicit its
    block of the prior given a :class:`PriorContext`.

    Subclasses must set ``self.name``, ``self.state_dim``, and
    ``self.disc_rate`` in ``__init__`` and implement the abstract
    methods. Optional methods (``disc_damped_block``, ``mult_block``)
    have sensible defaults.

    The ``is_regression`` class-level flag distinguishes "structural"
    components (``LocalLevel``, ``LocalTrend``, ``Fourier``) from
    "regression" components (``Regressors``). Structural components
    contribute to the fixed ``F`` and ``G`` of the model; regression
    components contribute only to the state-vector tail, with their
    ``F``-entries supplied at filter/forecast time as regressor values
    via the ``regressors`` argument of ``fwd_filter`` / ``forecast``.
    """

    name: str
    state_dim: int
    disc_rate: float
    is_regression: bool = False
    # Marks a component whose discount block is a SEASONAL one, so the grid can
    # anchor it to ``seasonal_prior`` rather than the level's discount prior
    # (see ``discount_grid._grid_model``). Set by type, not by name: the names
    # ``build_grid`` happens to use ("seasonal", "seasonal_weekly") cannot cover
    # a hand-built cell with three Fourier blocks, and component names must be
    # unique within a DLM.
    is_seasonal: bool = False
    # Distinguishes an exogenous regression tail (caller-supplied design matrix,
    # e.g. :class:`Regressors` driven by SNAP/promotion indicators) from an
    # autoregressive one (:class:`AR`, whose tail is the series' own lags).
    # The streaming driver reads this to choose the filter/forecast path:
    # exogenous -> ``format_exog_yts`` / ``forecast_exog`` (supplied X);
    # autoregressive -> ``format_lag_yts`` / ``forecast_iterated`` (lags of y).
    is_autoregressive: bool = False

    @abstractmethod
    def F_block(self) -> jnp.ndarray:
        """Observation-vector contribution. Shape ``(state_dim,)``."""

    @abstractmethod
    def G_block(self) -> jnp.ndarray:
        """State-transition block. Shape ``(state_dim, state_dim)``."""

    @abstractmethod
    def disc_norm_block(self) -> jnp.ndarray:
        """Per-state-slot 'normal' discount rate. Shape ``(state_dim,)``."""

    def disc_damped_block(self) -> jnp.ndarray:
        """Per-state-slot damped discount rate. Shape ``(state_dim,)``.

        Default: all ones (no damping). Override in components with
        damping (e.g. :class:`LocalTrend`).
        """
        return jnp.ones(self.state_dim)

    def monitor_inject_block(self) -> jnp.ndarray:
        """Per-state-slot monitor *injection* discount. Shape ``(state_dim,)``.

        When the error-monitor triggers (sustained signed forecast bias on an
        otherwise-confident state), the filter revises this component's grid
        discount DOWN to this value for that single step — a one-off variance
        injection that lets a lagging model catch up. The revised rate is fed
        as the grid INTO ``adapt_discount`` (the guard), which still has the
        last word: an already-uncertain state has the injection suppressed.

        Default: all ones (no injection — ``min(grid, 1) == grid``). Override
        in structural components: level/trend -> ``MONITOR_INJECT_LT`` (0.5),
        seasonal -> ``MONITOR_INJECT_SEAS`` (0.8).
        """
        return jnp.ones(self.state_dim)

    def mult_block(self) -> jnp.ndarray:
        """Multiplicative-component mask. Shape ``(state_dim,)``, 0/1.

        Default: all zeros (additive). Override for components with
        multiplicative slots.
        """
        return jnp.zeros(self.state_dim)

    @abstractmethod
    def elicit_prior(self, ctx: PriorContext) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(m0_block, C0_block)`` for this component.

        ``m0_block`` has shape ``(n_series, state_dim)``;
        ``C0_block`` has shape ``(n_series, state_dim, state_dim)``.
        Both are at t=-1 (one step before the first observation).
        Components are responsible for any roll-back of their own
        priors (e.g. Fourier multiplies by ``G_block``).
        """

    def _user_priors_or_none(self) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Helper for subclasses: return user-supplied (m0, C0) if both
        provided, else None to fall back to elicitation."""
        m0 = getattr(self, "_m0_user", None)
        C0 = getattr(self, "_C0_user", None)
        if m0 is not None and C0 is not None:
            return jnp.asarray(m0), jnp.asarray(C0)
        if (m0 is None) != (C0 is None):
            raise ValueError(
                f"Component {self.name!r}: must supply both m0 and C0 "
                "or neither. Partial overrides are not supported."
            )
        return None


# ---------------------------------------------------------------------------
# Concrete components
# ---------------------------------------------------------------------------


class LocalLevel(Component):
    """Local-level (random walk) component, state dim 1.

    Parameters
    ----------
    name : str
    disc_rate : float
    m0, C0 : optional arrays
        User-supplied prior at t=-1. ``m0`` has shape ``(n,)`` or
        ``(n, 1)``. ``C0`` has shape ``(n, 1, 1)``. If given, must be
        provided together.
    """

    _param_names = ("disc_rate",)

    def __init__(
        self,
        name: str,
        disc_rate=0.95,
        m0=None,
        C0=None,
    ):
        self.name = name
        self.state_dim = 1
        self._disc_rate_kind, self._disc_rate_value = _as_list_or_scalar(
            disc_rate, f"LocalLevel({name!r}).disc_rate"
        )
        self.disc_rate = _spec_base(self._disc_rate_kind, self._disc_rate_value)
        self._m0_user = m0
        self._C0_user = C0

    def F_block(self) -> jnp.ndarray:
        return jnp.asarray([1.0])

    def G_block(self) -> jnp.ndarray:
        return jnp.asarray([[1.0]])

    def disc_norm_block(self) -> jnp.ndarray:
        return jnp.asarray([self.disc_rate])

    def monitor_inject_block(self) -> jnp.ndarray:
        return jnp.asarray([MONITOR_INJECT_LT])

    def elicit_prior(self, ctx):
        user = self._user_priors_or_none()
        if user is not None:
            m0, C0 = user
            if m0.ndim == 1:
                m0 = m0[:, None]
            return m0, C0
        m0 = jnp.asarray(ctx.m0_lt[:, 0])[:, None]  # (n, 1)
        C0 = jnp.asarray(ctx.C0_lt[:, 0])[:, None, None]  # (n, 1, 1)
        return m0, C0

    def diffuse_prior(self, ctx):
        user = self._user_priors_or_none()
        if user is not None:
            m0, C0 = user
            if m0.ndim == 1:
                m0 = m0[:, None]
            return m0, C0
        # m0 = mean(y[:warmup]); C0 = 10 * var(y[:warmup])
        n = ctx.n_series
        m0 = ctx._diffuse_mu[:, None]  # (n, 1)
        C0 = (10.0 * ctx._diffuse_sigma2)[:, None, None]  # (n, 1, 1)
        return m0, C0

    def _swept_dims(self):
        """Return list of SweepDim for any list-valued tunables."""
        from DLMAX.ffs.sweep import SweepDim

        out = []
        if self._disc_rate_kind == "list":
            out.append(
                SweepDim(self.name, "disc_rate", self._disc_rate_value, short="D")
            )
        return out

    def _resolve(self, cell_for_component: dict):
        """Return a new fully-scalar LocalLevel from a cell-fragment.

        ``cell_for_component`` is the slice of the sweep cell for this
        component, e.g. ``{"disc_rate": 0.97}``.
        """
        return LocalLevel(
            name=self.name,
            disc_rate=cell_for_component.get("disc_rate", self._disc_rate_value),
            m0=self._m0_user,
            C0=self._C0_user,
        )


class LocalTrend(Component):
    """Local level-and-trend component, state dim 2.

    The state is ``(level, trend)``. The transition matrix is
    ``[[1, δ], [0, δ]]`` where ``δ`` is the damping factor. ``δ=1``
    is the standard linear-growth model; ``δ < 1`` damps the trend.
    ``δ=0`` zeroes the trend slot each step (equivalent to a level-only
    model with a redundant 2D state).

    The damped discount block is ``[1, δ²]`` — the level slot gets no
    damping and the trend slot gets ``δ²``, matching current
    ``Mcomp_DLM`` behaviour.

    Parameters
    ----------
    name : str
    disc_rate : float
    damping : float, default 1.0
        Trend damping factor. Allowed range [0, 1].
    multiplicative : bool or sequence of bool, default False
        If True, the trend slot (only) is flagged multiplicative. A list
        (e.g. ``[False, True]``) sweeps both at ``compile_universe`` time,
        mirroring ``Fourier.multiplicative``; a scalar (the default) leaves
        behaviour unchanged.
        Note: multiplicative trend is not exercised by current
        ``Mcomp_DLM`` and is not covered by equivalence tests against
        the existing baseline. h-step forecasts for multiplicative
        trend likely require multi-horizon model allocation to behave
        well — flagged for future work.
    m0, C0 : optional arrays
        User-supplied prior at t=-1. ``m0`` has shape ``(n, 2)``.
        ``C0`` has shape ``(n, 2, 2)``.
    """

    _param_names = ("disc_rate", "damping", "multiplicative")

    def __init__(
        self,
        name: str,
        disc_rate=0.95,
        damping=1.0,
        multiplicative: bool = False,
        m0=None,
        C0=None,
    ):
        self.name = name
        self.state_dim = 2
        self._disc_rate_kind, self._disc_rate_value = _as_list_or_scalar(
            disc_rate, f"LocalTrend({name!r}).disc_rate"
        )
        self._damping_kind, self._damping_value = _as_list_or_scalar(
            damping, f"LocalTrend({name!r}).damping"
        )
        self.disc_rate = _spec_base(self._disc_rate_kind, self._disc_rate_value)
        # Per-state discount: a 2-tuple ``(level, trend)`` is a single model with
        # different discounts on the level and trend states. A plain scalar sets
        # both. (A *list* stays a sweep — pass a list of 2-tuples to sweep
        # per-state settings.)
        if isinstance(self.disc_rate, (tuple, np.ndarray)):
            vals = tuple(float(x) for x in self.disc_rate)
            if len(vals) != self.state_dim:
                raise ValueError(
                    f"LocalTrend({name!r}).disc_rate as a per-state tuple must "
                    f"have {self.state_dim} entries (level, trend); got {len(vals)}."
                )
            for x in vals:
                if not 0.0 < x <= 1.0:
                    raise ValueError(
                        f"LocalTrend({name!r}).disc_rate entries must be in "
                        f"(0, 1]; got {x}."
                    )
            self.disc_rate = vals
        if self._damping_kind == "scalar":
            if not (
                0.0 <= self._damping_value <= 1.0
                or self._damping_value != self._damping_value
            ):  # NaN OK
                raise ValueError(
                    f"LocalTrend({name!r}): damping must be in [0, 1] "
                    f"or NaN, got {self._damping_value}."
                )
            self.damping = float(self._damping_value)
        else:
            self.damping = None
        self._multiplicative_kind, self._multiplicative_value = _as_list_or_scalar(
            multiplicative, f"LocalTrend({name!r}).multiplicative"
        )
        if self._multiplicative_kind == "scalar":
            self.multiplicative = bool(self._multiplicative_value)
        else:
            # Validate list contents — must be bools (not 0/1 ints, not
            # truthy strings; explicit to catch obvious mistakes). Mirrors
            # Fourier.multiplicative.
            for v in self._multiplicative_value:
                if not isinstance(v, bool):
                    raise TypeError(
                        f"LocalTrend({name!r}).multiplicative: list must contain "
                        f"bool values, got {type(v).__name__}."
                    )
            # Decided per-cell at resolve time.
            self.multiplicative = None
        self._m0_user = m0
        self._C0_user = C0

    def F_block(self) -> jnp.ndarray:
        return jnp.asarray([1.0, 0.0])

    def G_block(self) -> jnp.ndarray:
        d = self.damping
        return jnp.asarray([[1.0, d], [0.0, d]])

    def disc_norm_block(self) -> jnp.ndarray:
        d = self.disc_rate
        if isinstance(d, (tuple, np.ndarray)):        # per-state (level, trend)
            return jnp.asarray([float(d[0]), float(d[1])])
        return jnp.asarray([d, d])                     # scalar: level = trend

    def monitor_inject_block(self) -> jnp.ndarray:
        return jnp.asarray([MONITOR_INJECT_LT, MONITOR_INJECT_LT])

    def disc_damped_block(self) -> jnp.ndarray:
        # Mcomp_DLM convention: damped block is [1, damping**2] for δ > 0.
        # For δ = 0 ("no trend") the trend slot is zeroed at the prior, so
        # the damped factor is irrelevant — match Mcomp_DLM and store 1.0
        # rather than 0.0 to preserve structural equivalence.
        d2 = 1.0 if self.damping == 0.0 else self.damping**2
        return jnp.asarray([1.0, d2])

    def mult_block(self) -> jnp.ndarray:
        return jnp.asarray([0.0, 1.0 if self.multiplicative else 0.0])

    def elicit_prior(self, ctx):
        user = self._user_priors_or_none()
        if user is not None:
            return user
        n = ctx.n_series
        if self.damping == 0.0:
            # Mcomp_DLM zeroes the trend slot when trend_damping is NaN.
            m0_level = jnp.asarray(ctx.m0_lt[:, 0])[:, None]  # (n, 1)
            m0_trend = jnp.zeros((n, 1))
            m0 = jnp.concatenate([m0_level, m0_trend], axis=1)  # (n, 2)
            C0 = jnp.zeros((n, 2, 2)).at[:, 0, 0].set(jnp.asarray(ctx.C0_lt[:, 0]))
        else:
            m0 = jnp.asarray(ctx.m0_lt)  # (n, 2)
            C0 = vdiag(jnp.asarray(ctx.C0_lt))  # (n, 2, 2)
        return m0, C0

    def diffuse_prior(self, ctx):
        user = self._user_priors_or_none()
        if user is not None:
            return user
        # level slot: m0=mu, C0=10*sigma2
        # trend slot: m0=0,  C0=10*var(diff(y))
        n = ctx.n_series
        if self.damping == 0.0:
            # Mirror Mcomp_DLM "no trend" convention.
            m0 = jnp.stack(
                [ctx._diffuse_mu, jnp.zeros(n)],
                axis=1,
            )  # (n, 2)
            C0 = jnp.zeros((n, 2, 2)).at[:, 0, 0].set(10.0 * ctx._diffuse_sigma2)
        else:
            m0 = jnp.stack(
                [ctx._diffuse_mu, jnp.zeros(n)],
                axis=1,
            )  # (n, 2)
            diag = jnp.stack(
                [10.0 * ctx._diffuse_sigma2, ctx._diffuse_sigma2_d],
                axis=1,
            )  # (n, 2)
            C0 = vdiag(diag)  # (n, 2, 2)
        return m0, C0

    def _swept_dims(self):
        from DLMAX.ffs.sweep import SweepDim

        out = []
        if self._disc_rate_kind == "list":
            out.append(
                SweepDim(self.name, "disc_rate", self._disc_rate_value, short="D")
            )
        if self._damping_kind == "list":
            out.append(SweepDim(self.name, "damping", self._damping_value, short="T"))
        if self._multiplicative_kind == "list":
            out.append(
                SweepDim(
                    self.name, "multiplicative", self._multiplicative_value, short="MT"
                )
            )
        return out

    def _resolve(self, cell_for_component: dict):
        damping = cell_for_component.get("damping", self._damping_value)
        # NaN -> 0.0 (no-trend sentinel; matches mcomp_dlm convention).
        if isinstance(damping, float) and damping != damping:
            damping = 0.0
        multiplicative = cell_for_component.get(
            "multiplicative", self._multiplicative_value
        )
        return LocalTrend(
            name=self.name,
            disc_rate=cell_for_component.get("disc_rate", self._disc_rate_value),
            damping=damping,
            multiplicative=multiplicative,
            m0=self._m0_user,
            C0=self._C0_user,
        )


class Fourier(Component):
    """Fourier-form seasonal component, state dim ``2 * n_comps`` or
    ``period - 1``.

    Parameters
    ----------
    name : str
    period : int
        Seasonal period.
    disc_rate : float
    n_comps : int, optional
        Number of harmonic pairs. If None, full-rank Fourier
        (state_dim = period - 1, mathematically equivalent to seasonal
        dummies but fixed-F).
    multiplicative : bool, default False
        If True, the entire Fourier block is flagged multiplicative.
        Requires a level component to be present.
    selection : bool, default False
        Marks this component as eligible for online variable selection.
        Acts as metadata only in this commit; the variable-selection
        machinery interprets the flag.
    m0, C0 : optional arrays
        User-supplied prior at t=-1. Note that "at t=-1" means the
        prior should be the seasonal state one step before the first
        observation — internal elicitation rolls the data-derived
        prior back via ``G_block``; user-supplied values are used
        as-is.
    """

    _param_names = ("disc_rate", "multiplicative")
    is_seasonal = True

    def __init__(
        self,
        name: str,
        period: int,
        disc_rate=0.9,
        n_comps=None,
        multiplicative=False,  # was: multiplicative: bool = False
        selection: bool = False,
        inert: bool = False,
        m0=None,
        C0=None,
    ):
        self.name = name
        self.period = int(period)
        if n_comps is None:
            self.state_dim = self.period - 1
        else:
            self.state_dim = 2 * int(n_comps)
        self._disc_rate_kind, self._disc_rate_value = _as_list_or_scalar(
            disc_rate, f"Fourier({name!r}).disc_rate"
        )
        if self._disc_rate_kind == "scalar":
            # None disc_rate at scalar level -> inert sentinel
            if disc_rate is None:
                self.disc_rate = 1.0
                self.inert = True
            else:
                self.disc_rate = float(disc_rate)
                self.inert = bool(inert)
        else:
            self.disc_rate = None
            self.inert = False  # decided per-cell at resolve time
        self.n_comps = n_comps
        self._multiplicative_kind, self._multiplicative_value = _as_list_or_scalar(
            multiplicative, f"Fourier({name!r}).multiplicative"
        )
        if self._multiplicative_kind == "scalar":
            self.multiplicative = bool(self._multiplicative_value)
        else:
            # Validate list contents — must be bools (not 0/1 ints, not
            # truthy strings; explicit to catch obvious mistakes).
            for v in self._multiplicative_value:
                if not isinstance(v, bool):
                    raise TypeError(
                        f"Fourier({name!r}).multiplicative: list must contain "
                        f"bool values, got {type(v).__name__}."
                    )
            # Decided per-cell at resolve time.
            self.multiplicative = None
        self.selection = bool(selection)
        self._m0_user = m0
        self._C0_user = C0
        self._F, self._G = fourier_FG(self.period, self.n_comps, LH=False)
        self._L = None
        self._H = None

    def _ensure_LH(self):
        """Lazily compute L and H. Called by elicit_prior, not by diffuse_prior."""
        if self._H is None:
            _, _, self._L, self._H = fourier_FG(self.period, self.n_comps, LH=True)

    def F_block(self) -> jnp.ndarray:
        if self.inert:
            return jnp.zeros(self.state_dim)
        return self._F

    def G_block(self) -> jnp.ndarray:
        if self.inert:
            # Legacy Mcomp_DLM uses jnp.eye for the inert seasonal slot —
            # state is identically zero so the rotation is irrelevant
            # functionally, but structural equality requires eye.
            return jnp.eye(self.state_dim)
        return self._G

    def disc_norm_block(self) -> jnp.ndarray:
        if self.inert:
            # No discount on inert state — slot doesn't evolve.
            return jnp.ones(self.state_dim)
        return jnp.ones(self.state_dim) * self.disc_rate

    def monitor_inject_block(self) -> jnp.ndarray:
        # Seasonal kick (no injection on an inert slot).
        if self.inert:
            return jnp.ones(self.state_dim)
        return jnp.ones(self.state_dim) * MONITOR_INJECT_SEAS

    def mult_block(self) -> jnp.ndarray:
        if self.multiplicative:
            return jnp.ones(self.state_dim)
        return jnp.zeros(self.state_dim)

    def elicit_prior(self, ctx):

        # ... rest of method unchanged
        user = self._user_priors_or_none()
        if user is not None:
            return user

        if self.inert:
            n = ctx.n_series
            return (
                jnp.zeros((n, self.state_dim)),
                jnp.zeros((n, self.state_dim, self.state_dim)),
            )

        self._ensure_LH()

        if ctx.m0_seas is None or ctx.C0_seas is None:
            raise ValueError(
                f"Fourier({self.name!r}): no seasonal prior available "
                "from PriorContext (initial_state was called with no "
                "period). This is an internal bug — please report."
            )

        m_seas = jnp.asarray(ctx.m0_seas)  # shape (period, n)
        c_seas = jnp.asarray(ctx.C0_seas)  # shape (period, n)

        if self.multiplicative:
            level_mean = ctx._level_mean_or_raise()  # (n,)
            safe = jnp.where(level_mean == 0, 1.0, level_mean)  # (n,)
            m_seas = m_seas / safe[None, :]
            # Re-centre seasonal effects to sum to ``period``.
            m_seas = m_seas + (self.period - m_seas.sum(0)) / self.period
            c_seas = c_seas * (1.0 / safe[None, :] ** 2)

        # Map dummy-basis seasonal prior to Fourier basis via H.
        m0_block = (self._H @ m_seas).T  # (n, k)

        # Prior covariance per series: H @ diag(c_seas[:, i]) @ H.T
        n = ctx.n_series
        C0_block = jnp.stack(
            [self._H @ jnp.diag(c_seas[:, i]) @ self._H.T for i in range(n)],
            axis=0,
        )  # (n, k, k)

        # Roll the prior mean back one step: ``m_{t=-1} = m_{t=0} @ G``.
        # (Mcomp_DLM does this via ``m0 @ block_diag(eye(2), G_seas)``.)
        m0_block = m0_block @ self._G

        return m0_block, C0_block

    def diffuse_prior(self, ctx):
        # m0 = 0 per slot.
        # Additive: C0 = 10 * sigma2 per slot (diagonal).
        # Multiplicative: C0 = 10 * cv2 per slot.

        if self.inert:
            n = ctx.n_series
            return (
                jnp.zeros((n, self.state_dim)),
                jnp.zeros((n, self.state_dim, self.state_dim)),
            )

        # Active seasonal: honour a user-supplied prior (taken as-is, already in
        # the Fourier state-space basis at t=-1).
        user = self._user_priors_or_none()
        if user is not None:
            return user

        n = ctx.n_series
        k = self.state_dim
        p = self.period
        m0 = jnp.zeros((n, k))
        if self.multiplicative:
            scale = 10.0 * ctx._diffuse_cv2  # (n,)
        else:
            scale = 10.0 * ctx._diffuse_sigma2  # (n,)
        # Match legacy's `H @ diag(scale * ones(p)) @ H.T = scale * (H @ H.T)`.
        # For the Fourier basis, `H @ H.T = inv(L.T @ L)` is diagonal:
        # `2/p` for paired-harmonic slots and `1/p` for the Nyquist slot
        # (only present when p is even and n_comps is None).
        if self.n_comps is None and p % 2 == 0:
            hht_diag = jnp.concatenate(
                [(2.0 / p) * jnp.ones(k - 1), jnp.asarray([1.0 / p])]
            )
        else:
            hht_diag = (2.0 / p) * jnp.ones(k)
        C0 = scale[:, None, None] * jnp.diag(hht_diag)[None, :, :]  # (n, k, k)
        return m0, C0

    def _swept_dims(self):
        from DLMAX.ffs.sweep import SweepDim

        out = []
        if self._disc_rate_kind == "list":
            out.append(
                SweepDim(self.name, "disc_rate", self._disc_rate_value, short="D")
            )
        if self._multiplicative_kind == "list":
            out.append(
                SweepDim(
                    self.name, "multiplicative", self._multiplicative_value, short="M"
                )
            )
        return out

    def _resolve(self, cell_for_component: dict):
        disc_rate = cell_for_component.get("disc_rate", self._disc_rate_value)
        multiplicative = cell_for_component.get(
            "multiplicative", self._multiplicative_value
        )
        return Fourier(
            name=self.name,
            period=self.period,
            disc_rate=disc_rate,
            n_comps=self.n_comps,
            multiplicative=multiplicative,
            selection=self.selection,
            inert=False,
            m0=self._m0_user,
            C0=self._C0_user,
        )
# ---------------------------------------------------------------------------
# Regression component
# ---------------------------------------------------------------------------


class Regressors(Component):
    """Dynamic regression component.

    Adds ``n_regs`` regression coefficients to the state. Each
    coefficient evolves as a damped random walk:

    .. math:: \\beta_t = \\delta \\, \\beta_{t-1} + w_t

    where ``δ = damping``. ``δ = 1`` gives a pure random walk
    (standard dynamic regression); ``δ < 1`` damps coefficients
    toward zero, matching the convention used for the trend slot in
    :class:`LocalTrend`. The discount-factor handling mirrors
    ``LocalTrend`` exactly: per-slot ``disc_rate`` with a damped
    multiplier of ``δ²``, so the effective discount entering
    :func:`~DLMAX.dlm_core.adapt_discount` is ``disc_rate * δ²``.

    Unlike the structural components, this component's ``F_block`` is
    a placeholder of zeros. The actual regressor values are supplied
    at filter/forecast time via the ``regressors`` argument of
    :meth:`~DLMAX.dlm_core.uv_dlm.fwd_filter` /
    :meth:`~DLMAX.dlm_core.uv_dlm.forecast`. ``uv_dlm`` substitutes
    these values into the regression slots and uses
    ``self.regression_G`` (set by the builder to ``δ · I``) for the
    coefficient transition. Regression slots are placed at the **end**
    of the state vector regardless of the order in which the user
    calls ``add_component``, matching ``uv_dlm``'s layout assumption.

    Parameters
    ----------
    name : str
    n_regs : int
        Number of regression coefficients (regressor slots in the
        state).
    disc_rate : float
        Per-coefficient discount rate (shared across all ``n_regs``
        slots). Sweepable.
    damping : float, default 1.0
        Coefficient damping factor in ``[0, 1]``. Sweepable.
    m0, C0 : optional arrays
        User-supplied prior at ``t = -1``. ``m0`` has shape
        ``(n, n_regs)``. ``C0`` has shape ``(n, n_regs, n_regs)``.
        If omitted, a diffuse default is used: ``m0 = 0``,
        ``C0 = 10 · sigma_y^2 · I`` per series (matching the
        ``diffuse_prior`` convention used elsewhere in the builder).
    x_scale : optional float or array, shape ``(n_regs,)``
        Per-column scale (standard deviation) of the regressors over the
        data. When supplied, the diffuse default becomes *scale-aware*
        (a Zellner g-prior-style convention): ``C0_jj = g · sigma_y^2 /
        x_scale_j^2``, so the implied prior variance on each coefficient's
        *contribution* ``beta_j · X_j`` is ``g · sigma_y^2`` regardless of
        the regressor's units. Off-scale regressors then shrink sensibly
        instead of being handed a units-dependent prior. Leaving this
        ``None`` keeps the legacy flat ``10 · sigma_y^2 · I`` default
        (byte-for-byte unchanged). Ignored when ``C0`` is supplied
        explicitly.
    g : float, default 10.0
        Prior weight for the scale-aware convention (the a-priori number of
        units of ``sigma_y^2`` a regressor may explain). Only used when
        ``x_scale`` is supplied.

    Notes
    -----
    Multiple ``Regressors`` components can be added to a single DLM
    (e.g. to apply different discount rates to different blocks of
    regressors). They are concatenated in the state-vector tail in
    the order they were added.
    """

    _param_names = ("disc_rate", "damping")
    is_regression = True

    def __init__(
        self,
        name: str,
        n_regs: int,
        disc_rate=0.99,
        damping=1.0,
        m0=None,
        C0=None,
        x_scale=None,
        g=10.0,
    ):
        if int(n_regs) <= 0:
            raise ValueError(
                f"Regressors({name!r}): n_regs must be > 0, got {n_regs}."
            )
        self.name = name
        self.n_regs = int(n_regs)
        self.state_dim = self.n_regs

        self._disc_rate_kind, self._disc_rate_value = _as_list_or_scalar(
            disc_rate, f"Regressors({name!r}).disc_rate"
        )
        self._damping_kind, self._damping_value = _as_list_or_scalar(
            damping, f"Regressors({name!r}).damping"
        )

        self.disc_rate = (
            float(self._disc_rate_value)
            if self._disc_rate_kind == "scalar"
            else None
        )
        if self._damping_kind == "scalar":
            if not (0.0 <= self._damping_value <= 1.0):
                raise ValueError(
                    f"Regressors({name!r}): damping must be in [0, 1], "
                    f"got {self._damping_value}."
                )
            self.damping = float(self._damping_value)
        else:
            self.damping = None

        self._m0_user = m0
        self._C0_user = C0
        self._x_scale = None if x_scale is None else jnp.asarray(x_scale, float)
        self._g = float(g)

    def F_block(self) -> jnp.ndarray:
        # Placeholder zeros: regression slots' F-entries are supplied
        # at filter/forecast time via the ``regressors`` argument and
        # do not contribute to the model's static F. The builder
        # discards this block during compile (regression slots live in
        # the state-vector tail, not in self.F); the method is here
        # to satisfy the Component contract.
        return jnp.zeros(self.state_dim)

    def G_block(self) -> jnp.ndarray:
        # Damped identity. delta = 1 -> pure random walk; delta < 1 ->
        # coefficients shrink toward zero each step. The builder
        # passes this block as ``regression_G`` to ``uv_dlm``, which
        # block-diags it onto the structural G at filter time.
        return self.damping * jnp.eye(self.state_dim)

    def disc_norm_block(self) -> jnp.ndarray:
        return jnp.ones(self.state_dim) * self.disc_rate

    def disc_damped_block(self) -> jnp.ndarray:
        # Matches LocalTrend convention: damped multiplier is delta^2.
        # delta = 0 sentinel: store 1.0 (state would be zeroed by G anyway).
        d2 = 1.0 if self.damping == 0.0 else self.damping**2
        return jnp.ones(self.state_dim) * d2

    def mult_block(self) -> jnp.ndarray:
        # Regression coefficients are always additive in this API.
        return jnp.zeros(self.state_dim)

    def elicit_prior(self, ctx):
        # Diffuse default for regression coefficients: m0 = 0,
        # C0 = 10 * sigma_y^2 * I per series. Regressors have no
        # "regression-on-the-series" prior analogue, so we use the
        # data-scaled diffuse default in both the legacy and diffuse
        # paths. Users can override via m0/C0 kwargs.
        user = self._user_priors_or_none()
        if user is not None:
            return user
        n = ctx.n_series
        k = self.n_regs
        m0 = jnp.zeros((n, k))
        if self._x_scale is None:
            # Legacy flat diffuse default (unchanged): C0 = 10 * sigma_y^2 * I.
            C0 = (10.0 * ctx._diffuse_sigma2)[:, None, None] * jnp.eye(k)[None, :, :]
        else:
            # Scale-aware (g-prior style): C0_jj = g * sigma_y^2 / x_scale_j^2,
            # so the prior variance on each contribution beta_j * X_j is
            # g * sigma_y^2 irrespective of the regressor's units.
            xs2 = jnp.broadcast_to(self._x_scale, (k,)) ** 2          # (k,)
            diag = (self._g * ctx._diffuse_sigma2)[:, None] / xs2[None, :]  # (n, k)
            C0 = diag[:, :, None] * jnp.eye(k)[None, :, :]            # (n, k, k)
        return m0, C0

    def diffuse_prior(self, ctx):
        # Identical to elicit_prior: the data-scaled diffuse default
        # is appropriate for both the legacy (initial_state-based) and
        # warmup paths, since regression coefficients have no
        # observation-driven prior analogue.
        return self.elicit_prior(ctx)

    def _swept_dims(self):
        from DLMAX.ffs.sweep import SweepDim

        out = []
        if self._disc_rate_kind == "list":
            out.append(
                SweepDim(self.name, "disc_rate", self._disc_rate_value, short="D")
            )
        if self._damping_kind == "list":
            # "R" for regression-damping, to distinguish from
            # LocalTrend's "T" (trend-damping) when both appear in the
            # same descriptor.
            out.append(
                SweepDim(self.name, "damping", self._damping_value, short="R")
            )
        return out

    def _resolve(self, cell_for_component: dict):
        return Regressors(
            name=self.name,
            n_regs=self.n_regs,
            disc_rate=cell_for_component.get("disc_rate", self._disc_rate_value),
            damping=cell_for_component.get("damping", self._damping_value),
            m0=self._m0_user,
            C0=self._C0_user,
            x_scale=self._x_scale,
            g=self._g,
        )


class AR(Regressors):
    r"""Autoregressive component: a :class:`Regressors` block whose regressors
    are the series' **own lags** :math:`y_{t-1}, \ldots, y_{t-p}`, with a
    Minnesota shrink-to-random-walk prior.

    Mechanically identical to :class:`Regressors`: the ``p`` coefficients evolve
    as a discounted random walk (``disc_rate = 1`` -> ~fixed-coefficient AR;
    ``disc_rate < 1`` -> time-varying AR / TVAR), and their F-entries are
    supplied at filter/forecast time. Two differences:

    * **Minnesota prior** (Litterman): prior mean ``m0 = (1, 0, ..., 0)`` (a
      random walk) and prior covariance
      ``C0 = diag((\lambda/1)^2, (\lambda/2)^2, ..., (\lambda/p)^2)`` with
      ``\lambda`` fixed (default 0.2). The own-lag-1 coefficient starts at 1 and
      higher lags are shrunk toward 0 with lag-decaying tightness. Scale-free in
      the coefficients, so it needs neither the :class:`PriorContext` variance
      nor any change to the diffuse priors of the structural blocks (priors
      compose block-diagonally). Damping is fixed at 1 (pure RW coefficients);
      a ``damping < 1`` would fight the RW anchor, so shrinkage is left to the
      prior and the discount, not the transition.

    * ``is_autoregressive = True`` -- marks the block so the filter/forecast
      driver supplies the series' own lags as the regressors (not an exogenous
      design) and uses the iterated-expectations multi-step forecast (the future
      lags are themselves forecasts). Those two pieces live in the
      filter/forecast layer, not here.

    Parameters
    ----------
    name : str
    order : int
        AR order ``p`` (number of lags = number of coefficient slots = state
        dim). With the "TVAR(x), x = structural state size" convention this
        equals the state size of the structural model it mirrors.
    disc_rate : float
        Coefficient discount (time-variation rate). Sweepable; reuses the
        existing component discount grid.
    minnesota_lambda : float, default 0.2
        Minnesota overall-tightness; prior sd on the lag-``k`` coefficient is
        ``minnesota_lambda / k`` (so lag-1 sd = ``minnesota_lambda``).
    m0, C0 : array, optional
        Supply the coefficient prior outright, bypassing the Minnesota
        elicitation above — ``m0`` ``(n_series, order)`` and ``C0``
        ``(n_series, order, order)``. Use for a bespoke shrinkage pattern (e.g.
        geometric rather than harmonic lag decay, or prior mass on a seasonal
        lag). Both must be given together.
    """

    is_autoregressive = True

    def __init__(self, name, order, disc_rate=0.99, minnesota_lambda=0.2,
                 period=None, seasonal_weight=0.5, anchor=1.0, m0=None, C0=None):
        super().__init__(name=name, n_regs=order, disc_rate=disc_rate, damping=1.0,
                         m0=m0, C0=C0)
        if not minnesota_lambda > 0:
            raise ValueError(
                f"AR({name!r}): minnesota_lambda must be > 0, got {minnesota_lambda}."
            )
        if not 0.0 <= anchor <= 1.0:
            raise ValueError(f"AR({name!r}): anchor must be in [0, 1], got {anchor}.")
        self.order = int(order)
        self.minnesota_lambda = float(minnesota_lambda)
        # Prior mean of the (summed) unit-root persistence: 1.0 = pure RW; <1
        # (e.g. 0.95) hints at sub-integrated behaviour -> stationary prior with
        # bounded long-horizon forecast variance.
        self.anchor = float(anchor)
        # Seasonal anchor: the seasonal lag is the *period* (not the AR order).
        # None / period<2 / period>order -> non-seasonal (plain RW prior).
        self.period = int(period) if period and 1 < int(period) <= int(order) else None
        self.seasonal_weight = float(seasonal_weight)

    def elicit_prior(self, ctx):
        user = self._user_priors_or_none()
        if user is not None:
            return user
        n, p, lam = ctx.n_series, self.n_regs, self.minnesota_lambda
        lag_var = (lam / jnp.arange(1, p + 1)) ** 2            # Minnesota (lambda/k)^2
        m0 = jnp.zeros((n, p))
        if self.period is None:
            # Shrink to (sub-)RW: lag-1 mean = anchor (1.0 = pure RW).
            m0 = m0.at[:, 0].set(self.anchor)
        else:
            # Seasonal: split the (anchor) persistence between lag-1 and
            # lag-`period` (means sum to anchor -> non-explosive), and echo
            # lag-1's prior variance (lambda^2) onto the seasonal lag so it is
            # not shrunk hard. Index period-1 is the period-th lag (0-indexed).
            sw = self.seasonal_weight
            m0 = (m0.at[:, 0].set(self.anchor * (1.0 - sw))
                  .at[:, self.period - 1].set(self.anchor * sw))
            lag_var = lag_var.at[self.period - 1].set(lag_var[0])
        C0 = jnp.broadcast_to(jnp.diag(lag_var), (n, p, p))   # (n, p, p)
        return m0, C0

    def diffuse_prior(self, ctx):
        return self.elicit_prior(ctx)

    def _resolve(self, cell_for_component: dict):
        return AR(
            name=self.name,
            order=self.n_regs,
            disc_rate=cell_for_component.get("disc_rate", self._disc_rate_value),
            minnesota_lambda=self.minnesota_lambda,
            period=self.period,
            seasonal_weight=self.seasonal_weight,
            anchor=self.anchor,
        )


# ---------------------------------------------------------------------------
# Error specification
# ---------------------------------------------------------------------------


class ErrorSpec:
    def __init__(self, disc_rate=1.0, power = 1, nu0: float = 1.0):
        self._disc_rate_kind, self._disc_rate_value = _as_list_or_scalar(
            disc_rate, "ErrorSpec.disc_rate"
        )
        self._power_kind, self._power_value = _as_list_or_scalar(
            power, "ErrorSpec.power"
        )
        if self._disc_rate_kind == "scalar":
            if not 0.0 < self._disc_rate_value <= 1.0:
                raise ValueError(
                    f"ErrorSpec: disc_rate must be in (0, 1], got "
                    f"{self._disc_rate_value}."
                )
            self.disc_rate = float(self._disc_rate_value)
        else:
            self.disc_rate = None
        self.power = float(self._power_value) if self._power_kind == "scalar" else None
        self.nu0 = float(nu0)

    def _swept_dims(self):
        from DLMAX.ffs.sweep import SweepDim

        out = []
        if self._disc_rate_kind == "list":
            out.append(SweepDim("error", "disc_rate", self._disc_rate_value, short="D"))
        if self._power_kind == "list":
            out.append(SweepDim("error", "power", self._power_value, short="P"))
        return out

    def _resolve(self, cell_for_error: dict):
        return ErrorSpec(
            disc_rate=cell_for_error.get("disc_rate", self._disc_rate_value),
            power=cell_for_error.get("power", self._power_value),
            nu0=self.nu0,
        )


# ---------------------------------------------------------------------------
# DLM class
# ---------------------------------------------------------------------------


class DLM:
    """Configurable DLM specification.

    A DLM is built by adding components and (for Gaussian models)
    setting the error spec. ``compile(init_data)`` runs prior
    elicitation and returns a ``uv_dlm`` ready for filtering.

    Parameters
    ----------
    family : {'Gaussian'}
        Observation family. Only Gaussian is supported in this commit;
        exponential-family DLMs are forward-looking.
    n_series : int
        Number of series the model will be fit to. ``compile`` validates
        this against the shape of ``init_data``.

    Examples
    --------
    >>> dlm = DLM(family='Gaussian', n_series=10)
    >>> dlm.add_component(LocalTrend(name='trend', disc_rate=0.99,
    ...                              damping=0.99))
    >>> dlm.add_component(Fourier(name='annual', period=12, n_comps=2,
    ...                           disc_rate=0.95))
    >>> dlm.set_error(disc_rate=1.0, power=1, nu0=1)
    >>> uvdlm = dlm.compile(my_init_data)
    """

    SUPPORTED_FAMILIES = frozenset({"Gaussian"})

    def __init__(self, family: str = "Gaussian", n_series: Optional[int] = None):
        if family not in self.SUPPORTED_FAMILIES:
            raise NotImplementedError(
                f"family={family!r} not yet supported. "
                f"Supported: {sorted(self.SUPPORTED_FAMILIES)}."
            )
        self.family = family
        self.n_series = n_series
        self.components: List[Component] = []
        self.error_spec: Optional[ErrorSpec] = None
        # Error-monitoring sensitivity tau (the reused `monitor` slot). None
        # leaves every compiled model with monitor=False (legacy static path,
        # byte-identical). A scalar applies one tau to all models; a list makes
        # tau a sweep axis (the adaptive universe's fast/slow models).
        self._monitor_kind, self._monitor_value = "scalar", False

    def __repr__(self):
        comp_names = [c.name for c in self.components]
        err = "set" if self.error_spec is not None else "unset"
        return (
            f"DLM(family={self.family!r}, n_series={self.n_series}, "
            f"components={comp_names}, error={err})"
        )

    def add_component(self, component: Component) -> "DLM":
        """Append a component. Returns self for chaining."""
        if not isinstance(component, Component):
            raise TypeError(
                f"add_component: expected Component, got {type(component).__name__}."
            )
        if any(c.name == component.name for c in self.components):
            raise ValueError(f"Duplicate component name: {component.name!r}.")
        self.components.append(component)
        return self

    def set_error(self, disc_rate: float = 1.0, power: float = 1, nu0: float = 1.0) -> "DLM":
        """Set the observation-error spec. Required for Gaussian models;
        not allowed for exponential-family models (when those land)."""
        if self.family != "Gaussian":
            raise ValueError(
                f"set_error is only valid for family='Gaussian'; "
                f"this DLM has family={self.family!r}."
            )
        self.error_spec = ErrorSpec(disc_rate=disc_rate, power=power, nu0=nu0)
        return self

    def set_monitor(self, tau) -> "DLM":
        """Set the error-monitoring sensitivity tau (the `monitor` slot).

        ``tau`` is the threshold on the EWMA of |standardised one-step error|
        below which a model keeps its calm-regime discount (d=1*damped) and
        above which it forgets towards its per-component d_min. Pass:

        - ``False`` / ``None`` (default): no monitoring — every compiled model
          gets ``monitor=False`` and the legacy static-discount path runs
          (byte-identical to before this feature existed).
        - a scalar float: apply one tau to all models.
        - a list of floats: tau becomes a sweep axis (fast/slow models that
          are identical in calm and diverge only on a shock). This is how the
          adaptive universe replaces the legacy LT-discount grid.

        Returns self for chaining.
        """
        if tau is None:
            tau = False
        self._monitor_kind, self._monitor_value = _as_list_or_scalar(
            tau, "DLM.set_monitor.tau"
        )
        return self

    def _validate(self, init_data: pd.DataFrame) -> None:
        if self.n_series is None:
            raise ValueError(
                "n_series must be set before compile() (either as a "
                "constructor argument or via direct attribute assignment)."
            )
        if init_data.shape[1] != self.n_series:
            raise ValueError(
                f"init_data has {init_data.shape[1]} series, but "
                f"DLM.n_series={self.n_series}."
            )

        if not self.components:
            raise ValueError("DLM has no components.")
        structural_components = [
            c for c in self.components if not c.is_regression
        ]
        if not structural_components:
            raise ValueError(
                "DLM must have at least one structural component "
                "(LocalLevel, LocalTrend, or Fourier). Regression-only "
                "models are under-identified (V0 = 0)."
            )
        level_components = [
            c for c in self.components if isinstance(c, (LocalLevel, LocalTrend))
        ]
        if len(level_components) > 1:
            raise ValueError(
                "At most one of LocalLevel or LocalTrend is permitted "
                f"(got {[c.name for c in level_components]})."
            )


        # Multiplicative components require a level component.
        for c in self.components:
            if isinstance(c, Fourier) and c.multiplicative and not level_components:
                raise ValueError(
                    f"Fourier({c.name!r}, multiplicative=True) requires a "
                    "LocalLevel or LocalTrend component to be present."
                )
        if self.family == "Gaussian" and self.error_spec is None:
            raise ValueError("Gaussian DLM requires set_error(...) before compile().")

    def compile(
        self,
        init_data: pd.DataFrame,
        device=None,
        warmup_steps: Optional[int] = None,
        h: Optional[int] = None,
        monitor=None,
    ) -> uv_dlm:
        """Run prior elicitation and return a ``uv_dlm`` instance.

        Parameters
        ----------
        init_data : pd.DataFrame
            Time series used for prior elicitation. Shape (T, n_series),
            with each column a series. The mean feeds the initial location,
            the state variance is used to ensure diffuse initial state variance.
        device : NamedSharding, optional
            Defaults to ``DLMAX.ffs.devices.host_device``.
        warmup_steps : int, optional
            number of steps to keep discount rate = 1 for mximum learning
            of initial state
        h : int, optional
            Forecast horizon. If set, ``GH`` is precomputed for the
            returned ``uv_dlm``, enabling efficient ``forecast(h)``.

        Returns
        -------
        uv_dlm
            Ready for filtering. Equivalent to the output of
            :func:`Mcomp_DLM` for matching parameter combinations.
        """
        self._validate(init_data)

        # Reject list-valued tunables on the scalar-compile path.
        sweep_dims = []
        for c in self.components:
            sweep_dims.extend(c._swept_dims())
        if self.error_spec is not None:
            sweep_dims.extend(self.error_spec._swept_dims())
        if sweep_dims:
            names = [f"{d.component}.{d.param}" for d in sweep_dims]
            raise ValueError(
                f"DLM.compile() requires scalar parameters; found "
                f"list-valued: {names}. Use DLM.compile_universe() "
                "instead."
            )

        diffuse_only = warmup_steps is not None and warmup_steps > 0
        ctx = PriorContext(init_data, self.components, diffuse_only=diffuse_only)
        n = self.n_series

        # Partition components: structural (LocalLevel/LocalTrend/Fourier)
        # contribute to F and G; regression (Regressors) contribute only
        # to the state-vector tail, with F-entries supplied at filter
        # time via the ``regressors`` argument.
        structural = [c for c in self.components if not c.is_regression]
        regression = [c for c in self.components if c.is_regression]
        ordered = structural + regression
        n_regs_total = sum(c.n_regs for c in regression)

        # Concatenate / block-diag matrices. F and G are structural-only
        # (passed to uv_dlm as self.F and self.G); regression_G is
        # separate (passed as regression_G kwarg, used by _broadcast_G
        # at filter time).
        F = jnp.concatenate([c.F_block() for c in structural])
        G = block_diag(*[c.G_block() for c in structural])
        regression_G = (
            block_diag(*[c.G_block() for c in regression])
            if regression
            else None
        )
        # disc_norm / disc_damped / mult_comps / m0 / C0 cover the full
        # state (structural + regression, in that order).
        disc_norm = jnp.concatenate([c.disc_norm_block() for c in ordered])
        disc_damped = jnp.concatenate([c.disc_damped_block() for c in ordered])
        monitor_inject = jnp.concatenate([c.monitor_inject_block() for c in ordered])
        mult_comps = jnp.concatenate([c.mult_block() for c in ordered])

        # Per-component priors. Choose elicit_prior (legacy regression-based)
        # or diffuse_prior (warmup-friendly diffuse) based on warmup_steps.
        if warmup_steps is None or warmup_steps == 0:
            prior_method = "elicit_prior"
        else:
            prior_method = "diffuse_prior"
        m0_blocks, C0_blocks = zip(
            *(getattr(c, prior_method)(ctx) for c in ordered)
        )
        m0 = jnp.concatenate(m0_blocks, axis=1)  # (n, p_struct + n_regs_total)
        C0 = jnp.stack(
            [
                block_diag(*[C0_blocks[k][i] for k in range(len(ordered))])
                for i in range(n)
            ],
            axis=0,
        )  # (n, p_total, p_total)

        # Observation-variance prior derived from initial residuals.
        if prior_method == "elicit_prior":
            # Previous behaviour:
            # Observation-variance prior derived from initial residuals.
            # V0 = jnp.asarray(ctx.s2)
            # now set V0 = F_full C0 F_full.T * 10, same as system variance.
            # F_full is F zero-padded to the full state width so the
            # matmul aligns with C0 (which spans structural + regression).
            # Regression coefficients contribute zero to V0 at t=0,
            # consistent with regressors_0 = 0.
            if n_regs_total > 0:
                F_full = jnp.concatenate([F, jnp.zeros(n_regs_total)])
            else:
                F_full = F
            V0 = F_full @ C0 @ F_full.T * 10
        else:
            V0 = jnp.asarray(ctx._diffuse_sigma2)

        nu0 = jnp.ones(n) * self.error_spec.nu0

        if device is None:
            device = devices.host_device

        # Error-monitoring sensitivity tau (reused `monitor` slot). Default to
        # the DLM-level value (False unless set_monitor was called); a per-cell
        # value passed by compile_universe overrides it. False -> legacy static.
        if monitor is None:
            monitor = self._monitor_value

        result = uv_dlm(
            series_ids=jnp.arange(n),
            F=F,
            G=G,
            n_regressors=n_regs_total,
            regression_G=regression_G,
            m0=m0,
            C0=C0,
            V0=V0,
            nu0=nu0,
            disc_rates_norm=disc_norm,
            disc_rates_damped=disc_damped,
            monitor_inject=monitor_inject,
            variance_disc=self.error_spec.disc_rate,
            variance_power=self.error_spec.power,
            mult_comps=mult_comps,
            regressors_0=jnp.zeros(n_regs_total),
            device=device,
            h=h,
            monitor=monitor,
        )

        result.warmup_steps = warmup_steps if warmup_steps is not None else 0
        # Mark the regression tail's kind so the streaming driver picks the right
        # filter/forecast path. A model is autoregressive iff any of its
        # regression components is an AR block (its own lags); a plain exogenous
        # Regressors tail (supplied X) is not. Structural-only models (no
        # regression tail) leave this False — inert.
        result.is_autoregressive = any(
            getattr(c, "is_autoregressive", False) for c in regression
        )

        # Adapt/Wing: online RTRL discount learning happens inside fwd_filter, so
        # a discount-learning model looks like any other to the caller.
        kinds = [getattr(c, "_disc_rate_kind", "scalar") for c in self.components]
        _wu = warmup_steps if warmup_steps is not None else 6
        if "wing" in kinds:
            import numpy as _np
            from .discount_grid import _grid_model
            gm = _grid_model(self.components, float(self.error_spec.power))
            spec = next(c._disc_rate_value for c, k in zip(self.components, kinds)
                        if k == "wing")
            init = spec.init
            centre = (float(_np.mean(_np.asarray(init)))
                      if isinstance(init, (tuple, list, _np.ndarray)) else float(init))
            # β is a learned wingman param; start it at the grid convention 0.99
            # (the fixed default error disc_rate=1 would mean no β forgetting).
            _beta0 = float(self.error_spec.disc_rate)
            _beta0 = 0.99 if _beta0 >= 1.0 else _beta0
            result.enable_wing(
                gm, centre, offset=float(spec.offset), wings=int(spec.wings),
                warmup=_wu, beta_init=_beta0)
            # keep the components so AdaptiveBlock.from_cells can rebuild the grid
            # (grid_init needs them for the diffuse prior).
            result._wing["comps"] = tuple(self.components)
        elif "adapt" in kinds:
            gm, theta0, learn = _adapt_geometry(
                self.components, float(self.error_spec.power),
                float(self.error_spec.disc_rate))
            result.enable_adapt(gm, learn, theta0, warmup=_wu)
        return result

    def compile_universe(
        self,
        init_data: pd.DataFrame,
        h: Optional[int] = None,
        *,
        constraint=None,
        warmup_steps: Optional[int] = None,
        device=None,
        keyfn=None,
    ):
        """Build a dict of ``uv_dlm`` instances by sweeping list-valued params.

        Parameters
        ----------
        init_data : pd.DataFrame
        h : int, optional
            Forecast horizon, threaded through to each compiled model.
        constraint : callable, optional
            ``cell -> bool``. Cells where this returns False are skipped.
            ``cell`` is ``{component_name: {param: value}}`` plus an
            ``"error"`` group for the error spec.
        warmup_steps : int, optional
            Threaded through to ``.compile()``.
        device : NamedSharding, optional
        keyfn : callable, optional
            Custom cell-key formatter. Defaults to
            ``DLMAX.ffs.sweep.default_keyfn``. The ``make_ffs_universe``
            factory passes its own to produce the ``"AL0T0S0E0"`` key shape.

        Returns
        -------
        models : dict[str, uv_dlm]
            Keyed by cell-key (see ``keyfn``).
        descriptor : pd.DataFrame
            One row per cell; columns are the param paths
            (``"trend.disc_rate"``, ``"error.power"``, etc.) plus
            ``"key"``.
        """
        from DLMAX.ffs.sweep import sweep, default_keyfn, SweepDim

        self._validate(init_data)

        # Collect dims from components + error spec.
        dims = []
        for c in self.components:
            dims.extend(c._swept_dims())
        if self.error_spec is not None:
            dims.extend(self.error_spec._swept_dims())
        # Error-monitoring tau as a model-level sweep axis (adaptive universe).
        if self._monitor_kind == "list":
            dims.append(SweepDim("monitor", "tau", self._monitor_value, short="K"))

        if not dims:
            raise ValueError(
                "compile_universe: no list-valued tunables found. Use "
                ".compile() for scalar models, or pass at least one "
                "list-valued parameter to a component or set_error()."
            )

        keyfn = keyfn if keyfn is not None else default_keyfn
        models = {}
        rows = []
        for key, cell, _idx in sweep(dims, constraint=constraint, keyfn=keyfn):
            # Resolve each component to a fully-scalar instance.
            resolved = DLM(family=self.family, n_series=self.n_series)
            for c in self.components:
                resolved.add_component(c._resolve(cell.get(c.name, {})))
            if self.error_spec is not None:
                resolved.error_spec = self.error_spec._resolve(cell.get("error", {}))
            mon = cell.get("monitor", {}).get("tau", self._monitor_value)
            models[key] = resolved.compile(
                init_data,
                device=device,
                warmup_steps=warmup_steps,
                h=h,
                monitor=mon,
            )

            # Descriptor row: include every tunable, not just the swept
            # ones. Non-swept params get repeated across rows; this makes
            # the descriptor a self-contained record of each cell's full
            # parameterisation.
            row = {"key": key}
            for c in self.components:
                cell_for_c = cell.get(c.name, {})
                for p in _component_param_names(c):
                    if p in cell_for_c:
                        # Swept param: use the cell value (preserves None/NaN).
                        row[f"{c.name}.{p}"] = cell_for_c[p]
                    else:
                        # Non-swept: read scalar from the original component.
                        row[f"{c.name}.{p}"] = getattr(c, p)
            if self.error_spec is not None:
                err_cell = cell.get("error", {})
                for p in ("disc_rate", "power"):
                    if p in err_cell:
                        row[f"error.{p}"] = err_cell[p]
                    else:
                        row[f"error.{p}"] = getattr(self.error_spec, p)
            row["monitor.tau"] = cell.get("monitor", {}).get("tau", self._monitor_value)
            rows.append(row)

        descriptor = pd.DataFrame(rows)
        return models, descriptor

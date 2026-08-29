"""Dynamic-grid adaptive discount — per-step RTRL core.

The optimisation engine that learns the discount factors online, built on the
autodiff-stable QR kernel (:func:`~DLMAX.dlm_core.dlm_uv_fwd_qr_step`).
Per time step and per model it advances the filter at the current discount, carries
the forward sensitivity ``S = ∂state/∂θ`` (RTRL), and reads BOTH per-parameter
gradients off the one carry:

- ``g_sq`` — gradient of the frozen-variance squared error ``½·e²/sg(Q)`` (the
  quasi-likelihood objective for the mean-tracking discounts: the ``stop_gradient``
  on ``Q`` removes the sharpness channel and auto-weights relative-for-multiplicative
  / plain-for-additive, so there is no ``var_power`` branch);
- ``g_ls`` — gradient of the Student-t log score (the correct objective for the
  variance discount β).

This module is only the *engine* (filter step + streaming gradient). Its consumer
— a movable grid that pools ``g_sq[level]`` across a cell's wingmen and steps the
rest per worker, combined by a DMA — sits on top of it. The engine carries no
belief over the discount and does no quadrature: the gradient is read directly
off the filter carry, which is what makes it a per-step cost rather than a
per-step integration.

``grad_run`` runs the carry along a FIXED discount path (used by the sensitivity
gate and finite-difference checks); the online loop, where the discount evolves
step-to-step by a cross-model update, is a separate orchestration piece.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.scipy.special import gammaln
from jax.tree_util import Partial

from ..dlm_core import (
    AllocatorState, ForecastBundle, IdentityAggregator, LogScore,
                        iterated_obs_forecast,
                        PowerLawUpdate, allocator_step, dlm_uv_fcast_H,
                        dlm_uv_fwd_qr_step, finite_inv, init_alloc_state, var_scale_fn)


def neg_log_pred(e: jnp.ndarray, Q: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    """Negative log one-step predictive density, Student-t(nu, loc=0, scale²=Q)."""
    return (-gammaln((nu + 1.0) / 2.0) + gammaln(nu / 2.0)
            + 0.5 * jnp.log(nu * jnp.pi * Q)
            + (nu + 1.0) / 2.0 * jnp.log1p(e**2 / (nu * Q)))


def _logit(p):
    return jnp.log(p) - jnp.log1p(-p)


# Uniform clip box (same for all models, per the design — the DMA is the backstop).
CLIP_LO, CLIP_HI = 0.5, 0.9999
BETA_CLIP = (0.95, 1.0)
# DMA forgetting for DLMAX's PowerLawUpdate: pdm within a family's wingmen (in-class),
# mdm across the 16 families (between-class); both 0.90, matching AutoFFS. c = floor.
DMA_PDR = 0.90
DMA_MDR = 0.90
DMA_C = 1e-3
# Info-envelope covariance guard (adapt_discount port) — ON BY DEFAULT for every
# grid built through the block API. 0.5 = the static path's adapt coefficient.
# Pass ``adapt_guard=None`` to disable (deliberately awkward: the guard is a
# correctness feature, like the damped δ² correction, not an opt-in tweak). Fixes
# the M4H blow-up and the M4M long-horizon over-extrapolation.
ADAPT_GUARD_DEFAULT = 0.5

# ---------------------------------------------------------------------------
# The WING SPEC — the published default.
#
# These are the constants the M4 and M5 exhibits were produced under; they are
# the defaults of ``AutoFFS`` and ``AutoFFSUniverse`` so that
# ``AutoFFS(season_length=P, warmup=W)`` reproduces the published numbers with
# nothing else configured. Verified against the archived M5 universe config
# (grid_disc_prior [3 1 5 1], grid_seasonal_prior [4], grid_learn_dma 1,
# grid_additive_logscore 0, grid_decouple_trend 0, grid_offset 1.0,
# dma_pdr/mdr 0.9) and the hard-coded spec block in m4_wing_run.py.
#
# Anything not listed here already defaults correctly: offset 1.0, pdr/mdr 0.90
# (DMA_PDR/DMA_MDR), adapt_guard 0.5 (ADAPT_GUARD_DEFAULT), dampings DAMPINGS,
# additive_logscore/decouple_trend False, dma_prior None -> DMA_PRIOR (2, 1).
WING_DISC_PRIOR = (3.0, 1.0, 5.0, 1.0)   # N(3,1) on level/trend δ, N(5,1) on var β
WING_SEASONAL_PRIOR = 4.0                # N(4,1) on the seasonal Fourier blocks
WING_LEARN_DMA = True                    # SGDDMA: learn the forgetting rates online


@dataclass(frozen=True)
class GridModel:
    """Single-DLM geometry for the discount grid. ``state_to_block`` maps each
    state to its discount block
    (``0 .. n_blocks-1``); the discount vector is ``θ = [δ_block_0, …, β]``."""

    F: jnp.ndarray
    G: jnp.ndarray
    mc: jnp.ndarray
    var_power: float
    state_to_block: jnp.ndarray
    n_blocks: int
    #: Per-state (K,) bool. States flagged True carry a FROZEN discount δ=1 (never
    #: learned) — the mechanism for padded/inert slots when families of different
    #: state size are packed into one block (mirrors ``multi_model_dlm._init_from_dlms``,
    #: which pads with ``disc=1``). ``None`` -> no frozen states (bit-exact default).
    inert_mask: jnp.ndarray = None
    #: Bool. When True (decoupled trend + coupling), block 1 is a GROWTH block whose
    #: discount prior mean is the LEVEL block's (block 0) learned logit — a soft
    #: version of the fused level/growth coupling. False -> no coupling (bit-exact).
    couple_growth: bool = False
    #: Per-STATE (K,) damped-discount correction (``disc_damped_block``): the
    #: effective discount is ``θ · damped``, matching the SVD path's ``disc_rates *
    #: disc_rates_damped``. LocalTrend's growth slot gets δ²; everything else 1.0.
    #: ``None`` -> all ones (undamped, bit-exact default).
    damped: jnp.ndarray = None
    #: Per-BLOCK (n_blocks,) bool: True for a Fourier (seasonal) discount block.
    #: When a ``seasonal_prior`` mean is supplied, these blocks are anchored to it
    #: (e.g. N(4,1)) instead of the level/trend ``disc_prior`` mean. ``None`` -> no
    #: seasonal block (bit-exact default).
    seasonal_mask: jnp.ndarray = None
    #: Info-envelope covariance guard (``adapt_discount`` port): a scalar ``adapt``
    #: coefficient or ``None``. When set, the effective per-state discount is pulled
    #: toward 1 in proportion to the state's RELATIVE uncertainty
    #: ``info = |m|/√diag(C)`` — ``δ_eff = δ + (1-δ)·exp(-adapt·info)`` — so an
    #: already-uncertain state is not discounted further and its covariance cannot
    #: grow without bound (mirrors ``dlm_core.adapt_discount``). Applied INSIDE the
    #: jvp'd filter step, so it flows through the RTRL derivative. ``None`` ->
    #: off (bit-exact default).
    adapt_guard: float = None
    #: Width of the REGRESSION tail (exogenous or AR), 0 for a structural-only
    #: cell. The tail occupies states ``[reg_offset, reg_offset + n_regs)``; its
    #: ``F`` entries are zero here and are filled per step from the supplied
    #: regressors, exactly as the legacy multi path does. Offset rather than a
    #: negative index because ``_pad_grid_model`` appends inert slots AFTER the
    #: tail. ``0`` -> no tail (bit-exact default).
    n_regs: int = 0
    reg_offset: int = 0
    #: Per-BLOCK (n_blocks,) bool: True for a regression discount block. The tail
    #: gets its own block, so its discount is learned on the same footing as
    #: level / seasonal (``AR(disc_rate=Wing(...))``). Used by ``_clip_box``.
    regression_mask: jnp.ndarray = None

    @property
    def n_params(self) -> int:
        return self.n_blocks + 1


def _F_at(model: GridModel, x):
    """This step's observation vector: the static structural ``F`` with the
    regression tail filled from ``x`` ``(n_regs,)``.

    ``x is None`` or a structural-only cell returns ``model.F`` unchanged, so
    every existing path is bit-exact. The tail is addressed by ``reg_offset``
    rather than a negative index because ``_pad_grid_model`` appends inert slots
    AFTER it. ``x`` is DATA -- an exogenous row, or lagged observations -- so it
    is a constant to the RTRL jvp, exactly as ``y`` is.
    """
    if x is None or model.n_regs == 0:
        return model.F
    o = model.reg_offset
    return model.F.at[o:o + model.n_regs].set(jnp.asarray(x))


def lag_design(ys, order: int, history=None):
    """``(T, q, order)`` AR design over ``ys`` ``(T, q)``: ``out[t, s, j] =
    y_{t-1-j}``, i.e. slot ``j`` holds lag ``j+1``.

    The grid's analogue of :meth:`~DLMAX.dlm_core.multi_model_dlm.format_lags`,
    and deliberately the SAME convention so the two engines can be compared
    directly: ``history`` is the ``(order, q)`` block of observations
    IMMEDIATELY PRECEDING ``ys`` in TIME order (oldest row first), and ``None``
    zero-fills the first ``order`` steps so the tail contributes nothing until it
    has real lags.

    The lags are a deterministic function of the y stream, so they are built here
    rather than threaded through a filter-time buffer — the same reasoning the
    legacy path gives.

    NaN propagates: a missing observation makes the lags that depend on it NaN,
    hence ``f`` NaN, for the ``order`` steps it feeds. That matches the legacy
    path rather than diverging from it, and it is loud rather than silent.
    """
    ys = jnp.asarray(ys)
    T, q = ys.shape
    if order == 0:
        return jnp.zeros((T, q, 0), ys.dtype)
    if history is None:
        ext = jnp.concatenate([jnp.zeros((order, q), ys.dtype), ys], axis=0)
    else:
        ext = jnp.concatenate([jnp.asarray(history, ys.dtype)[-order:], ys], axis=0)
    # ext row (order + t) is y_t, so lag j at step t is ext[order + t - 1 - j].
    return jnp.stack([ext[order - 1 - j: order - 1 - j + T] for j in range(order)],
                     axis=-1)


def _filter_step(model: GridModel, state, theta, y, warm, x=None):
    """One QR filter step at discount params ``θ = [δ_blocks, β]``. ``warm`` (0/1)
    forces ``δ = 1`` while the diffuse prior settles (β not forced). Returns
    ``(new_state, ell_ls, ell_sq, f, q)`` — the log-score loss, the frozen-variance
    squared-error loss (both oriented as losses to MINIMISE), and the one-step
    predictive mean/variance."""
    nb = model.n_blocks
    delta = theta[:nb][model.state_to_block]
    if model.adapt_guard is not None:                     # info-envelope covariance guard
        state_var = jnp.sum(state["Z"] ** 2, axis=0)      # diag(C) per state (K,)
        info = jnp.abs(state["m"]) * finite_inv(jnp.sqrt(state_var))
        delta = delta + (1.0 - delta) * jnp.exp(-model.adapt_guard * info)
    if model.damped is not None:
        delta = delta * model.damped                      # effective disc = θ·δ²
    delta = jnp.where(warm > 0.5, 1.0, delta)
    if model.inert_mask is not None:
        delta = jnp.where(model.inert_mask, 1.0, delta)   # padded states: frozen δ=1
    disc_factor = jnp.diag((1.0 - delta) / delta)
    beta = theta[nb]
    nu_prior = state["nu"]
    new_state, md = dlm_uv_fwd_qr_step(
        disc_factor, beta, jnp.asarray(model.var_power), model.mc, state,
        {"F": _F_at(model, x), "G": model.G, "y": y})
    e = y - md["f"]
    ell_ls = neg_log_pred(e, md["q"], nu_prior)
    ell_sq = 0.5 * e**2 / lax.stop_gradient(md["q"])   # frozen-var → quasi-score
    return new_state, ell_ls, ell_sq, md["f"], md["q"]


def batch_loss(model: GridModel, init_state, ys, theta, warmup: int = 0, xs=None):
    """``(Σ ell_ls, Σ ell_sq)`` at a FIXED discount vector ``θ`` — the batch
    objectives whose gradients the streaming RTRL scores must reproduce.

    ``xs`` ``(T, n_regs)`` supplies the regression tail's row at each step, as in
    :func:`run_cell`; ``None`` is a structural-only run. Two scan bodies rather
    than a dummy leaf, for the reason :func:`grid_stream_scan` gives: ``None``
    cannot be a scan input, and a zero-width stand-in would still take the
    regressor branch inside the step. The ``xs is None`` body is byte-identical
    to what it was, which is what keeps the structural path bit-exact."""
    warm = (jnp.arange(ys.shape[0]) < warmup).astype(ys.dtype)

    if xs is None:
        def step(state, packed):
            y, w = packed
            new_state, ell_ls, ell_sq, _f, _q = _filter_step(model, state, theta, y, w)
            return new_state, (ell_ls, ell_sq)

        _, (ls, sq) = lax.scan(step, init_state, (ys, warm))
    else:
        def step(state, packed):
            y, w, x = packed
            new_state, ell_ls, ell_sq, _f, _q = _filter_step(model, state, theta, y, w, x)
            return new_state, (ell_ls, ell_sq)

        _, (ls, sq) = lax.scan(step, init_state, (ys, warm, jnp.asarray(xs)))
    return jnp.sum(ls), jnp.sum(sq)


def grad_run(model: GridModel, init_state, ys, theta_seq, warmup: int = 0, xs=None):
    """Forward RTRL along a per-step discount path ``theta_seq`` (T, P). Carries
    ``S = ∂state/∂θ`` and returns per-step losses and per-parameter gradients (in
    PROBABILITY space — the consumer applies the logit chain rule):

    ``{ell_ls, ell_sq (T,), g_ls, g_sq (T, P), state}`` where ``g_*[t]`` is the
    total ``∂ℓ_t/∂θ`` (accounting for the whole state history via ``S``).

    ``xs`` ``(T, n_regs)`` supplies the regression tail's row at each step;
    ``None`` is a structural-only run. Unlike :func:`batch_loss` the two branches
    share one ``body`` and the structural one passes ``x=None`` explicitly, which
    is what ``_filter_step`` already defaulted to — so the structural path is
    unchanged in RESULT rather than byte-identical in source, and
    ``test_structural_run_is_unchanged_by_the_new_argument`` asserts that at
    bit-exactness.

    The tail does NOT add derivative terms: a regressor row is DATA, a constant
    to the jvp exactly as ``y`` is, so it is closed over rather than
    differentiated. ``test_grid_rtrl_regressors.py`` is what holds that claim to
    account."""
    P = model.n_params
    eye = jnp.eye(P)
    S0 = {k: jnp.zeros((P,) + v.shape) for k, v in init_state.items()}
    warm = (jnp.arange(ys.shape[0]) < warmup).astype(ys.dtype)

    def body(state, S, y, theta, w, x):
        new_state, ell_ls, ell_sq, f, q = _filter_step(model, state, theta, y, w, x)

        def dir_i(S_slice, ei):
            (_, _, _, _, _), (ns_dot, ls_dot, sq_dot, _, _) = jax.jvp(
                lambda st, th: _filter_step(model, st, th, y, w, x),
                (state, theta), (S_slice, ei))
            return ns_dot, ls_dot, sq_dot

        S_new, g_ls, g_sq = vmap(dir_i)(S, eye)
        return (new_state, S_new), (ell_ls, ell_sq, g_ls, g_sq, f, q)

    if xs is None:
        def step(carry, packed):
            state, S = carry
            y, theta, w = packed
            return body(state, S, y, theta, w, None)

        scanned = (ys, theta_seq, warm)
    else:
        def step(carry, packed):
            state, S = carry
            y, theta, w, x = packed
            return body(state, S, y, theta, w, x)

        scanned = (ys, theta_seq, warm, jnp.asarray(xs))

    (final_state, _), (ell_ls, ell_sq, g_ls, g_sq, f, q) = lax.scan(
        step, (init_state, S0), scanned)
    return {"state": final_state, "ell_ls": ell_ls, "ell_sq": ell_sq,
            "g_ls": g_ls, "g_sq": g_sq, "f": f, "q": q}


# --------------------------------------------------------------------------- #
# The cell: 3 wingmen on one structure, sharing a moving level centre.
# --------------------------------------------------------------------------- #
def _dma_update(pdr, mdr, c=DMA_C):
    """DLMAX's ``PowerLawUpdate`` rule, pre-bound with the within-class (``pdm``) and
    between-class (``mdm``) forgetting rates + floor — the update the ``Allocator``
    uses. Fed to :func:`~DLMAX.dlm_core.allocator_step` for both the cell's single-class
    DMA and the grid's hierarchical one (no re-implementation)."""
    return Partial(PowerLawUpdate, dma_pdr=pdr, dma_mdr=mdr, c=c)


def _adam(g, m, v, t, lr, b1, b2, eps=1e-8):
    """One Adam step; returns ``(step, new_m, new_v)`` (descend by subtracting step)."""
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g**2
    mhat = m / (1.0 - b1 ** t)
    vhat = v / (1.0 - b2 ** t)
    return lr * mhat / (jnp.sqrt(vhat) + eps), m, v


def _clip_box(model, clip=None):
    """Logit clip box for ONE model: ``(lo, hi)``, each ``(n_params,)``.

    The floor is the load-bearing bound — it is what a block's covariance growth
    is ultimately limited by, and what is survivable depends on the data's
    frequency and on the block's OWN period, since δ costs ``(1/δ)**period`` over
    one cycle. The default 0.5 means "this block's covariance may double every
    step": fine at monthly, ruinous at hourly.

    ``clip`` may be

    * ``None`` — the module constants (``CLIP_LO``/``CLIP_HI``, β
      ``BETA_CLIP``);
    * a ``(lo, hi)`` pair of floats — one box for every discount block, β still
      on ``BETA_CLIP``;
    * a dict with any of ``"level"`` / ``"seasonal"`` / ``"beta"``, each a
      ``(lo, hi)`` pair — per-block, which is the point: a level wants to move
      and a period-8766 Fourier does not, so one floor cannot serve both.
      ``"seasonal"`` is placed by the model's own ``seasonal_mask``, ``"level"``
      covers every other discount block, and omitted keys fall back to the
      constants.
    """
    P, nb = model.n_params, model.n_blocks
    if clip is None:
        clip = {}
    elif not isinstance(clip, dict):                      # (lo, hi) for all blocks
        clip = {"level": (clip[0], clip[1]), "seasonal": (clip[0], clip[1])}
    lvl = clip.get("level", (CLIP_LO, CLIP_HI))
    seas = clip.get("seasonal", lvl)
    # the regression tail gets its OWN box, defaulting to the seasonal one: an AR
    # coefficient or regression weight is a slowly-drifting quantity, closer in
    # character to a seasonal amplitude than to a level.
    reg = clip.get("regression", seas)
    beta = clip.get("beta", BETA_CLIP)

    lo = jnp.full(P, _logit(float(lvl[0]))).at[-1].set(_logit(float(beta[0])))
    hi = jnp.full(P, _logit(float(lvl[1]))).at[-1].set(_logit(float(beta[1])))
    if model.seasonal_mask is not None:
        sm = jnp.asarray(model.seasonal_mask)              # (nb,) per discount block
        lo = lo.at[:nb].set(jnp.where(sm, _logit(float(seas[0])), lo[:nb]))
        hi = hi.at[:nb].set(jnp.where(sm, _logit(float(seas[1])), hi[:nb]))
    if model.regression_mask is not None:
        rm = jnp.asarray(model.regression_mask)
        lo = lo.at[:nb].set(jnp.where(rm, _logit(float(reg[0])), lo[:nb]))
        hi = hi.at[:nb].set(jnp.where(rm, _logit(float(reg[1])), hi[:nb]))
    return lo, hi


def forecast_origin(model: GridModel, state, theta, h: int, xh=None,
                    seed_lags=None):
    """h-step RAW predictive ``(loc (h,), q (h,), nu ())`` for ONE worker from its filter
    state and discount vector ``θ`` (level/seasonal δ + β), via
    :func:`~DLMAX.dlm_core.dlm_uv_fcast_H`. Returns the Student-t scale² ``q`` and DOF
    ``nu`` un-scaled — the DMA combination (DLMAX's ``_combined_predictive_sd``, the same
    Vincent/quantile average AutoFFS uses) applies the variance-matched factor, so the grid
    is combined identically to AutoFFS and only the DISCOUNT mechanism differs."""
    nb = model.n_blocks
    dstate = theta[:nb][model.state_to_block]
    if model.adapt_guard is not None:                     # info-envelope covariance guard
        state_var = jnp.sum(state["Z"] ** 2, axis=0)      # diag(C) per state (K,)
        info = jnp.abs(state["m"]) * finite_inv(jnp.sqrt(state_var))
        dstate = dstate + (1.0 - dstate) * jnp.exp(-model.adapt_guard * info)
    if model.damped is not None:
        dstate = dstate * model.damped                    # effective disc = θ·δ²
    disc = (1.0 - dstate) / dstate
    beta = theta[nb]
    k = model.F.shape[0]
    Gp, GH = jnp.eye(k), []
    for _ in range(h):
        Gp = model.G @ Gp
        GH.append(Gp)
    FH = jnp.tile(model.F, (h, 1))
    if xh is not None and model.n_regs:
        o = model.reg_offset                       # (h, n_regs) future exog rows
        FH = FH.at[:, o:o + model.n_regs].set(jnp.asarray(xh))
    DH = {"GH": jnp.stack(GH), "FH": FH}
    out = dlm_uv_fcast_H(disc, beta, jnp.asarray(model.var_power), model.mc,
                         {"m": state["m"], "C": state["Z"].T @ state["Z"],
                          "nu": state["nu"], "s": state["s"]}, DH)
    if seed_lags is not None and model.n_regs:
        # AUTOREGRESSIVE tail: F at horizon j carries forecasts made below it, so
        # FH cannot express it. The STATE propagation is unaffected (G is
        # block-diagonal and the coefficient block is the identity), so aH/RH are
        # taken verbatim and only the observation step is redone sequentially --
        # West & Harrison's iterated expectations. n_reg=0 reduces this EXACTLY
        # to the f/q above, so it is a strict generalisation of the static path.
        f_h, q_h = iterated_obs_forecast(
            out["m"], out["C"], state["s"], beta,
            jnp.asarray(model.var_power), model.mc, model.F, int(model.n_regs),
            jnp.asarray(seed_lags))
        return f_h, q_h, out["nu"][0]
    return out["f"], out["q"], out["nu"][0]     # nu is const over h → scalar


def _wing_step(carry, y, wrm, x=None, *, model, offsets, lo, hi, use_ls, mi, upd,
               lr_level, lr_worker, b1, b2, disc_prior=None, couple_sd=None,
               seasonal_prior=None, regression_prior=None):
    """One online RTRL step of a 3-wingman cell — the streaming core shared by
    :func:`run_cell` (grid) and the builder's ``Wing`` path (``uv_dlm``).

    ``carry`` = ``(st, S, c, lm, lv, wth, wm, wv, alloc, t)``; returns
    ``(new_carry, out)`` with ``out = {c, theta, w, ell_ls, f, q}``. No h-step
    forecast — callers that need it add it from the returned carry. Pure per-step
    logic; :func:`run_cell` wraps it in a ``lax.scan`` + the cutoff forecast.
    """
    st, S, c, lm, lv, wth, wm, wv, alloc, t = carry
    P = model.n_params
    # Learn the discount only on a NON-warmup, FINITE observation. A masked obs
    # (NaN — e.g. a late launcher's pre-launch days) carries no information and
    # its score gradient is NaN; gating the Adam/centre update on isfinite(y)
    # keeps the learned discount finite (the kernel already holds the STATE via
    # ignore_obs). Bit-identical for clean data (all-finite -> act = wrm < 0.5).
    act = (wrm < 0.5) & jnp.isfinite(y)

    lvl = jnp.clip(c + offsets, lo[0], hi[0])             # place wingmen (3,)
    wth = wth.at[:, 0].set(lvl)
    theta = jax.nn.sigmoid(wth)                           # (3, P)

    Ft = _F_at(model, x)                                  # this step's obs vector

    def one(st_i, S_i, th_i):
        # Primal via the kernel directly so the full md (incl v_sys) is available;
        # numerically identical to _filter_step (same disc/step/ell). _filter_step
        # is still used inside the jvp for the RTRL gradient.
        nb = model.n_blocks
        delta = th_i[:nb][model.state_to_block]
        if model.adapt_guard is not None:                     # info-envelope covariance guard
            state_var = jnp.sum(st_i["Z"] ** 2, axis=0)       # diag(C) per state (K,)
            info = jnp.abs(st_i["m"]) * finite_inv(jnp.sqrt(state_var))
            delta = delta + (1.0 - delta) * jnp.exp(-model.adapt_guard * info)
        if model.damped is not None:
            delta = delta * model.damped                      # effective disc = θ·δ²
        delta = jnp.where(wrm > 0.5, 1.0, delta)
        if model.inert_mask is not None:
            delta = jnp.where(model.inert_mask, 1.0, delta)   # padded states: frozen δ=1
        ns, md = dlm_uv_fwd_qr_step(
            jnp.diag((1.0 - delta) / delta), th_i[nb],
            jnp.asarray(model.var_power), model.mc, st_i,
            {"F": Ft, "G": model.G, "y": y})
        ell_ls = neg_log_pred(y - md["f"], md["q"], st_i["nu"])
        def dir_i(Ss, ei):
            (_, _, _, _, _), (ns_dot, ls_dot, sq_dot, _, _) = jax.jvp(
                lambda s, th: _filter_step(model, s, th, y, wrm, x),
                (st_i, th_i), (Ss, ei))
            return ns_dot, ls_dot, sq_dot
        S_new, g_ls, g_sq = vmap(dir_i)(S_i, jnp.eye(P))
        return ns, S_new, g_ls, g_sq, ell_ls, md["f"], md["q"], md["v_sys"]

    ns, S_new, g_ls, g_sq, ell_ls, f, q, v_sys = vmap(one)(st, S, theta)

    fc = ForecastBundle(f[:, None, None], q[:, None, None])
    alloc_n, wt = allocator_step(alloc, fc, y[None], LogScore, IdentityAggregator,
                                 upd, mi)
    w_new = wt[:, 0, 0]                                   # (3,) in-class weights

    chain = theta * (1.0 - theta)
    gsel = jnp.where(use_ls[None, :], g_ls, g_sq)         # objective-routed
    glogit = gsel * chain                                 # (3, P), logit-space

    t_n = t + act
    tb = jnp.maximum(t_n, 1.0)

    if disc_prior is not None:
        # MAP: logit-space Gaussian priors as the gradient of the LOSS (-log π),
        # weighted 1/t (running, ``tb`` scalar) so it acts like a per-datum prior.
        # ``disc_prior`` = (d_mean, d_sd) puts N(d_mean, d_sd) on the discount blocks
        # (0..nb-1) and leaves β unpriored; the 4-tuple (d_mean, d_sd, b_mean, b_sd)
        # ALSO puts N(b_mean, b_sd) on the variance discount β. Pulls δ (and β)
        # toward the prior mean, countering the RTRL collapse to the clip.
        nb = model.n_blocks
        pm = jnp.full(P, disc_prior[0])
        ps = jnp.full(P, disc_prior[1])
        if seasonal_prior is not None and model.seasonal_mask is not None:
            # Seasonal (Fourier) discount blocks are anchored HIGHER — N(seasonal_
            # prior, disc_prior[1]) instead of the level/trend N(disc_prior[0], .) —
            # they want to be stiffer for stability. seasonal_mask is (nb,); β keeps
            # the level mean.
            #
            # ``seasonal_prior`` is a scalar (one mean for every seasonal block) or
            # a per-block ``(nb,)`` vector — several seasonals in one cell can want
            # stiffnesses orders of magnitude apart, since what matters is the
            # covariance growth over the block's OWN period: δ=.982 costs
            # exp(0.44) over a 24-step daily cycle and exp(160) over an 8766-step
            # annual one. jnp.where broadcasts either shape.
            pm = pm.at[:nb].set(jnp.where(model.seasonal_mask,
                                          jnp.asarray(seasonal_prior),
                                          disc_prior[0]))
        if regression_prior is not None and model.regression_mask is not None:
            # the tail anchors like a seasonal by default (the caller passes
            # seasonal_prior through when regression_prior is unset): a drifting
            # AR coefficient is a slow quantity, not a level.
            pm = pm.at[:nb].set(jnp.where(model.regression_mask,
                                          jnp.asarray(regression_prior),
                                          pm[:nb]))
        prior_g = (wth - pm[None, :]) / (ps[None, :] ** 2 * tb)     # (3, P)
        if len(disc_prior) >= 4:
            prior_g = prior_g.at[:, nb].set(
                (wth[:, nb] - disc_prior[2]) / (disc_prior[3] ** 2 * tb))
        else:
            prior_g = prior_g.at[:, nb].set(0.0)                    # no β prior
        glogit = glogit + prior_g

    if couple_sd is not None:
        # Growth-to-level coupling: the growth block's (block 1) prior MEAN is the
        # LEVEL block's (block 0) current logit — a soft version of the fused
        # level/growth lock. Gated per-family by ``couple_growth`` (0 -> no change,
        # bit-exact) so only decoupled trend families are coupled; seasonal/inert
        # block-1's of other families are untouched. Anchors the growth to the
        # multi-step-correct level discount where it can't be identified (annual),
        # while the likelihood can still pull it away where it earns it (M/Q).
        cg = jnp.asarray(model.couple_growth, glogit.dtype)
        couple_b1 = (wth[:, 1] - wth[:, 0]) / (couple_sd ** 2 * tb)     # (3,)
        # The coupling REPLACES any fixed disc_prior anchor on the growth block:
        # subtract what disc_prior added to block 1 (its N(d_mean,d_sd) term) so a
        # coupled family's growth is anchored to the level, not to the constant.
        dp_b1 = ((wth[:, 1] - disc_prior[0]) / (disc_prior[1] ** 2 * tb)
                 if disc_prior is not None else 0.0)
        glogit = glogit.at[:, 1].add(cg * (couple_b1 - dp_b1))

    step_l, lm_u, lv_u = _adam(glogit[:, 0], lm, lv, tb, lr_level, b1, b2)   # (3,)
    pos = lvl - step_l                                    # tentative wing positions (3,)
    c_new = jnp.sum(w_new * pos)                          # DMA-weighted position = centre
    c_n = jnp.where(act, jnp.clip(c_new, lo[0], hi[0]), c)
    lm_n = jnp.where(act, lm_u, lm); lv_n = jnp.where(act, lv_u, lv)

    step_w, wm_u, wv_u = _adam(glogit, wm, wv, tb, lr_worker, b1, b2)
    wth_stepped = jnp.clip(wth - step_w, lo[None, :], hi[None, :])
    wth_n = jnp.where(act, wth.at[:, 1:].set(wth_stepped[:, 1:]), wth)
    wm_n = jnp.where(act, wm_u, wm); wv_n = jnp.where(act, wv_u, wv)

    out = {"c": c, "theta": theta, "w": w_new, "ell_ls": ell_ls, "f": f, "q": q,
           "v_sys": v_sys}
    return (ns, S_new, c_n, lm_n, lv_n, wth_n, wm_n, wv_n, alloc_n, t_n), out


def run_cell(model: GridModel, init_state, ys, *, offset: float = 1.0,
             lr_level: float = 0.05, lr_worker: float = 0.05, mu0_disc=None,
             mu0_beta=None, warmup: int = 0, pdr=DMA_PDR, mdr=DMA_MDR,
             dma_c=DMA_C, adam_betas=(0.9, 0.99), h: int = 0, cutoffs=None,
             additive_logscore: bool = False, disc_prior=None, couple_sd=None,
             seasonal_prior=None, regression_prior=None, xs=None, xh=None,
             seed_lags=None, clip=None, disc_init=None):
    """Online 3-wingman cell on ONE series (the movable-grid unit).

    ``xs`` ``(T, n_regs)`` supplies the regression tail's row at each step, and
    ``xh`` ``(T, h, n_regs)`` its FUTURE rows at each origin (read only on
    cutoff steps) -- the EXOGENOUS pair, where the caller knows the design over
    the horizon.

    ``seed_lags`` ``(T, n_regs)`` is the AUTOREGRESSIVE alternative: the lags
    that seed the iterated forecast emitted at a cutoff ``t``, most-recent-first
    ``[y_t, y_{t-1}, ...]``. It is the per-cell analogue of
    ``multi_model_dlm.format_seed_lag_yts``, which the legacy CV path uses.
    Without it an AR tail filters correctly but forecasts as though its
    coefficients were zero, and does so silently. Supply ``xs`` with it
    (the filter-time lag design); ``xh`` is then unused, since the future
    regressors ARE the forecasts.

    All ``None`` for a structural cell, which is the bit-exact path.

    ``h``/``cutoffs``: for the rolling backtest — at each step in ``cutoffs`` emit each
    wingman's ``h``-step origin forecast (``loc_h``/``sd_h`` ``(T,3,h)``, non-cutoff steps
    zero) from its post-update state + discount, so origin ``t`` uses exactly what the
    grid learned by ``t``. ``h=0`` (default) skips forecasting (the online-diagnostic path).

    The three wingmen share the model structure and a single moving **level centre**
    (logit); they sit at ``centre + {-offset, 0, +offset}``. Each step:

    1. place the wingmen at ``centre ± offset`` and advance all three one RTRL step
       (carrying ``S`` per wingman → per-parameter gradients ``g_ls``/``g_sq``);
    2. update the in-class pooling weights via DLMAX's single-class ``allocator_step``
       (``PowerLawUpdate``, ``pdm`` over the 3 wingmen — the genuine DLMAX DMA);
    3. move the **level centre** by *position weighting*: each wing Adam-steps its level
       by its own quasi-score (frozen-var squared-error) slope, then the new centre is the
       DMA-weighted mean of the resulting positions. This is ``consensus`` (toward the
       favoured wing) + ``derivative`` (toward where the slope says better, which can reach
       beyond the current wings), and it is robust — a bad wing move is performance-gated
       by the DMA;
    4. re-place the wingmen at the new ``centre ± offset``;
    5. Adam-step each wingman's own seasonal blocks (squared-error grad) and β
       (log-score grad) — block 0 (level/trend) is the pooled/shared one, blocks
       ``1..nb-1`` + β are per-worker.

    Discount learning is frozen during ``warmup`` (δ forced to 1 there anyway). Returns
    per-step trajectories (``c_traj``, ``theta_traj (T,3,P)``, ``w_traj (T,3)``, one-step
    ``f``/``q``) and the final states/discounts for origin forecasting."""
    P, nb = model.n_params, model.n_blocks
    lo, hi = _clip_box(model, clip)
    offsets = jnp.array([-offset, 0.0, offset])          # fast / centre / slow (logit)
    use_ls = jnp.zeros(P, bool).at[nb].set(True)         # β → log score; mean blocks → sq
    if additive_logscore:
        # additive families (var_power == 1) learn the discount blocks on LOG SCORE
        # too; non-additive keep quasi-score (log score failed for multiplicative).
        use_ls = jnp.where(model.var_power == 1.0, jnp.ones(P, bool), use_ls)
    b1, b2 = adam_betas
    mu0d = _logit(0.95) if mu0_disc is None else mu0_disc
    mu0b = _logit(0.99) if mu0_beta is None else mu0_beta

    st0 = {k: jnp.broadcast_to(v, (3,) + v.shape) for k, v in init_state.items()}
    S0 = {k: jnp.zeros((3, P) + v.shape) for k, v in init_state.items()}
    if disc_init is None:
        wth0 = jnp.full((3, P), mu0d).at[:, -1].set(mu0b)
    else:
        # THIS family's starting discounts, ``(P,)`` in logit space -- the
        # rolling analogue of grid_stream_static's per-family disc_init. A block
        # whose cells carry their own discounts (AdaptiveBlock.from_cells) must
        # start there on both paths: a long-period seasonal cannot be started at
        # the uniform 0.95 and learned back.
        wth0 = jnp.broadcast_to(jnp.clip(jnp.asarray(disc_init, dtype=float),
                                         lo, hi), (3, P))
    warm = (jnp.arange(ys.shape[0]) < warmup).astype(ys.dtype)

    mi = jnp.ones((3, 1), dtype=bool)                    # one class: the 3 wingmen
    upd = _dma_update(pdr, mdr, dma_c)
    alloc0 = init_alloc_state(3, 1, 1, mi, 1)
    if cutoffs is None:
        is_cut = jnp.zeros(ys.shape[0], bool)
    else:
        is_cut = jnp.zeros(ys.shape[0], bool).at[jnp.asarray(cutoffs)].set(True)

    # Python-level flags, so the branches resolve at trace time and a structural
    # cell scans exactly the tuple it always did. Zero-width stand-ins keep the
    # scan signature uniform without ever being read.
    has_x, has_xh = xs is not None, (bool(h) and xh is not None)
    has_sl = bool(h) and seed_lags is not None
    T_ = ys.shape[0]
    xs_s = jnp.asarray(xs) if has_x else jnp.zeros((T_, 0))
    xh_s = jnp.asarray(xh) if has_xh else jnp.zeros((T_, 0))
    sl_s = jnp.asarray(seed_lags) if has_sl else jnp.zeros((T_, 0))

    def cell_step(carry, packed):
        y, wrm, cut, x_t, xh_t, sl_t = packed
        new_carry, out = _wing_step(
            carry, y, wrm, x_t if has_x else None,
            model=model, offsets=offsets, lo=lo, hi=hi,
            use_ls=use_ls, mi=mi, upd=upd, lr_level=lr_level, lr_worker=lr_worker,
            b1=b1, b2=b2, disc_prior=disc_prior, couple_sd=couple_sd,
            seasonal_prior=seasonal_prior, regression_prior=regression_prior)
        if h:
            ns, c_n, wth_n = new_carry[0], new_carry[2], new_carry[5]
            # h-step origin forecast per wingman from the POST-update state + next
            # discount (level = new centre +/- offset), gated to cutoff steps.
            th_next = jax.nn.sigmoid(wth_n.at[:, 0].set(jnp.clip(c_n + offsets,
                                                                 lo[0], hi[0])))
            loc_h, q_h, nu_h = lax.cond(
                cut,
                lambda: vmap(lambda s_i, t_i: forecast_origin(
                    model, s_i, t_i, h, xh_t if has_xh else None,
                    sl_t if has_sl else None))(ns, th_next),
                lambda: (jnp.zeros((3, h)), jnp.zeros((3, h)), jnp.zeros(3)))
            out["loc_h"] = loc_h
            out["q_h"] = q_h
            out["nu_h"] = nu_h
            out["theta_used"] = th_next
        return new_carry, out

    carry0 = (st0, S0, mu0d, jnp.zeros(3), jnp.zeros(3), wth0,
              jnp.zeros((3, P)), jnp.zeros((3, P)), alloc0, 0.0)
    (st_f, S_f, c_f, lm_f, lv_f, wth_f, wm_f, wv_f, alloc_f, t_f), out = lax.scan(
        cell_step, carry0, (ys, warm, is_cut, xs_s, xh_s, sl_s))
    # re-place the wingmen at the FINAL centre so the returned discounts are
    # forecast-ready and consistent with ``c`` (the scan carries them one update behind).
    wth_f = wth_f.at[:, 0].set(jnp.clip(c_f + offsets, lo[0], hi[0]))
    res = {"c_traj": out["c"], "theta_traj": out["theta"], "w_traj": out["w"],
           "ell_ls": out["ell_ls"], "f": out["f"], "q": out["q"],
           "state": st_f, "wth": wth_f, "c": c_f, "w": out["w"][-1]}
    if h:
        res["loc_h"] = out["loc_h"]           # (T, 3, h) — nonzero at cutoffs
        res["q_h"] = out["q_h"]               # (T, 3, h)
        res["nu_h"] = out["nu_h"]             # (T, 3)
        res["theta_used"] = out["theta_used"]  # (T, 3, P)
    return res


# --------------------------------------------------------------------------- #
# The grid: 16 structures × 3 wingmen = 48 workers; DLMAX hierarchical allocator
# (families = classes, pdm within / mdm between) → mixture combine.
# --------------------------------------------------------------------------- #
import numpy as _np
import pandas as _pd
from jax.scipy.linalg import block_diag as _block_diag

from .dlm_builder import Fourier, LocalLevel, LocalTrend, PriorContext

DAMPINGS = (1.0, 0.99, 0.95, 0.0)      # undamped / damped / damped / level-only (0)
DISC_TREND, DISC_SEAS = 0.95, 0.99     # prior-centre discounts (component disc_rate)


def _components(damping, seasonal, period, *, mult, var_power, n_comps=None,
                period2=None, n_comps2=None):
    comps = ([LocalLevel(name="trend", disc_rate=DISC_TREND)] if damping == 0.0
             else [LocalTrend(name="trend", disc_rate=DISC_TREND, damping=damping)])
    if seasonal:
        comps.append(Fourier(name="seasonal", period=period, disc_rate=DISC_SEAS,
                             n_comps=n_comps, multiplicative=mult))
        # Dual seasonality (M4 Hourly): a SECOND Fourier so every seasonal model
        # carries BOTH a daily (period) and weekly (period2) cycle at once — the
        # "carry both" structure (vs a DMA-picks-one union), matching the static
        # m4_forecast_h. Only added when period2 is set.
        if period2 is not None:
            comps.append(Fourier(name="seasonal_weekly", period=period2,
                                 disc_rate=DISC_SEAS, n_comps=n_comps2,
                                 multiplicative=mult))
    return comps, var_power


def _order_comps(comps):
    """Structural components first, regression tail last -- the order
    ``DLM.compile`` itself uses (``ordered = structural + regression``). The grid
    took ``self.components`` in insertion order, which would put a tail in the
    middle if a caller added AR before a seasonal. No-op for structural-only
    cells, so bit-exact."""
    comps = list(comps)
    return ([c for c in comps if not getattr(c, "is_regression", False)]
            + [c for c in comps if getattr(c, "is_regression", False)])


def _grid_model(comps, var_power, decouple_trend: bool = False,
                couple_trend: bool = False) -> GridModel:
    comps = _order_comps(comps)
    F = jnp.concatenate([c.F_block() for c in comps])
    G = _block_diag(*[c.G_block() for c in comps])
    mc = jnp.concatenate([c.mult_block() for c in comps])
    # Per-state damped-discount correction (δ² on a LocalTrend's growth slot, 1
    # elsewhere): the effective discount is θ·damped, mirroring the SVD path.
    damped = jnp.concatenate([c.disc_damped_block() for c in comps])
    # Default: one discount block per component (level+growth of a LocalTrend
    # share one fused δ). ``decouple_trend``: the 2-state polynomial trend learns
    # a SEPARATE discount per state (level=block b, growth=block b+1) — the wing
    # still searches the level (block 0), the growth δ is RTRL-learned under the
    # same prior. Mirrors ``dlm_builder._adapt_geometry``'s per-state layout.
    # ``couple_trend`` flags such families so the growth block's prior is anchored
    # to the level (see ``_wing_step``); trend is the leading component -> growth
    # is block 1, so the flag alone (with the always-block-1 coupling) suffices.
    s2b, b, couple, seas, regs = [], 0, False, [], []
    for c in comps:
        # By type first (``Component.is_seasonal``), name second for anything
        # that predates the flag. Identical for every ``build_grid`` taxonomy —
        # its seasonals are Fourier AND carry those names — but it lets a
        # hand-built cell carry several Fourier blocks, which cannot all be
        # called "seasonal" (component names are unique within a DLM).
        is_seas = (getattr(c, "is_seasonal", False)
                   or c.name in ("seasonal", "seasonal_weekly"))
        if decouple_trend and c.name == "trend" and c.state_dim > 1:
            s2b.extend(range(b, b + c.state_dim)); b += c.state_dim
            couple = couple_trend
        else:
            s2b.extend([b] * c.state_dim)
            if is_seas:
                seas.append(b)
            if getattr(c, "is_regression", False):
                regs.append(b)
            b += 1
    seasonal_mask = _np.zeros(b, bool)          # per discount block (n_blocks,)
    seasonal_mask[seas] = True
    regression_mask = _np.zeros(b, bool)
    regression_mask[regs] = True
    # The tail is contiguous and last (``comps`` is ordered structural-then-
    # regression by _order_comps), so its offset is the structural state count.
    n_regs = sum(c.state_dim for c in comps if getattr(c, "is_regression", False))
    reg_offset = sum(c.state_dim for c in comps
                     if not getattr(c, "is_regression", False))
    return GridModel(F=F, G=G, mc=mc, var_power=var_power,
                     state_to_block=jnp.asarray(s2b, dtype=int), n_blocks=b,
                     couple_growth=couple, damped=damped,
                     seasonal_mask=jnp.asarray(seasonal_mask),
                     n_regs=n_regs, reg_offset=reg_offset,
                     regression_mask=jnp.asarray(regression_mask))


def build_grid(period, dampings=DAMPINGS, var_powers=None, n_comps=None,
               seasonal_mult=False, decouple_trend=False, couple_trend=False,
               period2=None, n_comps2=None):
    """The structure taxonomy: ``dampings × {seasonal off/on} × variants`` — each a
    3-wingman cell. ``period=None`` drops the seasonal structures. Returns a list of
    ``(name, GridModel, components)``; the name tags the class (e.g. ``T.99S-M``).

    ``n_comps`` truncates the seasonal Fourier to that many harmonics (``None`` =
    full ``period//2``). Keeps the state small for long periods (e.g. weekly M4:
    period 52 with ``n_comps=12`` -> 24 seasonal states instead of ~52).

    ``seasonal_mult`` (list-form ``var_powers`` only) makes the Fourier seasonal
    MULTIPLICATIVE while keeping the listed error law — i.e. it UNWELDS
    multiplicative seasonality from the multiplicative-error variance. So
    ``var_powers=[1.0], seasonal_mult=True`` gives level-scaling seasonality with
    additive (Gaussian) error, and ``[1.0, 0.25]`` adds a compound-Poisson variant.

    ``var_powers`` selects the error variants:

    * ``None`` (default) — the classic ``{A, M}`` pair: additive error
      (``var_power=1``, additive Fourier) and multiplicative error
      (``var_power=0``, multiplicative Fourier).
    * a list, e.g. ``[1.0, 0.25]`` — one **additive-Fourier** family per
      ``var_power`` value (the M5 compound-Poisson sweep: 1 = Gaussian,
      0.25 = near-Poisson). Names tag the value, e.g. ``T1S-v.25``.
    """
    seasonal_opts = (False,) if period is None else (False, True)
    if var_powers is None:
        variants = [("A", False, 1.0), ("M", True, 0.0)]        # classic A/M pair
    else:
        # additive (or, with seasonal_mult, MULTIPLICATIVE) Fourier at each var_power
        _pre = "Ms" if seasonal_mult else "v"
        variants = [(f"{_pre}{vp:g}".replace("0.", "."), seasonal_mult, float(vp))
                    for vp in var_powers]
    out = []
    for d in dampings:
        tag = "L" if d == 0.0 else f"T{d:g}".replace("0.", ".")
        for seasonal in seasonal_opts:
            for vname, mult, vp in variants:
                comps, var_power = _components(d, seasonal, period, mult=mult,
                                               var_power=vp, n_comps=n_comps,
                                               period2=period2, n_comps2=n_comps2)
                name = f"{tag}{'S' if seasonal else ''}-{vname}"
                out.append((name, _grid_model(comps, var_power,
                                              decouple_trend=decouple_trend,
                                              couple_trend=couple_trend),
                            tuple(comps)))
    return out


def grid_init(components, ys, warmup: int, var_power: float = 1.0, nu0: float = 1.0,
              component_priors=None):
    """Diffuse ``{m, Z, s, nu}`` for ONE series from its warmup window, via the same
    ``PriorContext`` AutoFFS uses, so the grid sees the identical prior.

    ``var_power`` (0 multiplicative, 1 additive) level-normalises the stored obs scale
    exactly as ``PriorContext`` does (``dlm_core`` ``s = V0 / var_scale(m0·F)``): the
    kernels multiply the stored ``s`` by ``var_scale = |f|^(2(1-var_power))`` each step,
    so a MULTIPLICATIVE worker must store the RELATIVE scale ``V0 / f0²`` — otherwise its
    obs variance starts at ``V0·f0²`` (astronomically wide at large levels)."""
    # Elicit the diffuse prior from the first ``warmup`` FINITE observations, not
    # ys[:warmup] verbatim: a late-launching series (e.g. M5) has leading NaN
    # (pre-launch, skipped by the filter), so ys[:warmup] can be all-NaN and the
    # prior would collapse to NaN. Skipping the leading NaN is bit-identical for
    # clean data (no leading NaN -> start=0 -> ys[:warmup]).
    _ys = _np.asarray(ys)
    _fin = _np.flatnonzero(_np.isfinite(_ys))
    _start = int(_fin[0]) if _fin.size else 0
    yw = _ys[_start:_start + warmup][None, :]                   # (1, warm)
    ctx = PriorContext(_pd.DataFrame(yw.T), list(components), diffuse_only=True)
    m0b, C0b, Fb = [], [], []
    for c in components:
        if component_priors is not None and c.name in component_priors:
            # Informative (hierarchical) prior for this component — a new
            # late-launcher warm-started from siblings (AutoFFSUniverse.add_series).
            # Sliced to this component's state_dim, since the grid's families carry
            # the same-named component at different sizes (LocalLevel k=1 vs
            # LocalTrend k=2; variable Fourier order) — each takes the leading
            # states of the (largest-component) sibling prior.
            pm, pC = component_priors[c.name]
            sd = c.state_dim
            m0c = jnp.asarray(pm)[..., :sd]                    # (1, sd)
            C0c = jnp.asarray(pC)[..., :sd, :sd]               # (1, sd, sd)
        else:
            m0c, C0c = c.diffuse_prior(ctx)                    # (1, sd), (1, sd, sd)
        m0b.append(jnp.asarray(m0c)[0]); C0b.append(jnp.asarray(C0c)[0])
        Fb.append(c.F_block())
    m0 = jnp.concatenate(m0b)
    C0 = _block_diag(*C0b)
    V0 = jnp.maximum(jnp.nanvar(jnp.asarray(yw)), 1e-6)
    var_scale = var_scale_fn(m0 @ jnp.concatenate(Fb), var_power)   # dlm_core.py:237,256
    s0 = V0 / var_scale
    ev, UC = jnp.linalg.eigh(C0)
    Z = jnp.sqrt(jnp.clip(ev, 1e-30, None))[:, None] * UC.T
    return {"m": m0, "Z": Z, "s": s0, "nu": jnp.asarray(nu0)}


def _hier_dma_traj(F, Q, obs, n_classes, pdr, mdr, c=DMA_C):
    """Hierarchical DMA weight trajectory via DLMAX's :func:`~DLMAX.dlm_core.allocator_step`
    (``PowerLawUpdate``): the ``n_classes`` families are the classes (3 wingmen each,
    contiguous), ``pdm`` forgets within a family, ``mdm`` across families. ``F``/``Q``:
    per-worker one-step mean/variance ``(T, M)``; ``obs`` ``(T,)``. Returns combined
    weights ``(T, M)`` (``pset × mset``, summing to 1 over the M workers each step)."""
    T, M = F.shape
    mi = jnp.zeros((M, n_classes), bool).at[jnp.arange(M), jnp.arange(M) // 3].set(True)
    upd = _dma_update(pdr, mdr, c)
    alloc0 = init_alloc_state(M, 1, n_classes, mi, 1)

    def step(alloc, xs):
        f, q, y = xs
        fc = ForecastBundle(f[:, None, None], q[:, None, None])
        alloc_n, w = allocator_step(alloc, fc, y[None], LogScore, IdentityAggregator,
                                    upd, mi)
        return alloc_n, w[:, 0, 0]

    _, Wt = lax.scan(step, alloc0, (F, Q, obs))
    return Wt


# --- SGDDMA: online-learned hierarchical forgetting rates (Type-II ML + prior) ------
DMA_LR = 0.05
DMA_PRIOR = (2.0, 1.0)      # N(2,1) logit-space prior on (pdr, mdr) -> anchors the
                            # learned DMA rate at sigmoid(2)=0.88, near the ~0.9 data
                            # optimum (N(3,1)=0.95 overshot -> +2% MASE on M4M).


def _sgd_dma_step(carry, f, q, y, w_flag, *, mi, c, dma_prior, lr, b1, b2, lo, hi):
    """ONE SGDDMA RTRL/MAP step for a single series — the shared core of the batch
    scan (:func:`_sgd_dma_traj`) and the streaming grid combine (``_grid_stream_step``).
    ``carry`` = ``(pset, mset, th, ma, va, Sp, Sm, t)``; returns ``(new_carry,
    weights (M,), (pdr, mdr))``. State ``(pset, mset)`` and the RTRL sensitivity
    ``S`` advance every step; the θ/Adam update is gated on a non-warmup finite obs."""
    pset, mset, th, ma, va, Sp, Sm, t = carry
    M = f.shape[0]

    def pm(pset, mset, th):
        pdr = jax.nn.sigmoid(th[0]); mdr = jax.nn.sigmoid(th[1])
        state = AllocatorState(pset[:, None, None], mset[:, None, None],
                               jnp.zeros((1, M, 1, 2)))
        fc = ForecastBundle(f[:, None, None], q[:, None, None])
        st_n, w = allocator_step(state, fc, y[None], LogScore, IdentityAggregator,
                                 _dma_update(pdr, mdr, c), mi)
        w = w[:, 0, 0]                                    # (M,)
        loc = (w * f).sum()
        var = jnp.maximum((w * (q + f ** 2)).sum() - loc ** 2, 1e-12)
        nll = 0.5 * (jnp.log(2 * jnp.pi * var) + (y - loc) ** 2 / var)
        return (st_n.pset[:, 0, 0], st_n.mset[:, 0, 0], nll), w

    (pp, mm, _nll), w = pm(pset, mset, th)               # primal (advances state)

    def dir_i(Sp_i, Sm_i, e_i):                          # RTRL directional jvp
        _primal, (pp_d, mm_d, nll_d) = jax.jvp(
            lambda a, b, t_: pm(a, b, t_)[0], (pset, mset, th), (Sp_i, Sm_i, e_i))
        return pp_d, mm_d, nll_d

    Sp_n, Sm_n, nll_g = vmap(dir_i)(Sp, Sm, jnp.eye(2))  # (2,M),(2,nc),(2,)
    act = (w_flag < 0.5) & jnp.isfinite(y)               # learn on real obs only
    tb = jnp.maximum(t + act, 1.0)
    g = nll_g + (th - dma_prior[0]) / (dma_prior[1] ** 2 * tb)    # MAP gradient
    stp, ma_u, va_u = _adam(g, ma, va, tb, lr, b1, b2)
    th_u = jnp.clip(th - stp, lo, hi)
    th_n = jnp.where(act, th_u, th)                      # gate θ/Adam on act
    ma_n = jnp.where(act, ma_u, ma); va_n = jnp.where(act, va_u, va)
    return (pp, mm, th_n, ma_n, va_n, Sp_n, Sm_n, t + act), w, jax.nn.sigmoid(th)


def _sgd_dma_carry0(M, mi, pdr0=DMA_PDR, mdr0=DMA_MDR):
    """Initial per-series SGDDMA carry: diffuse ``(pset, mset)`` (as the fixed
    allocator), θ at ``logit(pdr0/mdr0)``, zero Adam moments + RTRL sensitivity."""
    nc = int(jnp.asarray(mi).shape[1])
    a0 = init_alloc_state(M, 1, nc, mi, 1)
    return (a0.pset[:, 0, 0], a0.mset[:, 0, 0],
            jnp.array([_logit(pdr0), _logit(mdr0)]), jnp.zeros(2), jnp.zeros(2),
            jnp.zeros((2, M)), jnp.zeros((2, nc)), 0.0)


def _sgd_dma_traj(F, Q, obs, mi, *, c=DMA_C, dma_prior=DMA_PRIOR,
                  lr=DMA_LR, betas=(0.9, 0.99), warmup=0,
                  pdr0=DMA_PDR, mdr0=DMA_MDR):
    """SGDDMA weight trajectory (ONE series) over an arbitrary ``mi`` (M, C)
    model->class indicator — the grid's worker->family (``_hier_dma_traj_sgd``) OR
    the block-diagonal union (``_union_dma_weights``). The forgetting rates
    ``(pdr, mdr)`` are LEARNED online by RTRL/MAP — Type-II ML on the combined
    one-step log score + an ``N(dma_prior)`` logit-space prior (1/t-weighted so it
    anchors short series) — rather than fixed. Reuses ``allocator_step`` /
    ``PowerLawUpdate`` as the differentiated forward (the ``+c`` floor keeps every
    weight > 0, so the gradient is finite), so a FROZEN θ reproduces the fixed
    replay to float precision. State ``(pset, mset)`` advances every step (like the
    fixed path); only the θ/Adam update is gated on non-warmup finite obs, while the
    RTRL sensitivity ``S = ∂(pset,mset)/∂θ`` tracks throughout.

    Returns ``(Wt (T, M), DISCt (T, 2))`` — weights + learned ``(pdr, mdr)`` per step
    (probability space). ``mset`` between-class, ``pset`` within-class."""
    T, M = F.shape
    mi = jnp.asarray(mi)
    nc = int(mi.shape[1])
    b1, b2 = betas
    lo, hi = _logit(CLIP_LO), _logit(CLIP_HI)
    warm = (jnp.arange(T) < warmup).astype(F.dtype)

    def step(carry, xs):
        f, q, y, w_flag = xs
        new_c, w, disc = _sgd_dma_step(carry, f, q, y, w_flag, mi=mi, c=c,
                                       dma_prior=dma_prior, lr=lr, b1=b1, b2=b2,
                                       lo=lo, hi=hi)
        return new_c, (w, disc)

    carry0 = _sgd_dma_carry0(M, mi, pdr0, mdr0)
    _, (Wt, DISCt) = lax.scan(step, carry0, (F, Q, obs, warm))
    return Wt, DISCt


def _hier_dma_traj_sgd(F, Q, obs, n_classes, **kw):
    """Grid wrapper for :func:`_sgd_dma_traj`: the hierarchical worker->family
    indicator (3 wingmen per family, contiguous). ``**kw`` forwards the learning
    knobs (``c``, ``dma_prior``, ``warmup``, ...)."""
    M = F.shape[1]
    mi = jnp.zeros((M, n_classes), bool).at[jnp.arange(M), jnp.arange(M) // 3].set(True)
    return _sgd_dma_traj(F, Q, obs, mi, **kw)


def sgd_union_carry0(M, mi, q, pdr0=DMA_PDR, mdr0=DMA_MDR):
    """Per-series SGDDMA carry for the streaming UNION (``q`` copies of
    :func:`_sgd_dma_carry0`, series on the leading axis) — the learned analogue of
    the union ``AllocatorState``, so ``add_series`` pad/set/append are uniform
    axis-0 tree ops."""
    return vmap(lambda _i: _sgd_dma_carry0(M, mi, pdr0, mdr0))(jnp.arange(q))


def sgd_union_step_q(carry_q, f, q_, y, warm, *, mi, c=DMA_C, dma_prior=DMA_PRIOR,
                     lr=DMA_LR, betas=(0.9, 0.99)):
    """One streaming UNION SGDDMA step, vmapped over the ``q`` series (each combiner
    self-tunes its own rate). ``carry_q``: series-leading SGDDMA carry; ``f``/``q_``:
    ``(M, q)`` one-step mean/var; ``y``: ``(q,)``; ``warm``: scalar 0/1. Returns
    ``(new_carry_q, weights (M, q))`` — same weight layout as the fixed union step."""
    b1, b2 = betas
    lo, hi = _logit(CLIP_LO), _logit(CLIP_HI)

    def one(carry, fj, qj, yj):
        new_c, w, _disc = _sgd_dma_step(carry, fj, qj, yj, warm, mi=mi, c=c,
                                        dma_prior=dma_prior, lr=lr, b1=b1, b2=b2,
                                        lo=lo, hi=hi)
        return new_c, w
    new_c, w = vmap(one, in_axes=(0, 1, 1, 0))(carry_q, f, q_, y)   # w: (q, M)
    return new_c, w.T                                              # (M, q)


def run_grid(grid, init_states, ys, *, offset: float = 1.0, warmup: int = 0,
             pdr=DMA_PDR, mdr=DMA_MDR, dma_c=DMA_C, **cell_kw):
    """Run the whole grid on one series. Each ``(name, model, components)`` cell runs its
    3 wingmen (``run_cell``, in-class pooling via DLMAX's single-class allocator); the 48
    workers are then combined by DLMAX's HIERARCHICAL allocator over their one-step
    forecasts (families = classes, ``pdm`` within / ``mdm`` between) into a mixture
    predictive (weighted mean + law-of-total-variance).

    ``init_states``: one ``{m,Z,s,nu}`` per cell (see :func:`grid_init`). Returns the
    per-step combined ``loc``/``var``, the hierarchical ``weights (T, M)``, the stacked
    per-worker ``F``/``Q``, the cells, and the class ``names``."""
    cells = [run_cell(m, init, ys, offset=offset, warmup=warmup, pdr=pdr, mdr=mdr,
                      dma_c=dma_c, **cell_kw)
             for (_n, m, _c), init in zip(grid, init_states)]
    F = jnp.concatenate([c["f"] for c in cells], axis=1)         # (T, M)
    Q = jnp.concatenate([c["q"] for c in cells], axis=1)
    Wt = _hier_dma_traj(F, Q, ys, len(grid), pdr, mdr, dma_c)    # (T, M)
    loc = (Wt * F).sum(1)
    var = (Wt * (Q + F**2)).sum(1) - loc**2                      # mixture predictive var
    return {"loc": loc, "var": jnp.maximum(var, 1e-12), "weights": Wt,
            "F": F, "Q": Q, "cells": cells, "names": [n for n, _m, _c in grid]}


def run_grid_rolling(grid, init_states, ys, h, cutoffs, *, offset: float = 1.0,
                     warmup: int = 0, pdr=DMA_PDR, mdr=DMA_MDR, dma_c=DMA_C, **cell_kw):
    """Rolling-origin backtest on ONE series. One online learning pass per cell emits each
    worker's h-step origin forecast at the ``cutoffs`` (so origin ``t`` uses what the grid
    had learned by ``t``); the 48 workers are combined by DLMAX's hierarchical allocator
    (weights AT each cutoff) into a mixture predictive.

    Returns per-cutoff combined ``loc``/``sd`` ``(n_cut, h)`` for scoring + the per-worker
    detail (``loc_w``/``sd_w`` ``(n_cut, M, h)``, ``weights`` ``(n_cut, M)``) and cells for
    the components dump."""
    cutoffs = jnp.asarray(cutoffs)
    cells = [run_cell(m, init, ys, offset=offset, warmup=warmup, pdr=pdr, mdr=mdr,
                      dma_c=dma_c, h=h, cutoffs=cutoffs, **cell_kw)
             for (_n, m, _c), init in zip(grid, init_states)]
    from ..ffs_core import FFSPredictive, _combined_predictive_sd   # lazy: avoid import cycle
    F = jnp.concatenate([c["f"] for c in cells], axis=1)              # (T, M) 1-step, for DMA
    Q = jnp.concatenate([c["q"] for c in cells], axis=1)
    Wt = _hier_dma_traj(F, Q, ys, len(grid), pdr, mdr, dma_c)         # (T, M)
    LOC = jnp.concatenate([c["loc_h"] for c in cells], axis=1)        # (T, M, h)
    QH = jnp.concatenate([c["q_h"] for c in cells], axis=1)           # (T, M, h)
    NU = jnp.concatenate([c["nu_h"] for c in cells], axis=1)          # (T, M)

    Wc, LOCc, QHc, NUc = Wt[cutoffs], LOC[cutoffs], QH[cutoffs], NU[cutoffs]  # origins
    loc = (Wc[:, :, None] * LOCc).sum(1)                             # (n_cut, h)
    # Combine the workers' origin predictives with DLMAX's Vincent (quantile-average) SD —
    # the SAME _combined_predictive_sd AutoFFS/ffs_components use — so the ONLY difference
    # vs AutoFFS is the discount mechanism. Cutoffs are the "series" axis of the predictive.
    pred = FFSPredictive(loc=None, sd=None,
                         f_h=_np.asarray(LOCc.transpose(1, 0, 2)),   # (M, n_cut, h)
                         q_h=_np.asarray(QHc.transpose(1, 0, 2)),
                         nu=_np.asarray(NUc.transpose(1, 0)),        # (M, n_cut)
                         weights=_np.asarray(Wc.transpose(1, 0)))
    sd = _combined_predictive_sd(pred, "quantile").T                 # (n_cut, h)
    return {"loc": _np.asarray(loc), "sd": sd, "weights": _np.asarray(Wc),
            "loc_w": _np.asarray(LOCc), "q_w": _np.asarray(QHc), "nu_w": _np.asarray(NUc),
            "cutoffs": cutoffs, "cells": cells, "names": [n for n, _m, _c in grid]}


#: Non-zero root for inert (padded) states. The inert block is decoupled (G=I,
#: F=0) and its discount is frozen (delta=1), so this value is forecast-neutral and
#: bit-exact — it exists only so the RTRL ``qr`` autodiff sees a non-zero column
#: (a zero column, as multi_model_dlm._init_from_dlms uses, gives a NaN qr gradient).
_INERT_ROOT = 1.0


def _pad_grid_model(model: GridModel, K: int, nb: int) -> GridModel:
    """Pad a family's geometry to the grid-common ``(K, n_blocks)`` with inert
    trailing slots, so every family packs into one vmapped kernel. Mirrors
    ``multi_model_dlm._init_from_dlms``: F=0, G block-diag I, mc=0. The inert
    states are flagged in ``inert_mask`` (discount frozen at delta=1 -> an
    unobserved padded state cannot inflate)."""
    k = model.F.shape[0]
    inert = jnp.arange(K) >= k
    F = jnp.concatenate([model.F, jnp.zeros(K - k)]) if K > k else model.F
    G = jnp.eye(K).at[:k, :k].set(model.G) if K > k else model.G
    mc = jnp.concatenate([model.mc, jnp.zeros(K - k)]) if K > k else model.mc
    stb = (jnp.concatenate([model.state_to_block,
                            jnp.zeros(K - k, model.state_to_block.dtype)])
           if K > k else model.state_to_block)  # inert -> block 0 (frozen anyway)
    dmp = model.damped
    if dmp is not None and K > k:                     # pad damped with 1.0 (inert)
        dmp = jnp.concatenate([dmp, jnp.ones(K - k)])
    sm = model.seasonal_mask                          # (n_blocks,) -> pad to grid nb
    if sm is not None and nb > sm.shape[0]:
        sm = jnp.concatenate([sm, jnp.zeros(nb - sm.shape[0], bool)])
    rm = model.regression_mask                        # (n_blocks,) -> pad to nb
    if rm is not None and nb > rm.shape[0]:
        rm = jnp.concatenate([rm, jnp.zeros(nb - rm.shape[0], bool)])
    # reg_offset survives padding unchanged: inert slots are APPENDED, after the
    # tail, so the tail's position from the head does not move.
    return GridModel(F=F, G=G, mc=mc, var_power=model.var_power,
                     state_to_block=stb, n_blocks=nb, inert_mask=inert,
                     couple_growth=model.couple_growth, damped=dmp,
                     seasonal_mask=sm, n_regs=model.n_regs,
                     reg_offset=model.reg_offset, regression_mask=rm)


def _pad_grid_init(state, K: int):
    """Pad ``{m, Z, s, nu}`` to ``K``: m trailing 0, Z block-diag with a NON-ZERO
    inert root (:data:`_INERT_ROOT`). The inert block is decoupled and F=0, so it
    is forecast-neutral and bit-exact; the non-zero root only keeps the wing's
    ``qr`` autodiff finite (see :func:`_pad_grid_model`)."""
    k = state["m"].shape[0]
    if k == K:
        return state
    m = jnp.concatenate([state["m"], jnp.zeros(K - k)])
    Z = jnp.zeros((K, K)).at[:k, :k].set(state["Z"])
    Z = Z.at[jnp.arange(k, K), jnp.arange(k, K)].set(_INERT_ROOT)
    return {"m": m, "Z": Z, "s": state["s"], "nu": state["nu"]}


def run_grid_rolling_batch(grid, ys, cutoffs, h, *, warmup: int = 0, offset: float = 1.0,
                           pdr=DMA_PDR, mdr=DMA_MDR, dma_c=DMA_C, return_diag: bool = False,
                           return_blocks: bool = False, additive_logscore: bool = False,
                           disc_prior=None, learn_dma: bool = False, dma_prior=DMA_PRIOR,
                           couple_sd=None, seasonal_prior=None,
                           regression_prior=None,
                           adapt_guard=ADAPT_GUARD_DEFAULT,
                           xs=None, xh=None, seed_lags=None,
                           clip=None, disc_init=None):
    """Rolling backtest VMAPPED over a batch of SAME-LENGTH series (JAX-parallel). ``ys``:
    ``(q, T)`` online training data (already trimmed to ``L-h``). Each cell runs
    ``vmap(run_cell)`` over the ``q`` series; the hierarchical allocator is vmapped too;
    the combine uses DLMAX's Vincent SD with ``(series × cutoff)`` as the predictive series
    axis. Returns per-series combined ``loc``/``sd`` ``(q, n_cut, h)`` (numpy).

    Regression tail, series-major like ``ys`` and mapped over the same axis:
    ``xs`` ``(q, T, n_regs)`` filter-time design, ``xh`` ``(q, T, h, n_regs)``
    future EXOGENOUS rows at each origin, ``seed_lags`` ``(q, T, n_regs)``
    AUTOREGRESSIVE forecast seeds. All ``None`` -> the structural path, which is
    bit-exact to before this argument existed.

    ``return_diag`` also returns the learned MODEL at each cutoff for diagnostics:
    ``(weight, level_d, beta)`` each ``(q, n_cut, M)`` — the hierarchical DMA weight, the
    learned level discount, and β per worker (worker order = ``grid`` families × 3 wingmen).
    The combine path is untouched, so the ``loc``/``sd`` returned are identical either way."""
    from ..ffs_core import FFSPredictive, _combined_predictive_sd
    q = ys.shape[0]
    cutoffs = jnp.asarray(cutoffs)
    # Pad every family to a common (K, n_blocks) and run the WHOLE grid as ONE
    # vmapped kernel (families × wingmen × series) — compiled once, no python loop
    # over families (which recompiled per family shape and serialised execution).
    # Padding mirrors multi_model_dlm._init_from_dlms; inert slots are frozen
    # (δ=1) with a non-zero decoupled root, so they are forecast-neutral and the
    # RTRL qr autodiff stays finite. Bit-exact with the per-family loop it replaces.
    K = max(m.F.shape[0] for _n, m, _c in grid)
    nb = max(m.n_blocks for _n, m, _c in grid)
    padded = [_pad_grid_model(m, K, nb) for _n, m, _c in grid]
    Fs = jnp.stack([p.F for p in padded])
    Gs = jnp.stack([p.G for p in padded])
    MCs = jnp.stack([p.mc for p in padded])
    VPs = jnp.asarray([float(p.var_power) for p in padded])
    STBs = jnp.stack([p.state_to_block for p in padded])
    INs = jnp.stack([p.inert_mask for p in padded])
    CGs = jnp.asarray([bool(p.couple_growth) for p in padded])   # per-family coupling
    DMs = jnp.stack([p.damped for p in padded])                  # per-state δ² correction
    SMs = jnp.stack([p.seasonal_mask for p in padded])           # per-block seasonal flag
    RMs = jnp.stack([p.regression_mask for p in padded])         # per-block regression flag
    # Tail geometry is a property of the taxonomy, not of a family: GridBlock.n_regs
    # already refuses a grid whose families disagree, so take it from the first.
    n_regs_ = int(padded[0].n_regs)
    reg_offset_ = int(padded[0].reg_offset)
    # Per-family starting discounts, mapped over the family axis like Fs/Gs.
    # None -> a zero-width stand-in and the uniform 0.95 / beta 0.99 start.
    P_ = int(padded[0].n_params)
    DIs = (jnp.asarray(disc_init, dtype=float) if disc_init is not None
           else jnp.zeros((len(padded), 0)))
    inits_fam = []
    for _n, model, comps in grid:
        inits_j = [_pad_grid_init(grid_init(comps, ys[j], warmup, model.var_power), K)
                   for j in range(q)]
        inits_fam.append(jax.tree_util.tree_map(lambda *a: jnp.stack(a), *inits_j))
    inits_b = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *inits_fam)  # (nf, q, ..)

    # the tail arrays ride the SERIES axis exactly as ys does, and are shared
    # across families exactly as ys is; None -> a zero-width stand-in so the
    # vmap signature stays uniform and the structural path is untouched.
    q_ = ys.shape[0]
    rm_ax = [xs, xh, seed_lags]
    xs_b, xh_b, sl_b = [
        (jnp.asarray(a) if a is not None else jnp.zeros((q_, 0))) for a in rm_ax]
    has = [a is not None for a in rm_ax]

    def _cell(F_, G_, mc_, vp_, stb_, in_, cg_, dm_, sm_, rm_, di_, init_, y_,
              xs_, xh_, sl_):
        model = GridModel(F=F_, G=G_, mc=mc_, var_power=vp_,
                          state_to_block=stb_, n_blocks=nb, inert_mask=in_,
                          couple_growth=cg_, damped=dm_, seasonal_mask=sm_,
                          adapt_guard=adapt_guard, n_regs=n_regs_,
                          reg_offset=reg_offset_, regression_mask=rm_)
        return run_cell(model, init_, y_, warmup=warmup, h=h, cutoffs=cutoffs,
                        offset=offset, pdr=pdr, mdr=mdr, dma_c=dma_c,
                        additive_logscore=additive_logscore, disc_prior=disc_prior,
                        couple_sd=couple_sd, seasonal_prior=seasonal_prior,
                        regression_prior=regression_prior,
                        xs=xs_ if has[0] else None,
                        xh=xh_ if has[1] else None,
                        seed_lags=sl_ if has[2] else None,
                        clip=clip,
                        disc_init=di_ if disc_init is not None else None)
    _series = jax.vmap(_cell, in_axes=(None,) * 11 + (0, 0, 0, 0, 0))
    _fams = jax.vmap(_series, in_axes=(0,) * 12 + (None,) * 4)
    res = _fams(Fs, Gs, MCs, VPs, STBs, INs, CGs, DMs, SMs, RMs, DIs, inits_b, ys,
                xs_b, xh_b, sl_b)                                     # (nf,q,T,3[,h])

    def _fam(a):                              # (nf, q, T, 3[, h]) -> (q, T, nf*3[, h])
        a = jnp.moveaxis(a, 0, 2)             # (q, T, nf, 3[, h])
        return a.reshape(a.shape[:2] + (a.shape[2] * a.shape[3],) + a.shape[4:])
    F = _fam(res["f"]); Q = _fam(res["q"])                       # (q, T, M)
    LOC = _fam(res["loc_h"]); QH = _fam(res["q_h"]); NU = _fam(res["nu_h"])
    if learn_dma:                            # SGDDMA: learn the hierarchical (pdr,mdr)
        Wt, DMAt = jax.vmap(                  #  online per series (Wt (q,T,M), DMAt (q,T,2))
            lambda f, qq, y: _hier_dma_traj_sgd(f, qq, y, len(grid), c=dma_c,
                                                dma_prior=dma_prior, warmup=warmup),
            in_axes=(0, 0, 0))(F, Q, ys)
    else:
        Wt = jax.vmap(lambda f, qq, y: _hier_dma_traj(f, qq, y, len(grid), pdr, mdr, dma_c),
                      in_axes=(0, 0, 0))(F, Q, ys)                   # (q, T, M)
        DMAt = None

    Wc = Wt[:, cutoffs]; LOCc = LOC[:, cutoffs]; QHc = QH[:, cutoffs]; NUc = NU[:, cutoffs]
    loc = (Wc[..., None] * LOCc).sum(2)                             # (q, n_cut, h)
    nq, ncut, M, hh = LOCc.shape                                    # flatten (series×cutoff)
    pred = FFSPredictive(
        loc=None, sd=None,
        f_h=_np.asarray(LOCc.transpose(2, 0, 1, 3).reshape(M, nq * ncut, hh)),
        q_h=_np.asarray(QHc.transpose(2, 0, 1, 3).reshape(M, nq * ncut, hh)),
        nu=_np.asarray(NUc.transpose(2, 0, 1).reshape(M, nq * ncut)),
        weights=_np.asarray(Wc.transpose(2, 0, 1).reshape(M, nq * ncut)))
    sd = _combined_predictive_sd(pred, "quantile").T.reshape(nq, ncut, hh)  # (q, n_cut, h)
    if return_blocks:
        # Raw per-worker material for the orchestrator's union DMA: the
        # one-step trace F/Q (q, T, M) drives the union
        # Allocator; LOCc/QHc/NUc are the per-worker h-step predictives at the
        # cutoffs; the M workers are 3 wingmen per family (worker j -> family
        # j//3). The grid does NOT pre-combine — the orchestrator does.
        names = [f"{n}:{w}" for n, _m, _c in grid for w in ("fast", "cen", "slow")]
        blocks = {
            "F": _np.asarray(F), "Q": _np.asarray(Q),            # (q, T, M) one-step
            "LOCc": _np.asarray(LOCc), "QHc": _np.asarray(QHc),  # (q, n_cut, M, h)
            "NUc": _np.asarray(NUc),                             # (q, n_cut, M)
            "names": names, "n_families": len(grid),
        }
        return _np.asarray(loc), sd, blocks
    if not return_diag:
        return _np.asarray(loc), sd
    LVLc = _fam(res["theta_used"][..., 0])[:, cutoffs]             # (q, n_cut, M)
    BETc = _fam(res["theta_used"][..., -1])[:, cutoffs]
    diag = (_np.asarray(Wc), _np.asarray(LVLc), _np.asarray(BETc))
    if learn_dma:            # append learned (pdr,mdr) at the cutoffs -> 4-tuple
        diag = diag + (_np.asarray(DMAt[:, cutoffs]),)             # (q, n_cut, 2)
    return _np.asarray(loc), sd, diag


# ===================================================================== #
# Streaming (production) driver — a resumable carry for AutoFFSUniverse.
# ---------------------------------------------------------------------
# ``run_grid_rolling_batch`` re-learns the discounts from t=0 every call.
# Streaming instead HOLDS the carry (per-cell wing state + the family-DMA
# state) and advances it one origin at a time. ``grid_stream_scan(carry, ys)``
# produces the identical carry as the batch scan to the same point (validated
# bit-exact vs run_grid_rolling_batch), so ``grid_stream_forecast`` from the
# carry reproduces the batch cv emission. Static config (padded models + wing
# hyperparams) is obs-independent and rebuilt on load; only the carry persists.
# ===================================================================== #
def grid_stream_static(grid, *, offset: float = 1.0, lr_level: float = 0.05,
                       lr_worker: float = 0.05, adam_betas=(0.9, 0.99),
                       pdr=DMA_PDR, mdr=DMA_MDR, dma_c=DMA_C,
                       additive_logscore: bool = False, disc_prior=None,
                       learn_dma: bool = False, dma_prior=DMA_PRIOR, dma_lr=DMA_LR,
                       seasonal_prior=None, adapt_guard=ADAPT_GUARD_DEFAULT,
                       disc_init=None, clip=None, regression_prior=None):
    """Obs-independent streaming config: stacked padded model arrays + wing
    hyperparameters + the family-DMA indicator. Rebuilt (not persisted) on load.

    ``seasonal_prior`` is a scalar (one prior mean for every seasonal discount
    block) or a per-block vector of length ``nb`` — the latter for a cell with
    SEVERAL seasonals whose natural stiffnesses differ by orders of magnitude
    (ENTSOE: annual 8766 / weekly 168 / daily 24). Non-seasonal entries of the
    vector are ignored; ``seasonal_mask`` selects.

    ``clip`` overrides the discount clip box as ``(lo, hi)`` in probability
    space (``None`` -> the module constants). See :func:`_clip_box`. It does NOT
    move the SGDDMA rate box below, which keeps the module constants: bounding
    the STATE discount and bounding a MODEL WEIGHT's forgetting rate are
    unrelated, and on ENTSOE only the former mattered.

    ``disc_init`` is ``None`` (every discount block starts at 0.95 and β at
    0.99, as before) or an ``(nf, P)`` LOGIT-space array of per-family,
    per-block starting values. A block whose natural discount is far from 0.95
    cannot be started there and learned back: an annual Fourier at 0.95 grows
    its own covariance by ``(1/0.95)**8766`` over one cycle."""
    nf = len(grid)
    K = max(m.F.shape[0] for _n, m, _c in grid)
    nb = max(m.n_blocks for _n, m, _c in grid)
    padded = [_pad_grid_model(m, K, nb) for _n, m, _c in grid]
    P = nb + 1
    # Per family: a build_grid taxonomy has families with no seasonal block at
    # all, so a single family's seasonal_mask cannot label the others.
    _boxes = [_clip_box(p_, clip) for p_ in padded]
    lo = jnp.stack([b_[0] for b_ in _boxes])          # (nf, P)
    hi = jnp.stack([b_[1] for b_ in _boxes])
    M = nf * 3
    mi_h = jnp.zeros((M, nf), bool).at[jnp.arange(M), jnp.arange(M) // 3].set(True)
    return {
        "Fs": jnp.stack([p.F for p in padded]), "Gs": jnp.stack([p.G for p in padded]),
        "MCs": jnp.stack([p.mc for p in padded]),
        "VPs": jnp.asarray([float(p.var_power) for p in padded]),
        "STBs": jnp.stack([p.state_to_block for p in padded]),
        "INs": jnp.stack([p.inert_mask for p in padded]),
        "DMs": jnp.stack([p.damped for p in padded]),          # per-state δ² correction
        "SMs": jnp.stack([p.seasonal_mask for p in padded]),   # per-block seasonal flag
        # per-block regression flag + the tail geometry (uniform across families
        # in a block: they share one observation vector layout)
        "RMs": jnp.stack([p.regression_mask for p in padded]),
        "n_regs": int(padded[0].n_regs), "reg_offset": int(padded[0].reg_offset),
        # the tail's MAP prior mean; unset -> anchor it like a seasonal
        "regression_prior": (seasonal_prior if regression_prior is None
                             else regression_prior),
        "seasonal_prior": seasonal_prior, "adapt_guard": adapt_guard,
        "nf": nf, "K": K, "nb": nb, "P": P, "M": M, "lo": lo, "hi": hi,
        "offsets": jnp.array([-offset, 0.0, offset]),
        "use_ls": jnp.zeros(P, bool).at[nb].set(True),
        "use_ls_add": jnp.ones(P, bool),
        "additive_logscore": bool(additive_logscore),
        "disc_prior": disc_prior,
        "mu0d": _logit(0.95), "mu0b": _logit(0.99),
        "disc_init": (None if disc_init is None
                      else jnp.clip(jnp.asarray(disc_init, dtype=float),
                                    lo, hi)),
        "mi3": jnp.ones((3, 1), dtype=bool), "mi_h": mi_h,
        "upd": _dma_update(pdr, mdr, dma_c),
        "lr_level": lr_level, "lr_worker": lr_worker,
        "b1": adam_betas[0], "b2": adam_betas[1],
        # SGDDMA: learn the family (hierarchical) forgetting rates online per series.
        "learn_dma": bool(learn_dma), "dma_prior": dma_prior, "dma_lr": dma_lr,
        "dma_c": dma_c, "pdr": pdr, "mdr": mdr,
        "dma_lo": _logit(CLIP_LO), "dma_hi": _logit(CLIP_HI),
    }


def grid_stream_carry0(grid, static, ys, warmup: int, wing_centre=None,
                       component_priors=None, error_nu0=None):
    """Initial streaming carry ``(cell_carries, family_dma, weights)`` from each
    series' warmup window. ``ys``: ``(T, q)`` time-major (uses ``ys[:warmup]`` per
    series for the diffuse prior, exactly as ``run_cell``).

    Warm-start (``AutoFFSUniverse.add_series`` for a late launcher):
    ``component_priors`` seeds the DLM state from siblings (informative prior in
    :func:`grid_init`); ``wing_centre`` (a logit-space scalar) seeds the wing
    centre ``c`` — the analogous sibling discount — instead of the default
    ``logit(0.95)``; the wings stay at the fixed ``±offset``."""
    S = static
    q = ys.shape[1]
    K, P, nf, M = S["K"], S["P"], S["nf"], S["M"]
    nu0 = 1.0 if error_nu0 is None else float(error_nu0)
    inits_fam = []
    for _n, model, comps in grid:
        ij = [_pad_grid_init(grid_init(comps, _np.asarray(ys)[:, j], warmup,
                                       model.var_power, nu0=nu0,
                                       component_priors=component_priors), K)
              for j in range(q)]
        inits_fam.append(jax.tree_util.tree_map(lambda *a: jnp.stack(a), *ij))
    inits_b = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *inits_fam)  # (nf, q, ..)
    # Per-family starting discounts. ``disc_init`` None -> the uniform
    # logit(0.95) / logit(0.99) the wing has always used (bit-identical);
    # otherwise each family's own (P,) logit row, e.g. from the cells a caller
    # compiled. The wing CENTRE starts at block 0's value, since block 0 is the
    # wing-searched one.
    if S.get("disc_init") is None:
        wth_f = jnp.broadcast_to(
            jnp.full((3, P), S["mu0d"]).at[:, -1].set(S["mu0b"]), (nf, 3, P))
        c0_f = jnp.full((nf,), S["mu0d"])
    else:
        di = jnp.asarray(S["disc_init"])                                # (nf, P)
        wth_f = jnp.broadcast_to(di[:, None, :], (nf, 3, P))
        c0_f = di[:, 0]
    if wing_centre is not None:
        c0_f = jnp.full((nf,), float(wing_centre))

    def cell0(init_one, wth0, c0):
        st0 = {k: jnp.broadcast_to(v, (3,) + v.shape) for k, v in init_one.items()}
        S0 = {k: jnp.zeros((3, P) + v.shape) for k, v in init_one.items()}
        alloc0 = init_alloc_state(3, 1, 1, S["mi3"], 1)
        return (st0, S0, c0, jnp.zeros(3), jnp.zeros(3), wth0,
                jnp.zeros((3, P)), jnp.zeros((3, P)), alloc0, 0.0)
    cc0 = vmap(vmap(cell0, in_axes=(0, None, None)),
               in_axes=(0, 0, 0))(inits_b, wth_f, c0_f)                # (nf, q, ..)
    if S.get("learn_dma"):     # per-series SGDDMA carry (learns family pdr/mdr)
        hier0 = vmap(lambda _i: _sgd_dma_carry0(M, S["mi_h"], S["pdr"], S["mdr"])
                     )(jnp.arange(q))
    else:
        hier0 = vmap(lambda _i: init_alloc_state(M, 1, nf, S["mi_h"], 1))(jnp.arange(q))
    Wc0 = jnp.full((q, M), 1.0 / M)
    return (cc0, hier0, Wc0)


def _grid_stream_step(static, carry, yt, warm, xt=None):
    """One streaming origin: advance every cell (``_wing_step``) then the family
    DMA. ``yt``: ``(q,)``; ``warm``: scalar 0/1. Returns the new carry."""
    S = static
    cc, hier, _Wc = carry

    def per_cell(F_, G_, mc_, vp_, stb_, in_, dm_, sm_, rm_, lo_, hi_, cc_one,
                 y_s, x_s):
        model = GridModel(F=F_, G=G_, mc=mc_, var_power=vp_, state_to_block=stb_,
                          n_blocks=S["nb"], inert_mask=in_, damped=dm_, seasonal_mask=sm_,
                          adapt_guard=S["adapt_guard"], n_regs=S["n_regs"],
                          reg_offset=S["reg_offset"], regression_mask=rm_)
        uls = (jnp.where(vp_ == 1.0, S["use_ls_add"], S["use_ls"])
               if S["additive_logscore"] else S["use_ls"])
        return _wing_step(cc_one, y_s, warm, x_s, model=model, offsets=S["offsets"],
                          lo=lo_, hi=hi_, use_ls=uls, mi=S["mi3"],
                          upd=S["upd"], lr_level=S["lr_level"], lr_worker=S["lr_worker"],
                          b1=S["b1"], b2=S["b2"], disc_prior=S["disc_prior"],
                          seasonal_prior=S["seasonal_prior"],
                          regression_prior=S["regression_prior"])
    # xt: (q, n_regs) this step's regressor rows, or None for a structural grid.
    # Mapped over series like yt; SHARED across families, since every family in a
    # block sees the same exogenous row / the same lags.
    xt_ax = None if xt is None else 0
    pc_s = vmap(per_cell, in_axes=(None,) * 11 + (0, 0, xt_ax))   # over series
    pc_f = vmap(pc_s, in_axes=(0,) * 12 + (None, None))           # over families
    new_cc, out = pc_f(S["Fs"], S["Gs"], S["MCs"], S["VPs"], S["STBs"], S["INs"],
                       S["DMs"], S["SMs"], S["RMs"], S["lo"], S["hi"], cc, yt, xt)
    q = yt.shape[0]
    F = jnp.moveaxis(out["f"], 0, 1).reshape(q, S["M"])              # (q, M) family-major
    Q = jnp.moveaxis(out["q"], 0, 1).reshape(q, S["M"])

    if S.get("learn_dma"):     # SGDDMA: learn the family pdr/mdr online per series
        def hier_one(dma_carry, f_row, q_row, y_s):
            new_c, w, _disc = _sgd_dma_step(
                dma_carry, f_row, q_row, y_s, warm, mi=S["mi_h"], c=S["dma_c"],
                dma_prior=S["dma_prior"], lr=S["dma_lr"], b1=S["b1"], b2=S["b2"],
                lo=S["dma_lo"], hi=S["dma_hi"])
            return new_c, w
    else:
        def hier_one(alloc, f_row, q_row, y_s):
            fc = ForecastBundle(f_row[:, None, None], q_row[:, None, None])
            alloc_n, w = allocator_step(alloc, fc, y_s[None], LogScore,
                                        IdentityAggregator, S["upd"], S["mi_h"])
            return alloc_n, w[:, 0, 0]
    new_h, Wc = vmap(hier_one)(hier, F, Q, yt)                        # (q, M)
    return (new_cc, new_h, Wc), F, Q          # F, Q: (q, M) per-worker one-step


def grid_stream_scan(static, carry, ys, warm, xs=None, return_trace=False):
    """Advance the carry over ``ys`` ``(T, q)`` with per-step ``warm`` ``(T,)``.
    Equivalent to ``T`` calls of :func:`_grid_stream_step` (``lax.scan``).

    ``return_trace`` additionally returns ``(F, Q)`` ``(T, q, M)`` — the
    per-worker one-step mean/variance at each origin (already computed inside the
    step). This is the trace the union DMA is driven over to build its carry
    across the window; it matches ``cv_trajectory``'s ``f1_full``/``q1_full``."""
    # Two scan bodies rather than a dummy leaf: None cannot be a scan input, and
    # a zero-width stand-in would not be None inside the step, so the structural
    # path would take the regressor branch and vmap a mismatched axis. Keeping
    # the no-regressor scan on a separate branch is what keeps the structural
    # results bit-exact.
    if xs is None:
        def step(c, packed):
            y_t, w_t = packed
            c_new, F, Q = _grid_stream_step(static, c, y_t, w_t, None)
            return c_new, ((F, Q) if return_trace else None)
        carry, tr = lax.scan(step, carry, (ys, warm))
    else:
        def step(c, packed):
            y_t, w_t, x_t = packed
            c_new, F, Q = _grid_stream_step(static, c, y_t, w_t, x_t)
            return c_new, ((F, Q) if return_trace else None)
        carry, tr = lax.scan(step, carry, (ys, warm, jnp.asarray(xs)))
    return (carry, tr) if return_trace else carry


def grid_stream_step(static, carry, yt, warm, xt=None, return_trace=False):
    """Public single-origin advance (``fwd_filter``). ``warm`` scalar 0/1.

    ``return_trace`` also returns the per-worker one-step ``(F, Q)`` ``(q, M)``
    used for ``yt`` — the SAME quantity ``grid_stream_scan``'s trace emits, so the
    union DMA is driven identically in the update and fit paths."""
    carry, F, Q = _grid_stream_step(static, carry, yt, jnp.asarray(warm, float), xt)
    return (carry, F, Q) if return_trace else carry


def grid_stream_forecast(static, carry, h, xh=None, seed_lags=None):
    """h-step forecast from a streaming carry (no re-filtering). Returns
    ``(loc, sd)`` ``(q, h)`` plus the per-worker predictives for the union DMA."""
    from ..ffs_core import FFSPredictive, _combined_predictive_sd
    S = static
    cc, _hier, Wc = carry
    q = Wc.shape[0]

    def fc_cell(F_, G_, mc_, vp_, stb_, in_, dm_, lo_, hi_, cc_one, xh_one,
                sl_one):
        model = GridModel(F=F_, G=G_, mc=mc_, var_power=vp_, state_to_block=stb_,
                          n_blocks=S["nb"], inert_mask=in_, damped=dm_,
                          adapt_guard=S["adapt_guard"], n_regs=S["n_regs"],
                          reg_offset=S["reg_offset"])
        st, _Sn, c, lm, lv, wth, wm, wv, alloc, t = cc_one
        th_next = jax.nn.sigmoid(wth.at[:, 0].set(jnp.clip(c + S["offsets"],
                                                           lo_[0], hi_[0])))
        return vmap(lambda s_i, t_i: forecast_origin(
            model, s_i, t_i, h, xh_one, sl_one))(st, th_next)
    # xh: (q, h, n_regs) future EXOGENOUS rows; seed_lags: (q, n_regs) the known
    # observations seeding an AUTOREGRESSIVE tail. Both per series, None -> the
    # static-F path.
    xh_ax = None if xh is None else 0
    sl_ax = None if seed_lags is None else 0
    fc_s = vmap(fc_cell, in_axes=(None,) * 9 + (0, xh_ax, sl_ax))  # over series
    fc_f = vmap(fc_s, in_axes=(0,) * 10 + (None, None))            # over families
    LOC, QH, NU = fc_f(S["Fs"], S["Gs"], S["MCs"], S["VPs"], S["STBs"], S["INs"],
                       S["DMs"], S["lo"], S["hi"], cc, xh, seed_lags)
    LOCc = jnp.moveaxis(LOC, 0, 1).reshape(q, S["M"], h)
    QHc = jnp.moveaxis(QH, 0, 1).reshape(q, S["M"], h)
    NUc = jnp.moveaxis(NU, 0, 1).reshape(q, S["M"])
    loc = (Wc[:, :, None] * LOCc).sum(1)
    pred = FFSPredictive(loc=None, sd=None,
                         f_h=_np.asarray(LOCc.transpose(1, 0, 2)),
                         q_h=_np.asarray(QHc.transpose(1, 0, 2)),
                         nu=_np.asarray(NUc.transpose(1, 0)),
                         weights=_np.asarray(Wc.transpose(1, 0)))
    sd = _combined_predictive_sd(pred, "quantile").T
    return _np.asarray(loc), sd, {"LOCc": _np.asarray(LOCc), "QHc": _np.asarray(QHc),
                                  "NUc": _np.asarray(NUc), "Wc": _np.asarray(Wc)}


def run_grid_batch(arr, srs_ids, cutoffs, h, period, warmup, offset=1.0, return_diag=False):
    """Dask worker entry (the ``run_svi_batch`` analogue) — vmapped rolling backtest for a
    batch of SAME-LENGTH series. ``arr``: ``(L, q)`` column-stacked (as
    ``AutoFFS._iter_batches`` yields). Trains the online pass on rows ``[0, L-h)`` and
    emits h-step forecasts at ``cutoffs``. Returns ``(srs_ids, loc, sd, obs)`` — ``loc``/
    ``sd``/``obs`` shaped ``(q, n_cut, h)``, all numpy (pickles cleanly from the worker).

    ``return_diag`` appends ``(names, weight, level_d, beta)`` — ``names`` the M worker
    tags (grid families × 3 wingmen), and ``weight``/``level_d``/``beta`` each
    ``(q, n_cut, M)`` — the learned discount MODEL at each origin, for diagnostics."""
    from . import discount_grid as _dg           # ensure the module is importable on workers
    a = _np.asarray(arr, float)
    L = a.shape[0]
    ys = jnp.asarray(a[:L - h].T)                 # (q, L-h)
    grid = _dg.build_grid(period=period)
    out = _dg.run_grid_rolling_batch(grid, ys, _np.asarray(cutoffs), h,
                                     warmup=warmup, offset=offset, return_diag=return_diag)
    cuts = _np.asarray(cutoffs)
    obs = _np.stack([_np.stack([a[t + 1:t + 1 + h, j] for t in cuts])
                     for j in range(a.shape[1])])   # (q, n_cut, h)
    if not return_diag:
        loc, sd = out
        return list(srs_ids), _np.asarray(loc), _np.asarray(sd), obs
    loc, sd, (W, LVL, BET) = out
    names = [f"{n}:{w}" for n, _m, _c in grid for w in ("fast", "cen", "slow")]
    return list(srs_ids), _np.asarray(loc), _np.asarray(sd), obs, names, W, LVL, BET

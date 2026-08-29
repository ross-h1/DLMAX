"""Core DLM implementation: univariate state-space filtering and forecasting
using an SVD square-root form for numerical stability, plus a multi-model
wrapper that packs many DLM specifications into a single batched computation.

The square-root form requires 64-bit floats, so
``config.update("jax_enable_x64", True)`` is set at import time. That is a
module-level side effect: importing this module changes JAX's global
precision for the process.
"""

import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax import config, device_put, grad, jit, value_and_grad, vmap
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.lax import scan
from jax.numpy.linalg import svd
from jax.scipy.linalg import block_diag
import jax.scipy.stats as stats
from typing import NamedTuple
from jax.tree_util import Partial

from DLMAX.smoother import SmootherError


config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# x64 enforcement helpers
# -----------------------------------------------------------------------------


def enable_x64() -> None:
    """Enable JAX 64-bit float mode.

    Convenience wrapper around ``jax.config.update("jax_enable_x64",
    True)``. DLMAX requires 64-bit floats for the SVD square-root form
    of the DLM filter to remain numerically stable, particularly for
    long series (e.g. multi-year daily data, where 32-bit
    factorisations diverge after a few hundred steps).

    Called once on import of ``dlm_core``, but exposed as a function so
    callers can re-enable it explicitly if they have toggled it off
    elsewhere in their session.
    """
    config.update("jax_enable_x64", True)


def require_x64() -> None:
    """Raise ``RuntimeError`` if JAX 64-bit float mode is not enabled.

    Called from DLM constructors as belt-and-braces: ``dlm_core``'s
    import already enables x64, but downstream code occasionally
    toggles it off and the resulting numerical issues are subtle and
    expensive to debug.
    """
    try:
        enabled = config.read("jax_enable_x64")
    except AttributeError:
        # Fallback for older JAX versions where `read` isn't available.
        enabled = config.values.get("jax_enable_x64", False)
    if not enabled:
        raise RuntimeError(
            "DLMAX requires JAX 64-bit floats (jax_enable_x64=True). "
            "Call `DLMAX.dlm_core.enable_x64()` or set the environment "
            "variable JAX_ENABLE_X64=1 before constructing models."
        )


# -----------------------------------------------------------------------------
# vmap utilities
# -----------------------------------------------------------------------------

vdiag = vmap(jnp.diag)
vvdiag = vmap(vdiag)
axis0dot = vmap(jnp.dot, in_axes=[0, 0])
axis1dot = vmap(jnp.dot, in_axes=[1, 1])
axis01dot = vmap(axis0dot, in_axes=[0, 0])


# -----------------------------------------------------------------------------
# uv_dlm
# -----------------------------------------------------------------------------


class uv_dlm(object):
    """Univariate DLM applied in parallel across ``q`` series.

    For a state vector of length ``p``:

    Parameters
    ----------
    series_ids : array-like
        Identifiers for each series (length ``q``).
    F : jnp.ndarray
        Observation vector, shape ``(p,)`` (shared across series).
    G : jnp.ndarray
        State transition matrix, shape ``(p, p)`` (shared across series).
    n_regressors : int
        Number of additional dynamic regression components beyond ``F``.
    m0 : jnp.ndarray
        Initial state mean, shape ``(q, p)``.
    C0 : jnp.ndarray
        Initial state covariance, shape ``(q, p, p)``.
    V0 : jnp.ndarray
        Initial observation-variance estimates, shape ``(q,)``.
    nu0 : jnp.ndarray
        Initial degrees of freedom for the variance prior, shape ``(q,)``.
    disc_rates_norm : jnp.ndarray
        Per-state discount factors, shape ``(p,)``.
    disc_rates_damped : jnp.ndarray, optional
        Per-state damping factors, shape ``(p,)``. Defaults to ones.
    monitor_inject : jnp.ndarray, optional
        Per-state revised discount fed into the monitor when it fires
        (e.g. 0.5 level/trend, 0.8 seasonal). ``1.0`` everywhere (the default)
        disables injection, reproducing the static-grid behaviour exactly.
    variance_disc : float
        Variance discount factor.
    variance_power : float
        Power relating state to observation variance (0 = log, 0.5 = sqrt,
        1 = identity).
    mult_comps : jnp.ndarray, optional
        Binary mask of multiplicative state components, shape ``(p,)``.
    regressors_0 : jnp.ndarray, optional
        Regressor values at t=0, used to initialise the variance scale.
    regression_G : jnp.ndarray, optional
        Transition block for the regression coefficients, shape
        ``(n_regressors, n_regressors)``. Defaults to the identity
        (random-walk coefficients).
    device : jax.sharding.NamedSharding
        Sharding on which to place state. Must be provided.
    h : int, optional
        Forecast horizon. If given, ``G``-powers are pre-computed.
    monitor : bool
        Whether to run model-monitoring in downstream code.
    adapt : float or None, default 0.5
        Sensitivity of the state-adaptive discount. The applied discount is

            disc = (disc_rates + (1 - disc_rates) * exp(-adapt * info)) * disc_rates_damped

        with ``info = |m| / sqrt(diag(C))``, so delta falls toward the floor
        ``disc_rates * disc_rates_damped`` as a state becomes well determined and
        sits at ``disc_rates_damped`` (least forgetting) while it is uncertain.

        ``adapt=None`` selects a **static** discount,
        ``disc = disc_rates * disc_rates_damped`` — byte-identical to the grid
        that :meth:`multi_model_dlm._forecast_disc_factor_vec` already uses on the
        forecast path, so filter and forecast then agree by construction. Note
        this is *not* reachable by setting ``disc_rates=1``: ``disc_rates_damped``
        is owned by the components (``LocalTrend.disc_damped_block`` returns
        ``[1, damping**2]``, the phi-squared correction that stops G's damping
        fighting the discount), so overloading it would corrupt damped-trend
        models.

        A static discount makes the model fixed linear-Gaussian, which is the
        precondition for exact retrospective smoothing and FFBS — see
        :mod:`DLMAX.smoother`. The default preserves the previous behaviour
        exactly.
    """

    def __init__(
            self,
            series_ids,
            F,
            G,
            n_regressors,
            m0,
            C0,
            V0,
            nu0,
            disc_rates_norm,
            disc_rates_damped=None,
            monitor_inject=None,   # per-state revised discount when the monitor
            #                        fires (0.5 lt / 0.8 seas; 1.0 = no injection).
            variance_disc=1,
            variance_power=1,
            mult_comps=None,
            regressors_0=None,
            regression_G=None,
            device=None,
            h=None,
            monitor=False,   # error-monitoring tau slot; False=no monitoring.
            #                  (inert in the filter; read by the adaptive
            #                  discount path. Default False so monitor>0 cleanly
            #                  means "adaptive" — see multi.forecast / _adapt_tau.)
            adapt=0.5,       # state-adaptive discount sensitivity; None = static
            #                  discount (disc_rates * disc_rates_damped).
    ):
        require_x64()
        if device is None:
            raise ValueError(
                "`device` must be a jax.sharding.NamedSharding; see "
                "DLMAX.ffs.devices.configure_devices for the usual construction."
            )
        self.device = device
        self.monitor = monitor
        self.n_devices = len(self.device.device_set)
        self.series_ids = series_ids
        self.n_series = V0.shape[0]

        leftovers = self.n_series % self.n_devices
        self.n_pad = (self.n_devices - leftovers) if leftovers else 0

        self.F = F
        self.G = G
        self.n_regressors = n_regressors
        # Regression transition matrix. Stored on the model so
        # ``_broadcast_G`` can use it as the default ``Gt_extra``,
        # baking coefficient damping into the model rather than
        # requiring callers to thread ``Gt`` through every filter call.
        if n_regressors > 0:
            if regression_G is None:
                regression_G = jnp.eye(n_regressors)
            else:
                regression_G = jnp.asarray(regression_G)
                if regression_G.shape != (n_regressors, n_regressors):
                    raise ValueError(
                        f"regression_G has shape {regression_G.shape}, "
                        f"expected ({n_regressors}, {n_regressors})."
                    )
        self.regression_G = regression_G
        # Pre-build extended shardings for 2D / 3D arrays. Trailing
        # ``None``s leave the added axes replicated across devices,
        # preserving the original ``device.reshape(n_devices, 1, ...)``
        # intent (shard the leading n_series axis, leave the rest alone).
        # ``NamedSharding`` does not expose ``.reshape``, so the sharding
        # must be rebuilt with an explicit spec rather than reshaped.
        _spec = tuple(self.device.spec)
        self._device_2d = NamedSharding(self.device.mesh, P(*_spec, None))
        self._device_3d = NamedSharding(self.device.mesh, P(*_spec, None, None))
        self.m0 = device_put(pad(m0, self.n_pad), self.device)
        self.C0 = device_put(pad(C0, self.n_pad), self.device)
        self.V0 = device_put(pad(V0, self.n_pad), self.device)
        self.nu = device_put(pad(nu0, self.n_pad), self.device)

        UC, SC, VCT = svd_sqrt(self.C0)

        self.mult_comps = jnp.zeros_like(F) if mult_comps is None else mult_comps
        self.base_comps = 1 - self.mult_comps

        self.disc_rates = disc_rates_norm
        self.disc_rates_damped = (
            jnp.ones_like(self.disc_rates)
            if disc_rates_damped is None
            else disc_rates_damped
        )
        # Per-state monitor injection discount (the revised grid rate fed into the
        # guard on a trigger). 1.0 everywhere => no injection (legacy behaviour).
        self.monitor_inject = (
            jnp.ones_like(self.disc_rates)
            if monitor_inject is None
            else monitor_inject
        )

        self.variance_disc = variance_disc
        self.variance_power = variance_power
        # None => static discount. See the `adapt` docstring entry.
        self.adapt = adapt

        if regressors_0 is None:
            regressors_0 = jnp.zeros(0)
        var_scale = var_scale_fn(
            self.m0 @ jnp.concatenate([self.F, regressors_0]), self.variance_power
        )

        self.model = None
        self.h = h
        if self.h is not None:
            # Precompute G, G^2, ..., G^h as a one-shot Python loop: it runs
            # once at construction, not per step, so h is small enough that a
            # scan would not pay.
            GH = [self.G]
            for _ in range(h - 1):
                GH.append(self.G @ GH[-1])
            self.GH = jnp.stack(GH)[jnp.newaxis, ...]

        self.dlm_state = {
            "m": self.m0,
            # QR-native covariance root: C = Z' Z. Seeded from the one-shot
            # svd_sqrt(C0) above (Z = diag(SC) @ UC'), so the QR and SVD forms
            # start from numerically identical state. The SVD roots {UC, SC} are
            # recoverable on demand via the `dlm_svd_state` property.
            "Z": axis0dot(vdiag(SC), UC.swapaxes(-1, -2)),
            "s": self.V0 / var_scale,
            "nu": self.nu,
            # EWMA of the SIGNED standardised one-step error per series, for the
            # error-monitor intervention (signed-bias detector, West & Harrison
            # automatic-monitoring analogue). Init at 0 (no bias => calm, |S| <
            # threshold). Carried/stacked like the rest of the state; only drives
            # behaviour when `monitor` (tau = SD multiple) > 0, so the static path
            # is numerically untouched.
            "S": jnp.zeros_like(self.V0 / var_scale),
        }
        # RTRL discount-learning carry (set by enable_adapt for Adapt models);
        # None => the plain QR filter path in fwd_filter.
        self._adapt = None
        self._wing = None      # 3-wingman cell carry (enable_wing, Wing models)
        # Per-step filtered trajectory, accumulated only when fwd_filter is
        # called with trajectory=True. See `trajectory` / `clear_trajectory`.
        self._traj = None

    # ------------------------------------------------------------------
    # discount
    # ------------------------------------------------------------------

    def _disc_mtx(self):
        """Per-series discount factor matrices ``(1-delta)/delta``, shape (q, k, k).

        ``adapt=None`` gives the static grid ``disc_rates * disc_rates_damped``,
        identical for every series and independent of the data — the fixed
        linear-Gaussian case. Otherwise the state-adaptive envelope is applied
        per series.
        """
        if self.adapt is None:
            disc = self.disc_rates * self.disc_rates_damped
            n = self.n_series + self.n_pad
            return jnp.broadcast_to(
                jnp.diag((1.0 - disc) / disc), (n, disc.shape[-1], disc.shape[-1])
            )
        vadapt = vmap(_adapt_discount_z, in_axes=(0, 0, None, None, None))
        return vadapt(
            self.dlm_state["Z"],
            self.dlm_state["m"],
            self.disc_rates,
            self.disc_rates_damped,
            self.adapt,
        )

    # ------------------------------------------------------------------
    # filtered trajectory
    # ------------------------------------------------------------------

    @property
    def trajectory(self):
        """Stacked filtered trajectory, or ``None`` if none was accumulated.

        Populated by calling :meth:`fwd_filter` with ``trajectory=True``. Keys
        ``m`` (T, q, p), ``Z`` (T, q, p, p), ``s`` / ``nu`` (T, q, 1), sliced to
        the unpadded ``n_series``. ``a_t = G m_{t-1}`` and the prior root
        ``NR_t = Z_{t-1} G' diag(1/sqrt(delta))`` are recomputable and so are not
        stored; ``Z`` is ~99% of the volume, ``T*q*p*p*8`` bytes.
        """
        if self._traj is None:
            return None
        return {k: jnp.stack(v) for k, v in self._traj.items()}

    def clear_trajectory(self):
        """Discard any accumulated trajectory."""
        self._traj = None

    def _require_trajectory(self, caller):
        traj = self.trajectory
        if traj is None:
            raise SmootherError(
                f"{caller} needs a filtered trajectory, but none was accumulated. "
                "Re-run the filter with fwd_filter(..., trajectory=True). "
                "(Not done implicitly: it would silently repeat the whole filter.)"
            )
        return traj

    @property
    def applied_disc(self):
        """The discount actually applied per state, ``disc_rates * disc_rates_damped``.

        Exact only for ``adapt=None``; under the adaptive envelope this is the
        floor rather than the realised value.
        """
        return self.disc_rates * self.disc_rates_damped

    # ------------------------------------------------------------------
    # retrospective smoothing / sampling  (see DLMAX.smoother)
    # ------------------------------------------------------------------

    def smooth(self):
        """Retrospective (RTS) smoothing over the accumulated trajectory.

        Requires a prior ``fwd_filter(..., trajectory=True)`` run. Returns the
        :func:`DLMAX.smoother.rts_smooth` dict, plus ``exact`` (bool) and
        ``approximations`` (list of str) recording whether the model features in
        play make the result exact or exact-only-conditionally. See the
        :mod:`DLMAX.smoother` docstring for the classification.
        """
        from DLMAX.smoother import rts_smooth, _validate

        traj = self._require_trajectory("smooth()")
        approximate, reasons = _validate(self, sampling=False,
                                         allow_approximate=True)
        out = rts_smooth(traj, self.G, self.applied_disc)
        out["exact"] = not approximate
        out["approximations"] = reasons
        return out

    def backward_sample(self, key, n_draws=1, right_factor=None,
                        allow_approximate=False):
        """FFBS: draw whole state trajectories from the joint posterior.

        Requires a prior ``fwd_filter(..., trajectory=True)`` run. Raises
        :class:`DLMAX.smoother.SmootherError` when the model is one for which the
        draw would not target the posterior (multiplicative components, adaptive
        discount, RTRL / wing learning) unless ``allow_approximate=True``.

        Returns an array of shape ``(n_draws, T, n_series, p)``.
        """
        from DLMAX.smoother import ffbs, _validate

        traj = self._require_trajectory("backward_sample()")
        _validate(self, sampling=True, allow_approximate=allow_approximate)
        return ffbs(key, traj, self.G, self.applied_disc,
                    n_draws=n_draws, right_factor=right_factor)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _broadcast_G(self, Gt_extra=None):
        """Build a per-series batched G tensor, optionally block-diag with
        regression block ``Gt_extra`` (defaults to ``self.regression_G``,
        which is ``eye(n_regressors)`` unless overridden at construction).
        """
        if Gt_extra is None:
            Gt_extra = self.regression_G
        base = block_diag(self.G, Gt_extra)[jnp.newaxis, :]
        return device_put(
            pad(base * jnp.ones((self.n_series, 1, 1)), self.n_pad),
            self._device_3d,
        )

    # ------------------------------------------------------------------
    # filtering
    # ------------------------------------------------------------------

    def fwd_filter(self, yt, regressors=None, Gt=None, intervention=None,
                   trajectory=False):
        """Advance the filter by one observation for every series.

        Performs a single forward-filtering SVD update step (prior → posterior)
        across all ``q`` series in parallel, mutating ``self.dlm_state`` in
        place and storing the one-step-ahead predictive moments on
        ``self.model`` (read them back with :meth:`params`).

        Parameters
        ----------
        yt : array, shape ``(n_series,)``
            The observation for each series at this step.
        regressors : array, optional, shape ``(n_series, n_regressors)``
            Dynamic-regression values at this step, appended to the tail of
            ``F``. ``None`` (default) runs the purely structural update.
        Gt : array, optional
            Override for the regression-coefficient transition block (defaults
            to ``self.regression_G``). Only used when ``regressors`` is given.
        intervention : optional
            Reserved; not currently applied.
        trajectory : bool, default False
            Accumulate this step's filtered state onto :attr:`trajectory`, for
            later retrospective smoothing / FFBS (:mod:`DLMAX.smoother`). Off by
            default: zero cost and no behaviour change when unused. Memory grows
            as ``T*q*p*p*8`` bytes, dominated by the covariance root ``Z``.
            Only ``{m, Z, s, nu}`` are kept — ``a_t`` and the prior root ``NR_t``
            are recomputable from them. Use :meth:`clear_trajectory` to reset.

        Returns
        -------
        ForecastBundle
            One-step-ahead predictive ``loc`` and ``var`` (variance), each
            shaped ``(n_series,)``. The same moments are also stored on
            ``self.model`` (see :meth:`params`).
        """
        if self._adapt is not None:
            return self._adapt_fwd_filter(yt)
        if self._wing is not None:
            return self._wing_fwd_filter(yt)

        if regressors is not None:
            Ft = device_put(
                pad(
                    jnp.concatenate(
                        [
                            self.F[jnp.newaxis, :] * jnp.ones((self.n_series, 1)),
                            regressors,
                        ],
                        axis=1,
                    ),
                    self.n_pad,
                ),
                self._device_2d,
            )
            Gt_batched = self._broadcast_G(Gt)
        else:
            Ft = jnp.ones((self.n_series + self.n_pad, 1)) * self.F[jnp.newaxis, :]
            Gt_batched = (
                jnp.ones((self.n_series + self.n_pad, 1, 1)) * self.G[jnp.newaxis, :, :]
            )

        #data_t = {
        #    "F": Ft,
        #    "G": Gt_batched,
        #    "y": device_put(jnp.atleast_2d(pad(yt, self.n_pad)), self.device),
        #}

        data_t = {
            "F": Ft,
            "G": Gt_batched,
            "y": device_put(jnp.atleast_2d(pad(yt, self.n_pad)).T, self.device),
        }

        disc_mtx = self._disc_mtx()

        # Broadcast variance_disc / variance_power / mult_comps to (n_series, ...)
        # for the vmap. They're stored as scalars (or 1D for mult_comps) on a
        # single uv_dlm; vdlm_uv_fwd_svd_step requires axis 0 to be n_series.
        n = self.n_series + self.n_pad
        var_disc_b = jnp.broadcast_to(
            jnp.atleast_1d(self.variance_disc).reshape(-1)[:1], (n, 1)
        )
        var_power_b = jnp.broadcast_to(
            jnp.atleast_1d(self.variance_power).reshape(-1)[:1], (n, 1)
        )
        mult_b = jnp.broadcast_to(
            jnp.atleast_2d(self.mult_comps), (n, self.mult_comps.shape[-1])
        )

        self.dlm_state, self.model = vdlm_uv_fwd_qr_step(
            disc_mtx,
            var_disc_b,
            var_power_b,
            mult_b,
            self.dlm_state,
            data_t,
        )

        # self.dlm_state, self.model = vdlm_uv_fwd_svd_step(
        #    disc_mtx,
        #    jnp.atleast_2d(self.variance_disc),
        #    jnp.atleast_2d(self.variance_power),
        #    jnp.atleast_2d(self.mult_comps),
        #    self.dlm_state,
        #    data_t,
        # )

        # For a local-level (p=1) model, vmap'd outputs drop the trailing dim.
        # Restore it so downstream code sees a consistent (q, p) shape.
        if self.dlm_state["m"].ndim == 1:
            self.dlm_state["m"] = self.dlm_state["m"][:, jnp.newaxis]

        if trajectory:
            if self._traj is None:
                self._traj = {k: [] for k in ("m", "Z", "s", "nu")}
            for k in self._traj:
                v = self.dlm_state[k][: self.n_series]
                self._traj[k].append(jnp.atleast_2d(v) if v.ndim == 1 else v)

        return ForecastBundle(
            self.model["f"][: self.n_series].reshape(self.n_series),
            self.model["q"][: self.n_series].reshape(self.n_series),
        )

    def params(self):
        """Return the stored one-step-ahead predictive moments.

        Returns
        -------
        dict
            The contents of ``self.model`` from the most recent
            :meth:`fwd_filter` call (predictive mean ``f``, variance ``q`` and
            related quantities), sliced back to the unpadded ``n_series`` rows.
        """
        return {k: v[: self.n_series] for k, v in self.model.items()}

    @property
    def discounts(self):
        """Current per-state discount rates, shape ``(n_series, k)``.

        Fixed / ``[list]`` model: the static ``disc_rates`` broadcast over
        series. ``Adapt`` model: the ONLINE-LEARNED discount as of the last
        :meth:`fwd_filter` — the block θ mapped to states via the geometry.
        """
        if self._wing is not None:
            # DMA-weighted effective per-state discount over the wingmen, per series.
            w = self._wing
            gm = w["gm"]
            _, _, c, _, _, wth = w["carry"][:6]              # c (q,), wth (q, wings, P)
            offsets, lo, hi = w["offsets"], w["lo"], w["hi"]

            def per_series(c_i, wth_i, w_i):
                th = 1.0 / (1.0 + jnp.exp(-wth_i.at[:, 0].set(
                    jnp.clip(c_i + offsets, lo[0], hi[0]))))  # (wings, P)
                delta = th[:, gm.state_to_block]              # (wings, k)
                return jnp.sum(w_i[:, None] * delta, axis=0)  # (k,)

            return vmap(per_series)(c, wth, w["w"])           # (q, k)
        if self._adapt is None:
            d = jnp.atleast_1d(self.disc_rates)
            return jnp.broadcast_to(d, (self.n_series, d.shape[-1]))
        a = self._adapt
        theta = 1.0 / (1.0 + jnp.exp(-a["wth"]))          # (q, P), probability space
        return theta[:, a["gm"].state_to_block]            # (q, k) per-state δ

    @property
    def discount_grad(self):
        """Last-step RTRL log-score gradient ``∂ℓ/∂θ`` per ``(series, param)``,
        shape ``(n_series, P)`` — or ``None`` for a non-``Adapt`` model. The
        signal driving the online discount; useful for diagnostics/tutorials."""
        return None if self._adapt is None else self._adapt.get("g_ls")

    @property
    def wing_grid(self):
        """For a ``Wing`` model: the moving grid as of the last :meth:`fwd_filter`.

        Returns ``{"discounts": (n_series, wings), "weights": (n_series, wings)}``
        — each wingman's centre-block (level) discount and its DMA weight — so the
        grid can be inspected/plotted without reaching into internals. ``None`` for
        a non-``Wing`` model.
        """
        if self._wing is None:
            return None
        w = self._wing
        c = w["carry"][2]                                    # (q,) centre (logit)
        lvl = jnp.clip(c[:, None] + w["offsets"][None, :], w["lo"][0], w["hi"][0])
        return {"discounts": 1.0 / (1.0 + jnp.exp(-lvl)), "weights": w["w"]}

    @property
    def dlm_svd_state(self):
        """The filter state in SVD-root form ``{m, UC, SC, s, nu, ...}``.

        Reconstructs the SVD covariance roots from the QR root ``Z`` on demand:
        ``C = Z'Z`` and ``Z = U diag(S) Vᵀ`` give ``C = V diag(S²) Vᵀ``, so
        ``UC = V`` (right singular vectors) and ``SC = S`` (singular values of
        ``Z``). Lets SVD-state consumers — ``multi_model_dlm`` construction and
        the ``svd_fwd_filter`` fallback — read the QR-native state unchanged.
        """
        Z = self.dlm_state["Z"]
        _U, S, Vt = jnp.linalg.svd(Z)
        out = {k: v for k, v in self.dlm_state.items() if k != "Z"}
        out["UC"] = Vt.swapaxes(-1, -2)
        out["SC"] = S
        return out

    def svd_fwd_filter(self, yt):
        """One filter step through the SVD kernel (non-differentiable fallback).

        Structural path only. Runs ``dlm_uv_fwd_svd_step`` via the reconstructed
        SVD state, then re-roots the posterior back to ``Z`` so storage stays
        QR-native. For long / ill-conditioned *pure*-filtering runs where SVD's
        per-step re-orthonormalisation is preferred over a QR root carried across
        thousands of steps; not a path for discount learning (use ``fwd_filter``).
        """
        svd_state = self.dlm_svd_state
        n = self.n_series + self.n_pad
        Ft = jnp.ones((n, 1)) * self.F[jnp.newaxis, :]
        Gt = jnp.ones((n, 1, 1)) * self.G[jnp.newaxis, :, :]
        data_t = {"F": Ft, "G": Gt,
                  "y": device_put(jnp.atleast_2d(pad(yt, self.n_pad)).T, self.device)}
        if self.adapt is None:
            disc = self.disc_rates * self.disc_rates_damped
            disc_mtx = jnp.broadcast_to(
                jnp.diag((1.0 - disc) / disc), (n, disc.shape[-1], disc.shape[-1]))
        else:
            disc_mtx = vmap(adapt_discount, in_axes=(0, 0, 0, None, None, None))(
                svd_state["UC"], svd_state["SC"], svd_state["m"],
                self.disc_rates, self.disc_rates_damped, self.adapt)
        var_disc_b = jnp.broadcast_to(jnp.atleast_1d(self.variance_disc).reshape(-1)[:1], (n, 1))
        var_power_b = jnp.broadcast_to(jnp.atleast_1d(self.variance_power).reshape(-1)[:1], (n, 1))
        mult_b = jnp.broadcast_to(jnp.atleast_2d(self.mult_comps), (n, self.mult_comps.shape[-1]))
        new_svd, self.model = vdlm_uv_fwd_svd_step(
            disc_mtx, var_disc_b, var_power_b, mult_b, svd_state, data_t)
        if new_svd["m"].ndim == 1:
            new_svd["m"] = new_svd["m"][:, jnp.newaxis]
        self.dlm_state = {
            "m": new_svd["m"],
            "Z": axis0dot(vdiag(new_svd["SC"]), new_svd["UC"].swapaxes(-1, -2)),
            "s": new_svd["s"],
            "nu": new_svd["nu"],
        }
        return ForecastBundle(
            self.model["f"][: self.n_series].reshape(self.n_series),
            self.model["q"][: self.n_series].reshape(self.n_series),
        )

    # ------------------------------------------------------------------
    # online RTRL discount learning (Adapt models)
    # ------------------------------------------------------------------
    def enable_adapt(self, grid_model, learn, theta0, *, lr: float = 0.05,
                     clip=(0.5, 0.9999), warmup: int = 6):
        """Turn on online RTRL discount learning (for an ``Adapt`` model).

        After this, :meth:`fwd_filter` runs a single-worker RTRL step per series:
        filter with the current discount θ, carry the forward sensitivity
        ``S = ∂state/∂θ``, and take one online Adam step on the learnable θ
        entries — the discount adapts as the data streams, single pass. Called by
        ``DLM.compile`` when the builder carries ``Adapt``-marked discounts.

        ``grid_model`` is the θ-geometry (``_grid_model(components, var_power)``);
        ``learn`` is the ``(P,)`` bool mask over the ``P`` discount params
        (blocks + β); ``theta0`` the initial discounts (probability space).
        """
        import numpy as _np
        q = self.n_series
        P = int(grid_model.n_params)
        st = {k: self.dlm_state[k][:q] for k in ("m", "Z", "s", "nu")}
        theta0 = jnp.clip(jnp.asarray(theta0), clip[0], clip[1])   # keep logit finite
        self._adapt = {
            "gm": grid_model, "P": P, "lr": float(lr),
            "lo": float(_np.log(clip[0] / (1 - clip[0]))),
            "hi": float(_np.log(clip[1] / (1 - clip[1]))),
            "learn": jnp.asarray(learn, dtype=bool),
            "warmup": int(warmup), "step": 0, "t": 0.0,
            "wth": jnp.broadcast_to(jnp.log(theta0 / (1 - theta0)), (q, P)),
            "S": {k: jnp.zeros((q, P) + v.shape[1:]) for k, v in st.items()},
            "m": jnp.zeros((q, P)), "v": jnp.zeros((q, P)),
        }
        self.dlm_state = st           # drop the monitor slot; adapt owns learning
        return self

    def _adapt_fwd_filter(self, yt):
        import jax
        from DLMAX.ffs.discount_grid import _filter_step, _adam
        a = self._adapt
        gm, P, lr = a["gm"], a["P"], a["lr"]
        lo, hi, learn = a["lo"], a["hi"], a["learn"]
        y = jnp.asarray(yt).reshape(-1)[: self.n_series]
        warm = 1.0 if a["step"] < a["warmup"] else 0.0
        act = warm < 0.5
        if act:
            a["t"] = a["t"] + 1.0
        tb = max(a["t"], 1.0)
        eye = jnp.eye(P)
        nb, mc, vpw = gm.n_blocks, gm.mc, jnp.asarray(gm.var_power)
        F0, G0, s2b = gm.F, gm.G, gm.state_to_block

        def single(st_i, S_i, wth_i, m_i, v_i, y_i):
            theta = jax.nn.sigmoid(wth_i)
            # predictive step at the current discount — the FULL kernel md, so
            # model.model / params() carry the same fields as the fixed path
            # (m, Z, A, s, a, f, q, nu, v_sys, spec_variance).
            delta = jnp.where(warm > 0.5, 1.0, theta[:nb][s2b])
            disc_factor = jnp.diag((1.0 - delta) / delta)
            ns, md = dlm_uv_fwd_qr_step(
                disc_factor, theta[nb], vpw, mc, st_i, {"F": F0, "G": G0, "y": y_i})
            # RTRL gradient via jvp of the (loss-returning) _filter_step
            def dir_i(Ss, ei):
                (_, _, _, _, _), (ns_dot, ls_dot, _, _, _) = jax.jvp(
                    lambda s, th: _filter_step(gm, s, th, y_i, warm),
                    (st_i, theta), (Ss, ei))
                return ns_dot, ls_dot

            S_new, g_ls = vmap(dir_i)(S_i, eye)
            glogit = g_ls * theta * (1.0 - theta) * learn          # freeze non-learned; logit chain
            step_, m_u, v_u = _adam(glogit, m_i, v_i, tb, lr, 0.9, 0.99)
            wth_n = jnp.where(act, jnp.clip(wth_i - step_, lo, hi), wth_i)
            return (ns, S_new, wth_n,
                    jnp.where(act, m_u, m_i), jnp.where(act, v_u, v_i), md, g_ls)

        ns, S_new, wth_n, m_n, v_n, md, g_ls = vmap(single)(
            self.dlm_state, a["S"], a["wth"], a["m"], a["v"], y)
        self.dlm_state = ns
        a["S"], a["wth"], a["m"], a["v"], a["g_ls"] = S_new, wth_n, m_n, v_n, g_ls
        a["step"] += 1
        self.model = md
        return ForecastBundle(
            md["f"].reshape(self.n_series), md["q"].reshape(self.n_series))

    def _adapt_forecast(self, h):
        """h-step forecast for an ``Adapt`` model, using the ONLINE-LEARNED
        discount. Delegates to ``discount_grid.forecast_origin`` per series with
        the current θ and the same ``GridModel`` geometry the filter used, so the
        forecast is consistent with what was learned (no discount/damping
        double-count). Returns ``(n_series, h)`` loc/var."""
        import jax
        from DLMAX.ffs.discount_grid import forecast_origin
        a = self._adapt
        gm = a["gm"]
        theta = jax.nn.sigmoid(a["wth"])                 # (q, P) learned discounts

        def one(st_i, th_i):
            loc, qv, _nu = forecast_origin(gm, st_i, th_i, h)
            return loc, qv

        loc, qv = vmap(one)(self.dlm_state, theta)        # (q, h), (q, h)
        return ForecastBundle(loc, qv)

    # ------------------------------------------------------------------
    # online RTRL discount-GRID learning (Wing models; single series)
    # ------------------------------------------------------------------
    def enable_wing(self, grid_model, centre_init, *, offset=1.0, wings=3,
                    lr=0.05, warmup=6, beta_init=0.99):
        """Turn on online RTRL discount-GRID learning (a ``Wing`` model).

        Sets up a single 3-wingman cell (moving centre + in-class DMA) for the
        series; :meth:`fwd_filter` then advances it one step via
        ``discount_grid._wing_step`` and returns the DMA-combined predictive.
        The both-axes learner: the wingmen straddle the centre (discrete/DMA) and
        the centre moves by RTRL (continuous). ``n_series == 1`` for now.
        """
        import numpy as _np
        from jax import tree_util as _tu
        from DLMAX.ffs.discount_grid import _dma_update, _clip_box, DMA_PDR, DMA_MDR, DMA_C
        q, gm = self.n_series, grid_model
        P = int(gm.n_params)
        lo, hi = _clip_box(gm)
        offsets = jnp.linspace(-offset, offset, wings)       # [-off, .., +off]
        c0s = float(_np.log(centre_init / (1 - centre_init)))
        mu0b = float(_np.log(beta_init / (1 - beta_init)))
        st = {k: self.dlm_state[k][:q] for k in ("m", "Z", "s", "nu")}          # (q, ...)
        # each series -> `wings` wingmen: state (q, wings, ...); S = ∂state/∂θ zeros.
        st3 = {k: jnp.broadcast_to(v[:, None], (q, wings) + v.shape[1:]) for k, v in st.items()}
        S3 = {k: jnp.zeros((q, wings, P) + v.shape[1:]) for k, v in st.items()}
        wth1 = jnp.full((wings, P), c0s).at[:, -1].set(mu0b)  # block 0 reset each step; β=mu0b
        wth0 = jnp.broadcast_to(wth1, (q, wings, P))
        mi = jnp.ones((wings, 1), dtype=bool)
        alloc0 = _tu.tree_map(lambda x: jnp.broadcast_to(x, (q,) + x.shape),
                              init_alloc_state(wings, 1, 1, mi, 1))
        zc = jnp.zeros((q, wings))
        self._wing = {
            "gm": gm, "P": P, "offsets": offsets, "lo": lo, "hi": hi,
            "use_ls": jnp.zeros(P, bool).at[gm.n_blocks].set(True),
            "mi": mi, "upd": _dma_update(DMA_PDR, DMA_MDR, DMA_C),
            "lr": float(lr), "warmup": int(warmup), "step": 0, "wings": int(wings),
            "w": jnp.ones((q, wings)) / wings,     # last DMA weights (q, wings)
            "carry": (st3, S3, jnp.full((q,), c0s), zc, zc, wth0,
                      jnp.zeros((q, wings, P)), jnp.zeros((q, wings, P)),
                      alloc0, jnp.zeros((q,))),
        }
        return self

    def _wing_fwd_filter(self, yt):
        from DLMAX.ffs.discount_grid import _wing_step
        w = self._wing
        y = jnp.asarray(yt).reshape(-1)[: self.n_series]
        wrm = jnp.asarray(1.0 if w["step"] < w["warmup"] else 0.0)

        def step_fn(carry, yi):
            return _wing_step(carry, yi, wrm, model=w["gm"], offsets=w["offsets"],
                              lo=w["lo"], hi=w["hi"], use_ls=w["use_ls"], mi=w["mi"],
                              upd=w["upd"], lr_level=w["lr"], lr_worker=w["lr"],
                              b1=0.9, b2=0.99)

        new_carry, out = vmap(step_fn)(w["carry"], y)         # cell per series
        w["carry"] = new_carry
        w["w"] = out["w"]                                     # (q, wings)
        w["step"] += 1
        # DMA-combine the wingmen per series: mean + quantile-avg sd; the
        # state-contribution v_sys is the DMA-weighted average over the wingmen.
        f, qv, wt = out["f"], out["q"], out["w"]              # (q, wings)
        f_c = jnp.sum(wt * f, axis=1)                         # (q,)
        sd_c = (jnp.sum(wt * (f + 1.96 * jnp.sqrt(qv)), axis=1) - f_c) / 1.96
        q_c = sd_c ** 2
        v_sys_c = jnp.sum(wt * out["v_sys"], axis=1)          # (q,)
        self.model = {"f": f_c, "q": q_c, "v_sys": v_sys_c,
                      "spec_variance": jnp.clip(q_c - v_sys_c, 0.0, None)}
        return ForecastBundle(f_c.reshape(self.n_series), q_c.reshape(self.n_series))

    def _wing_forecast(self, h):
        """h-step forecast for a ``Wing`` model: forecast each wingman from its
        state at its current discount, DMA-combine with the last weights."""
        import jax
        from DLMAX.ffs.discount_grid import forecast_origin
        w = self._wing
        gm = w["gm"]
        st3, _, c, _, _, wth = w["carry"][:6]                 # (q, wings, ...), (q,), (q, wings, P)
        offsets, lo, hi = w["offsets"], w["lo"], w["hi"]

        def per_series(st3_i, c_i, wth_i, w_i):
            th = jax.nn.sigmoid(wth_i.at[:, 0].set(jnp.clip(c_i + offsets, lo[0], hi[0])))
            loc, qv = vmap(lambda s, t: forecast_origin(gm, s, t, h)[:2])(st3_i, th)
            wt = w_i[:, None]
            loc_c = jnp.sum(wt * loc, axis=0)                 # (h,)
            sd_c = (jnp.sum(wt * (loc + 1.96 * jnp.sqrt(qv)), axis=0) - loc_c) / 1.96
            return loc_c, sd_c ** 2

        loc, var = vmap(per_series)(st3, c, wth, w["w"])      # (q, h)
        return ForecastBundle(loc, var)

    # ------------------------------------------------------------------
    # forecasting
    # ------------------------------------------------------------------

    def forecast(self, h, regressors=None, Gt=None, regressors_var=None):
        """h-step forecast for every series.

        ``regressors`` supplies the future regressor values over the horizon
        (shape ``h x n_series x n_regressors``); they fill the regression tail
        of F at each horizon. By default they are treated as *known*. Pass
        ``regressors_var`` (same shape) to mark the regressors as *stochastic* —
        the supplied per-horizon variance is propagated into the predictive
        variance ``q`` (via :func:`dlm_uv_fcast_step_exog`), so a forecast made
        on uncertain future regressors carries that uncertainty. ``Gt``
        overrides the regression-coefficient transition (defaults to
        ``self.regression_G``).

        Returns
        -------
        ForecastBundle
            h-step predictive ``loc`` and ``var`` (variance), each shaped
            ``(n_series, h)``. The raw per-horizon moments are also stored on
            ``self.fc_model``.
        """
        if self._adapt is not None:
            return self._adapt_forecast(h)
        if self._wing is not None:
            return self._wing_forecast(h)

        if regressors_var is not None and regressors is None:
            raise ValueError(
                "regressors_var supplied without regressors; provide the "
                "regressor means alongside their variances."
            )
        # Build F over the horizon
        F_base = pad(
            self.F[jnp.newaxis, jnp.newaxis, :] * jnp.ones((self.n_series, h, 1)),
            self.n_pad,
        )
        if regressors is not None:
            # regressors passed as h x n x k; reshape to n x h x k for parallel exec
            regressors = device_put(
                pad(regressors.swapaxes(0, 1), self.n_pad),
                self._device_3d,
            )
            F = jnp.concatenate([F_base, regressors], axis=2)
            G = self._broadcast_G(Gt)
        else:
            F = F_base
            G = (
                self._broadcast_G(Gt)
                if Gt is not None
                else (
                    jnp.ones((self.n_series + self.n_pad, 1, 1))
                    * self.G[jnp.newaxis, :, :]
                )
            )

        fc_data = {
            "F": F.swapaxes(0, 1),
            "G": jnp.repeat(G[jnp.newaxis, ...], h, 0),
        }

        init = {k: self.dlm_state[k] for k in ["m", "s", "nu"]}
        init["C"] = axis0dot(
            self.dlm_state["Z"].swapaxes(-1, -2), self.dlm_state["Z"]
        )  # C = Z' Z
        # W = ((1-δ)/δ) C with the combined level/trend discount δ·δ_damped,
        # matching multi_model_dlm.forecast. self.disc_rates here is 1-D
        # (n_states,), so jnp.diag (not the batched vdiag) builds the (k, k)
        # diagonal that broadcasts over init["C"] (n_series, k, k).
        W = init["C"] * jnp.diag(
            (1 - self.disc_rates * self.disc_rates_damped)
            / (self.disc_rates * self.disc_rates_damped)
        )

        if regressors_var is None:
            def scan_fn(state, data):
                return vdlm_uv_fcast_step(
                    W,
                    self.variance_disc,
                    self.variance_power,
                    self.mult_comps,
                    state,
                    data,
                )
        else:
            n_reg = self.n_regressors
            # per-horizon regressor variance, padded/transposed to match F
            v = pad(jnp.asarray(regressors_var).swapaxes(0, 1), self.n_pad)
            fc_data["v"] = v.swapaxes(0, 1)            # (h, n_series + n_pad, n_reg)

            def scan_fn(state, data):
                return vdlm_uv_fcast_step_exog(
                    W,
                    self.variance_disc,
                    self.variance_power,
                    self.mult_comps,
                    n_reg,
                    state,
                    data,
                )

        _, self.fc_model = scan(scan_fn, init=init, xs=fc_data)

        return ForecastBundle(
            self.fc_model["f"][:, : self.n_series].T,
            # q carries a trailing singleton from the (1, 1) quadratic form
            # F.T @ R @ F, which f (a scalar) does not. reshape before the
            # transpose so var matches loc exactly at (n_series, h).
            self.fc_model["q"][:, : self.n_series].reshape(h, self.n_series).T,
        )


# -----------------------------------------------------------------------------
# multi_model_dlm
# -----------------------------------------------------------------------------


def _stack_and_broadcast(values, q, dlm_compute, reshape_to):
    """Stack per-model leading axis, broadcast across ``q`` series, reshape
    to the flat ``(p, ...)`` layout, then place on ``dlm_compute``.

    The reshape is applied before ``device_put`` because JAX cannot reshape
    across a sharded dimension without an explicit ``out_sharding``.
    """
    arr = jnp.stack(values)
    arr = jnp.expand_dims(arr, axis=1) * jnp.ones((1, q) + (1,) * (arr.ndim - 1))
    return device_put(arr.reshape(reshape_to), dlm_compute)


class multi_model_dlm(object):
    """Packs a dict of ``uv_dlm`` instances into a single flat-batched filter.

    The DMA workhorse: a universe of candidate models is stacked on a leading
    model axis and filtered as one ``(n_models * n_series)`` batch, so the whole
    model space is advanced per observation in a single compiled kernel. Models
    of differing state dimension are padded to a common size with inert trailing
    slots, so structural and AR/TVAR models can be packed together.

    Construct from a dict of per-series :class:`uv_dlm` models, or build an
    empty instance and restore a persisted universe with :meth:`load`.

    Parameters
    ----------
    dlms : dict[str, uv_dlm] or None
        Per-model ``uv_dlm`` instances keyed by name. If ``None``, the universe
        is left uninitialised for a subsequent :meth:`load`.
    device : jax.sharding.NamedSharding, optional
        Sharding on which to place the packed state. Defaults to the FFS host
        device.

    Notes
    -----
    Key methods: :meth:`fwd_filter` / :meth:`scan_filter` ingest observations
    (single step / whole series); :meth:`forecast`, :meth:`forecast_iterated`
    and :meth:`forecast_exog` emit h-step predictive moments for the structural,
    AR/TVAR and exogenous-regression cases respectively; the ``format_*``
    helpers shape observation and regressor batches for those calls; and
    :meth:`save` / :meth:`load` round-trip the packed universe to HDF5.
    """

    def __init__(self, dlms=None, device=None):
        require_x64()
        if device is None:
            from DLMAX.ffs import devices as ffs_devices
            device = ffs_devices.host_device
        self.dlm_compute = device
        if dlms is not None:
            self._init_from_dlms(dlms)

    def _init_from_dlms(self, dlms):
        self.dlm_state = {}
        first = next(iter(dlms.values()))

        self.q = first.n_series
        self.nm = len(dlms)
        self.p = self.q * self.nm
        # Total state dim includes the regression tail. The structural F/G are
        # zero-/block-diag-padded into the regression slots below, and the
        # regressor F values are supplied at filter time. For n_regressors == 0
        # (the structural-only universe) everything reduces exactly to the
        # prior behaviour (bit-exact).
        #
        # Mixed universes (structural + TVAR) pack at a common total state dim:
        # each model's len(F) + n_regressors must agree. The regression tail is
        # the last `self.n_regressors` slots (max over models); a per-model mask
        # (reg_mask) gates the lag-fill so it touches only the AR/TVAR models and
        # leaves the structural models' F untouched.
        # Pad every model to a common total state dim K = max(len(F)+n_reg).
        # Smaller models (e.g. structural models packed alongside a higher-order
        # TVAR class — a no-seasonal universe with AR(4), k=5, vs LocalTrend k=2)
        # get inert trailing slots: F=0, G block-diag I, m=0, C0=0 (SC=0, UC
        # identity block), disc=1 (no evolution). No-op when all k agree (the
        # usual case) -> bit-exact with the prior packing.
        totals = [len(d.F) + d.n_regressors for d in dlms.values()]
        self.k = max(totals)
        self.n_regressors = max(d.n_regressors for d in dlms.values())
        K = self.k

        def _pad_last(a, fill):
            extra = K - a.shape[-1]
            if extra == 0:
                return a
            return jnp.concatenate(
                [a, jnp.full(a.shape[:-1] + (extra,), fill, a.dtype)], axis=-1)

        def _pad_UC(a):
            kt = a.shape[-1]
            if kt == K:
                return a
            return vmap(lambda u: block_diag(u, jnp.eye(K - kt, dtype=a.dtype)))(a)

        def _pad_G_mat(g):
            kt = g.shape[-1]
            return g if kt == K else block_diag(g, jnp.eye(K - kt, dtype=g.dtype))

        def _pad_Z(a):
            # QR root (q, k, k) -> (q, K, K) with zeros: C = Z'Z, so zero-padding
            # the root gives block_diag(C, 0) — inert extra states, matching the
            # SVD padding's zero-variance trailing slots.
            kt = a.shape[-1]
            if kt == K:
                return a
            return jnp.pad(a, ((0, 0), (0, K - kt), (0, K - kt)))
        # Per-(model, series) flag: True where the model has an AR/regression
        # tail. Model-major (p,), matching the packed layout.
        is_reg = jnp.array([d.n_regressors > 0 for d in dlms.values()])
        self.reg_mask = device_put(
            jnp.repeat(is_reg, self.q), self.dlm_compute
        )
        # Exogenous (caller-supplied X) vs autoregressive (own-lag) regression
        # tail. An exog universe drives the streaming filter/forecast via
        # format_exog_yts / forecast_exog; an AR universe via format_lag_yts /
        # forecast_iterated. No tail (n_regressors == 0) -> False, leaving the
        # structural path bit-exact. Per-model is_autoregressive is stamped by
        # the DLM builder (dlm_builder.DLM.compile).
        self.exog_regressors = bool(self.n_regressors > 0) and not any(
            getattr(d, "is_autoregressive", False) for d in dlms.values()
        )

        # state dict: stack on model axis, then flatten (nm, q, ...) -> (p, ...).
        # Pad the k-sized state to K: m/SC append 0, UC gets an identity block;
        # s/nu are scalar (per-series) and pass through.
        # Both the members and multi_model_dlm are QR-native ({m, Z, s, nu, S}):
        # pad each member's root Z (and mean m) to the common dim K and stack.
        member_states = [d.dlm_state for d in dlms.values()]
        for param in member_states[0].keys():
            if param == "Z":
                arrs = [_pad_Z(v[param]) for v in member_states]
            elif param == "m":
                arrs = [_pad_last(v[param], 0.0) for v in member_states]
            else:
                arrs = [v[param] for v in member_states]
            new_shape = (self.p,) + arrs[0].shape[1:]
            self.dlm_state[param] = device_put(
                jnp.stack(arrs), self.dlm_compute,
            ).reshape(new_shape)

        self.mdl_keys = list(dlms.keys())
        self.npad = dlms[self.mdl_keys[0]].n_pad

        def _full_F(d):
            # Structural F padded with zeros in the regression slots (filled
            # with regressor values at filter time).
            return (d.F if d.n_regressors == 0
                    else jnp.concatenate([d.F, jnp.zeros(d.n_regressors)]))

        def _full_G(d):
            # Structural G block-diagonalised with the regression transition.
            return d.G if d.n_regressors == 0 else block_diag(d.G, d.regression_G)

        self.F = device_put(
            jnp.stack([_pad_last(_full_F(d), 0.0) for d in dlms.values()]),
            self.dlm_compute,
        )
        self.G = device_put(
            jnp.stack([_pad_G_mat(_full_G(d)) for d in dlms.values()]),
            self.dlm_compute,
        )

        # GH is only present if models were created with h != None.
        # The per-model uv_dlm.GH is built from the *structural* G (before the
        # regression block), so for n_regressors == 0 stacking it is exact. With
        # a regression tail it would be the wrong rank — rebuild GH from the full
        # block-diag G so the regression block (G_reg = I) propagates through the
        # forecast horizon.
        try:
            if self.n_regressors == 0:
                self.GH = device_put(
                    jnp.stack([d.GH for d in dlms.values()]), self.dlm_compute
                )
            else:
                h_fc = next(iter(dlms.values())).GH.shape[1]
                Gf = jnp.stack([_pad_G_mat(_full_G(d)) for d in dlms.values()])  # (nm,K,K)
                powers = [Gf]
                for _ in range(h_fc - 1):
                    powers.append(jnp.einsum("mij,mjk->mik", Gf, powers[-1]))
                # (nm, h, k, k) -> (nm, 1, h, k, k), matching the structural stack
                self.GH = device_put(
                    jnp.stack(powers, axis=1)[:, jnp.newaxis, ...], self.dlm_compute
                )
        except AttributeError:
            self.GH = None

        # per-model scalars/vectors → broadcast across q → flatten
        self.disc_rates = _stack_and_broadcast(
            [_pad_last(d.disc_rates, 1.0) for d in dlms.values()],
            self.q,
            self.dlm_compute,
            (self.p, self.k),
        )
        self.disc_rates_damped = _stack_and_broadcast(
            [_pad_last(d.disc_rates_damped, 1.0) for d in dlms.values()],
            self.q,
            self.dlm_compute,
            (self.p, self.k),
        )
        self.monitor_inject = _stack_and_broadcast(
            [_pad_last(d.monitor_inject, 1.0) for d in dlms.values()],
            self.q,
            self.dlm_compute,
            (self.p, self.k),
        )
        self.variance_disc = _stack_and_broadcast(
            [d.variance_disc for d in dlms.values()],
            self.q,
            self.dlm_compute,
            (self.p, 1),
        )
        self.variance_power = _stack_and_broadcast(
            [d.variance_power for d in dlms.values()],
            self.q,
            self.dlm_compute,
            (self.p, 1),
        )
        self.mult_comps = _stack_and_broadcast(
            [_pad_last(d.mult_comps, 0.0) for d in dlms.values()],
            self.q,
            self.dlm_compute,
            (self.p, self.k),
        )
        self.monitor = _stack_and_broadcast(
            [d.monitor for d in dlms.values()],
            self.q,
            self.dlm_compute,
            (self.p, 1),
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, fname):
        """Persist the packed universe to an HDF5 file.

        Writes the filtered state (``dlm_state``), the universe dimensions and
        model keys, and the static structural arrays (``F``, ``G``, ``GH``,
        discount / variance / multiplicative-component parameters) so the
        universe can be restored with :meth:`load`.

        Parameters
        ----------
        fname : str or path-like
            Destination HDF5 path.
        """
        with h5py.File(fname, "w") as f:
            g = f.create_group("dlm_state")
            for param, arr in self.dlm_state.items():
                g.create_dataset(param, data=arr)

            g = f.create_group("dims")
            dims = {
                "q": self.q,
                "nm": self.nm,
                "p": self.p,
                "k": self.k,
                "npad": self.npad,
                "mdl_keys": self.mdl_keys,
            }
            for param, val in dims.items():
                g.create_dataset(param, data=val)

            g = f.create_group("Other")
            for param in [
                "F",
                "G",
                "GH",
                "disc_rates",
                "disc_rates_damped",
                "monitor_inject",
                "variance_disc",
                "variance_power",
                "mult_comps",
                "monitor",
            ]:
                g.create_dataset(param, data=getattr(self, param))

            # Regression-tail descriptors. n_regressors / reg_mask / G already
            # encode the (block-diag) regression transition, but the tail kind
            # and packed mask must round-trip so a reloaded universe keeps
            # filtering/forecasting with regressors instead of silently
            # degrading to structural (getattr default 0). Absent on legacy
            # structural files -> restored as 0 / empty / False (no tail).
            g.create_dataset("n_regressors", data=int(self.n_regressors))
            g.create_dataset("reg_mask", data=np.asarray(self.reg_mask))
            g.create_dataset(
                "exog_regressors", data=int(getattr(self, "exog_regressors", False))
            )

    def load(self, fname):
        """Restore a packed universe previously written by :meth:`save`.

        Populates ``self.dlm_state``, the dimensions / model keys and the static
        arrays from ``fname``, placing every array on ``self.dlm_compute``.
        Universes saved before the monitor-injection field existed restore with
        ``monitor_inject`` defaulted to ones (legacy behaviour, no injection).

        Parameters
        ----------
        fname : str or path-like
            Source HDF5 path, as written by :meth:`save`.
        """
        with h5py.File(fname, "r") as f:
            self.dlm_state = {
                param: device_put(jnp.asarray(f["dlm_state"][param]), self.dlm_compute)
                for param in f["dlm_state"].keys()
            }

            g = f["dims"]
            # Read dimensions as Python ints for consistency with save-time types.
            self.q = int(g["q"][()])
            self.nm = int(g["nm"][()])
            self.p = int(g["p"][()])
            self.k = int(g["k"][()])
            self.npad = int(g["npad"][()])
            self.mdl_keys = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in g["mdl_keys"][()]
            ]

            g = f["Other"]
            for param in [
                "F",
                "G",
                "GH",
                "disc_rates",
                "disc_rates_damped",
                "variance_disc",
                "variance_power",
                "mult_comps",
                "monitor",
            ]:
                setattr(
                    self,
                    param,
                    device_put(jnp.asarray(g[param]), self.dlm_compute),
                )
            # Back-compat: universes persisted before the monitor-injection field
            # existed have no `monitor_inject` group; default to ones (no injection,
            # i.e. legacy behaviour) so they restore cleanly.
            if "monitor_inject" in g:
                self.monitor_inject = device_put(
                    jnp.asarray(g["monitor_inject"]), self.dlm_compute
                )
            else:
                self.monitor_inject = jnp.ones_like(self.disc_rates)

            # Regression-tail descriptors (back-compat: legacy structural files
            # have none -> no tail, structural path unchanged).
            if "n_regressors" in g:
                self.n_regressors = int(g["n_regressors"][()])
                self.reg_mask = device_put(
                    jnp.asarray(g["reg_mask"]), self.dlm_compute
                )
                self.exog_regressors = bool(int(g["exog_regressors"][()]))
            else:
                self.n_regressors = 0
                self.reg_mask = device_put(
                    jnp.zeros(self.p, dtype=bool), self.dlm_compute
                )
                self.exog_regressors = False

    # ------------------------------------------------------------------
    # filtering / forecasting
    # ------------------------------------------------------------------

    def format_yts(self, ys):
        """convenience function to format a batch of ys for batch processing
        and place them on the appropriate device"""
        t = len(ys)
        return device_put(
            jnp.stack(
                [
                    (pad(ys[i], self.npad) * jnp.ones((self.nm, 1))).reshape(self.p)
                    for i in range(t)
                ]
            ),
            self.dlm_compute,
        )

    def format_lags(self, ys, order, history=None):
        """Build the AR lag array for a batch of observations.

        The autoregressive analogue of :meth:`format_yts`: instead of
        broadcasting an exogenous regressor, the regressors at step ``t`` are
        the series' own lagged values ``[y_{t-1}, ..., y_{t-order}]``. These
        are a deterministic function of the y stream, so they are precomputed
        here (outside the filter scan) rather than threaded through a buffer.

        Parameters
        ----------
        ys : array, shape ``(T, n_series)``
            The observation batch (unpadded series axis).
        order : int
            AR order — number of lags.
        history : array, optional, shape ``(order, n_series)``
            The last ``order`` observations preceding this batch, used to fill
            the lags of the first ``order`` steps when continuing a filter
            across batches. ``None`` (default, an initial batch) zeroes the
            warmup region so the AR tail contributes nothing until it has
            ``order`` real observations.

        Returns
        -------
        array, shape ``(T, n_series, order)``
            ``out[t, s, j] = y_{t-1-j}`` for series ``s`` (lag ``j+1``), with
            unavailable warmup lags set to 0. Pass each ``out[t]`` slice to
            :meth:`fwd_filter` as ``regressors``.
        """
        ys = jnp.asarray(ys)
        T, n = ys.shape
        if history is None:
            ext = jnp.concatenate([jnp.zeros((order, n), ys.dtype), ys], axis=0)
        else:
            history = jnp.asarray(history)
            ext = jnp.concatenate([history[-order:], ys], axis=0)
        # ext has `order` rows before row 0 of ys; lag j+1 at step t is
        # ext[order + t - 1 - j].
        return jnp.stack(
            [ext[order - 1 - j : order - 1 - j + T] for j in range(order)],
            axis=-1,
        )

    def format_lag_yts(self, ys, history=None):
        """Scan-ready lag array: the ``(T, p, n_regressors)`` analogue of
        :meth:`format_yts`.

        Builds :meth:`format_lags` (order = ``self.n_regressors``) and then
        broadcasts the lags across models and pads the series axis to ``q``,
        producing one ``(p, n_regressors)`` regressor slice per step — exactly
        the shape :func:`_multi_fwd_filter_step` expects as ``regressors``.
        Model-major ordering matches ``format_yts`` and the packed state.

        Returns an empty ``(T, p, 0)`` array when there is no regression tail.
        """
        ys = jnp.asarray(ys)
        T = ys.shape[0]
        n_reg = self.n_regressors
        if n_reg == 0:
            return jnp.zeros((T, self.p, 0), ys.dtype)
        lags = self.format_lags(ys, n_reg, history=history)        # (T, n, n_reg)
        # pad series axis to q, then broadcast across the nm models (model-major)
        lags = jnp.concatenate(
            [lags, jnp.zeros((T, self.npad, n_reg), lags.dtype)], axis=1
        )                                                          # (T, q, n_reg)
        lags = lags[:, jnp.newaxis] * jnp.ones((1, self.nm, 1, 1))  # (T, nm, q, n_reg)
        return device_put(lags.reshape(T, self.p, n_reg), self.dlm_compute)

    def format_seed_lag_yts(self, ys, history=None):
        """Scan-ready *seed* lags including the current observation:
        ``seed[t, s, j] = y_{t-j}`` for ``j = 0 .. n_regressors-1`` (most-recent
        first, lag 0 = ``y_t``), shaped ``(T, p, n_regressors)``.

        Whereas :meth:`format_lag_yts` gives the lags entering the *filter* step
        at ``t`` (``y_{t-1} ...``), this gives the lags that seed the *iterated
        forecast* emitted at a cutoff ``t``: the first forecast horizon's lag is
        ``y_t``. Empty ``(T, p, 0)`` when there is no regression tail.
        """
        ys = jnp.asarray(ys)
        T, n = ys.shape
        n_reg = self.n_regressors
        if n_reg == 0:
            return jnp.zeros((T, self.p, 0), ys.dtype)
        pad_rows = n_reg - 1
        if history is None or pad_rows == 0:
            ext = jnp.concatenate([jnp.zeros((pad_rows, n), ys.dtype), ys], axis=0)
        else:
            ext = jnp.concatenate([jnp.asarray(history)[-pad_rows:], ys], axis=0)
        # ext has pad_rows rows before row 0; seed lag j at step t is
        # ext[pad_rows + t - j].
        seed = jnp.stack(
            [ext[pad_rows - j : pad_rows - j + T] for j in range(n_reg)], axis=-1
        )                                                          # (T, n, n_reg)
        seed = jnp.concatenate(
            [seed, jnp.zeros((T, self.npad, n_reg), seed.dtype)], axis=1
        )                                                          # (T, q, n_reg)
        seed = seed[:, jnp.newaxis] * jnp.ones((1, self.nm, 1, 1))
        return device_put(seed.reshape(T, self.p, n_reg), self.dlm_compute)

    def format_exog_yts(self, X):
        """Scan-ready exogenous-regressor array: the supplied-``X`` analogue of
        :meth:`format_lag_yts`. Whereas the AR formatter derives the regression
        tail from lags of ``y``, here the caller supplies the regressor matrix
        directly (shared across all models — the shared-regressor design).

        Parameters
        ----------
        X : array, shape ``(T, n_series, n_regressors)``
            Per-step exogenous regressor values, aligned to the same time axis
            and series order as the observations. The last axis must match
            ``self.n_regressors`` (the packed tail size).

        Returns
        -------
        array, shape ``(T, p, n_regressors)`` — one ``(p, n_regressors)`` slice
        per step, model-major and series-padded to ``q``, exactly the shape
        :func:`_multi_fwd_filter_step` expects as ``regressors``. Empty
        ``(T, p, 0)`` when there is no regression tail.
        """
        X = jnp.asarray(X)
        T = X.shape[0]
        n_reg = self.n_regressors
        if n_reg == 0:
            return jnp.zeros((T, self.p, 0), X.dtype)
        if X.ndim != 3 or X.shape[-1] != n_reg:
            raise ValueError(
                f"X must have shape (T, n_series, {n_reg}); got {tuple(X.shape)}."
            )
        # pad series axis to q, then broadcast across the nm models (model-major)
        X = jnp.concatenate(
            [X, jnp.zeros((T, self.npad, n_reg), X.dtype)], axis=1
        )                                                          # (T, q, n_reg)
        X = X[:, jnp.newaxis] * jnp.ones((1, self.nm, 1, 1))       # (T, nm, q, n_reg)
        return device_put(X.reshape(T, self.p, n_reg), self.dlm_compute)

    def prepared_step(self):
        """No-op hook retained for API compatibility with the streaming driver."""
        return

    def _monitor_tau(self, enabled=True):
        """Per-(model,series) tau vector for the signed-error monitor, or
        ``None`` for the static-grid path.

        Mirrors :func:`DLMAX.ffs_core._adapt_tau` so the standalone drivers
        (:meth:`fwd_filter`, :meth:`scan_filter`) make the same static-vs-
        adaptive decision as the AutoFFS scan path, without importing from
        ``ffs_core`` (which would be circular). The discriminator is the
        monitor VALUE, not its dtype: a legacy ``monitor=False`` coerces to a
        float-zero array, whereas an adaptive universe stamps strictly-positive
        tau (>= 1) into the monitor slot. Any positive entry => adaptive
        (reshape to ``(p,)`` to match ``dlm_state["S"]``); all-zero / absent =>
        ``None``, so ``_multi_fwd_filter_step`` takes the static branch
        (bit-exact with the legacy structural path).

        Parameters
        ----------
        enabled : bool, default True
            When False, force the static path regardless of the universe's
            monitor slot (the ``monitor=False`` caller override).
        """
        if not enabled:
            return None
        mon = getattr(self, "monitor", None)
        if mon is None:
            return None
        mon = jnp.asarray(mon)
        if bool(jnp.any(mon > 0)):
            return mon.reshape(-1)
        return None

    def fwd_filter(self, yt, regressors=None, Gt=None, monitor=True):
        """Advance the whole universe by one observation.

        Single forward-filtering step over the packed ``(n_models * n_series)``
        batch: updates ``self.dlm_state`` in place and returns the one-step-ahead
        predictive moments for every (model, series). For a whole series in one
        compiled scan, use :meth:`scan_filter`.

        Parameters
        ----------
        yt : array, shape ``(n_series,)``
            The observation for each series at this step (broadcast across all
            models internally).
        regressors : array, optional, shape ``(n_series, n_regressors)``
            Per-series regressor values (e.g. AR lags from :meth:`format_lags`),
            broadcast across models. ``None`` runs the structural path.
        Gt : optional
            Reserved; not currently applied.
        monitor : bool, default True
            Enable the signed-error monitor's adaptive-discount intervention
            *if the universe is adaptive* (i.e. carries a positive tau in its
            monitor slot, as built by ``make_ffs_universe_adaptive``). On a
            static/legacy universe there is no per-state tau to use, so the
            static-grid path runs regardless and the step is bit-exact.
            ``monitor=False`` forces the static path even on an adaptive
            universe. See :meth:`_monitor_tau`.

        Returns
        -------
        ForecastBundle
            One-step-ahead predictive ``loc`` and ``var`` (variance), each
            shaped ``(n_models, n_series)``. The filtered state is also stored
            on ``self.dlm_state`` / ``self.model``.
        """

        # new
        multi_params = (
            self.F,
            self.G,
            self.nm,
            self.k,
            self.q,
            self.p,
            self.disc_rates,
            self.disc_rates_damped,
            self.monitor_inject,
            self.variance_disc,
            self.variance_power,
            self.mult_comps,
        )

        # Broadcast per-series regressors (e.g. AR lags) across models, mirroring
        # the yt broadcast: (n_series, n_reg) -> padded (q, n_reg) -> (p, n_reg).
        # None leaves Ft as the static structural F (bit-exact structural path).
        reg_t = None
        if regressors is not None:
            regressors = jnp.asarray(regressors)
            reg_t = device_put(
                (pad(regressors, self.npad)[None] * jnp.ones((self.nm, 1, 1))).reshape(
                    self.p, regressors.shape[-1]
                ),
                self.dlm_compute,
            )

        self.dlm_state, self.model, f, q = _multi_fwd_filter_step(
            self.dlm_state,
            multi_params,
            device_put(
                pad(yt, self.npad) * jnp.ones((self.nm, 1)), self.dlm_compute
            ).reshape(self.p),
            regressors=reg_t,
            reg_mask=getattr(self, "reg_mask", None),
            tau=self._monitor_tau(monitor),
        )

        return ForecastBundle(f, q)

    def scan_filter(self, ys, regressors=None, update_state=True, monitor=True):
        """Run the multi-model forward filter over a whole series in one
        compiled ``lax.scan`` — the multi-level analogue of the single-step
        :meth:`fwd_filter`, and the standalone driver for a regression/AR
        multi-DLM *outside* the AutoFFS streaming machinery.

        Threads the per-step regressor slices through the filter, returns the
        stacked one-step-ahead predictive moments, and (by default) advances
        ``self.dlm_state`` to the end of the series. Runs the signed-error
        monitor's adaptive-discount path on an adaptive universe and the
        static-grid path otherwise, exactly as :meth:`fwd_filter` does.

        Parameters
        ----------
        ys : array, shape ``(T, n_series)``
            The observations.
        regressors : array, optional, shape ``(T, p, n_reg)``
            Per-step regressor slices, as produced by :meth:`format_exog_yts`
            (exogenous) or :meth:`format_lag_yts` (AR lags). ``None`` runs the
            purely structural filter.
        update_state : bool, default True
            Store the final filtered state back on ``self.dlm_state``.
        monitor : bool, default True
            Enable the signed-error monitor *if the universe is adaptive*; on a
            static/legacy universe the static-grid path runs regardless (the
            step is bit-exact). ``monitor=False`` forces the static path. The
            tau is resolved once via :meth:`_monitor_tau` and held fixed across
            the scan (it is a concrete None/array decided outside the trace,
            as ``lax.scan`` requires). See :meth:`fwd_filter`.

        Returns
        -------
        ForecastBundle
            Per-step one-step-ahead predictive ``loc`` and ``var`` (variance)
            for every (model, series), each shaped ``(T, nm, q)``.
        """
        ys = jnp.asarray(ys)
        T = ys.shape[0]
        multi_params = (
            self.F, self.G, self.nm, self.k, self.q, self.p,
            self.disc_rates, self.disc_rates_damped, self.monitor_inject,
            self.variance_disc, self.variance_power, self.mult_comps,
        )
        reg_mask = getattr(self, "reg_mask", None)
        # Resolve tau ONCE outside the trace: _multi_fwd_filter_step branches on
        # `tau is None` (Python-level), so it must be a concrete None/array, not
        # a traced value. Held fixed across the scan.
        tau = self._monitor_tau(monitor)

        # The single-step kernel emits s/nu as (p, 1) (and m as (p, k)); the
        # freshly-packed state carries s/nu as (p,). Normalise the scan's
        # initial carry to the post-step layout so the carry shape is stable
        # across iterations (lax.scan requires input/output carry types match).
        init_state = dict(self.dlm_state)
        for key in ("s", "nu"):
            v = init_state[key]
            if v.ndim == 1:
                init_state[key] = v[:, jnp.newaxis]

        # observations -> model-major (T, p), mirroring fwd_filter's per-step
        # broadcast (pad series to q, replicate across the nm models).
        Yq = jnp.concatenate(
            [ys, jnp.zeros((T, self.npad), ys.dtype)], axis=1
        )                                                          # (T, q)
        Y = device_put(
            (Yq[:, jnp.newaxis] * jnp.ones((1, self.nm, 1))).reshape(T, self.p),
            self.dlm_compute,
        )

        if regressors is None:
            def step(state, y_t):
                new_state, _model, f, q = _multi_fwd_filter_step(
                    state, multi_params, y_t, reg_mask=reg_mask, tau=tau,
                )
                return new_state, (f, q)

            final_state, (f, q) = scan(step, init_state, Y)
        else:
            regressors = device_put(jnp.asarray(regressors), self.dlm_compute)

            def step(state, xs):
                y_t, reg_t = xs
                new_state, _model, f, q = _multi_fwd_filter_step(
                    state, multi_params, y_t,
                    regressors=reg_t, reg_mask=reg_mask, tau=tau,
                )
                return new_state, (f, q)

            final_state, (f, q) = scan(step, init_state, (Y, regressors))

        if update_state:
            self.dlm_state = final_state
        return ForecastBundle(f, q)

    def _forecast_disc_factor_vec(self):
        """Per-(model,series,state) forecast discount factor ``(1-disc)/disc``,
        shape ``(p, k)`` (the discount VECTOR consumed by ``dlm_uv_fcast_H``).

        Always the static grid: disc = disc_rates * disc_rates_damped (byte-
        identical to the original static path). The signed-error monitor's
        variance injection is a one-off applied DURING filtering, so by the
        forecast origin it already lives in the state covariance ``C``; the
        h-step projection forward just uses the plain grid forgetting (the
        injection is not projected over the horizon).
        """
        disc = self.disc_rates * self.disc_rates_damped
        return (1.0 - disc) / disc

    def forecast(self, h):
        """h-step forecast for every (model, series); returns a ForecastBundle
        whose ``loc`` and ``var`` are each shaped ``(nm, q, h)``.

        Uses a nested vmap — inner over series, outer over models — so the
        model-only operands (GH, FH, variance/mult-comp constants) are SHARED
        across the q series (in_axes=None) rather than replicated. The flat
        path (``_forecast_flat``) materialised GH as (nm*q, h, k, k)
        (~2.2 MB/series, ~33 GB unbatched), which was both the forecast-stage
        memory blow-up and, since the step is memory-bandwidth-bound, ~half its
        streamed bytes. The nested form never builds it. Numerically identical
        to ``_forecast_flat`` (guarded by tests/test_nested_forecast_equiv.py).
        """
        nm, q, k = self.nm, self.q, self.k

        # Per-series quantities, built on the flat (p, ...) layout then folded
        # to (nm, q, ...). p is model-major (see _stack_and_broadcast), so both
        # the reshape and the [:, 0] model-constant extraction below are valid.
        init = {key: self.dlm_state[key] for key in ["m", "s", "nu"]}
        init["C"] = axis0dot(
            self.dlm_state["Z"].swapaxes(-1, -2), self.dlm_state["Z"],  # C = Z' Z
        )
        disc_factor = self._forecast_disc_factor_vec()    # (p, k) vector

        def _to_mq(a):                       # (p, ...) -> (nm, q, ...)
            return a.reshape((nm, q) + a.shape[1:])

        init_mq = {key: _to_mq(val) for key, val in init.items()}

        # Model-only operands kept compact (nm, ...) and shared across the q
        # series by the inner vmap (in_axes=None) — never broadcast to (p, ...).
        # disc_factor is per-state and series-invariant, so it joins them.
        disc_factor_m = _to_mq(disc_factor)[:, 0]            # (nm, k)
        variance_disc = _to_mq(self.variance_disc)[:, 0]      # (nm, 1)
        variance_power = _to_mq(self.variance_power)[:, 0]    # (nm, 1)
        mult_comps = _to_mq(self.mult_comps)[:, 0]            # (nm, k)
        DH = {
            "FH": self.F[:, jnp.newaxis, :] * jnp.ones((1, h, 1)),  # (nm, h, k)
            "GH": self.GH[:, 0, :h, ...],                          # (nm, h, k, k)
        }

        f, qv = _jvdlm_uv_fcast_H_nested_fq(
            disc_factor_m, variance_disc, variance_power, mult_comps, init_mq, DH,
        )
        return ForecastBundle(f.reshape(nm, q, h), qv.reshape(nm, q, h))

    def forecast_iterated(self, h, seed_lags):
        """Iterated-expectations h-step forecast for an AR/TVAR universe.

        The standard :meth:`forecast` assumes F is known over the horizon, which
        is false when the regression tail carries the model's own lagged
        observations. This routine reuses the ordinary *state* propagation
        (``a_j = G^j m``, ``R_j`` — correct because G is block-diagonal and the
        AR block is the identity) and then redoes the *observation* step
        sequentially via :func:`iterated_obs_forecast`, feeding each horizon's
        predictive mean forward as the next lag.

        With ``n_regressors == 0`` it returns exactly the same ``(f, q)`` as
        :meth:`forecast` (the lag tail is empty, the scan reduces to the static-F
        kernel), so it is a strict generalisation.

        Parameters
        ----------
        h : int
            Forecast horizon.
        seed_lags : array, shape ``(n_series, n_regressors)``
            The last ``n_regressors`` observations per series, most-recent-first
            (``[y_t, y_{t-1}, ...]``), seeding the lags of the first horizons.
            Broadcast across models (all models share the series' y).

        Returns
        -------
        ForecastBundle
            Predictive ``loc`` and ``var`` (variance), each shaped ``(nm, q, h)``.
        """
        nm, q, k, p = self.nm, self.q, self.k, self.p
        n_reg = self.n_regressors

        # --- state propagation, reused verbatim from the flat forecast kernel.
        init = {key: self.dlm_state[key] for key in ["m", "s", "nu"]}
        # Stationarise the AR coefficients before forecasting: reflect any
        # non-stationary roots inside the unit circle so the iterated variance
        # recursion Q_j = s + sum (phi_k^2 + Var phi_k) Q_{j-k} cannot diverge
        # for near-unit-root / explosive coefficients. numpy over only the TVAR
        # rows (reg_mask) — not inside a lax.scan here, so the C-loop root solve
        # is fine — covering AutoFFS.predict, AutoFFSUniverse.forecast and direct
        # callers. G_ar = I, so the flipped block propagates unchanged.
        if AR_STATIONARITY_EPS is not None and n_reg > 0:
            rm = np.asarray(
                getattr(self, "reg_mask", np.ones(p, dtype=bool))
            ).astype(bool)
            if rm.any():
                reg0 = k - n_reg
                m_np = np.asarray(init["m"]).copy()           # (p, k)
                sub = m_np[rm]
                sub[:, reg0:] = _flip_ar_np(sub[:, reg0:], AR_STATIONARITY_EPS)
                m_np[rm] = sub
                init["m"] = jnp.asarray(m_np)
        init["C"] = axis0dot(
            self.dlm_state["Z"].swapaxes(-1, -2), self.dlm_state["Z"],  # C = Z' Z
        )
        disc_factor = (
            1 - self.disc_rates * self.disc_rates_damped
        ) / (self.disc_rates * self.disc_rates_damped)            # (p, k) vector
        Ft = self.F.reshape(nm, 1, k)[jnp.newaxis, ...] * jnp.ones((h, 1, q, 1))
        DH = {
            "FH": jnp.moveaxis(Ft, 0, 2).reshape(p, h, k),
            "GH": (self.GH[:, :, :h, ...] * jnp.ones((1, q, 1, 1, 1))).reshape(
                p, h, k, k
            ),
        }
        prop = _jvdlm_uv_fcast_H(
            disc_factor, self.variance_disc, self.variance_power,
            self.mult_comps, init, DH
        )
        aH, RH = prop["m"], prop["C"]                 # (p, h, k), (p, h, k, k)

        # --- per-(model, series) iterated observation step.
        s0 = jnp.ravel(self.dlm_state["s"])           # (p,)
        vd = jnp.ravel(self.variance_disc)            # (p,)
        vp = jnp.ravel(self.variance_power)           # (p,)
        Fp = (self.F.reshape(nm, 1, k) * jnp.ones((1, q, 1))).reshape(p, k)
        # seed_lags (q, n_reg) -> model-major (p, n_reg)
        seed = (
            jnp.asarray(seed_lags)[jnp.newaxis] * jnp.ones((nm, 1, 1))
        ).reshape(p, n_reg)

        # Per-model gate: structural models packed alongside TVAR (sharing the
        # tail slots as trend/seasonal state) get the standard forecast.
        is_reg = getattr(
            self, "reg_mask", jnp.ones(p, dtype=bool)
        )

        def _one(a, R, s, d, pw, m, F, sl, ir):
            return iterated_obs_forecast(a, R, s, d, pw, m, F, n_reg, sl, ir)

        f, qv = vmap(_one)(aH, RH, s0, vd, vp, self.mult_comps, Fp, seed, is_reg)
        return ForecastBundle(f.reshape(nm, q, h), qv.reshape(nm, q, h))

    def forecast_exog(self, h, future_mean, future_var=None):
        """h-step forecast with *supplied* exogenous regressors over the horizon
        — the deterministic / stochastic-regressor analogue of
        :meth:`forecast_iterated`.

        The regression tail's future values are provided by the caller rather
        than forecast by the model (as the AR path does). With
        ``future_var=None`` the regressors are treated as *known* (predictive
        variance = observation noise + coefficient uncertainty). Pass
        ``future_var`` to propagate a *stochastic* regressor's own forecast
        variance through the predictive variance — the same second-moment term
        the AR path applies to its self-forecast lags (see
        :func:`_regressor_var_prop`). No stationarity guard is needed: there is
        no self-feedback, so the variance recursion cannot diverge.

        With ``n_regressors == 0`` there is no tail to fill; call
        :meth:`forecast` instead.

        Parameters
        ----------
        h : int
            Forecast horizon.
        future_mean : array, shape ``(h, n_series, n_regressors)``
            Future regressor values per horizon. Shared across all models.
        future_var : array, optional, shape ``(h, n_series, n_regressors)``
            Future regressor variances per horizon. Defaults to zeros (known X).

        Returns
        -------
        ForecastBundle
            Predictive ``loc`` and ``var`` (variance), each shaped ``(nm, q, h)``.
        """
        nm, q, k, p = self.nm, self.q, self.k, self.p
        n_reg = self.n_regressors
        if n_reg == 0:
            raise ValueError(
                "forecast_exog requires a regression tail (n_regressors > 0); "
                "use forecast for a purely structural universe."
            )

        # --- state propagation: identical to forecast_iterated (G's regression
        # block is the damped identity, so the coefficients evolve under their
        # own discount; aH/RH come straight from the ordinary forecast kernel).
        init = {key: self.dlm_state[key] for key in ["m", "s", "nu"]}
        init["C"] = axis0dot(
            self.dlm_state["Z"].swapaxes(-1, -2), self.dlm_state["Z"],  # C = Z' Z
        )
        disc_factor = (
            1 - self.disc_rates * self.disc_rates_damped
        ) / (self.disc_rates * self.disc_rates_damped)            # (p, k) vector
        Ft = self.F.reshape(nm, 1, k)[jnp.newaxis, ...] * jnp.ones((h, 1, q, 1))
        DH = {
            "FH": jnp.moveaxis(Ft, 0, 2).reshape(p, h, k),
            "GH": (self.GH[:, :, :h, ...] * jnp.ones((1, q, 1, 1, 1))).reshape(
                p, h, k, k
            ),
        }
        prop = _jvdlm_uv_fcast_H(
            disc_factor, self.variance_disc, self.variance_power,
            self.mult_comps, init, DH
        )
        aH, RH = prop["m"], prop["C"]                 # (p, h, k), (p, h, k, k)

        # --- per-(model, series) observation step with supplied regressors.
        s0 = jnp.ravel(self.dlm_state["s"])           # (p,)
        vd = jnp.ravel(self.variance_disc)            # (p,)
        vp = jnp.ravel(self.variance_power)           # (p,)
        Fp = (self.F.reshape(nm, 1, k) * jnp.ones((1, q, 1))).reshape(p, k)

        # future_mean/var (h, n_series, n_reg) -> model-major (p, h, n_reg),
        # matching the (nm, q, h, ...) ordering of aH and DH["FH"].
        def _to_p_h_nreg(A):
            A = jnp.asarray(A)
            if A.ndim != 3 or A.shape[0] != h or A.shape[-1] != n_reg:
                raise ValueError(
                    f"future regressor array must have shape (h={h}, n_series, "
                    f"{n_reg}); got {tuple(A.shape)}."
                )
            A = jnp.concatenate(
                [A, jnp.zeros((h, self.npad, n_reg), A.dtype)], axis=1
            )                                                  # (h, q, n_reg)
            A = A[:, jnp.newaxis] * jnp.ones((1, nm, 1, 1))    # (h, nm, q, n_reg)
            return jnp.moveaxis(A, 0, 2).reshape(p, h, n_reg)  # (nm, q, h, n_reg)

        x_traj = _to_p_h_nreg(future_mean)
        v_traj = (
            jnp.zeros_like(x_traj) if future_var is None
            else _to_p_h_nreg(future_var)
        )

        is_reg = getattr(self, "reg_mask", jnp.ones(p, dtype=bool))

        def _one(a, R, s, d, pw, m, F, xt, vt, ir):
            return exog_obs_forecast(a, R, s, d, pw, m, F, n_reg, xt, vt, ir)

        f, qv = vmap(_one)(
            aH, RH, s0, vd, vp, self.mult_comps, Fp, x_traj, v_traj, is_reg
        )
        return ForecastBundle(f.reshape(nm, q, h), qv.reshape(nm, q, h))

    def _forecast_flat(self, h):
        """Reference flat-vmap forecast: replicates the model operands across
        the q series and maps a single flat (nm*q) vmap. Retained as the
        regression target for ``forecast`` (test_nested_forecast_equiv). Prefer
        ``forecast`` — this materialises the (nm*q, h, k, k) GH tensor.
        """
        Ft = self.F.reshape(self.nm, 1, self.k)[jnp.newaxis, ...] * jnp.ones(
            (h, 1, self.q, 1)
        )

        init = {key: self.dlm_state[key] for key in ["m", "s", "nu"]}
        init["C"] = axis0dot(
            self.dlm_state["Z"].swapaxes(-1, -2), self.dlm_state["Z"],  # C = Z' Z
        )

        disc_factor = self._forecast_disc_factor_vec()    # (p, k) vector

        DH = {
            "FH": jnp.moveaxis(Ft, 0, 2).reshape(self.p, h, self.k),
            "GH": (self.GH[:, :, :h, ...] * jnp.ones((1, self.q, 1, 1, 1))).reshape(
                self.p, h, self.k, self.k
            ),
        }
        fc_model = _jvdlm_uv_fcast_H(
            disc_factor,
            self.variance_disc,
            self.variance_power,
            self.mult_comps,
            init,
            DH,
        )
        return ForecastBundle(
            fc_model["f"].reshape(self.nm, self.q, h),
            fc_model["q"].reshape(self.nm, self.q, h),
        )


def _format_yt(yt, n_pad, n_models, n_flat):
    return pad(yt, n_pad) * jnp.ones((n_models, 1)).reshape(n_flat)


# Error-monitor (signed-bias intervention) constants. The monitor tracks an EWMA
# of the SIGNED standardised one-step error with a 10-step half-life; the trigger
# fires when |S| exceeds tau * (its stationary SD). For a zero-mean unit-variance
# error the EWMA's stationary variance is alpha/(2-alpha), so MONITOR_S_SD is its
# SD and the threshold tau*MONITOR_S_SD reads directly as "tau standard deviations"
# (tau, the reused monitor slot, defaults to 3). See _multi_fwd_filter_step.
MONITOR_HALFLIFE = 10
MONITOR_ALPHA = 1.0 - 0.5 ** (1.0 / MONITOR_HALFLIFE)            # ~0.0670 EWMA weight
MONITOR_S_SD = (MONITOR_ALPHA / (2.0 - MONITOR_ALPHA)) ** 0.5    # ~0.1862 stationary SD


def _multi_fwd_filter_step(dlm_state, multi_params, y_t, warmup_flag=0.0,
                           regressors=None, reg_mask=None, tau=None, alpha=0.3):
    """Single-step multi-model filter update.

    Parameters
    ----------
    dlm_state : dict
        Current state (m, UC, SC, s, nu).
    multi_params : tuple
        Static packed-universe parameters, in order:
        ``(F, G, nm, k, q, p, disc_rates, disc_rates_damped, monitor_inject,
        variance_disc, variance_power, mult_comps)``.
    y_t : array
        Observations at time t.
    warmup_flag : scalar, default 0.0
        If 1.0, this step is treated as a warmup step: ``disc_mtx`` is
        zeroed so the system-noise contribution W = 0 for this step.
        The state's covariance evolves only via G, and the observation
        update proceeds normally. If 0.0 (the default), this is a
        no-op and the step behaves exactly as before.
    regressors : array, optional
        Per ``(model, series)`` regressor values for this step, shape
        ``(p, n_reg)`` — these fill the **last ``n_reg`` slots** of ``Ft``
        (the regression tail; ``self.F`` carries zeros there). For AR/TVAR
        models the values are the series' lagged ``y``. ``None`` (the
        default) leaves ``Ft`` as the static structural F — bit-exact with
        the prior structural-only behaviour.
    reg_mask : array, optional
        Per ``(model, series)`` boolean of shape ``(p,)`` selecting which
        models actually have an AR/regression tail. In a mixed universe
        (structural + TVAR) the lags must land only in the TVAR models;
        ``where(reg_mask, regressors, Ft_tail)`` leaves structural models' F
        untouched. ``None`` (a homogeneous regression multi) fills every
        model's tail.
    """
    (
        F,
        G,
        nm,
        k,
        q,
        p,
        disc_rates,
        disc_rates_damped,
        monitor_inject,
        variance_disc,
        variance_power,
        mult_comps,
    ) = multi_params

    Ft = (F.reshape(nm, 1, k) * jnp.ones((1, q, 1))).reshape(p, k)
    Gt = (G.reshape(nm, 1, k, k) * jnp.ones((1, q, 1, 1))).reshape(p, k, k)

    if regressors is not None:
        # Fill the regression tail's F-entries (last n_reg slots) with the
        # supplied per-(model, series) regressor values. In a mixed universe
        # reg_mask gates the fill so structural models keep their static F.
        n_reg = regressors.shape[-1]
        if reg_mask is None:
            filled = regressors
        else:
            filled = jnp.where(reg_mask[:, jnp.newaxis], regressors, Ft[:, k - n_reg:])
        Ft = Ft.at[:, k - n_reg:].set(filled)

    data_t = {"F": Ft, "G": Gt, "y": y_t}

    # S is the per-(model,series) carried error-monitor EWMA (signed). Pulled up
    # here because the trigger (computed from S_{t-1}) drives BOTH the disc
    # injection and the S reset below.
    S_prev = dlm_state["S"]
    if tau is None:
        disc_mtx = vadapt_discount_z(
            dlm_state["Z"],
            dlm_state["m"],
            disc_rates,
            disc_rates_damped,
            0.5,
        )
        inject_flag = None
    else:
        # Signed-error monitor intervention (opt-in via the `monitor`/tau slot,
        # tau = SD multiple, default 3). The trigger fires when the carried
        # signed-error EWMA |S_{t-1}| exceeds tau * stationary-SD. On a triggered
        # step the grid discount is revised DOWN to ``monitor_inject`` (0.5
        # level/trend, 0.8 seasonal) — a one-off variance injection — and fed as
        # the grid INTO ``adapt_discount`` (the guard), which still has the last
        # word: an already-uncertain state has the injection suppressed, so the
        # state variance cannot diverge. Non-triggered steps are bit-identical to
        # the static path. Causal — uses S_{t-1}; S refreshed/reset below.
        inject_flag = jnp.abs(S_prev) > (tau * MONITOR_S_SD)        # (p,)
        grid = jnp.where(
            inject_flag[:, jnp.newaxis],
            jnp.minimum(disc_rates, monitor_inject),
            disc_rates,
        )
        disc_mtx = vadapt_discount_z(
            dlm_state["Z"],
            dlm_state["m"],
            grid,
            disc_rates_damped,
            0.5,
        )

    # Warmup: zero out W for this step. ``disc_mtx`` is the per-model
    # ``diag((1-disc)/disc)`` matrix; multiplying by (1 - warmup_flag)
    # gives W = 0 for warmup steps, identity behaviour otherwise.
    disc_mtx = disc_mtx * (1.0 - warmup_flag)

    # The inner SVD step is S-agnostic (its vmap in_axes lists fixed state keys),
    # so strip S out for the call and re-attach it after — keeps the hot path
    # byte-for-byte unchanged.
    inner_state = {key: val for key, val in dlm_state.items() if key != "S"}
    inner_state, model = _jvdlm_uv_fwd_qr_step(
        disc_mtx, variance_disc, variance_power, mult_comps, inner_state, data_t
    )

    if tau is None:
        inner_state["S"] = S_prev                       # carry unchanged
    else:
        # Refresh the per-series EWMA of the SIGNED standardised one-step error
        # for the next step's trigger (causal: this step used S_{t-1}). The EWMA
        # weight is MONITOR_ALPHA (10-step half-life), independent of the DMA/FFS
        # alpha. On a triggered (injected) step the monitor resets to 0; held
        # fixed on warmup so the diffuse-prior transient doesn't poison S.
        f_flat = model["f"].reshape(-1)                 # (p,1) -> (p,) to match y
        q_flat = model["q"].reshape(-1)
        z = (data_t["y"] - f_flat) * finite_inv(jnp.sqrt(q_flat))   # signed
        S_new = MONITOR_ALPHA * z + (1.0 - MONITOR_ALPHA) * S_prev
        S_new = jnp.where(inject_flag, 0.0, S_new)      # reset to 0 on trigger
        inner_state["S"] = jnp.where(warmup_flag > 0.0, S_prev, S_new)
    dlm_state = inner_state

    if dlm_state["m"].ndim == 1:
        dlm_state["m"] = dlm_state["m"][:, jnp.newaxis]

    return dlm_state, model, model["f"].reshape(nm, q), model["q"].reshape(nm, q)


# -----------------------------------------------------------------------------
# discount adaptation
# -----------------------------------------------------------------------------

# West & Harrison *block* (congruence) discount — the model's discount geometry,
# applied unconditionally. The covariance evolution is ``R = B P B`` with
# ``B = diag(1/sqrt(delta))`` (so ``R_ij = P_ij / sqrt(delta_i delta_j)``), which
# for a common within-block delta reduces to ``P/delta`` and PRESERVES the prior
# correlation structure. Filter: scale the columns of the prior square-root by
# 1/sqrt(delta). Forecast: the W&H-standard CONSTANT origin-W added additively
# over the horizon (``W = B C B - C``; the discount is NOT re-applied each
# forecast step). Wired through the multi-model FILTER (``dlm_uv_fwd_svd_step`` /
# ``dlm_uv_fwd_qr_step``, hence single-uv ``update``) and the h-step FORECAST
# (``dlm_uv_fcast_H``) — the AutoFFS / AutoFFSUniverse CV path used by M1M/M4/M5.


def adapt_discount(UC, SC, m, disc_default, disc_damped, adapt):
    state_var = jnp.diag(UC @ jnp.diag(SC**2) @ UC.T)
    info = jnp.abs(m) * finite_inv(jnp.sqrt(state_var))
    disc = (disc_default + (1 - disc_default) * jnp.exp(-adapt * info)) * disc_damped
    return jnp.diag((1 - disc) / disc)


def _adapt_discount_z(Z, m, disc_default, disc_damped, adapt):
    """``adapt_discount`` on the QR root ``Z`` (``C = Z'Z``).

    The only state-covariance quantity ``adapt_discount`` uses is
    ``diag(C)`` = per-state variances, which on the root is
    ``diag(Z'Z) = (Z**2).sum(0)`` — a column square-sum, no SVD. Bit-identical
    to ``adapt_discount`` (verified in the Phase A parity gate).
    """
    state_var = jnp.sum(Z ** 2, axis=0)
    info = jnp.abs(m) * finite_inv(jnp.sqrt(state_var))
    disc = (disc_default + (1 - disc_default) * jnp.exp(-adapt * info)) * disc_damped
    return jnp.diag((1 - disc) / disc)


# across series (q) / across models (nm)
vadapt_discount = vmap(adapt_discount, in_axes=(0, 0, 0, 0, 0, None))
# Root form for the QR-native packed filter: maps Z, m, disc, disc_damped over
# the model axis (adapt fixed). state_var = (Z**2).sum(0) — no per-step SVD.
vadapt_discount_z = vmap(_adapt_discount_z, in_axes=(0, 0, 0, 0, None))
vvadapt_discount = vmap(vadapt_discount, in_axes=(1, 1, 1, None, None, None))


# Error-monitoring discount (continuous, standardised-error analogue of West &
# Harrison automatic monitoring). S is the EWMA of |standardised one-step error|,
# tau the per-(model,series) sensitivity threshold (the reused `monitor` slot),
# d_min the per-state floor (the `disc_rates` passed as now). In calm (S <= tau)
# the envelope is 1 -> disc = disc_damped (structural damping only); as the EWMA
# exceeds tau, disc decays toward d_min * disc_damped (emergency forgetting).
EWMA_ERR_ALPHA = 0.3        # fast EWMA weight on |z|; v1 constant


def adapt_discount_err(S, tau, d_min, disc_damped):
    excess = jnp.maximum(S - tau, 0.0)                       # scalar per (model, series)
    disc = (d_min + (1 - d_min) * jnp.exp(-excess)) * disc_damped   # (k,) per state
    return jnp.diag((1 - disc) / disc)


# vmaps over the flattened (model x series) axis, mirroring vadapt_discount's
# output shape (p, k, k); S, tau are (p,), d_min/disc_damped are (p, k).
vadapt_discount_err = vmap(adapt_discount_err, in_axes=(0, 0, 0, 0))


# -----------------------------------------------------------------------------
# observation / variance helpers
# -----------------------------------------------------------------------------


def bayes_factor(error, sigma1, sigma2):
    log_odds = (
        jnp.log(sigma2 / sigma1)
        + (error**2 / (2 * sigma2**2))
        - (error**2 / (2 * sigma1**2))
    )
    return jnp.exp(log_odds)


def var_scale_fn(f, var_power, var_floor=1e-12):
    """Observation variance scale as a power of the mean.

    ``var_power``: 0 → log transform, 0.5 → sqrt, 0.25 → compound Poisson,
    1 → no transform.
    """
    return jnp.maximum(jnp.abs(f) ** (2 * (1 - var_power)), var_floor)


def F_fn(F, a, mc, nmc):
    """Observation function combining additive and multiplicative components.

    ``mc``/``nmc`` are binary indicators of multiplicative / non-multiplicative
    state components.
    """
    return (a * nmc) @ F * (1 + (a * mc) @ F)


F_grad = grad(F_fn, 1)
F_val_n_grad = value_and_grad(F_fn, 1)
vF_val_n_grad = vmap(F_val_n_grad, in_axes=(0, 0, None, None))


# -----------------------------------------------------------------------------
# single-step filter (SVD form)
# -----------------------------------------------------------------------------


# Reconditioning floor for the SVD failure guard (see dlm_uv_fwd_svd_step): on a
# non-finite observation-update SVD the reverted prior covariance has its small
# singular values floored to ``max(SC) * _SVD_RCOND`` (condition cap 1/_SVD_RCOND)
# so the next step can update instead of refiring. Only ever applied on an actual
# failure, so well-conditioned steps are bit-identical.
_SVD_RCOND = 1e-4


def dlm_uv_fwd_svd_step(disc_factor, variance_disc, var_power, mc, Stm1, Dt):
    a = Stm1["m"] @ Dt["G"].T
    F = F_grad(Dt["F"], a, mc, 1 - mc)
    f = F_fn(Dt["F"], a, mc, 1 - mc)
    error = Dt["y"] - f

    var_scale = var_scale_fn(f, var_power)
    s = Stm1["s"] * var_scale

    ignore_obs = jnp.isnan(error)

    # time update
    SCUCGT = jnp.diag(Stm1["SC"]) @ Stm1["UC"].T @ Dt["G"].T
    # West & Harrison block (congruence) discount: R = B (G C G') B with
    # B = diag(1/√δ). SCUCGT is a square-root factor of P = G C G'
    # (P = SCUCGT.T @ SCUCGT), so B P B = (SCUCGT @ B).T (SCUCGT @ B): just scale
    # the COLUMNS of the root by 1/√δ. The per-state δ is recovered from the
    # diagonal of the supplied factor matrix: diag(disc_factor) = (1-δ)/δ, so
    # 1 + (1-δ)/δ = 1/δ and b = sqrt(1/δ). The congruence is exact and PSD.
    b = jnp.sqrt(1.0 + jnp.diag(disc_factor))          # (k,) = 1/sqrt(delta)
    NR = SCUCGT * b[jnp.newaxis, :]                     # column scaling
    # NW is a diagnostic-only model output (unused downstream); report the increment.
    NW = NR - SCUCGT
    UNR, SNR, VNRT = svd_sqrt(NR)
    UR = VNRT.T
    SR = SNR**2

    # observation update
    R = NR.T @ NR
    q = s + F.T @ R @ F
    A = R @ F / q
    p = UR.shape[0]
    SRUSIlAF = (
        jnp.diag(SR) @ UR.T @ (jnp.eye(p) - (jnp.atleast_2d(F).T @ jnp.atleast_2d(A)))
    )
    NC = jnp.concatenate([SRUSIlAF, jnp.sqrt(s) * jnp.atleast_2d(A)], 0)
    UNC, SNC, VNCT = svd_sqrt(NC)
    UC = VNCT.T
    SC = SNC**2
    m = jnp.where(ignore_obs, a, a + A * error)

    # variance update
    d = s * Stm1["nu"]
    d = jnp.where(ignore_obs, d, variance_disc * d + (error**2) * s / q)
    nu = jnp.where(ignore_obs, Stm1["nu"], variance_disc * Stm1["nu"] + 1.0)
    s_t = d / nu
    SC = SC * jnp.sqrt(s_t / s)
    s_t = jnp.where(ignore_obs, Stm1["s"], s_t / var_scale)

    # Missing observation: the posterior IS the prior, so the covariance takes
    # R_t = G C G'/δ — whose root in this convention (C = UC diag(SC²) UC') is
    # exactly the already-computed (UR, SR). Reverting to the incoming
    # (Stm1["UC"], Stm1["SC"]) = C_{t-1} dropped the step's discount forgetting
    # and, under non-identity G, left the mean rotated while the covariance was
    # not. No rescale: s_t reverts to Stm1["s"] below, and (UR, SR) already
    # carries C_{t-1}'s scale.
    UC = jnp.where(ignore_obs, UR, UC)
    SC = jnp.where(ignore_obs, SR, SC)

    # Robustness guard: a severely ill-conditioned observation update can make
    # svd_sqrt(NC) return non-finite (LAPACK gesdd convergence failure on a
    # near-singular sqrt-covariance — seen on high-frequency multiplicative-error
    # models at large levels). Detect it and skip the observation update for this
    # (model, series) step, reverting to the time-projected prior mean ``a`` with
    # a RECONDITIONED prior covariance (small singular values floored relative to
    # the largest) so the next step can update rather than refire. Fires ONLY on
    # an actual non-finite result, so well-conditioned steps stay bit-identical.
    svd_failed = ~(jnp.all(jnp.isfinite(SC)) & jnp.all(jnp.isfinite(UC))
                   & jnp.all(jnp.isfinite(m)))
    SC_recond = jnp.maximum(Stm1["SC"], jnp.max(Stm1["SC"]) * _SVD_RCOND)
    m = jnp.where(svd_failed, a, m)
    UC = jnp.where(svd_failed, Stm1["UC"], UC)
    SC = jnp.where(svd_failed, SC_recond, SC)
    s_t = jnp.where(svd_failed, Stm1["s"], s_t)
    nu = jnp.where(svd_failed, Stm1["nu"], nu)

    return (
        {"m": m, "UC": UC, "SC": SC, "s": s_t, "nu": nu},
        {
            "m": m,
            "UC": UC,
            "SC": SC,
            "A": A,
            "s": s_t,
            "a": a,
            "UR": UR,
            "SR": SR,
            "f": f,
            "q": q,
            "NW": NW,
            "nu": nu,
            "v_sys": F.T @ NR.T @ NR @ F,
            "spec_variance": s_t * var_scale,
        },
    )


vdlm_uv_fwd_svd_step = vmap(dlm_uv_fwd_svd_step, in_axes=(0, 0, 0, 0, 0, 0))
# Module-level jitted version used by multi_model_dlm.fwd_filter so we don't
# re-jit on every call.
_jvdlm_uv_fwd_svd_step = jit(vdlm_uv_fwd_svd_step)


# -----------------------------------------------------------------------------
# single-step filter (QR square-root form)
# -----------------------------------------------------------------------------
#
# Autodiff-STABLE sibling of ``dlm_uv_fwd_svd_step``. Same statistical model
# (shares ``F_fn`` / ``F_grad`` / ``var_scale_fn`` and the West & Harrison
# variance learning), but it carries the covariance ROOT ``Z`` directly
# (``C = Z.T @ Z``) and uses a QR factorisation in place of ``svd_sqrt``. SVD
# gradients are ill-conditioned near repeated/zero singular values, so forward-
# or reverse-mode autodiff through the SVD step is unstable; the QR step keeps
# the tangent √κ-conditioned. This is the step an online gradient-based
# discount learner differentiates through.
#
# It is a PARALLEL forward form, not a drop-in for the SVD scan: the state layout
# differs — ``Stm1 = {"m", "Z", "s", "nu"}`` with ``Z`` a (k, k) root, NOT the
# SVD step's ``{"m", "UC", "SC", "s", "nu"}``. Predictive moments (f, q, nu) and
# the log-score gradient w.r.t. the discounts are bit-exact to the SVD kernel;
# see ``tests/test_qr_kernel.py``.
#
# Uses the West & Harrison block (congruence) discount exactly as the SVD step:
# ``R = B P B`` with ``B = diag(1/sqrt(delta))``. The observation update depends
# on the prior-covariance root ``NR`` only through ``R = NR.T @ NR``.
#
# ``disc_factor`` is the same (k, k) diagonal matrix the SVD step takes
# (``diag((1-δ)/δ)``); ``variance_disc`` (β), ``var_power``, ``mc``, ``Dt`` are
# identical. A missing observation (NaN ``y``) projects the mean and freezes the
# covariance / variance, mirroring the SVD step's ``ignore_obs`` branch (inert on
# finite data, so the bit-exactness gate is unaffected). Unlike the SVD step it
# carries no ``svd_sqrt`` failure guard — the QR root is PSD by construction and
# far better conditioned, so there is nothing to recondition.


def dlm_uv_fwd_qr_step(disc_factor, variance_disc, var_power, mc, Stm1, Dt):
    m, Z = Stm1["m"], Stm1["Z"]
    s, nu = Stm1["s"], Stm1["nu"]
    k = Dt["F"].shape[0]

    a = m @ Dt["G"].T
    F = F_grad(Dt["F"], a, mc, 1 - mc)          # effective obs vector ∂f/∂a
    f = F_fn(Dt["F"], a, mc, 1 - mc)
    error = Dt["y"] - f
    ignore_obs = jnp.isnan(error)

    var_scale = var_scale_fn(f, var_power)
    s_pred = s * var_scale                       # prior obs variance this step

    # time update: NR is a root of the prior state covariance R = GCG'/δ.
    ZGT = Z @ Dt["G"].T                          # (k, k) root of P = GCG'  (P = ZGT.T @ ZGT)
    # Congruence R = B P B, B = diag(1/sqrt(delta)): scale root columns by 1/sqrt(delta).
    b = jnp.sqrt(1.0 + jnp.diag(disc_factor))          # (k,) = 1/sqrt(delta)
    NR = ZGT * b[jnp.newaxis, :]                       # (k, k)

    # observation update (scale-free filtering recursion + Joseph posterior root)
    NRF = NR @ F
    q = s_pred + NRF @ NRF
    A = (NR.T @ NRF) / q
    m_upd = a + A * error
    Pre = jnp.concatenate(
        [NR @ (jnp.eye(k) - jnp.outer(F, A)),
         jnp.sqrt(s_pred) * A[jnp.newaxis, :]], axis=0)
    _, Zp = jnp.linalg.qr(Pre)                   # (k, k) posterior root at scale s_pred

    # variance learning (West & Harrison); nu is deterministic in the discounts
    nu_upd = variance_disc * nu + 1.0
    s_upd = (variance_disc * s_pred * nu + error**2 * s_pred / q) / nu_upd
    Z_upd = Zp * jnp.sqrt(s_upd / s_pred)        # rescale covariance to the new variance
    s_upd_norm = s_upd / var_scale               # store level-normalised s

    # Missing observation: the posterior IS the prior. The mean takes the
    # time-evolved ``a``; the covariance must likewise take ``R_t = G C G'/δ``,
    # whose root is ``NR``. The *incoming* root ``Z`` (= C_{t-1}) will not do,
    # on two counts: it omits that step's discount forgetting, so a gap of k
    # steps loses δ^-k of variance inflation and leaves the model overconfident
    # across the gap; and under non-identity G it leaves the mean rotated while
    # the covariance is not, so C[i,j] no longer describes m[i], m[j].
    # No rescale: nothing about the variance changes on a skipped step
    # (s_new = s, nu_new = nu), and NR already carries C_{t-1}'s scale.
    m_new = jnp.where(ignore_obs, a, m_upd)
    Z_new = jnp.where(ignore_obs, NR, Z_upd)
    s_new = jnp.where(ignore_obs, s, s_upd_norm)
    nu_new = jnp.where(ignore_obs, nu, nu_upd)

    return (
        {"m": m_new, "Z": Z_new, "s": s_new, "nu": nu_new},
        {
            "m": m_new,
            "Z": Z_new,
            "A": A,
            "s": s_new,
            "a": a,
            "f": f,
            "q": q,
            "nu": nu_new,
            "v_sys": NRF @ NRF,
            "spec_variance": s_new * var_scale,
        },
    )


vdlm_uv_fwd_qr_step = vmap(dlm_uv_fwd_qr_step, in_axes=(0, 0, 0, 0, 0, 0))
_jvdlm_uv_fwd_qr_step = jit(vdlm_uv_fwd_qr_step)


# -----------------------------------------------------------------------------
# forecasting step (non-SVD, uses covariance directly)
# -----------------------------------------------------------------------------


def dlm_uv_fcast_step(W, variance_disc, var_power, mc, Stm1, Dt):
    a = Stm1["m"] @ Dt["G"].T
    f, F = F_val_n_grad(Dt["F"], a, mc, 1 - mc)
    var_scale = var_scale_fn(f, var_power)
    s = Stm1["s"] * var_scale
    R = Dt["G"] @ Stm1["C"] @ Dt["G"].swapaxes(-1, -2) + W
    d = s * Stm1["nu"] * variance_disc
    nu = variance_disc * Stm1["nu"]
    s = d / nu
    q = s + F.T @ R @ F
    return (
        {"m": a, "C": R, "s": s / var_scale, "nu": nu},
        {"m": a, "C": R, "s": s / var_scale, "q": q, "nu": nu, "f": f},
    )


vdlm_uv_fcast_step = vmap(dlm_uv_fcast_step, in_axes=(0, None, None, None, 0, 0))


def dlm_uv_fcast_step_exog(W, variance_disc, var_power, mc, n_reg, Stm1, Dt):
    """:func:`dlm_uv_fcast_step` with an added stochastic-regressor variance term.

    Identical to :func:`dlm_uv_fcast_step` for the predictive mean and the
    state/observation variance, but adds the supplied regressor's own forecast
    uncertainty: ``sum_k (phi_k^2 + Var(phi_k)) Var(x_k)`` from the per-horizon
    regressor variance ``Dt['v']`` (shape ``(n_reg,)``), via
    :func:`_regressor_var_prop`. So a forecast made on *uncertain* future
    regressors carries that uncertainty into ``q``. With ``Dt['v'] == 0`` it is
    byte-for-byte :func:`dlm_uv_fcast_step` (the known/deterministic regressor
    case). The regression coefficients are the state tail ``a[k - n_reg:]``.
    """
    a = Stm1["m"] @ Dt["G"].T
    f, F = F_val_n_grad(Dt["F"], a, mc, 1 - mc)
    var_scale = var_scale_fn(f, var_power)
    s = Stm1["s"] * var_scale
    R = Dt["G"] @ Stm1["C"] @ Dt["G"].swapaxes(-1, -2) + W
    d = s * Stm1["nu"] * variance_disc
    nu = variance_disc * Stm1["nu"]
    s = d / nu
    reg0 = a.shape[-1] - n_reg
    prop = _regressor_var_prop(a[reg0:], jnp.diagonal(R)[reg0:], Dt["v"])
    q = s + F.T @ R @ F + prop
    return (
        {"m": a, "C": R, "s": s / var_scale, "nu": nu},
        {"m": a, "C": R, "s": s / var_scale, "q": q, "nu": nu, "f": f},
    )


vdlm_uv_fcast_step_exog = vmap(
    dlm_uv_fcast_step_exog, in_axes=(0, None, None, None, None, 0, 0)
)


def dlm_uv_fcast_H(disc_factor, variance_disc, var_power, mc, Stm1, DH):
    """Vectorised h-step forecast for a single series.

    Inputs
    ------
    disc_factor : per-state discount factor vector ``(1-δ)/δ``, shape ``(k,)``.
        (Previously this argument was the precomputed system covariance ``W``;
        it is now the discount vector so the kernel builds the block congruence
        ``W = B C B - C`` internally.)
    Stm1 : dict with 'm', 'C', 'nu', 's' at the start of the forecast window
    DH : dict with 'GH' (h x k x k) and 'FH' (h x k)
    """
    h = len(DH["GH"])
    aH = axis0dot(
        Stm1["m"][jnp.newaxis, ...] * jnp.ones((h, 1)),
        DH["GH"].swapaxes(-1, -2),
    )
    # System noise increment W from the origin covariance, then the W&H-standard
    # forecast: RH_j = G^j C G^j' + Σ_{i<j} G^i W G^i' (constant origin-W, added
    # additively -> ~linear variance growth). Block (congruence) geometry:
    # W = B C B - C, B = diag(1/sqrt(delta)); its diagonal equals (1-d_i)/d_i * C_ii,
    # with the off-diagonals added by the congruence.
    b = jnp.sqrt(1.0 + disc_factor)                    # (k,) = 1/sqrt(delta)
    BB = b[:, jnp.newaxis] * b[jnp.newaxis, :]         # (k,k) = outer(b, b)
    W = BB * Stm1["C"] - Stm1["C"]
    PH = axis0dot(
        axis0dot(DH["GH"], Stm1["C"] * jnp.ones((h, 1, 1))),
        DH["GH"].swapaxes(-1, -2),
    )
    WH = jnp.concatenate(
        [
            W[jnp.newaxis, ...],
            axis0dot(
                axis0dot(DH["GH"][:-1, ...], W * jnp.ones((h - 1, 1, 1))),
                DH["GH"][:-1, ...].swapaxes(-1, -2),
            ),
        ],
        axis=0,
    )
    RH = PH + WH.cumsum(axis=0)
    f, F = vF_val_n_grad(DH["FH"], aH, mc, 1 - mc)
    var_scale = var_scale_fn(f, var_power)
    s = Stm1["s"] * var_scale
    d = s * Stm1["nu"] * variance_disc
    nu = variance_disc * Stm1["nu"]
    s = d / nu
    q = s + axis0dot(axis0dot(F, RH), F)
    return {
        "m": aH,
        "C": RH,
        "s": s / var_scale,
        "q": q,
        "nu": nu * jnp.ones(h),
        "f": f,
    }


# Stationarity guard for the AR/TVAR forecast. None disables it (forecasts use
# the raw filtered AR coefficients — the prior behaviour). A float `eps` flips
# any AR characteristic root on/outside the unit circle to just inside, so the
# iterated variance recursion Q_j = s + sum phi_k^2 Q_{j-k} cannot diverge for
# near-unit-root / explosive coefficients. Larger eps tames the residual
# variance level more (root radius capped near 1-eps); 1e-3 just stops the
# divergence. Only affects models with a regression tail (n_reg > 0); the
# structural path is untouched.
AR_STATIONARITY_EPS = 1e-3


def _flip_to_stationary(phi, eps=AR_STATIONARITY_EPS):
    """Return AR coefficients with any non-stationary roots reflected inside the
    unit circle (root flipping; stationary coefficients are returned unchanged).

    Mirrors the numpy reference
        lam = roots([1, -phi]); bad = |lam| >= 1-eps
        lam = where(bad, (1-eps)*lam/|lam|^2, lam); phi* = -poly(lam)[1:]
    via the companion eigenvalues and a convolution reconstruction (verified
    identical to the numpy version to ~1e-5). ``phi`` shape ``(p,)``."""
    p = phi.shape[0]
    C = (jnp.zeros((p, p), phi.dtype)
         .at[0, :].set(phi)
         .at[jnp.arange(1, p), jnp.arange(p - 1)].set(1.0))
    lam = jnp.linalg.eigvals(C)                            # companion eigenvalues
    bad = jnp.abs(lam) >= 1.0 - eps
    lam = jnp.where(bad, (1.0 - eps) * lam / (jnp.abs(lam) ** 2), lam)
    coeffs = jnp.array([1.0 + 0.0j])
    for i in range(p):                                     # p static -> unrolled
        coeffs = jnp.convolve(coeffs, jnp.stack([jnp.asarray(1.0 + 0.0j), -lam[i]]))
    return -jnp.real(coeffs[1:])


def _flip_ar_np(phi, eps=AR_STATIONARITY_EPS):
    """Numpy, per-row root flipping over a batch of AR coefficient vectors.

    ``phi`` is ``(n, order)`` real (only the TVAR rows — typically 5·q of them,
    not the whole p·n_reg universe). numpy's ``np.roots`` loops per row in C with
    no block-diagonal, so this is O(n) memory and fast (~ms for a few thousand
    rows) where the jax ``vmap(eig)`` over all p rows OOMs on the cluster. This
    is the canonical reference of :func:`_flip_to_stationary`, applied off-device
    before the forecast (see ffs_core._run_cv_batch)."""
    out = np.array(phi, dtype=float, copy=True)
    for i in range(out.shape[0]):
        lam = np.roots(np.r_[1.0, -out[i]])
        bad = np.abs(lam) >= 1.0 - eps
        # Guard the denominator on the kept (non-bad) roots, which may be exactly
        # zero (e.g. the Minnesota lag-k prior = 0) and would otherwise raise a
        # spurious 0/0 in the discarded branch of the where.
        denom = np.where(bad, np.abs(lam) ** 2, 1.0)
        lam = np.where(bad, (1.0 - eps) * lam / denom, lam)
        out[i] = -np.poly(lam)[1:].real
    return out


def _regressor_var_prop(a_tail, r_tail, v_tail):
    """Forecast-variance contribution of a *stochastic* regressor tail.

    For a regression term ``phi * x`` with independent coefficient ``phi``
    (filtered mean ``a_tail``, variance ``r_tail``) and regressor ``x``
    (forecast variance ``v_tail``), the exact product variance is

        Var(phi * x) = E[phi]^2 Var(x) + E[x]^2 Var(phi) + Var(phi) Var(x).

    The ``E[x]^2 Var(phi)`` term is already carried by the ``gradF' R gradF``
    (``F' R F``) quadratic form, which evaluates F at the regressor *mean*.
    This helper returns the remaining

        sum_k (a_tail_k^2 + r_tail_k) * v_tail_k
        = E[phi]^2 Var(x) + Var(phi) Var(x),

    summed over the tail. It is zero when the regressor is *known*
    (``v_tail == 0``), so a deterministic future regressor adds nothing.

    Shared by the AR/TVAR iterated forecast (where ``v_tail`` is the model's
    own forecast variance, fed forward), the multi-model exogenous forecast,
    and ``uv_dlm.forecast`` (where ``v_tail`` is a supplied regressor
    variance). ``a_tail``, ``r_tail``, ``v_tail`` are all shape ``(n_reg,)``.
    """
    return jnp.sum((a_tail ** 2 + r_tail) * v_tail)


def iterated_obs_forecast(
    aH, RH, s0, variance_disc, var_power, mc, F_struct, n_reg, seed_lags, is_reg=True
):
    """Iterated-expectations observation forecast for a state-dependent F.

    The standard h-step forecast (``dlm_uv_fcast_H``) assumes the regression
    vector F is known for every future horizon. That fails when F at horizon j
    depends on the model's own forecasts — the AR/TVAR case, where the
    regression tail carries the lagged observations ``[y_{t+j-1}, ...,
    y_{t+j-p}]``. West & Harrison's iterated-expectations treatment handles this
    by feeding each horizon's predictive mean forward as the next lag and
    propagating its variance.

    The *state* propagation ``a_j = G^j m``, ``R_j`` is unchanged by the
    state-dependent F (G is block-diagonal and the AR-coefficient block is the
    identity, so the coefficients stay at their filtered mean). So ``aH``/``RH``
    are taken verbatim from the ordinary forecast kernel; this routine only
    redoes the cheap *observation* step ``y = F_j' theta_j + nu`` sequentially.

    Parameters
    ----------
    aH : (h, k)   state-mean trajectory   a_j = G^j m         (from dlm_uv_fcast_H)
    RH : (h, k, k) state-cov trajectory   R_j                 (from dlm_uv_fcast_H)
    s0 : scalar    observation variance scale at the origin   (Stm1['s'])
    variance_disc, var_power : scalars    variance machinery  (mirror the kernel)
    mc : (k,)      multiplicative-component indicator
    F_struct : (k,) structural F; the regression tail (last ``n_reg`` slots) is
        zero and is overwritten each horizon with the forecast lags.
    n_reg : int    AR order (size of the regression tail); 0 reduces this
        routine *exactly* to the standard ``f, q``.
    seed_lags : (n_reg,) the known observations ``[y_t, y_{t-1}, ...,
        y_{t-n_reg+1}]`` (most-recent first) that seed the lags of the first
        ``n_reg`` horizons.
    is_reg : bool or bool array, default True
        Per-model gate for a mixed (structural + TVAR) universe. When False the
        regression tail is *not* fed forward — ``F_j == F_struct`` and the
        propagation term is dropped — so a structural model packed alongside
        TVAR models (sharing ``n_reg`` slots that are really trend/seasonal
        state) reduces to the standard forecast. ``True`` (default) runs the
        full AR iteration.

    Returns
    -------
    f_h : (h,)  predictive means   q_h : (h,)  predictive variances

    Notes
    -----
    The variance recursion is the second-moment propagation of
    ``y_{t+j} = F_struct' theta_struct + sum_k phi_k * yhat_{j-k} + nu``:

        Q_j = s_j                                (observation noise)
            + gradF_j' R_j gradF_j               (state uncertainty at mean lags
                                                  — structural + AR-coef at fixed
                                                  lags)
            + sum_k (phi_k^2 + Var(phi_k)) Q_{j-k}   (lag-forecast propagation +
                                                      coef x lag interaction)

    where ``phi_k = a_j[reg_k]`` is the mean AR coefficient on lag k and
    ``Q_{j-k}`` is that lag's forecast variance (0 for the seeded known
    observations). With ``n_reg == 0`` the last term vanishes, ``F_j == F_struct``
    and this is byte-for-byte the standard kernel's ``f, q``. With no structural
    block and fixed coefficients it is the classic AR(p) MA-variance recursion.
    """
    reg0 = aH.shape[1] - n_reg

    # NOTE: the AR stationarity guard is NOT applied here. Running jnp.linalg.eig
    # per row inside the per-(model, series) vmap block-diagonalises across all p
    # rows on the cluster's jaxlib (an O((p·n_reg)^2) dense matrix -> OOM at M4
    # scale), and it is wasted on the structural rows anyway. Instead the AR
    # coefficients are reflected inside the unit circle once, in numpy, over just
    # the TVAR rows, before the forecast — see _flip_ar_np and its use in
    # ffs_core._run_cv_batch. aH already carries the stationarised coefficients.

    def step(carry, xs):
        buf_f, buf_q = carry           # (n_reg,) each, most-recent-first
        a_j, R_j = xs                  # (k,), (k, k)
        if n_reg > 0:
            # is_reg gates the tail-fill: structural models (is_reg False) keep
            # F_struct so the shared tail slots stay trend/seasonal state.
            F_j = jnp.where(is_reg, F_struct.at[reg0:].set(buf_f), F_struct)
        else:
            F_j = F_struct
        f_j, gradF = F_val_n_grad(F_j, a_j, mc, 1 - mc)
        var_scale = var_scale_fn(f_j, var_power)
        s_j = s0 * var_scale
        quad = gradF @ R_j @ gradF
        if n_reg > 0:
            phi = a_j[reg0:]
            r_ar = jnp.diagonal(R_j)[reg0:]
            prop = jnp.where(is_reg, _regressor_var_prop(phi, r_ar, buf_q), 0.0)
            new_f = jnp.concatenate([f_j[jnp.newaxis], buf_f[:-1]])
            new_q = jnp.concatenate([(s_j + quad + prop)[jnp.newaxis], buf_q[:-1]])
        else:
            prop = 0.0
            new_f, new_q = buf_f, buf_q
        q_j = s_j + quad + prop
        return (new_f, new_q), (f_j, q_j)

    init = (
        jnp.asarray(seed_lags).reshape(n_reg),
        jnp.zeros(n_reg),
    )
    _, (f_h, q_h) = scan(step, init, (aH, RH))
    return f_h, q_h


def exog_obs_forecast(
    aH, RH, s0, variance_disc, var_power, mc, F_struct, n_reg,
    x_traj, v_traj, is_reg=True,
):
    """Observation forecast for a state-dependent F driven by *supplied*
    exogenous regressors (the deterministic / stochastic-regressor analogue of
    :func:`iterated_obs_forecast`).

    Where the AR/TVAR routine *forecasts* its own regression tail and feeds it
    forward, here the tail is filled at each horizon from a caller-supplied
    regressor trajectory ``x_traj`` (with optional forecast variance
    ``v_traj``). Because the regressors are known up front there is no
    self-feedback, so the horizons are independent given ``aH``/``RH`` — this is
    a plain :func:`vmap` over the horizon, not a sequential scan, and no
    stationarity guard is needed (the variance recursion cannot diverge).

    The state propagation ``aH = G^j m``, ``RH = R_j`` is unchanged by the
    state-dependent F (G is block-diagonal; the regression block is the damped
    identity ``delta * I``, so the coefficients evolve under their own
    discount). ``aH``/``RH`` are therefore taken verbatim from the ordinary
    forecast kernel, exactly as in the AR path.

    Parameters
    ----------
    aH : (h, k)    state-mean trajectory   a_j = G^j m
    RH : (h, k, k) state-cov trajectory    R_j
    s0 : scalar    observation variance scale at the origin (``Stm1['s']``)
    variance_disc, var_power : scalars     variance machinery (mirror the kernel)
    mc : (k,)      multiplicative-component indicator
    F_struct : (k,) structural F; the regression tail (last ``n_reg`` slots) is
        zero and is overwritten each horizon with the supplied regressor values.
    n_reg : int    number of exogenous regressors (size of the tail).
    x_traj : (h, n_reg)  the future regressor values per horizon (the regressor
        *means*).
    v_traj : (h, n_reg)  the future regressor *variances* per horizon. Pass
        zeros for known/deterministic regressors (then the predictive variance
        reduces to obs-noise + coefficient uncertainty); pass the regressor's
        own forecast variance for a stochastic regressor.
    is_reg : bool, default True
        Per-model gate for a mixed (structural + regression) universe. When
        False the tail is not filled and the propagation term is dropped, so a
        structural model packed alongside regression models reduces to the
        standard forecast.

    Returns
    -------
    f_h : (h,)  predictive means   q_h : (h,)  predictive variances
    """
    reg0 = aH.shape[1] - n_reg

    def per_h(a_j, R_j, x_j, v_j):
        if n_reg > 0:
            F_j = jnp.where(is_reg, F_struct.at[reg0:].set(x_j), F_struct)
        else:
            F_j = F_struct
        f_j, gradF = F_val_n_grad(F_j, a_j, mc, 1 - mc)
        var_scale = var_scale_fn(f_j, var_power)
        s_j = s0 * var_scale
        quad = gradF @ R_j @ gradF
        if n_reg > 0:
            phi = a_j[reg0:]
            r_co = jnp.diagonal(R_j)[reg0:]
            prop = jnp.where(is_reg, _regressor_var_prop(phi, r_co, v_j), 0.0)
        else:
            prop = 0.0
        return f_j, s_j + quad + prop

    f_h, q_h = vmap(per_h)(aH, RH, x_traj, v_traj)
    return f_h, q_h


# Module-level jitted, vmap'd version for multi_model_dlm.forecast
_jvdlm_uv_fcast_H = jit(vmap(dlm_uv_fcast_H, in_axes=(0, 0, 0, 0, 0, 0)))

def _dlm_uv_fcast_H_fq(disc_factor, variance_disc, var_power, mc, Stm1, DH):
    """Forecast kernel returning only ``(f, q)``.

    ``multi_model_dlm.forecast`` needs just the predictive mean ``f`` and
    variance ``q``; the full ``dlm_uv_fcast_H`` also returns the per-series
    state trajectory (``m``, ``C`` = the (h, k, k) covariance, ``s``, ``nu``).
    Returning only ``f, q`` lets XLA dead-code-eliminate the assembly of those
    outputs, which would otherwise materialise as a second (nm*q, h, k, k) /
    (nm*q, h, k) batched result alongside the inputs.
    """
    out = dlm_uv_fcast_H(disc_factor, variance_disc, var_power, mc, Stm1, DH)
    return out["f"], out["q"]


# Nested form used by multi_model_dlm.forecast: the inner vmap maps over series
# and SHARES the model-only operands (GH, FH, variance/mult-comp constants) via
# in_axes=None instead of replicating them across q; the outer vmap maps over
# models. Returns only (f, q) (see _dlm_uv_fcast_H_fq). Numerically identical to
# the flat kernel above (verified to ~1e-14), but never materialises the
# (nm*q, h, k, k) GH input. The flat kernel is still used by ``_forecast_flat``
# (the regression target for ``forecast``, guarded by
# tests/test_nested_forecast_equiv.py), so keep both.
_jvdlm_uv_fcast_H_nested_fq = jit(
    vmap(
        # disc_factor (1st arg) is now the per-state discount VECTOR, a model-only
        # operand like variance_disc/mult_comps — shared across series by the inner
        # vmap (in_axes=None), mapped over models by the outer.
        vmap(_dlm_uv_fcast_H_fq, in_axes=(None, None, None, None, 0, None)),
        in_axes=(0, 0, 0, 0, 0, 0),
    )
)


def dlm_uv_fcast_svd_step(NW, variance_disc, var_power, mc, Stm1, Dt):
    a = Stm1["m"] @ Dt["G"].T
    F = F_grad(Dt["F"], a, mc, 1 - mc)
    f = F_fn(Dt["F"], a, mc, 1 - mc)
    var_scale = var_scale_fn(f, var_power)
    s = Stm1["s"] * var_scale

    SCUCGT = (
        jnp.diag(Stm1["SC"]) @ Stm1["UC"].swapaxes(-1, -2) @ Dt["G"].swapaxes(-1, -2)
    )
    NR = jnp.concatenate([SCUCGT, NW], 0)
    UNR, SNR, VNRT = svd_sqrt(NR)
    UR = VNRT.swapaxes(-1, -2)
    SR = SNR**2
    iNC = jnp.concatenate(
        [jnp.atleast_2d(F) @ UR / jnp.sqrt(s), jnp.diag(finite_inv(SR))], 0
    )
    UiNC, SiNC, ViNCT = svd_sqrt(iNC)
    UC = UR @ ViNCT.T
    SC = finite_inv(SiNC**2)

    # `m = a`: for 1-D m, `Dt["G"] @ Stm1["m"]` equals the value already
    # computed above, so it is not recomputed.
    m = a

    d = s * Stm1["nu"] * variance_disc
    nu = variance_disc * Stm1["nu"]
    s = d / nu
    q = s + F.T @ NR.T @ NR @ F

    return (
        {"m": m, "UC": UC, "SC": SC, "s": s / var_scale, "nu": nu},
        {"m": m, "UC": UC, "SC": SC, "s": s / var_scale, "q": q, "nu": nu, "f": f},
    )


vdlm_uv_fcast_svd_step = vmap(
    dlm_uv_fcast_svd_step, in_axes=(0, None, None, None, 0, 0)
)


# -----------------------------------------------------------------------------
# low-level helpers
# -----------------------------------------------------------------------------


def svd_sqrt(M, hermitian=False):
    UM, DM, VMT = svd(M)
    return UM, jnp.sqrt(DM), VMT


def pad(a, n, p=0):
    """Pad ``a`` along axis 0 with ``n`` trailing rows of constant value ``p``."""
    if n == 0:
        return a
    pad_shape = (n,) + a.shape[1:]
    return jnp.concatenate([a, jnp.full(pad_shape, p, dtype=a.dtype)], axis=0)


def seasonal_prior(c0):
    """Zero-sum seasonal prior covariance given a diagonal ``c0``."""
    C0 = jnp.diag(c0)
    l = jnp.ones(len(c0))
    U = l @ C0 @ l
    A = C0 @ l / U
    return C0 - jnp.outer(A, A) * U


def fourier_FG(p, ncomps=None, LH=True):
    """Fourier-form seasonal F and G for period ``p``.

    If ``ncomps`` is None, uses the full set. Returns ``(F, G)`` or
    ``(F, G, L, H)`` if ``LH`` is True, where ``L`` maps state to seasonal
    pattern and ``H = (L'L)^{-1} L'`` is its left inverse.
    """

    def Gp(j, period):
        omega = 2 * jnp.pi * j / period
        return jnp.asarray(
            [
                [jnp.cos(omega), jnp.sin(omega)],
                [-jnp.sin(omega), jnp.cos(omega)],
            ]
        )

    if ncomps is None:
        if p % 2 == 1:  # odd period
            m = (p + 1) // 2
            k = 2 * m - 2
            F = jnp.tile(jnp.asarray([1, 0]), k // 2)
            G = block_diag(*[Gp(i, p) for i in range(1, m)])
        else:  # even period: Nyquist block
            m = p // 2
            k = 2 * m - 2
            F = jnp.concatenate([jnp.tile(jnp.asarray([1, 0]), k // 2), jnp.ones(1)])
            G = block_diag(*([Gp(i, p) for i in range(1, m)] + [-jnp.ones(1)]))
    else:
        F = jnp.tile(jnp.asarray([1, 0]), ncomps)
        G = block_diag(*[Gp(i, p) for i in range(1, ncomps + 1)])

    if not LH:
        return F, G

    L = F[jnp.newaxis, :]
    Gi = G
    for _ in range(int(p) - 1):
        L = jnp.concatenate([L, (F @ Gi)[jnp.newaxis, :]], 0)
        Gi = Gi @ G
    H = jnp.linalg.inv(L.T @ L) @ L.T
    return F, G, L, H


def finite_inv(x, eps_truncate=False):
    """Reciprocal with 1/0 → 0 (or clipped to 1/sqrt(eps) if ``eps_truncate``)."""
    if eps_truncate:
        eps = jnp.finfo(x.dtype).eps
        return jnp.minimum(1 / x, 1 / jnp.sqrt(eps))
    y = 1 / x
    return jnp.where(jnp.isinf(y), 0.0, y)


# -----------------------------------------------------------------------------
# Dynamic Model Averaging
# -----------------------------------------------------------------------------


def DMA_model_update(
    obs,
    forecast_loc,
    forecast_sd,
    pset_prior,
    mset_prior,
    c,
    dma_pdr,
    dma_mdr,
    mi,
):
    """One-step DMA update.

    Shapes (conceptually): (n, T, h, nm). That is, n series, nm parameter
    sets grouped into a smaller number of model classes by the indicator ``mi``.
    The model-class update is allowed to evolve at a different rate
    (``dma_mdr``) than the parameter-set update (``dma_pdr``).
    """
    nm, n = pset_prior.shape
    nc, _ = mset_prior.shape

    liks = stats.norm.pdf(obs[:, jnp.newaxis], forecast_loc, forecast_sd)
    # liks is n_srs x n_models
    # mi is n_models x n_classes
    # pset_prior is n_models x n_srs
    # mset_prior is n_classes x n_srs
    tmp = (liks @ mi) / mi.sum(0)
    pM = tmp / tmp.sum(1)[:, jnp.newaxis]
    pSgM = liks / (liks @ mi @ mi.T)

    w_p = pset_prior * pSgM.T
    w_m = mset_prior * pM.T

    pset_post = w_p**dma_pdr + c / nm
    pset_post = pset_post.T / (pset_post.T @ mi @ mi.T)

    mset_post = w_m**dma_mdr + c / nc
    mset_post = mset_post.T / mset_post.sum(0)[:, jnp.newaxis]

    return pset_post.T, mset_post.T


j_DMA_model_update = jit(DMA_model_update)


class ForecastBundle(NamedTuple):
    """Predictive output: location plus (variance-stored) spread.

    The standard return type of the filter / forecast methods of
    :class:`uv_dlm` and :class:`multi_model_dlm`. The spread is stored as the
    predictive **variance** (:attr:`var`); the standard deviation is available
    via the :attr:`sd` property (``sqrt(var)``). Storing variance keeps the
    exact predictive variance available to downstream scoring without a
    ``sqrt`` → square round-trip.

    Attributes
    ----------
    loc : jnp.ndarray
        Predictive mean. The shape depends on the producer — e.g.
        ``(n_models, n_series, h)`` for ``forecast``, or flat
        ``(n_models * n_series,)`` for a single ``fwd_filter`` step.
    var : jnp.ndarray
        Predictive variance, same shape as ``loc``.
    """

    loc: jnp.ndarray  # (n_models, n_series, h)
    var: jnp.ndarray  # predictive variance, same shape as loc

    @property
    def sd(self):
        """Predictive standard deviation, ``sqrt(var)``."""
        return jnp.sqrt(self.var)


class AllocatorState(NamedTuple):
    pset: jnp.ndarray  # current parameter weights (nmodels, n_series)
    mset: jnp.ndarray  # current class weights (nclasses, n_series)
    forecast_history: jnp.ndarray  # (h, nm, n_series, 2) rolling buffer
    #  last axis = (loc, sd)
    # possibly h history of the forecasts


def allocator_step(
    state, forecast, obs, scoring_rule, horizon_aggregator, update_rule, model_indicator
):
    """
    Pure update — this is the canonical thing. Scan, jit, vmap here."""
    raw_scores = jnp.exp(scoring_rule(forecast, obs))
    aggregate_scores = horizon_aggregator(raw_scores)
    pset_post, mset_post = update_rule(
        scores=aggregate_scores,
        pset_prior=state.pset,
        mset_prior=state.mset,
        mi=model_indicator,
    )

    def wt_fn(pset_post, mset_post, model_indicator):
        return pset_post * (mset_post.T @ model_indicator.T).T

    vwt_fn = vmap(wt_fn, in_axes=(2, 2, None))

    # weights = pset_post * (mset_post.T @ model_indicator.T).T
    weights = vwt_fn(pset_post, mset_post, model_indicator)

    state = state._replace(pset=pset_post, mset=mset_post)
    return state, jnp.moveaxis(weights, 0, 2)


def LogScore(forecast, obs):
    # returns scores at each horizon
    # positively orientated - higher score = better forecast
    return stats.norm.logpdf(obs[:, jnp.newaxis], forecast.loc, forecast.sd)


def IdentityAggregator(scores, h=1):
    return scores @ jnp.ones((h, 1))


def PowerLawUpdate(dma_pdr, dma_mdr, c, scores, pset_prior, mset_prior, mi):

    def score_fn(liks, pset_prior, mset_prior):
        # liks is n_srs x n_models
        # mi is n_models x n_classes
        # pset_prior is n_models x n_srs
        # mset_prior is n_classes x n_srs

        #process
        #calculate prob of todays obs
        #multiply with prior
        #apply time discount

        nm = pset_prior.shape[0]
        nc = mset_prior.shape[0]
        tmp = (liks @ mi) / mi.sum(0)
        tmp_sum = tmp.sum(1)[:, jnp.newaxis]
        pM = tmp / jnp.where(tmp_sum == 0, 1, tmp_sum)
        denom = liks @ mi @ mi.T
        pSgM = liks / jnp.where(denom == 0, 1, denom)
        w_p = pset_prior * pSgM.T
        w_m = mset_prior * pM.T
        # ``w**disc`` with w==0 gives 0 (correct) but a NaN GRADIENT (0·log 0) — a
        # hazard only for the SGDDMA autodiff (fixed-rate DMA never differentiates
        # these). The double-``where`` keeps the forward BIT-EXACT (0**disc == 0)
        # while giving a finite gradient; the ``isnan`` guards below still catch a
        # NaN base (a diverged model's score). M1M forward is unchanged.
        w_p_s = jnp.where(w_p > 0, w_p, 1.0)
        pset_post = jnp.where(w_p > 0, w_p_s**dma_pdr, 0.0) + c / nm
        pset_post = jnp.where(jnp.isnan(pset_post),pset_prior, pset_post)
        denom = pset_post.T @ mi @ mi.T
        pset_post = pset_post.T / jnp.where(denom == 0, 1, denom)
        w_m_s = jnp.where(w_m > 0, w_m, 1.0)
        mset_post = jnp.where(w_m > 0, w_m_s**dma_mdr, 0.0) + c / nc
        mset_post = jnp.where(jnp.isnan(mset_post),mset_prior, mset_post)
        mset_sum = mset_post.sum(0)[:, jnp.newaxis]
        mset_post = mset_post.T / jnp.where(mset_sum == 0, 1, mset_sum)
        return pset_post, mset_post

    sfn = vmap(score_fn, in_axes=(2, 2, 2))
    pp, mp = sfn(scores.swapaxes(0, 1), pset_prior, mset_prior)

    return pp.swapaxes(0, 2), mp.swapaxes(0, 2)
    # return jnp.moveaxis(pp, 0, -1), jnp.moveaxis(mp, 0, -1)


def _t_vincent_scale_factor(nu):
    r"""Per-component factor that turns a Student-t **scale** :math:`\sqrt{q}`
    into a variance-matched Gaussian SD.

    For :math:`\nu > 2` this is :math:`\sqrt{\nu/(\nu-2)}` (the variance-matched
    factor); for :math:`\nu \le 2` (infinite t-variance — only on very short
    series) it falls back to the 95%-quantile-implied factor
    :math:`t^{-1}_\nu(0.975)/1.96`, finite for any :math:`\nu > 0`. As
    :math:`\nu \to \infty` the factor :math:`\to 1` (the Gaussian limit).

    Mirrors :func:`DLMAX.ffs_core._t_vincent_sd` so that
    :meth:`Allocator.combine` can reproduce
    :func:`DLMAX.ffs_core._combined_predictive_sd` (``method="quantile"``).
    The :math:`\nu \le 2` branch has no closed form, so it is evaluated on the
    host via SciPy; under the usual large ``nu`` it is skipped entirely (the
    branch is never taken), keeping the common path pure-JAX.
    """
    nu = jnp.asarray(nu, dtype=jnp.result_type(float))
    var_factor = jnp.sqrt(nu / (nu - 2.0))
    if not bool(jnp.any(nu <= 2.0)):
        return var_factor
    from scipy.stats import t as t_dist

    z95 = 1.959963984540054  # norm.isf(0.025)
    quant = jnp.asarray(np.asarray(t_dist.ppf(0.975, np.asarray(nu))) / z95)
    return jnp.where(nu > 2.0, var_factor, quant)


class Allocator:
    """Class to do model allocation"""

    def __init__(
        self,
        scoring_rule,
        update_rule,
        horizon_aggregator=IdentityAggregator,
        model_indicator=None,
        device=None,
    ):
        if device is None:
            raise ValueError(
                "`device` must be a jax.sharding.NamedSharding; see "
                "DLMAX.ffs.devices.configure_devices for the usual construction."
            )
        self.device = device
        self.scoring_rule = scoring_rule
        self.update_rule = update_rule
        self.horizon_aggregator = horizon_aggregator
        self.mi = jnp.asarray(model_indicator)
        self.state = None

    def init(self, n_models, n_series, n_classes, model_indicator, h=1):
        self.state = device_put(
            init_alloc_state(n_models, n_series, n_classes, model_indicator, h),
            self.device,
        )

    # def update(self, forecast, obs):
    #    """Single-step ergonomic API. Mutates self.state.
    #    raw scores ALWAYS higher score = better forecast!"""
    #    self.state, weights = allocator_step(
    #        self.state,
    #        forecast,
    #        obs,
    #        self.scoring_rule,
    #        self.horizon_aggregator,
    #        self.update_rule,
    #        self.mi,
    #    )
    #    return weights

    def update(self, forecast, obs):
        """Mutating single-step API for online use."""
        self.state, weights = self.prepared_step()(self.state, forecast, obs)
        return weights

    def prepared_step(self):
        """Return a pure step function with rules and mi pre-bound.

        The returned function has signature:
            (state, forecast, obs) -> (new_state, weights)

        Intended for use inside lax.scan or jax.jit where the rules
        must be closed over at trace time.
        """
        return Partial(
            allocator_step,
            scoring_rule=self.scoring_rule,
            horizon_aggregator=self.horizon_aggregator,
            update_rule=self.update_rule,
            model_indicator=self.mi,
        )

    def combine(self, weights, forecasts, nu=None):
        """DMA-weighted mean and quantile-averaged SD of a set of forecasts.

        Returns a :class:`ForecastBundle`. The combined spread is a quantile
        (Vincent) average of the per-component standard deviations; it is stored
        as ``var = sd**2`` so that ``result.sd`` returns it back (the only
        round-trip in the bundle API, and off the bit-exact CV path).

        Parameters
        ----------
        weights : array, shape ``(n_models, n_series, h)``
            Final DMA model weights.
        forecasts : ForecastBundle
            Per-component predictives. ``forecasts.sd`` is the per-component
            Student-t **scale** :math:`\\sqrt{q}` (the bundle stores ``var = q``).
        nu : optional, scalar or array broadcastable to ``forecasts.sd``
            Per-(model, series) Student-t degrees of freedom. When given, each
            component scale is converted to a variance-matched Gaussian SD via
            :func:`_t_vincent_scale_factor` before averaging, so the combined SD
            reproduces :func:`DLMAX.ffs_core._combined_predictive_sd`
            (``method="quantile"`` / :func:`~DLMAX.ffs_core._t_vincent_sd`).
            ``None`` (default) is the Gaussian limit :math:`\\nu \\to \\infty`
            (a very large ``nu``): the factor is exactly 1 and the result is
            bit-identical to the prior pure-Gaussian quantile average.
        """
        loc, sd_pm = forecasts.loc, forecasts.sd
        # Optional Student-t correction. forecasts.sd is the t SCALE sqrt(q);
        # scaling it by the variance-matched factor sqrt(nu/(nu-2)) makes the
        # quantile average below match ffs_core._t_vincent_sd exactly. nu=None
        # leaves the scale as-is (Gaussian limit, factor 1) — bit-identical.
        if nu is not None:
            factor = _t_vincent_scale_factor(nu)
            while factor.ndim < sd_pm.ndim:
                factor = factor[..., jnp.newaxis]
            sd_pm = sd_pm * factor
        # NaN-robust mixture: if a component forecast is non-finite for a given
        # (series, horizon), drop it and renormalise the DMA weights over the
        # survivors, so one diverged model cannot poison the whole series'
        # combined forecast. The model's weight is left at its prior by
        # PowerLawUpdate's NaN-score guard, so power discounting reweights it back
        # gradually. Bit-identical wherever all component forecasts are finite.
        finite = jnp.isfinite(loc) & jnp.isfinite(sd_pm)
        any_bad = ~jnp.all(finite, axis=0, keepdims=True)          # (1, n_series, h)
        w = jnp.where(finite, weights, 0.0)
        wsum = jnp.sum(w, axis=0, keepdims=True)
        w = w / jnp.where(wsum > 0, wsum, 1.0)
        w_eff = jnp.where(any_bad, w, weights)                     # all-finite cols -> original weights
        loc_s = jnp.where(finite, loc, 0.0)
        sd_s = jnp.where(finite, sd_pm, 0.0)
        fd = (loc_s * w_eff).sum(0)
        sd = (((loc_s + 1.96 * sd_s) * w_eff).sum(0) - fd) / 1.96
        return ForecastBundle(fd, sd ** 2)


def init_alloc_state(n_models, n_series, n_classes, model_indicator, horizon=1):
    return AllocatorState(
        pset=jnp.ones((n_models, n_series, horizon))
        / jnp.asarray(
            [model_indicator.sum(0)[model_indicator[i]] for i in range(n_models)]
        )[:, jnp.newaxis, :],
        mset=jnp.ones((n_classes, n_series, horizon)) / n_classes,
        forecast_history=jnp.zeros((horizon, n_models, n_series, 2)),
    )

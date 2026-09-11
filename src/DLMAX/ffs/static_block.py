"""``StaticBlock`` — the standard static-discount universe as a :class:`~DLMAX.ffs.block.Block`.

Wraps the packed ``multi_model_dlm`` universe (fixed / static-grid discounts,
combined by DMA) behind the block face.

It exposes the **rolling-origin CV** mode (:meth:`forecast_rolling`), which
delegates to the ``_run_cv_batch`` core and so reproduces ``StaticFFS``'s
CV **bit-for-bit** — the seam ``AutoFFS`` routes through, which
is what makes the block path transparent for a single block.

It also exposes the **streaming production mode** (``scan_filter`` /
``fwd_filter`` / ``forecast``), so ``AutoFFS(blocks=[StaticBlock(...)])`` fits,
steps and forecasts exactly as a grid block does — one vocabulary whichever
engine is underneath.

Streaming and CV differ in ONE respect, and it is inherent rather than a defect:
``_build_multi_and_dma`` elicits its diffuse prior from the whole array it is
given, which is the entire batch under CV. A stream cannot do that without
look-ahead, so it elicits from ``ys[:warmup]`` — as ``GridBlock`` does, and as is
required for step-equals-scan to hold. Given the SAME window the two agree to
float precision (~4e-16 on the per-model h-step location; see
``tests/test_static_block_streaming.py``).

Persistence (``save``/``load``) is NOT implemented: a persistent static universe
is ``AutoFFSUniverse(grid_period=None)``, which streams through the legacy multi
path rather than through this block.
"""

from __future__ import annotations

import numpy as np

_MONITOR_DEFAULT = object()  # sentinel: default monitor_tau to ffs_core.MONITOR_TAU


class StaticBlock:
    """The static-discount universe, configured, presented as a block.

    Holds the universe-defining config (the same knobs ``AutoFFS`` carries) and
    produces a per-model CV trajectory for a batch via :meth:`forecast_rolling`.
    """

    def __init__(self, *, season_length, n_seas_comps, warmup: int = 0,
                 dma_pdr: float = 0.90, dma_mdr: float = 0.90,
                 include_ar: bool = False, adaptive: bool = False,
                 tau_values=None, var_disc_values=None,
                 monitor_tau=_MONITOR_DEFAULT, universe_builder=None,
                 h_template: int = 18):
        self.season_length = season_length
        self.n_seas_comps = n_seas_comps
        self.warmup = warmup
        self.dma_pdr = dma_pdr
        self.dma_mdr = dma_mdr
        self.include_ar = include_ar
        self.adaptive = adaptive
        self.tau_values = tau_values
        self.var_disc_values = var_disc_values
        if monitor_tau is _MONITOR_DEFAULT:
            # match AutoFFS/StaticFFS: the static-grid error monitor is ON.
            from DLMAX.ffs_core import MONITOR_TAU  # lazy: avoid import cycle
            monitor_tau = MONITOR_TAU
        self.monitor_tau = monitor_tau
        self.universe_builder = universe_builder
        # Longest horizon the STREAMING face can forecast. The packed universe
        # precomputes its h-step template at build time (unlike the grid, which
        # builds GH per call), so this is fixed when the universe is. Only the
        # streaming path uses it; CV passes its own h through forecast_rolling.
        self.h_template = int(h_template)
        self.q = None    # number of series; set when a batch is seen
        self._nm = None  # number of packed models; set from the trajectory
        self._multi = None       # pre-built universe (from_multi); else built per batch
        self._model_desc = None  # its descriptor (carries the 'Class' column)
        # Streaming state — None until the first scan_filter. Held separately
        # from the CV path, which stays stateless and bit-identical.
        self._dma = None         # Allocator over the packed universe
        self._dma_state = None   # its persistable carry
        self._dma_step = None    # prepared_step, bound once
        self._mi = None          # model -> class indicator
        self._weights = None     # latest (M, q) DMA weights
        self._t = 0              # observations absorbed

    @classmethod
    def from_multi(cls, multi, model_desc, *, warmup: int = 0,
                   dma_pdr: float = 0.90, dma_mdr: float = 0.90):
        """Wrap a user-assembled packed universe as a StaticBlock.

        The transparent, from-scratch path for the FFS paper: build a dict of
        fixed-discount ``uv_dlm`` cells with :meth:`DLM.compile_universe`, pack
        them into a ``multi_model_dlm``, and present the pair here — the block
        then drives the identical CV core as the config-built ``StaticBlock``.

        Parameters
        ----------
        multi : multi_model_dlm
            The packed universe (``multi.nm`` models, ``multi.q`` series).
        model_desc : pd.DataFrame
            One row per model in packed order, carrying a ``'Class'`` column
            (the between-class DMA layer). ``compile_universe``'s descriptor
            plus a ``'Class'`` assignment.
        warmup, dma_pdr, dma_mdr : as the constructor.
        """
        blk = cls(season_length=None, n_seas_comps=None, warmup=warmup,
                  dma_pdr=dma_pdr, dma_mdr=dma_mdr)
        blk._multi = multi
        blk._model_desc = model_desc
        blk._nm = int(multi.nm)
        return blk

    @property
    def nm(self):
        """Number of packed candidate models (set after the first batch)."""
        return self._nm

    # -- rolling-origin driving mode (the CV path) -----------------------------
    def forecast_rolling(self, srs_ids, arr, cutoff_t_idx, h,
                         warmup_steps: int = 0, capture_trace: bool = False):
        """Single-pass rolling-origin CV for a batch, emitting per-model h-step
        forecasts at ``cutoff_t_idx``. Delegates to ``_run_cv_batch`` — identical
        to the legacy CV path — and returns its ``_CVTrajectory``.
        """
        from DLMAX.ffs_core import _run_cv_batch  # lazy: avoid import cycle

        prebuilt = None
        if self._multi is not None:
            # User-assembled universe (from_multi): build the allocator for the
            # given multi and run the identical CV core over it.
            from DLMAX.ffs_core import _dma_for_multi

            dma, model_indicator = _dma_for_multi(
                int(self._multi.nm), int(np.asarray(arr).shape[1]),
                self._model_desc, self.dma_pdr, self.dma_mdr,
            )
            prebuilt = (self._multi, dma, model_indicator)

        traj = _run_cv_batch(
            srs_ids, arr, cutoff_t_idx, self.season_length, self.n_seas_comps, h,
            self.dma_pdr, self.dma_mdr, warmup_steps, self.include_ar,
            self.adaptive, self.tau_values, self.var_disc_values, capture_trace,
            self.monitor_tau, self.universe_builder, _prebuilt=prebuilt,
        )
        self.q = int(np.asarray(arr).shape[1])
        self._nm = int(traj.f_h.shape[1])
        return traj

    def cv_trajectory(self, srs_ids, arr, cutoffs, h, regressors=None):
        """Uniform block CV face for the ``blocks=`` API: a
        union-ready ``_CVTrajectory`` (per-model h-step + one-step trace +
        model_indicator), using the block's own ``warmup``. Mirrors
        ``GridBlock.cv_trajectory`` so the orchestrator drives a list of blocks
        uniformly. Delegates to :meth:`forecast_rolling` with ``capture_trace``.

        ``regressors`` is part of the uniform face, not a capability: a static
        block carries no regression tail, so a supplied design would go
        nowhere. Refusing it is the same contract ``GridBlock`` enforces --
        silently ignoring the caller's design is the failure this face exists
        to prevent.
        """
        if regressors is not None:
            raise ValueError(
                "regressors supplied to a StaticBlock, which has no regression "
                "tail: the design would go nowhere. Use a GridBlock/AdaptiveBlock "
                "whose cells carry a Regressors or AR component.")
        return self.forecast_rolling(
            srs_ids, arr, cutoffs, h, warmup_steps=self.warmup, capture_trace=True
        )

    @property
    def model_indicator(self):
        """``(M, C)`` bool model->class map, for the union allocator.

        Mirrors ``GridBlock.model_indicator`` so a list of mixed block types can
        be combined at the union. Available once the universe exists.
        """
        if self._mi is None:
            raise RuntimeError(
                "StaticBlock.model_indicator before the universe is built: "
                "call scan_filter (or forecast_rolling) first.")
        return np.asarray(getattr(self._mi, "values", self._mi), dtype=bool)

    # -- streaming production mode --------------------------------------------

    def _ensure_stream(self, ys, init_window=None):
        """Build the universe + allocator on first use, prior elicited from ``ys``.

        Uses the SAME :func:`_build_multi_and_dma` the CV path uses, so the
        streamed universe IS the universe CV builds rather than a second
        definition free to drift from it. A ``from_multi`` block already carries
        its universe and only needs the allocator.
        """
        if self._dma is not None:
            return
        from DLMAX.ffs_core import _build_multi_and_dma, _dma_for_multi
        a = np.asarray(ys, dtype=float)
        # Elicit from the WARMUP WINDOW, not from everything this first call
        # happens to carry. _build_multi_and_dma uses the whole array it is
        # given once warmup_steps > 0, so passing the full history would make
        # the prior depend on how the caller chunked the stream --
        # scan_filter(all) and scan_filter(warmup) + fwd_filter(rest) would
        # then disagree. Slicing here matches GridBlock, whose carry is elicited
        # from ys[:warmup]. The window is still filtered afterwards, as there.
        # ``init_window`` overrides the elicitation window. Default None takes
        # ys[:warmup], which is what a stream can see; passing the batch that CV
        # would use reproduces the CV prior exactly, and is how the equivalence
        # test isolates this (the only difference between the two faces).
        if init_window is not None:
            init = np.asarray(init_window, dtype=float)
        elif self.warmup and self.warmup > 0:
            init = a[:self.warmup]
        else:
            init = a
        if self._multi is None:
            self._multi, self._dma, self._mi = _build_multi_and_dma(
                init, self.season_length, self.n_seas_comps, self.h_template,
                self.dma_pdr, self.dma_mdr, self.warmup,
                include_ar=self.include_ar, adaptive=self.adaptive,
                tau_values=self.tau_values, var_disc_values=self.var_disc_values,
                monitor_tau=self.monitor_tau,
                universe_builder=self.universe_builder)
        else:
            self._dma, self._mi = _dma_for_multi(
                int(self._multi.nm), int(a.shape[1]), self._model_desc,
                self.dma_pdr, self.dma_mdr)
        self._nm = int(self._multi.nm)
        self.q = int(a.shape[1])   # full width, not the init slice's
        self._dma_state = self._dma.state
        self._dma_step = self._dma.prepared_step()
        self._t = 0

    def fwd_filter(self, yt, return_trace=False, regressors=None):
        """Advance one observation ``yt`` ``(q,)`` across every packed model.

        ``return_trace`` also returns the per-model one-step ``(F, Q)``
        ``(q, M)`` — the same layout ``GridBlock.fwd_filter`` emits, so the union
        DMA is driven identically whichever block type it is combining.
        """
        import jax.numpy as jnp
        from DLMAX.dlm_core import ForecastBundle
        if regressors is not None:
            raise ValueError(
                "regressors supplied to a StaticBlock, which has no regression "
                "tail: the design would go nowhere. Use a GridBlock/AdaptiveBlock "
                "whose cells carry a Regressors or AR component.")
        if self._dma is None:
            raise RuntimeError(
                "StaticBlock.fwd_filter before scan_filter (no state): the prior "
                "is elicited from a warmup window, so the first call must be "
                "scan_filter over one.")
        y = jnp.asarray(np.asarray(yt, dtype=float))
        # Warmup is per STEP, as in the scan: inside the window the discount
        # matrix is zeroed and the observational variance held. Without this the
        # streaming face would filter the warmup window untreated and diverge
        # from the batch path over exactly those observations.
        warm = 1.0 if self._t < int(self.warmup or 0) else 0.0
        f, q = self._multi.fwd_filter(y, warmup_flag=warm)     # (nm, q) each
        self._dma_state, w = self._dma_step(
            self._dma_state, ForecastBundle(f[..., None], q[..., None]), y)
        self._weights = np.asarray(w[:, :, 0])                 # (M, q)
        self._t += 1
        if return_trace:
            return self, (np.asarray(f).T, np.asarray(q).T)    # (q, M)
        return self

    def scan_filter(self, ys, *, return_trace=False, regressors=None, **_ignored):
        """Advance (or initialise) over ``ys`` ``(T, q)`` time-major.

        The first call must include the warmup window: the diffuse prior is
        elicited from it, exactly as the batch path. Implemented as a loop over
        :meth:`fwd_filter`, so the step and scan faces agree BY CONSTRUCTION
        rather than by a tolerance — unlike the grid, whose scan is a fused
        ``lax.scan`` and therefore reduces in a different order from its step.
        """
        a = np.asarray(ys, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"scan_filter(ys) wants (T, q); got {a.shape}.")
        self._ensure_stream(a)
        Fs, Qs = [], []
        for t in range(a.shape[0]):
            if return_trace:
                _b, (F, Q) = self.fwd_filter(a[t], return_trace=True)
                Fs.append(F); Qs.append(Q)
            else:
                self.fwd_filter(a[t])
        if return_trace:
            return self, (np.stack(Fs), np.stack(Qs))          # (T, q, M)
        return self

    def forecast(self, h, *args, **kwargs):
        """``h``-step predictive from the held state, combined under the DMA.

        Returns ``(loc (q, h), sd (h, q), components)`` in ``GridBlock``'s
        layout, so the orchestrator's single-block combine consumes either block
        type without knowing which it has.
        """
        from DLMAX.ffs_core import _union_predictive_combine
        if self._dma is None or self._weights is None:
            raise RuntimeError(
                "StaticBlock.forecast before scan_filter (no state).")
        if int(h) > self.h_template:
            raise ValueError(
                f"StaticBlock.forecast(h={h}) exceeds h_template="
                f"{self.h_template}: the packed universe precomputes its h-step "
                f"template when it is built, so the horizon is fixed then. "
                f"Construct with StaticBlock(..., h_template={int(h)}).")
        bundle = self._multi.forecast(h)
        f_h = np.asarray(bundle.loc)                           # (M, q, h)
        q_h = np.asarray(bundle.var)
        nu = np.asarray(self._multi.dlm_state["nu"]).reshape(self._nm, self.q)
        W = self._weights                                      # (M, q)
        loc, sd, _bounds = _union_predictive_combine(
            W, f_h, q_h, nu, None, "quantile")
        comp = {"Wc": W.T,                                     # (q, M)
                "LOCc": np.moveaxis(f_h, 0, 1),                # (q, M, h)
                "QHc": np.moveaxis(q_h, 0, 1),
                "NUc": nu.T}                                   # (q, M)
        return loc, sd, comp

    # Persistence stays unimplemented: a PERSISTENT static universe is
    # AutoFFSUniverse(grid_period=None), which streams through the legacy multi
    # path rather than through this block. The in-memory streaming above is what
    # AutoFFS needs; wiring save/load would only duplicate that route.
    def save(self, fname):
        raise NotImplementedError(
            "StaticBlock has no persistence (save): it streams in memory only. "
            "Use AutoFFSUniverse(grid_period=None) for a persistent static "
            "universe.")

    def load(self, fname):
        raise NotImplementedError(
            "StaticBlock has no persistence (load): it streams in memory only. "
            "Use AutoFFSUniverse(grid_period=None) for a persistent static "
            "universe.")

    def __repr__(self):
        q = "unfitted" if self.q is None else f"q={self.q}"
        return (f"StaticBlock(season_length={self.season_length}, "
                f"nm={self._nm}, {q})")

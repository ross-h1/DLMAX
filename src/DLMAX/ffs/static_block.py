"""``StaticBlock`` — the standard static-discount universe as a :class:`~DLMAX.ffs.block.Block`.

Wraps the packed ``multi_model_dlm`` universe (fixed / static-grid discounts,
combined by DMA) behind the block face.

It exposes the **rolling-origin CV** mode (:meth:`forecast_rolling`), which
delegates to the ``_run_cv_batch`` core and so reproduces ``StaticFFS``'s
CV **bit-for-bit** — the seam ``AutoFFS`` routes through, which
is what makes the block path transparent for a single block.

The streaming production mode (``scan_filter`` / ``forecast``) is NOT
implemented here: this block is the batch/CV engine, and a caller wanting to
stream a static universe forward should use ``AutoFFSUniverse`` with an
explicit ``grid_period=None``. The methods raise rather than returning
something plausible. ``StaticBlock`` satisfies the ``Block`` protocol, which
checks names rather than behaviour.
"""

from __future__ import annotations

import numpy as np

_PROD_MSG = (
    "StaticBlock has no streaming production mode ({name}): it is the batch / "
    "rolling-origin engine. Use forecast_rolling for the CV path, or "
    "AutoFFSUniverse(grid_period=None) to stream a static universe forward."
)

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
                 monitor_tau=_MONITOR_DEFAULT, universe_builder=None):
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
        self.q = None    # number of series; set when a batch is seen
        self._nm = None  # number of packed models; set from the trajectory
        self._multi = None       # pre-built universe (from_multi); else built per batch
        self._model_desc = None  # its descriptor (carries the 'Class' column)

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

    # -- streaming production mode: not implemented, see the module docstring --
    def scan_filter(self, ys, *args, **kwargs):
        raise NotImplementedError(_PROD_MSG.format(name="scan_filter"))

    def fwd_filter(self, yt, *args, **kwargs):
        raise NotImplementedError(_PROD_MSG.format(name="fwd_filter"))

    def forecast(self, h, *args, **kwargs):
        raise NotImplementedError(_PROD_MSG.format(name="forecast"))

    def save(self, fname):
        raise NotImplementedError(_PROD_MSG.format(name="save"))

    def load(self, fname):
        raise NotImplementedError(_PROD_MSG.format(name="load"))

    def __repr__(self):
        q = "unfitted" if self.q is None else f"q={self.q}"
        return (f"StaticBlock(season_length={self.season_length}, "
                f"nm={self._nm}, {q})")

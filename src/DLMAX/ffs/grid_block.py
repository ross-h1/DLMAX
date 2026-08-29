"""``GridBlock`` — the dynamic-discount grid as a :class:`~DLMAX.ffs.block.Block`.

Wraps the RTRL discount grid (:mod:`DLMAX.ffs.discount_grid`) behind the block
face so the orchestration layer can drive it inside a
:class:`~DLMAX.ffs.universe.Universe`, alone or alongside other blocks combined
by the top-level DMA.

Two driving modes
-----------------
* **Rolling-origin** (:meth:`forecast_rolling`, :meth:`cv_trajectory`) — learn
  the discounts online while emitting each worker's h-step forecast *at the
  cutoffs*, so origin ``t`` uses only what the grid had learned by ``t``. This
  is the backtest mode, and the one the published results run on.
* **Production** (:meth:`scan_filter` to now, then :meth:`forecast`) — advance a
  resumable carry over observations and forecast forward from it. This is what
  ``AutoFFSUniverse`` streams and what ``AutoFFS.fit``/``predict`` hold in
  memory.

Where the DMA sits
------------------
A block is a construction and engine boundary, **not** a DMA level. There are
two independent uses of averaging here, with no feedback between them:

- the grid's **within-cell** wingman pooling, which moves the discount centres,
  is the block's own *learning* signal. It is self-contained and never fed from
  outside: the discounts a grid learns must depend on the grid and the data, not
  on which other blocks happen to share the universe.
- the **combination** is a single two-level ``Allocator`` the orchestrator runs
  over the *union* of all blocks' models — the grid's per-worker predictives
  concatenated with any other block's — with classes being the grid families
  concatenated with the other blocks' classes.

So the grid exposes its **per-worker** predictives (``LOCc``/``QHc``/``NUc``,
with ``Wc`` its own weights) for that union DMA rather than pre-combining. For a
single-block universe the union over one block's workers reproduces the block's
own internal combine, which is why a one-block union is not a second layer of
averaging.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from DLMAX.ffs.discount_grid import (
    ADAPT_GUARD_DEFAULT,
    DAMPINGS,
    DMA_C,
    DMA_MDR,
    DMA_PDR,
    DMA_PRIOR,
    build_grid,
    lag_design,
    run_grid_rolling_batch,
)

class GridBlock:
    """The adaptive-discount grid presented as a block.

    Parameters
    ----------
    grid : list[(name, GridModel, components)]
        The structure taxonomy from :func:`~DLMAX.ffs.discount_grid.build_grid`
        (16 families x 3 wingmen by default). Prefer :meth:`build`.
    period : int or None
        Seasonal period; ``None`` drops the seasonal structures.
    warmup : int
        Warmup steps for the diffuse-prior settle (discounts forced to 1 there).
    offset, pdr, mdr, dma_c :
        Wingman logit offset and the DMA forgetting/floor, forwarded to the grid.
    """

    def __init__(self, grid, *, period, warmup, offset: float = 1.0,
                 pdr: float = DMA_PDR, mdr: float = DMA_MDR, dma_c: float = DMA_C,
                 additive_logscore: bool = False, disc_prior=None,
                 decouple_trend: bool = False, learn_dma: bool = False,
                 dma_prior=None, couple_sd=None, seasonal_prior=None,
                 regression_prior=None,
                 adapt_guard=ADAPT_GUARD_DEFAULT, disc_init=None, clip=None):
        self._grid = grid
        self.period = period
        self.warmup = warmup
        self.offset = offset
        self.pdr = pdr
        self.mdr = mdr
        self.dma_c = dma_c
        self.additive_logscore = additive_logscore
        self.disc_prior = disc_prior
        # Baked into ``grid`` (state_to_block) at build; kept for save/rebuild.
        self.decouple_trend = decouple_trend
        # SGDDMA: learn this block's family (hierarchical) forgetting rates online.
        self.learn_dma = learn_dma
        self.dma_prior = dma_prior
        # Growth-to-level coupling strength (None -> off): the decoupled growth
        # discount's prior mean is the level's live logit (see _wing_step).
        self.couple_sd = couple_sd
        # Block-specific seasonal discount prior mean (None -> use disc_prior for
        # all blocks): seasonal Fourier blocks get N(seasonal_prior, disc_sd).
        # Scalar, or a per-block (nb,) vector when one cell carries several
        # seasonals of very different periods.
        self.seasonal_prior = seasonal_prior
        # Discount prior mean for a REGRESSION/AR tail block. None -> the
        # kernel falls back to ``seasonal_prior`` (long-standing behaviour,
        # harmless where a cell has no seasonal). Set it explicitly once a
        # cell carries BOTH a seasonal and a tail, where one knob cannot
        # serve two blocks that want different discounts.
        self.regression_prior = regression_prior
        # Per-family, per-block STARTING discounts, (nf, P) in logit space
        # (None -> the uniform 0.95 / β 0.99 the wing has always used). A block
        # whose natural discount is nowhere near 0.95 cannot be started there and
        # learned back — it destroys its own state first. See
        # ``AdaptiveBlock.from_cells``, which fills this in from the cells.
        self.disc_init = disc_init
        # Discount clip box (lo, hi) in probability space; None -> the module
        # constants. The FLOOR is the real bound on a block's covariance growth,
        # and what is survivable is frequency- and period-dependent: the default
        # 0.5 lets a block's covariance double every step, which is ruinous on
        # hourly data (ENTSOE needs 0.99 and diverges below it).
        self.clip = clip
        # Info-envelope covariance guard (adapt_discount port): scalar adapt coeff
        # or None. Pulls the effective discount toward 1 for already-uncertain
        # states so their covariance can't grow without bound (see discount_grid).
        self.adapt_guard = adapt_guard
        self.q = None  # number of series; set when data is first seen
        # Streaming (production) state — built lazily on first scan_filter.
        self._static = None   # obs-independent config (rebuilt on load)
        self._carry = None    # resumable carry (cell wing state + family DMA)
        self._t = 0           # observations ingested (for the warmup flag)
        # Trailing observations for an AR tail, (n_regs, q) in TIME order -- the
        # `history` that makes each chunk's lag design continuous with the last,
        # and the seed for the iterated forecast. DATA, not filter state, so it
        # lives here rather than in the wing carry pytree: no carry-structure
        # change, and old saved blocks load unchanged (absent -> None).
        self._lags = None

    @classmethod
    def build(cls, period, *, warmup, dampings=DAMPINGS, var_powers=None,
              offset: float = 1.0, pdr: float = DMA_PDR, mdr: float = DMA_MDR,
              dma_c: float = DMA_C, n_comps=None, seasonal_mult=False,
              additive_logscore=False, disc_prior=None, decouple_trend=False,
              learn_dma=False, dma_prior=None, couple_trend=False, couple_sd=None,
              period2=None, n_comps2=None, seasonal_prior=None,
              regression_prior=None,
              adapt_guard=ADAPT_GUARD_DEFAULT, clip=None):
        """Construct the grid taxonomy for ``period`` and wrap it as a block.

        ``var_powers`` (see :func:`~DLMAX.ffs.discount_grid.build_grid`): ``None``
        for the classic ``{A, M}`` variants, or a list e.g. ``[1.0, 0.25]`` for the
        additive-Fourier compound-Poisson sweep (M5). ``n_comps`` truncates the
        seasonal Fourier to that many harmonics (``None`` = full ``period//2``) —
        e.g. weekly M4 uses period 52 with ``n_comps=12``. ``seasonal_mult`` makes
        the Fourier seasonal multiplicative at the listed error law (level-scaling
        seasonality WITHOUT the multiplicative-error variance)."""
        return cls(build_grid(period, dampings, var_powers, n_comps=n_comps,
                              seasonal_mult=seasonal_mult,
                              decouple_trend=decouple_trend, couple_trend=couple_trend,
                              period2=period2, n_comps2=n_comps2),
                   period=period, warmup=warmup, offset=offset, pdr=pdr, mdr=mdr,
                   dma_c=dma_c, additive_logscore=additive_logscore,
                   disc_prior=disc_prior, decouple_trend=decouple_trend,
                   learn_dma=learn_dma, dma_prior=dma_prior, couple_sd=couple_sd,
                   seasonal_prior=seasonal_prior,
                   regression_prior=regression_prior, adapt_guard=adapt_guard,
                   clip=clip)

    # -- block shape -----------------------------------------------------------
    @property
    def nm(self) -> int:
        """Number of DMA members (workers): 3 wingmen per family."""
        return len(self._grid) * 3

    @property
    def n_classes(self) -> int:
        """Number of DMA classes (families)."""
        return len(self._grid)

    @property
    def names(self):
        """Per-worker tags, worker order = families x {fast, cen, slow}."""
        return [f"{n}:{w}" for n, _m, _c in self._grid
                for w in ("fast", "cen", "slow")]

    @property
    def n_regs(self) -> int:
        """Width of the regression tail, 0 for a structural-only block.

        Read from the families' ``GridModel``. Callers need it to tell an
        exogenous block from a structural one WITHOUT filtering anything — the
        universe uses it to reject the two silent no-ops: exogenous data supplied
        to a block that has no tail (the data goes nowhere), and a block with a
        tail driven with no data (the tail filters against a zero F row and
        scores as though it were absent).

        Families are required to agree. A mixed block would need per-family tail
        widths, which the padding could express but nothing yet builds -- see the
        brief's open question 2 -- so disagreement is an error rather than a
        silently-chosen maximum.
        """
        widths = {int(m.n_regs) for _n, m, _c in self._grid}
        if len(widths) > 1:
            raise ValueError(
                f"families disagree on the regression tail width: {sorted(widths)}. "
                "A mixed-width block is not supported (see wing_regressors_brief "
                "open question 2).")
        return widths.pop() if widths else 0

    @property
    def is_autoregressive(self) -> bool:
        """Whether the tail is AUTOREGRESSIVE (its own lagged observations)
        rather than EXOGENOUS (a caller-supplied design).

        The distinction is not cosmetic — it decides where the regressors come
        from and how the block is forecast. An AR tail supplies its own design
        from the y stream and is forecast by iterated expectations; an exogenous
        one needs the caller's matrix and its known future rows. Read from the
        ``AR`` component's ``is_autoregressive`` marker, which
        :class:`~DLMAX.ffs.dlm_builder.AR` sets and :class:`Regressors` does not.

        Families must agree, for the same reason they must on ``n_regs``.
        """
        kinds = {any(getattr(c, "is_autoregressive", False) for c in comps)
                 for _n, _m, comps in self._grid}
        if len(kinds) > 1:
            raise ValueError(
                "families disagree on whether the tail is autoregressive. Mixing "
                "AR and exogenous tails in one block is not supported: they take "
                "their regressors from different places.")
        return bool(kinds.pop()) if kinds else False

    @property
    def seed_lags(self):
        """``(q, n_regs)`` most-recent-first ``[y_t, y_{t-1}, ...]``, or ``None``.

        What :func:`~DLMAX.dlm_core.iterated_obs_forecast` wants to seed an AR
        forecast at this block's current origin. Derived from the held buffer
        (which is stored TIME-ordered, the legacy convention) rather than stored
        twice — one quantity, converted at the point of use.
        """
        if self._lags is None or not self.n_regs:
            return None
        return np.asarray(self._lags)[::-1].T          # (n_regs, q) -> (q, n_regs)

    def _update_lags(self, ys):
        """Absorb ``ys`` ``(T, q)`` into the trailing-observation buffer.

        Kept as the last ``n_regs`` rows in TIME order, so it is exactly the
        ``history`` argument :func:`lag_design` and the legacy ``format_lags``
        take. Short of ``n_regs`` rows it is zero-filled at the FRONT, matching
        ``history=None``'s "no real lag yet" — which does conflate a missing lag
        with an observed zero, as the legacy path also does.

        Only an AUTOREGRESSIVE tail keeps one. An exogenous tail's regressors come
        from the caller at every step, past and future alike, so trailing
        observations would be stored, padded and persisted without ever being
        read.
        """
        r = self.n_regs
        if not r or not self.is_autoregressive:
            return
        a = np.asarray(ys, dtype=float)
        prev = (np.zeros((0, a.shape[1])) if self._lags is None
                else np.asarray(self._lags, dtype=float))
        buf = np.concatenate([prev, a], axis=0)[-r:]
        if buf.shape[0] < r:                      # first call, fewer rows than lags
            buf = np.concatenate(
                [np.zeros((r - buf.shape[0], buf.shape[1])), buf], axis=0)
        self._lags = buf

    def _ar_design(self, ys):
        """This chunk's AR design, seeded from the buffer so it is CONTINUOUS
        across calls.

        The reason the buffer has to exist at all: building the design from
        ``ys`` alone zero-fills its first ``n_regs`` rows, which is right for the
        very first chunk and wrong for every one after it. A universe that fits
        then updates would otherwise silently blank the tail at each origin.
        """
        return lag_design(jnp.asarray(np.asarray(ys, dtype=float)),
                          self.n_regs, history=self._lags)

    @property
    def model_indicator(self):
        """Worker->family ``(nm, n_classes)`` bool indicator (worker ``j`` in
        family ``j // 3``). The block's contribution to the union DMA's
        block-diagonal indicator; matches the inline matrix built in
        :meth:`cv_trajectory`."""
        M, nf = self.nm, self.n_classes
        mi = np.zeros((M, nf), dtype=bool)
        mi[np.arange(M), np.arange(M) // 3] = True
        return mi

    # -- rolling-origin tail designs ------------------------------------------
    def _rolling_designs(self, arr, h, regressors=None):
        """``(xs, xh, seed_lags)`` for the rolling/CV path, series-major.

        The CV analogue of what :meth:`scan_filter` / :meth:`forecast` do on the
        streaming path, and of ``multi_model_dlm.format_lag_yts`` /
        ``format_seed_lag_yts`` on the legacy one. ``arr`` is the FULL ``(L, q)``
        panel; the online pass sees rows ``[0, L-h)``.

        * structural block -> ``(None, None, None)``, the bit-exact path;
        * AUTOREGRESSIVE tail -> the filter design is built from the y stream
          (``lag_design``) and the forecast seed is ``[y_t, y_{t-1}, ...]`` at
          each step, most-recent-first. ``xh`` stays ``None``: the future
          regressors ARE the forecasts, which is what the iterated forecast in
          ``forecast_origin`` is for;
        * EXOGENOUS tail -> ``regressors`` ``(L, q, n_regs)`` is required, split
          into the filter rows and, at each step ``t``, the ``h`` future rows
          that the origin at ``t`` would use.

        Raising on a tail with no design is deliberate: filtering against a zero
        F row is silent and produces plausible numbers (see
        ``AutoFFSUniverse._check_exog_supported``).
        """
        r = self.n_regs
        if not r:
            if regressors is not None:
                raise ValueError(
                    "regressors supplied to a block with no regression tail "
                    "(n_regs=0): the design would go nowhere. Drop it, or use a "
                    "block whose cells carry a Regressors/AR component.")
            return None, None, None
        a = jnp.asarray(np.asarray(arr, dtype=float))
        L = a.shape[0]
        T = L - h                                   # the online training rows
        if self.is_autoregressive:
            if regressors is not None:
                raise ValueError(
                    "an AR tail builds its own design from the y stream; "
                    "supplying `regressors` would silently override it.")
            xs = lag_design(a[:T], r)                        # (T, q, r)
            # seed at step t is [y_t, y_{t-1}, ...] -- lag_design shifted by one,
            # i.e. the same array with y_t prepended. Reuse it rather than
            # re-deriving the indexing.
            sl = lag_design(a[:T + 1], r)[1:]                # (T, q, r)
            return (jnp.transpose(xs, (1, 0, 2)),            # -> (q, T, r)
                    None,
                    jnp.transpose(sl, (1, 0, 2)))
        if regressors is None:
            raise ValueError(
                f"this block has an EXOGENOUS regression tail (n_regs={r}) but no "
                "`regressors` were supplied, so it would filter against a zero F "
                "row at every step and contribute nothing while occupying state "
                "and a discount block. Supply the design, or use AR for a tail "
                "that builds its own.")
        x = jnp.asarray(np.asarray(regressors, dtype=float))  # (L, q, r)
        if x.shape[:2] != (L, a.shape[1]) or x.shape[2] != r:
            raise ValueError(
                f"regressors must be (L, q, n_regs) = {(L, a.shape[1], r)}; "
                f"got {tuple(x.shape)}")
        xs = x[:T]                                            # (T, q, r)
        # future rows the origin at t would use: x[t+1 : t+1+h]
        xh = jnp.stack([x[t + 1: t + 1 + h] for t in range(T)])   # (T, h, q, r)
        return (jnp.transpose(xs, (1, 0, 2)),                 # (q, T, r)
                jnp.transpose(xh, (2, 0, 1, 3)),              # (q, T, h, r)
                None)

    # -- rolling-origin driving mode (the validated CV / backtest path) --------
    def forecast_rolling(self, arr, cutoffs, h, *, return_diag: bool = False,
                         regressors=None):
        """Online rolling-origin pass over ``arr`` ``(L, q)``, emitting h-step
        forecasts at ``cutoffs``.

        Trains the online learning pass on rows ``[0, L-h)`` and combines the
        workers' origin predictives with the grid's own hierarchical DMA. Returns
        ``(loc, sd)`` each ``(q, n_cut, h)`` (numpy); with ``return_diag`` also
        the ``(weight, level_d, beta)`` learned-model detail per origin. Delegates
        to :func:`~DLMAX.ffs.discount_grid.run_grid_rolling_batch` — identical to
        the standalone grid path.
        """
        a = np.asarray(arr, dtype=float)
        L = a.shape[0]
        self.q = a.shape[1]
        ys = jnp.asarray(a[: L - h].T)  # (q, L-h)
        xs, xh, sl = self._rolling_designs(a, h, regressors)
        return run_grid_rolling_batch(
            self._grid, ys, np.asarray(cutoffs), h,
            warmup=self.warmup, offset=self.offset,
            pdr=self.pdr, mdr=self.mdr, dma_c=self.dma_c,
            return_diag=return_diag, couple_sd=self.couple_sd,
            seasonal_prior=self.seasonal_prior,
            regression_prior=getattr(self, "regression_prior", None),
            adapt_guard=self.adapt_guard,
            additive_logscore=self.additive_logscore, disc_prior=self.disc_prior,
            learn_dma=self.learn_dma,
            dma_prior=self.dma_prior if self.dma_prior is not None else DMA_PRIOR,
            clip=getattr(self, "clip", None),
            disc_init=getattr(self, "disc_init", None),
            xs=xs, xh=xh, seed_lags=sl,
        )

    # -- rolling-origin driving mode (learn online, emit at the cutoffs) ------
    def cv_trajectory(self, srs_ids, arr, cutoffs, h, regressors=None):
        """Rolling-origin CV producing a ``_CVTrajectory`` in the union-DMA
        layout: per-worker h-step predictives at the cutoffs plus the
        per-worker one-step trace, so the orchestrator's union
        Allocator combines the grid's workers alongside the static block's
        models. The grid does *not* pre-combine here.

        ``arr`` ``(L, q)``; the online pass trains on rows ``[0, L-h)`` (so the
        one-step trace has ``T = L-h`` — cutoffs must be ``< L-h``).
        """
        from DLMAX.ffs_core import _CVTrajectory  # lazy: avoid import cycle

        a = np.asarray(arr, dtype=float)
        L = a.shape[0]
        self.q = a.shape[1]
        ys = jnp.asarray(a[: L - h].T)  # (q, L-h)
        xs, xh, sl = self._rolling_designs(a, h, regressors)
        _loc, _sd, blk = run_grid_rolling_batch(
            self._grid, ys, np.asarray(cutoffs), h,
            warmup=self.warmup, offset=self.offset,
            pdr=self.pdr, mdr=self.mdr, dma_c=self.dma_c, return_blocks=True,
            additive_logscore=self.additive_logscore, disc_prior=self.disc_prior,
            couple_sd=self.couple_sd, seasonal_prior=self.seasonal_prior,
            regression_prior=getattr(self, "regression_prior", None),
            adapt_guard=self.adapt_guard,
            learn_dma=self.learn_dma,
            dma_prior=self.dma_prior if self.dma_prior is not None else DMA_PRIOR,
            clip=getattr(self, "clip", None),
            disc_init=getattr(self, "disc_init", None),
            xs=xs, xh=xh, seed_lags=sl,
        )
        M = int(blk["F"].shape[2])
        nf = int(blk["n_families"])
        mi = np.zeros((M, nf), dtype=bool)
        mi[np.arange(M), np.arange(M) // 3] = True  # worker -> family
        # to _CVTrajectory layout: f_h (n_cut, M, q, h); f1_full (T, M, q)
        return _CVTrajectory(
            srs_ids=tuple(srs_ids),
            f_h=np.transpose(blk["LOCc"], (1, 2, 0, 3)),
            q_h=np.transpose(blk["QHc"], (1, 2, 0, 3)),
            nu=np.transpose(blk["NUc"], (1, 2, 0)),
            weights=None,  # union DMA recomputes; grid's own weights unused here
            cutoff_t_idx=np.asarray(cutoffs),
            f1_full=np.transpose(blk["F"], (1, 2, 0)),
            q1_full=np.transpose(blk["Q"], (1, 2, 0)),
            model_indicator=mi,
        )

    # -- production driving mode (scan-to-now + forecast-forward) --------------
    # Streaming face for AutoFFSUniverse: hold a resumable carry and advance it
    # one origin at a time. grid_stream_scan == the batch scan to the same point
    # (validated bit-exact), so forecast() reproduces the cv_trajectory emission.
    def _ensure_static(self):
        if self._static is None:
            from DLMAX.ffs.discount_grid import grid_stream_static
            _dp = {} if self.dma_prior is None else {"dma_prior": self.dma_prior}
            self._static = grid_stream_static(
                self._grid, offset=self.offset, pdr=self.pdr, mdr=self.mdr,
                dma_c=self.dma_c, additive_logscore=self.additive_logscore,
                disc_prior=self.disc_prior, learn_dma=self.learn_dma,
                seasonal_prior=self.seasonal_prior,
                regression_prior=getattr(self, "regression_prior", None),
                adapt_guard=self.adapt_guard,
                disc_init=getattr(self, "disc_init", None),
                clip=getattr(self, "clip", None), **_dp)
        return self._static

    def scan_filter(self, ys, wing_centre=None, component_priors=None,
                    error_nu0=None, return_trace=False, regressors=None):
        """Advance (or initialise) the streaming carry over ``ys`` ``(T, q)``
        time-major. The first call must include the ``warmup`` window (the
        diffuse prior is elicited from ``ys[:warmup]``, exactly as the batch
        path).

        ``wing_centre`` / ``component_priors`` warm-start a fresh carry from
        siblings (``AutoFFSUniverse.add_series``): the wing centre and the DLM
        state prior respectively. Ignored once the carry exists.

        ``return_trace`` also returns the per-worker one-step trace
        ``(F, Q)`` ``(T, q, M)`` over ``ys`` — what the union DMA is driven over
        to build its carry across the fit window (matches ``cv_trajectory``'s
        ``f1_full``/``q1_full``)."""
        from DLMAX.ffs.discount_grid import grid_stream_carry0, grid_stream_scan
        a = jnp.asarray(np.asarray(ys, dtype=float))
        T = a.shape[0]
        self.q = a.shape[1]
        static = self._ensure_static()
        if self._carry is None:
            self._carry = grid_stream_carry0(
                self._grid, static, a, self.warmup, wing_centre=wing_centre,
                component_priors=component_priors, error_nu0=error_nu0)
            self._t = 0
        warm = (jnp.arange(self._t, self._t + T) < self.warmup).astype(a.dtype)
        if regressors is not None:
            xs = jnp.asarray(np.asarray(regressors, dtype=float))   # (T,q,n_regs)
        elif self.n_regs and self.is_autoregressive:
            # An AR tail's design IS the y stream, so the block builds it rather
            # than making every caller re-derive it (and get the continuation
            # wrong: ``ys`` alone zero-fills the first n_regs rows of EVERY
            # chunk, which is right only for the first). An explicit
            # ``regressors`` still wins, so a caller can drive the tail by hand.
            xs = self._ar_design(a)
        else:
            xs = None
        out = grid_stream_scan(static, self._carry, a, warm, xs,
                               return_trace=return_trace)
        # after the scan: the buffer must describe observations up to and
        # including this chunk, for the next chunk's design and for the forecast
        # seed, but NOT before the design above was built from the previous one.
        self._update_lags(a)
        if return_trace:
            self._carry, (F, Q) = out
            self._t += T
            return self, (np.asarray(F), np.asarray(Q))
        self._carry = out
        self._t += T
        return self

    # -- capacity padding (persisted): the batch carry is stored at a fixed
    # `capacity` with placeholder slots + an active mask, so the jitted kernels
    # compile once and add_series fills a slot (no reshape, zero recompiles) --
    def pad_to(self, cap):
        """Pad the carry's series axis to ``cap`` with placeholder slots
        (replicate slot 0). Returns the active mask (real q True). No-op if
        already at/over ``cap``."""
        import jax
        if self._carry is None or self.q >= cap:
            return np.ones(self.q or 0, dtype=bool)
        pad = cap - self.q
        cc, hier, Wc = self._carry

        def rep(x, ax):
            r = jnp.repeat(jnp.take(x, jnp.array([0]), axis=ax), pad, axis=ax)
            return jnp.concatenate([x, r], axis=ax)
        self._carry = (jax.tree_util.tree_map(lambda x: rep(x, 1), cc),
                       jax.tree_util.tree_map(lambda x: rep(x, 0), hier), rep(Wc, 0))
        if self._lags is not None:                  # (n_regs, q): pad the series axis
            l = np.asarray(self._lags)
            self._lags = np.concatenate(
                [l, np.repeat(l[:, :1], pad, axis=1)], axis=1)
        active = np.concatenate([np.ones(self.q, bool), np.zeros(pad, bool)])
        self.q = cap
        return active

    def set_slot(self, idx, other):
        """Fill series slot ``idx`` with ``other``'s slot 0 (a freshly-fit single
        series) — add_series filling an inactive placeholder. No reshape."""
        import jax
        cc, hier, Wc = self._carry
        cc2, hier2, Wc2 = other._carry
        self._carry = (
            jax.tree_util.tree_map(lambda a, b: a.at[:, idx].set(b[:, 0]), cc, cc2),
            jax.tree_util.tree_map(lambda a, b: a.at[idx].set(b[0]), hier, hier2),
            Wc.at[idx].set(Wc2[0]))
        if self._lags is not None and other._lags is not None:
            self._lags = np.asarray(self._lags).copy()
            self._lags[:, idx] = np.asarray(other._lags)[:, 0]
        return self

    def fwd_filter(self, yt, return_trace=False, regressors=None):
        """Advance one origin. ``yt`` ``(q,)`` — one time-step across series.

        ``return_trace`` also returns the per-worker one-step ``(F, Q)`` ``(q, M)``
        used for ``yt`` (identical to the scan's trace) — what the union DMA is
        driven over in the update path so it matches the fit path exactly."""
        from DLMAX.ffs.discount_grid import grid_stream_step
        if self._carry is None:
            raise RuntimeError("GridBlock.fwd_filter before scan_filter (no carry).")
        warm = 1.0 if self._t < self.warmup else 0.0
        y = np.asarray(yt, dtype=float)
        if regressors is not None:
            xt = jnp.asarray(np.asarray(regressors, dtype=float))       # (q, n_regs)
        elif self.n_regs and self.is_autoregressive:
            # this step's lag row is [y_{t-1}, y_{t-2}, ...] — the buffer, which
            # holds observations up to t-1, reversed to most-recent-first.
            xt = jnp.asarray(self.seed_lags)
        else:
            xt = None
        out = grid_stream_step(
            self._ensure_static(), self._carry,
            jnp.asarray(y), warm, xt, return_trace=return_trace)
        self._update_lags(y[None, :])
        if return_trace:
            self._carry, F, Q = out
            self._t += 1
            return self, (np.asarray(F), np.asarray(Q))
        self._carry = out
        self._t += 1
        return self

    def forecast(self, h, exog_future=None, seed_lags=None):
        """h-step forecast from the current carry (no re-filtering). Returns
        ``(loc, sd, components)`` — ``(q, h)`` combined + the per-worker
        predictives (``LOCc``/``QHc``/``NUc``/``Wc``) for the union DMA.

        ``exog_future`` ``(q, h, n_regs)`` supplies an EXOGENOUS tail's future
        rows. ``seed_lags`` ``(q, n_regs)`` instead drives an AUTOREGRESSIVE tail
        -- the known observations ``[y_t, y_{t-1}, ...]``, most recent first --
        through the iterated-expectation forecast, because horizon j's row there
        contains forecasts made at horizons below it.

        An AR block SEEDS ITSELF from the observations it has filtered, so
        ``seed_lags`` need only be passed to override them. That is what makes a
        reopened universe able to forecast an AR tail at all: the lags are
        persisted with the block, and nothing outside it knows the y stream.
        The default applies only to an autoregressive tail — defaulting it for an
        EXOGENOUS one would quietly feed the model's own forecasts back in as
        though they were the caller's regressors."""
        from DLMAX.ffs.discount_grid import grid_stream_forecast
        if self._carry is None:
            raise RuntimeError("GridBlock.forecast before scan_filter (no carry).")
        xh = (None if exog_future is None
              else jnp.asarray(np.asarray(exog_future, dtype=float)))
        if seed_lags is None and xh is None and self.n_regs and self.is_autoregressive:
            seed_lags = self.seed_lags
        sl = (None if seed_lags is None
              else jnp.asarray(np.asarray(seed_lags, dtype=float)))
        return grid_stream_forecast(self._ensure_static(), self._carry, h, xh, sl)

    def append_series(self, other):
        """Concatenate another GridBlock's carry along the series (q) axis.

        Valid because series are independent in the wing grid (the vmap is over
        series; the family/wingman DMA is per-series — no cross-series coupling),
        so a series fit alone and appended is identical to having fit it in the
        batch. Both blocks must share the same grid. Used by
        ``AutoFFSUniverse.add_series`` to fold a warm-started late-launcher into
        an existing batch. Each block must be filtered to the same calendar
        origin (their own full histories to 'now')."""
        import jax
        if other._carry is None:
            return self
        if self._carry is None:
            self._carry, self.q, self._t = other._carry, other.q, other._t
            self._lags = other._lags
            return self
        cc, hier, Wc = self._carry
        cc2, hier2, Wc2 = other._carry
        cc_new = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=1), cc, cc2)   # (nf, q, ..)
        hier_new = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0), hier, hier2)  # (q, ..)
        Wc_new = jnp.concatenate([Wc, Wc2], axis=0)                  # (q, M)
        self._carry = (cc_new, hier_new, Wc_new)
        if self._lags is not None and other._lags is not None:
            # (n_regs, q): the series axis grows with the carry's. Both blocks are
            # filtered to the same origin, so the rows line up in time.
            self._lags = np.concatenate(
                [np.asarray(self._lags), np.asarray(other._lags)], axis=1)
        self.q = (self.q or 0) + (other.q or 0)
        self._t = max(self._t, other._t)
        return self

    def save(self, fname, group="grid_state", mode="w"):
        """Round-trip the streaming carry to HDF5. The static config (padded
        models + wing hyperparams) is rebuilt from the grid on load, so only the
        dynamic carry (a pytree of arrays), ``t_ingested`` and ``q`` persist.

        ``group``/``mode`` let several blocks share one batch file for the
        multi-block universe (each writes its own ``/blocks/<i>`` group, the
        first with ``mode='w'`` and the rest ``'a'``). Defaults are byte-identical
        to the single-block ``/grid_state`` layout."""
        import pickle
        import h5py
        import jax
        if self._carry is None:
            raise RuntimeError("GridBlock.save before scan_filter (no carry).")
        leaves, treedef = jax.tree_util.tree_flatten(self._carry)
        with h5py.File(fname, mode) as f:
            if group in f:
                del f[group]
            g = f.create_group(group)
            g.attrs["treedef"] = np.void(pickle.dumps(treedef))
            g.attrs["t"] = int(self._t)
            g.attrs["q"] = int(self.q) if self.q is not None else -1
            g.attrs["n_leaves"] = len(leaves)
            for i, lf in enumerate(leaves):
                g.create_dataset(f"leaf_{i}", data=np.asarray(lf))
            # AR trailing observations, (n_regs, q). Its OWN dataset rather than a
            # carry leaf: the lags are data, not filter state, so the carry pytree
            # keeps its structure and every block saved before this still loads.
            if self._lags is not None:
                g.create_dataset("lags", data=np.asarray(self._lags))

    def load(self, fname, group="grid_state"):
        """Restore a carry saved by :meth:`save` into this block (its grid must
        match the one that produced the carry). ``group`` selects the block's
        subgroup in a shared multi-block batch file."""
        import pickle
        import h5py
        import jax
        with h5py.File(fname, "r") as f:
            g = f[group]
            treedef = pickle.loads(g.attrs["treedef"].tobytes())
            self._t = int(g.attrs["t"])
            q = int(g.attrs["q"])
            self.q = q if q >= 0 else None
            leaves = [jnp.asarray(g[f"leaf_{i}"][()]) for i in range(int(g.attrs["n_leaves"]))]
            # absent for a structural block, and optional in the stored format
            # -> None, which is the no-lag-buffer case.
            self._lags = np.asarray(g["lags"][()]) if "lags" in g else None
        self._carry = jax.tree_util.tree_unflatten(treedef, leaves)
        self._ensure_static()
        return self

    def __repr__(self):
        q = "unfitted" if self.q is None else f"q={self.q}"
        return (f"GridBlock(period={self.period}, families={self.n_classes}, "
                f"workers={self.nm}, {q})")


class AdaptiveBlock(GridBlock):
    """An adaptive-discount grid assembled from a user's ``Wing`` cells.

    ``GridBlock`` builds its family taxonomy internally (via ``build_grid``);
    ``AdaptiveBlock.from_cells`` instead assembles the grid from ``Wing``
    ``uv_dlm`` cells the user has compiled — one cell per family — so a user can
    roll their own adaptive grid transparently and hand it to
    ``AutoFFS(blocks=[...])``. Behaviourally identical to a ``GridBlock`` over
    the same families ("adaptive" = SGD-learned discounts).
    """

    @classmethod
    def from_cells(cls, cells, *, warmup=None, offset=None,
                   pdr: float = DMA_PDR, mdr: float = DMA_MDR, dma_c: float = DMA_C,
                   disc_prior=None, seasonal_prior=None, regression_prior=None,
                   learn_dma: bool = False,
                   dma_prior=None, additive_logscore: bool = False,
                   adapt_guard=ADAPT_GUARD_DEFAULT, use_cell_disc: bool = True,
                   clip=None):
        """Build from compiled ``Wing`` cells (each a ``uv_dlm`` with a
        ``disc_rate=Wing(...)`` discount). Each cell contributes one family (its
        ``GridModel`` + components). ``warmup``/``offset`` default to the cells'
        own values (assumed uniform across the cells).

        ``disc_prior`` / ``seasonal_prior`` / ``regression_prior`` /
        ``learn_dma`` / ``dma_prior`` /
        ``additive_logscore`` / ``adapt_guard`` are the same knobs
        :meth:`GridBlock.build` takes, and default the same way — so a
        hand-assembled block can carry the published WING spec instead of
        silently running un-regularised with fixed DMA rates.

        ``use_cell_disc`` (default True) starts each discount block at the
        ``disc_rate`` its component was compiled with, and β at the error
        spec's, instead of the wing's uniform 0.95 / 0.99. Without it the cells'
        own discounts are silently discarded — fatal for a long-period seasonal,
        which cannot be started at 0.95 and learned back (an annual Fourier
        grows its covariance by ``(1/0.95)**8766`` over one cycle before the
        learner has a chance). Pass False for the old uniform start."""
        cells = list(cells)
        if not cells or any(getattr(c, "_wing", None) is None for c in cells):
            raise ValueError(
                "AdaptiveBlock.from_cells expects compiled Wing cells — build each "
                "with a component whose disc_rate is Wing(...), then compile().")
        grid = []
        for i, c in enumerate(cells):
            w = c._wing
            if "comps" not in w:
                raise ValueError("Wing cell is missing its components; recompile it.")
            grid.append((f"cell{i}", w["gm"], w["comps"]))
        w0 = cells[0]._wing
        off = float(w0["offsets"][-1]) if offset is None else float(offset)
        wu = int(w0["warmup"]) if warmup is None else int(warmup)
        di = _cells_disc_init(cells, grid) if use_cell_disc else None
        return cls(grid, period=None, warmup=wu, offset=off, pdr=pdr, mdr=mdr,
                   dma_c=dma_c, disc_prior=disc_prior, seasonal_prior=seasonal_prior,
                   regression_prior=regression_prior,
                   learn_dma=learn_dma, dma_prior=dma_prior,
                   additive_logscore=additive_logscore, adapt_guard=adapt_guard,
                   disc_init=di, clip=clip)


def _cells_disc_init(cells, grid):
    """``(nf, P)`` logit-space starting discounts read off the compiled cells.

    Block ``b`` is component ``b`` — :func:`discount_grid._grid_model` walks the
    components in order, one discount block each (the only exception,
    ``decouple_trend``, is unreachable from ``DLM.compile``). A ``Wing``
    ``disc_rate`` contributes its ``init``. β (the last slot) comes from the
    cell's own wing carry, which already holds the error spec's rate in logit
    space. Blocks past a cell's component count are padding — inert, δ frozen at
    1 — so their value is arbitrary; they take the level's.
    """
    nb = max(int(m.n_blocks) for _n, m, _c in grid)
    rows = []
    for cell, (_n, _m, comps) in zip(cells, grid):
        vals = []
        for c in comps:
            dr = getattr(c, "disc_rate", None)
            dr = getattr(dr, "init", dr)               # Wing -> its centre
            vals.append(float(np.mean(np.asarray(dr, dtype=float))))
        vals += [vals[0]] * (nb - len(vals))           # inert padding blocks
        beta = float(np.asarray(cell._wing["carry"][5])[0, 0, -1])   # already logit
        rows.append([float(np.log(v) - np.log1p(-v)) for v in vals[:nb]] + [beta])
    return np.asarray(rows, dtype=float)

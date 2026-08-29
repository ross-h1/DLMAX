"""The ``Block`` protocol — the unit the orchestration layer drives.

A *block* is the composable model unit that ``AutoFFS`` (fit over a fixed
window) and ``AutoFFSUniverse`` (sequential update with persistence and
unbalanced panels) orchestrate. The orchestrators consume a *list* of
blocks combined by a top-level DMA and never construct models themselves;
model construction lives in the factories that emit blocks.

The target face
---------------
A block is defined by the face it presents upward, **not** by having a
:class:`~DLMAX.dlm_core.multi_model_dlm` inside it:

    For each series x horizon, a block returns a predictive distribution on the
    common *observation* scale, plus a running class score so the top-level DMA
    can weight the whole block against the others.

:class:`~DLMAX.dlm_core.multi_model_dlm` is the *default engine* behind a block
(a static structural universe, or a log-transform block). The adaptive-discount
grid's RTRL engine and a future DGLM / coupled-state filter are other engines
presenting the same face.

A block produces its predictive by one of three internal modes, all presenting
the same face:

* a **single model**;
* an **internal DMA** over competing members (static universe; the grid's
  families -> wingmen);
* **structural composition** of sub-models into one predictive — e.g. a DCMM
  hurdle (Bernoulli occurrence x conditional Poisson/Gaussian amount) composing
  into one zero-inflated predictive. Composition multiplies components into a
  joint likelihood; it is *not* averaging and does not appear in the block list
  as competing entries.

What a block owns
-----------------
The design draws the protocol so the orchestrator stops reaching into block
internals. A block owns:

1. its **predictive family** (Gaussian / Student-t / lognormal / negative
   binomial / ...);
2. its **kernel / update rule** (SVD, QR, conjugate-guide DGLM, inclusion
   moves);
3. its **membership** (add / remove / gather members) — a static block slices
   its series axis; a hierarchical-coupled block cannot be sliced and
   re-elicits instead;
4. its **batching unit** — length-grouping is an orchestrator optimisation that
   assumes independent series; a coupled block is its own batch.

Heterogeneity comes in two forms: *within* a block (flat-batch members varying
over discount / var_power / family via select-not-branch, one ``vmap``, keeping
the packing win) and *across* blocks (the list — genuinely different engines or
data ingestion). The list is for heterogeneity, never for splitting
homogeneous work.

Scope of the protocol below
---------------------------
It is typed to the filter / forecast / persistence face, which is the part
every engine shares. The richer face the design describes above — a
family-carrying ``Predictive`` return, the class score, the membership methods
— is not part of the protocol, so ``Block`` stays satisfiable by any engine
presenting the common face. It is deliberately not exported in the top-level
``DLMAX`` public ``__all__``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Block(Protocol):
    """Structural interface a model block presents to the orchestration layer.

    Runtime-checkable: ``isinstance(obj, Block)`` verifies the presence of the
    members below. It checks names, not signatures, so it is a structural
    contract rather than a typed one.

    Members
    -------
    q : int
        Number of series carried by the block.
    nm : int
        Number of packed candidate models (the DMA member axis). ``1`` for a
        single-model block.

    Notes
    -----
    ``fwd_filter`` / ``scan_filter`` ingest observations (single step / whole
    series), advancing the block's filter state in place; ``forecast`` emits the
    block's h-step predictive; ``save`` / ``load`` round-trip the block's state
    to HDF5. Engines that expose forecast *variants* (e.g. the AR/TVAR
    ``forecast_iterated`` and exogenous ``forecast_exog`` on
    :class:`multi_model_dlm`) satisfy the protocol through ``forecast``; the
    variant dispatch is an engine-internal detail behind that single face.
    """

    q: int
    nm: int

    def fwd_filter(self, yt):
        """Advance the block's filter state by one observation vector."""
        ...

    def scan_filter(self, ys):
        """Advance the block's filter state over a whole series block."""
        ...

    def forecast(self, h):
        """Emit the block's h-step-ahead predictive."""
        ...

    def save(self, fname):
        """Persist the block's state to ``fname`` (HDF5)."""
        ...

    def load(self, fname):
        """Restore the block's state from ``fname`` (HDF5)."""
        ...
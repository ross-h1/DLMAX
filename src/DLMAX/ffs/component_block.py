"""``ComponentBlock`` — a block built from the component DSL, not the standard grid.

Wrap a ``DLM`` you built with the component builder and drop it straight into
``AutoFFS(blocks=[...])`` — no ``universe_builder`` plumbing:

    dlm = (DLM(family="Gaussian")
           .add_component(LocalTrend(name="trend", disc_rate=[0.9, 0.95, 0.99]))
           .set_error())
    AutoFFS(blocks=[ComponentBlock(dlm)]).cross_validation(wide, h=6)

A list-valued spec (e.g. a discount grid) becomes the block's DMA'd model set; a
fully-scalar spec is a single-model block. This is the "same face at every level"
rung: components -> DLM -> block -> AutoFFS.

Under the hood it derives a ``universe_builder`` from the DLM and routes through
the same ``StaticBlock`` machinery, so it is bit-identical to constructing the
equivalent ``StaticBlock(universe_builder=...)`` by hand. ``n_series`` is taken
from the batch, so the same block drives batches of any width.
"""

from __future__ import annotations

import pandas as pd

from DLMAX.ffs.static_block import StaticBlock


class ComponentBlock(StaticBlock):
    """A block whose model set is a DSL-built ``DLM`` (single model or a sweep).

    Parameters
    ----------
    dlm : DLMAX.ffs.dlm_builder.DLM
        A DLM specification (components + error). ``n_series`` need not be set —
        it is filled per batch from the data.
    warmup : int
        Warmup steps (diffuse-prior settle), like any block.
    dma_pdr, dma_mdr : float
        The block's within-/between-class DMA forgetting for its own model set.
    """

    def __init__(self, dlm, *, warmup: int = 0, dma_pdr: float = 0.90,
                 dma_mdr: float = 0.90):
        self._dlm = dlm
        super().__init__(
            season_length=None, n_seas_comps=None, warmup=warmup,
            dma_pdr=dma_pdr, dma_mdr=dma_mdr, monitor_tau=None,
            universe_builder=self._universe_builder,
        )

    def _has_sweep(self) -> bool:
        dlm = self._dlm
        if getattr(dlm, "_monitor_kind", None) == "list":
            return True
        if any(c._swept_dims() for c in dlm.components):
            return True
        if dlm.error_spec is not None and dlm.error_spec._swept_dims():
            return True
        return False

    def _universe_builder(self, init_data, h, ctx):
        """The ``(init_data, h, ctx) -> (models, model_desc)`` seam, derived from
        the DLM. Sets ``n_series`` from the batch; a swept spec compiles a
        universe, a scalar spec a single model; tags a single DMA ``Class``."""
        self._dlm.n_series = init_data.shape[1]
        warmup = getattr(ctx, "warmup_steps", None)
        if self._has_sweep():
            models, desc = self._dlm.compile_universe(
                init_data, h=h, warmup_steps=warmup)
            desc = desc.copy()
        else:
            model = self._dlm.compile(init_data, h=h, warmup_steps=warmup)
            models = {"m": model}
            desc = pd.DataFrame([{"key": "m"}])
        # One DMA class for the block's own model set (within-class pooling over
        # the sweep). The orchestrator's union DMA still competes this block's
        # models against the other blocks'.
        desc["Class"] = 0
        return models, desc

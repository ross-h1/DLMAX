"""The ``Universe`` — the model set the orchestration layer drives.

A universe is a *list of blocks* combined by a top-level DMA allocator. It is
the currency that ``AutoFFS`` (fit over a fixed window) and ``AutoFFSUniverse``
(sequential update with persistence) consume: the orchestrators drive a
universe through time and run its allocator, and never construct models
themselves.

With a single block, ``model_indicator`` is that block's flat within-block
``Class`` grouping and the DMA has one level. With several, the indicator is
derived from block membership and the DMA becomes two-level: within a block,
then across blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from DLMAX.ffs.block import Block


@dataclass
class Universe:
    """A model universe: a list of blocks plus the DMA allocator over them.

    Parameters
    ----------
    blocks : list[Block]
        The blocks combined by ``dma``. One element for a single universe (the
        packed ``multi_model_dlm``, or one grid); several to combine engines.
    dma : Allocator
        The dynamic model averaging allocator combining the blocks' forecasts.
    model_indicator : optional
        The model->class indicator (a DataFrame) wiring the DMA between-class
        layer. Carried through from construction for a single block; derived
        from block membership when there are several.
    model_desc : optional
        The per-model descriptor (a DataFrame, ``Class`` column) from the
        universe builder, when available.
    """

    blocks: List[Block]
    dma: Any
    model_indicator: Any = None
    model_desc: Any = None

    @property
    def multi(self) -> Block:
        """The sole block, for callers that assume a single-block universe.

        Raises when the universe holds more than one, since such a caller has
        no way to express which block it meant.
        """
        if len(self.blocks) != 1:
            raise ValueError(
                "Universe.multi is only defined for a single-block universe; "
                f"this one has {len(self.blocks)} blocks."
            )
        return self.blocks[0]

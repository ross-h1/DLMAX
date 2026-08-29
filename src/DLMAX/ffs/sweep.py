"""Cartesian-product sweep over named dimensions, with optional constraint.

Universe-agnostic. Knows nothing about DLMs, components, or models.
Used by ``DLM.compile_universe`` (and intended to be reusable for
variable selection / hyperparameter sweeps in future).
"""

from itertools import product
from typing import Callable, Iterator, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Dimension descriptor
# ---------------------------------------------------------------------------


class SweepDim:
    """One axis of the Cartesian product.

    Parameters
    ----------
    component : str
        Top-level group key in the cell dict. For DLM use, this is the
        component name (e.g. "trend", "seasonal") or "error" for the
        error spec.
    param : str
        Parameter name within the group (e.g. "disc_rate", "damping").
    values : list
        Discrete options for this dimension. Length-1 is fine; empty
        is rejected at construction.
    short : str, optional
        Short tag used in the cell key. Defaults to the first letter
        of ``param``, capitalised.
    """

    def __init__(
        self,
        component: str,
        param: str,
        values: list,
        short: Optional[str] = None,
    ):
        if not isinstance(values, list):
            raise TypeError(
                f"SweepDim({component}.{param}): values must be a list, "
                f"got {type(values).__name__}."
            )
        if len(values) == 0:
            raise ValueError(
                f"SweepDim({component}.{param}): values is empty — at "
                "least one option required."
            )
        self.component = component
        self.param = param
        self.values = values
        self.short = short if short is not None else param[0].upper()

    def __repr__(self):
        return f"SweepDim({self.component}.{self.param}, " f"n={len(self.values)})"


# ---------------------------------------------------------------------------
# Cell key formatting
# ---------------------------------------------------------------------------


def default_keyfn(cell: dict, dims: Sequence[SweepDim], indices: Sequence[int]) -> str:
    """Default cell key: ``trend.D0.L1|seasonal.D2|error.D0.P0``.

    Components are grouped; inside a group, params are joined by ".";
    groups are joined by "|". Each (param, idx) becomes ``{short}{idx}``.
    """
    by_component: dict[str, list[str]] = {}
    for dim, idx in zip(dims, indices):
        by_component.setdefault(dim.component, []).append(f"{dim.short}{idx}")
    return "|".join(f"{comp}.{'.'.join(tags)}" for comp, tags in by_component.items())


# ---------------------------------------------------------------------------
# Sweep generator
# ---------------------------------------------------------------------------


def sweep(
    dims: Sequence[SweepDim],
    constraint: Optional[Callable[[dict], bool]] = None,
    keyfn: Callable = default_keyfn,
) -> Iterator[Tuple[str, dict, Tuple[int, ...]]]:
    """Yield ``(key, cell, indices)`` for each accepted Cartesian-product point.

    ``cell`` is a nested dict ``{component: {param: value}}`` with the
    resolved scalar value at each (component, param) position.
    ``indices`` is the per-dim index tuple, parallel to ``dims``.

    If ``constraint`` is supplied, it is called with ``cell`` and the
    cell is yielded only if the callable returns truthy.
    """
    for indices in product(*(range(len(d.values)) for d in dims)):
        cell: dict[str, dict] = {}
        for dim, idx in zip(dims, indices):
            cell.setdefault(dim.component, {})[dim.param] = dim.values[idx]
        if constraint is not None and not constraint(cell):
            continue
        yield keyfn(cell, dims, indices), cell, tuple(indices)

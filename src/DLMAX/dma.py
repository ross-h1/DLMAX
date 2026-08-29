"""Allocator-side helpers for the DMA path.

Utilities for constructing the inputs the ``Allocator`` consumes from
``DLM.compile_universe`` output. Kept separate from ``ffs_core`` so
non-FFS callers (e.g. paper-side factories built on
``compile_universe``) don't pull in the FFS-specific machinery.

Provides:

- ``make_model_indicator`` — build the (n_models × n_classes)
  indicator matrix the ``Allocator`` consumes.
- ``as_forecast_bundle`` — wrap the raw ``(f, q)`` tuple output of
  ``multi_model_dlm.fwd_filter`` / ``forecast`` as a ``ForecastBundle``
  with the variance-to-SD conversion the scoring rules expect.
"""

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import jax.numpy as jnp

from DLMAX.dlm_core import ForecastBundle


def make_model_indicator(
    descriptor: pd.DataFrame,
    by: Optional[Union[str, Callable[[pd.Series], object]]] = None,
) -> pd.DataFrame:
    """Build the (n_models × n_classes) model-indicator matrix for the DMA allocator.

    The model indicator tells the allocator which cells to group together
    for class-level averaging. Pass the resulting DataFrame's ``.values``
    to ``Allocator(model_indicator=..., ...)``.

    Parameters
    ----------
    descriptor : pd.DataFrame
        One row per cell. Typically the second return value of
        ``DLM.compile_universe`` or ``assemble_models``.
    by : str or callable, optional
        - ``None`` (default): single class with all cells assigned to it.
          Returns a single column of ones. Use when no class structure
          is wanted (e.g. a small power-sweep universe).
        - ``str``: name of a column in ``descriptor`` whose values are
          used directly as class labels. E.g. ``by="Class"`` reproduces
          the legacy ``assemble_models`` schema; ``by="error.power"``
          groups cells by variance-power on a ``compile_universe``
          descriptor.
        - ``callable``: applied per-row via ``descriptor.apply(by, axis=1)``.
          The return value is the class label.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_models, n_classes)`` with bool dtype (matching the
        legacy ``pd.get_dummies`` output that ``init_alloc_state``'s
        boolean-masking logic depends on). Indexed by
        ``descriptor["key"]`` if that column exists, else by
        ``descriptor.index``. Row order matches ``descriptor`` row order,
        which matches ``compile_universe``'s sweep iteration order and
        therefore the stacking order of ``multi_model_dlm(models)``.

    Examples
    --------
    Single class (toy / smoke-test universe)::

        models, desc = dlm.compile_universe(panel)
        mi = make_model_indicator(desc)
        # mi.shape == (len(models), 1); all values 1

    Group by a descriptor column::

        mi = make_model_indicator(desc, by="error.power")
        # one column per distinct error.power value

    Custom class assignment::

        def class_fn(row):
            return "additive" if row["error.power"] == 1.0 else "log"
        mi = make_model_indicator(desc, by=class_fn)

    Legacy schema (assemble_models output)::

        ATS, model_desc = assemble_models(...)
        mi = make_model_indicator(model_desc, by="Class")
        # equivalent to pd.get_dummies(model_desc["Class"])
    """
    n = len(descriptor)
    if "key" in descriptor.columns:
        idx = pd.Index(descriptor["key"], name="key")
    else:
        idx = descriptor.index

    if by is None:
        return pd.DataFrame(
            np.ones((n, 1), dtype=bool),
            index=idx,
            columns=["all"],
        )



    if isinstance(by, str):
        if by not in descriptor.columns:
            raise KeyError(
                f"make_model_indicator: column {by!r} not in descriptor. "
                f"Available columns: {list(descriptor.columns)}."
            )
        classes = descriptor[by]
    elif callable(by):
        classes = descriptor.apply(by, axis=1)
    else:
        raise TypeError(
            f"make_model_indicator: 'by' must be None, str, or callable; "
            f"got {type(by).__name__}."
        )

    dummies = pd.get_dummies(classes)
    dummies.index = idx
    return dummies

def as_forecast_bundle(
    f,
    q,
    *,
    n_models: int,
    n_series: int,
    h: int = 1,
) -> ForecastBundle:
    """Wrap raw ``(f, q)`` filter / forecast output as a ``ForecastBundle``.

    The filter / forecast methods now return a :class:`ForecastBundle`
    directly; this helper remains for wrapping *raw* ``(f, q)`` arrays (e.g.
    reshaping a flat one-step output to the canonical 3-D layout). Both ``f``
    and ``q`` encode the predictive mean and **variance**; the bundle stores
    the variance and exposes ``sqrt`` via its ``sd`` property, so this helper
    just reshapes to ``(n_models, n_series, h)``.

    Parameters
    ----------
    f, q : jnp.ndarray
        Predictive mean and variance from
        ``multi_model_dlm.fwd_filter`` (1-step, flat) or
        ``multi_model_dlm.forecast(h)`` (h-step, 3D).
    n_models : int
        Number of models in the multi-model filter
        (``multi.nm``).
    n_series : int
        Number of series (``multi.q``).
    h : int, default 1
        Forecast horizon.

    Returns
    -------
    ForecastBundle
        ``loc`` and ``var`` each shaped ``(n_models, n_series, h)``.

    Examples
    --------
    Reshaping raw one-step moments into the canonical 3-D bundle::

        bundle = as_forecast_bundle(
            f_flat, q_flat, n_models=multi.nm, n_series=multi.q, h=1
        )
        weights = dma.update(bundle, obs=y_t)
    """
    f = jnp.asarray(f).reshape(n_models, n_series, h)
    q = jnp.asarray(q).reshape(n_models, n_series, h)
    return ForecastBundle(loc=f, var=q)
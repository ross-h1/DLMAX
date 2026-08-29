"""DLMAX — Dynamic Linear Models in JAX.

DLMAX provides Bayesian dynamic linear model (DLM) forecasting with dynamic
model averaging (DMA), exposed through a high-level fit / update / predict
API over long-format DataFrames.

The supported public API is re-exported here and enumerated in ``__all__``.
Importing directly from submodules (``DLMAX.ffs_core``, ``DLMAX.dlm_core``,
...) continues to work but is not part of the API contract and may change
without notice.

Quick start
-----------
``df_history`` is long format — columns ``(unique_id, ds, y)``::

    >>> from DLMAX import AutoFFS
    >>> model = AutoFFS(season_length=12).fit(df_history)
    >>> forecast = model.predict(h=12, level=[80, 95])

``predict`` returns one row per ``(unique_id, ds)`` with the point forecast,
its predictive SD, and the requested interval bounds. To extend the model as
data arrives, ``model.update(df_new)`` costs one filter pass over the new rows
— no refit.

Three entry points, by what you need to hold:

* :class:`AutoFFS` — fit / update / predict with the state in memory, plus
  :meth:`~AutoFFS.cross_validation` for rolling-origin backtests. Start here.
* :class:`AutoFFSUniverse` — the same model with state on disk, for panels that
  outlive the process or do not fit in memory, and for series that arrive later
  (``add_series``). Agrees with :class:`AutoFFS` bitwise.
``DLMAX.ffs_core.StaticFFS`` runs the same API over a FIXED-discount universe
instead of the wing. Reachable, but not part of the API contract above.
"""

from DLMAX.dlm_core import (
    enable_x64,
    multi_model_dlm,
    require_x64,
    uv_dlm,
)
from DLMAX.datasets import available_datasets, load_dataset
from DLMAX.ffs.devices import configure_devices
from DLMAX.ffs_core import (
    AutoFFS,
    AutoFFSUniverse,
    FFSPredictive,
    UniverseContext,
)
from DLMAX.smoother import SmootherError, ffbs, rts_smooth

__all__ = [
    # Forecasting API
    "AutoFFS",
    "AutoFFSUniverse",
    "FFSPredictive",
    "UniverseContext",
    # Core model classes
    "uv_dlm",
    "multi_model_dlm",
    # Retrospective smoothing / backward sampling. SmootherError is raised by
    # uv_dlm.smooth() / .backward_sample(), and is exported because a caller
    # cannot otherwise handle a documented failure of an exported method without
    # importing from a submodule.
    # rts_smooth / ffbs are exported but PROVISIONAL: their signatures are
    # expected to change (see the DLMAX.smoother docstring), so they are
    # explicitly carved out of the stability guarantee the rest of this list
    # carries. Prefer the uv_dlm methods where they suffice.
    "SmootherError",
    "rts_smooth",
    "ffbs",
    # Setup helpers
    "configure_devices",
    "enable_x64",
    "require_x64",
    # Example datasets
    "load_dataset",
    "available_datasets",
]

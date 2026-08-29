# DLMAX — Dynamic Linear Models in JAX

DLMAX is a JAX-based framework for **Dynamic Linear Models (DLMs)** with
**Discount Model Averaging (DMA)**. It fits and forecasts large panels of time
series under a Bayesian state-space formulation, averaging over a universe of
candidate models whose discount factors control how quickly each adapts. The
whole forward filter is written in JAX, so it vectorises across series and runs
on CPU or GPU.

DLMAX is the engine behind the paper *Forecasting: Fast and Slow*. The name
nods to the idea that a panel is best served not by one model but by a
*universe* of fast- and slow-adapting models, combined online by their
realised predictive performance.

> **Status: alpha (v0.1.0).** The numerics are tested and stable — the
> forecasts here are the ones behind the paper — but this is a `0.x` release and
> the public API may change between minor versions, per
> [semantic versioning](https://semver.org/#spec-item-4). Pin an exact version
> if you depend on it: `pip install awen-dlmax==0.1.0`.
>
> Bug reports and questions are welcome via
> [GitHub Issues](https://github.com/ross-h1/DLMAX/issues). If you hit a
> numerical problem, the most useful report includes the model spec, the shape
> of the panel, and whether 64-bit floats were enabled.

## Features

- **Component-based model builder** — compose `LocalLevel`, `LocalTrend`,
  `Fourier` (seasonal), and `Regressors` (exogenous) terms into a DLM.
- **Discount Model Averaging** — maintain a universe of models differing in
  their discount factors and combine them by online predictive score.
- **Online-learned discounts** — the default "wing" grid does not fix the
  discount factors at all: each family carries three wingmen and the centre is
  moved by online gradient descent, so the adaptation rate is learned per
  series rather than chosen. This is the method of *Forecasting: Fast and Slow*.
- **Panel-scale** — vectorised across series; length-grouped batching handles
  unbalanced panels; optional Dask distribution across a cluster.
- **Streaming API** — `fit` once, then `update` incrementally as new
  observations arrive, with HDF5 persistence of filter state.
- **Probabilistic forecasts** — Student-t predictive intervals at arbitrary
  horizons and coverage levels.
- **DataFrame interface** — long or wide input, accepted interchangeably
  (see [Input format](#input-format)).

## Installation

DLMAX requires Python ≥ 3.11. The distribution is published as
**`awen-dlmax`**; the import name is `DLMAX`.

```bash
pip install awen-dlmax
```

For a GPU build of JAX (Linux + CUDA 12):

```bash
pip install "awen-dlmax[gpu]"
```

To reproduce published results, pin the exact version they were produced under
— `pip install awen-dlmax==0.1.0`.

DLMAX runs in double precision; enable 64-bit JAX before importing anything
that builds a model:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Input format

Series are supplied as a pandas DataFrame in either of two shapes, detected
automatically.

**Long** — one row per observation, three columns:

| unique_id | ds | y |
|---|---|---|
| series_a | 2015-01-01 | 10.4 |
| series_a | 2015-02-01 | 11.1 |
| series_b | 2015-01-01 | 27.3 |

`unique_id` names the series, `ds` is the timestamp (or an integer index), `y`
is the observation. Series may be of different lengths and need not share a
calendar.

**Wide** — one row per period, one column per series:

| | series_a | series_b |
|---|---|---|
| **2015-01-01** | 10.4 | 27.3 |
| **2015-02-01** | 11.1 | 28.0 |

The index is the timestamp and the column names are the series ids. A series
that starts late or ends early is simply `NaN` in those cells; missing values
are carried through the filter as missing rather than imputed.

Both give identical results — wide input is converted to long internally, and
forecasts are returned keyed the way the input was.

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
from DLMAX import AutoFFS

# A small monthly panel in long format: columns (unique_id, ds, y).
rng = np.random.default_rng(0)
dates = pd.date_range("2015-01-01", periods=72, freq="MS")
frames = []
for sid in ["series_a", "series_b", "series_c"]:
    trend = np.linspace(0, 5, len(dates))
    seasonal = 2.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = rng.normal(0, 0.5, len(dates))
    frames.append(pd.DataFrame({
        "unique_id": sid,
        "ds": dates,
        "y": 10 + trend + seasonal + noise,
    }))
df = pd.concat(frames, ignore_index=True)

# Fit a model universe (monthly seasonality) and forecast 12 steps ahead
# with 80% and 95% predictive intervals.
model = AutoFFS(season_length=12)
model.fit(df, freq="MS")
forecast = model.predict(h=12, level=[80, 95])

print(forecast.head())
```

`predict` returns a long-format DataFrame with the point forecast and the
requested interval bounds per `(unique_id, ds)`.

### Streaming updates

When the next period's observations arrive, `update` advances the filter over
the new rows only — no refit. Every fitted series must appear, and the panel
advances together.

```python
# next month's observation for each series
next_ds = dates[-1] + pd.offsets.MonthBegin(1)
df_new = pd.DataFrame({
    "unique_id": ["series_a", "series_b", "series_c"],
    "ds": [next_ds] * 3,
    "y": [16.1, 15.4, 15.9],
})

model.update(df_new)            # one filter pass over the new rows only
forecast = model.predict(h=12)  # forecast from the advanced state
```

`fit(history)` then `update(new)` gives the same answer as `fit(history + new)`
— the filter is sequential, so how the data is fed in cannot change the result.

## Key concepts

- **`AutoFFS`** — the high-level estimator. Builds the model universe sized to
  your data and exposes `fit` / `update` / `predict` for forecasting forward,
  and `cross_validation` for rolling-origin backtests.
- **Model universe** — the set of candidate DLMs being averaged. The default is
  the **wing grid** (below). Pass `blocks=[...]` for a different model set, or
  `blocks=[StaticBlock(...)]` for a fixed-discount universe.
- **Discount factor** — governs how quickly a model forgets: 1 never forgets,
  smaller adapts faster. The wing *learns* these. Separately, `dma_pdr` /
  `dma_mdr` govern how fast the *averaging* re-weights toward
  better-performing models.
- **`AutoFFSUniverse`** — the same model with state persisted to disk. Use it
  when state must outlive the process, the panel does not fit in memory, or
  series arrive over time (`add_series`). It agrees with `AutoFFS` bitwise.

### The wing grid, in six words

These terms recur throughout the API and the source:

| term | meaning |
|---|---|
| **grid** | the whole candidate model set for a series |
| **family** | one structure in that set — a choice of trend damping, seasonality on/off, and error law (additive or multiplicative) |
| **wing** | the mechanism that learns a discount rather than fixing it: a family carries trial discounts either side of a centre, and the centre moves toward whichever is scoring better |
| **wingman** | one of those trials. Three per family: one at the centre, one either side |
| **worker** | a single model — one family at one wingman offset. A 16-family grid has 48 workers |
| **union DMA** | the top-level averaging, run over *every* worker of *every* block at once, rather than combining each block separately and then combining the combinations |

So a forecast is: each worker filters the series, the wing moves each family's
discount centre using how its own wingmen scored, and the union DMA weights all
workers by recent predictive performance to produce the reported forecast.

## Awkward data

Real panels have series that cross zero, stop and start, or sit flat through
their warmup window. [`docs/INVARIANTS.md`](docs/INVARIANTS.md) documents what
DLMAX does in each case and why — missing values are carried as *no
information* rather than imputed, the variance link is symmetric about zero,
and the diffuse prior survives a partly-missing warmup. Worth reading if your
panel has any of those, and before changing the core filter.

## Development

```bash
git clone https://github.com/ross-h1/DLMAX.git
cd DLMAX
uv sync --extra dev      # or: pip install -e ".[dev]"
pytest                   # runs the fast suite
pytest -m slow           # also runs the slower numerical tests
```

Some tests exercise a private M1-competition dataset and are skipped unless
`DLMAX_M1M_PATH` points at a local copy; the default suite needs no external
data.

## Citation

If you use DLMAX or its methods in academic work, please cite the accompanying
paper. See [`CITATION.cff`](CITATION.cff), or use GitHub's *Cite this
repository* button.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

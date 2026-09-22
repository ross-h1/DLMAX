# Changelog

Notable changes to DLMAX (published on PyPI as `awen-dlmax`).

Every release carries a **Numerical behaviour** section. SemVer describes the
API and says nothing about the change that actually costs you: a release can
keep every signature identical and still move every number. Anything that can
change a forecast is recorded there, with the affected path named.

## 0.3.0

### Added

- **`fwd_filter(yt)` accepts a labelled observation** on `AutoFFS` and
  `AutoFFSUniverse` — a `pd.Series` keyed by series id, or the one-row frame
  `df.iloc[[t]]`. `df.iloc[t]` is already a Series indexed by the ids and
  *named* by the timestamp, so stepping a frame now carries identification and
  calendar without being asked for them separately:

  ```python
  for t in range(n):
      bundle = model.fwd_filter(wide.iloc[t])   # ids and dates come along
  ```

  Three things follow.

  **Matching is by label.** The observation is reordered onto the fitted ids
  under the same two-sided check `update(df_new)` enforces: every fitted series
  must appear, no unfitted one may. Column order can no longer silently
  mis-assign — the failure the positional form cannot see. An array is still
  taken in fitted (or manifest) order and costs no alignment, so it remains the
  fast path for a per-step loop.

  **Self-initialisation adopts the labels.** The warmup buffer previously named
  series `s0`… and counted the calendar from 0, because it had been handed a
  bare array; given labels it uses them, and infers `freq` from the buffered
  timestamps. This is the only route for a **ragged panel**, where `fit` is not
  available at all: `fit` requires one shared calendar and an observation for
  every series in the window, which a warmup-length slice of a ragged frame does
  not give. Passing the frame to `fit` first remains the way to name a balanced
  panel.

  **Below three timestamps the integer calendar is kept.** One gap does not
  identify a calendar frequency — 91 days is `QS-DEC`, or `QE`, or nothing — so
  a short warmup adopts the ids and declines to guess the dates.

- **`warmup_steps` is now honoured by the engines, not merely stamped on them.**
  `dlm_builder` had written `result.warmup_steps` onto every compiled model
  since April and nothing in `dlm_core` ever read it. Now:

  - `uv_dlm` takes `warmup_steps` as a constructor argument and counts its own
    observations, so `fwd_filter` applies the window without being told. It had
    no warmup mechanism at all before — the gap 0.2.0's `warmup_flag` note
    described for `multi_model_dlm`, which that release closed only for `multi`.
  - `fwd_filter(..., warmup_flag=)` on both classes defaults to `None`, meaning
    "use my own window"; an explicit `0.0`/`1.0` still forces the step. Explicit
    wins, which is why every existing caller is unaffected.
  - `multi_model_dlm` **inherits** `warmup_steps` from the `uv_dlm` instances it
    packs, taking the max when they disagree, and persists it with its counter.
  - `multi_model_dlm.scan_filter` honours warmup for the first time. It had no
    such parameter, so every step took the default of `0.0` — while
    `fwd_filter`'s docstring asserted the scan had "always applied this per
    step". That was true of `ffs_core`'s scan builders and false of the method
    below it. It is now true of both.
  - `DEFAULT_LEARNER_WARMUP` (6) is named in `dlm_builder`. A `Wing`/`Adapt`
    model compiled without an explicit `warmup_steps` previously carried 0 on
    the model and 6 on its overlay simultaneously; one number now reaches both.
  - The RTRL overlays share the model's counter instead of keeping private ones.

- **The Quintana/West matrix-normal DLM is now part of the library**, as
  `DLMAX.mvdlm`. It had lived in a research project since August; the
  scaffolding was already here (`smoother.py` was written to serve "a `uv_dlm`
  and, in future, a matrix-normal (Quintana/West) multivariate DLM", and
  `ffbs`'s `right_factor` is the `Sigma` hook), so this connects the two ends.

  The model is `y_t' = F_t'Theta_t + nu_t`, `Theta_t = G Theta_{t-1} + Omega_t`
  with `Omega_t ~ N(0, W_t (x) Sigma)`. The `(x) Sigma` structure is what it
  buys: `C_t`, `R_t`, `Q_t` and `A_t` are common to every series **and free of
  the data**, so the whole covariance path is built once, for any number of
  series, and reused across a rolling origin. The price is that the design must
  be common to all series — which is the Zellner SUR condition, and is exactly
  what a VAR gives you.

  - **It speaks the library's vocabulary.** `fwd_filter(yt)` for one step,
    `scan_filter()` for many, `forecast(h)` returning a `ForecastBundle` —
    the same three names the engines and the FFS blocks use for the same three
    things. The functional shims (`dlm_params`, `dlm_forward`,
    `dlm_back_sample`, `dlm_forecast`) keep their exact signatures and return
    shapes, including `dlm_forecast`'s `var` being the state term with no
    observation half.

  - **A time-varying design**, so known regressors and a VAR both work:
    `_build_covariance_path(..., X=)` drives the last `k` slots of `F_t` per
    step. `X=None` reproduces the constant-`F` path bitwise, and a *constant*
    `X` reproduces folding that `x` into `F` — two routes to one model, checked
    against each other. What does not survive is reuse across origins: the path
    becomes a function of the regressor window, so it must be keyed on it.

  - **`forecast_sample(h, ...)` forecasts a VAR by simulation**, because the
    Kronecker structure does not survive iteration. For a VAR(1) with `Theta`
    known, `Var(y_{t+2} | y_t, Theta) = Sigma + B Sigma B'`, which is not a
    multiple of `Sigma` — so the predictive is `Q_1 Sigma` at one step and is
    not `(x) Sigma` from two steps on. `uv_dlm`'s `iterated_obs_forecast`
    returns a scalar `q_h` per series and legitimately can, its series being
    independent; a VAR's whole point is that they are not. Drawing `Theta`
    rather than propagating its moments carries the coefficient uncertainty
    for free. The in-sample phase stays closed form, and no backward smoothing
    is involved.

  - **`Sigma` is learned, by discount Wishart** — `n_t = d n_{t-1} + 1`,
    `S_t = d S_{t-1} + e_t e_t'/Q_t` — in a new `DLMAX.wishart`. This
    generalises a line the model already had, which tracked the `q` diagonal
    variances with no discount and no cross terms. Two notes:

    **It is opt-in** (`mv_dlm(..., delta_sigma=)`, default `None`). The full
    path is `(T, q, q)`; at `q = 304` over 131 periods that is ~97 MB per run,
    to hold something a caller supplying its own `Sigma` never reads. With the
    default the model runs the diagonal recursion it always ran, bit for bit,
    and `params()` gains no keys — key presence is static pytree structure, so
    an unconditional key would retrace a jitted caller's kernel.

    **The recursion carries `S_t / n_t`, not `S_t`.** Algebraically identical;
    numerically not. In that form it reproduces the existing scalar recursion
    *exactly*, so the two paths are pinned to each other bitwise rather than to
    a tolerance. Carrying `S` agrees only to 3.4e-16 — machine precision, but
    enough slack to hide a discount attached to the wrong term.

    The learned `Sigma` is surfaced whole as `sigma_scale`/`sigma_dof` and
    through `params()`, and **it feeds the one-step predictive**: choosing a
    variance discount and then reporting intervals computed from an
    undiscounted estimate would be half a feature. `scale` keeps one meaning —
    the model's estimate of `diag(Sigma)` — and `delta_sigma` chooses how it is
    estimated. At `delta_sigma = 1` that substitution is bitwise a no-op on
    both the predictive and the filtered path, and the parameter defaults to
    `None`, so nothing moves unless a discount is actually chosen.

  - **Reached as `DLMAX.mvdlm`, not through the top-level namespace.** It is
    deliberately absent from `__all__` and from the docs reference: the
    migration plan's gate for promoting it was "once the API has settled
    against real use", and it has had none outside the project it came from.
    Importing the submodule works and is supported; what is withheld is the
    implication of stability that exporting it alongside `uv_dlm` would carry.
    The same applies to `DLMAX.wishart`.

### Fixed

- **`import tqdm` in `ffs_core` broke a clean install of 0.2.0.** `tqdm` was
  never a declared dependency — it arrived through `numpyro` until that was
  removed — so `pip install awen-dlmax==0.2.0; import DLMAX` raised
  `ModuleNotFoundError`. Every environment we develop in happened to have it.
  Removed rather than declared: it appeared at one line, in a reference
  implementation with no callers.
- **The smoother no longer smooths a warmup window silently.** `smooth()` and
  `backward_sample()` pass `applied_disc`, a time-constant vector, so against a
  trajectory covering a warmup window the backward recursion rebuilt `R_t` with
  forgetting inflation those steps never had. It now reports as approximate;
  `backward_sample` raises unless `allow_approximate=True`. The error is
  confined to the window — everything from step `N` on is exact.

### Numerical behaviour

- **The warmup work moves no existing caller.** Every FFS entry point passes
  `warmup_flag` explicitly, so the new self-counting path is unreachable from
  them; `_multi_fwd_filter_step` is byte-identical; and the grid engine imports
  only free functions from `dlm_core`, never the classes, so `AutoFFS(blocks=
  [GridBlock])` and grid-mode `AutoFFSUniverse` — the M4 and M5 producers —
  cannot see any of it. Verified bitwise on the published M4 call path. What
  does change is a bare caller stepping a model compiled with `warmup_steps>0`,
  which is the case that was wrong.
- **Labels identify; they do not compute.** The array, `Series` and one-row
  `DataFrame` faces are **bitwise identical** at every step and in the resulting
  forecast. `fwd_filter` returns plain arrays in its `ForecastBundle` whatever
  goes in, so there is one return type and no per-step pandas construction.
- **`mvdlm` adopts the library's square-root convention, and only `sqrtH`
  moves.** The research implementation took an SVD and *reflected* an
  indefinite `H`; `smoother.py`'s `_sym_sqrt` clips and reports `psd_clip`. The
  library now has one convention, which is a deliberate divergence from the
  frozen oracle rather than a fix: `H` is positive semi-definite throughout
  (minimum eigenvalue `+2.7e-17` across 1728 matrices), so reflect and clip
  agree on the matrix and differ only in the root they choose. 30 of the 32
  oracle fields are bitwise unchanged; `sqrtH` and the sampled state path move,
  and 400k draws through either root recover `H` to within Monte Carlo error.
  Anything consuming `sqrtH` as *a* root is unaffected; anything that pinned
  its sign pattern is not.

### Corrections to 0.2.0's notes

- **A warmup step zeroes `W`, and does nothing else to the filter.** 0.2.0's
  `warmup_flag` note said the scan "zeroes the former and holds the latter" of
  the discount matrix and the observational variance. Only the first clause is
  right. `s`/`nu` update normally on a warmup step — they are gated by
  `ignore_obs` (a NaN observation) and by nothing else. The quantity actually
  held is the monitor's signed-error EWMA `dlm_state["S"]`
  (`_multi_fwd_filter_step`), and only on an adaptive (`tau`) universe. The
  grid engine documented this correctly throughout (`_filter_step`: "β not
  forced"); the multi engine's docstrings did not. **No behaviour changed — the
  code always did this; the description of it was wrong.**

## 0.2.0

### Breaking

- **`forecast(h)` now means the same thing on every class**: project the state
  currently held, `h` steps ahead. That was already true of `uv_dlm`,
  `multi_model_dlm`, the blocks and `AutoFFSUniverse`; `AutoFFS` is now
  consistent with them.
- **`AutoFFS.forecast(df, h)` — the one-shot fit-and-predict — is removed.**
  The name could not carry two meanings. Use `fit(df).forecast(h)`:

  ```python
  # 0.1.0
  AutoFFS(season_length=12).forecast(df, h=24, level=[80, 95])
  # 0.2.0
  AutoFFS(season_length=12).fit(df).forecast(h=24, level=[80, 95])
  ```

  `AutoFFS.forecast(df)` raises a `TypeError` naming the replacement. Note
  `forecast(df, h=...)` instead raises Python's "multiple values for argument
  'h'", because `df` binds to the `h` parameter before any check can run.
  `StaticFFS.forecast(df, h)` is unchanged — the legacy class keeps the
  statsforecast-style one-shot.

### Deprecated

- **`AutoFFS.predict(h)`** is an alias of `forecast(h)` and warns. Removed in
  0.3.0.

### Added

- **`fwd_filter(yt)` on `AutoFFS` and `AutoFFSUniverse`** — advance exactly one
  observation and return the one-step-ahead predictive, pairing with `update`
  the way `fwd_filter`/`scan_filter` pair on the engines. `update` computes that
  predictive on every step (it drives the DMA weights) and discards it.
  `return_trace=True` also yields the per-worker `(F, Q)` — the quantity CV
  accumulates as `f1_full`/`q1_full`.

  **No `fit()` call is required.** `uv_dlm.fwd_filter` steps from construction
  because it is handed `m0`/`C0`; `AutoFFS` derives its prior from data, so it
  buffers the opening observations and elicits the prior once there are enough —
  the same computation `fit` runs over those rows, and bitwise identical to
  `fit(window)` then stepping. Those steps return **NaN**: no prior exists yet,
  so there is no predictive, and NaN says so rather than a plausible number. With
  several blocks it buffers to the longest `warmup`, since prior elicitation,
  per-block learning suppression and the union DMA's warmup gate are each keyed
  on a block's own count.
- **`AutoFFSUniverse.fwd_filter(..., persist=True)` and `flush()`.** `update`
  persists on every call, and an M5-sized batch runs to hundreds of MB, so a
  per-step save costs seconds against a filter step of milliseconds.
  `persist=False` defers the write to `flush()`; it is also how you dry-run a
  universe, since a carry that is never flushed leaves no trace.
- **`StaticBlock` streams**: `scan_filter` / `fwd_filter` / `forecast`, so
  `AutoFFS(blocks=[StaticBlock(...)])` fits, steps and forecasts exactly as a
  grid block does. It previously raised on all three — it was built as the CV
  seam, and streaming a static universe existed only behind
  `AutoFFSUniverse(grid_period=None)`, which made the block constructible but
  unusable (the `Block` protocol checks names, not behaviour) and forced anyone
  wanting a streamed static universe onto the disk-backed orchestrator for a
  *block* capability rather than for persistence.

  It delegates to the packed `multi_model_dlm` it already wraps, built through
  the same `_build_multi_and_dma` the CV path uses, so the two are one universe
  rather than two definitions free to drift. `h_template` (default 18) caps the
  streaming horizon: the packed universe precomputes its h-step template when
  built, unlike the grid which builds `GH` per call. Persistence stays
  unimplemented — a *persistent* static universe is still
  `AutoFFSUniverse(grid_period=None)`.
- **`multi_model_dlm.fwd_filter(..., warmup_flag=0.0)`.** The scan path has
  always applied a per-step warmup flag; the step face could not, so a caller
  driving the filter one observation at a time filtered the warmup window
  untreated and could not reproduce the scan over it.
- **`DLM.design(init_data, warmup_steps=None)` returns the model DEFINITION**
  without constructing a filter: `F`, `G`, `regression_G`, the per-state
  discounts, `mult_comps`, `monitor_inject`, and the elicited `m0`/`C0`/`V0`/`nu0`,
  as a frozen `DlmDesign`. `compile()` is now exactly `design()` plus
  `uv_dlm(...)`, so the two cannot drift.

  For callers that run their own filter. `disc_norm` and `disc_damped` come back
  SEPARATELY rather than pre-multiplied, because a damped trend puts
  `disc * phi**2` on the growth slot and a caller may need to override that —
  composing them is the caller's business, and doing it by mutating a built
  `uv_dlm` is worse. `DlmDesign.applied_disc` gives the product when that is what
  is wanted.

  The motivating case is a Quintana/West multivariate DLM built on DLMAX's
  components: it needs the structure and the discounts, supplies its own priors
  (a `C0` shared across series and scale-free, a `V0` that is a prior scale
  rather than an observation variance), and must not inherit filter runtime
  (`device`, `adapt`, `n_pad`, `monitor`, variance discounting) it would only
  have to override.

- **`ffbs(smoothed=...)` accepts a SHARED covariance path.** If the smoothed
  path's series axis is length 1 while `traj["m"]` carries `q` series, the
  gains broadcast across all of them. This is the Quintana/West case the
  `right_factor` argument already anticipates: `C*` is common to every series,
  so a per-series copy is pure redundancy — at `T=95, q=304, p=14` it is 90 MB
  of duplicated `B`/`sqrtH`, and 304 identical matmuls where one would do.
  Pass `right_factor` explicitly when using this; the default
  `L = diag(sqrt(s_T))` is built from `smoothed["scale"]` and cannot be formed
  from a one-series path.

### Packaging

- **`numpyro` and `matplotlib` are no longer dependencies.** Neither was
  imported anywhere in the package — `numpyro` appeared nowhere in the
  repository except the dependency list itself. Between them they took a clean
  install from 10 packages to 21, pulling in `contourpy`, `pillow`, `fonttools`,
  `kiwisolver`, `cycler`, `pyparsing` (matplotlib's rendering stack) and
  `multipledispatch`, `tqdm` (numpyro's). 0.1.0 shipped the same list.

  `matplotlib` moves to the `dev` extra, which is what the `docs/tutorials`
  notebooks need it for. If you relied on DLMAX to install either package
  transitively, declare it yourself.

  Cannot change a forecast: nothing in any code path imported them.

- **The `dev` extra is removed; development tooling is a `[dependency-groups]`
  entry.** `dev` was defined twice with different contents — as a published
  extra (`pytest>=7.0`, xarray, ruff, jupyter) and as a PEP 735 group
  (`pytest>=9.0.3`). `uv sync` installs the group, `uv sync --extra dev` the
  extra, so the command the README documented and the default one built
  different environments; `ruff` sat in the extra and was therefore never
  actually installed. The group is now the single definition, which also keeps
  dev tooling out of published metadata — `pip install awen-dlmax[dev]` should
  not exist, because `ruff` is not a feature of DLMAX.

  The `gpu` and `xarray` extras are unaffected. If you installed `[dev]`, use
  `uv sync` (or `--group dev`) from a clone instead.

### Fixed

- **Negative h-step predictive variance** reaching the Vincent SD combine.
  `dlm_uv_fcast_H` returns `q = s + F'RH F`; the state term is a quadratic form
  in a covariance, so `q >= s` holds analytically, but rounding could drive it
  negative when `RH` is large and ill-conditioned. It then hit `np.sqrt`, went
  NaN, and was silently dropped by the `isfinite` guard downstream — costing
  that worker its DMA weight for the cell with nothing in the output to say so.
  The state term is now clamped at zero, restoring the analytic bound. inf and
  NaN still propagate, so a genuinely diverged component is not rescued.

### Unchanged, deliberately

- **`rts_smooth` and `ffbs` remain PROVISIONAL.** 0.2.0 adds a capability to
  `ffbs` (the shared covariance path), which could be read as the signature
  firming up. It is not: both stay explicitly carved out of the stability
  guarantee the rest of `__all__` carries, and their signatures may still change.
  Prefer the `uv_dlm` methods where they suffice.

### Numerical behaviour

- **The `forecast`/`predict` work changes nothing.** `AutoFFS.forecast(h)`
  returns exactly what `0.1.0`'s `predict(h)` returned, and `fwd_filter`
  surfaces a quantity that was already computed.
- **`warmup_flag` defaults to `0.0`, so no existing caller moves** — but it does
  change results for anyone who was stepping `multi_model_dlm.fwd_filter`
  manually across a warmup window, which was previously filtered with the
  discount matrix live and the observational variance updating, where the scan
  zeroes the former and holds the latter. That path now matches the scan.
- **`StaticBlock` streaming and its CV path differ only in prior elicitation.**
  `_build_multi_and_dma` elicits the diffuse prior from the whole array it is
  given, which under CV is the entire batch; a stream cannot do that without
  look-ahead, so it elicits from `ys[:warmup]` as `GridBlock` does — which is
  also what makes step-equals-scan hold. Given the SAME window the two agree to
  **4.4e-16** on the per-model h-step location. Per step the faces are bitwise
  identical (`m`, `s`, `nu`, `Z`). The CV path itself is untouched.
- **`DLM.design()` is a pure extraction; `compile()` is unchanged.** Verified
  bit-identical across six configurations (level-only, trend+Fourier,
  trend+Fourier+AR, each with and without `warmup_steps`, so both the
  `elicit_prior` and `diffuse_prior` paths): every one of `F`, `G`, `m0`, `C0`,
  `V0`, `nu`, `disc_rates`, `disc_rates_damped`, `mult_comps`, `regression_G`
  and `monitor_inject` matches the pre-change output exactly, and `design()`
  agrees with what `compile()` consumed. Full suite: 390 passed.

- **The `ffbs` shared-path change moves nothing for existing callers.** Two
  `einsum("qij,qj->qi", ...)` became broadcasting `matmul`s, which is the same
  operation when the leading axes already match. Verified bit-identical
  (`max|diff| 0.000e+00`) on both the default-`L` and `chol(Sigma)` paths at a
  fixed key, and a broadcast shared path is bit-identical to explicitly tiling
  that path to `q` copies.
- **The variance clamp changes results only where they were already broken.**
  It is a no-op wherever `F'RH F >= 0`, which is everywhere the filter is
  well-conditioned: the M4 and M5 paths never trigger it (zero occurrences
  across two 30-origin M5 rolls), and the full suite — including the
  bit-exactness contracts — is unchanged. It DOES move results for
  configurations run near marginal stability: an hourly panel with an annual
  Fourier and discounts floored at 0.99 saw the warning 3–5 times per run, and
  those cells now keep a worker that was previously dropped.
- Step-vs-scan equivalence, for reference: the multi-block path is **bitwise**
  (both faces share `_multiblock_step`), while the single-block path differs by
  about one ulp (2.2e-16 measured) because `update` runs the block's
  `scan_filter` there and `lax.scan` reduces in a different order from a single
  step call. Tests assert `rtol=1e-12`.

## 0.1.0

First public release. Wing-grid `AutoFFS` / `AutoFFSUniverse`, the block API,
and the `uv_dlm` / `multi_model_dlm` engines.

### Numerical behaviour

- Baseline. Note that results for models run near marginal stability (a high
  discount floor over a long filter, as in an hourly panel with an annual
  Fourier) are sensitive to thread count as well as to library version: such a
  configuration reproduces bitwise only on matching hardware and threading.

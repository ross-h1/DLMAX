"""Regression and AR tails through ``AutoFFS.cross_validation``.

A tail attached to a block -- ``Regressors`` for exogenous covariates, ``AR``
for the series' own lags -- must reach the rolling path, not merely occupy
state. The failure mode this guards is silent: a tail that is filtered against
a zero ``F`` row grows the state, allocates and RTRL-learns a discount block,
and returns forecasts indistinguishable from the no-tail model.

These tests drive the ORCHESTRATOR rather than the kernel, which is the seam
where that can happen: the tail is configured on the block but fed by the
rolling driver above it.

The central assertion is EQUIVALENCE with the streaming path. A block's
rolling and streaming faces are the same model scanned to the same point, so
they must agree, tail or no tail -- which also pins the block's full
configuration (``disc_prior``, ``learn_dma``, ``clip``, ``disc_init``), since
dropping any of them makes a block disagree with itself.
"""
import numpy as np
import pandas as pd
import pytest

from DLMAX import ffs_core
from DLMAX.ffs.dlm_builder import AR, DLM, LocalTrend, Regressors, Wing
from DLMAX.ffs.grid_block import AdaptiveBlock

N, T, H, WARMUP = 5, 40, 6, 4

# Tolerance for "these two code paths are the same computation".
# Cross-platform float noise on these assertions runs at 1-10x machine epsilon
# (2.2e-16 to 2.3e-15); a genuine disagreement between the paths is 1e-2 or
# larger. 1e-12 sits far above the former and far below the latter.
EQUIV_RTOL = 1e-12


def _panel(seed=0, exog_coef=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (T, N))
    y = np.cumsum(rng.normal(0, 1, (T, N)), axis=0) + 100 + exog_coef * x
    cols = [f"s{i}" for i in range(N)]
    return pd.DataFrame(y, columns=cols), pd.DataFrame(x, columns=cols)


def _long(df, value):
    return (df.reset_index().melt(id_vars="index", var_name="unique_id",
                                  value_name=value)
            .rename(columns={"index": "ds"}))


def _block(Y, tail, learn_dma=True):
    """Two damping families, optionally with a tail. `from_cells` deliberately:
    it is the path that carries per-cell discounts and a clip box, and the path
    the config route (`GridBlock.build`) cannot express."""
    cells = []
    for damping in (1.0, 0.95):
        c = DLM(family="Gaussian", n_series=N)
        c.add_component(LocalTrend(name="trend", damping=damping,
                                   disc_rate=Wing(0.95, offset=1.0)))
        if tail == "exog":
            c.add_component(Regressors(name="x", n_regs=1, disc_rate=0.99))
        elif tail == "ar":
            # zero-centred, NOT the default Minnesota shrink-to-random-walk:
            # the level already carries a random walk.
            c.add_component(AR(name="ar", order=1, disc_rate=0.99,
                               m0=np.zeros((N, 1)), C0=np.full((N, 1, 1), 0.25)))
        c.set_error(disc_rate=0.99, power=1.0)
        cells.append(c.compile(init_data=Y, warmup_steps=WARMUP, h=H))
    return AdaptiveBlock.from_cells(
        cells, warmup=WARMUP, offset=1.0, disc_prior=(3.0, 1.0, 5.0, 1.0),
        seasonal_prior=3.0, learn_dma=learn_dma, adapt_guard=0.5)


def _cv(Y, tail, learn_dma=True, **kw):
    m = ffs_core.AutoFFS(blocks=[_block(Y, tail, learn_dma)], season_length=None,
                         dma_pdr=0.9, dma_mdr=0.9, max_batch_size=50,
                         adaptive=False, learn_dma=learn_dma)
    return m.cross_validation(_long(Y, "y"), h=H, n_windows=1,
                              warmup_steps=WARMUP, **kw)["AutoFFS"].to_numpy()


# --------------------------------------------------------------------------
# a tail must CHANGE the answer
# --------------------------------------------------------------------------
def test_ar_tail_changes_cross_validation():
    """An AR tail must move the CV forecast.

    A difference at float-noise level (~1e-14) would mean the tail is present in
    the state but absent from the result."""
    Y, _ = _panel()
    base, ar = _cv(Y, "none"), _cv(Y, "ar")
    assert np.nanmax(np.abs(ar - base)) > 1e-3


def test_exog_tail_changes_cross_validation():
    Y, X = _panel(exog_coef=8.0)
    base = _cv(Y, "none")
    got = _cv(Y, "exog", exog=_long(X, "x1"))
    assert np.nanmax(np.abs(got - base)) > 1e-3


# --------------------------------------------------------------------------
# equivalence with the streaming path
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tail", ["none", "ar"])
@pytest.mark.parametrize("learn_dma", [False, True])
def test_rolling_matches_streaming(tail, learn_dma):
    """The block's rolling face and its streaming face are the same model
    scanned to the same point, so they must agree -- with a tail as without.

    ``rtol=EQUIV_RTOL`` rather than exact: the two faces are the same
    computation, which float arithmetic realises to within rounding rather than
    bitwise, since XLA lowers the reductions differently per platform (~2e-15
    apart under ``learn_dma=True``). An exact assertion would pin one platform's
    lowering rather than the property being claimed.

    The tolerance still separates that from anything real: the failure modes
    this guards against move the answer by 1e-2 or more, ten orders of
    magnitude above 1e-12.
    """
    Y, _ = _panel()
    arr = Y.to_numpy()
    cut = T - H - 1
    rolling = np.asarray(
        _block(Y, tail, learn_dma).forecast_rolling(arr, [cut], H)[0])[:, 0, :]
    b = _block(Y, tail, learn_dma)
    b.scan_filter(arr[: T - H])
    streaming = np.asarray(b.forecast(H)[0])
    if streaming.shape != rolling.shape:
        streaming = streaming.T
    np.testing.assert_allclose(rolling, streaming, rtol=EQUIV_RTOL, atol=0)


# --------------------------------------------------------------------------
# the guard: every mismatch fails loudly rather than silently no-opping
# --------------------------------------------------------------------------
def test_tail_without_design_raises():
    Y, _ = _panel()
    with pytest.raises(ValueError, match="EXOGENOUS regression tail"):
        _cv(Y, "exog")


def test_design_without_tail_raises():
    Y, X = _panel()
    with pytest.raises(ValueError, match="no regression tail"):
        _cv(Y, "none", exog=_long(X, "x1"))


def test_exog_with_ar_tail_raises():
    """An AR tail builds its own design; a supplied one would be ignored."""
    Y, X = _panel()
    with pytest.raises(ValueError, match="builds its own design"):
        _cv(Y, "ar", exog=_long(X, "x1"))


def test_static_block_refuses_regressors():
    from DLMAX.ffs.static_block import StaticBlock
    sb = StaticBlock(season_length=None, n_seas_comps=None, warmup=WARMUP)
    with pytest.raises(ValueError, match="no regression tail"):
        sb.cv_trajectory(["a"], np.zeros((10, 1)), [5], H, regressors=np.zeros((10, 1, 1)))


def test_incomplete_exog_raises():
    """A design that does not cover every ds fails silently if tolerated: the
    uncovered steps filter against a zero ``F`` row."""
    Y, X = _panel(exog_coef=8.0)
    short = _long(X, "x1")
    short = short[short["ds"] < T - 5]          # drop the tail of the calendar
    with pytest.raises(ValueError, match="does not cover every ds"):
        _cv(Y, "exog", exog=short)


# --------------------------------------------------------------------------
# legacy multi_model_dlm engine: exogenous CV
# --------------------------------------------------------------------------
def _legacy_universe(init_data, h, ctx):
    d = DLM(family="Gaussian", n_series=init_data.shape[1])
    d.add_component(LocalTrend(name="trend", disc_rate=[0.95, 0.99], damping=1.0))
    d.add_component(Regressors(name="x", n_regs=1, disc_rate=0.99))
    d.set_error(disc_rate=0.99, power=[1.0])
    models, desc = d.compile_universe(init_data, h=h, warmup_steps=ctx.warmup_steps)
    desc = desc.copy()
    desc["Class"] = "all"
    return models, desc.set_index("key")


def _legacy_cv(Y, X=None, n_windows=1):
    m = ffs_core.StaticFFS(season_length=None, universe_builder=_legacy_universe,
                               max_batch_size=50)
    kw = {} if X is None else {"exog": _long(X, "x1")}
    return m.cross_validation(_long(Y, "y"), h=H, n_windows=n_windows,
                              warmup_steps=WARMUP, **kw)["AutoFFS"].to_numpy()


@pytest.mark.parametrize("n_windows", [1, 3])
def test_legacy_exog_cv_runs(n_windows):
    """Both emit paths: the n_windows==1 fast path and the multi-cutoff scan."""
    Y, X = _panel(exog_coef=9.0)
    got = _legacy_cv(Y, X, n_windows=n_windows)
    assert np.isfinite(got).all()


def test_legacy_exog_future_rows_reach_the_forecast():
    """Perturbing ONLY the horizon rows must move the forecast.

    The sharp test for this seam: a design that reaches the filter but not
    forecast_origin still changes the answer (via the filtered state), so
    "the forecast moved" is not evidence on its own. Only the FUTURE rows
    isolate the emission.
    """
    Y, X = _panel(exog_coef=9.0)
    cut = T - H - 1
    X2 = X.copy()
    X2.iloc[cut + 1:] += 50.0
    base, moved = _legacy_cv(Y, X), _legacy_cv(Y, X2)
    assert np.nanmax(np.abs(base - moved)) > 1.0


def test_legacy_exog_tail_without_design_raises():
    Y, _ = _panel()
    with pytest.raises(ValueError, match="EXOGENOUS regression tail"):
        _legacy_cv(Y)


# --------------------------------------------------------------------------
# cross-engine sanity
# --------------------------------------------------------------------------
def test_wing_ar_moves_like_legacy_ar():
    """The wing's AR tail and the legacy engine's ``include_ar`` should push
    the forecast the same way.

    Deliberately NOT an equality test: the two engines differ in model set and
    in how discounts are handled (RTRL-learned wing vs fixed replay), so the
    numbers will not match. What it catches is a wing AR wired backwards or at
    the wrong scale -- using the path that is already known to work, since the
    earlier include_ar A/B (yearly helps on M1Y/M4Y, hurts on M4Q) ran on it.
    """
    Y, _ = _panel()
    d_wing = _cv(Y, "ar") - _cv(Y, "none")

    def legacy(include_ar):
        m = ffs_core.StaticFFS(season_length=None, include_ar=include_ar,
                                   max_batch_size=50)
        return m.cross_validation(_long(Y, "y"), h=H, n_windows=1,
                                  warmup_steps=WARMUP)["AutoFFS"].to_numpy()

    d_legacy = legacy(True) - legacy(False)

    ok = np.isfinite(d_wing) & np.isfinite(d_legacy)
    assert ok.sum() > 10
    a, b = d_wing[ok], d_legacy[ok]
    # same direction
    assert np.corrcoef(a, b)[0, 1] > 0.3
    # same order of magnitude (within 10x either way)
    ratio = np.sqrt((a ** 2).mean()) / np.sqrt((b ** 2).mean())
    assert 0.1 < ratio < 10.0, f"AR effect scale differs {ratio:.2f}x"

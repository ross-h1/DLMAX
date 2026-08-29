"""Tests for the dynamic-grid discount RTRL core (``ffs/discount_grid.py``).

Headline: the **sensitivity-carry gate** — with the discount FIXED, the sum of the
per-step forward-RTRL gradients must equal the batch gradient of the total loss
w.r.t. the discount, for BOTH objectives (log score and frozen-variance squared
error). That validates the ``S = ∂state/∂θ`` carry reproduces reverse-mode. Plus a
finite-difference cross-check. Self-contained (models built from DLMAX primitives),
mirroring tests/test_qr_kernel.py.
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import block_diag

import DLMAX.dlm_core as dc
from DLMAX.ffs import discount_grid as dg


def _fourier_full(period):
    Fb, Gb = [], []
    for j in range(1, period // 2 + 1):
        w = 2.0 * math.pi * j / period
        if period % 2 == 0 and j == period // 2:
            Fb.append(jnp.array([1.0])); Gb.append(jnp.array([[math.cos(w)]]))
        else:
            Fb.append(jnp.array([1.0, 0.0]))
            Gb.append(jnp.array([[math.cos(w), math.sin(w)],
                                 [-math.sin(w), math.cos(w)]]))
    return jnp.concatenate(Fb), block_diag(*Gb)


def _make_model(damping, seasonal, variant, period=6):
    if damping == 0.0:
        Ft, Gt, nt = jnp.array([1.0]), jnp.array([[1.0]]), 1
    else:
        Ft = jnp.array([1.0, 0.0])
        Gt = jnp.array([[1.0, damping], [0.0, damping]])
        nt = 2
    mult = variant == "M"
    var_power = 0.0 if mult else 1.0
    if seasonal:
        Fs, Gs = _fourier_full(period)
        F = jnp.concatenate([Ft, Fs]); G = block_diag(Gt, Gs)
        ns = period - 1
        state_to_block = jnp.asarray([0] * nt + [1] * ns)
        n_blocks = 2
    else:
        F, G, ns = Ft, Gt, 0
        state_to_block = jnp.zeros(nt, dtype=int)
        n_blocks = 1
    mc = jnp.zeros(F.shape[0])
    if mult and ns > 0:
        mc = mc.at[nt:nt + ns].set(1.0)
    return dg.GridModel(F=F, G=G, mc=mc, var_power=var_power,
                        state_to_block=state_to_block, n_blocks=n_blocks)


def _init_state(model, ys, warm=12, nu0=1.0):
    d = model.F.shape[0]
    yw = np.asarray(ys)[:warm]
    m0 = jnp.zeros(d).at[0].set(float(np.mean(yw)))
    C0 = jnp.eye(d) * float(max(np.var(yw), 1e-6)) * 10.0
    s0 = float(max(np.var(np.diff(yw)), 1e-6))
    ev, UC = jnp.linalg.eigh(C0)
    SC = jnp.sqrt(jnp.clip(ev, 1e-30, None))
    return {"m": m0, "Z": SC[:, None] * UC.T,
            "s": jnp.asarray(s0), "nu": jnp.asarray(nu0)}


def _synthetic_series(T=90, period=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = 100.0 + 0.4 * t + 6.0 * np.sin(2 * np.pi * t / period) + rng.normal(0, 2, T)
    return jnp.asarray(np.maximum(y, 1.0))


_MODELS = [
    ("L-A", _make_model(0.0, False, "A")),
    ("LT-A", _make_model(0.95, False, "A")),
    ("LTS-A", _make_model(0.95, True, "A")),
    ("L-M", _make_model(0.0, False, "M")),
    ("LTS-M", _make_model(0.95, True, "M")),
]

WARMUP = 12


def _theta(model, disc=0.95, beta=0.99):
    return jnp.full(model.n_params, disc).at[-1].set(beta)


@pytest.mark.parametrize("name,model", _MODELS)
def test_sensitivity_gate(name, model):
    """Σ_t g_t (forward RTRL, fixed θ) == batch gradient (reverse-mode), both losses."""
    ys = _synthetic_series()
    init = _init_state(model, ys)
    theta = _theta(model)
    theta_seq = jnp.tile(theta, (ys.shape[0], 1))

    out = dg.grad_run(model, init, ys, theta_seq, warmup=WARMUP)
    g_ls_sum = np.asarray(out["g_ls"].sum(axis=0))
    g_sq_sum = np.asarray(out["g_sq"].sum(axis=0))

    ref_ls = np.asarray(jax.grad(lambda th: dg.batch_loss(model, init, ys, th, WARMUP)[0])(theta))
    ref_sq = np.asarray(jax.grad(lambda th: dg.batch_loss(model, init, ys, th, WARMUP)[1])(theta))

    assert np.allclose(g_ls_sum, ref_ls, rtol=1e-6, atol=1e-8), (name, g_ls_sum, ref_ls)
    assert np.allclose(g_sq_sum, ref_sq, rtol=1e-6, atol=1e-8), (name, g_sq_sum, ref_sq)


def test_finite_difference_logscore():
    """FD check of the log-score batch gradient (validates the reverse-mode reference
    the gate compares against). Only the log-score objective is FD-checkable: the
    squared-error loss carries ``stop_gradient(Q)``, so a finite perturbation moves the
    real Q in the denominator (a term the analytic gradient deliberately drops), and FD
    would test the wrong quantity — the gate covers the squared-error gradient instead."""
    name, model = "LTS-M", _make_model(0.95, True, "M")
    ys = _synthetic_series()
    init = _init_state(model, ys)
    theta = _theta(model)
    eps = 1e-6
    f = lambda th: float(dg.batch_loss(model, init, ys, th, WARMUP)[0])
    ana = np.asarray(jax.grad(lambda th: dg.batch_loss(model, init, ys, th, WARMUP)[0])(theta))
    fd = np.zeros_like(ana)
    for i in range(model.n_params):
        fd[i] = (f(theta.at[i].add(eps)) - f(theta.at[i].add(-eps))) / (2 * eps)
    assert np.allclose(ana, fd, rtol=1e-4, atol=1e-5), (ana, fd)


def test_beta_invisible_to_squared_error():
    """The quasi-likelihood decoupling, verified: β does NOT leak into the frozen-var
    squared-error objective (∂(Σ ell_sq)/∂β ≈ 0), so the mean-discount signal cannot
    move β — only the log score can. (Mean discounts DO have a nonzero sq-gradient.)"""
    for name, model in _MODELS:
        ys = _synthetic_series()
        init = _init_state(model, ys)
        theta = _theta(model)
        g_sq = np.asarray(jax.grad(lambda th: dg.batch_loss(model, init, ys, th, WARMUP)[1])(theta))
        assert abs(g_sq[-1]) < 1e-8, (name, "beta sq-grad", g_sq[-1])
        assert np.any(np.abs(g_sq[:-1]) > 1e-6), (name, "mean sq-grad all ~0", g_sq)


def test_losses_finite():
    for name, model in _MODELS:
        ys = _synthetic_series()
        init = _init_state(model, ys)
        out = dg.grad_run(model, init, ys, jnp.tile(_theta(model), (ys.shape[0], 1)),
                          warmup=WARMUP)
        for k in ("ell_ls", "ell_sq", "g_ls", "g_sq"):
            assert np.all(np.isfinite(np.asarray(out[k]))), (name, k)


# --------------------------------------------------------------------------- #
# The cell (3 wingmen, pooled level centre, DMA-weighted).
# --------------------------------------------------------------------------- #
def test_cell_runs_and_adapts():
    """The cell runs, produces finite forecasts, the level centre MOVES from its
    prior, the wingmen hold centre ± offset, and the pooling weights stay valid."""
    for name, model in _MODELS:
        ys = _synthetic_series(T=120)
        init = _init_state(model, ys)
        out = dg.run_cell(model, init, ys, offset=1.0, warmup=WARMUP)
        for k in ("c_traj", "theta_traj", "w_traj", "f", "q"):
            assert np.all(np.isfinite(np.asarray(out[k]))), (name, k)
        # centre adapted away from the logit(0.95) prior
        assert abs(float(out["c"]) - float(dg._logit(0.95))) > 1e-3, (name, "centre static")
        # wingmen level logits == centre + {-off,0,+off} (last step), ordered
        wth_lvl = np.asarray(out["wth"])[:, 0]
        c = float(out["c"])
        assert np.allclose(wth_lvl, np.clip([c - 1.0, c, c + 1.0], dg._logit(0.5),
                                            dg._logit(0.9999)), atol=1e-6), (name, wth_lvl)
        # weights: a simplex at every step
        w = np.asarray(out["w_traj"])
        assert np.allclose(w.sum(1), 1.0, atol=1e-6) and np.all(w >= 0), name


def test_cell_inclass_weights_from_allocator():
    """The cell's in-class pooling weights come from DLMAX's single-class allocator:
    a valid simplex over the 3 wingmen at every step, discriminating by the end."""
    name, model = "LTS-M", _make_model(0.95, True, "M")
    ys = _synthetic_series(T=120)
    init = _init_state(model, ys)
    out = dg.run_cell(model, init, ys, warmup=WARMUP)
    w = np.asarray(out["w_traj"])
    assert np.allclose(w.sum(1), 1.0, atol=1e-6) and np.all(w >= 0)
    assert w[-1].max() > 1.0 / 3                     # not stuck uniform


def test_cell_beta_per_worker_diverges():
    """β is per-worker (not pooled), so the three wingmen's β can differ by the end."""
    name, model = "LTS-M", _make_model(0.95, True, "M")
    ys = _synthetic_series(T=150)
    init = _init_state(model, ys)
    out = dg.run_cell(model, init, ys, warmup=WARMUP, lr_worker=0.1)
    beta = np.asarray(out["wth"])[:, -1]   # per-wingman β logit, final
    assert np.all(np.isfinite(beta))
    # at least moved from the logit(0.99) prior (per-worker learning is live)
    assert np.any(np.abs(beta - float(dg._logit(0.99))) > 1e-3), beta


# --------------------------------------------------------------------------- #
# The grid (16 structures × 3 wingmen, global DMA, mixture combine).
# --------------------------------------------------------------------------- #
def test_build_grid_taxonomy():
    assert len(dg.build_grid(period=4)) == 16     # 4 damping × 2 seasonal × 2 error
    assert len(dg.build_grid(period=None)) == 8   # seasonal structures dropped
    names = [n for n, _m, _c in dg.build_grid(period=4)]
    assert len(set(names)) == 16                  # all distinct class tags


def test_run_grid_combines():
    grid = dg.build_grid(period=6)
    ys = _synthetic_series(T=120, period=6)
    inits = [dg.grid_init(comps, ys, WARMUP, m.var_power) for _n, m, comps in grid]
    out = dg.run_grid(grid, inits, ys, offset=1.0, warmup=WARMUP)
    W = 3 * len(grid)                             # 48 workers
    assert out["weights"].shape[1] == W
    assert np.all(np.isfinite(np.asarray(out["loc"])))
    assert np.all(np.asarray(out["var"]) > 0)
    w = np.asarray(out["weights"])
    assert np.allclose(w.sum(1), 1.0, atol=1e-6) and np.all(w >= 0)
    # the global DMA actually discriminates (not stuck uniform by the end)
    assert w[-1].max() > 1.5 / W


def test_forecast_origin():
    """h-step origin RAW predictive per worker: right shapes, finite loc, positive q,
    scalar valid nu (the combine, not this, applies the variance factor)."""
    for name, model in _MODELS:
        ys = _synthetic_series()
        init = _init_state(model, ys)
        theta = _theta(model)
        out = dg.grad_run(model, init, ys[:40], jnp.tile(theta, (40, 1)), warmup=WARMUP)
        loc, q, nu = dg.forecast_origin(model, out["state"], theta, h=6)
        assert loc.shape == (6,) and q.shape == (6,) and np.ndim(nu) == 0, name
        assert np.all(np.isfinite(np.asarray(loc))), name
        assert np.all(np.asarray(q) > 0) and float(nu) > 0, name


def test_run_grid_rolling():
    """Rolling backtest: per-cutoff combined h-step forecast is finite + right shape;
    per-worker detail present; combine weights a valid simplex at each origin."""
    grid = dg.build_grid(period=6)
    ys = _synthetic_series(T=120, period=6)
    inits = [dg.grid_init(comps, ys, WARMUP, m.var_power) for _n, m, comps in grid]
    h, cutoffs = 6, [90, 100, 110]
    out = dg.run_grid_rolling(grid, inits, ys, h, cutoffs, warmup=WARMUP)
    n_cut, W = len(cutoffs), 3 * len(grid)
    assert out["loc"].shape == (n_cut, h) and out["sd"].shape == (n_cut, h)
    assert np.all(np.isfinite(np.asarray(out["loc"]))), "loc"
    assert np.all(np.asarray(out["sd"]) > 0), "sd"
    assert out["loc_w"].shape == (n_cut, W, h)
    w = np.asarray(out["weights"])
    assert np.allclose(w.sum(1), 1.0, atol=1e-6) and np.all(w >= 0)


def test_run_grid_batch():
    """Dask worker entry: vmapped batch runs, right output shapes, obs gathered right."""
    grid = dg.build_grid(period=6)
    rng = np.random.default_rng(2)
    L, q, h = 60, 4, 6
    arr = np.stack([100 + 0.4 * np.arange(L) + rng.normal(0, 2, L) for _ in range(q)], axis=1)
    cutoffs = (53, 51, 49)   # L-h-1-i*step, the _origins convention (obs fits in [0,L))
    sids, loc, sd, obs = dg.run_grid_batch(arr, [f"s{j}" for j in range(q)], cutoffs, h, 6, 12)
    assert loc.shape == (q, 3, h) and sd.shape == (q, 3, h) and obs.shape == (q, 3, h)
    assert np.all(np.isfinite(loc)) and np.all(sd > 0)
    assert np.allclose(obs[0, 0], arr[cutoffs[0] + 1:cutoffs[0] + 1 + h, 0])   # obs alignment


def test_run_grid_batch_diag():
    """return_diag adds (names, weight, level_d, beta) without changing loc/sd."""
    grid = dg.build_grid(period=6)
    M = 3 * len(grid)
    rng = np.random.default_rng(3)
    L, q, h = 60, 4, 6
    arr = np.stack([100 + 0.4 * np.arange(L) + rng.normal(0, 2, L) for _ in range(q)], axis=1)
    cutoffs = (53, 51, 49)
    base = dg.run_grid_batch(arr, [f"s{j}" for j in range(q)], cutoffs, h, 6, 12)
    diag = dg.run_grid_batch(arr, [f"s{j}" for j in range(q)], cutoffs, h, 6, 12,
                             return_diag=True)
    sids, loc, sd, obs, names, W, LVL, BET = diag
    assert np.array_equal(base[1], loc) and np.array_equal(base[2], sd)   # combine untouched
    assert len(names) == M and names[0].endswith(":fast")
    assert W.shape == (q, 3, M) and LVL.shape == (q, 3, M) and BET.shape == (q, 3, M)
    assert np.allclose(W.sum(-1), 1.0)                          # DMA weights a simplex
    assert np.all((LVL >= 0.5 - 1e-9) & (LVL <= 0.9999 + 1e-9)) and np.all(BET > 0)

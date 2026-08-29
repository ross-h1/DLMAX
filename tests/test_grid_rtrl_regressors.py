"""The RTRL sensitivity carry, with a regression tail attached.

The regressors brief (FFS ``notes/wing_regressors_brief.md`` §3) rests on one
claim, and it is the claim that made the work cheap:

    regressors are DATA — constants to the jvp, exactly as ``y`` is — so the
    ``S = ∂state/∂θ`` recursion is structurally unchanged and the tail adds no
    new derivative terms.

Tests that check the tail reaches the filter and the forecast do not cover
this: they say nothing about whether the LEARNING signal through it is right.
A wrong sensitivity carry would leave those green while quietly learning the
wrong discount for the tail's own block.

So this is ``test_discount_grid.py``'s sensitivity gate re-run with a tail: with
θ FIXED, the sum of the per-step forward-RTRL gradients must equal the batch
gradient of the total loss, for both objectives. Self-contained models built from
DLMAX primitives, in that file's style.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from DLMAX.ffs import discount_grid as dg

T, NREG, WARMUP = 60, 2, 12
# coefficients for the generating process, sized to whatever tail is asked for
BETA = np.array([1.5, -0.7, 0.9, -1.2])


def _tail_model(*, n_regs=NREG, trend=False):
    """Local level (or damped trend) + an ``n_regs``-wide regression tail.

    The tail is its own discount block — block 1 — which is the whole point: its
    coefficients' drift rate is learned on the same footing as the level's. G is
    the identity there (random-walk coefficients) and its F entries are zero,
    filled per step from the regressors by ``_F_at``.
    """
    if trend:
        Ft, Gt, nt = jnp.array([1.0, 0.0]), jnp.array([[1.0, 0.95], [0.0, 0.95]]), 2
    else:
        Ft, Gt, nt = jnp.array([1.0]), jnp.array([[1.0]]), 1
    F = jnp.concatenate([Ft, jnp.zeros(n_regs)])
    G = jnp.eye(nt + n_regs).at[:nt, :nt].set(Gt)
    return dg.GridModel(
        F=F, G=G, mc=jnp.zeros(nt + n_regs), var_power=1.0,
        state_to_block=jnp.asarray([0] * nt + [1] * n_regs),
        n_blocks=2, n_regs=n_regs, reg_offset=nt,
        regression_mask=jnp.asarray([False, True]))


def _structural_model():
    return dg.GridModel(F=jnp.array([1.0]), G=jnp.array([[1.0]]),
                        mc=jnp.zeros(1), var_power=1.0,
                        state_to_block=jnp.zeros(1, dtype=int), n_blocks=1)


def _data(seed=1, n_regs=NREG):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (T, n_regs))
    lvl = np.cumsum(rng.normal(0, 0.2, T)) + 20.0
    assert n_regs <= BETA.size, "widen BETA for a tail this wide"
    y = lvl + X @ BETA[:n_regs] + rng.normal(0, 0.4, T)
    return jnp.asarray(y), jnp.asarray(X)


def _init_state(model, ys, warm=WARMUP):
    d = model.F.shape[0]
    yw = np.asarray(ys)[:warm]
    m0 = jnp.zeros(d).at[0].set(float(np.mean(yw)))
    C0 = jnp.eye(d) * float(max(np.var(yw), 1e-6)) * 10.0
    ev, UC = jnp.linalg.eigh(C0)
    return {"m": m0, "Z": jnp.sqrt(jnp.clip(ev, 1e-30, None))[:, None] * UC.T,
            "s": jnp.asarray(float(max(np.var(np.diff(yw)), 1e-6))),
            "nu": jnp.asarray(1.0)}


def _theta(model, disc=0.95, beta=0.99):
    return jnp.full(model.n_params, disc).at[-1].set(beta)


_TAILED = [("level+tail", _tail_model()),
           ("trend+tail", _tail_model(trend=True)),
           ("level+wide-tail", _tail_model(n_regs=4))]


@pytest.mark.parametrize("name,model", _TAILED)
def test_rtrl_matches_batch_gradient_with_a_tail(name, model):
    """Σ_t g_t (forward RTRL, fixed θ) == batch gradient (reverse-mode).

    The claim stated as arithmetic. It covers the tail's OWN
    discount block, not just the level's — a sensitivity carry that dropped the
    tail's contribution would show up in ``g[1]`` alone.
    """
    ys, X = _data(n_regs=model.n_regs)
    init = _init_state(model, ys)
    theta = _theta(model)
    theta_seq = jnp.tile(theta, (T, 1))

    out = dg.grad_run(model, init, ys, theta_seq, warmup=WARMUP, xs=X)
    g_ls_sum = np.asarray(out["g_ls"].sum(axis=0))
    g_sq_sum = np.asarray(out["g_sq"].sum(axis=0))

    ref_ls = np.asarray(jax.grad(
        lambda th: dg.batch_loss(model, init, ys, th, WARMUP, xs=X)[0])(theta))
    ref_sq = np.asarray(jax.grad(
        lambda th: dg.batch_loss(model, init, ys, th, WARMUP, xs=X)[1])(theta))

    assert np.isfinite(g_ls_sum).all() and np.isfinite(ref_ls).all()
    assert np.allclose(g_ls_sum, ref_ls, rtol=1e-6, atol=1e-8), (name, g_ls_sum, ref_ls)
    assert np.allclose(g_sq_sum, ref_sq, rtol=1e-6, atol=1e-8), (name, g_sq_sum, ref_sq)


def test_finite_difference_logscore_with_a_tail():
    """FD check of the tailed batch gradient, validating the reverse-mode
    reference the gate above compares against.

    Only the log score is FD-checkable: the squared-error loss carries
    ``stop_gradient(Q)``, so a finite perturbation moves the real Q in the
    denominator — a term the analytic gradient deliberately drops — and FD would
    test the wrong quantity. Same reasoning as ``test_discount_grid``'s.
    """
    model = _tail_model()
    ys, X = _data()
    init = _init_state(model, ys)
    theta = _theta(model)
    eps = 1e-6

    f = lambda th: float(dg.batch_loss(model, init, ys, th, WARMUP, xs=X)[0])
    ana = np.asarray(jax.grad(
        lambda th: dg.batch_loss(model, init, ys, th, WARMUP, xs=X)[0])(theta))
    fd = np.array([(f(theta.at[i].add(eps)) - f(theta.at[i].add(-eps))) / (2 * eps)
                   for i in range(model.n_params)])
    assert np.allclose(ana, fd, rtol=1e-4, atol=1e-5), (ana, fd)


def test_the_tail_carries_real_gradient_signal():
    """Guard against a vacuous gate.

    If the regressors never reached the jvp, the two sides above would agree on a
    tail-free quantity and pass while testing nothing — the failure mode that has
    already cost this project twice (NaN-vs-NaN, and permuted regressor columns).
    So: the tail block's gradient must be materially non-zero, and SCALING the
    regressors must move it.
    """
    model = _tail_model()
    ys, X = _data()
    init = _init_state(model, ys)
    theta = _theta(model)

    g = lambda x: np.asarray(jax.grad(
        lambda th: dg.batch_loss(model, init, ys, th, WARMUP, xs=x)[0])(theta))
    g1, g2 = g(X), g(X * 10.0)

    assert np.isfinite(g1).all()
    assert abs(g1[1]) > 1e-6, ("tail block gradient is ~0 — tail inert?", g1)
    assert not np.allclose(g1, g2), ("scaling the regressors did not move the "
                                     "gradient — the tail is not reaching the jvp")


def test_structural_run_is_unchanged_by_the_new_argument():
    """``xs=None`` is the old code path, bit-for-bit.

    ``grad_run``/``batch_loss`` gained an ``xs`` argument; a structural model must
    not notice. Bit-exact, not merely close — the standard every other structural
    guard in this work was held to."""
    model = _structural_model()
    ys, _X = _data()
    init = _init_state(model, ys)
    theta = _theta(model)
    theta_seq = jnp.tile(theta, (T, 1))

    a = dg.grad_run(model, init, ys, theta_seq, warmup=WARMUP)
    b = dg.grad_run(model, init, ys, theta_seq, warmup=WARMUP, xs=None)
    for k in ("g_ls", "g_sq", "ell_ls", "ell_sq", "f", "q"):
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))

    l_a = dg.batch_loss(model, init, ys, theta, WARMUP)
    l_b = dg.batch_loss(model, init, ys, theta, WARMUP, xs=None)
    np.testing.assert_array_equal(np.asarray(l_a), np.asarray(l_b))


def test_a_zero_width_tail_reduces_to_the_structural_run():
    """``n_regs = 0`` with an empty ``xs`` equals the structural run.

    This drives the REGRESSOR scan branch (a zero-width leaf) against a model
    whose ``_F_at`` short-circuits on ``n_regs == 0``, so it is a narrow claim:
    the tail's plumbing is an extension of the structural path and degrades to it
    rather than running alongside it. The free-reduction property the iterated
    forecast is tested on (``test_grid_ar_forecast``), applied to the learning
    signal."""
    model = _structural_model()
    ys, _X = _data()
    init = _init_state(model, ys)
    theta = _theta(model)
    theta_seq = jnp.tile(theta, (T, 1))

    ref = dg.grad_run(model, init, ys, theta_seq, warmup=WARMUP)
    got = dg.grad_run(model, init, ys, theta_seq, warmup=WARMUP,
                      xs=jnp.zeros((T, 0)))
    for k in ("g_ls", "g_sq", "ell_ls", "ell_sq", "f", "q"):
        np.testing.assert_allclose(np.asarray(got[k]), np.asarray(ref[k]),
                                   rtol=0, atol=0)

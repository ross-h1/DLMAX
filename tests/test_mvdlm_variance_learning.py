"""Learning ``Sigma`` inside ``mv_dlm``, rather than being handed it.

Where this sits
---------------
The Quintana/West collapse is scale-free: ``C_t``, ``R_t``, ``Q_t`` and ``A_t``
are common to every series and free of the data, and that holds for ANY
``Sigma``. So the filter never needed to know ``Sigma`` -- and consequently
``mv_dlm`` never learned it. BayesFR supplied it externally from its factor
model, and the model's own ``scale`` tracked only the ``q`` diagonal variances,
by

    scale = (error ** 2 / Q + scale * n) / (n + 1)

which is a recursion with no discount and no off-diagonal. For a VAR that is
not enough: the cross-series covariance is the object of interest, and there is
no external ``Sigma`` to fall back on.

``DLMAX.wishart`` holds the generalised recursion. This file is about wiring it
in, and the wiring has one overriding constraint.

The constraint: the default path must not move
----------------------------------------------
``dlm_forward`` and the ``params`` pytree are load-bearing for BayesFR --
``csrec`` jits the Gibbs kernel with ``params`` as an argument, and
``HF_GIBBS_OPT_direct`` unpacks ``dlm_forward``'s 4-tuple. So variance learning
is **opt-in**: with ``delta_sigma=None`` (the default) the model runs the cheap
diagonal recursion it always ran, bit for bit, and the ``params`` pytree gains
no keys.

That is not merely conservative. The full path is ``(T, q, q)``; at BayesFR's
``q = 304`` over 131 periods that is ~97 MB allocated on every Gibbs sweep, to
hold a quantity that project does not use. Opt-in is the right default on
memory grounds alone.

What makes this safe rather than a fork
---------------------------------------
``test_wishart.py`` pins ``diag(wishart) == today's scale`` **bitwise**, so the
two paths cannot silently diverge; and ``test_the_two_paths_agree_at_delta_one``
below asserts that same identity at the ``mv_dlm`` level, through the real
filter rather than on synthetic errors. The diagonal path is an optimisation of
the general one, and is tested as such.

The prior
---------
``n0`` and ``S0``. ``S0`` defaults to ``diag(V0)`` -- the observation-variance
prior the model already takes -- and ``n0`` to ``1``, which is what the existing
recursion hardcodes. Those defaults are exactly what make the ``delta_sigma=1``
reduction hold, so they are not arbitrary: any other choice would be a
different model, and would announce itself by breaking the oracle.
"""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from DLMAX.mvdlm import mv_dlm, congruence, dlm_params, dlm_forward

T, Q_DIM, P = 60, 4, 2


def _model(**kw):
    rng = np.random.default_rng(31)
    Y = np.cumsum(rng.normal(0, 1, (T, Q_DIM)), axis=0) + 10.0
    F = jnp.array([[1.0], [0.0]])
    G = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    delta = congruence(jnp.array([0.95, 0.98]))
    return mv_dlm(data=Y, F=F, G=G, m0=jnp.zeros((P, Q_DIM)),
                  C0=jnp.eye(P), V0=jnp.ones((1, Q_DIM)),
                  disc_mtx=delta, **kw), Y


# --------------------------------------------------------------------------
# the default must not move
# --------------------------------------------------------------------------

def test_learning_is_off_by_default():
    m, _ = _model()
    assert m.delta_sigma is None
    m.scan_filter()
    assert m.sigma_scale is None
    assert m.sigma_dof is None


def test_the_default_params_pytree_gains_no_keys():
    """``csrec`` jits its Gibbs kernel with this dict as an argument. Adding
    keys unconditionally would retrace it and, worse, would make the default
    model carry a (T, q, q) array it never reads."""
    m, _ = _model()
    base = set(m.params().keys())
    assert "sigma_scale" not in base and "sigma_dof" not in base


def test_the_functional_shim_is_untouched():
    """``dlm_forward``'s 4-tuple and its shapes are what BayesFR unpacks."""
    _, Y = _model()
    F = jnp.array([[1.0], [0.0]])
    G = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    p = dlm_params(Y=Y, m0=jnp.zeros((P, Q_DIM)), C0=jnp.eye(P), F=F, G=G,
                   V0=jnp.ones((1, Q_DIM)),
                   disc_mtx=congruence(jnp.array([0.95, 0.98])))
    a, m_, n, scale = dlm_forward(jnp.asarray(Y), p, jnp.zeros((P, Q_DIM)),
                                  jnp.ones((1, Q_DIM)))
    assert scale.shape == (T, 1, Q_DIM)
    assert a.shape == (T, P, Q_DIM)


# --------------------------------------------------------------------------
# THE gate, at the model level
# --------------------------------------------------------------------------

def test_the_two_paths_agree_at_delta_one():
    """``diag(sigma_scale)`` must reproduce the diagonal ``scale`` EXACTLY.

    ``test_wishart.py`` proves this for the recursion in isolation, on
    synthetic errors. This proves the WIRING: that the errors handed to the
    Wishart recursion are the filter's own one-step errors, normalised by the
    filter's own ``Q_t``, at the same step. Feeding it ``a`` instead of ``m``,
    or ``Q_{t-1}``, or the error before the update, would all leave the
    isolated test passing and this one failing.
    """
    ref, _ = _model()
    ref.scan_filter()
    got, _ = _model(delta_sigma=1.0)
    got.scan_filter()
    diag = jnp.diagonal(got.sigma_scale, axis1=1, axis2=2)
    np.testing.assert_array_equal(np.asarray(diag),
                                  np.asarray(ref._filtered[3]).reshape(T, Q_DIM))


def test_the_dof_matches_the_counter_at_delta_one():
    got, _ = _model(delta_sigma=1.0)
    got.scan_filter()
    np.testing.assert_array_equal(np.asarray(got.sigma_dof),
                                  np.arange(2, T + 2, dtype=float))


def test_the_streaming_face_agrees_with_the_scan():
    """``fwd_filter`` and ``scan_filter`` are two faces of one recursion, and
    the library's contract is that they agree. Learning must not open a gap
    between them."""
    a, Y = _model(delta_sigma=0.97)
    a.scan_filter()
    b, _ = _model(delta_sigma=0.97)
    for t in range(T):
        b.fwd_filter(jnp.asarray(Y[t]))
    np.testing.assert_allclose(np.asarray(b.sigma_scale_now),
                               np.asarray(a.sigma_scale[-1]),
                               rtol=1e-12, atol=0)
    assert abs(float(b.sigma_dof_now) - float(a.sigma_dof[-1])) < 1e-12


# --------------------------------------------------------------------------
# what learning buys
# --------------------------------------------------------------------------

def test_the_off_diagonal_is_populated():
    m, _ = _model(delta_sigma=0.99)
    m.scan_filter()
    S = np.asarray(m.sigma_scale[-1])
    assert S.shape == (Q_DIM, Q_DIM)
    off = S[~np.eye(Q_DIM, dtype=bool)]
    assert np.abs(off).max() > 1e-6, "no cross-series covariance was learned"
    np.testing.assert_array_equal(S, S.T)


def test_a_known_Sigma_is_recovered_end_to_end():
    """Plan gate 7. Data generated from a nearly-static state plus ``N(0, Sigma)``
    noise: the filter's one-step errors are then the observation noise itself,
    and the learned scale must be ``Sigma``.

    This is the test that catches a transposed or mis-scaled outer product at
    the WIRING level -- ``delta_sigma = 1`` agreeing with the diagonal would not,
    since the diagonal is invariant to a transpose.
    """
    Sigma = np.array([[1.0, 0.7, 0.0, -0.3],
                      [0.7, 2.0, 0.2, 0.0],
                      [0.0, 0.2, 0.5, 0.1],
                      [-0.3, 0.0, 0.1, 1.5]])
    n = 4000
    rng = np.random.default_rng(99)
    level = np.array([5.0, -2.0, 1.0, 0.0])
    Y = level + rng.multivariate_normal(np.zeros(Q_DIM), Sigma, n)
    F = jnp.array([[1.0]])
    G = jnp.eye(1)
    m = mv_dlm(data=Y, F=F, G=G, m0=jnp.asarray(level).reshape(1, Q_DIM),
               C0=jnp.eye(1) * 1e-8, V0=jnp.ones((1, Q_DIM)),
               disc_mtx=jnp.ones((1, 1)), delta_sigma=1.0)
    m.scan_filter()
    got = np.asarray(m.sigma_scale[-1])
    rel = np.abs(got - Sigma).max() / np.abs(Sigma).max()
    assert rel < 0.08, f"recovered Sigma off by {rel:.1%}\n{got}"


def test_params_surfaces_the_learned_scale_when_on():
    """The plan records ``scale`` as presently DEAD -- computed, returned, and
    discarded by both callers. Part 3 is what makes it load-bearing, so it has
    to be reachable through the pytree the callers actually read."""
    m, _ = _model(delta_sigma=0.98)
    m.scan_filter()
    p = m.params()
    assert p["sigma_scale"].shape == (T, Q_DIM, Q_DIM)
    assert p["sigma_dof"].shape == (T,)
    assert all(hasattr(v, "shape") for v in p.values()), \
        "params must stay a flat all-array pytree -- csrec jits it"


# --------------------------------------------------------------------------
# refusals and edges
# --------------------------------------------------------------------------

def test_an_invalid_discount_is_refused_at_construction():
    """Better at construction than T steps later inside a scan."""
    with pytest.raises(ValueError, match="delta_sigma"):
        _model(delta_sigma=0.0)
    with pytest.raises(ValueError, match="delta_sigma"):
        _model(delta_sigma=1.2)


def test_the_prior_scale_can_be_given():
    S0 = np.diag([4.0, 1.0, 9.0, 0.25])
    m, _ = _model(delta_sigma=1.0, S0=S0, n0=3.0)
    m.scan_filter()
    assert float(m.sigma_dof[0]) == 4.0
    got = np.asarray(m.sigma_scale[0])
    assert np.all(np.isfinite(got))


def test_a_discount_actually_discounts():
    flat, _ = _model(delta_sigma=1.0)
    flat.scan_filter()
    tracked, _ = _model(delta_sigma=0.9)
    tracked.scan_filter()
    assert not np.allclose(np.asarray(flat.sigma_scale[-1]),
                           np.asarray(tracked.sigma_scale[-1]))
    assert float(tracked.sigma_dof[-1]) < float(flat.sigma_dof[-1])


def test_delta_sigma_one_leaves_the_predictive_bitwise_unchanged():
    """The safety property that makes routing learning into the predictive
    free: at ``delta_sigma = 1`` the discount-Wishart diagonal IS the old
    running average, so turning learning on changes nothing until a discount
    is actually chosen."""
    plain, Y = _model()
    learn, _ = _model(delta_sigma=1.0)
    for t in range(15):
        a = plain.fwd_filter(jnp.asarray(Y[t]))
        b = learn.fwd_filter(jnp.asarray(Y[t]))
        np.testing.assert_array_equal(np.asarray(a.var), np.asarray(b.var))


def test_the_predictive_tracks_the_variance_discount():
    """Choosing a variance discount must move the intervals.

    The alternative -- learning a discounted Sigma and then reporting
    predictive variances computed from an undiscounted one -- would be half a
    feature: the model would hold an estimate it declined to use for its own
    uncertainty. So ``delta_sigma`` reaches the predictive, and this asserts it
    rather than leaving it to inspection.
    """
    plain, Y = _model()
    learn, _ = _model(delta_sigma=0.9)
    diffs = []
    for t in range(30):
        a = plain.fwd_filter(jnp.asarray(Y[t]))
        b = learn.fwd_filter(jnp.asarray(Y[t]))
        diffs.append(np.abs(np.asarray(a.var) - np.asarray(b.var)).max())
    assert max(diffs) > 1e-6, (
        "the variance discount made no difference to any one-step predictive")


def test_the_returned_scale_is_the_learned_diagonal():
    """``scale`` has one meaning -- the model's estimate of ``diag(Sigma)`` --
    and ``delta_sigma`` chooses how it is estimated. Returning the undiscounted
    diagonal next to a discounted ``sigma_scale`` would be two answers to one
    question."""
    m, _ = _model(delta_sigma=0.93)
    a, m_, n, scale = m.scan_filter()
    want = jnp.diagonal(m.sigma_scale, axis1=1, axis2=2).reshape(T, 1, Q_DIM)
    np.testing.assert_array_equal(np.asarray(scale), np.asarray(want))


def test_the_scan_scale_is_bitwise_unchanged_at_delta_one():
    """...and the substitution above is a no-op where it must be."""
    ref, _ = _model()
    ref.scan_filter()
    got, _ = _model(delta_sigma=1.0)
    got.scan_filter()
    np.testing.assert_array_equal(np.asarray(got._filtered[3]),
                                  np.asarray(ref._filtered[3]))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly", "--tb=short"]))

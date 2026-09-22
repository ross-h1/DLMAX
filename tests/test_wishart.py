"""The discount-Wishart recursion for a matrix-normal observation variance.

What this is
------------
A matrix-normal DLM observes ``y_t' = F_t'Theta_t + nu_t`` with
``nu_t ~ N(0, v_t Sigma)``. ``Sigma`` is the one thing the Quintana/West
collapse does NOT make free: the ``(x) Sigma`` factorisation holds for any
``Sigma``, so the filter can run scale-free, but somebody still has to say what
``Sigma`` is. Today ``mvdlm`` tracks only its DIAGONAL, by

    scale = (error ** 2 / Q + scale * n) / (n + 1)

which is the recursion below at ``delta = 1``, elementwise. This module is that
line generalised in the two directions it is missing: the full ``q x q`` outer
product, and a discount.

    n_t = delta * n_{t-1} + 1
    S_t = delta * S_{t-1} + e_t e_t' / Q_t          Sigma | D_t ~ IW(n_t, S_t)

Per Prado, Ferreira & West (2nd ed) the Bartlett-decomposition route gives the
identical formulation to the sequential matrix-beta model, so this IS the
standard recursion and not a variant of it.

Why the SCALE matrix, not S
---------------------------
The functions carry ``S_t / n_t`` -- the point estimate -- rather than the
unnormalised ``S_t``. Algebraically identical; numerically not, and the
difference is the whole reason this file can be written as a specification:

* carrying ``S`` and reporting ``diag(S_t)/n_t`` agrees with today's line to
  3.4e-16 -- machine precision, but NOT bitwise, because the operations land
  in a different order;
* carrying ``S_t/n_t`` reproduces today's operation sequence exactly, and
  agrees **bitwise**.

So the recursion in scale form is

    n_t  = delta * n_{t-1} + 1
    S~_t = (e_t e_t'/Q_t + S~_{t-1} * (delta * n_{t-1})) / n_t

and ``test_delta_one_is_bitwise_the_existing_scalar_recursion`` is an exact
oracle rather than a tolerance. That is not a cosmetic preference: a tolerance
of 1e-13 would pass for a recursion that had the discount attached to the wrong
term, and an exact oracle will not.

What is deliberately NOT here
-----------------------------
The inverse-Wishart posterior MEAN. Turning ``(S~, n)`` into ``E[Sigma | D_t]``
needs a degrees-of-freedom convention, and the conventions in the literature
differ by exactly the kind of ``q + 1`` that no test written against this module
could catch -- the recursion is correct under all of them. That is a modelling
judgement belonging at the call site, where the convention can be stated
alongside the prior it is paired with. Shipping a plausible-looking helper here
would launder a judgement into an implementation detail.

Likewise absent: any check relating the limiting degrees of freedom to ``q``.
``n_t -> 1/(1 - delta)``, so ``delta = 0.99`` sustains ``n = 100`` however long
the series, and ``delta = 0.9`` only 10. Two reasons not to police that.

First, the discount makes ``n_t`` non-integer at essentially every step, and the
classical Wishart -- a sum of ``n`` outer products -- is defined only for
integer ``n``. It is the **Bartlett decomposition** that defines it for real
``n``, and Uhlig's singular matrix-beta that carries the evolution below
``q``. So a small, fractional dof is the construction working as designed, not
a violation to be caught.

Second, and concretely: the returned ``S~`` stays full rank and Cholesky-able
even when ``n < q``, because the geometric sum never drops a direction and the
prior is carried forward at weight ``delta^t n_0 / n_t``. Measured at ``q = 20``,
``delta = 0.9`` (limiting dof 10) over 800 steps: rank 20, condition number 136,
Cholesky fine. What a small dof costs is **precision, not well-posedness** --
the same run recovers ``Sigma`` to only ~75% relative error, which is what an
effective sample of 10 buys for a 20x20 covariance. Choosing ``delta`` is a
bias-variance trade, and the caller makes it; ``limiting_dof`` exists so they
can see which side of it they are on.

Where dof does bite is DRAWING ``Sigma`` rather than estimating it -- a
nonsingular draw needs rank, and no amount of discounting creates it. That is
the caller's problem at the point of the draw, not this module's.
"""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from DLMAX.wishart import wishart_step, wishart_scan, limiting_dof

Q_DIM, T = 4, 80


@pytest.fixture(scope="module")
def errors_and_Q():
    rng = np.random.default_rng(20260922)
    e = jnp.asarray(rng.normal(size=(T, Q_DIM)))
    Q = jnp.asarray(rng.uniform(0.8, 2.0, T))
    S0 = jnp.diag(jnp.asarray(rng.uniform(0.5, 2.0, Q_DIM)))
    return e, Q, S0


def _todays_line(e, Q, s0):
    """The EXACT line from ``mv_dlm.scan_filter``, run elementwise -- and run
    inside a ``lax.scan``, as the real one is.

    Reproduced here rather than imported because it is the specification being
    generalised: this file must keep pinning it even after the caller is
    rewritten to use the new recursion.

    The ``scan`` is not incidental. XLA fuses a scan differently from an eager
    Python loop, and the two land ~1e-15 apart -- so the same recursion written
    as a loop would NOT compare bitwise, and a test that did so would be
    pinning the trace shape rather than the arithmetic. The claim being made is
    that the new recursion reproduces today's *code path*, which is a scan.
    """
    def step(carry, xs):
        scale, n = carry
        e_t, Q_t = xs
        scale = (e_t ** 2 / Q_t + scale * n) / (n + 1)
        return (scale, n + 1), scale

    _, out = jax.lax.scan(step, (s0, 1), (e, Q))
    return out


# --------------------------------------------------------------------------
# THE gate -- the exact oracle
# --------------------------------------------------------------------------

def test_delta_one_is_bitwise_the_existing_scalar_recursion(errors_and_Q):
    """At ``delta = 1`` the diagonal must reproduce today's ``scale`` EXACTLY.

    Not to a tolerance. The scale-form recursion preserves the existing
    operation order, so this is available as bitwise equality, and taking it
    is what makes the rest of this file a specification rather than a hope.
    """
    e, Q, S0 = errors_and_Q
    scale, n = wishart_scan(S0, 1.0, e, Q, delta=1.0)
    want = _todays_line(e, Q, jnp.diag(S0))
    got = jnp.diagonal(scale, axis1=1, axis2=2)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_delta_one_dof_is_the_existing_counter(errors_and_Q):
    """``n`` after t steps is ``t + 1``, matching the ``n + 1`` in the scan."""
    e, Q, S0 = errors_and_Q
    _, n = wishart_scan(S0, 1.0, e, Q, delta=1.0)
    np.testing.assert_array_equal(np.asarray(n), np.arange(2, T + 2, dtype=float))


# --------------------------------------------------------------------------
# step / scan consistency
# --------------------------------------------------------------------------

def test_scan_is_the_step_iterated(errors_and_Q):
    """The compiled path and the obvious loop must not drift apart."""
    e, Q, S0 = errors_and_Q
    for delta in (1.0, 0.98):
        S, n = S0, 1.0
        want_S, want_n = [], []
        for t in range(T):
            S, n = wishart_step(S, n, e[t], Q[t], delta)
            want_S.append(S)
            want_n.append(n)
        got_S, got_n = wishart_scan(S0, 1.0, e, Q, delta=delta)
        # Not bitwise: ``wishart_scan`` compiles a ``lax.scan`` while the loop
        # above runs eagerly, and XLA fuses the two differently. 1e-13 is four
        # orders tighter than any real defect and two looser than the ~1e-15
        # the fusion difference actually costs.
        np.testing.assert_allclose(np.asarray(got_S),
                                   np.asarray(jnp.stack(want_S)),
                                   rtol=1e-13, atol=0)
        np.testing.assert_allclose(np.asarray(got_n),
                                   np.asarray(jnp.stack(want_n)),
                                   rtol=1e-13, atol=0)


# --------------------------------------------------------------------------
# the degrees of freedom
# --------------------------------------------------------------------------

def test_dof_follows_the_closed_form(errors_and_Q):
    """``n_t = delta^t n_0 + (1 - delta^t)/(1 - delta)``."""
    e, Q, S0 = errors_and_Q
    delta, n0 = 0.95, 3.0
    _, n = wishart_scan(S0, n0, e, Q, delta=delta)
    t = np.arange(1, T + 1)
    want = delta ** t * n0 + (1 - delta ** t) / (1 - delta)
    np.testing.assert_allclose(np.asarray(n), want, rtol=1e-12, atol=0)


def test_dof_converges_to_the_limit():
    """The discount buys a bounded memory: ``n -> 1/(1 - delta)``.

    This is the property that decides whether the learned Sigma is usable at a
    given width: the IW is proper only for ``n > q - 1``, so ``delta = 0.99``
    sustains ``n = 100`` no matter how long the series, and at BayesFR's
    ``q = 304`` it never becomes proper. The caller needs that number, which is
    why ``limiting_dof`` is exported.
    """
    delta, steps = 0.9, 600
    rng = np.random.default_rng(2)
    e = jnp.asarray(rng.normal(size=(steps, Q_DIM)))
    Q = jnp.ones(steps)
    _, n = wishart_scan(jnp.eye(Q_DIM), 1.0, e, Q, delta=delta)
    assert abs(float(n[-1]) - limiting_dof(delta)) < 1e-9
    assert abs(limiting_dof(0.99) - 100.0) < 1e-9
    assert float(n[0]) < float(n[-1])          # approached from below


def test_limiting_dof_is_infinite_without_a_discount():
    assert limiting_dof(1.0) == np.inf


# --------------------------------------------------------------------------
# what the generalisation is FOR: the off-diagonal, and the Q normalisation
# --------------------------------------------------------------------------

def test_the_off_diagonal_is_learned(errors_and_Q):
    """The point of the whole exercise.

    Today's recursion tracks ``q`` variances and no covariances, so a model
    built on it cannot express the cross-series dependence a VAR exists to
    capture. Errors drawn with a strongly correlated ``Sigma`` must produce a
    scale matrix with that correlation in it -- an implementation that kept
    ``diag(e**2)`` would pass every diagonal test in this file.
    """
    Sigma = np.array([[1.0, 0.8, 0.0, 0.0],
                      [0.8, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, -0.6],
                      [0.0, 0.0, -0.6, 1.0]])
    rng = np.random.default_rng(7)
    e = jnp.asarray(rng.multivariate_normal(np.zeros(Q_DIM), Sigma, 40_000))
    Q = jnp.ones(40_000)
    scale, _ = wishart_scan(jnp.eye(Q_DIM), 1.0, e, Q, delta=1.0)
    got = np.asarray(scale[-1])
    np.testing.assert_allclose(got, Sigma, atol=0.03)
    assert abs(got[0, 1]) > 0.5, "the (0,1) covariance was not learned at all"
    assert got[2, 3] < -0.3, "a NEGATIVE covariance must survive too"


def test_a_known_Sigma_is_recovered_through_a_varying_Q(errors_and_Q):
    """``e_t ~ N(0, Q_t Sigma)``, so ``E[e_t e_t'/Q_t] = Sigma`` and the scale
    matrix targets ``Sigma`` itself, not ``Sigma`` times some average of ``Q``.

    This is the test that catches the ``/Q`` being dropped or applied twice --
    with ``Q == 1`` everywhere those mistakes are invisible, so ``Q`` is drawn
    away from 1 deliberately and over a wide range.
    """
    Sigma = np.array([[2.0, 0.5, 0.1, 0.0],
                      [0.5, 1.0, 0.2, 0.1],
                      [0.1, 0.2, 3.0, 0.4],
                      [0.0, 0.1, 0.4, 0.5]])
    rng = np.random.default_rng(11)
    n = 60_000
    Qs = rng.uniform(0.5, 4.0, n)
    L = np.linalg.cholesky(Sigma)
    e = (rng.normal(size=(n, Q_DIM)) @ L.T) * np.sqrt(Qs)[:, None]
    scale, _ = wishart_scan(jnp.eye(Q_DIM), 1.0, jnp.asarray(e),
                            jnp.asarray(Qs), delta=1.0)
    got = np.asarray(scale[-1])
    rel = np.abs(got - Sigma).max() / np.abs(Sigma).max()
    assert rel < 0.05, f"recovered Sigma is off by {rel:.1%}\n{got}"
    # and the specific failure of ignoring Q: that yields Sigma * E[Q] ~ 2.25x
    assert np.abs(got - Sigma * Qs.mean()).max() / np.abs(Sigma).max() > 0.5


# --------------------------------------------------------------------------
# the discount actually discounts
# --------------------------------------------------------------------------

def test_the_discount_tracks_a_variance_shift(errors_and_Q):
    """A step change in the error scale: the discounted estimate must move to
    the new level, the undiscounted one must not (it averages over all history).

    This pins the discount to the RIGHT term. A recursion that discounted the
    degrees of freedom but not the accumulated scale would still show ``n``
    converging, and ``test_dof_follows_the_closed_form`` would still pass.
    """
    rng = np.random.default_rng(5)
    e = np.concatenate([rng.normal(size=(300, Q_DIM)),
                        rng.normal(size=(300, Q_DIM)) * 4.0])
    Q = jnp.ones(600)
    e = jnp.asarray(e)
    tracked, _ = wishart_scan(jnp.eye(Q_DIM), 1.0, e, Q, delta=0.95)
    flat, _ = wishart_scan(jnp.eye(Q_DIM), 1.0, e, Q, delta=1.0)
    v_tracked = float(jnp.trace(tracked[-1])) / Q_DIM
    v_flat = float(jnp.trace(flat[-1])) / Q_DIM
    assert v_tracked > 10.0, f"discounted estimate did not reach the new level ({v_tracked:.2f})"
    assert v_flat < 10.0, f"undiscounted estimate should lag ({v_flat:.2f})"
    assert v_tracked > v_flat * 1.3


def test_the_discount_is_on_the_scale_not_only_the_dof(errors_and_Q):
    """Direct form of the same point, without the statistics.

    With a single observation from a flat start, ``delta`` changes the weight
    given to the prior scale. Write the answer out longhand and compare.
    """
    e = jnp.array([[3.0, 0.0, 0.0, 0.0]])
    Q = jnp.array([2.0])
    S0, n0, delta = jnp.eye(Q_DIM) * 5.0, 4.0, 0.9
    scale, n = wishart_scan(S0, n0, e, Q, delta=delta)
    want_n = delta * n0 + 1.0
    want = (np.outer(np.asarray(e[0]), np.asarray(e[0])) / 2.0
            + np.asarray(S0) * (delta * n0)) / want_n
    np.testing.assert_allclose(np.asarray(scale[0]), want, rtol=1e-14, atol=0)
    assert abs(float(n[0]) - want_n) < 1e-14


# --------------------------------------------------------------------------
# structural properties
# --------------------------------------------------------------------------

def test_the_scale_stays_exactly_symmetric(errors_and_Q):
    """``outer(e, e)`` is symmetric to the bit, and the recursion is affine in
    it, so symmetry should hold exactly -- no ``(S + S.T)/2`` patch-up. If it
    does not, the implementation is forming the outer product asymmetrically
    (an ``e[:, None] @ e[None, :]`` via a route that rounds differently), which
    would slowly break every Cholesky downstream.
    """
    e, Q, S0 = errors_and_Q
    scale, _ = wishart_scan(S0, 1.0, e, Q, delta=0.97)
    s = np.asarray(scale)
    np.testing.assert_array_equal(s, s.swapaxes(1, 2))


def test_the_scale_stays_positive_definite(errors_and_Q):
    """A positive-definite start plus positive-semidefinite increments stays
    positive definite, for any ``delta`` in (0, 1]. This is what lets the
    caller take a Cholesky without a guard."""
    e, Q, S0 = errors_and_Q
    for delta in (1.0, 0.99, 0.8):
        scale, _ = wishart_scan(S0, 1.0, e, Q, delta=delta)
        eig = np.linalg.eigvalsh(np.asarray(scale))
        assert eig.min() > 0, f"delta={delta}: min eigenvalue {eig.min():.3e}"


def test_shapes_and_dtypes(errors_and_Q):
    e, Q, S0 = errors_and_Q
    scale, n = wishart_scan(S0, 1.0, e, Q, delta=0.99)
    assert scale.shape == (T, Q_DIM, Q_DIM)
    assert n.shape == (T,)
    assert scale.dtype == jnp.float64 and n.dtype == jnp.float64
    assert np.all(np.isfinite(np.asarray(scale)))


def test_it_runs_under_jit(errors_and_Q):
    """``csrec`` jits the Gibbs kernel around this, so no Python branch may
    depend on a traced value."""
    e, Q, S0 = errors_and_Q
    f = jax.jit(lambda S, e_, Q_: wishart_scan(S, 1.0, e_, Q_, delta=0.99))
    got, _ = f(S0, e, Q)
    want, _ = wishart_scan(S0, 1.0, e, Q, delta=0.99)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=1e-13, atol=0)


def test_a_single_step_is_shaped_like_its_input(errors_and_Q):
    e, Q, S0 = errors_and_Q
    S, n = wishart_step(S0, 1.0, e[0], Q[0], 0.99)
    assert S.shape == (Q_DIM, Q_DIM)
    assert jnp.ndim(n) == 0


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [0.0, -0.1, 1.5, np.nan])
def test_an_invalid_discount_is_refused(errors_and_Q, delta):
    """``delta`` in (0, 1]. Zero would discard the prior entirely and leave
    ``n_t = 1``, which is not a degenerate case worth supporting silently."""
    e, Q, S0 = errors_and_Q
    with pytest.raises(ValueError, match="delta"):
        wishart_scan(S0, 1.0, e, Q, delta=delta)


def test_a_non_square_scale_is_refused(errors_and_Q):
    e, Q, _ = errors_and_Q
    with pytest.raises(ValueError):
        wishart_scan(jnp.ones((Q_DIM, Q_DIM + 1)), 1.0, e, Q, delta=1.0)


def test_a_mismatched_error_width_is_refused(errors_and_Q):
    e, Q, S0 = errors_and_Q
    with pytest.raises(ValueError):
        wishart_scan(S0, 1.0, e[:, :Q_DIM - 1], Q, delta=1.0)


def test_mismatched_error_and_Q_lengths_are_refused(errors_and_Q):
    e, Q, S0 = errors_and_Q
    with pytest.raises(ValueError):
        wishart_scan(S0, 1.0, e, Q[:-1], delta=1.0)


def test_a_non_positive_dof_is_refused(errors_and_Q):
    e, Q, S0 = errors_and_Q
    with pytest.raises(ValueError, match="n0"):
        wishart_scan(S0, 0.0, e, Q, delta=1.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly", "--tb=short"]))

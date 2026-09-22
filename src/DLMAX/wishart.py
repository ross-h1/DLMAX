"""Learning a matrix-normal DLM's observation covariance, by discount Wishart.

A matrix-normal DLM observes ``y_t' = F_t'Theta_t + nu_t`` with
``nu_t ~ N(0, v_t Sigma)``. The Quintana/West collapse is scale-free, so the
filter runs without knowing ``Sigma`` at all -- which is convenient, and leaves
somebody else to say what ``Sigma`` is. For a VAR there is no somebody else:
the cross-series covariance is the object of interest.

The recursion is the standard sequential one,

    n_t = delta * n_{t-1} + 1
    S_t = delta * S_{t-1} + e_t e_t' / Q_t          Sigma | D_t ~ IW(n_t, S_t)

with ``e_t`` the one-step forecast error and ``Q_t`` the scale-free predictive
variance. Per Prado, Ferreira & West (2nd ed) the Bartlett-decomposition route
yields the identical formulation to the sequential matrix-beta model, so this
is the standard recursion rather than a variant of it.

Why it carries ``S / n``
------------------------
These functions carry the SCALE matrix ``S_t / n_t`` -- the point estimate --
not the unnormalised ``S_t``. The two are algebraically identical and
numerically are not:

* carrying ``S`` and reporting ``diag(S_t)/n_t`` agrees with the scalar
  recursion ``mvdlm`` already used to 3.4e-16 -- machine precision, but not
  bitwise, because the operations land in a different order;
* carrying ``S_t / n_t`` reproduces that operation sequence exactly, and agrees
  **bitwise**.

Hence the form actually implemented:

    n_new     = delta * n + 1.0
    scale_new = (outer(e, e) / Q + scale * (delta * n)) / n_new

The grouping is load-bearing and should not be factored or reassociated. It
buys an exact oracle instead of a tolerance, and a tolerance of 1e-13 would
admit a recursion with the discount attached to the wrong term.

Degrees of freedom
------------------
``n_t -> 1/(1 - delta)``, a bounded memory: ``delta = 0.99`` sustains an
effective sample of 100 however long the series runs, ``delta = 0.9`` only 10.
Two things follow that are easy to get backwards.

The discount makes ``n_t`` non-integer at essentially every step, and the
classical Wishart -- a sum of ``n`` outer products -- is defined only for
integer ``n``. It is the Bartlett decomposition that defines it for real ``n``,
and Uhlig's singular matrix-beta that carries the evolution below ``q``. A
small fractional dof is therefore the construction working as designed, not a
violation to be policed, and nothing here refuses one.

And the returned scale matrix stays full rank and Cholesky-able even when
``n < q``, because the geometric sum never drops a direction and the prior is
carried forward at weight ``delta^t n_0 / n_t``. What a small dof costs is
precision, not well-posedness. Where rank genuinely bites is DRAWING ``Sigma``
rather than estimating it, which is the caller's problem at the point of the
draw. :func:`limiting_dof` is exported so the caller can see where they stand.

What is deliberately absent
---------------------------
Any conversion of ``(scale, n)`` into an inverse-Wishart posterior mean. That
needs a degrees-of-freedom convention, and the conventions in the literature
differ by exactly the sort of ``q + 1`` that no test written against this
module could catch -- the recursion is correct under all of them. It is a
modelling judgement, and belongs at the call site where it can be stated next
to the prior it is paired with.
"""
import jax.numpy as jnp
from jax.lax import scan

__all__ = ["wishart_step", "wishart_scan", "limiting_dof"]


def limiting_dof(delta):
    """The fixed point of ``n_t = delta n_{t-1} + 1``, i.e. the effective
    sample size the discount sustains. ``inf`` when nothing is discounted."""
    return float("inf") if delta >= 1.0 else 1.0 / (1.0 - delta)


def _check_delta(delta):
    # `not (0 < d <= 1)` rather than two comparisons, so a NaN is refused too.
    if not (0.0 < float(delta) <= 1.0):
        raise ValueError(f"delta must lie in (0, 1]; got {delta}")


def wishart_step(scale, n, error, Q, delta):
    """One update.

    Parameters
    ----------
    scale : array ``(q, q)``   current scale matrix ``S/n``
    n : scalar                 current degrees of freedom
    error : array ``(q,)``     one-step forecast error
    Q : scalar                 scale-free predictive variance
    delta : float              discount in (0, 1]

    Notes
    -----
    No validation: this runs inside a compiled scan, so it must contain no
    Python branch on a traced value. :func:`wishart_scan` validates once, up
    front, where it is free.
    """
    n_new = delta * n + 1.0
    scale_new = (jnp.outer(error, error) / Q + scale * (delta * n)) / n_new
    return scale_new, n_new


def wishart_scan(scale0, n0, errors, Qs, delta):
    """Run the recursion over ``T`` steps.

    Parameters
    ----------
    scale0 : array ``(q, q)``  prior scale matrix
    n0 : float                 prior degrees of freedom, positive
    errors : array ``(T, q)``  one-step forecast errors
    Qs : array ``(T,)``        scale-free predictive variances
    delta : float              discount in (0, 1]

    Returns
    -------
    tuple
        ``(scale_path, n_path)`` of shapes ``(T, q, q)`` and ``(T,)``. The
        first entry is the state AFTER the first observation, not the prior.
    """
    _check_delta(delta)
    scale0 = jnp.asarray(scale0, dtype=jnp.float64)
    errors = jnp.asarray(errors, dtype=jnp.float64)
    Qs = jnp.asarray(Qs, dtype=jnp.float64)
    if scale0.ndim != 2 or scale0.shape[0] != scale0.shape[1]:
        raise ValueError(f"scale0 must be square; got {scale0.shape}")
    if errors.ndim != 2 or errors.shape[1] != scale0.shape[0]:
        raise ValueError(
            f"errors must be (T, {scale0.shape[0]}); got {errors.shape}")
    if Qs.shape[0] != errors.shape[0]:
        raise ValueError(f"Qs has {Qs.shape[0]} steps, errors {errors.shape[0]}")
    if float(n0) <= 0.0:
        raise ValueError(f"n0 must be positive; got {n0}")

    def step(carry, xs):
        S, n = carry
        e, Q = xs
        S, n = wishart_step(S, n, e, Q, delta)
        return (S, n), (S, n)

    _, out = scan(step, (scale0, jnp.asarray(n0, dtype=jnp.float64)),
                  (errors, Qs))
    return out

"""A time-varying design for the matrix-normal DLM: known regressors.

What this buys, and what it costs
---------------------------------
The Quintana/West collapse needs one ``F_t`` shared by every series. For a VAR
that is automatic -- every equation has the same right-hand side, which is the
Zellner SUR condition -- so ``F_t = [structural; x_t']`` with ``x_t`` common.
**A per-series design would break the collapse entirely** and put you back to
``q`` separate covariance recursions.

What survives with a time-varying ``F``:

* **shared across series** -- yes, by the common-regressor restriction;
* **independent of the data** -- yes. ``C*`` depends on ``F_t``, ``G``,
  ``delta`` and ``C0``, never on ``Y``, so the one-run construction stands.

What does not: **reuse across rolling origins**. The path becomes a function of
the regressor window, so it must be keyed on that window rather than treated as
a constant of the model.

The oracles
-----------
Two, both exact, and neither requiring a frozen artefact.

1. ``X=None`` must reproduce the existing constant-``F`` path **bitwise**. This
   is a change that must cost nothing when unused.

2. A **constant** ``X`` must equal folding that ``x`` into ``F`` and running the
   constant path. Driving the tail and extending the design are two routes to
   the same model, and they must agree exactly. This is what pins the
   arithmetic: it catches a wrong ``Q``, a wrong ``A``, and -- because the
   comparison is against a genuinely different code path -- a plausible-looking
   recursion that happens to be self-consistent.

The third thing tested is alignment. ``x_t`` must be the row used at step ``t``,
because the design at ``t`` is what forms the predictive for ``y_t``. An
off-by-one there produces a perfectly finite, perfectly plausible path.
"""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.linalg import block_diag

from DLMAX.mvdlm import _build_covariance_path, congruence

T, P_STRUCT, K = 24, 2, 3
P = P_STRUCT + K


@pytest.fixture(scope="module")
def model():
    """A local level + growth, plus a K-wide regression tail."""
    F = jnp.array([[1.0], [0.0]])
    Gs = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    G = block_diag(Gs, jnp.eye(K))
    C0 = jnp.eye(P)
    delta_M = congruence(jnp.concatenate(
        [jnp.array([0.95, 0.98]), jnp.full(K, 0.99)]))
    return dict(F=F, G=G, C0=C0, delta_M=delta_M)


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(11).normal(size=(T, K))


def _build(m, X=None):
    return _build_covariance_path(T=T, F=m["F"], G=m["G"], C0=m["C0"],
                                  delta_M=m["delta_M"], X=X)


# --------------------------------------------------------------------------
# oracle 1: unused costs nothing
# --------------------------------------------------------------------------

def test_no_regressors_is_bitwise_the_constant_path(model):
    """``X=None`` must leave the existing behaviour untouched, exactly. The
    structural model here is P-wide with a zero tail in F, which is what a
    caller who is not using regressors has."""
    F_full = jnp.concatenate([model["F"], jnp.zeros((K, 1))])
    ref = _build_covariance_path(T=T, F=F_full, G=model["G"], C0=model["C0"],
                                 delta_M=model["delta_M"])
    got = _build_covariance_path(T=T, F=F_full, G=model["G"], C0=model["C0"],
                                 delta_M=model["delta_M"], X=None)
    for field in ("C", "R", "Q", "A", "B", "S", "H", "sqrtH"):
        np.testing.assert_array_equal(
            np.asarray(getattr(got, field)), np.asarray(getattr(ref, field)),
            err_msg=field)


# --------------------------------------------------------------------------
# oracle 2: a constant X is the same model as an extended F
# --------------------------------------------------------------------------

def test_constant_regressors_equal_folding_them_into_F(model):
    """THE arithmetic gate. Driving the tail with a constant ``x`` and writing
    that ``x`` into ``F`` are two routes to one model, by different code paths,
    and they must agree exactly."""
    x = np.random.default_rng(3).normal(size=K)
    via_tail = _build(model, X=np.tile(x, (T, 1)))
    F_folded = jnp.concatenate([model["F"], jnp.asarray(x).reshape(K, 1)])
    folded = _build_covariance_path(T=T, F=F_folded, G=model["G"],
                                    C0=model["C0"], delta_M=model["delta_M"])

    # Everything that does not touch the design must be BITWISE equal. These
    # are the bulk of the path, and they are built by identical code in both
    # routes, so anything less than exact here is a real defect.
    for field in ("C", "R", "B", "S", "H"):
        np.testing.assert_array_equal(
            np.asarray(getattr(via_tail, field)),
            np.asarray(getattr(folded, field)), err_msg=field)

    # Q and A contract with the design, and the two routes reach the same
    # contraction by different shapes: the constant branch broadcasts a shared
    # (1, p) against (T, p, p), the varying one carries a batched (T, 1, p).
    # XLA lowers those as different dot_generals, so they accumulate in a
    # different order and land one ulp apart on 2 of 24 entries. Demanding
    # bitwise across two lowerings would pin a platform, not a property.
    # A divides by Q, which lifts the relative gap to ~3.5e-15; 1e-13 keeps
    # two orders of margin while still being far tighter than any real defect
    # could hide under.
    for field in ("Q", "A"):
        np.testing.assert_allclose(
            np.asarray(getattr(via_tail, field)),
            np.asarray(getattr(folded, field)),
            rtol=1e-13, atol=0, err_msg=field)


# --------------------------------------------------------------------------
# the derived quantities, at the time-varying design
# --------------------------------------------------------------------------

def test_Q_uses_the_design_at_each_step(model, X):
    """``Q_t = 1 + F_t' R_t F_t`` with ``F_t = [F; x_t]``. Using a constant F
    here is the obvious mistake and leaves Q smooth and plausible."""
    cov = _build(model, X=X)
    R = np.asarray(cov.R)
    Fs = np.asarray(model["F"]).reshape(-1)
    Ft = np.concatenate([np.tile(Fs, (T, 1)), X], axis=1)          # (T, P)
    want = 1.0 + np.einsum("ti,tij,tj->t", Ft, R, Ft).reshape(T, 1, 1)
    np.testing.assert_allclose(np.asarray(cov.Q), want, rtol=1e-12, atol=0)


def test_A_uses_the_design_at_each_step(model, X):
    """``A_t = R_t F_t / Q_t``."""
    cov = _build(model, X=X)
    R, Q = np.asarray(cov.R), np.asarray(cov.Q)
    Fs = np.asarray(model["F"]).reshape(-1)
    Ft = np.concatenate([np.tile(Fs, (T, 1)), X], axis=1)
    want = np.einsum("tij,tj->ti", R, Ft)[:, :, None] / Q
    np.testing.assert_allclose(np.asarray(cov.A), want, rtol=1e-12, atol=0)


def test_R_is_still_free_of_F(model, X):
    """``R_t = (G C_{t-1} G') / Delta`` carries no design at all -- only Q and
    A see ``F_t``. If R started depending on the design, the smoother (which
    also carries no F) would silently disagree with the filter."""
    cov = _build(model, X=X)
    G = np.asarray(model["G"])
    C = np.asarray(cov.C)
    C_prev = np.concatenate([np.asarray(model["C0"])[None], C[:-1]], axis=0)
    want = (G @ C_prev @ G.T) / np.asarray(model["delta_M"])[None]
    np.testing.assert_allclose(np.asarray(cov.R), want, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------------------
# alignment -- the silent failure
# --------------------------------------------------------------------------

def test_the_design_is_aligned_to_its_own_step(model, X):
    """``x_t`` is the row used at step ``t``: the design at ``t`` forms the
    predictive for ``y_t``. An off-by-one gives a finite, smooth, entirely
    plausible path, so it is checked by shifting X and demanding a difference."""
    base = _build(model, X=X)
    rolled = _build(model, X=np.roll(X, 1, axis=0))
    assert not np.allclose(np.asarray(base.Q), np.asarray(rolled.Q)), (
        "shifting the design changed nothing: the path is not using x_t at t")


def test_a_wrong_length_design_is_refused(model):
    with pytest.raises((ValueError, TypeError)):
        _build(model, X=np.zeros((T - 1, K)))


def test_a_design_as_wide_as_the_state_is_refused(model):
    """There must be a structural part left over.

    Note what is NOT checked: a design narrower than the state but wider than
    the caller intended. The function cannot know the intended structural/tail
    split -- any ``k < p`` is arithmetically legal -- so it validates what it
    can see. An earlier version of this test asserted that ``K+1`` was refused,
    which was simply wrong: with ``p = 5`` a 4-wide tail leaves one structural
    slot and is a perfectly good model.
    """
    with pytest.raises(ValueError, match="cannot be the whole state"):
        _build(model, X=np.zeros((T, P)))


# --------------------------------------------------------------------------
# the properties the collapse still needs
# --------------------------------------------------------------------------

def test_the_path_still_cannot_see_the_data(model, X):
    """Adding a design must not smuggle in an observation argument: the
    collapse depends on the path being computable without Y."""
    import inspect
    params = set(inspect.signature(_build_covariance_path).parameters)
    assert params == {"T", "F", "G", "C0", "delta_M", "X"}, params


def test_shapes_and_finiteness(model, X):
    cov = _build(model, X=X)
    assert np.asarray(cov.C).shape == (T, P, P)
    assert np.asarray(cov.Q).shape == (T, 1, 1)
    assert np.asarray(cov.A).shape == (T, P, 1)
    for field in ("C", "R", "Q", "A", "B", "S", "H", "sqrtH"):
        v = np.asarray(getattr(cov, field))
        assert np.all(np.isfinite(v)), field
        assert v.dtype == np.float64, field


def test_a_time_varying_design_actually_changes_the_path(model, X):
    """Guards the two oracles above from passing because the design is ignored
    altogether."""
    varying = _build(model, X=X)
    constant = _build(model, X=np.tile(X[0], (T, 1)))
    assert not np.allclose(np.asarray(varying.Q), np.asarray(constant.Q))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly", "--tb=short"]))

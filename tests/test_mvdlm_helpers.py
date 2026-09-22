"""Discount and prior helpers for the matrix-normal (Quintana/West) DLM.

Three pure functions, in ``DLMAX.mvdlm``:

``congruence(disc_rates)``
    Per-state discount **vector** -> the West & Harrison congruence **matrix**,
    ``D[i,j] = sqrt(d_i d_j)``.

``component_discount_matrix(shapes, delta)``
    The same, for a model described as blocks: one rate per block, expanded to
    one rate per state, then made congruent.

``seasonal_prior(c0)``
    The zero-sum prior covariance for a seasonal-dummy block, given its
    diagonal.

Why the congruence matrix exists at all
---------------------------------------
The matrix-normal DLM discounts with ``R = P / D`` **elementwise**, where
``P = G C G'``. For that to be the same operation as the univariate engine's
``R = D_v P D_v`` with ``D_v = diag(1/sqrt(delta))``, the matrix entries must be
exactly ``sqrt(d_i d_j)``. ``test_congruence_is_the_WH_R_eq_DPD_identity``
asserts precisely that equivalence, and it is the reason this function is not
simply ``outer(d, d)`` or ``diag(d)``: both of those agree with it on the
diagonal and disagree off it, so a diagonal-only test would pass either.

``seasonal_prior`` is deliberately **singular**. With ``period`` dummy states
summing to a level, ``1'theta`` is not identified, so the prior is made
rank-deficient along ``1`` (W&H sec 8.6). That is not a defect to be
regularised away: it is the path ``smoother._B_and_H``'s pseudo-inverse exists
for, and the reason ``test_smoother.py::test_rank_deficient_prior_uses_pseudo_inverse``
exists. A version that quietly returned something full-rank would pass a naive
"is it a covariance matrix" check and break the smoother's reason for being.

Expected values below are taken from the reference implementation this port
replaces (``BayesFR/mvdlm.py``), so the port is pinned to it rather than merely
to a property that several wrong answers also satisfy.
"""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from DLMAX.mvdlm import congruence, component_discount_matrix, seasonal_prior


# --------------------------------------------------------------------------
# congruence
# --------------------------------------------------------------------------

def test_congruence_exact_reference_values():
    """Pinned against the implementation this replaces."""
    got = np.asarray(congruence(np.array([0.99, 0.995, 0.995])))
    want = np.array([
        [0.99,              0.992496851380396, 0.992496851380396],
        [0.992496851380396, 0.995,             0.995],
        [0.992496851380396, 0.995,             0.995],
    ])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)


def test_congruence_is_sqrt_of_the_outer_product():
    d = np.array([0.90, 0.95, 0.99, 0.999])
    got = np.asarray(congruence(d))
    np.testing.assert_allclose(got, np.sqrt(np.outer(d, d)), rtol=0, atol=1e-15)


def test_congruence_diagonal_is_the_rates_themselves():
    """sqrt(d_i d_i) = d_i. Necessary, and nowhere near sufficient -- see the
    module docstring on why outer(d, d) and diag(d) must also be excluded."""
    d = np.array([0.90, 0.95, 0.99])
    np.testing.assert_allclose(np.diag(np.asarray(congruence(d))), d,
                               rtol=0, atol=1e-15)


def test_congruence_is_symmetric():
    c = np.asarray(congruence(np.array([0.9, 0.95, 0.99, 0.999, 0.8])))
    np.testing.assert_array_equal(c, c.T)


def test_congruence_is_the_WH_R_eq_DPD_identity():
    """THE load-bearing property.

    The matrix path computes ``R = P / Delta`` elementwise; the univariate
    engine computes ``R = D P D`` with ``D = diag(1/sqrt(delta))``. The whole
    reason this helper produces sqrt(d_i d_j) is that those two must coincide.
    If they do not, the covariance path and the kernel that produced it
    disagree, silently.
    """
    rng = np.random.default_rng(0)
    d = np.array([0.90, 0.95, 0.99, 0.999])
    A = rng.normal(size=(4, 4))
    P = A @ A.T                       # any symmetric PSD P will do

    R_matrix = P / np.asarray(congruence(d))
    D = np.diag(1.0 / np.sqrt(d))
    R_kernel = D @ P @ D

    np.testing.assert_allclose(R_matrix, R_kernel, rtol=1e-14, atol=0)


def test_congruence_is_float64():
    """The filter runs in x64; a float32 discount matrix would silently halve
    the precision of every R_t built from it."""
    assert np.asarray(congruence(np.array([0.99, 0.995]))).dtype == np.float64


# --------------------------------------------------------------------------
# component_discount_matrix
# --------------------------------------------------------------------------

def test_component_discount_matrix_expands_blocks_then_makes_congruent():
    """``shapes`` gives one (rows, cols) per component; ``delta`` one rate per
    component. A (4, 4) block means four states that all share its rate.

    This is GDPRun's quarterly model: a level plus a 4-state seasonal dummy,
    at rates 0.99 and 0.995, giving a 5-state model.
    """
    got = np.asarray(component_discount_matrix([(1, 1), (4, 4)], [0.99, 0.995]))
    assert got.shape == (5, 5)
    np.testing.assert_allclose(np.diag(got),
                               np.array([0.99, 0.995, 0.995, 0.995, 0.995]),
                               rtol=0, atol=1e-15)
    # and it really is the congruence of that expanded vector, not just a
    # matrix with the right diagonal
    rates = np.array([0.99, 0.995, 0.995, 0.995, 0.995])
    np.testing.assert_allclose(got, np.sqrt(np.outer(rates, rates)),
                               rtol=0, atol=1e-15)


def test_component_discount_matrix_cross_block_entry():
    """The level/seasonal cross term, recorded in the migration plan as
    0.992497 -- a value that has been checked against independently."""
    got = np.asarray(component_discount_matrix([(1, 1), (4, 4)], [0.99, 0.995]))
    assert got[0, 1] == pytest.approx(0.992496851380396, abs=1e-15)


def test_component_discount_matrix_matches_congruence_of_expanded_rates():
    got = np.asarray(component_discount_matrix(
        [(2, 2), (3, 3), (1, 1)], [0.9, 0.99, 0.95]))
    expanded = np.array([0.9, 0.9, 0.99, 0.99, 0.99, 0.95])
    np.testing.assert_allclose(got, np.asarray(congruence(expanded)),
                               rtol=0, atol=1e-15)


def test_component_discount_matrix_rejects_a_length_mismatch():
    """One rate per block. Zipping to the shorter of the two would drop states
    or rates and say nothing -- a wrong model that runs.

    This was NOT in the first version of this test, and the first
    implementation written against it duly used a bare zip(). Added after the
    fact, which is the honest place to record it.
    """
    with pytest.raises(ValueError):
        component_discount_matrix([(1, 1), (4, 4)], [0.99])
    with pytest.raises(ValueError):
        component_discount_matrix([(1, 1)], [0.99, 0.995])


# --------------------------------------------------------------------------
# seasonal_prior
# --------------------------------------------------------------------------

def test_seasonal_prior_exact_reference_values():
    got = np.asarray(seasonal_prior(np.ones(4)))
    want = np.full((4, 4), -0.25) + np.eye(4)      # 0.75 on the diagonal
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)


def test_seasonal_prior_is_singular_along_ones():
    """The defining property: 1'theta is not identified, so the prior carries
    no mass in that direction. C0 @ 1 must be exactly the zero vector."""
    for p in (4, 7, 12):
        C = np.asarray(seasonal_prior(np.ones(p)))
        np.testing.assert_allclose(C @ np.ones(p), np.zeros(p),
                                   rtol=0, atol=1e-12)


def test_seasonal_prior_has_rank_one_less_than_its_size():
    for p in (4, 12):
        C = np.asarray(seasonal_prior(np.ones(p)))
        assert np.linalg.matrix_rank(C) == p - 1


def test_seasonal_prior_is_symmetric_and_psd():
    """Singular, but never negative-definite: it is still a covariance."""
    C = np.asarray(seasonal_prior(np.array([1.0, 2.0, 0.5, 4.0, 1.5])))
    np.testing.assert_allclose(C, C.T, rtol=0, atol=1e-14)
    w = np.linalg.eigvalsh(C)
    assert w.min() > -1e-12, f"negative eigenvalue {w.min()}"


def test_seasonal_prior_handles_a_non_uniform_diagonal():
    """c0 need not be constant. The zero-sum projection is still exact, and the
    result is NOT simply diag(c0) minus a constant."""
    c0 = np.array([1.0, 2.0, 0.5, 4.0])
    C = np.asarray(seasonal_prior(c0))
    np.testing.assert_allclose(C @ np.ones(4), np.zeros(4), rtol=0, atol=1e-12)
    assert not np.allclose(np.diag(C), c0), "projection must change the diagonal"


if __name__ == "__main__":
    # Runnable as a SCRIPT as well as collected by pytest, because the agent
    # harness executes `python <test_file>` rather than invoking pytest. Without
    # this, running the file would define the test functions, call none of them,
    # exit 0 -- and report a pass for any implementation at all.
    import sys
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly", "--tb=short"]))

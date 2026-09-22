"""The shared covariance path of the matrix-normal (Quintana/West) DLM.

What is being built
-------------------
In the Quintana/West model the state noise is ``W_t (x) Sigma`` and the prior is
``Theta_0 ~ N(m_0, C_0 (x) Sigma)``. That Kronecker structure collapses the
filter: ``C_t``, ``R_t``, ``Q_t`` and ``A_t`` come out **common to every series
and independent of the data**. Only the mean ``m_t`` is per-series.

So the whole covariance path can be computed ONCE, on a single dummy series,
and reused for all ``q`` of them. That is what ``_build_covariance_path``
produces and ``CovariancePath`` holds. At BayesFR's shape it replaces 304
copies of a 13x13 recursion with one.

The data-independence is not an approximation. The oracle these tests compare
against was generated at ``q=4`` and its ``C``, ``A``, ``Q`` and ``B`` sums are
identical, to all printed digits, to a ``q=24`` run and to a ``q=304`` run made
in August on entirely different (real) data.

How it is computed
------------------
Not by re-deriving the recursion. The scale-free covariance recursion IS
DLMAX's own QR filter, so the path is obtained by running a one-series
``uv_dlm`` with a static discount and reading its trajectory back out. That
inherits the library's numerics -- QR covariance roots, the Joseph-form
posterior -- instead of growing a second implementation free to drift from the
first.

Why the assertions are exact
----------------------------
``rtol=0, atol=0``. This is a port, and its whole purpose is to be the same
computation in a new home; a tolerance here would let a subtly different
recursion through, and the difference would then show up much later as a
mismatch no one could localise. The oracle was frozen before any deliberate
change, so anything that moves is an accident.

One thing NOT asserted exactly: ``sqrtH``. Any ``M`` with ``M M' = H`` is a
valid root, and ``H`` is degenerate early on -- at ``t=0`` several singular
values are tied to within 1e-12 relative, because the seasonal states are
undifferentiated before data accumulates. The root is therefore defined only up
to rotation within that subspace, and its entries are not reproducible across
implementations. What IS reproducible is ``sqrtH @ sqrtH.T == H``, which is
what gets checked.
"""
import os

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from DLMAX.mvdlm import CovariancePath, _build_covariance_path

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@pytest.fixture(scope="module", params=["M4", "M12"])
def oracle(request):
    """Quarterly (p=5) and monthly (p=13) reference paths."""
    return np.load(os.path.join(FIXTURES, f"oracle_qw_{request.param}.npz"))


def _build(o):
    return _build_covariance_path(
        T=o["in_Y"].shape[0], F=o["in_F"], G=o["in_G"], C0=o["in_C0"],
        delta_M=o["in_delta_M"])


# --------------------------------------------------------------------------
# shape and type
# --------------------------------------------------------------------------

def test_returns_a_covariance_path_with_the_documented_fields(oracle):
    cov = _build(oracle)
    assert isinstance(cov, CovariancePath)
    for field in ("C", "R", "Q", "A", "B", "H", "sqrtH", "S"):
        assert hasattr(cov, field), f"CovariancePath has no {field!r}"
    assert cov.T == oracle["in_Y"].shape[0]


def test_shapes_match_the_oracle(oracle):
    cov = _build(oracle)
    for field in ("C", "R", "Q", "A", "B", "S", "H", "sqrtH"):
        want = oracle[f"params_{field}"].shape
        got = np.asarray(getattr(cov, field)).shape
        assert got == want, f"{field}: {got} != {want}"


def test_is_float64_throughout(oracle):
    cov = _build(oracle)
    for field in ("C", "R", "Q", "A", "B", "S", "H", "sqrtH"):
        assert np.asarray(getattr(cov, field)).dtype == np.float64, field


# --------------------------------------------------------------------------
# exact parity with the frozen oracle
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["C", "R", "Q", "A", "B", "S", "H"])
def test_exact_parity_with_the_oracle(oracle, field):
    """Bitwise. See the module docstring on why this is not a tolerance."""
    got = np.asarray(getattr(_build(oracle), field))
    np.testing.assert_array_equal(got, oracle[f"params_{field}"])


def test_sqrtH_is_a_root_of_H(oracle):
    """sqrtH's ENTRIES are not reproducible (H is degenerate early, so the root
    is defined only up to rotation) but the product must be H."""
    cov = _build(oracle)
    sqrtH = np.asarray(cov.sqrtH)
    H = np.asarray(cov.H)
    np.testing.assert_allclose(sqrtH @ np.swapaxes(sqrtH, -1, -2), H,
                               rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------
# the properties that make the collapse valid
# --------------------------------------------------------------------------

def test_the_builder_cannot_see_the_data(oracle):
    """Data-independence is the property the whole design rests on, and here it
    is STRUCTURAL: the builder takes no observations, so it cannot consult them
    even in principle.

    The first version of this test built the path twice with identical
    arguments and compared the results, then claimed in its docstring that "a
    version that reached for the data would fail here". It would not have: that
    only proves determinism. Recorded rather than quietly replaced, because a
    test whose docstring overstates it is worse than one that admits its scope.
    """
    import inspect
    params = set(inspect.signature(_build_covariance_path).parameters)
    # X is a DESIGN, not observations -- a known regressor matrix, which the
    # path may depend on. This test exists to keep the OBSERVATIONS out, and it
    # earned its keep by failing when X was added: the allowed set is explicit
    # so a new argument has to be justified rather than slipped in.
    assert params == {"T", "F", "G", "C0", "delta_M", "X"}, params
    for forbidden in ("Y", "y", "data", "obs", "observations", "m0", "V0"):
        assert forbidden not in params, (
            f"{forbidden!r} is an argument: the path could depend on the data")


def test_the_path_is_deterministic(oracle):
    """Same inputs, same answer. Weaker than the above and worth having
    separately, since the builder drives a filter internally."""
    a, b = _build(oracle), _build(oracle)
    for field in ("C", "R", "Q", "A", "B"):
        np.testing.assert_array_equal(np.asarray(getattr(a, field)),
                                      np.asarray(getattr(b, field)))


def test_Q_is_one_plus_the_quadratic_form(oracle):
    """``Q_t = 1 + F' R_t F``, scale-free (V == 1). The ``1`` is the
    observation term; dropping it is a silent under-statement of every
    predictive variance."""
    cov = _build(oracle)
    F = np.asarray(oracle["in_F"]).reshape(-1, 1)
    R = np.asarray(cov.R)
    want = 1.0 + np.einsum("ij,tjk,kl->til", F.T, R, F)
    np.testing.assert_allclose(np.asarray(cov.Q), want, rtol=1e-12, atol=0)


def test_A_is_R_F_over_Q(oracle):
    """The adaptive vector. ``A_t = R_t F / Q_t``."""
    cov = _build(oracle)
    F = np.asarray(oracle["in_F"]).reshape(-1, 1)
    R, Q = np.asarray(cov.R), np.asarray(cov.Q)
    np.testing.assert_allclose(np.asarray(cov.A), (R @ F) / Q,
                               rtol=1e-12, atol=0)


def test_R_is_the_discounted_prior_covariance(oracle):
    """``R_t = (G C_{t-1} G') / Delta`` ELEMENTWISE, seeded with C0.

    The elementwise division by the congruence matrix is the same operation as
    the kernel's ``D P D``; see test_mvdlm_helpers. If these disagree, the path
    and the filter that produced it are computing different models.
    """
    cov = _build(oracle)
    G = np.asarray(oracle["in_G"])
    C0 = np.asarray(oracle["in_C0"])
    C = np.asarray(cov.C)
    C_prev = np.concatenate([C0[None], C[:-1]], axis=0)
    want = (G @ C_prev @ G.T) / np.asarray(oracle["in_delta_M"])[None]
    np.testing.assert_allclose(np.asarray(cov.R), want, rtol=1e-10, atol=1e-14)


def test_covariances_are_symmetric_and_psd(oracle):
    cov = _build(oracle)
    for field in ("C", "R"):
        X = np.asarray(getattr(cov, field))
        np.testing.assert_allclose(X, np.swapaxes(X, -1, -2),
                                   rtol=1e-10, atol=1e-14, err_msg=field)
        w = np.linalg.eigvalsh((X + np.swapaxes(X, -1, -2)) / 2)
        assert w.min() > -1e-9, f"{field} has eigenvalue {w.min()}"


def test_everything_is_finite(oracle):
    cov = _build(oracle)
    for field in ("C", "R", "Q", "A", "B", "S", "H", "sqrtH"):
        X = np.asarray(getattr(cov, field))
        assert np.all(np.isfinite(X)), f"{field} has non-finite entries"


if __name__ == "__main__":
    # Runnable as a script: the agent harness executes `python <test_file>`
    # rather than invoking pytest, and a bare pytest module would define these
    # functions, run none of them and exit 0.
    import sys
    sys.exit(pytest.main([__file__, "-q", "-p", "no:randomly", "--tb=short"]))

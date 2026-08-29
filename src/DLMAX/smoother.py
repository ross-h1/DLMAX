"""Retrospective smoothing (RTS) and forward-filtering backward-sampling (FFBS).

Both operate on a **filtered trajectory** rather than on a model object, so the
same code serves a :class:`~DLMAX.dlm_core.uv_dlm` and, in future, a matrix-normal
(Quintana/West) multivariate DLM. The model classes provide thin convenience
methods that delegate here, mirroring DLMAX's existing split between free-function
kernels and ergonomic wrappers.

Square-root form throughout
---------------------------
The filter carries the covariance as a root ``Z`` with ``C = Z'Z``, never ``C``
itself. Two consequences shape the code:

* The prior root is **reconstructed, not stored**::

      NR_t = Z_{t-1} G' diag(1/sqrt(delta))          R_t = NR_t' NR_t

  which is exactly what ``dlm_uv_fwd_qr_step`` computes internally. Nothing extra
  needs to be persisted by the filter.

* ``B_t = C_t G' R_{t+1}^-1`` is formed by QR-factoring ``NR`` and applying two
  triangular solves, rather than by an explicit ``pinv`` of ``R``.

Covariances are SCALE-FREE
--------------------------
The filter carries ``C_t = s_t C*_t``, with ``s_t`` the running observation-variance
estimate. Running the backward recursion on those raw covariances mixes scales —
``R_{t+1}`` sits at ``s_t`` while ``S_{t+1}`` arrives at ``s_{t+1}`` — and under
variance learning the result is not monotone: smoothed variance can exceed filtered
variance (measured at +5.7 on a 5-state seasonal model). That mixed quantity is what
``DLM_LIB_2`` computes, and it is hard to interpret.

This module therefore divides the scale out (``C* = Z'Z / s``) and runs everything
scale-free. ``S``, ``H`` and ``sqrtH`` are all ``⊗ V`` quantities: multiply by an
observation variance to get an actual covariance. Means (``s``) and gains (``B``)
are unaffected — the scale cancels in ``C G' R^-1``.

For convenience ``S_at_sT`` gives the smoothed covariance at the terminal variance
estimate ``s_T``, the usual "retrospective variance" for a plain ``uv_dlm``. A
Quintana/West model ignores it and supplies Σ through ``right_factor`` instead.

Indefinite H
------------
With **component** discounting at differing rates, the implied system noise
``W = D P D - P`` is not guaranteed PSD (e.g. ``P=[[1,1],[1,1]]``, ``δ=(0.9,0.99)``
gives ``det(W) = -0.0024``), so ``H = C - B R B'`` can be genuinely indefinite. This
is a property of the discount specification, not of the implementation.
``sqrtH`` is the nearest PSD root — negative eigenvalues clipped to zero — and the
size of that clip is reported as ``psd_clip`` so it is never silent.

Exactness
---------
The recursions are exact for a **fixed-discount, additive, Gaussian** model
(``uv_dlm(adapt=None)``, ``mult_comps`` unset). Three model features weaken that,
and are handled differently by :func:`rts_smooth` and :func:`ffbs`:

===========================  ==========================  ====================
feature                      smoothing                   backward sampling
===========================  ==========================  ====================
``var_power != 1``           exact given variance path   permitted, flagged
adaptive discount / monitor  exact given realised delta  rejected
``mult_comps.any()``         approximate (extended RTS)  rejected
===========================  ==========================  ====================

The first two keep the observation mean **linear**, so the model stays
conditionally Gaussian and the result is exact conditional on a realised path.
``mult_comps`` makes the mean itself bilinear in the state
(``F_fn = (a*nmc)@F * (1 + (a*mc)@F)``), so the filter is an EKF and no
conditioning recovers exactness. An approximate retrospective *estimate* is a
defensible thing to return; an approximate backward *sample* inside a Gibbs
sweep is a chain that misses its target, so :func:`ffbs` refuses it unless
``allow_approximate=True``.

API stability
-------------
:class:`SmootherError`, :func:`rts_smooth` and :func:`ffbs` are all re-exported
from ``DLMAX``. ``SmootherError`` carries the package's usual stability
guarantee — a caller must be able to handle a documented failure of
``uv_dlm.smooth()`` / ``.backward_sample()`` without importing from a submodule.

The two free functions are exported but **provisional**, and are explicitly
carved out of that guarantee. They have not yet been exercised by a real
consumer, and their signatures are expected to change: the trajectory dict may
gain keys or become a typed object, ``G`` / ``disc`` should probably be read off
the model rather than passed (``uv_dlm`` already exposes ``G`` and
``applied_disc``, so the current signature invites a mismatch), and
``right_factor`` may generalise into a noise model once a matrix-normal DLM
lands. Any of those would be a BREAKING change to a caller who took the export
as a promise — so prefer the ``uv_dlm`` methods where they suffice, and call the
free functions directly only if you are willing to track changes.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.lax import scan

__all__ = ["rts_smooth", "ffbs", "SmootherError"]


class SmootherError(RuntimeError):
    """Raised when smoothing / sampling is requested on an unsupported model."""


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _prior_root(Z_prev, G, b):
    """Root ``NR`` of the prior covariance ``R = G C G' / delta``, ``C = Z'Z``.

    Mirrors ``dlm_uv_fwd_qr_step`` exactly: ``NR = (Z G') diag(b)`` with
    ``b = 1/sqrt(delta)``, a column scaling, so ``R = NR'NR = D P D`` is the
    West & Harrison component-discount congruence.
    """
    return (Z_prev @ G.T) * b[jnp.newaxis, :]


def _B_and_H(Z, NR, G, rcond=1e-12):
    """Backward gain ``B = C G' R^+`` and its variance ``H = C - B R B'``.

    ``R = NR'NR``, so the SVD ``NR = U diag(sv) V'`` gives ``R = V diag(sv^2) V'``
    directly — the prior covariance's eigendecomposition without ever forming
    ``R``.

    **``R`` is routinely rank-deficient and the inverse must be a pseudo-inverse.**
    The standard West & Harrison seasonal prior (``seasonal_prior``, and
    ``dlm_core.seasonal_prior``) projects out the sum direction so that the
    seasonal effects sum to zero, which makes ``C0`` — and hence every ``R`` —
    exactly one rank short. Measured on a 13-state monthly model: rank 12/13,
    condition ~1e16, smallest singular value ~1e-17. A triangular solve against
    that divides by ~1e-17 and returns garbage in the null direction; ``H`` is
    unaffected (``B R B'`` annihilates it) but the backward *mean*
    ``m_t + B_t(theta_{t+1} - a_{t+1})`` is not, so smoothed means and FFBS draws
    come out wrong.

    Singular values below ``rcond * max`` are treated as zero, giving the
    minimum-norm ``B`` — the same choice ``jnp.linalg.pinv`` makes.
    """
    _u, sv, Vt = jnp.linalg.svd(NR)
    s2 = sv ** 2
    inv_s2 = jnp.where(s2 > rcond * jnp.max(s2), 1.0 / jnp.where(s2 > 0, s2, 1.0), 0.0)
    CGT = Z.T @ (Z @ G.T)                                    # C G'
    Rinv = (Vt.T * inv_s2) @ Vt                              # R^+
    B = CGT @ Rinv
    H = Z.T @ Z - B @ (Vt.T * s2) @ Vt @ B.T                 # C - B R B'
    return B, H


def _sym_sqrt(H):
    """Nearest PSD square root: eigendecompose, clip negatives, take sqrt.

    Returns ``(root, clip)`` where ``root @ root.T`` is the PSD projection of ``H``
    and ``clip`` is the largest magnitude of eigenvalue clipped (0.0 if ``H`` was
    already PSD). See the module docstring on indefinite ``H``.
    """
    Hs = (H + H.swapaxes(-1, -2)) / 2.0
    w, v = jnp.linalg.eigh(Hs)
    clip = jnp.max(jnp.clip(-w, 0.0))
    return v * jnp.sqrt(jnp.clip(w, 0.0))[..., jnp.newaxis, :], clip


def _validate(model, *, sampling, allow_approximate):
    """Guard the approximate paths. See the module docstring table."""
    reasons = []
    if getattr(model, "mult_comps", None) is not None and bool(
        jnp.any(model.mult_comps)
    ):
        reasons.append(
            "mult_comps is set: the observation mean is bilinear in the state, so "
            "the filter is an EKF and the backward recursion is an extended (not "
            "exact) RTS. No conditioning recovers exactness"
        )
    if getattr(model, "adapt", None) is not None:
        reasons.append(
            "the discount is state-adaptive (adapt is not None), so W_t depends on "
            "the filtered state; results are exact only conditional on the realised "
            "discount path. Construct with uv_dlm(..., adapt=None) for a fixed "
            "discount"
        )
    if getattr(model, "_adapt", None) is not None:
        reasons.append("enable_adapt (RTRL discount learning) is active")
    if getattr(model, "_wing", None) is not None:
        reasons.append("enable_wing is active")

    approximate = bool(reasons)
    if approximate and sampling and not allow_approximate:
        raise SmootherError(
            "backward sampling is not exact for this model:\n  - "
            + "\n  - ".join(reasons)
            + "\nAn approximate draw inside a Gibbs sweep gives a chain that does "
              "not target the posterior. Pass allow_approximate=True to override."
        )
    return approximate, reasons


# -----------------------------------------------------------------------------
# retrospective smoothing
# -----------------------------------------------------------------------------


def rts_smooth(traj, G, disc):
    """West & Harrison retrospective (RTS) smoothing, in square-root form.

    Parameters
    ----------
    traj : dict
        Filtered trajectory as produced by ``uv_dlm.fwd_filter(trajectory=True)``:
        ``m`` (T, q, p), ``Z`` (T, q, p, p), ``s`` (T, q, 1). Extra keys ignored.
    G : array, shape (p, p)
        State transition matrix (constant).
    disc : array, shape (p,)
        Applied per-state discount ``delta`` — for ``uv_dlm(adapt=None)`` this is
        ``disc_rates * disc_rates_damped``.

    Returns
    -------
    dict
        ``s`` (T, q, p) smoothed state means; ``S`` (T, q, p, p) **scale-free**
        smoothed state covariances; ``S_at_sT`` the same at the terminal variance
        estimate; ``B`` (T-1, q, p, p) backward gains; ``H`` / ``sqrtH``
        (T-1, q, p, p) scale-free backward-sampling variances and their nearest
        PSD roots; ``psd_clip`` the largest eigenvalue magnitude clipped in
        forming ``sqrtH``; ``scale`` (T, q, 1) the filtered variance path.
        Index ``t`` of ``B``/``H`` refers to the step from ``t`` to ``t+1``.

    Notes
    -----
    The recursion carries no ``F``, so it is identical for additive and
    multiplicative models — which is exactly why the latter must be guarded
    upstream rather than detected here.
    """
    m, Z = jnp.asarray(traj["m"]), jnp.asarray(traj["Z"])
    b = 1.0 / jnp.sqrt(jnp.asarray(disc))
    # Scale-free roots: C = s C*, so Z* = Z / sqrt(s). Everything below is (x) V.
    scale = jnp.asarray(traj["s"]).reshape(m.shape[0], m.shape[1], 1)
    Z = Z / jnp.sqrt(scale)[..., jnp.newaxis]

    v_prior_root = vmap(_prior_root, in_axes=(0, None, None))
    v_B_and_H = vmap(_B_and_H, in_axes=(0, 0, None))

    # NR[t] is the prior root at t+1, built from the posterior root at t.
    NR = v_prior_root(Z[:-1].reshape((-1,) + Z.shape[2:]), G, b).reshape(Z[:-1].shape)
    B, H = vmap(v_B_and_H, in_axes=(0, 0, None))(Z[:-1], NR, G)
    a_next = jnp.einsum("ij,tqj->tqi", G, m[:-1])                 # a_{t+1} = G m_t
    R_next = NR.swapaxes(-1, -2) @ NR

    C = Z.swapaxes(-1, -2) @ Z

    def back(carry, xs):
        s_next, S_next = carry
        m_t, C_t, B_t, a_t1, R_t1 = xs
        s_t = m_t + jnp.einsum("qij,qj->qi", B_t, s_next - a_t1)
        S_t = C_t - B_t @ (R_t1 - S_next) @ B_t.swapaxes(-1, -2)
        return (s_t, S_t), (s_t, S_t)

    (_, _), (s_hist, S_hist) = scan(
        back, (m[-1], C[-1]), (m[:-1], C[:-1], B, a_next, R_next), reverse=True
    )

    sqrtH, clip = _sym_sqrt(H)
    S = jnp.concatenate([S_hist, C[-1][jnp.newaxis]], 0)
    return {
        "s": jnp.concatenate([s_hist, m[-1][jnp.newaxis]], 0),
        "S": S,
        "S_at_sT": S * scale[-1][jnp.newaxis, ..., jnp.newaxis],
        "B": B,
        "H": H,
        "sqrtH": sqrtH,
        "psd_clip": clip,
        "scale": scale,
    }


# -----------------------------------------------------------------------------
# forward-filtering backward-sampling
# -----------------------------------------------------------------------------


def ffbs(key, traj, G, disc, n_draws=1, right_factor=None, smoothed=None):
    """Draw state trajectories from the joint posterior (Fruehwirth-Schnatter;
    Carter & Kohn).

    The backward draw is

    .. code-block:: text

        Theta_T = m_T   + sqrt(C_T) Z_T L'
        Theta_t = h_t   + sqrtH_t   Z_t L' ,   h_t = m_t + B_t(Theta_{t+1} - a_{t+1})

    Parameters
    ----------
    key : PRNGKey
    traj : dict
        Filtered trajectory (see :func:`rts_smooth`).
    G, disc
        As for :func:`rts_smooth`.
    n_draws : int, default 1
        Number of independent trajectories.
    right_factor : array, optional
        ``L`` above, shape ``(q, q)``, such that the state noise is
        ``H_t (x) L L'``. Since ``H`` here is **scale-free** (see the module
        docstring), ``L`` is where the observation-variance scale enters.

        * ``None`` (default): ``L = diag(sqrt(s_T))`` — each series independent, at
          its own terminal variance estimate. The usual ``uv_dlm`` reading.
        * Quintana/West: pass ``chol(Sigma)``. The series then couple through Σ
          exactly, and because ``C*`` is common across series the induced
          cross-series correlation is precisely Σ's.

        The univariate case is therefore not a special branch — it is this same
        expression with a diagonal right factor.
    smoothed : dict, optional
        Reuse an existing :func:`rts_smooth` result instead of recomputing it.

    Returns
    -------
    array, shape ``(n_draws, T, q, p)``
    """
    sm = rts_smooth(traj, G, disc) if smoothed is None else smoothed
    m = jnp.asarray(traj["m"])
    T, q, p = m.shape

    a_next = jnp.einsum("ij,tqj->tqi", G, m[:-1])
    # Terminal draw uses the scale-free filtered posterior C*_T = S[-1], matching
    # the scale-free sqrtH used at every earlier step. Note m_T, not a_T: the
    # final observation's update belongs in the terminal state draw.
    sqrtC_T, _ = _sym_sqrt(sm["S"][-1])

    # Default L = diag(sqrt(s_T)): independent series at their terminal variance.
    L = (jnp.diag(jnp.sqrt(sm["scale"][-1].reshape(q)))
         if right_factor is None else jnp.asarray(right_factor))

    def shape_noise(root, z):
        """root @ z, then couple across series by L on the right."""
        return L @ jnp.einsum("qij,qj->qi", root, z)      # (q,q) @ (q,p)

    def one(k):
        kT, krest = random.split(k)
        theta_T = m[-1] + shape_noise(sqrtC_T, random.normal(kT, (q, p)))

        def back(theta_next, xs):
            m_t, B_t, a_t1, sqrtH_t, z_t = xs
            h = m_t + jnp.einsum("qij,qj->qi", B_t, theta_next - a_t1)
            theta = h + shape_noise(sqrtH_t, z_t)
            return theta, theta

        zs = random.normal(krest, (T - 1, q, p))
        _, thetas = scan(back, theta_T, (m[:-1], sm["B"], a_next, sm["sqrtH"], zs),
                         reverse=True)
        return jnp.concatenate([thetas, theta_T[jnp.newaxis]], 0)

    return vmap(one)(random.split(key, n_draws))

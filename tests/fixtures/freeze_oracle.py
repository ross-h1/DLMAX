"""Freeze a reference oracle for the Quintana/West MVDLM. Recreates Phase 0.75.

    python diagnostics/freeze_oracle.py [--out diagnostics/oracle_qw.npz]

Why this exists again
---------------------
The original ``diagnostics/oracle_t96.npz`` and the three scripts that used it
(``freeze_oracle``, ``check_mvdlm_parity``, ``check_csrec_wiring``) were deleted
in ``fa8f6bd`` when ``DLM_LIB_2`` was retired. The ``.npz`` was 10.3 MB and was
never committed, so nothing of it survives but eight checksums quoted in
``DLMAX_MIGRATION_PLAN.md``. Those are against the pre-11-Aug
``compat_theta_T=True`` convention and the pre-5-Sep pre-drawn-``rvs`` random
stream, so they are not reproducible today even in principle.

The consequence is that **the only artefact ever to have proved ``mvdlm``
correct is gone**, at exactly the point where the code is to be moved into
DLMAX and then extended. This regenerates one under current conventions.

Three deliberate departures from the original
---------------------------------------------
1. **Self-contained, not driven from ``GDPRun.py``.** The oracle has to survive
   the move into DLMAX, and DLMAX cannot depend on BayesFR's GDP data. Every
   input here is built from a fixed seed, so the artefact is reproducible
   anywhere with no data on disk.

2. **A modest ``q``.** The original used ``q=304`` and produced 10.3 MB, which
   is why it was never committed -- and not committing it is the whole failure.
   The cross-series structure is broadcast over ``q``; nothing about the code
   paths changes with its size, so a smaller panel exercises the same
   arithmetic and yields an artefact small enough to live in git.

3. **Both seasonal periods.** ``GDPRun`` is quarterly (``M=4``, ``p=5``); the
   original checksums are ``(96,13,13)``, i.e. monthly (``M=12``, ``p=13``).
   Rather than guess which was meant, freeze both.

What it pins, and what it deliberately does not
-----------------------------------------------
The **DLM boundary alone**. ``chol_r`` is a fixed synthetic PSD matrix in factor
form plus a diagonal -- dense and correlated, so it genuinely exercises the
cross-series coupling -- rather than a factor-model draw. The gate therefore
does not depend on the Gibbs chain converging, which is the property that makes
it usable as a unit test.

``C0`` carries ``seasonal_prior``, which is **singular by construction** (the
dummy form is over-parameterised along ``1``). That is not incidental: it is the
path ``smoother._B_and_H``'s pseudo-inverse exists for, and the reason
``tests/test_smoother.py::test_rank_deficient_prior_uses_pseudo_inverse``
exists. An oracle that avoided it would miss the one numerically delicate part.

Freeze this BEFORE changing ``sqrtH`` to DLMAX's clipping convention. That
change is deliberate and expected to move ``sqrtH``; the point of freezing first
is that the delta is measured rather than absorbed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import block_diag

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mvdlm

T = 96          # periods, as the original
Q = 4           # series. Small on purpose: see departure (2) above.
SEED = 20260922
KEY = 0         # PRNG key for the backward draw
H = 12          # forecast horizon, as the original
DISC = [0.99, 0.995]        # GDPRun.py:164
N_FACTORS = 5               # for the synthetic chol_r


def build_model(M):
    """``(F, G, C0, delta_M, p)`` for a level + ``M``-state seasonal dummy.

    Mirrors ``GDPRun.py:303-348`` exactly: ``F`` is one in the level and the
    first seasonal slot and zero elsewhere; ``G`` is the level block-diagonal
    with a cyclic permutation; the discount matrix is the W&H congruence of the
    per-state rates.
    """
    I = jnp.eye(M)
    F = jnp.concatenate([jnp.ones((2, 1)), jnp.zeros((M - 1, 1))])
    G = block_diag(jnp.ones((1, 1)),
                   jnp.concatenate([I[:, [-1]], I[:, 0:M - 1]], 1))
    C0 = block_diag(jnp.ones((1, 1)), mvdlm.seasonal_prior(jnp.ones(M)))
    delta_M = mvdlm.component_discount_matrix([(1, 1), (M, M)], DISC)
    return F, G, jnp.asarray(C0), delta_M, M + 1


def build_data(rng, p, q):
    """Synthetic panel, priors, and a dense correlated ``chol_r``."""
    # A seasonal-plus-level signal per series, so the filter has structure to
    # track rather than noise alone.
    t = np.arange(T)[:, None]
    level = np.cumsum(rng.normal(0, 0.3, (T, q)), axis=0)
    seas = np.sin(2 * np.pi * t / 12 + rng.uniform(0, 2 * np.pi, (1, q)))
    Y = 100.0 + level + 2.0 * seas + rng.normal(0, 0.5, (T, q))

    m0 = jnp.asarray(np.vstack([Y[0], np.zeros((p - 1, q))]))   # (p, q)
    V0 = jnp.asarray(Y[:12].var(axis=0)[None, :])               # (1, q)

    # chol_r: factor form + diagonal. Dense and correlated by construction, so
    # the right-factor coupling in ffbs is genuinely exercised; a diagonal one
    # would let a transposed or mis-broadcast Sigma pass unnoticed.
    L = rng.normal(0, 1.0, (q, N_FACTORS))
    Sigma = L @ L.T + np.diag(rng.uniform(0.5, 1.5, q))
    chol_r = np.linalg.cholesky(Sigma)
    return jnp.asarray(Y), m0, V0, jnp.asarray(chol_r)


def freeze(M, out):
    rng = np.random.default_rng(SEED + M)
    F, G, C0, delta_M, p = build_model(M)
    Y, m0, V0, chol_r = build_data(rng, p, Q)

    params = mvdlm.dlm_params(Y=Y, m0=m0, C0=C0, F=F, G=G, V0=V0,
                              disc_mtx=delta_M)
    (a, m, n, scale), ts = mvdlm.dlm_back_sample(
        Y, random.PRNGKey(KEY), chol_r, params)
    fa, fR, f, var = mvdlm.dlm_forecast(params, H)

    arrays = {f"params_{k}": np.asarray(v) for k, v in params.items()}
    arrays.update({
        "in_Y": np.asarray(Y), "in_m0": np.asarray(m0), "in_C0": np.asarray(C0),
        "in_F": np.asarray(F), "in_G": np.asarray(G), "in_V0": np.asarray(V0),
        "in_delta_M": np.asarray(delta_M), "in_chol_r": np.asarray(chol_r),
        "bs_a": np.asarray(a), "bs_m": np.asarray(m),
        "bs_n": np.asarray(n), "bs_scale": np.asarray(scale),
        "bs_ts": np.asarray(ts),
        "fc_a": np.asarray(fa), "fc_R": np.asarray(fR),
        "fc_state_mean": np.asarray(f), "fc_state_var": np.asarray(var),
    })
    np.savez_compressed(out, **arrays)

    nonfinite = {k: int((~np.isfinite(v)).sum())
                 for k, v in arrays.items() if v.dtype.kind == "f"}
    bad = {k: c for k, c in nonfinite.items() if c}
    size = os.path.getsize(out)
    print(f"\n=== M={M}  p={p}  T={T}  q={Q}  ->  {out}  ({size/1024:.0f} KB) ===")
    print(f"  arrays: {len(arrays)}   non-finite: {bad if bad else 'none'}")
    for k in ("params_C", "params_A", "params_Q", "params_sqrtH", "params_B",
              "bs_ts", "fc_state_mean", "fc_state_var"):
        if k in arrays:
            v = arrays[k]
            print(f"  {k:16s} {str(v.shape):16s} sum={v.sum():+.10e}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    any_bad = {}
    for M in (4, 12):
        out = os.path.join(args.outdir, f"oracle_qw_M{M}.npz")
        any_bad.update(freeze(M, out))
    print("\nOK" if not any_bad else f"\nNON-FINITE ENTRIES: {any_bad}")

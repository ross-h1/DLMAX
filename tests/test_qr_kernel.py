"""Bit-exactness gate for the QR square-root filter step.

``dlm_uv_fwd_qr_step`` is the autodiff-stable sibling of ``dlm_uv_fwd_svd_step``:
same statistical model, but it carries the covariance root ``Z`` (``C = Z.T @ Z``)
and factorises with QR instead of ``svd_sqrt`` so gradients through the step stay
well-conditioned. These tests pin the contract that made it worth building — the
QR step must reproduce the SVD kernel's one-step predictive moments AND the
log-score gradient w.r.t. the discounts, across additive/multiplicative error and
multiplicative seasonality.

Self-contained: models are built from the DLMAX component algebra directly and
driven with synthetic positive data (multiplicative models need ``f > 0``).

- primal moments (f, q, prior-nu): QR vs SVD, ~1e-8.
- gradient dL/dθ over the discount vector + β: QR autodiff vs a central finite
  difference of the SVD kernel's own log score, ~1e-4 (FD-limited).
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.scipy.linalg import block_diag
from jax.scipy.special import gammaln

import DLMAX.dlm_core as dc


# --------------------------------------------------------------------------- #
# Model construction, from the DLMAX component algebra.
# --------------------------------------------------------------------------- #
def _fourier_full(period):
    """Full-rank Fourier seasonal, matching DLMAX's Fourier component layout."""
    Fb, Gb = [], []
    for j in range(1, period // 2 + 1):
        w = 2.0 * math.pi * j / period
        if period % 2 == 0 and j == period // 2:      # Nyquist: single state
            Fb.append(jnp.array([1.0]))
            Gb.append(jnp.array([[math.cos(w)]]))
        else:
            Fb.append(jnp.array([1.0, 0.0]))
            Gb.append(jnp.array([[math.cos(w), math.sin(w)],
                                 [-math.sin(w), math.cos(w)]]))
    return jnp.concatenate(Fb), block_diag(*Gb)


def _make_configs(period=6):
    """(name, F, G, mc, var_power, block_sizes) for level/trend/trend+seasonal
    × additive/multiplicative. Multiplicative ⇒ var_power 0 and, when seasonal,
    a multiplicative-seasonality mask (the only nonlinear observation)."""
    configs = []
    for damping in (0.0, 0.95):                       # level-only vs damped trend
        if damping == 0.0:
            Ft, Gt, tsz = jnp.array([1.0]), jnp.array([[1.0]]), 1
        else:
            Ft = jnp.array([1.0, 0.0])
            Gt = jnp.array([[1.0, damping], [0.0, damping]])
            tsz = 2
        for seasonal in (False, True):
            for variant in ("A", "M"):
                mult = variant == "M"
                var_power = 0.0 if mult else 1.0
                if seasonal:
                    Fs, Gs = _fourier_full(period)
                    F = jnp.concatenate([Ft, Fs])
                    G = block_diag(Gt, Gs)
                    ssz = period - 1
                    blocks = (tsz, ssz)
                else:
                    F, G, ssz, blocks = Ft, Gt, 0, (tsz,)
                mc = jnp.zeros(F.shape[0])
                if mult and ssz > 0:                  # multiplicative seasonality
                    mc = mc.at[tsz:tsz + ssz].set(1.0)
                tag = ("L" if damping == 0.0 else "T") + ("S" if seasonal else "")
                configs.append((f"{tag}-{variant}", F, G, mc, var_power, blocks))
    return configs


def _init_states(F, ys, warm=12, nu0=1.0):
    """Diffuse-ish prior: level/trend/seasonal states with a wide covariance, obs
    variance from the warmup window. Returns matched SVD (UC, SC) and QR (Z) roots
    of the SAME C0 (C0 = UC diag(SC**2) UC.T = Z.T Z)."""
    d = F.shape[0]
    yw = np.asarray(ys)[:warm]
    m0 = jnp.zeros(d).at[0].set(float(np.mean(yw)))
    C0 = jnp.eye(d) * float(max(np.var(yw), 1e-6)) * 10.0
    s0 = float(max(np.var(np.diff(yw)), 1e-6))
    ev, UC = jnp.linalg.eigh(C0)
    SC = jnp.sqrt(jnp.clip(ev, 1e-30, None))
    svd_init = {"m": m0, "UC": UC, "SC": SC,
                "s": jnp.asarray(s0), "nu": jnp.asarray(nu0)}
    qr_init = {"m": m0, "Z": SC[:, None] * UC.T,
               "s": jnp.asarray(s0), "nu": jnp.asarray(nu0)}
    return svd_init, qr_init


# --------------------------------------------------------------------------- #
# Scans returning one-step predictive (f, q, prior-nu) for each kernel.
# --------------------------------------------------------------------------- #
def _disc_matrix(theta_disc):
    return jnp.diag((1.0 - theta_disc) / theta_disc)


def _svd_run(theta_disc, beta, var_power, mc, F, G, init, ys):
    dm = _disc_matrix(theta_disc)

    def step(carry, y):
        st, md = dc.dlm_uv_fwd_svd_step(
            dm, beta, jnp.asarray(var_power), mc, carry, {"F": F, "G": G, "y": y})
        return st, (md["f"], md["q"], st["nu"])

    _, (f, q, nu_post) = lax.scan(step, init, ys)
    nu_prior = jnp.concatenate([jnp.atleast_1d(init["nu"]), nu_post[:-1]])
    return f, q, nu_prior


def _qr_run(theta_disc, beta, var_power, mc, F, G, init, ys):
    dm = _disc_matrix(theta_disc)

    def step(carry, y):
        st, md = dc.dlm_uv_fwd_qr_step(
            dm, beta, jnp.asarray(var_power), mc, carry, {"F": F, "G": G, "y": y})
        return st, (md["f"], md["q"], st["nu"])

    _, (f, q, nu_post) = lax.scan(step, init, ys)
    nu_prior = jnp.concatenate([jnp.atleast_1d(init["nu"]), nu_post[:-1]])
    return f, q, nu_prior


def _neg_log_pred(e, Q, n):
    return (-gammaln((n + 1.0) / 2.0) + gammaln(n / 2.0)
            + 0.5 * jnp.log(n * jnp.pi * Q)
            + (n + 1.0) / 2.0 * jnp.log1p(e**2 / (n * Q)))


def _qr_logscore(theta, var_power, mc, F, G, init, ys):
    """theta = [δ_state..., β]; QR sibling's Σ −log p (the autodiff target)."""
    k = F.shape[0]
    f, q, nu = _qr_run(theta[:k], theta[k], var_power, mc, F, G, init, ys)
    return jnp.sum(_neg_log_pred(ys - f, q, nu))


def _svd_logscore(theta, var_power, mc, F, G, init, ys):
    k = F.shape[0]
    f, q, nu = _svd_run(theta[:k], theta[k], var_power, mc, F, G, init, ys)
    return jnp.sum(_neg_log_pred(ys - f, q, nu))


def _synthetic_series(T=90, period=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    level = 100.0 + 0.4 * t
    season = 6.0 * np.sin(2 * np.pi * t / period)
    y = level + season + rng.normal(0.0, 2.0, T)
    return jnp.asarray(np.maximum(y, 1.0))            # strictly positive (mult models)


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
def test_qr_primal_matches_svd():
    """QR one-step predictive moments (f, q, prior-nu) == SVD kernel, all
    structures."""
    period = 6
    ys = _synthetic_series(period=period)
    for name, F, G, mc, var_power, blocks in _make_configs(period):
        svd_init, qr_init = _init_states(F, ys)
        k = F.shape[0]
        theta_disc = jnp.full(k, 0.95)
        beta = jnp.asarray(0.95)
        f_s, q_s, n_s = _svd_run(theta_disc, beta, var_power, mc, F, G, svd_init, ys)
        f_q, q_q, n_q = _qr_run(theta_disc, beta, var_power, mc, F, G, qr_init, ys)
        assert np.allclose(f_q, f_s, atol=1e-6, rtol=1e-8), f"{name} f"
        assert np.allclose(q_q, q_s, rtol=1e-8, atol=1e-9), f"{name} q"
        assert np.allclose(n_q, n_s, atol=1e-9), f"{name} nu"


def test_qr_logscore_gradient_matches_svd_fd():
    """QR autodiff dL/dθ (θ = discount vector + β) == central finite difference of
    the SVD kernel's own log score. This is the property SVD autodiff can't give."""
    period = 6
    ys = _synthetic_series(period=period)
    eps = 1e-6
    grad_fn = jax.grad(_qr_logscore)
    for name, F, G, mc, var_power, blocks in _make_configs(period):
        svd_init, qr_init = _init_states(F, ys)
        k = F.shape[0]
        theta = jnp.concatenate([jnp.full(k, 0.95), jnp.asarray([0.95])])
        g_qr = np.asarray(grad_fn(theta, var_power, mc, F, G, qr_init, ys))
        # reference: FD of the SVD kernel log score, per component
        g_fd = np.zeros(k + 1)
        for j in range(k + 1):
            lp = float(_svd_logscore(theta.at[j].add(eps), var_power, mc, F, G, svd_init, ys))
            lm = float(_svd_logscore(theta.at[j].add(-eps), var_power, mc, F, G, svd_init, ys))
            g_fd[j] = (lp - lm) / (2 * eps)
        denom = np.maximum(np.abs(g_fd), 1e-8)
        rel = np.max(np.abs(g_qr - g_fd) / denom)
        assert rel < 1e-4, f"{name} grad rel={rel:.2e}"


def test_qr_selfconsistent_grad_fwd_rev():
    """Forward- and reverse-mode autodiff through the QR step agree (the SVD step
    can't be trusted here — repeated singular values make its tangent blow up)."""
    period = 6
    ys = _synthetic_series(period=period)
    name, F, G, mc, var_power, blocks = _make_configs(period)[4]   # a seasonal model
    _, qr_init = _init_states(F, ys)
    k = F.shape[0]
    theta = jnp.concatenate([jnp.full(k, 0.95), jnp.asarray([0.95])])
    g_rev = np.asarray(jax.grad(_qr_logscore)(theta, var_power, mc, F, G, qr_init, ys))
    g_fwd = np.asarray(jax.jacfwd(_qr_logscore)(theta, var_power, mc, F, G, qr_init, ys))
    assert np.allclose(g_rev, g_fwd, rtol=1e-9, atol=1e-12)
    assert np.all(np.isfinite(g_rev))


def test_qr_missing_obs_projects_to_prior():
    """A NaN observation makes the posterior equal the PRIOR: the mean projects
    (m→a) and the covariance advances to R = G C G'/δ. The variance state is
    untouched. Finite steps stay bit-exact.

    Previously this asserted the covariance was *frozen* at C_{t-1}, which
    dropped the step's discount forgetting and, under non-identity G, left the
    mean rotated while the covariance was not. See tests/test_ignore_obs_prior.py.
    """
    period = 6
    name, F, G, mc, var_power, blocks = _make_configs(period)[2]   # LT-A (trend)
    ys = _synthetic_series(period=period)
    _, qr_init = _init_states(F, ys)
    disc = jnp.full(F.shape[0], 0.95)
    dm = _disc_matrix(disc)
    beta = jnp.asarray(0.95)
    st0, _ = dc.dlm_uv_fwd_qr_step(dm, beta, jnp.asarray(var_power), mc, qr_init,
                                   {"F": F, "G": G, "y": ys[12]})
    st_nan, md_nan = dc.dlm_uv_fwd_qr_step(
        dm, beta, jnp.asarray(var_power), mc, st0,
        {"F": F, "G": G, "y": jnp.asarray(np.nan)})

    a = st0["m"] @ G.T
    C0 = np.asarray(st0["Z"].T @ st0["Z"])
    D = np.diag(1.0 / np.sqrt(np.asarray(disc)))
    R = D @ (np.asarray(G) @ C0 @ np.asarray(G).T) @ D          # congruence discount

    assert np.allclose(np.asarray(st_nan["m"]), np.asarray(a))          # mean projected
    assert np.allclose(np.asarray(st_nan["Z"].T @ st_nan["Z"]), R)      # cov -> prior
    assert not np.allclose(np.asarray(st_nan["Z"].T @ st_nan["Z"]), C0)  # not frozen
    assert float(st_nan["s"]) == float(st0["s"])
    assert float(st_nan["nu"]) == float(st0["nu"])

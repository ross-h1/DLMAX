"""Discount and prior helpers for the matrix-normal (Quintana/West) DLM.

The first piece of that model to land in DLMAX. Three pure functions, no state
and no IO:

:func:`congruence`
    Per-state discount **vector** -> the West & Harrison congruence **matrix**,
    ``D[i, j] = sqrt(d_i d_j)``.
:func:`component_discount_matrix`
    The same for a model described as blocks: one rate per block, expanded to
    one rate per state, then made congruent.
:func:`seasonal_prior`
    The zero-sum prior covariance for a seasonal-dummy block, from its diagonal.

Why the congruence matrix exists
--------------------------------
The matrix-normal filter discounts elementwise, ``R = P / D`` with
``P = G C G'``. The univariate engine discounts by congruence,
``R = D_v P D_v`` with ``D_v = diag(1/sqrt(delta))``. Those are the same
operation precisely when ``D[i, j] = sqrt(d_i d_j)`` — which is what this
returns, and why it is neither ``outer(d, d)`` nor ``diag(d)``. Both of those
agree on the diagonal and differ off it, so the two halves of the library would
compute different things while each looked correct.

Taking the **vector** as canonical is what lets DLMAX's ``Component`` API drive
this model: components emit per-state discount vectors, and the congruence is
formed here rather than being a second thing components must know how to emit.
"""
import jax.numpy as jnp
from jax import random, vmap
from jax.lax import scan

from DLMAX.wishart import wishart_step, wishart_scan


def congruence(disc_rates):
    """Per-state discount rates ``(p,)`` -> congruence matrix ``(p, p)``.

    Parameters
    ----------
    disc_rates : array, shape ``(p,)``
        One discount in ``(0, 1]`` per state.

    Returns
    -------
    array, shape ``(p, p)``
        ``D[i, j] = sqrt(d_i d_j)``, symmetric, with ``d`` on the diagonal.
        Always float64: the filter runs in x64, and a float32 discount matrix
        would silently halve the precision of every covariance built from it.
    """
    d = jnp.asarray(disc_rates, dtype=jnp.float64)
    # sqrt of the product, NOT the product of the square roots. The two are
    # mathematically identical and differ by an ulp; this form is the one the
    # implementation being ported uses, and delta_M feeds the covariance path,
    # so the oracle gate for that move is exact rather than to-tolerance. The
    # only argument for the other form is underflow in the product, which
    # cannot arise for discounts in (0, 1].
    return jnp.sqrt(d[:, None] * d[None, :])


def _rates_from_shapes(shapes, delta):
    """``[(n, n), ...]`` + one rate per block -> the per-state rate vector.

    Only the row count of each shape is read: a block contributes that many
    states, all sharing its rate.
    """
    if len(shapes) != len(delta):
        raise ValueError(
            f"{len(shapes)} shapes against {len(delta)} rates: one rate per "
            "block is required. Silently zipping to the shorter of the two "
            "would drop states or rates without saying so.")
    return jnp.concatenate(
        [jnp.full(int(s[0]), float(r), dtype=jnp.float64)
         for s, r in zip(shapes, delta)])


def component_discount_matrix(shapes, delta):
    """Block description -> congruence matrix.

    Parameters
    ----------
    shapes : list of (rows, cols)
        One entry per component. A ``(4, 4)`` block means four states that
        share that component's rate.
    delta : list of float
        One discount per component, in the same order.

    Returns
    -------
    array, shape ``(p, p)`` where ``p`` is the total row count.

    Examples
    --------
    A level plus a 4-state seasonal dummy, at 0.99 and 0.995, gives a 5x5
    matrix whose diagonal is ``[0.99, 0.995, 0.995, 0.995, 0.995]`` and whose
    level/seasonal cross term is ``sqrt(0.99 * 0.995) = 0.9924968...``.
    """
    return congruence(_rates_from_shapes(shapes, delta))


def seasonal_prior(c0):
    """Zero-sum prior covariance for a seasonal-dummy block, given its diagonal.

    With ``period`` dummy states summing to a level, ``1'theta`` is not
    identified, so the prior is made to carry no mass in that direction. From
    ``C = diag(c0)``, with ``l = 1``, ``U = l'Cl`` and ``A = Cl/U``, the result
    is ``C - outer(A, A) U``. West & Harrison sec 8.6.

    Parameters
    ----------
    c0 : array, shape ``(period,)``
        The intended diagonal.

    Returns
    -------
    array, shape ``(period, period)``
        Symmetric, positive-semi-definite, and **singular** — ``C @ 1`` is
        exactly zero and the rank is ``period - 1``.

    Notes
    -----
    The singularity is the point of the function, not a defect to regularise.
    It is the case :func:`DLMAX.smoother._B_and_H`'s pseudo-inverse exists for;
    adding a ridge to make it invertible would remove the very path that code
    is written to handle, and would do so without any test noticing.
    """
    c = jnp.asarray(c0, dtype=jnp.float64)
    # For diagonal C, Cl is just c and U = l'Cl is sum(c), so the projection
    # reduces to diag(c) - outer(c, c)/sum(c). Written via A and U to stay
    # recognisable as the W&H expression.
    U = jnp.sum(c)
    A = c / U
    return jnp.diag(c) - jnp.outer(A, A) * U


class CovariancePath:
    """The scale-free filtered and smoothed covariance path of a QW model.

    Shared by every series and by every period with the same ``F``, ``G``,
    ``C0`` and discount. Every attribute is **data-independent**:

    ``C`` ``(T, p, p)``, ``R`` ``(T, p, p)``, ``Q`` ``(T, 1, 1)``,
    ``A`` ``(T, p, 1)``, ``B`` ``(T-1, p, p)``, ``H`` ``(T, p, p)``,
    ``sqrtH`` ``(T, p, p)``, ``S`` ``(T, p, p)``.

    A plain container rather than a dataclass or a pytree: it is built once,
    outside any traced region, and held. The filter's own per-step carry is
    what crosses ``jit`` boundaries.
    """

    def __init__(self, C, R, Q, A, B, H, sqrtH, S, delta_M, psd_clip):
        self.C, self.R, self.Q, self.A = C, R, Q, A
        self.B, self.H, self.sqrtH, self.S = B, H, sqrtH, S
        self.delta_M = delta_M
        self.psd_clip = psd_clip

    @property
    def T(self):
        return self.C.shape[0]


def _build_covariance_path(T, F, G, C0, delta_M, X=None):
    """Run the QR filter ONCE, on a single dummy series, and read the path off.

    The Kronecker structure ``W_t (x) Sigma`` makes ``C``, ``R``, ``Q`` and
    ``A`` common to every series and independent of the data, so one run gives
    the path for all of them. At BayesFR's shape that is one 13x13 recursion
    instead of 304 copies.

    The data really is irrelevant, so zeros are filtered: driving it with
    anything else would invite the reader to wonder whether the choice mattered.
    Note the signature takes no observations at all — that is the property, made
    structural.

    Not a re-derivation of the recursion. The scale-free covariance recursion IS
    ``dlm_uv_fwd_qr_step``, reached through a one-series :class:`uv_dlm` with a
    static discount, so this inherits the library's QR roots and Joseph-form
    posterior rather than growing a second implementation free to drift.

    Parameters
    ----------
    X : array ``(T, k)``, optional
        A time-varying design, **common to every series**. The effective design
        at step ``t`` is ``[F_structural ; X[t]]``, where ``F_structural`` is
        the leading ``p - k`` entries of ``F``.

        Common is not a convenience: it is what keeps the ``(x) Sigma``
        collapse valid. A VAR satisfies it automatically -- every equation has
        the same right-hand side, which is the Zellner SUR condition -- but a
        per-series design would break the collapse and put you back to ``q``
        separate recursions.

        What survives with a time-varying design: the path is still shared
        across series, and still independent of ``Y``. What does not: reuse
        across rolling ORIGINS. The path becomes a function of the regressor
        window rather than a constant of the model, so a caller holding one
        must key it on that window.
    """
    from DLMAX.dlm_core import uv_dlm
    from DLMAX.smoother import rts_smooth, _sym_sqrt
    from DLMAX.ffs import devices

    p = G.shape[0]
    disc = jnp.diag(delta_M)        # per-state rate; the congruence is off-diag
    Fv = jnp.asarray(F).reshape(-1)

    if X is None:
        k, Fv_struct = 0, Fv
        reg_kw = {"n_regressors": 0}
    else:
        X = jnp.asarray(X, dtype=jnp.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be (T, k); got shape {X.shape}.")
        if X.shape[0] != T:
            raise ValueError(
                f"X has {X.shape[0]} rows against T={T}. The design must cover "
                "exactly the steps being filtered; a shorter or longer one "
                "mis-indexes silently.")
        k = int(X.shape[1])
        if k >= p:
            raise ValueError(
                f"X is {k} wide against a {p}-state model: the design tail "
                "cannot be the whole state, there must be a structural part.")
        # uv_dlm takes the STRUCTURAL design and fills the last n_regressors
        # slots per step from `regressors`, which is exactly the shape wanted
        # here -- so drive that rather than inventing a second mechanism.
        Fv_struct = Fv[:p - k]
        reg_kw = dict(
            n_regressors=k,
            regression_G=jnp.eye(k),  # the tail's own transition
            # Three arguments that must be given together, none of which names
            # its cause when omitted: without regressors_0 an internal concat
            # comes out the wrong length, and mult_comps otherwise defaults to
            # the STRUCTURAL width rather than p.
            regressors_0=jnp.zeros(k),
            mult_comps=jnp.zeros(p),
        )

    model = uv_dlm(
        series_ids=[0], F=Fv_struct,
        G=G if k == 0 else jnp.asarray(G)[:p - k, :p - k],
        m0=jnp.zeros((1, p)), C0=jnp.asarray(C0)[jnp.newaxis],
        V0=jnp.ones(1), nu0=jnp.ones(1),
        # adapt=None makes the discount static, which leaves disc_rates_norm
        # inert and disc_rates_damped as the applied rate. This is the one place
        # that slot is used for something other than a damped trend's phi**2,
        # and it is safe only because the adaptive envelope is switched off.
        disc_rates_norm=jnp.ones(p),
        disc_rates_damped=disc,
        variance_disc=1.0, variance_power=1.0,
        device=devices.host_device, adapt=None, **reg_kw,
    )
    if k == 0:
        for _ in range(T):
            model.fwd_filter(jnp.zeros(1), trajectory=True)
    else:
        for t in range(T):
            model.fwd_filter(jnp.zeros(1), regressors=X[t][jnp.newaxis, :],
                             trajectory=True)

    traj = model.trajectory
    # C* = Z'Z / s. With var_power == 1 the stored s is the un-normalised scale,
    # so this is exact; a variance law would make C depend on the fitted mean
    # and the whole collapse would fail.
    Z = traj["Z"]                                       # (T, 1, p, p)
    s = jnp.asarray(traj["s"]).reshape(T, 1, 1, 1)
    C = ((Z.swapaxes(-1, -2) @ Z) / s)[:, 0]            # (T, p, p), scale-free

    sm = rts_smooth({"m": traj["m"], "Z": Z, "s": traj["s"]}, G, disc)

    # rts_smooth returns H for t = 0..T-2 (the step t -> t+1). The terminal
    # entry is C_T, because Theta_T is drawn from the filtered posterior.
    H_full = jnp.concatenate([sm["H"][:, 0], C[-1][jnp.newaxis]], 0)

    # The library's root, not a second convention. _sym_sqrt uses eigh and
    # CLIPS a negative eigenvalue, reporting how much as psd_clip; the code this
    # was ported from used svd, which takes the root of the SINGULAR values and
    # so REFLECTS an indefinite H instead, silently.
    #
    # The two agree wherever H is positive semi-definite, which on everything
    # measured so far it is -- minimum eigenvalue +2.7e-17 over 1728 matrices
    # across both model sizes, i.e. zero negative eigenvalues. So this is a
    # change of root CHOICE (eigh ascending against svd descending), not of
    # arithmetic: sqrtH's entries move, H does not.
    #
    # It still changes every backward draw. Any M with M M' = H is a valid root
    # and they are distributionally equivalent, but for a GIVEN random variate
    # they give different realisations. That is why it is a separate commit
    # from the port, with the oracle frozen beforehand.
    sqrtH, psd_clip = _sym_sqrt(H_full)

    # R_t is the prior at t, built from C_{t-1}; R_0 uses C0. The division by
    # delta_M is ELEMENTWISE -- delta_M is the congruence matrix, and P/Delta is
    # the same operation as the kernel's D P D, not P @ inv(Delta).
    C_prev = jnp.concatenate([jnp.asarray(C0)[jnp.newaxis], C[:-1]], 0)
    # R carries NO design -- and neither does rts_smooth, which is why they can
    # be built from the same C without either knowing about F.
    R = (1.0 / delta_M)[jnp.newaxis] * (G @ C_prev @ G.T)
    if k == 0:
        Fc = Fv.reshape(p, 1)
        Q = 1.0 + Fc.T @ R @ Fc                         # (T, 1, 1); the 1 is V
        A = (R @ Fc) / Q                                # (T, p, 1)
    else:
        # The effective design per step, (T, p). Only Q and A see it.
        Ft = jnp.concatenate(
            [jnp.broadcast_to(Fv_struct, (T, p - k)), X], axis=1)
        Fc = Ft[:, :, jnp.newaxis]                      # (T, p, 1)
        # Written as the same matmul the constant branch uses rather than an
        # einsum. The two are mathematically identical but contract in a
        # different order, which put the constant-design reduction an ulp out
        # on 2 of 24 entries -- and that reduction is the gate for this whole
        # change, so it is worth being bitwise rather than nearly.
        Q = 1.0 + jnp.swapaxes(Fc, -1, -2) @ R @ Fc     # (T, 1, 1)
        A = (R @ Fc) / Q                                # (T, p, 1)

    return CovariancePath(
        C=C, R=R, Q=Q, A=A,
        B=sm["B"][:, 0], H=H_full, sqrtH=sqrtH,
        S=sm["S"][:, 0], delta_M=delta_M,
        # The larger of the two clips: the smoother's own, and ours on H_full.
        psd_clip=float(jnp.maximum(sm["psd_clip"], psd_clip)),
    )


class mv_dlm:
    """Quintana/West matrix-normal DLM with vector observations.

    ::

        y_t'    = F' Theta_t + nu_t' ,   nu_t ~ N(0, Sigma)        Sigma is q x q
        Theta_t = G Theta_{t-1} + Om_t,  Om_t ~ N(0, W_t (x) Sigma)
        Theta_0 ~ N(m_0, C_0 (x) Sigma)

    The ``(x) Sigma`` structure is what makes this cheap: ``C_t``, ``R_t``,
    ``Q_t`` and ``A_t`` are common across all ``q`` series and independent of
    the data, so they are built once (:class:`CovariancePath`) and only the mean
    ``m_t`` is per-series. The series couple *only* in the backward draw,
    through ``Sigma^(1/2)``.

    Parameters
    ----------
    data : array ``(T, q)``
    F : array ``(p, 1)``
    G : array ``(p, p)``
    m0 : array ``(p, q)``      initial state mean, per series
    C0 : array ``(p, p)``      initial state covariance, scale-free (``(x) Sigma``)
    V0 : array ``(1, q)``      prior observation variances, per series
    disc_mtx : array ``(p, p)``
        Component-discount matrix; use :func:`component_discount_matrix`.
    cov_path : CovariancePath, optional
        Reuse a path built for an earlier period with identical ``F``, ``G``,
        ``C0`` and discount. Since the path does not depend on the data, this is
        free across a rolling origin and is the single largest saving available.

    Notes
    -----
    **Everything here is scale-free**: ``V == 1``, so predictive variances are
    in units of ``Sigma``. ``Sigma`` is not held by the model -- it is supplied
    at :meth:`backward_sample` time as ``chol(Sigma)``. Multiply a returned
    variance by ``diag(Sigma)`` for the per-series quantity.

    The API follows the rest of the library: :meth:`fwd_filter` advances one
    observation and returns the one-step-ahead predictive, :meth:`scan_filter`
    advances many, :meth:`forecast` projects from the held state. The engines
    (``uv_dlm``, ``multi_model_dlm``) and the FFS blocks use the same three
    names for the same three things.
    """

    def __init__(self, data, F, G, m0, C0, V0, disc_mtx, cov_path=None,
                 mult_comps=None, delta_sigma=None, n0=1.0, S0=None):
        if delta_sigma is not None and not (0.0 < float(delta_sigma) <= 1.0):
            # Checked here rather than T steps later inside a scan, where it
            # would surface as a NaN with nothing to attribute it to.
            raise ValueError(
                f"delta_sigma must lie in (0, 1]; got {delta_sigma}")
        Y = jnp.asarray(data)
        if bool(jnp.any(~jnp.isfinite(Y))):
            n_bad = int(jnp.sum(~jnp.isfinite(Y)))
            raise ValueError(
                f"data has {n_bad} non-finite entries. A gap makes the filter "
                "skip that step, desynchronising that series' covariance path "
                "from the shared one -- which invalidates the single-run "
                "construction this model is built on. Handle missingness with a "
                "data-augmentation Gibbs block, or group series by missingness "
                "pattern and build one path per pattern.")
        if mult_comps is not None and bool(jnp.any(jnp.asarray(mult_comps))):
            # Required by the migration plan and never written until now.
            raise ValueError(
                "mult_comps makes the observation equation bilinear in the "
                "state, so the forward pass is an EKF and the backward "
                "recursion an extended -- not conjugate -- smoother. It would "
                "run cleanly and return plausible approximate numbers with "
                "nothing raising, which is exactly why this refuses instead.")

        self.data, self.F, self.G = Y, jnp.asarray(F), jnp.asarray(G)
        self.m0, self.V0 = jnp.asarray(m0), jnp.atleast_2d(V0)
        self.disc_mtx = jnp.asarray(disc_mtx)
        self.T, self.q = Y.shape
        self.p = self.G.shape[0]

        if cov_path is None:
            cov_path = _build_covariance_path(
                self.T, self.F, self.G, jnp.asarray(C0), self.disc_mtx)
        elif cov_path.T < self.T:
            raise ValueError(
                f"cov_path covers {cov_path.T} steps, data has {self.T}.")
        self.cov = cov_path
        self._filtered = None
        self._t = 0                      # observations absorbed by fwd_filter
        self._carry = (self.m0, 1, self.V0)

        # -- Sigma learning, opt-in -------------------------------------
        # OFF by default, and deliberately so. The full path is (T, q, q);
        # at BayesFR's q = 304 over 131 periods that is ~97 MB allocated on
        # every Gibbs sweep to hold something that project does not read --
        # it takes Sigma from its factor model. With delta_sigma None the
        # model runs exactly the diagonal recursion it always ran, and
        # params() gains no keys, so the pytree csrec jits is unchanged.
        self.delta_sigma = delta_sigma
        self.n0 = float(n0)
        self.S0 = (jnp.diag(self.V0.reshape(self.q)) if S0 is None
                   else jnp.asarray(S0))
        self.sigma_scale = None          # (T, q, q), filled by scan_filter
        self.sigma_dof = None            # (T,)
        # the streaming face's running state
        self.sigma_scale_now = self.S0 if delta_sigma is not None else None
        self.sigma_dof_now = self.n0 if delta_sigma is not None else None

    # -- streaming face ----------------------------------------------------

    def fwd_filter(self, yt):
        """Advance one observation; return the one-step-ahead predictive.

        The STEP face, with the same contract as ``uv_dlm.fwd_filter`` and
        ``multi_model_dlm.fwd_filter``: the predictive belongs to the carry as
        it stood BEFORE ``yt``, which is what ``yt`` is scored against.

        Parameters
        ----------
        yt : array ``(q,)``

        Returns
        -------
        ForecastBundle
            ``loc`` ``(q,)`` and ``var`` ``(q,)``, the per-series predictive
            variance ``Q_t * scale_{t-1}``.

        Notes
        -----
        ``Q_t`` is common to every series -- that is the Kronecker structure,
        not an approximation -- and ``scale`` is the running estimate of
        ``diag(Sigma)``, so the product is the per-series quantity. This mirrors
        ``uv_dlm.fwd_filter``, whose variance is likewise ``s * (1 + F'RF)``
        with the LEARNED observation scale rather than a scale-free factor.

        ``scale`` is taken from the carry as it stood BEFORE ``yt``, since a
        one-step-ahead predictive may only use ``D_{t-1}``. It is the DIAGONAL
        of ``S_t / n_t`` at ``delta_Sigma = 1``. The scale-free factor on its
        own is ``self.cov.Q``, if that is what you want.

        **Variance learning enters here.** With ``delta_sigma`` set, ``scale``
        above is the diagonal of the discount-Wishart estimate rather than the
        undiscounted running average -- choosing a variance discount and then
        reporting intervals that ignore it would be half a feature. At
        ``delta_sigma = 1`` the two are bitwise identical, and the parameter
        defaults to ``None``, so no existing caller moves. The learned
        ``Sigma`` is also surfaced whole as :attr:`sigma_scale`, for the draw in
        :meth:`forecast_sample` and as ``right_factor`` for ``ffbs``.
        """
        from DLMAX.dlm_core import ForecastBundle
        if self._t >= self.cov.T:
            raise ValueError(
                f"fwd_filter past the covariance path: {self._t} steps taken, "
                f"path covers {self.cov.T}. The path is built for a fixed T.")
        y = jnp.asarray(yt).reshape(self.q)
        m, n, scale = self._carry
        A, Q = self.cov.A[self._t], self.cov.Q[self._t]

        a = self.G @ m                                   # (p, q)
        f = (self.F.T @ a).reshape(self.q)               # (q,)
        error = y.reshape(1, self.q) - self.F.T @ a      # (1, q)
        m_new = a + A @ error
        scale_new = (error ** 2 / Q + scale * n) / (n + 1)

        # The predictive belongs to D_{t-1}, so the PRIOR scale is used -- the
        # one in the carry above, not scale_new. With learning on that is the
        # prior discount-Wishart scale, which self.sigma_scale_now still holds:
        # it is advanced below, after this line.
        prior = (scale if self.delta_sigma is None
                 else jnp.diagonal(self.sigma_scale_now).reshape(1, self.q))
        var = (Q.reshape(()) * prior).reshape(self.q)
        self._carry = (m_new, n + 1, scale_new)
        if self.delta_sigma is not None:
            self.sigma_scale_now, self.sigma_dof_now = wishart_step(
                self.sigma_scale_now, self.sigma_dof_now,
                error.reshape(self.q), Q.reshape(()), self.delta_sigma)
        self._t += 1
        return ForecastBundle(f, var)

    def scan_filter(self, ys=None, m0=None, V0=None):
        """Advance over the whole series in one compiled scan.

        The SCAN face. Returns ``(a, m, n, scale)`` stacked over time -- the
        per-series mean recursion against the shared ``A``/``Q``. The covariance
        path is NOT recomputed: it does not depend on the data.
        """
        Y = self.data if ys is None else jnp.asarray(ys)
        m0 = self.m0 if m0 is None else jnp.asarray(m0)
        V0 = self.V0 if V0 is None else jnp.atleast_2d(V0)
        Ft = self.F.T

        def step(state, xs):
            m, n, scale = state
            y, A, Q = xs
            a = self.G @ m
            error = y - Ft @ a
            m = a + A @ error
            scale = (error ** 2 / Q + scale * n) / (n + 1)
            return (m, n + 1, scale), (a, m, n + 1, scale, error)

        T = Y.shape[0]
        _, out = scan(step, (m0, 1, V0),
                      (Y, self.cov.A[:T], self.cov.Q[:T]))
        self._filtered = out[:4]
        if self.delta_sigma is not None:
            # A second pass over the errors the scan already produced, rather
            # than carrying a (q, q) accumulator through it. The cost is one
            # (T, q) array; the gain is that the recursion stays in one tested
            # place instead of being inlined here as well.
            self.sigma_scale, self.sigma_dof = wishart_scan(
                self.S0, self.n0, out[4].reshape(T, self.q),
                self.cov.Q[:T].reshape(T), self.delta_sigma)
            # `scale` has ONE meaning -- the model's estimate of diag(Sigma) --
            # and delta_sigma chooses how it is estimated. Returning the
            # undiscounted diagonal alongside a discounted sigma_scale would
            # hand the caller two different answers to the same question.
            # At delta_sigma = 1 this substitution is bitwise a no-op.
            a_, m_, n_, _ = self._filtered
            self._filtered = (a_, m_, n_,
                              jnp.diagonal(self.sigma_scale, axis1=1,
                                           axis2=2).reshape(T, 1, self.q))
        return self._filtered

    # -- retrospective -----------------------------------------------------

    def backward_sample(self, sigma_tril, key=None, ys=None, m0=None, V0=None,
                        reuse_filtered=False):
        """One FFBS draw of the state path, coupled across series by ``Sigma``.

        ``sigma_tril`` is ``chol(Sigma)`` ``(q, q)`` -- the only place ``Sigma``
        enters. It reaches :func:`DLMAX.smoother.ffbs` as ``right_factor``, so
        the draw is matrix-normal with ``sqrtH`` on the left (state, scale-free)
        and ``Sigma^(1/2)`` on the right (series).

        The covariance path is passed with a singleton series axis, which
        ``ffbs`` broadcasts: ``C*`` is common to every series, so a per-series
        copy is pure redundancy.
        """
        from DLMAX.smoother import ffbs
        if key is None:
            raise ValueError("backward_sample needs a PRNG key")
        if reuse_filtered and self._filtered is not None:
            a, m, n, scale = self._filtered
        else:
            a, m, n, scale = self.scan_filter(ys=ys, m0=m0, V0=V0)

        states = ffbs(
            key, {"m": m.swapaxes(1, 2)}, self.G, jnp.diag(self.disc_mtx),
            n_draws=1, right_factor=jnp.asarray(sigma_tril),
            smoothed={"S": self.cov.S[:, jnp.newaxis],
                      "B": self.cov.B[:, jnp.newaxis],
                      "sqrtH": self.cov.sqrtH[:-1][:, jnp.newaxis]},
        )[0]                                              # (T, q, p)
        ts = (self.F.T @ states.swapaxes(1, 2)).squeeze(1)
        return (a, m, n, scale), ts

    def smoothed_means(self):
        """``s_t = m_t + B_t (s_{t+1} - a_{t+1})``, the retrospective means."""
        if self._filtered is None:
            self.scan_filter()
        _, m, _, _ = self._filtered
        a_next = self.G @ m[:-1]

        def back(s_next, xs):
            m_t, B_t, a_t1 = xs
            return (m_t + B_t @ (s_next - a_t1),) * 2

        _, s_hist = scan(back, m[-1], (m[:-1], self.cov.B, a_next),
                         reverse=True)
        return jnp.concatenate([s_hist, m[-1][jnp.newaxis]], 0)

    # -- forecasting -------------------------------------------------------

    def _forecast_arrays(self, h, state=None):
        """``(a, R, f, var)`` over the horizon. ``var`` is ``F' R_h F``, the
        STATE component only -- see :meth:`forecast` on the two conventions."""
        if state is None:
            if self._filtered is None:
                self.scan_filter()
            state = (self._filtered[1][-1], self.cov.C[-1])
        m, C = state
        Ft = self.F.T
        # W is frozen at the origin (W&H p.199): the h-step projection uses one
        # fixed evolution covariance rather than re-deriving it each step.
        W = ((1.0 - self.disc_mtx) / self.disc_mtx) * (self.G @ C @ self.G.T)

        def step(carry, _):
            m, C = carry
            a = self.G @ m
            R = self.G @ C @ self.G.T + W
            return (a, R), (a, R, Ft @ a, Ft @ R @ self.F)

        _, out = scan(step, (m, C), jnp.ones(h))
        return out

    def forecast(self, h, state=None):
        """``h``-step predictive from the held state.

        Returns
        -------
        ForecastBundle
            ``loc`` ``(h, q)`` and ``var`` ``(h, q)``, **scale-free**: the
            predictive is ``N(loc, var_h Sigma)``.

        Notes
        -----
        ``var`` here INCLUDES the observation term, ``1 + F' R_h F``, so it is
        the same quantity :meth:`fwd_filter` returns at ``h = 1``. The legacy
        :func:`dlm_forecast` shim returns ``F' R_h F`` WITHOUT it, because its
        callers add the observation half themselves. The two conventions are
        deliberate and the difference is exactly ``1``; this one is the
        predictive, that one is the state component.
        """
        from DLMAX.dlm_core import ForecastBundle
        _a, _R, f, var = self._forecast_arrays(h, state)
        loc = jnp.asarray(f).reshape(h, self.q)
        v = (1.0 + jnp.asarray(var).reshape(h, 1))
        return ForecastBundle(loc, jnp.broadcast_to(v, (h, self.q)))

    def forecast_sample(self, h, sigma_tril, key, n_lags, n_draws=1,
                        seed_lags=None, state=None):
        """Simulate the VAR forward from the held state. ``(n_draws, h, q)``.

        The counterpart of :meth:`backward_sample`, which samples the past.

        Why simulation, and not iterated expectations
        ---------------------------------------------
        **The Kronecker structure does not survive multi-step forecasting.**
        For a VAR(1) with ``Theta`` known,

            Var(y_{t+2} | y_t, Theta) = Sigma + Theta' Sigma Theta

        which is not a scalar multiple of ``Sigma`` unless
        ``Theta' Sigma Theta`` happens to be proportional to it. So the
        predictive is ``Q_1 Sigma`` at one step and is **not** ``(x) Sigma``
        from two steps on.

        ``uv_dlm``'s ``iterated_obs_forecast`` returns a SCALAR ``q_h`` per
        series, and can, because its series are independent and each carries
        its own lags. The VAR analogue cannot: the whole point is that the
        series are coupled. Carrying the exact ``q x q`` predictive analytically
        means the coefficient-uncertainty term generalises to a matrix
        quadratic; simulating carries it for free, by drawing ``Theta``.

        A version that kept the scalar ``q_h (x) Sigma`` form would run
        cleanly, return entirely plausible numbers, and understate forecast
        uncertainty.

        The recursion
        -------------
        ::

            Sigma, Theta_T ~ posterior           Theta_T ~ N(m_T, C_T (x) Sigma)
            for k = 1..h:
                Theta_{t+k} ~ N(G Theta_{t+k-1}, W (x) Sigma)
                y_{t+k}     ~ N(Theta'_{t+k} F_{t+k}, Sigma)
                y_{t+k} shifts into F_{t+k+1}

        No backward pass is involved: FFBS produces the HISTORICAL state path,
        which forecasting never touches. Drawing ``Theta_T`` rather than fixing
        it at ``m_T`` is what makes the forecast origin resampled across draws,
        which the migration plan noted it never was.

        Two conventions, stated rather than implied:

        * ``W`` is frozen at the origin, ``((1-Delta)/Delta) (G C_T G')``. Once
          ``Theta`` is drawn rather than tracked there is no ``C_{t+k}`` to
          derive it from; this is the W&H p.199 convention and what
          :meth:`forecast` already does.
        * The observation variance is ``Sigma`` exactly. The ``Q_t`` inflation
          exists to carry uncertainty in ``Theta``; simulation carries it by
          drawing, so applying both would double-count.

        Parameters
        ----------
        h : int
        sigma_tril : array ``(q, q)``
            ``chol(Sigma)``. The only place ``Sigma`` enters.
        key : PRNGKey
        n_lags : int
            Lag order. The design tail must be ``n_lags * q`` wide, since a
            VAR's design carries every series' lags -- that is what makes the
            design common across equations, and the collapse valid.
        n_draws : int
        seed_lags : array ``(n_lags, q)``, optional
            Most-recent-first, so row 0 is ``y_T``. Defaults to the tail of the
            data. This is DLMAX's ``format_seed_lag_yts`` convention: the first
            forecast horizon's lag is ``y_T``.
        state : ``(m_T, C_T)``, optional
        """
        from DLMAX.smoother import _sym_sqrt

        L = jnp.asarray(sigma_tril)
        k = int(n_lags) * self.q
        p_struct = self.p - k
        if p_struct < 0:
            raise ValueError(
                f"n_lags={n_lags} over {self.q} series needs a {k}-wide design "
                f"tail, but the state is only {self.p} wide. A VAR's design "
                "carries every series' lags, so the tail is n_lags * q.")

        if state is None:
            if self._filtered is None:
                self.scan_filter()
            state = (self._filtered[1][-1], self.cov.C[-1])
        m_T, C_T = state
        W = ((1.0 - self.disc_mtx) / self.disc_mtx) * (self.G @ C_T @ self.G.T)
        rootC = _sym_sqrt(jnp.asarray(C_T)[jnp.newaxis])[0][0]
        rootW = _sym_sqrt(jnp.asarray(W)[jnp.newaxis])[0][0]

        if seed_lags is None:
            seed_lags = self.data[-int(n_lags):][::-1]      # most-recent-first
        seed = jnp.asarray(seed_lags).reshape(int(n_lags), self.q)

        F_struct = self.F.reshape(self.p)[:p_struct]

        def one_draw(key_d):
            k0, k_path = random.split(key_d)
            theta = _matrix_normal(k0, m_T, rootC, L)

            def step(carry, key_t):
                theta, lags = carry
                k_th, k_obs = random.split(key_t)
                theta = _matrix_normal(k_th, self.G @ theta, rootW, L)
                # The design at t+k: structural part, then the lag block,
                # most-recent-first and flattened series-within-lag.
                F_t = jnp.concatenate([F_struct, lags.reshape(-1)])
                mean = theta.T @ F_t                        # (q,)
                y = mean + L @ random.normal(k_obs, (self.q,), dtype=mean.dtype)
                # y_{t+k} becomes lag 1 for the next step; the oldest drops.
                lags = jnp.concatenate([y[jnp.newaxis], lags[:-1]], axis=0)
                return (theta, lags), y

            _, ys = scan(step, (theta, seed), random.split(k_path, h))
            return ys                                       # (h, q)

        return vmap(one_draw)(random.split(key, n_draws))    # (n_draws, h, q)

    def params(self):
        """The flat, all-array parameter dict the functional shims consume.

        Every value is an array because ``csrec`` jits its Gibbs kernel and
        passes this straight in, so the dict is traced; key PRESENCE is static
        pytree structure, which is why the backend marker is a key whose value
        is merely a flag.
        """
        out = {"m0": self.m0, "F": self.F, "G": self.G,
               "C": self.cov.C, "R": self.cov.R, "Q": self.cov.Q,
               "A": self.cov.A, "S": self.cov.S, "s": self.smoothed_means(),
               "H": self.cov.H, "sqrtH": self.cov.sqrtH, "B": self.cov.B,
               "delta_M": self.disc_mtx, "V0": self.V0,
               "_mvdlm": jnp.asarray(1.0)}
        if self.sigma_scale is not None:
            # Added only when learning is on: key presence is static pytree
            # structure, so an unconditional key would retrace csrec's kernel.
            out["sigma_scale"] = self.sigma_scale
            out["sigma_dof"] = self.sigma_dof
        return out


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------
# Pure functions of arrays, taking the ``params`` dict rather than a model
# object. That is not stylistic: a caller jits a kernel and passes the dict
# straight in, so the whole thing is traced and every value must be an array.
# Key PRESENCE is static pytree structure, which is why the backend marker is a
# key whose value is merely a flag.


def dlm_params(Y, m0, C0, F, G, V0, disc_mtx=None, shapes=None, delta=None,
               cov_path=None, disc_rates=None):
    """Build a model and return its parameter dict.

    Exactly one of ``disc_mtx``, ``disc_rates`` or ``(shapes, delta)`` is
    needed. ``disc_rates`` is the preferred form: it is what DLMAX's components
    emit, and it lets a caller set a single state's rate -- a damped trend's
    growth slot, say -- without hand-building a congruence matrix. Taking the
    vector as canonical is what lets the ``Component`` API drive this model.

    **Variance learning is not reachable from here**, deliberately. ``Sigma``
    learning is a property of the model object (``mv_dlm(delta_sigma=)``); this
    functional path exists so a caller can jit a kernel around the parameter
    dict, and the callers that do so supply their own ``Sigma``. Adding it here
    would change the pytree those kernels are traced against for no gain.
    """
    if disc_mtx is None:
        if disc_rates is None:
            if shapes is None or delta is None:
                raise ValueError(
                    "specify disc_mtx, disc_rates, or both shapes and delta")
            disc_rates = _rates_from_shapes(shapes, delta)
        disc_mtx = congruence(disc_rates)
    return mv_dlm(data=Y, F=F, G=G, m0=m0, C0=C0, V0=V0, disc_mtx=disc_mtx,
                  cov_path=cov_path).params()


def dlm_forward(Y, params, m0=None, v0=None):
    """The per-series filtering recursion against the cached scale-free path.

    Returns ``(a, m, n, scale)`` stacked over ``t``: prior mean, posterior
    mean, degrees of freedom, and the running observation-variance estimate.

    Split out from :func:`dlm_back_sample` because callers want only this --
    a one-step forecast ``F'a`` needs no backward draw, and obtaining it by
    running one and discarding the result is pure waste.

    Cheap because ``A`` and ``Q`` never depend on ``Y``: this is a scan over
    cached arrays.
    """
    G, Ft = params["G"], params["F"].T
    m0 = params["m0"] if m0 is None else m0
    V0 = params["V0"] if v0 is None else jnp.atleast_2d(v0)

    def fwd(state, xs):
        m, n, scale = state
        y, A, Q = xs
        a = G @ m
        error = y - Ft @ a
        m = a + A @ error
        scale = (error ** 2 / Q + scale * n) / (n + 1)
        return (m, n + 1, scale), (a, m, n + 1, scale)

    _, out = scan(fwd, (m0, 1, V0), (Y, params["A"], params["Q"]))
    return out


def dlm_back_sample(Y, key, sigma_tril, params, m0=None, v0=None, exog=None):
    """Filter forward, then draw the state trajectory with
    :func:`DLMAX.smoother.ffbs`.

    ``sigma_tril`` is ``chol(Sigma)``. It reaches ``ffbs`` as ``right_factor``,
    so the draw is matrix-normal with ``sqrtH`` on the left (state, scale-free)
    and ``Sigma^(1/2)`` on the right (series) -- the series couple there and
    nowhere else.

    ``smoothed`` carries the SHARED path, ``B``/``sqrtH``/``S`` with a series
    axis of 1, which ``ffbs`` broadcasts over the ``q`` series. That is the
    whole point of the Quintana/West form: ``C*`` is common to every series, so
    a per-series copy would be tens of megabytes of duplicated gains.

    Takes a PRNG key rather than pre-drawn normals, because ``ffbs`` owns its
    own draws.
    """
    from DLMAX.smoother import ffbs
    if exog is not None:
        # Refused on identity, not on value. Inspecting the array with
        # bool(jnp.any(...)) raises TracerBoolConversionError under jit -- which
        # surfaces as a confusing trace error rather than this message -- and
        # letting an all-zero design through would make the contract "exog == 0"
        # rather than "no exog", which is not a contract anyone could rely on.
        raise NotImplementedError(
            "exogenous regressors are not supported by mvdlm yet. The model "
            "requires a design COMMON to all series, which is what makes the "
            "Kronecker collapse valid; a per-series design would break it.")

    a, m, n, scale = dlm_forward(Y, params, m0, v0)

    # ffbs wants means as (T, q, p); the forward scan stacks them (T, p, q).
    traj = {"m": m.swapaxes(1, 2)}
    # sqrtH is cached over the full horizon (H_full appends C[-1]); ffbs's scan
    # covers the T-1 transitions and takes the terminal draw from S[-1].
    smoothed = {"S": params["S"][:, jnp.newaxis],
                "B": params["B"][:, jnp.newaxis],
                "sqrtH": params["sqrtH"][:-1][:, jnp.newaxis]}

    states = ffbs(key, traj, params["G"], jnp.diag(params["delta_M"]),
                  n_draws=1, right_factor=jnp.asarray(sigma_tril),
                  smoothed=smoothed)[0]                       # (T, q, p)
    return (a, m, n, scale), (params["F"].T @ states.swapaxes(1, 2)).squeeze(1)


def dlm_forecast(args, h):
    """``h``-step projection from the smoothed terminal state. Pure arrays.

    W&H p.199: ``W`` is fixed from the forecast origin and held over the
    horizon rather than re-derived at each step.

    Returns ``(a, R, f, var)``. **``var`` is the STATE component only**,
    ``F' R_h F``, with no observation term -- deliberately, because callers add
    the observation half themselves and adding it here would double-count.
    :meth:`mv_dlm.forecast` returns the other convention, ``1 + F' R_h F``,
    which is the predictive. The difference is exactly one, in either direction
    invisible in the output, so the two are kept apart and both are tested.
    """
    G, F, delta = args["G"], args["F"], args["delta_M"]
    m, C = args["s"][-1], args["S"][-1]
    W = ((1.0 - delta) / delta) * (G @ C @ G.T)

    def step(carry, _):
        m, C = carry
        a = G @ m
        R = G @ C @ G.T + W
        return (a, R), (a, R, F.T @ a, F.T @ R @ F)

    _, out = scan(step, (m, C), jnp.ones(h))
    return out


__all__ = ["mv_dlm", "CovariancePath", "dlm_params", "dlm_forward",
           "dlm_back_sample", "dlm_forecast", "congruence",
           "component_discount_matrix", "seasonal_prior"]


def _matrix_normal(key, mean, root_left, root_right):
    """One draw from ``N(mean, (LL') (x) (RR'))`` for a ``(p, q)`` mean.

    ``Theta = mean + L Z R'`` with ``Z`` standard normal: the left root shapes
    the state dimension, the right root the series dimension, and the Kronecker
    structure is exactly that separability.

    ``root_left`` comes from :func:`DLMAX.smoother._sym_sqrt` rather than a
    Cholesky because the covariances here are routinely SINGULAR -- a seasonal
    dummy prior is rank-deficient by construction -- and ``cholesky`` returns
    NaN on those while the symmetric square root does not.
    """
    z = random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + root_left @ z @ root_right.T

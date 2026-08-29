"""``Predictive`` — family-carrying forecast distributions + DMA combination.

The output side of the block face. A block emits per-model ``Predictive``s
on the **observation scale**; the *family* (Gaussian / Student-t / lognormal /
...) is a property of the block, which is what lets blocks of different kinds
combine through one seam. :func:`combine` reduces a set of per-model
predictives to the final reported forecast given the DMA weights.

Every ``Predictive`` is vectorised — its parameters are arrays (any batch shape,
e.g. ``(n_series, h)`` or, for a stacked model ensemble, ``(nm, n_series, h)``).
It exposes ``mean`` / ``var`` / ``sd`` / ``quantile(p)`` / ``log_score(y)``, all
family-specific but with a uniform interface, so :func:`combine` is
family-agnostic (Vincent averaging needs only ``.sd`` / ``.quantile``; the
mixture form needs only ``.log_score``).

Combine forms
-------------
- ``"vincent"`` (default) — weighted mean plus a Vincentised
  (quantile-averaged) SD, the combined forecast reported as
  ``N(loc, sd_vincent)`` for the log-score, with per-level quantile-averaged
  intervals. This is the combine the reported forecasts use, and it agrees
  bitwise with ``_t_vincent_sd`` / ``_t_quantile_average`` in the Student-t
  case.
- ``"mixture"`` — the proper mixture ``p(y) = Σ w_m p_m(y)``, scored as
  ``logsumexp_m(log w_m + log p_m(y))``. Distributionally the more correct
  object, but it is sharper than the second moment warrants on this model set,
  so it is offered rather than defaulted: switching changes every reported
  number and is a calibration decision, not a drop-in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm, t as t_dist

_Z95 = 1.959963984540054  # norm.isf(0.025); matches ffs_core._t_vincent_sd


@runtime_checkable
class Predictive(Protocol):
    """A vectorised predictive distribution on the observation scale."""

    @property
    def mean(self): ...
    @property
    def var(self): ...
    @property
    def sd(self): ...
    def quantile(self, p: float): ...
    def log_score(self, y): ...


class GaussianPredictive:
    """``N(mean, var)`` on the observation scale."""

    def __init__(self, mean, var):
        self._mean = np.asarray(mean, float)
        self._var = np.asarray(var, float)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    @property
    def sd(self):
        return np.sqrt(self._var)

    def quantile(self, p):
        return self._mean + self.sd * norm.ppf(p)

    def log_score(self, y):
        return norm.logpdf(np.asarray(y, float), self._mean, self.sd)


class StudentTPredictive:
    """``y ~ T(loc, sqrt(scale2), nu)`` — the per-model DMA component.

    ``sd`` is the **variance-matched** SD (``sqrt(scale2 · nu/(nu-2))`` for
    ``nu > 2``, falling back to the 97.5%-quantile-implied Gaussian SD for
    ``nu <= 2``), matching ``ffs_core._t_vincent_sd``'s per-component ``sd_m`` so
    the Vincent combine is bit-reproducible.
    """

    def __init__(self, loc, scale2, nu):
        self._loc = np.asarray(loc, float)
        self._scale2 = np.asarray(scale2, float)
        self._nu = np.asarray(nu, float)

    @property
    def mean(self):
        return self._loc

    @property
    def var(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            infl = np.where(self._nu > 2.0, self._nu / (self._nu - 2.0), np.nan)
        return self._scale2 * _broadcast_nu(infl, self._scale2)

    @property
    def sd(self):
        scale = np.sqrt(self._scale2)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_factor = np.sqrt(self._nu / (self._nu - 2.0))       # nu > 2
        quant_factor = t_dist.ppf(0.975, self._nu) / _Z95           # any nu > 0
        factor = np.where(self._nu > 2.0, var_factor, quant_factor)
        return scale * _broadcast_nu(factor, scale)

    def quantile(self, p):
        scale = np.sqrt(self._scale2)
        return self._loc + scale * _broadcast_nu(t_dist.ppf(p, self._nu), scale)

    def log_score(self, y):
        scale = np.sqrt(self._scale2)
        return t_dist.logpdf(np.asarray(y, float), _broadcast_nu(self._nu, scale),
                             loc=self._loc, scale=scale)


class LogNormalPredictive:
    """``log(y) ~ N(mu_log, sigma2_log)`` — the log block's observation-scale
    predictive. ``mean`` / ``var`` are the lognormal moments; ``log_score``
    carries the change-of-variables Jacobian (``-log y``) so it is directly
    comparable, on the observation scale, with a Gaussian block's log score."""

    def __init__(self, mu_log, sigma2_log):
        self._mu = np.asarray(mu_log, float)
        self._s2 = np.asarray(sigma2_log, float)

    @property
    def mean(self):
        return np.exp(self._mu + 0.5 * self._s2)

    @property
    def var(self):
        return (np.exp(self._s2) - 1.0) * np.exp(2.0 * self._mu + self._s2)

    @property
    def sd(self):
        return np.sqrt(self.var)

    def quantile(self, p):
        return np.exp(self._mu + np.sqrt(self._s2) * norm.ppf(p))

    def log_score(self, y):
        y = np.asarray(y, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return norm.logpdf(np.log(y), self._mu, np.sqrt(self._s2)) - np.log(y)


def _broadcast_nu(nu_like, ref):
    """Broadcast a ``(nm, n_series)``-shaped per-model quantity to ``ref``'s
    ``(nm, n_series, h)`` when ``ref`` carries a trailing horizon axis."""
    nu_like = np.asarray(nu_like, float)
    if ref.ndim == nu_like.ndim + 1:
        return nu_like[..., None]
    return nu_like


class CombinedPredictive:
    """Result of :func:`combine` (``form="vincent"``): a combined forecast
    reported as ``N(mean, sd_vincent)`` for the log-score, with Vincent
    (quantile-averaged) intervals from the underlying components. Holds the
    components + weights so the quantile average stays exact per level rather
    than collapsing to the Gaussian's symmetric interval."""

    def __init__(self, mean, sd, stacked_quantile, weights_finite):
        self._mean = np.asarray(mean, float)     # (n_series, h)
        self._sd = np.asarray(sd, float)         # (n_series, h)
        self._stacked_quantile = stacked_quantile  # p -> (nm_total, n_series, h)
        self._w = weights_finite                 # (nm_total, n_series, h) nan-safe weights

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._sd ** 2

    @property
    def sd(self):
        return self._sd

    def quantile(self, p):
        qm = self._stacked_quantile(p)           # (nm_total, n_series, h)
        return (np.where(np.isfinite(qm), qm, 0.0) * self._w).sum(axis=0)

    def interval(self, level):
        p = 0.5 + level / 200.0
        return self.quantile(1.0 - p), self.quantile(p)

    def log_score(self, y):
        return norm.logpdf(np.asarray(y, float), self._mean, self._sd)


def combine(components, weights, *, form: str = "vincent"):
    """Combine per-model ``Predictive``s with DMA ``weights`` into the final
    forecast.

    Parameters
    ----------
    components : list[Predictive]
        Per-block predictives, each **stacked** on a leading model axis
        (``nm_block, n_series, h`` for means/quantiles; ``nm_block, n_series``
        for nu-like). Concatenated along the model axis internally, so blocks of
        different families combine directly.
    weights : list[np.ndarray]
        Per-block DMA weights ``(nm_block, n_series)``, aligned to
        ``components`` — the union ``Allocator``'s output, split by block.
    form : {"vincent", "mixture"}
        See module docstring. ``"vincent"`` is the current-FFS default.

    Returns
    -------
    CombinedPredictive (form="vincent") or MixturePredictive (form="mixture").
    """
    from DLMAX.ffs_core import _nan_safe_w  # lazy: avoid import cycle

    means = np.concatenate([np.asarray(c.mean, float) for c in components], axis=0)
    sds = np.concatenate([np.asarray(c.sd, float) for c in components], axis=0)
    w = np.concatenate([np.asarray(x, float) for x in weights], axis=0)  # (nm, n_series)

    finite = np.isfinite(means) & np.isfinite(sds)          # (nm, n_series, h)
    w_safe = _nan_safe_w(w, finite)                         # (nm, n_series, h)
    mean = (np.where(finite, means, 0.0) * w_safe).sum(axis=0)   # (n_series, h)

    if form == "mixture":
        return MixturePredictive(components, weights)
    if form != "vincent":
        raise ValueError(f"form must be 'vincent' or 'mixture', got {form!r}")

    sd = (np.where(finite, sds, 0.0) * w_safe).sum(axis=0)   # Vincent SD (n_series, h)

    def stacked_quantile(p):
        return np.concatenate([np.asarray(c.quantile(p), float) for c in components], axis=0)

    return CombinedPredictive(mean, sd, stacked_quantile, w_safe)


class MixturePredictive:
    """Result of :func:`combine` (``form="mixture"``): the proper DMA mixture
    ``p(y) = Σ w_m p_m(y)``. Sharper/looser than Vincent is a calibration
    question; this form MOVES the numbers vs the current path (the retest
    option, not the default)."""

    def __init__(self, components, weights):
        self._components = components
        self._weights = [np.asarray(x, float) for x in weights]

    @property
    def mean(self):
        means = np.concatenate([np.asarray(c.mean, float) for c in self._components], axis=0)
        w = np.concatenate(self._weights, axis=0)
        from DLMAX.ffs_core import _nan_safe_w
        finite = np.isfinite(means)
        w_safe = _nan_safe_w(w, finite)
        return (np.where(finite, means, 0.0) * w_safe).sum(axis=0)

    def log_score(self, y):
        # logsumexp_m( log w_m + log p_m(y) ) over the union of models.
        logs = np.concatenate(
            [np.atleast_1d(c.log_score(y))[None] if np.ndim(c.log_score(y)) == 0
             else np.asarray(c.log_score(y), float) for c in self._components],
            axis=0,
        )
        w = np.concatenate(self._weights, axis=0)
        logw = np.log(np.where(w > 0, w, np.nan))
        # broadcast logw (nm, n_series) to logs (nm, n_series, h)
        if logs.ndim == logw.ndim + 1:
            logw = logw[..., None]
        a = logs + logw
        m = np.nanmax(a, axis=0, keepdims=True)
        return (m[0] + np.log(np.nansum(np.exp(a - m), axis=0)))

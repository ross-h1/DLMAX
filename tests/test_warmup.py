"""What a warmup step means, pinned at the engine.

A warmup step forces ``W = 0`` for that step -- the discount is held at 1, so
the state covariance evolves only through ``G`` with no forgetting inflation.
**That is the whole effect.** The observational variance ``s``/``nu`` updates
normally; it is gated by ``ignore_obs`` (a NaN observation) and by nothing else.

This file exists because the absence of a test here let three docstrings claim
for two releases that warmup also "holds the observational variance", which no
kernel has ever done. ``test_warmup_zeroes_W_and_touches_nothing_else`` is the
executable statement of the contract.

There was a ``tests/test_warmup.py`` before (commit 040fc24); it was deleted in
cf60fb2 as collateral, because it used the removed ``Mcomp_DLM`` as its oracle.
This keeps its three-part shape -- no-op guarantee, semantics, prior/faces --
and needs no oracle: every property is a comparison between two runs of the
engine itself.

A note on isolation that bites every test here. ``warmup_steps`` does TWO jobs:
it sets the W=0 window AND selects the diffuse prior over the legacy OLS
elicitation (``DLM.design``). So ``build(6)`` and ``build(0)`` differ in their
priors as well as their windows, and comparing them measures both at once. To
isolate the window, build at the same ``warmup_steps`` and null the window on
the comparator with ``_no_window`` below.
"""
import numpy as np
import pandas as pd
import pytest
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from DLMAX.dlm_core import (multi_model_dlm, resolve_warmup_flag,
                            dlm_uv_fwd_qr_step)
from DLMAX.ffs.dlm_builder import DLM, LocalTrend, Adapt, Wing
from DLMAX.ffs import devices

T, W = 24, 6


@pytest.fixture(scope="module")
def y():
    rng = np.random.default_rng(11)
    return np.cumsum(rng.normal(0, 1.0, 60)) + 100.0


def _build(y, warmup_steps, h=2):
    d = DLM(n_series=1)
    d.add_component(LocalTrend(name="t", disc_rate=0.95))
    d.set_error(disc_rate=0.99)
    return d.compile(init_data=pd.DataFrame({"y": y[:12]}),
                     warmup_steps=warmup_steps, h=h)


def _no_window(m):
    """Same model, same PRIOR, no warmup window -- the isolation the module
    docstring describes."""
    m.warmup_steps = 0
    return m


def _drive(m, y, n, flags=None):
    out = []
    for t in range(n):
        f = None if flags is None else flags[t]
        b = m.fwd_filter(jnp.asarray([y[t]]), warmup_flag=f)
        out.append((float(np.asarray(b.loc)[0]), float(np.asarray(b.var)[0])))
    return np.asarray(out)


def _universe(y, warmup_steps):
    d = DLM(n_series=1)
    d.add_component(LocalTrend(name="t", disc_rate=[0.95, 0.98]))
    d.set_error(disc_rate=0.99)
    models, _desc = d.compile_universe(
        init_data=pd.DataFrame({"y": y[:12]}), warmup_steps=warmup_steps, h=2)
    return multi_model_dlm(models, devices.dlm_compute)


# ---------------------------------------------------------------------------
# 1. The no-op guarantee
# ---------------------------------------------------------------------------
# Every model compiled without the option has warmup_steps == 0, and must be
# bit-identical to the engine as it stood before warmup was readable at all.
# uv_dlm.fwd_filter guards the multiply at PYTHON level for this reason: it is
# not jitted, so a zero warmup emits no operation rather than a * 1.0.

def test_zero_warmup_is_bitwise_unchanged(y):
    a = _drive(_build(y, 0), y, T)
    b = _drive(_no_window(_build(y, 0)), y, T)
    np.testing.assert_array_equal(a, b)


def test_zero_warmup_multi_is_bitwise_unchanged(y):
    a = _universe(y, None)
    b = _universe(y, None)
    assert a.warmup_steps == 0
    fa = a.scan_filter(jnp.asarray(y[:T]).reshape(T, 1))
    fb = b.scan_filter(jnp.asarray(y[:T]).reshape(T, 1))
    np.testing.assert_array_equal(np.asarray(fa.loc), np.asarray(fb.loc))
    np.testing.assert_array_equal(np.asarray(a.dlm_state["m"]),
                                  np.asarray(b.dlm_state["m"]))


def test_resolver_rule():
    """Explicit wins; None consults the window. The one definition both
    engines share, so they cannot drift on it again."""
    assert resolve_warmup_flag(1.0, 99, 0) == 1.0      # explicit beats exhausted
    assert resolve_warmup_flag(0.0, 0, 10) == 0.0      # explicit beats inside
    assert resolve_warmup_flag(None, 3, 6) == 1.0      # inside the window
    assert resolve_warmup_flag(None, 6, 6) == 0.0      # boundary is exclusive
    assert resolve_warmup_flag(None, 0, 0) == 0.0      # no window


# ---------------------------------------------------------------------------
# 2. The semantics
# ---------------------------------------------------------------------------

def test_warmup_zeroes_W_and_touches_nothing_else(y):
    """THE contract. One kernel step at warm=1 vs warm=0 from an identical
    state.

    The wrong docstrings said warmup "holds the observational variance". Held
    would mean ``s_post == s_prior``. It does not: the West & Harrison variance
    update runs on a warmup step exactly as on any other, gated only by
    ``ignore_obs``. So the refutation is that ``s`` MOVES OFF ITS PRIOR under
    warmup -- not that it is unchanged between the two runs.

    It is worth being exact about why ``s`` differs between them at all, since
    that is easy to misread as evidence for the old claim. ``s_upd`` divides by
    ``q``, the predictive variance, which carries the covariance and therefore
    the discount. Warmup changes ``s``'s INPUT; it does not gate its update.
    ``nu``, which the kernel comments note is deterministic in the discounts,
    is identical between the two -- as is ``f``, being F(Gm), with no covariance
    in it. ``q`` is not, since it has.
    """
    k = 2
    G = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    F = jnp.array([1.0, 0.0])
    s_prior, nu_prior = 1.5, 9.0
    state = {"m": jnp.array([100.0, 0.5]), "Z": jnp.eye(k) * 2.0,
             "s": jnp.array(s_prior), "nu": jnp.array(nu_prior)}
    data = {"F": F, "G": G, "y": jnp.array([101.0])}
    disc = jnp.diag(jnp.array([(1 - 0.95) / 0.95, (1 - 0.98) / 0.98]))

    live, m_live = dlm_uv_fwd_qr_step(disc, 0.99, 1.0, jnp.zeros(k), state, data)
    warm, m_warm = dlm_uv_fwd_qr_step(disc * 0.0, 0.99, 1.0, jnp.zeros(k),
                                      state, data)

    # NOT held: the variance update ran on the warmup step, as on any other.
    assert not np.isclose(float(np.asarray(warm["s"]).ravel()[0]), s_prior)
    assert not np.isclose(float(np.asarray(warm["nu"]).ravel()[0]), nu_prior)

    # nu is deterministic in the discounts, so warmup cannot move it; nor f.
    np.testing.assert_array_equal(np.asarray(live["nu"]), np.asarray(warm["nu"]))
    np.testing.assert_array_equal(np.asarray(m_live["f"]), np.asarray(m_warm["f"]))

    # What warmup DOES change: the covariance, and what depends on it.
    assert not np.array_equal(np.asarray(live["Z"]), np.asarray(warm["Z"]))
    assert not np.array_equal(np.asarray(m_live["q"]), np.asarray(m_warm["q"]))


def test_warmup_propagates_the_covariance_through_G_alone(y):
    """W = 0 means R_t = G C G' exactly -- no forgetting inflation.

    In the QR kernel the discount enters as ``b = sqrt(1 + diag(disc_factor))``,
    so a zeroed disc_factor gives b = 1 and the prior root is just ZG'.
    """
    k = 2
    G = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    Z = jnp.array([[2.0, 0.5], [0.0, 1.5]])
    state = {"m": jnp.array([100.0, 0.5]), "Z": Z,
             "s": jnp.array(1.0), "nu": jnp.array(9.0)}
    # A missing observation isolates the prior propagation: the posterior IS
    # the prior, so the returned Z is exactly R_t's root.
    data = {"F": jnp.array([1.0, 0.0]), "G": G, "y": jnp.array([jnp.nan])}
    warm, _ = dlm_uv_fwd_qr_step(jnp.zeros((k, k)), 0.99, 1.0, jnp.zeros(k),
                                 state, data)
    C_warm = np.asarray(warm["Z"]).T @ np.asarray(warm["Z"])
    C_expect = np.asarray(G) @ (np.asarray(Z).T @ np.asarray(Z)) @ np.asarray(G).T
    np.testing.assert_allclose(C_warm, C_expect, rtol=1e-12, atol=1e-12)


def test_auto_equals_an_explicit_flag_sequence(y):
    """Self-counting is exactly [1]*W + [0]* — no off-by-one at the boundary."""
    auto = _drive(_build(y, W), y, T)
    expl = _drive(_build(y, W), y, T, flags=[1.0] * W + [0.0] * (T - W))
    np.testing.assert_array_equal(auto, expl)


def test_explicit_flag_overrides_the_window(y):
    """Explicit 0.0 on a warmup model reproduces the same model with no window
    (same prior — see the module docstring on isolation)."""
    forced = _drive(_build(y, W), y, T, flags=[0.0] * T)
    nowin = _drive(_no_window(_build(y, W)), y, T)
    np.testing.assert_array_equal(forced, nowin)


def test_explicit_flag_does_not_suspend_the_counter(y):
    """The window is a property of observations absorbed, not of how the caller
    labelled them."""
    m = _build(y, W)
    m.fwd_filter(jnp.asarray([y[0]]), warmup_flag=0.0)
    assert m._warm_t == 1


def test_counter_is_positional_so_a_nan_consumes_a_slot(y):
    """Matches GridBlock's jnp.arange(_t, _t+T) < warmup and StaticBlock's _t,
    both of which count NaN rows. Discount LEARNING is gated on finiteness
    (discount_grid._wing_step); the warmup flag itself is not."""
    yn = y.copy()
    yn[1] = np.nan

    def run(m, flags=None):
        for t in range(6):
            m.fwd_filter(jnp.asarray([yn[t]]),
                         warmup_flag=None if flags is None else flags[t])
        return np.asarray(m.dlm_state["m"])

    auto = run(_build(y, 3))
    counted = run(_no_window(_build(y, 3)), [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    skipped = run(_no_window(_build(y, 3)), [1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(auto, counted)
    assert not np.array_equal(auto, skipped)


# ---------------------------------------------------------------------------
# 3. Inheritance, faces and persistence
# ---------------------------------------------------------------------------

def test_multi_inherits_warmup_from_its_members(y):
    assert _universe(y, W).warmup_steps == W
    assert _universe(y, None).warmup_steps == 0


def test_multi_takes_the_max_over_disagreeing_members(y):
    """Mirrors AutoFFS._warmup_target: a shorter-window member simply filters
    the remaining rows normally, as it would on its own."""
    a, b = _build(y, 3), _build(y, 7)
    assert multi_model_dlm({"a": a, "b": b},
                           devices.dlm_compute).warmup_steps == 7


def test_multi_scan_equals_step_loop_with_warmup_on(y):
    """The gap this closes: scan_filter took no warmup parameter at all, so it
    could not reproduce a stepwise caller over a warmup window.

    Not bitwise — a fused lax.scan reduces in a different order from a Python
    step loop, as test_fwd_filter_face documents for the same pair.
    """
    a = _universe(y, W)
    a.scan_filter(jnp.asarray(y[:T]).reshape(T, 1))
    b = _universe(y, W)
    for t in range(T):
        b.fwd_filter(jnp.asarray([y[t]]))
    np.testing.assert_allclose(np.asarray(a.dlm_state["m"]),
                               np.asarray(b.dlm_state["m"]), rtol=1e-10, atol=0)


def test_multi_scan_warmup_actually_does_something(y):
    """Guards against the above passing because both sides ignore warmup."""
    on = _universe(y, W)
    on.scan_filter(jnp.asarray(y[:T]).reshape(T, 1))
    off = _no_window(_universe(y, W))
    off.scan_filter(jnp.asarray(y[:T]).reshape(T, 1))
    assert not np.array_equal(np.asarray(on.dlm_state["m"]),
                              np.asarray(off.dlm_state["m"]))


def test_chunked_scan_does_not_rewarm(y):
    """The counter offset earns its keep: a scan resumed mid-window must not
    re-apply warmup to rows it has already absorbed."""
    whole = _universe(y, W)
    whole.scan_filter(jnp.asarray(y[:T]).reshape(T, 1))
    chunked = _universe(y, W)
    chunked.scan_filter(jnp.asarray(y[:4]).reshape(4, 1))
    chunked.scan_filter(jnp.asarray(y[4:T]).reshape(T - 4, 1))
    np.testing.assert_allclose(np.asarray(whole.dlm_state["m"]),
                               np.asarray(chunked.dlm_state["m"]),
                               rtol=1e-10, atol=0)


def test_persistence_round_trips_window_and_counter(y, tmp_path):
    """Both, not just the window: a universe reopened mid-warmup with a reset
    counter would re-warm rows it has already absorbed."""
    fn = str(tmp_path / "u.h5")
    u = _universe(y, W)
    u.fwd_filter(jnp.asarray([y[0]]))
    u.fwd_filter(jnp.asarray([y[1]]))
    u.save(fn)
    r = multi_model_dlm()
    r.load(fn)
    assert (r.warmup_steps, r._warm_t) == (W, 2)


def test_legacy_file_without_warmup_fields_loads_as_zero(y, tmp_path):
    """Back-compat: files written before warmup was inherited have neither
    field, and 0/0 is exactly how they behaved."""
    import h5py
    fn = str(tmp_path / "old.h5")
    _universe(y, W).save(fn)
    with h5py.File(fn, "a") as f:
        del f["dims"]["warmup_steps"]
        del f["dims"]["warm_t"]
    r = multi_model_dlm()
    r.load(fn)
    assert (r.warmup_steps, r._warm_t) == (0, 0)


# ---------------------------------------------------------------------------
# 4. One clock: the builder and the RTRL overlays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [Adapt(init=0.95), Wing(init=0.95)])
def test_learner_defaults_to_one_number_on_both(y, spec):
    """A None-compiled learner used to carry 0 on the model and 6 on the
    overlay. The default is relocated, not revised: it is still 6."""
    from DLMAX.ffs.dlm_builder import DEFAULT_LEARNER_WARMUP
    d = DLM(n_series=1)
    d.add_component(LocalTrend(name="t", disc_rate=spec))
    d.set_error(disc_rate=0.99)
    m = d.compile(init_data=pd.DataFrame({"y": y[:12]}), warmup_steps=None, h=2)
    carry = m._adapt if m._adapt is not None else m._wing
    assert m.warmup_steps == DEFAULT_LEARNER_WARMUP == carry["warmup"]


def test_fixed_discount_model_still_defaults_to_no_warmup(y):
    """Only a learner gets the non-zero default."""
    d = DLM(n_series=1)
    d.add_component(LocalTrend(name="t", disc_rate=0.95))
    d.set_error(disc_rate=0.99)
    m = d.compile(init_data=pd.DataFrame({"y": y[:12]}), warmup_steps=None, h=2)
    assert m.warmup_steps == 0


@pytest.mark.parametrize("spec", [Adapt(init=0.95), Wing(init=0.95)])
def test_overlays_share_the_model_counter(y, spec):
    """The overlays kept private step counters; they now read self._warm_t.
    The sequences coincide because fwd_filter always routes to the overlay on a
    learner model and nothing else advances the count."""
    d = DLM(n_series=1)
    d.add_component(LocalTrend(name="t", disc_rate=spec))
    d.set_error(disc_rate=0.99)
    m = d.compile(init_data=pd.DataFrame({"y": y[:12]}), warmup_steps=4, h=2)
    for t in range(9):
        m.fwd_filter(jnp.asarray([y[t]]))
    assert m._warm_t == 9
    carry = m._adapt if m._adapt is not None else m._wing
    assert "step" not in carry          # the private counter is gone
    assert carry["warmup"] == m.warmup_steps == 4


# ---------------------------------------------------------------------------
# 5. The smoother, which warmup otherwise breaks silently
# ---------------------------------------------------------------------------

def test_smoother_reports_warmup_as_approximate(y):
    """smooth()/backward_sample() pass applied_disc, a time-CONSTANT vector, so
    the backward recursion rebuilds R_t with inflation the warmup steps never
    had. Approximate, and it must say so rather than return quietly."""
    from DLMAX.smoother import SmootherError
    m = _build(y, W)
    m.adapt = None
    for t in range(T):
        m.fwd_filter(jnp.asarray([y[t]]), trajectory=True)
    out = m.smooth()
    assert out["exact"] is False
    assert any("warmup_steps" in r for r in out["approximations"])
    with pytest.raises(SmootherError, match="warmup_steps"):
        m.backward_sample(jax.random.PRNGKey(0))


def test_smoother_is_silent_without_a_warmup_window(y):
    m = _build(y, 0)
    m.adapt = None
    for t in range(T):
        m.fwd_filter(jnp.asarray([y[t]]), trajectory=True)
    out = m.smooth()
    assert not any("warmup" in r for r in out["approximations"])


def test_smoother_warmup_error_is_confined_to_the_window(y):
    """The flag promises "exact only from step warmup_steps on; pass a
    trajectory that starts after the window". This makes that promise
    executable, in two halves.

    FIRST: smoothing the full trajectory and smoothing only its post-warmup
    slice agree from step N on. That the contamination cannot propagate forward
    is structural -- RTS runs backward, so an error in R_{t+1} at a warmup step
    reaches only s_t for t < N -- rather than a delicate numerical fact. The
    test is here to pin the USER-FACING advice, and it would catch a change to
    rts_smooth that broke the confinement (a forward-backward pass, say, or any
    global normalisation across t).

    SECOND: over a window that is ENTIRELY warmup, the smoother's constant-
    discount reconstruction really does differ from the truth, and in the
    documented direction. The filter used delta = 1 there, so the correct
    reconstruction is disc = 1; ``applied_disc`` instead inflates R_t by 1/delta,
    which under-states the backward gain B and so damps the correction. That is
    the error the guard exists to announce, and it is conservative: it leaves the
    earliest states closer to their filtered values, in exactly the stretch where
    the prior was diffuse.
    """
    from DLMAX.smoother import rts_smooth

    N = W
    m = _build(y, N)
    m.adapt = None
    for t in range(T):
        m.fwd_filter(jnp.asarray([y[t]]), trajectory=True)
    traj = m.trajectory

    # --- first half: the escape hatch --------------------------------------
    full = rts_smooth(traj, m.G, m.applied_disc)
    tail = rts_smooth({k: v[N:] for k, v in traj.items()}, m.G, m.applied_disc)
    np.testing.assert_allclose(np.asarray(full["s"])[N:], np.asarray(tail["s"]),
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(full["S"])[N:], np.asarray(tail["S"]),
                               rtol=1e-12, atol=0)

    # --- second half: inside the window, the assumption is wrong ------------
    head = {k: v[:N] for k, v in traj.items()}
    assumed = rts_smooth(head, m.G, m.applied_disc)          # what DLMAX does
    truth = rts_smooth(head, m.G, jnp.ones_like(m.applied_disc))  # delta = 1
    assert not np.allclose(np.asarray(assumed["s"]), np.asarray(truth["s"]))
    # R over-stated -> B under-stated -> the backward correction is damped
    assert (np.linalg.norm(np.asarray(assumed["B"]))
            < np.linalg.norm(np.asarray(truth["B"])))

# Numerical behaviour

Some of what DLMAX does with awkward data is not obvious from the API, and a
few of the choices look wrong until the reason is stated. This page documents
both: what the library does, and why it does it that way.

## Series that cross or touch zero

The observation variance scales as a power of the mean:

```python
def var_scale_fn(f, var_power, var_floor=1e-12):
    return jnp.maximum(jnp.abs(f) ** (2 * (1 - var_power)), var_floor)
```

With `var_power < 1` the scale is a non-integer power of the forecast mean, so
a **negative** mean would raise a negative base to a fractional power and give
`NaN`. Taking the absolute value *before* exponentiating avoids that; doing it
after cannot, because the `NaN` has already been produced.

The consequence for a user: a multiplicative or compound-Poisson model
(`var_power` of 0, 0.25, 0.5) can be run on a series whose level passes through
or below zero. The variance is scaled by `|f|`, so it is symmetric about zero
rather than undefined below it. The `var_floor` keeps the scale positive when
`f` is exactly zero, which would otherwise collapse the observation variance.

If you need the variance to be a function of the *signed* level, this is not
the model to use.

## Missing observations

A missing value (`NaN` in `y`) is treated as **no information**, not as a zero
and not as an error. On such a step the state evolves and its uncertainty grows
by the discount, but nothing is assimilated: the posterior is the prior.

The observation-variance state is carried forward unchanged:

```python
s_t = jnp.where(ignore_obs, Stm1["s"], s_t / var_scale)
```

Rescaling it on a step where no observation arrived would let the variance
estimate drift on the strength of data that was never seen. This is what makes
pre-launch periods, gaps and structurally-missing observations safe to leave as
`NaN` rather than imputing them — imputation would inject information the model
then treats as real.

## Flat or partly-missing warmup windows

The diffuse prior is elicited from the warmup window:

```python
self._diffuse_sigma2 = jnp.maximum(jnp.nanvar(Y, axis=0), 1e-6)
```

Two things are guarded. `nanvar` rather than `var`, so a window containing
`NaN` — the ordinary case for a series that has not launched yet — still yields
a usable prior instead of `NaN` for the whole series. And a floor of `1e-6`, so
a *flat* window (constant, all-zero, all-identical) does not give `V0 = C0 = 0`
and a degenerate prior.

The floor is absolute, not relative. For data spanning very different scales a
relative floor would be more principled; in practice the absolute one is
sufficient, but it is worth knowing if your series are all far below unit
scale.

## Model averaging under pathological data

If a model-averaging score update produces `NaN`, the affected model's
posterior falls back to its prior rather than propagating:

```python
pset_post = jnp.where(jnp.isnan(pset_post), pset_prior, pset_post)
mset_post = jnp.where(jnp.isnan(mset_post), mset_prior, mset_post)
```

Without it, one pathological series could take the whole model set with it. The
effect is that such a series stops updating its weights rather than destroying
them; its forecasts continue from the last good weighting.

## Priors from the component builder

`DLM.compile()` leaves explicitly elicited priors untouched, and does not use
the prior context when setting `V0`. This keeps `NaN` values in the initial
data from contaminating a prior that the caller supplied deliberately.

---

*A note for contributors:* the first three items above are load-bearing and
read like mistakes — an `abs` in an unusual place, a `where` that looks
redundant beside its neighbours, a floor that looks arbitrary. Each is a fix
for a specific numerical failure, and the natural-looking simplification
reintroduces it.

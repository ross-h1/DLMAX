"""Wide / tidy input for the canonical ``AutoFFS``.

Feed a wide frame (index = time, columns = series ids, values = y) instead of
the long ``(unique_id, ds, y)`` form. Wide is auto-detected and converted;
the result must match the equivalent long input exactly, and ragged panels are
just trailing NaN in the wide grid. Legacy stays long-only (frozen).
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import pytest

from DLMAX.ffs_core import AutoFFS, _wide_to_long


def _ragged():
    """3 ragged series (different lengths = trailing NaN in the wide grid)."""
    rng = np.random.default_rng(0)
    L = 24
    lengths = [24, 18, 21]
    cols = {}
    for i, n in enumerate(lengths):
        y = 100.0 + 10 * i + np.cumsum(rng.normal(0.5, 1.0, n))
        col = np.full(L, np.nan)
        col[:n] = y
        cols[f"s{i}"] = col
    wide = pd.DataFrame(cols, index=np.arange(L))  # index = integer time
    # hand-built long, INDEPENDENT of _wide_to_long
    rows = [
        {"unique_id": f"s{i}", "ds": t, "y": float(cols[f"s{i}"][t])}
        for i, n in enumerate(lengths)
        for t in range(n)
    ]
    return wide, pd.DataFrame(rows)


def test_wide_to_long_adapter():
    wide = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [np.nan, 3.0, 4.0]},
                        index=[10, 11, 12])
    long = _wide_to_long(wide)
    assert set(long.columns) == {"ds", "unique_id", "y"}
    assert len(long) == 4  # NaN cells dropped
    assert sorted(long.loc[long.unique_id == "a", "ds"]) == [10, 11]
    assert sorted(long.loc[long.unique_id == "b", "ds"]) == [11, 12]


def test_wide_input_matches_long():
    wide, long = _ragged()
    kw = dict(h=3, n_windows=1, freq=1)
    cv_wide = AutoFFS(season_length=None, n_seas_comps=None).cross_validation(
        wide, warmup_steps=4, **kw)
    cv_long = AutoFFS(season_length=None, n_seas_comps=None).cross_validation(
        long, warmup_steps=4, **kw)
    pd.testing.assert_frame_equal(
        cv_wide.reset_index(drop=True), cv_long.reset_index(drop=True))


def test_long_still_works():
    _wide, long = _ragged()
    out = AutoFFS(season_length=None, n_seas_comps=None).cross_validation(
        long, h=3, n_windows=1, freq=1, warmup_steps=4)
    assert set(out["unique_id"]) == {"s0", "s1", "s2"}


def test_wide_multiindex_columns_are_flattened():
    """MultiIndex columns flatten to a single unique_id joined with '_'.

    Without this, reset_index has to label the index column ("ds", "") to match
    the column depth and the melt fails looking for a plain "ds".
    """
    wide = pd.DataFrame(
        {("FOODS", "s0"): [1.0, 2.0, 3.0], ("HOBBIES", "s1"): [4.0, 5.0, np.nan]},
        index=[10, 11, 12])
    wide.columns = pd.MultiIndex.from_tuples(wide.columns, names=["dept", "item"])
    long = _wide_to_long(wide)
    assert set(long.columns) == {"ds", "unique_id", "y"}
    assert sorted(long["unique_id"].unique()) == ["FOODS_s0", "HOBBIES_s1"]
    assert len(long) == 5  # the NaN cell is dropped as usual


def test_wide_multiindex_matches_pre_flattened_frame():
    wide, _long = _ragged()
    flat = wide.copy()
    mi = wide.copy()
    mi.columns = pd.MultiIndex.from_tuples([("g", c) for c in wide.columns])
    flat.columns = [f"g_{c}" for c in wide.columns]
    kw = dict(h=3, n_windows=1, freq=1)
    a = AutoFFS(season_length=None, n_seas_comps=None).cross_validation(
        mi, warmup_steps=4, **kw)
    b = AutoFFS(season_length=None, n_seas_comps=None).cross_validation(
        flat, warmup_steps=4, **kw)
    # the MultiIndex result additionally carries the restored levels, so
    # compare on the shared columns -- the NUMBERS must be identical
    shared = [c for c in b.columns]
    pd.testing.assert_frame_equal(a[shared].reset_index(drop=True),
                                  b.reset_index(drop=True))


def test_wide_multiindex_duplicate_ids_rejected():
    """('a', 'b_c') and ('a_b', 'c') both flatten to 'a_b_c' — refuse rather
    than silently merge two series."""
    wide = pd.DataFrame({("a", "b_c"): [1.0, 2.0], ("a_b", "c"): [3.0, 4.0]},
                        index=[10, 11])
    wide.columns = pd.MultiIndex.from_tuples(wide.columns)
    with pytest.raises(ValueError, match="duplicate series ids"):
        _wide_to_long(wide)

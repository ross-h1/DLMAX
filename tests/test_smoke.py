"""
Fast smoke tests for the DLMAX import graph.

These run in well under a second and exist to catch the most common
ways a refactor can break the package — circular imports, missing
public re-exports, mistyped class names. They make no assumptions
about M1M data or any other external state.
"""

import pytest


def test_import_dlmax():
    """Top-level package imports cleanly."""
    import DLMAX  # noqa: F401


def test_import_subpackages():
    """Both legacy and new import paths resolve."""
    import DLMAX.dlm_core  # noqa: F401
    import DLMAX.ffs_core  # noqa: F401
    import DLMAX.ffs  # noqa: F401


def test_autoffs_constructible():
    """The AutoFFS class can be instantiated with default-like args.

    Doesn't fit anything; just checks the constructor and __repr__
    work end to end without runtime errors.
    """
    from DLMAX.ffs_core import AutoFFS

    model = AutoFFS(season_length=12)
    assert "AutoFFS" in repr(model)
    assert model.is_fitted is False

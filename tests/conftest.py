"""
Shared pytest configuration and fixtures for DLMAX tests.

Custom marks
------------
``slow``
    Tests that touch the full M1M dataset and take more than a few
    seconds. Excluded from the default ``pytest`` invocation. Run them
    with ``pytest -m slow`` or ``pytest -m "slow or not slow"``.

Environment variables
---------------------
``DLMAX_M1M_PATH``
    Filesystem path to the M1M HDF5 data file (e.g.
    ``/path/to/M1M.h5``). Tests that depend on M1M data check
    this variable; if unset, those tests are skipped with a clear
    message rather than failing.

    Set in ``~/.zshrc`` (or shell equivalent) for a persistent setup::

        export DLMAX_M1M_PATH=/path/to/M1M.h5
"""

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Custom marks
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom marks so pytest doesn't warn about them."""
    config.addinivalue_line(
        "markers",
        "slow: tests that take more than a few seconds to run (default skipped)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless the user explicitly opts in.

    A test is run if any of these is true:
      * pytest is invoked with ``-m slow`` or ``-m "slow and ..."``
      * pytest is invoked with ``-m "not slow"`` excluding it (handled by pytest)
      * the test has no ``slow`` mark

    Otherwise, slow tests are skipped at collection time.
    """
    selected_mark_expr = config.getoption("-m") or ""
    if "slow" in selected_mark_expr:
        # User asked for slow tests explicitly; do nothing special.
        return
    skip_slow = pytest.mark.skip(reason="slow test (use `-m slow` to run)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Fixtures: M1M data path
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def m1m_path():
    """Path to the M1M HDF5 file, or skip if not configured.

    Reads from the ``DLMAX_M1M_PATH`` environment variable. Skips the
    test (rather than fails) if the variable is unset or the file
    doesn't exist on disk — this keeps the test suite runnable on
    machines without the M1M data, which matters for CI and for
    contributors who don't have the dataset.

    Returns
    -------
    pathlib.Path
        Absolute path to the M1M HDF5 file.
    """
    raw = os.environ.get("DLMAX_M1M_PATH")
    if not raw:
        pytest.skip(
            "DLMAX_M1M_PATH environment variable not set. "
            "Set it to the path of M1M.h5 to run M1M-dependent tests."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        pytest.skip(f"DLMAX_M1M_PATH={path!s} does not exist or is not a file.")
    return path

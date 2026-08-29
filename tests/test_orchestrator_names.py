"""``AutoFFS`` extends ``StaticFFS``: same API, different default model set.

``StaticFFS`` / ``StaticFFSUniverse`` build a FIXED-discount universe.
``AutoFFS`` / ``AutoFFSUniverse`` subclass them and default to the wing, whose
discounts are learned online. They are distinct classes, not aliases, so the
two model sets can be run against each other on identical data.
"""

import jax

jax.config.update("jax_enable_x64", True)

import DLMAX
from DLMAX.ffs_core import (
    AutoFFS,
    StaticFFS,
    AutoFFSUniverse,
    StaticFFSUniverse,
)


def test_wing_orchestrators_extend_the_static_ones():
    assert issubclass(AutoFFS, StaticFFS)
    assert issubclass(AutoFFSUniverse, StaticFFSUniverse)
    # distinct classes, so the two model sets are comparable on the same data
    assert AutoFFS is not StaticFFS
    assert AutoFFSUniverse is not StaticFFSUniverse


def test_package_exports_the_wing_orchestrators():
    assert DLMAX.AutoFFS is AutoFFS
    assert DLMAX.AutoFFSUniverse is AutoFFSUniverse


def test_static_orchestrator_is_importable():
    """Reachable from ``ffs_core`` for a fixed-discount run."""
    from DLMAX.ffs_core import StaticFFS as _S
    assert _S is StaticFFS

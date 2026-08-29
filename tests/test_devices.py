"""Tests for DLMAX.ffs.devices and the configure_devices entry point.

These tests exercise the configuration surface only — they don't
construct DLM models or run the filter, so they're fast and have no
hidden device requirements beyond JAX itself.
"""

import pytest


def test_module_globals_populated_at_import():
    """After import, the three shardings are NamedSharding instances."""
    from jax.sharding import NamedSharding
    from DLMAX.ffs import devices

    assert isinstance(devices.dlm_compute, NamedSharding)
    assert isinstance(devices.allocation_compute, NamedSharding)
    assert isinstance(devices.host_device, NamedSharding)


def test_force_cpu_succeeds():
    """`configure_devices('cpu')` runs without error and produces shardings."""
    from jax.sharding import NamedSharding
    from DLMAX.ffs import devices

    devices.configure_devices("cpu")

    assert isinstance(devices.dlm_compute, NamedSharding)
    assert isinstance(devices.allocation_compute, NamedSharding)
    assert isinstance(devices.host_device, NamedSharding)

    # Reset to defaults so subsequent tests get auto-detection.
    devices.configure_devices()


def test_invalid_compute_raises():
    """Unknown compute strings are rejected before any state mutation."""
    from DLMAX.ffs import devices

    with pytest.raises(ValueError, match="must be 'cpu', 'gpu', or None"):
        devices.configure_devices("tpu")


def test_kwarg_override_replaces_auto_default():
    """A user-supplied sharding replaces the auto-constructed one."""
    from jax import devices as jax_devices, make_mesh
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

    from DLMAX.ffs import devices

    custom_mesh = make_mesh(
        (1,), ("custom",), axis_types=(AxisType.Auto,), devices=jax_devices("cpu")
    )
    custom = NamedSharding(custom_mesh, P("custom"))

    devices.configure_devices("cpu", dlm=custom)

    assert devices.dlm_compute is custom
    # alloc and host should still be auto-built defaults (not the custom).
    assert devices.allocation_compute is not custom
    assert devices.host_device is not custom

    devices.configure_devices()  # reset


def test_back_compat_reexport():
    """`configure_devices` is still importable from `DLMAX.ffs_core`."""
    from DLMAX.ffs_core import configure_devices as reexported
    from DLMAX.ffs.devices import configure_devices as original

    assert reexported is original


def test_require_x64_passes_when_enabled():
    """When x64 is enabled (the default after dlm_core import), check passes."""
    from DLMAX.dlm_core import require_x64

    require_x64()  # Should not raise.


def test_enable_x64_idempotent():
    """`enable_x64` can be called repeatedly with no error."""
    from DLMAX.dlm_core import enable_x64

    enable_x64()
    enable_x64()

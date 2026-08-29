"""Device and sharding configuration for DLMAX.

Provides ``configure_devices``, which sets up three module-level
``NamedSharding`` instances used throughout the FFS pipeline:

* ``dlm_compute`` — sharding for DLM parameters and state.
* ``allocation_compute`` — sharding for DMA / allocator state.
* ``host_device`` — host-side sharding for inputs that need to live
  on CPU regardless of where compute happens.

The defaults are auto-detected on module import: GPU 0 if any GPU is
visible to JAX, otherwise CPU. To force CPU (e.g. for benchmarking
against CPU-only baselines), call ``configure_devices('cpu')`` after
import. To inject custom shardings (e.g. multi-GPU strategies), pass
them via the ``dlm``, ``alloc``, ``host`` keyword arguments.

Re-running ``configure_devices`` cleanly replaces the previous
shardings. Consumers that hold the live module attributes (i.e. refer
to them as ``devices.dlm_compute``) see the update; code that captured
the names with ``from DLMAX.ffs.devices import dlm_compute`` will not
— that's why DLMAX-internal code uses the attribute form.
"""

from typing import Optional

from jax import devices as _jax_devices, make_mesh
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P


# Module-level shardings, populated by ``configure_devices``.
dlm_compute: Optional[NamedSharding] = None
allocation_compute: Optional[NamedSharding] = None
host_device: Optional[NamedSharding] = None


def _resolve_compute(compute: Optional[str]) -> str:
    """Return the resolved compute target ('cpu' or 'gpu').

    If ``compute`` is None, auto-detects: GPU if any visible to JAX,
    else CPU.
    """
    if compute is not None:
        if compute not in {"cpu", "gpu"}:
            raise ValueError(
                f"`compute` must be 'cpu', 'gpu', or None, got {compute!r}"
            )
        return compute

    try:
        if _jax_devices("gpu"):
            return "gpu"
    except RuntimeError:
        pass
    return "cpu"


def configure_devices(
    compute: Optional[str] = None,
    device_id: int = 0,
    *,
    dlm: Optional[NamedSharding] = None,
    alloc: Optional[NamedSharding] = None,
    host: Optional[NamedSharding] = None,
) -> None:
    """Configure the module-level DLMAX shardings.

    Parameters
    ----------
    compute : {'cpu', 'gpu', None}, default None
        Target platform for DLM and allocator compute. ``None`` selects
        GPU if any is visible to JAX, otherwise CPU. Use ``'cpu'``
        explicitly to force CPU even when a GPU is present (e.g. for
        fair timing comparisons against CPU-only baselines).
    device_id : int, default 0
        GPU index when ``compute='gpu'``. Ignored on CPU. DLMAX does
        not currently shard across multiple GPUs by default; multi-GPU
        users should construct their own meshes and pass them via the
        ``dlm`` / ``alloc`` / ``host`` keyword arguments.
    dlm, alloc, host : NamedSharding, optional
        Custom shardings that override the auto-constructed defaults
        for the DLM compute, allocator compute, and host shardings
        respectively. Useful for advanced sharding strategies such as
        custom mesh topologies or multi-device replication.

    Notes
    -----
    Sets module-level globals ``dlm_compute``, ``allocation_compute``,
    and ``host_device``. Internal DLMAX code refers to these via
    attribute access on this module (e.g. ``devices.dlm_compute``) so
    that updates from later ``configure_devices`` calls are seen.

    The meshes are created with ``axis_types=(AxisType.Auto,)`` so
    that JAX can infer output shardings for reshapes and contractions
    — without this, recent JAX versions reject many operations DLMAX
    relies on.
    """
    global dlm_compute, allocation_compute, host_device

    resolved = _resolve_compute(compute)

    if resolved == "gpu":
        try:
            available = _jax_devices("gpu")
        except RuntimeError:
            raise ValueError("No GPU device found")
        if not available:
            raise ValueError("No GPU device found")
        if device_id >= len(available):
            raise ValueError(
                f"GPU device id {device_id} exceeds available GPUs "
                f"({len(available)})"
            )
        compute_devices = [available[device_id]]
    else:  # 'cpu'
        compute_devices = _jax_devices("cpu")

    auto = (AxisType.Auto,)

    # Build defaults for any sharding not overridden by kwargs.
    if dlm is None:
        mesh_dlm = make_mesh(
            (1,), ("dlm_compute",), axis_types=auto, devices=compute_devices
        )
        dlm = NamedSharding(mesh_dlm, P("dlm_compute"))

    if alloc is None:
        mesh_alloc = make_mesh(
            (1,), ("allocation_compute",), axis_types=auto, devices=compute_devices
        )
        alloc = NamedSharding(mesh_alloc, P("allocation_compute"))

    if host is None:
        mesh_host = make_mesh(
            (1,), ("host",), axis_types=auto, devices=_jax_devices("cpu")
        )
        host = NamedSharding(mesh_host, P("host"))

    dlm_compute = dlm
    allocation_compute = alloc
    host_device = host


# Set sensible defaults at module import time. Auto-detects GPU vs CPU.
configure_devices()

"""Plain ``evoxc`` plumbing shared by the neuroevolution graph modules.

This module holds no graph and no task constant.  It is the backend/device
resolution, the host<->device staging, the payload-leaf helpers and the
``GenomeTensors`` operand names that ``evolution_graphs`` and ``policy_graphs``
both build on, so the two graph modules only speak about their own contract.

Nothing here imports ``torch``, ``genesis`` or ``tensordict``; the only imports
are ``numpy`` and ``evoxc``.

Device handling
---------------
``resolve_backend_device()`` honors explicit ``backend``/``device`` arguments,
then the ``EVOXC_BACKEND`` / ``EVOXC_DEVICE`` environment variables, and
defaults to the ``numpy`` backend on ``cpu:0``.  Selecting a CUDA or ROCm
target is therefore pure configuration; a non-CPU device also needs a compiler
backend such as ``iree`` or ``xla``.

Staging
-------
``_to_device_array()`` moves a host array onto the executable's device and is a
no-op on CPU, and ``_to_numpy()`` brings an output tensor back to the host.
``_staged()`` additionally checks a runtime argument against the row (and
optionally column) count the graph was compiled for.
"""

from __future__ import annotations

import os

import numpy as np

import evoxc

# --- Backend / device resolution ----------------------------------------------------


def _parse_device(spec) -> evoxc.Device:
    """Parse ``"kind[:index]"`` (or pass a :class:`evoxc.Device` through)."""
    if isinstance(spec, evoxc.Device):
        return spec
    if isinstance(spec, str):
        kind, _, index = spec.partition(":")
        kind = kind.strip()
        if not kind:
            raise ValueError(f"malformed device spec {spec!r}")
        return evoxc.Device(kind, int(index) if index.strip() else 0)
    raise TypeError(f"device must be a str or evoxc.Device, got {type(spec)!r}")


def resolve_backend_device(backend=None, device=None):
    """Resolve ``(backend, device)``: explicit args > env vars > cpu/numpy default."""
    resolved_backend = backend or os.environ.get("EVOXC_BACKEND") or "numpy"

    if device is None:
        env_device = os.environ.get("EVOXC_DEVICE", "").strip()
        resolved_device = _parse_device(env_device) if env_device else evoxc.Device("cpu", 0)
    else:
        resolved_device = _parse_device(device)
    return resolved_backend, resolved_device


# --- Graph constants ----------------------------------------------------------------


def _constant(array: np.ndarray) -> evoxc.SymbolicTensor:
    """Embed a host numpy array as a graph constant."""
    return evoxc.constant(evoxc.from_numpy(np.ascontiguousarray(array)))


# --- Host <-> device staging --------------------------------------------------------


def _to_device_array(executable, array: np.ndarray):
    """Stage a host array onto the executable's device (no-op for cpu).

    Re-selects the executable's own backend first, so staging stays correct when several
    backends serve the same device kind in one process (mixed-backend staging).
    """
    arr = np.asarray(array)
    if arr.ndim > 0:
        arr = np.ascontiguousarray(arr)
    device = getattr(executable, "device", None)
    if device is not None and device.kind != "cpu":
        backend = getattr(executable, "backend", None)
        if backend:
            # IREE and XLA can both serve cuda/rocm in one process; Tensor.to
            # resolves the transfer provider of the last looked-up backend, so
            # re-select this executable's backend before staging its inputs.
            evoxc.backends.get(backend)
        return evoxc.from_numpy(arr).to(device)
    return arr


def _to_numpy(tensor) -> np.ndarray:
    """Materialize an output tensor as a host numpy array."""
    device = getattr(tensor, "device", None)
    if device is not None and device.kind != "cpu":
        tensor = tensor.to(evoxc.Device("cpu", 0))
    return tensor.numpy()


# --- Executable metadata ------------------------------------------------------------


def _device_name(device) -> str:
    """``"kind:index"`` of a resolved device, for cache keys and log lines."""
    return f"{device.kind}:{device.index}"


def _attach(executable, **metadata):
    """Stash build-time metadata on an ``evoxc`` executable (it has a ``__dict__``).

    ``Executable.backend`` / ``.device`` / ``.functions`` are read-only properties, so
    the metadata names deliberately avoid them.
    """
    for name, value in metadata.items():
        setattr(executable, name, value)
    return executable


def _positive_int(value, name: str) -> int:
    """Validate a strict positive integer build argument."""
    ivalue = int(value)
    if ivalue != value or ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return ivalue


def _rows_of(executable) -> int:
    """The row count a graph was compiled for."""
    rows = getattr(executable, "rows", None)
    if rows is None:
        raise TypeError(f"{executable!r} was not built by this module (no '.rows' metadata)")
    return int(rows)


def _staged(executable, name: str, value, width=None, *, dtype=np.float32) -> np.ndarray:
    """Stage one runtime argument, checking its shape against the compiled graph."""
    rows = _rows_of(executable)
    expected = (rows,) if width is None else (rows, width)
    arr = np.asarray(value, dtype=dtype)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected} to match the compiled graph, got {arr.shape}")
    return np.ascontiguousarray(arr)


# --- Genome payload leaves ----------------------------------------------------------

#: ``(name, ...)`` of the four ``tensorneat.genome.utils.GenomeTensors`` payload leaves.
POP_OPERANDS = ("node_keys", "node_attrs", "conn_keys", "conn_attrs")


def _host(value) -> np.ndarray:
    """Materialize one payload leaf as a host numpy array.

    ``state.pop`` / ``evoxc.tree_leaves(state.species)`` hand back ``evoxc.Tensor``
    leaves, and ``np.asarray`` on those yields a 0-d object array, so tensors are
    routed through ``_to_numpy`` instead.
    """
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "numpy"):
        arr = _to_numpy(value)
    else:
        arr = np.asarray(value)
    if arr.dtype == object:
        raise TypeError(
            "payload leaf is a 0-d object array; materialize an evoxc tensor with `.numpy()` "
            "(see evox_runtime._to_numpy / evox_runtime._host) before passing it in"
        )
    return arr


def _outputs(result) -> tuple:
    """Normalize a ``evoxc.run`` result: a single-leaf graph returns the tensor bare."""
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def _contiguous(array: np.ndarray) -> np.ndarray:
    """``np.ascontiguousarray`` that leaves 0-d leaves 0-d (it would promote to ``(1,)``)."""
    return array if array.ndim == 0 else np.ascontiguousarray(array)


def _spec_for(array, name: str) -> evoxc.TensorSpec:
    """A ``TensorSpec`` matching one payload leaf (shape + dtype + name)."""
    arr = _host(array)
    return evoxc.TensorSpec(tuple(int(dim) for dim in arr.shape), evoxc.dtype(arr.dtype), name=name)


def _pop_specs(pop_arrays):
    """``TensorSpec`` list for the four ``GenomeTensors`` payload arrays."""
    arrays = tuple(pop_arrays)
    if len(arrays) != len(POP_OPERANDS):
        raise ValueError(f"pop_arrays must hold the {len(POP_OPERANDS)} GenomeTensors payloads, got {len(arrays)}")
    specs = []
    for array, name in zip(arrays, POP_OPERANDS, strict=True):
        arr = _host(array)
        specs.append(evoxc.TensorSpec(tuple(int(dim) for dim in arr.shape), evoxc.dtype(arr.dtype), name=name))
    return specs

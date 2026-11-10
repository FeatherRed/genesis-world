"""Torch-free helpers shared by the entity layer: row selection, index collections and host copies.

The entity layer must stay importable and usable where PyTorch is not installed: entities then read solver state
through torch-free device views and hand index collections back to the solvers as contiguous NumPy arrays. Every
helper below keeps the exact PyTorch behaviour when PyTorch is installed, so a call site can use it unconditionally.

Importing this module touches PyTorch only through 'genesis.utils.torch_compat', and it imports the device-buffer
module lazily, so that the entity layer keeps importing before Quadrants is initialized.
"""

from __future__ import annotations

import numpy as np

from genesis.utils.torch_compat import HAS_TORCH, require_torch, torch

__all__ = ["as_device_index", "as_index_array", "first_row", "is_device_view", "stack_values", "to_numpy"]

# Host dtype of an index collection: 'gs.tc_int' is 32-bit wide whatever the simulation precision.
_INDEX_DTYPE = np.int32


def is_device_view(value) -> bool:
    """Whether 'value' is a torch-free device view, which is the zero-copy read of a solver buffer."""
    from genesis.utils.device_buffer import DeviceView  # local import: this module loads before Quadrants is initialized

    return isinstance(value, DeviceView)


def first_row(value):
    """Torch-free equivalent of 'value[..., 0, :]', for a device view or a NumPy array of at least two axes."""
    if HAS_TORCH and isinstance(value, torch.Tensor):
        return value[..., 0, :]
    ndim = len(value.shape)
    if ndim < 2:
        import genesis as gs  # local import: 'genesis' imports this module while it is loading

        gs.raise_exception(f"Expecting at least two dimensions to select the first row, got shape '{value.shape}'.")
    return value[(slice(None),) * (ndim - 2) + (0, slice(None))]


def as_index_array(values):
    """Contiguous 'int32' host array of an index collection, which the solvers accept without PyTorch."""
    if is_device_view(values) or (HAS_TORCH and isinstance(values, torch.Tensor)):
        values = to_numpy(values)
    return np.ascontiguousarray(values, dtype=_INDEX_DTYPE)


def as_device_index(values):
    """Index collection as the tensor or array that the solver setters and getters accept."""
    if HAS_TORCH:
        import genesis as gs  # local import: 'genesis' imports this module while it is loading

        return torch.as_tensor(values, dtype=gs.tc_int, device=gs.device).contiguous()
    return as_index_array(values)


def to_numpy(value):
    """Best-effort host copy of a device view, a PyTorch tensor or an array; never call it on a hot path."""
    if HAS_TORCH and isinstance(value, torch.Tensor):
        from genesis.utils.misc import tensor_to_array  # local import: 'misc' imports PyTorch at module scope

        return np.asarray(tensor_to_array(value))
    if is_device_view(value):
        return np.asarray(value.to_numpy())
    return np.asarray(value)


def stack_values(values, dim=0):
    """Stack the values along 'dim', with PyTorch when it is installed and with NumPy otherwise."""
    values = list(values)
    if HAS_TORCH:
        return torch.stack(values, dim=dim)
    if not all(isinstance(value, np.ndarray) for value in values):
        require_torch("genesis.engine.entities.torch_free.stack_values")
    return np.stack(values, axis=dim)

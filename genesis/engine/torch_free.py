"""Torch-free helpers shared by the scene and the simulator: environment indices, step counters and error masks.

The scene and the simulator must stay usable where PyTorch is not installed: the counter of the steps every environment
ran then lives in a Quadrants field, the error flags of the solvers are merged into one device-resident buffer, and the
selection of the environments stays on the host as a contiguous 'int32' index array.
Every helper below keeps the exact PyTorch behaviour when PyTorch is installed, so a call site can use it
unconditionally.

Importing this module touches PyTorch only through 'genesis.utils.torch_compat', and it imports the device-buffer and
solver helpers lazily, so that the scene layer keeps importing before Quadrants is initialized.
"""

from __future__ import annotations

import numpy as np
import quadrants as qd

import genesis as gs

__all__ = [
    "INDEX_DTYPE",
    "envs_index",
    "envs_index_field",
    "kernel_clear_counters",
    "kernel_increment_counters",
    "kernel_or_mask",
    "kernel_zero_mask",
    "mask_field",
    "mask_indices",
    "steps_field",
    "steps_from_values",
    "steps_values",
]

# Host dtype of the index collections and of the step counters: 'gs.tc_int' is 32-bit wide whatever the simulation
# precision.
INDEX_DTYPE = np.int32


# ------------------------------------------------------------------------------------------------
# Environment indices
# ------------------------------------------------------------------------------------------------


def envs_index(envs_idx, n_envs: int) -> np.ndarray:
    """Selection of the environments of a call, as the contiguous 'int32' host array a torch-free scene works with.

    The accepted forms and the errors raised are those of 'sanitize_index', whose numpy counterpart this is (see
    'genesis.engine.solvers.torch_free').
    """
    # Local import: the solver helpers load after this module, which the scene layer imports first
    from genesis.engine.solvers.torch_free import sanitize_index_values

    return sanitize_index_values(envs_idx, -1, n_envs, 0, "envs_idx")


def mask_indices(mask) -> np.ndarray:
    """Indices of the environments a mask flags, which a torch-free scene gives as a device view or a host array."""
    from genesis.utils.device_buffer import DeviceView  # local: this module loads before Quadrants

    if isinstance(mask, DeviceView):
        mask = mask.to_numpy()
    return np.ascontiguousarray(np.nonzero(np.asarray(mask).reshape(-1))[0], dtype=INDEX_DTYPE)


def envs_index_field(envs_idx, n_envs: int, name: str = "envs_idx"):
    """Selection of the environments of a call, uploaded as the device tensor a kernel iterates over."""
    # Local import: the solver helpers load after this module, which the scene layer imports first
    from genesis.engine.solvers.torch_free import index_field

    return index_field(envs_idx, n_envs, name)


# ------------------------------------------------------------------------------------------------
# Step counters
# ------------------------------------------------------------------------------------------------


def steps_field(n_envs: int):
    """Allocate the counter of the steps every environment ran since its last reset, which Quadrants zeroes."""
    return qd.field(gs.qd_int, shape=(n_envs,))


def mask_field(n_envs: int):
    """Allocate the mask of the environments that flagged an error, which Quadrants zeroes, as the solvers keep one.

    The buffer holds one boolean flag per environment, so that the mask a torch-free scene returns has the dtype and the
    values of the torch one. Apple Metal stores 'gs.qd_bool' as 'int32', which is why every write casts explicitly.
    """
    return qd.field(gs.qd_bool, shape=(n_envs,))


def steps_values(steps) -> np.ndarray:
    """Host copy of the step counters, which a torch-free scene reads where the torch one keeps a tensor."""
    return np.asarray(steps.to_numpy(), dtype=INDEX_DTYPE)


def steps_from_values(steps, values) -> None:
    """Write the step counters back from a host array, as a restore of a checkpoint does."""
    steps.from_numpy(np.ascontiguousarray(values, dtype=INDEX_DTYPE))


@qd.kernel
def kernel_increment_counters(counters: qd.Tensor):
    """Add one step to every environment."""
    for i_b in range(counters.shape[0]):
        counters[i_b] += 1


@qd.kernel
def kernel_clear_counters(counters: qd.Tensor, envs_idx: qd.Tensor):
    """Reset the counter of the given environments."""
    for i_b_ in range(envs_idx.shape[0]):
        counters[envs_idx[i_b_]] = 0


# ------------------------------------------------------------------------------------------------
# Error masks
# ------------------------------------------------------------------------------------------------


@qd.kernel
def kernel_zero_mask(mask: qd.Tensor):
    """Clear the mask of the environments that flagged an error."""
    for i_b in range(mask.shape[0]):
        mask[i_b] = qd.cast(False, gs.qd_bool)


@qd.kernel
def kernel_or_mask(mask: qd.Tensor, solver_mask: qd.Tensor):
    """Add the error flags of one solver to the mask of the whole simulator, in place on the device.

    Both buffers hold one boolean flag per environment, which are compared against 0 before their disjunction and cast
    back, so that the kernel compiles whatever 'gs.qd_bool' is stored as (Apple Metal stores it as 'int32').
    """
    for i_b in range(mask.shape[0]):
        mask[i_b] = qd.cast((mask[i_b] != 0) | (solver_mask[i_b] != 0), gs.qd_bool)

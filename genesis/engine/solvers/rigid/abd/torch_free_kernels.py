"""Quadrants kernels the torch-free write path of the articulated state accessors needs.

PyTorch hands a setter a tensor view of the state it overwrites in place, so a write covering part of a row patches that
row where it lies.
Without it the row is rebuilt on the device instead, which the two kernels below do (see
'KinematicSolver.set_dofs_position').
Nothing here imports PyTorch: the buffers are plain kernel arguments, holding the environment axis first or last as each
kernel states, and the state is the usual 'DynState' struct.
"""

import quadrants as qd

from genesis.engine.solvers.rigid.data import DynState
from genesis.utils.data import eoa

__all__ = [
    "kernel_copy_dofs_pos",
    "kernel_scatter_dofs_pos",
]


@qd.kernel(fastcache=True)
def kernel_copy_dofs_pos(out: qd.types.ndarray(), dyn_state: DynState):
    """Copy the dof positions of every environment into `out`, a batch-first buffer of shape (n_envs, n_dofs).

    This kernel exists only on the torch-free write path, where a setter rebuilds the whole row of dof positions it
    patches instead of overwriting a tensor view of the state.
    The environments are the last axis of the state and the first axis of `out`.
    Every entry of `out` is written once and read by one thread, so the loops of the accessor kernels need no
    serialization on top of the memory-order axes.
    """
    for i_b, i_d in qd.ndrange(out.shape[0], out.shape[1], axes=qd.static(eoa(out))):
        out[i_b, i_d] = dyn_state.dofs.pos[i_d, i_b]


@qd.kernel(fastcache=True)
def kernel_scatter_dofs_pos(
    out: qd.types.ndarray(),
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
):
    """Write `values` into the given dofs of the given environments, `out` being batch-first and `values` batch-last.

    This kernel exists only on the torch-free write path, where the values a setter writes are device buffers rather
    than the rows of a tensor view of the state.
    `out` holds one row per environment of the solver and `values` one column per environment, so the value written to
    the dof `dofs_idx[i_d_]` of the environment `envs_idx[i_b_]` is `values[i_d_, i_b_]`.
    Every entry selected is written once and read by one thread, so the loops of the accessor kernels need no
    serialization on top of the memory-order axes.
    """
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(values))):
        out[envs_idx[i_b_], dofs_idx[i_d_]] = values[i_d_, i_b_]

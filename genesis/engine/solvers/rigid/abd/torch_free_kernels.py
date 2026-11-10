"""Quadrants kernels the torch-free write path of the articulated state accessors needs.

PyTorch hands a setter a tensor view of the state it overwrites in place, so a write covering part of a row patches that
row where it lies.
Without it the row is rebuilt on the device instead, which the two kernels below do (see
'KinematicSolver.set_dofs_position').
A DLPack producer cannot be written into a field either, whose batch-last view is not contiguous, so the staged kernels
below read the batch-last staging buffer of a 'DlpackStager' instead, one column per environment (see
'KinematicSolver.set_qpos').
The 'kernel_set_*'/'kernel_control_*' kernels of 'accessor.py' cannot read that buffer, since they index their values
batch-first ('values[i_b, i_d]') while the staged buffer holds the batch axis last; the staged kernels below mirror them
entry for entry instead, and reproduce their result bit-exactly.
A qpos write stages its producer batch-first instead, and 'kernel_set_qpos_rows_staged' picks the row of each
environment, row 0, its row in the selection, or the row of the environment itself (see 'RigidSolver.set_qpos').
A drone keeps the accumulated revolutions of its propellers in a device buffer in a torch-free scene, and its RPM values
reach the device staged batch-first, so the two kernels below advance that accumulator and expand a single row of RPM to
every environment (see 'RigidSolver.set_drone_propellers_rpm').
Nothing here imports PyTorch: the buffers are plain kernel arguments, holding the environment axis first or last as each
kernel states, and the state is the usual 'DynState' struct.
"""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.data import DynState, RigidInfo
from genesis.utils.data import eoa

__all__ = [
    "kernel_broadcast_drone_rpm",
    "kernel_control_dofs_position_velocity_staged",
    "kernel_control_dofs_staged",
    "kernel_copy_dofs_pos",
    "kernel_scatter_dofs_pos",
    "kernel_set_dofs_position_staged",
    "kernel_set_dofs_velocity_staged",
    "kernel_set_qpos_rows_staged",
    "kernel_set_qpos_staged",
    "kernel_update_drone_propeller_revs",
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


# ------------------------------------------------------------------------------------------------
# Staged writes of a DLPack producer
# ------------------------------------------------------------------------------------------------
# The kernels below exist only on the torch-free staged-write path: a DLPack producer is first copied, device to device,
# into the batch-last staging buffer of a 'DlpackStager', which holds one column per environment of the solver, and the
# value written to the entry `idx[i_]` of the environment `envs_idx[i_b_]` is then `values[i_, envs_idx[i_b_]]`, the
# index being the position of the entry in the selection rather than the entry itself, as a host write reads it.
# The staged buffer is not a field, so these kernels take it as the plain ndarray it is.


@qd.kernel(fastcache=True)
def kernel_set_qpos_staged(
    qs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
):
    """Write the staged qpos values into the given qpos of the given environments.

    The kernel is the counterpart of 'kernel_set_qpos' for a staged buffer, which holds one column per environment of
    the solver rather than one row of the values written, and one row per qpos of the selection rather than per qpos.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(rigid_info.qpos))):
        rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = values[i_q_, envs_idx[i_b_]]


@qd.kernel(fastcache=True)
def kernel_set_qpos_rows_staged(
    qs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    rigid_info: RigidInfo,
    rigid_config: qd.template(),
    is_broadcast: qd.template(),
    is_tile: qd.template(),
    is_row: qd.template(),
):
    """Write the staged qpos values into the given qpos of the given environments, `values` being batch-first.

    The kernel mirrors 'kernel_set_qpos' for a staged buffer, which holds one of four forms of rows of values:
    a bare element row of shape (len(qs_idx),), which every selected environment takes;
    a single row of shape (1, len(qs_idx)), which every selected environment takes as well;
    one row per environment named by `envs_idx`, in its order, whose value written to the qpos `qs_idx[i_q_]` of the
    environment `envs_idx[i_b_]` is `values[i_b_, i_q_]`;
    and the whole tile of the solver, which holds one row per environment, each selected environment taking the row of
    its own index, `values[envs_idx[i_b_], i_q_]`, the rows of the environments it leaves out being ignored.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_q_, i_b_ in qd.ndrange(qs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(rigid_info.qpos))):
        # A producer of a single dimension holds the element itself, which every selected environment takes
        if qd.static(is_row):
            rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = values[i_q_]
        else:
            # The row a selected environment takes is its own row of the whole tile, row 0 of a broadcast producer, and
            # its position in the selection otherwise
            i_b_values = envs_idx[i_b_] if qd.static(is_tile) else (0 if qd.static(is_broadcast) else i_b_)
            rigid_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = values[i_b_values, i_q_]


@qd.kernel(fastcache=True)
def kernel_set_dofs_velocity_staged(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    dyn_state: DynState,
    rigid_config: qd.template(),
):
    """Write the staged dof velocities into the given dofs of the given environments.

    The kernel is the counterpart of 'kernel_set_dofs_velocity' for a staged buffer, which holds one column per
    environment of the solver rather than one row of the values written, and one row per dof of the selection rather than
    per dof.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(dyn_state.dofs.vel))):
        i_b = envs_idx[i_b_]
        dyn_state.dofs.vel[dofs_idx[i_d_], i_b] = values[i_d_, i_b]


@qd.kernel(fastcache=True)
def kernel_set_dofs_position_staged(
    out: qd.types.ndarray(),
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
):
    """Write the staged dof positions into the given dofs of the given environments, `out` being batch-first.

    The kernel fills the batch-first scratch buffer that 'kernel_copy_dofs_pos' has pre-filled with the current
    positions, before 'kernel_set_dofs_position' derives the configuration of the joints from it.
    """
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(values))):
        out[envs_idx[i_b_], dofs_idx[i_d_]] = values[i_d_, envs_idx[i_b_]]


@qd.kernel(fastcache=True)
def kernel_control_dofs_staged(
    mode: qd.template(),
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    values: qd.types.ndarray(),
    dyn_state: DynState,
    rigid_config: qd.template(),
):
    """Write the staged control value of the given dofs of the given environments, in the given control mode.

    The kernel mirrors 'kernel_control_dofs_force', 'kernel_control_dofs_velocity' and
    'kernel_control_dofs_position' for a staged buffer, which holds one column per environment of the solver and one
    row per dof of the selection rather than per dof.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(dyn_state.dofs.ctrl_mode))):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        if qd.static(mode == gs.CTRL_MODE.FORCE):
            dyn_state.dofs.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.FORCE
            dyn_state.dofs.ctrl_force[i_d, i_b] = values[i_d_, i_b]
        if qd.static(mode == gs.CTRL_MODE.VELOCITY):
            dyn_state.dofs.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.VELOCITY
            dyn_state.dofs.ctrl_vel[i_d, i_b] = values[i_d_, i_b]
        if qd.static(mode == gs.CTRL_MODE.POSITION):
            dyn_state.dofs.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
            dyn_state.dofs.ctrl_pos[i_d, i_b] = values[i_d_, i_b]
            dyn_state.dofs.ctrl_vel[i_d, i_b] = 0.0


@qd.kernel(fastcache=True)
def kernel_control_dofs_position_velocity_staged(
    dofs_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    pos: qd.types.ndarray(),
    vel: qd.types.ndarray(),
    dyn_state: DynState,
    rigid_config: qd.template(),
):
    """Write the staged control position and velocity of the given dofs of the given environments.

    The kernel mirrors 'kernel_control_dofs_position_velocity' for two staged buffers, which each hold one column per
    environment of the solver and one row per dof of the selection rather than per dof.
    The two buffers must be distinct: a 'DlpackStager' reuses one buffer per shape, so staging both targets into the
    same stager would leave both arguments holding the values staged last (see 'RigidSolver._stage_values').
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d_, i_b_ in qd.ndrange(dofs_idx.shape[0], envs_idx.shape[0], axes=qd.static(eoa(dyn_state.dofs.ctrl_mode))):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dyn_state.dofs.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
        dyn_state.dofs.ctrl_pos[i_d, i_b] = pos[i_d_, i_b]
        dyn_state.dofs.ctrl_vel[i_d, i_b] = vel[i_d_, i_b]


# ------------------------------------------------------------------------------------------------
# Drone propellers
# ------------------------------------------------------------------------------------------------
# A drone keeps the accumulated revolutions of its propellers in a buffer of its own, held batch-last like the state,
# which a torch-free scene stores on the device rather than on the host: the kernels below advance it from the RPM
# values a producer staged batch-first, and expand a single row of RPM to every environment of the scene.


@qd.kernel(fastcache=True)
def kernel_update_drone_propeller_revs(
    propellers_revs: qd.types.ndarray(), propellers_rpm: qd.types.ndarray(), revs_limit: float
):
    """Advance the accumulated revolutions of every propeller of every environment by its RPM.

    `propellers_revs` is the accumulator, batch-last of shape (n_propellers, n_envs), and `propellers_rpm` the RPM
    values staged batch-first, of shape (n_envs, n_propellers), which is the layout 'kernel_set_drone_rpm' reads.
    The value the accumulator wraps around to is `revs_limit`, the revolutions one full turn of a propeller takes,
    as the torch path computes it on the host (`60 / dt`).
    Every entry of the accumulator is read and written by exactly one thread, so the loop needs no serialization on top
    of the memory-order axes.
    """
    for i_pp, i_b in qd.ndrange(
        propellers_revs.shape[0], propellers_revs.shape[1], axes=qd.static(eoa(propellers_revs))
    ):
        propellers_revs[i_pp, i_b] = (propellers_revs[i_pp, i_b] + propellers_rpm[i_b, i_pp]) % revs_limit


@qd.kernel(fastcache=True)
def kernel_broadcast_drone_rpm(row: qd.types.ndarray(), out: qd.types.ndarray(), is_row: qd.template()):
    """Write the RPM values of a single row to every environment of `out`, a batch-first buffer (n_envs, n_propellers).

    The row is read as a bare row of `n_propellers` values when `is_row`, and as a single-row tile otherwise, which is
    the shape a producer holding one row comes in: `(n_propellers,)` or `(1, n_propellers)` (see
    'RigidSolver.set_drone_propellers_rpm').
    Every entry of `out` is written once and read by one thread, so the loop needs no serialization on top of the
    memory-order axes.
    """
    for i_b, i_pp in qd.ndrange(out.shape[0], out.shape[1], axes=qd.static(eoa(out))):
        if qd.static(is_row):
            out[i_b, i_pp] = row[i_pp]
        else:
            out[i_b, i_pp] = row[0, i_pp]

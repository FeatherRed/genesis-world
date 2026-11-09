"""Device-resident (zero-copy) rollout of one generation of the Go2 TensorNEAT trainer.

This module is the opt-in alternative to the host-staged loop of ``Go2NeatEnv.evaluate``: the same
generation, the same graphs, the same resets and the same bookkeeping, but every value that crosses a
graph boundary stays on the accelerator.  ``Go2NeatEnv.evaluate`` dispatches here when the env was
built with ``residency="device"``; the numpy path stays the reference implementation, so a difference
between the two residencies can only come from this bridge.  Device residency needs a non-CPU
``evoxc`` device (the env refuses to be built otherwise) and imports IREE lazily.

Interop contract (torch / Genesis <-> IREE on one ROCm or CUDA device)
---------------------------------------------------------------------
``evoxc.run`` enforces a strict run boundary: every tensor leaf must ALREADY live on the executable's
device, and a non-CPU executable rejects host (numpy) inputs.  The bridge only ever hands device
buffers over, through DLPack in both directions:

* torch device tensor -> graph input (``_operand``): ``capsule = tensor.__dlpack__()``, then
  ``hal.from_dlpack_capsule(capsule)`` wraps the *same* memory as an IREE HAL buffer view, and
  ``evoxc.core.Tensor(IreeDevicePayload(rt.DeviceArray(hal, view), executable.device))`` is the
  operand -- the call form of ``evoxc.backends.adapters.iree._device_array_from_dlpack``.  The tensor
  must be contiguous (``torch.__dlpack__`` exports its buffer as compact) and on that device.
* graph output -> torch (``_torch``): an output tensor's payload is an ``IreeDevicePayload`` whose
  ``__dlpack__`` exports the IREE device buffer, so ``torch.from_dlpack(tensor)`` aliases IREE's
  memory (used for the raw action).  The ``evoxc`` tensor stays referenced until torch consumed it.
* graph output -> graph input: an executable's output is passed straight to the next one (the
  observation fed to the policy graph, and ``returns`` / ``alive`` / ``episode_length`` fed back to
  the advance graph); ``IreeExecutable.run`` passes a same-device ``IreeDevicePayload`` through with
  no staging, so those values never reach torch at all.
* ``HalDevice``: taken from the first device payload this module sees
  (``tensor.data.device_array._device``), i.e. from the operands staged at the start of a generation;
  the adapter keeps one HAL device per ``core.Device``, so it is the executables' runtime device.
* bool / int32 leaves cross like floats: torch's ``kDLBool`` / ``kDLInt`` capsules import as IREE
  ``i1`` / ``i32`` buffers and export back as ``torch.bool`` / ``torch.int32`` (verified on this
  host), so no graph dtype override and no ``payload.to_host()`` fallback is needed for them.

Synchronization: IREE's HIP queue is *not* torch's stream and the DLPack import adds no fence, so
this module fences explicitly -- twice per step and once per generation:

* ``_sync()`` after the state read and before the advance graph imports the buffers Genesis just wrote
  (torch/quadrants -> IREE);
* ``_sync()`` right after the policy graph returned and before torch reads the action buffer it just
  wrote (IREE -> torch);
* ``_sync()`` once per generation, after the reset writes: the reset qpos tile and the PD targets are
  torch tensors the graphs import by DLPack, so the generation's first reads must not run ahead.

Nothing else needs a fence: ``IreeExecutable.run`` blocks on its own queue before returning, so an
exported output is complete by the time it is handed on.

What stays on the device
------------------------
Observations, raw and clipped actions, ``last_actions``, ``returns``, ``alive``, ``episode_length``,
the PD targets, the per-generation init-qpos tile and the reset rows gathered from it.  The state is
read through zero-copy views of the LIVE solver buffers (``qd_to_torch(..., copy=False)``), not
through the cloning getters:

* ``solver.rigid_info.qpos`` -- batch dim last, ``[x, y, z, qw, qx, qy, qz, dof0..dof11]``, the same
  layout as ``env.init_qpos`` (column 7 onwards are dofs);
* ``solver.dyn_state.dofs.pos`` / ``.vel`` -- the 12 actuated dofs of ``env._control_dof_slice``;
* ``robot.get_vel()`` / ``get_ang()`` -- kept as the getters: they compose the link-frame velocity
  ``cd_vel + cd_ang x (pos - root_COM)`` (plus the authored-origin offset) for ONE link row, entirely
  on the device; re-deriving that here would duplicate a Genesis convention for no traffic.

Host touches that remain (exhaustive)
-------------------------------------
Per step:

* ``done`` (``rows`` bytes) -- ``payload.to_host()``, for ``np.flatnonzero`` (the reset rows), the
  ``end_reasons`` classification and the host ``alive`` mirror;
* ``error_mask.any()`` (1 byte) -- decides ``scene.clear_errno()``, exactly the numpy path's test;
* on a step where a row ended: the ``base_rpy_deg`` rows and the ``error_mask`` vector (a few hundred
  bytes) handed to the env's own ``_classify_endings``, so the rule has one owner.

Per generation: the ``returns`` (``rows`` floats) and ``fitness`` (``pop_size`` floats) readbacks (the
seeds and init qpos are the host arrays the env already computed), the pop payload (four leaves,
host -> device) staged once, and the ``(rows, 19)`` init-qpos tile.

The device copies per step are unavoidable but device-to-device (no host memory): the base pos / base
quat / dof pos / dof vel blocks are ``.contiguous()`` / ``index_select`` materializations of
transposed views, plus the joint-order gather of the PD targets and the ``torch.clamp`` of the action.

Parity
------
The bridge changes no ordering and no arithmetic: the same graphs run on the same values in the same
sequence, and both residencies produce identical seeds, episode returns, fitness, step counts and
``end_reasons``.
"""

from __future__ import annotations

import time

import numpy as np
import torch

import genesis as gs

from genesis.utils.geom import inv_quat, quat_to_xyz, transform_quat_by_quat
from genesis.utils.misc import qd_to_torch

from go2_neat_kernels import (
    CLIP_ACTIONS,
    DEFAULT_COMMANDS,
    NUM_ACTIONS,
    _outputs,
    _tile_commands,
    _tile_scaled_commands,
    _tile_standing,
)


def evaluate_device(env, *, executables, pop_arrays, seeds, init_qpos, row_init_qpos, horizon, stop_when_done):
    """Simulate one generation of ``env`` with every graph operand device-resident.

    Args:
        env (Go2NeatEnv): The environment whose scene, executables and static metadata are used.
        executables (tuple): ``(obs, policy, advance, fitness)`` executables, all on one device.
        pop_arrays (tuple): The four genome payload leaves of the generation (host numpy).
        seeds (np.ndarray): The generation's ordered episode seeds, passed through to the result.
        init_qpos (np.ndarray): ``(3, 19)`` perturbed init qpos of the generation, passed through.
        row_init_qpos (np.ndarray): ``(rows, 19)`` policy-major tiling of ``init_qpos``.
        horizon (int): ``scene.step()`` calls of the generation.
        stop_when_done (bool): Stop as soon as every row has ended.

    Returns:
        dict: The same mapping ``Go2NeatEnv.evaluate`` returns, so
        ``env_base.validate_evaluation`` and ``trainer.NeuroevolutionTrainer`` accept it unchanged.
    """
    rollout = _DeviceRollout(env, executables)
    return rollout.run(
        pop_arrays=pop_arrays,
        seeds=seeds,
        init_qpos=init_qpos,
        row_init_qpos=row_init_qpos,
        horizon=horizon,
        stop_when_done=stop_when_done,
    )


class _DeviceRollout:
    """The zero-copy bridge and one generation of the device-resident loop.

    One instance serves one generation: the cached solver views and the HAL device belong to the
    scene of that generation, and a reset writes the buffers in place, so the views stay valid for
    the whole loop.
    """

    def __init__(self, env, executables):
        self.env = env
        self.obs_executable, self.policy_executable, self.advance_executable, self.fitness_executable = executables
        self.rows = int(env.rows)
        self.device = self.policy_executable.device
        self.backend = self.policy_executable.backend
        for executable in executables:
            if executable.device != self.device:
                raise ValueError(f"every executable must live on {self.device}, got {executable.device}")
        if env.robot.n_qs != 7 + NUM_ACTIONS:
            raise ValueError(
                f"device residency assumes the Go2 free base (7 qs) plus {NUM_ACTIONS} dofs, "
                f"got {env.robot.n_qs} qs"
            )
        # Zero-valued / unit-valued device state the graphs consume at reset and at the first step.
        self._zeros = {
            "base_ang_vel_world": torch.zeros((self.rows, 3), dtype=gs.tc_float, device=gs.device),
            "base_quat": torch.zeros((self.rows, 4), dtype=gs.tc_float, device=gs.device),
            "dof_vel": torch.zeros((self.rows, NUM_ACTIONS), dtype=gs.tc_float, device=gs.device),
            "actions": torch.zeros((self.rows, NUM_ACTIONS), dtype=gs.tc_float, device=gs.device),
        }
        self._zeros["base_quat"][:, 0] = 1.0
        self._slot = torch.as_tensor(env._qpos_dof_slot, dtype=torch.int64, device=gs.device)
        self._actions_dof_idx = torch.as_tensor(env.actions_dof_idx, dtype=torch.int64, device=gs.device)
        self._default_dof_pos = torch.as_tensor(env.default_dof_pos, dtype=gs.tc_float, device=gs.device)
        self._inv_init_quat = None
        self._views = {}
        self._hal = None

    # --- DLPack bridge ---------------------------------------------------------------

    def _note_hal(self, tensor):
        """Remember the HAL device of a payload (the one place it is discovered)."""
        if self._hal is None:
            array = getattr(tensor.data, "device_array", None)
            if array is not None:
                self._hal = array._device

    def _operand(self, tensor):
        """Wrap a device torch tensor as an ``evoxc`` operand sharing its memory (zero-copy)."""
        import evoxc
        import evoxc.core
        import iree.runtime as rt
        from evoxc.backends.adapters.iree import IreeDevicePayload

        if self._hal is None:
            raise RuntimeError("the HAL device is not known yet; stage an operand or run a graph first")
        view = self._hal.from_dlpack_capsule(tensor.__dlpack__())
        return evoxc.core.Tensor(IreeDevicePayload(rt.DeviceArray(self._hal, view), self.device))

    def _torch(self, tensor):
        """View an ``evoxc`` output tensor as a torch tensor sharing its memory (zero-copy)."""
        return torch.from_dlpack(tensor)

    def _stage(self, array):
        """Copy one host array onto the device (the only host -> device staging of the loop)."""
        import evoxc

        # IREE and XLA can both serve rocm/cuda in one process, and a transfer resolves the provider
        # of the last backend looked up, so re-select this executable's backend before staging.
        evoxc.backends.get(self.backend)
        tensor = evoxc.from_numpy(array).to(self.device)
        self._note_hal(tensor)
        return tensor

    def _host(self, tensor):
        """Materialize one output on the host (the payload owns an explicit readback)."""
        return tensor.data.to_host()

    @staticmethod
    def _sync():
        """Fence the torch stream against IREE's queue (torch maps HIP on ROCm)."""
        torch.cuda.synchronize()

    # --- state ----------------------------------------------------------------------

    def _view(self, field):
        """Zero-copy batch-first view of a live solver buffer (cached for this generation)."""
        view = self._views.get(id(field))
        if view is None:
            view = qd_to_torch(field, transpose=True, copy=False)
            self._views[id(field)] = view
        return view

    def _read_state(self):
        """Read the stepped state as device tensors, without any host traffic.

        The two state blocks are zero-copy views of the live solver buffers; the four materialized
        blocks below are device copies of those views because a DLPack export must be contiguous and
        the dof block is stored in dof order while the graphs index it in joint order.
        """
        solver = self.env.robot._solver
        qpos = self._view(solver.rigid_info.qpos)
        dofs_pos = self._view(solver.dyn_state.dofs.pos)[:, self.env._control_dof_slice]
        dofs_vel = self._view(solver.dyn_state.dofs.vel)[:, self.env._control_dof_slice]
        return {
            "base_pos": qpos[:, :3].contiguous(),
            "base_quat": transform_quat_by_quat(self._inv_init_quat, qpos[:, 3:7].contiguous()),
            "base_lin_vel_world": self.env.robot.get_vel().contiguous(),
            "base_ang_vel_world": self.env.robot.get_ang().contiguous(),
            "dof_pos": dofs_pos.index_select(1, self._slot),
            "dof_vel": dofs_vel.index_select(1, self._slot),
            "error_mask": self.env.scene.get_error_envs_mask(),
        }

    # --- graphs ---------------------------------------------------------------------

    def _stage_pop(self, pop_arrays):
        """Stage the four genome payload leaves once per generation."""
        arrays = tuple(np.asarray(array) for array in pop_arrays)
        shapes = tuple(self.policy_executable.pop_shapes)
        if len(arrays) != len(shapes):
            raise ValueError(f"pop_arrays must hold the {len(shapes)} genome payload leaves, got {len(arrays)}")
        operands = []
        for array, shape in zip(arrays, shapes, strict=True):
            if array.shape != shape:
                raise ValueError(f"pop_arrays leaf must keep the built shape {shape}, got {array.shape}")
            operands.append(self._stage(np.ascontiguousarray(array)))
        return operands

    def _reset_obs(self, row_init_qpos):
        """The observation of a fresh reset, computed from the known reset values."""
        import evoxc

        dof_pos = row_init_qpos[:, 7:].index_select(1, self._slot)
        outputs = evoxc.run(
            self.obs_executable,
            self._operand(self._zeros["base_ang_vel_world"]),
            self._operand(self._zeros["base_quat"]),
            self._operand(dof_pos),
            self._operand(self._zeros["dof_vel"]),
            self._operand(self._zeros["actions"]),
            _tile_standing(self.obs_executable.standing_flat, self.rows),
            _tile_scaled_commands(DEFAULT_COMMANDS, self.rows),
        )
        obs = _outputs(outputs)[0]
        self._note_hal(obs)
        return obs

    def _run_policy(self, pop_operands, obs):
        """One policy forward pass; ``obs`` is the previous graph's output (no staging)."""
        import evoxc

        outputs = evoxc.run(self.policy_executable, *pop_operands, obs)
        actions = _outputs(outputs)[0]
        self._note_hal(actions)
        return actions

    def _run_advance(self, state, actions, last_actions, returns, alive, episode_length):
        """One advance step; mirrors ``go2_neat_kernels.run_advance`` operand for operand."""
        import evoxc

        executable = self.advance_executable
        outputs = evoxc.run(
            executable,
            self._operand(state["base_pos"]),
            self._operand(state["base_quat"]),
            self._operand(state["base_lin_vel_world"]),
            self._operand(state["base_ang_vel_world"]),
            self._operand(state["dof_pos"]),
            self._operand(state["dof_vel"]),
            actions,
            self._operand(last_actions),
            returns,
            alive,
            episode_length,
            self._operand(state["error_mask"]),
            _tile_standing(executable.standing_flat, self.rows),
            _tile_commands(DEFAULT_COMMANDS, self.rows),
            _tile_scaled_commands(DEFAULT_COMMANDS, self.rows),
            executable.max_episode_length,
            executable.dt,
            executable.tan_termination,
            CLIP_ACTIONS,
        )
        obs, _reward, returns, alive, done, episode_length = _outputs(outputs)
        self._note_hal(obs)
        return obs, returns, alive, done, episode_length

    def _fitness(self, returns):
        """The fitness graph on the device-resident returns of the generation."""
        import evoxc

        executable = self.fitness_executable
        outputs = evoxc.run(executable, returns, executable.n_policies, executable.episodes_per_policy)
        return np.asarray(_outputs(outputs)[0].data.to_host(), dtype=np.float32)

    # --- generation -----------------------------------------------------------------

    def run(self, *, pop_arrays, seeds, init_qpos, row_init_qpos, horizon, stop_when_done):
        """Simulate the generation and return its statistics mapping."""
        # Imported here because ``go2_neat_env`` imports this module lazily from ``evaluate``.
        from go2_neat_env import END_REASONS

        env = self.env
        rows = self.rows
        start = time.perf_counter()

        pop_operands = self._stage_pop(pop_arrays)
        row_init_qpos = torch.as_tensor(np.ascontiguousarray(row_init_qpos), dtype=gs.tc_float, device=gs.device)
        # The per-row init orientation the graphs' relative quaternion is measured from, owned by this
        # rollout: the env keeps its own host copy for the numpy path, which this bridge does not touch.
        self._inv_init_quat = inv_quat(row_init_qpos[:, 3:7].contiguous())
        env.robot.set_qpos(row_init_qpos, envs_idx=None, zero_velocity=True, skip_forward=True)
        env.scene.clear_errno()  # drop any flag the previous generation left behind
        self._sync()  # the reset landed in torch's stream; the first graph reads it next
        obs = self._reset_obs(row_init_qpos)

        returns = self._operand(torch.zeros(rows, dtype=gs.tc_float, device=gs.device))
        alive = self._operand(torch.ones(rows, dtype=torch.bool, device=gs.device))
        episode_length = self._operand(torch.zeros(rows, dtype=torch.int32, device=gs.device))
        last_actions = torch.zeros((rows, NUM_ACTIONS), dtype=gs.tc_float, device=gs.device)
        ending = np.full(rows, END_REASONS["unfinished"], dtype=np.int8)
        host_alive = np.ones(rows, dtype=bool)

        steps = 0
        while steps < horizon:
            actions = self._run_policy(pop_operands, obs)
            self._sync()  # IREE wrote the action buffer; torch reads it next
            targets = last_actions * env.action_scale + self._default_dof_pos
            env.robot.control_dofs_position(targets[:, self._actions_dof_idx], env._control_dof_slice)
            env.scene.step()
            steps += 1

            state = self._read_state()
            self._sync()  # Genesis wrote the state buffers; the advance graph reads them next
            obs, returns, alive, done, episode_length = self._run_advance(
                state, actions, last_actions, returns, alive, episode_length
            )
            # The graph clipped the raw action itself, so the next control target matches the
            # ``action_rate`` term of the reward.
            last_actions = torch.clamp(self._torch(actions), -CLIP_ACTIONS, CLIP_ACTIONS)

            done_host = np.asarray(self._host(done), dtype=bool)
            error_mask = state["error_mask"]
            if done_host.any():
                reset_idx = torch.as_tensor(np.flatnonzero(done_host), dtype=torch.int64, device=gs.device)
                # One tiny readback when a row ended: the env's own classification is reused so the
                # rule (and ``end_reasons``) has a single owner, and its inputs are the very values
                # the numpy path classifies.
                rpy_deg = quat_to_xyz(state["base_quat"], rpy=True, degrees=True)
                ending = env._classify_endings(
                    ending,
                    done_host,
                    rpy_deg.detach().cpu().numpy().astype(np.float32),
                    error_mask.detach().cpu().numpy().astype(bool),
                )
                env.robot.set_qpos(
                    row_init_qpos.index_select(0, reset_idx),
                    envs_idx=reset_idx,
                    zero_velocity=True,
                    skip_forward=True,
                )
                last_actions.index_fill_(0, reset_idx, 0.0)
            # ``alive`` never re-arms, so the host mirror is enough to detect the end of the
            # generation without reading the graph's ``alive`` output.
            host_alive &= ~done_host
            if bool(error_mask.any()):
                # The graph already latched the error into ``alive``/``returns``, and a row that is
                # dead is never reset again, so the flag has to be dropped here or the next
                # ``scene.step()`` raises (see 'Scene.get_error_envs_mask').
                env.scene.clear_errno()
            if stop_when_done and not host_alive.any():
                break
        wall_time_s = time.perf_counter() - start

        returns_host = np.asarray(self._host(returns), dtype=np.float32)
        fitness = self._fitness(returns)
        env_steps = steps * rows
        return {
            "seeds": seeds,
            "episode_returns": returns_host.reshape(env.pop_size, env.episodes_per_policy).copy(),
            "returns": returns_host,
            "fitness": fitness,
            "steps": steps,
            "env_steps": env_steps,
            "wall_time_s": wall_time_s,
            "steps_per_s": steps / wall_time_s,
            "env_steps_per_s": env_steps / wall_time_s,
            "cache_hits": env.cache.hits,
            "cache_misses": env.cache.misses,
            "end_reasons": {name: int(np.count_nonzero(ending == code)) for name, code in END_REASONS.items()},
            "init_qpos": init_qpos,
        }

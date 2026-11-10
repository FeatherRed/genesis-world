"""Device-resident (zero-copy) rollout of one generation of the Go2 TensorNEAT trainer.

This module is the opt-in alternative to the host-staged loop of ``Go2NeatEnv.evaluate``: the same
generation, the same graphs, the same resets and the same bookkeeping, but every value that crosses a
graph boundary stays on the accelerator.  ``Go2NeatEnv.evaluate`` dispatches here when the env was
built with ``residency="device"``; the numpy path stays the reference implementation, so a difference
between the two residencies can only come from this bridge.  Device residency needs a non-CPU
``evoxc`` device (the env refuses to be built otherwise) and imports IREE lazily.

No Torch
--------
This loop does not import Torch and does not perform a single Torch operation, so a whole generation
runs in a process where ``import torch`` fails.  Every device-side value is an ``evoxc`` tensor or a
``genesis.utils.device_buffer.DeviceView``, the two setters (``control_dofs_position`` and
``set_qpos``) take them as DLPack producers and stage them device to device, and the fences are
``qd.sync()`` (Quadrants' own stream sync) plus ``IreeExecutable.run``'s blocking return.

Interop contract (Genesis / Quadrants <-> IREE on one ROCm or CUDA device)
-------------------------------------------------------------------------
``evoxc.run`` enforces a strict run boundary: every tensor leaf must ALREADY live on the executable's
device, and a non-CPU executable rejects host (numpy) inputs.  The bridge only ever hands device
buffers over, through DLPack in both directions:

* live solver buffer -> graph input (``_operand``): ``DeviceView(field)[low:high]`` is a contiguous,
  batch-last ``(12, rows)`` slice of ``solver.dyn_state.dofs.pos`` / ``.vel``, and
  ``capsule = view.__dlpack__()``, ``hal.from_dlpack_capsule(capsule)`` wraps the *same* memory as an
  IREE HAL buffer view; ``evoxc.core.Tensor(IreeDevicePayload(rt.DeviceArray(hal, view), device))``
  is the operand -- the call form of ``evoxc.backends.adapters.iree._device_array_from_dlpack``.
* graph output -> setter: the control graph's ``(rows, 12)`` targets go straight into
  ``robot.control_dofs_position(targets, _control_dof_slice)``, whose ``_stage_values`` copies them
  device to device with ``hipMemcpy2DAsync``; the per-generation ``(rows, 19)`` init-qpos tile goes
  the same way into ``robot.set_qpos``, so an auto-reset copies nothing from the host.  ``set_qpos``
  has one masked-write contract in both residencies: a whole ``(rows, 19)`` tile plus the selected
  rows, each env's row being picked in-kernel.
* graph output -> graph input: an executable's output is passed straight to the next one (the
  observation to the policy graph, the action to the control graph, the state blocks to the advance
  graph, ``returns`` / ``alive`` / ``episode_length`` back into the advance graph); ``IreeExecutable.
  run`` passes a same-device ``IreeDevicePayload`` through with no staging, so those values never
  leave the device.
* ``episode_returns`` is the compiled ``(rows,)`` -> ``(pop_size, episodes)`` reshape graph
  (``go2_device_kernels.run_episode_returns``), not a host copy: ``DeviceArray.reshape`` / any numpy
  function on a device array returns a HOST ``numpy.ndarray``, so nothing in this module reshapes,
  slices or ``np.*``-es a device value.
* ``HalDevice``: taken from the first device payload this module sees
  (``tensor.data.device_array._device``), i.e. from the operands staged at the start of a generation;
  the adapter keeps one HAL device per ``core.Device``, so it is the executables' runtime device.

Synchronization: IREE's HIP queue is *not* Quadrants' stream and the DLPack import adds no fence, so
this module fences explicitly -- once per step and once per reset:

* ``_sync()`` after ``scene.step()``, before the state graph imports the buffers Genesis just wrote
  (Quadrants -> IREE);
* ``_sync()`` once per reset writes: the init-qpos tile and the PD targets are device tensors the
  graphs import by DLPack, so the generation's first reads must not run ahead.

Nothing else needs a fence: ``IreeExecutable.run`` blocks on its own queue before returning, so an
exported output is complete by the time it is handed on, including to the setters.

What stays on the device
------------------------
Observations, raw and clipped actions, ``last_actions``, ``returns``, ``alive``, ``episode_length``,
the PD targets, the per-generation init-qpos tile, the two joint permuations, the default-target
tile, the state blocks and the reset observation.

Host touches that remain (exhaustive)
-------------------------------------
Per step:

* ``done`` (``rows`` bytes) -- ``payload.to_host()``, for ``np.flatnonzero`` (the reset rows), the
  ``end_reasons`` classification and the host ``alive`` mirror that ``stop_when_done`` checks;
* on a step where a row ended: the ``base_quat`` rows and the ``error_flag`` vector (a few hundred
  bytes) handed to the env's own ``_classify_endings``, so the rule has one owner.  ``error_flag`` is
  only read here because a flagged row is always a ``done`` row: the advance graph folds the mask
  into ``died``, and a row that ended is teleported home before its flag could recur.  ``clear_errno``
  drops the flag of every row that reports one, dead or alive.

Per generation: the ``(3,)`` seeds, the four genome payload leaves (already device-resident under the
trainer; host arrays are staged when this function is called directly) and the ``(rows, 19)``
init-qpos tile -- all once, before the first step.  The returned training leaves (``returns``,
``episode_returns``, ``fitness``, ``seeds``) stay device tensors.

The termination decision is not a host decision: it happens inside ``run_advance``, and the host only
reads the resulting ``done`` mask.

Parity
------
The bridge changes no ordering and no arithmetic: the same graphs run on the same values in the same
sequence, and both residencies produce identical seeds, episode returns, fitness, step counts and
``end_reasons``.
"""

from __future__ import annotations

import time

import numpy as np
import quadrants as qd

from genesis.utils.device_buffer import DeviceView
from genesis.utils.geom import inv_quat, quat_to_xyz

from go2_device_kernels import default_targets_tile, run_control, run_episode_returns, run_state
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
        executables (tuple): ``(obs, policy, advance, fitness, control, state, episode_returns)``
            executables, all on
            one device.
        pop_arrays (tuple): The four genome payload leaves of the generation, device-resident under
            the trainer (host arrays are staged, for a direct call).
        seeds (np.ndarray): The generation's ordered episode seeds; returned as a device tensor so the
            caller can consume it without a host roundtrip.
        init_qpos (np.ndarray): ``(3, 19)`` perturbed init qpos of the generation, passed through.
        row_init_qpos (np.ndarray): ``(rows, 19)`` policy-major tiling of ``init_qpos``.
        horizon (int): ``scene.step()`` calls of the generation.
        stop_when_done (bool): Stop as soon as every row has ended.

    Returns:
        dict: The same mapping ``Go2NeatEnv.evaluate`` returns, with the four training leaves
        device-resident, so ``env_base.validate_evaluation_device`` and
        ``trainer.NeuroevolutionTrainer`` accept it unchanged.
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
    """The zero-copy loop of one generation.

    One instance serves one generation: the cached solver views and the HAL device belong to the scene
    of that generation, and a reset writes the buffers in place, so the views stay valid for the whole
    loop.
    """

    def __init__(self, env, executables):
        self.env = env
        (
            self.obs_executable,
            self.policy_executable,
            self.advance_executable,
            self.fitness_executable,
            self.control_executable,
            self.state_executable,
            self.episode_returns_executable,
        ) = executables
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
        self._dof_low = int(env._control_dof_slice.start)
        self._dof_high = int(env._control_dof_slice.stop)
        # Host metadata the graphs specialize on; staged once per generation (``_stage_generation``).
        self._standing = _tile_standing(self.obs_executable.standing_flat, self.rows)
        self._commands = _tile_commands(DEFAULT_COMMANDS, self.rows)
        self._commands_scaled = _tile_scaled_commands(DEFAULT_COMMANDS, self.rows)
        self._default_targets = default_targets_tile(self.rows, env.default_dof_pos, env.actions_dof_idx)
        self._identity_quat = np.zeros((self.rows, 4), dtype=np.float32)
        self._identity_quat[:, 0] = 1.0
        self._zero_vel = np.zeros((self.rows, 3), dtype=np.float32)
        self._zero_actions = np.zeros((self.rows, NUM_ACTIONS), dtype=np.float32)
        self._init_dof_pos = None
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

    def _operand(self, value):
        """Wrap one device value as an ``evoxc`` operand sharing its memory (zero-copy).

        An ``evoxc`` tensor (a graph output, or something already staged) is passed through; a
        ``DeviceView`` of a live solver buffer is imported through its DLPack capsule.
        """
        if self._hal is None:
            raise RuntimeError("the HAL device is not known yet; stage an operand or run a graph first")
        import evoxc.core
        import iree.runtime as rt
        from evoxc.backends.adapters.iree import IreeDevicePayload

        view = self._hal.from_dlpack_capsule(value.__dlpack__())
        return evoxc.core.Tensor(IreeDevicePayload(rt.DeviceArray(self._hal, view), self.device))

    def _live(self, value):
        """A live solver buffer or getter result as a graph operand (copied on the device if strided).

        ``robot.get_pos/get_quat/get_vel/get_ang`` and ``scene.get_error_envs_mask`` return a
        contiguous ``DeviceView`` in a Torch-free process (and a contiguous tensor when Torch is
        present); the error mask is boolean in both residencies.  The compaction below only covers a
        backend that ever hands out a strided view, and it stays on the device either way.
        """
        flag = getattr(value, "is_contiguous", None)
        flagged = flag() if callable(flag) else flag
        if flagged is False:
            value = value.contiguous()
        return self._operand(value)

    def _stage(self, array):
        """Copy one host array onto the device (the only host -> device staging of the loop)."""
        import evoxc

        # IREE and XLA can both serve rocm/cuda in one process, and a transfer resolves the provider
        # of the last backend looked up, so re-select this executable's backend before staging.
        evoxc.backends.get(self.backend)
        tensor = evoxc.from_numpy(np.ascontiguousarray(array)).to(self.device)
        self._note_hal(tensor)
        return tensor

    def _host(self, tensor):
        """Materialize one output on the host (the payload owns an explicit readback)."""
        return tensor.data.to_host()

    @staticmethod
    def _sync():
        """Drain Quadrants' stream so the graphs (on IREE's queue) see the finished state."""
        qd.sync()

    # --- state ----------------------------------------------------------------------

    def _reset_rows(self, envs_idx):
        """Teleport the ended rows back to the generation's init pose (device to device).

        ``set_qpos`` has a single masked-write contract: it takes the whole ``(rows, n_qs)`` tile plus
        the selection and picks each env's row in its own kernel, staging device to device, so this
        call is identical with and without Torch.
        """
        self.env.robot.set_qpos(
            self._tile, envs_idx=np.asarray(envs_idx, dtype=np.int32), zero_velocity=True, skip_forward=True
        )

    def _block(self, field):
        """Contiguous batch-last ``(12, rows)`` slice of a live ``(n_dofs, rows)`` solver buffer."""
        view = self._views.get(id(field))
        if view is None:
            view = DeviceView(field)[self._dof_low : self._dof_high]
            if not view.is_contiguous:
                # Only a strided source reaches this branch; IREE refuses strided DLTensors, so the
                # slice is compacted on the device (never through the host).
                view = view.contiguous()
            self._views[id(field)] = view
        return view

    def _read_state(self):
        """Read the stepped state as device values, without any host traffic."""
        solver = self.env.robot._solver
        return run_state(
            self.state_executable,
            base_quat_world=self._live(self.env.robot.get_quat()),
            dof_pos_block=self._operand(self._block(solver.dyn_state.dofs.pos)),
            dof_vel_block=self._operand(self._block(solver.dyn_state.dofs.vel)),
            inv_init_quat=self._inv_init_quat,
            error_mask=self._live(self.env.scene.get_error_envs_mask()),
        )

    # --- graphs ---------------------------------------------------------------------

    def _pop_operands(self, pop_arrays):
        """Hand the four genome payload leaves over: pass-through when device-resident."""
        from device_runtime import assert_device_resident, is_device_tensor

        shapes = tuple(self.policy_executable.pop_shapes)
        arrays = tuple(pop_arrays)
        if len(arrays) != len(shapes):
            raise ValueError(f"pop_arrays must hold the {len(shapes)} genome payload leaves, got {len(arrays)}")
        operands = []
        for array, shape in zip(arrays, shapes, strict=True):
            if is_device_tensor(array):
                # The trainer path: the leaves never leave the device, so nothing is staged.
                operand = assert_device_resident(array, self.device, "pop_arrays leaf")
            else:
                operand = self._stage(array)
            if tuple(int(dim) for dim in operand.shape) != tuple(shape):
                raise ValueError(f"pop_arrays leaf must keep the built shape {shape}, got {operand.shape}")
            operands.append(operand)
        return operands

    def _reset_obs(self):
        """The observation of a fresh reset, computed from the known reset values (device-side)."""
        import evoxc

        outputs = evoxc.run(
            self.obs_executable,
            self._zero_vel_op,
            self._identity_quat_op,
            self._init_dof_pos,
            self._zero_actions_op,
            self._zero_actions_op,
            self._standing,
            self._commands_scaled,
        )
        return _outputs(outputs)[0]

    def _run_policy(self, pop_operands, obs):
        """One policy forward pass; ``obs`` is the previous graph's output (no staging)."""
        import evoxc

        return _outputs(evoxc.run(self.policy_executable, *pop_operands, obs))[0]

    def _run_advance(self, state_base_quat, dof_pos, dof_vel, actions, last_actions, returns, alive, episode_length,
                     error_flag):
        """One advance step; mirrors ``go2_neat_kernels.run_advance`` operand for operand."""
        import evoxc

        executable = self.advance_executable
        outputs = evoxc.run(
            executable,
            self._live(self.env.robot.get_pos()),
            state_base_quat,
            self._live(self.env.robot.get_vel()),
            self._live(self.env.robot.get_ang()),
            dof_pos,
            dof_vel,
            actions,
            last_actions,
            returns,
            alive,
            episode_length,
            error_flag,
            self._standing,
            self._commands,
            self._commands_scaled,
            executable.max_episode_length,
            executable.dt,
            executable.tan_termination,
            CLIP_ACTIONS,
        )
        obs, _reward, returns, alive, done, episode_length = _outputs(outputs)
        return obs, returns, alive, done, episode_length

    def _fitness(self, returns):
        """The fitness graph on the device-resident returns of the generation."""
        import evoxc

        executable = self.fitness_executable
        return _outputs(evoxc.run(executable, returns, executable.n_policies, executable.episodes_per_policy))[0]

    def _episode_returns(self, returns):
        """The policy-major ``(pop_size, episodes)`` view of ``returns``, as a graph.

        ``DeviceArray.reshape`` and every other numpy function on a device array return a HOST
        copy, so the shape is changed by the compiled ``episode_returns`` graph instead (see
        ``go2_device_kernels``); the result stays on the device.
        """
        return run_episode_returns(self.episode_returns_executable, returns)

    # --- generation -----------------------------------------------------------------

    def _stage_generation(self, row_init_qpos):
        """Stage the per-generation metadata tiles and write the reset pose of every row."""
        tile = np.ascontiguousarray(np.asarray(row_init_qpos, dtype=np.float32))
        init_quat = np.ascontiguousarray(tile[:, 3:7])
        self._inv_init_quat = self._stage(inv_quat(init_quat))
        # The reset observation's joint block is metadata (the perturbed init qpos), so it is staged
        # once per generation; a reset inside the loop re-writes the whole tile already on the device.
        self._init_dof_pos = self._stage(tile[:, 7:][:, self.env._qpos_dof_slot])
        self._tile = self._stage(tile)
        self._default_targets_op = self._stage(self._default_targets)
        self._zero_vel_op = self._stage(self._zero_vel)
        self._zero_actions_op = self._stage(self._zero_actions)
        self._identity_quat_op = self._stage(self._identity_quat)
        self.env.robot.set_qpos(self._tile, envs_idx=None, zero_velocity=True, skip_forward=True)
        self.env.scene.clear_errno()  # drop any flag the previous generation left behind
        self._sync()  # the reset landed in Quadrants' stream; the first graph reads it next

    def run(self, *, pop_arrays, seeds, init_qpos, row_init_qpos, horizon, stop_when_done):
        """Simulate the generation and return its statistics mapping."""
        # Imported here because ``go2_neat_env`` imports this module lazily from ``evaluate``.
        from go2_neat_env import END_REASONS

        env = self.env
        rows = self.rows
        start = time.perf_counter()

        pop_operands = self._pop_operands(pop_arrays)
        self._stage_generation(row_init_qpos)
        obs = self._reset_obs()
        done = self._stage(np.zeros(rows, dtype=bool))
        returns = self._stage(np.zeros(rows, dtype=np.float32))
        alive = self._stage(np.ones(rows, dtype=bool))
        episode_length = self._stage(np.zeros(rows, dtype=np.int32))
        last_actions = self._stage(np.zeros((rows, NUM_ACTIONS), dtype=np.float32))
        # The action a step executes is the one the policy emitted on the *previous* step
        # (``go2_env``'s ``simulate_action_latency``: the numpy path writes ``last_actions`` of the
        # previous iteration), so the targets of step 0 are the default pose and the control graph
        # runs at the END of an iteration, producing the targets of the next one.
        targets = self._default_targets_op
        ending = np.full(rows, END_REASONS["unfinished"], dtype=np.int8)
        host_alive = np.ones(rows, dtype=bool)

        steps = 0
        while steps < horizon:
            actions = self._run_policy(pop_operands, obs)
            env.robot.control_dofs_position(targets, env._control_dof_slice)
            env.scene.step()
            steps += 1
            self._sync()  # Genesis wrote the state buffers; the state graph reads them next

            base_quat, dof_pos, dof_vel, error_flag = self._read_state()
            obs, returns, alive, done, episode_length = self._run_advance(
                base_quat, dof_pos, dof_vel, actions, last_actions, returns, alive, episode_length, error_flag
            )
            # The targets and the ``action_rate`` reference of the NEXT step, exactly where the numpy
            # path clips the action and zeroes the rows that just ended (``last_actions[reset] = 0``).
            targets, last_actions = run_control(
                self.control_executable,
                raw_actions=actions,
                done=done,
                default_targets=self._default_targets_op,
                clip_actions=CLIP_ACTIONS,
            )
            done_host = np.asarray(self._host(done), dtype=bool)
            if done_host.any():
                # One tiny readback when a row ended: the env's own classification is reused so the
                # rule (and ``end_reasons``) has a single owner, and its inputs are the very values
                # the numpy path classifies.
                reset_idx = np.flatnonzero(done_host).astype(np.int32)
                flags = np.asarray(self._host(error_flag), dtype=bool)
                rpy_deg = quat_to_xyz(np.asarray(self._host(base_quat), dtype=np.float32), rpy=True, degrees=True)
                ending = env._classify_endings(ending, done_host, rpy_deg, flags)
                self._reset_rows(reset_idx)
                if flags.any():
                    # The graph already latched the error into ``alive``/``returns``, and a row that is
                    # dead is never reset again, so the flag has to be dropped here or the next
                    # ``scene.step()`` raises (see 'Scene.get_error_envs_mask').
                    env.scene.clear_errno()
                self._sync()  # the reset landed; the next graph must not read ahead of it
            # ``alive`` never re-arms, so the host mirror is enough to detect the end of the generation
            # without reading the graph's ``alive`` output.
            host_alive &= ~done_host
            if stop_when_done and not host_alive.any():
                break
        wall_time_s = time.perf_counter() - start

        fitness = self._fitness(returns)
        env_steps = steps * rows
        return {
            "seeds": self._stage(np.ascontiguousarray(seeds, dtype=np.int64)),
            "episode_returns": self._episode_returns(returns),
            "returns": returns,
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

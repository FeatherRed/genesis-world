"""Device-resident (zero-copy) rollout of one generation of the drone TensorNEAT hover task.

:class:`DeviceRollout` scores one generation of the TensorNEAT hover task exactly like the host-staged
loop of :meth:`hover_neat_env.HoverNeatEnv.evaluate`, but every value that crosses a graph boundary
stays on the accelerator, and nothing on the path of the loop performs a Torch operation.  It is
selected with ``HoverNeatEnv(..., residency="device")`` and imported lazily by that env, so the
default numpy residency never imports this module nor IREE.

Why it exists
-------------
The numpy residency is the correctness-first boundary: every read materializes a host array
(``drone.get_pos`` / ``get_quat`` / ``get_vel`` / ``get_ang``) and every write stages one back
(``drone.set_propellers_rpm``).  That is one full state round trip per ``scene.step()``, which
dominates the wall time at the 24-row batch size of the TensorNEAT pilot.  This module removes all
of it: the getters and the actuator write take the torch-free DLPack path of Genesis itself, so only
the values that are genuinely host-side (the command draws) touch the host.

No Torch
--------
This module imports no Torch, at module scope or inside a helper, and calls no Torch operation: a
whole generation runs in a process where ``import torch`` fails.  ``hover_neat_check.py --imports``
scans it, and ``hover_neat_check.py --torch-free`` proves the runtime claim.  The counterparts of the
Torch-side primitives are the torch-free ones Genesis ships: ``genesis.utils.device_buffer.DeviceView``
reads a Quadrants buffer without leaving the device, ``DroneEntity.set_propellers_rpm`` takes a
DLPack producer and stages it on the device, and the getters of a torch-free scene already return the
device views this loop imports.

Interop contract (Genesis / Quadrants <-> IREE on one ROCm, CUDA or CPU device)
-----------------------------------------------------------------------------
``evoxc.run`` enforces a strict run boundary: every tensor leaf must already live on the executable's
device.  The bridge only ever hands device values over, through DLPack in both directions:

* live value -> graph input (``_live`` + ``_operand``): a torch-free ``drone.get_pos`` /
  ``get_quat`` / ``get_vel`` / ``get_ang`` is a ``DeviceView`` of the live solver buffer, already in
  the operand layout ``hover_advance`` takes (``get_vel`` even carries the
  ``cd_vel + cd_ang x (pos - root_COM)`` transport term the entity getter composes); the scene's
  ``(rows,)`` solver-error mask is the same kind of view.  A strided view is compacted with
  ``DeviceView.contiguous()``, on the device, because IREE refuses a strided DLTensor; then
  ``capsule = view.__dlpack__()`` and ``hal.from_dlpack_capsule(capsule)`` wrap the *same* memory as
  an IREE HAL buffer view, which ``evoxc.core.Tensor(IreeDevicePayload(...))`` completes -- the call
  form of ``evoxc.backends.adapters.iree._device_array_from_dlpack``.
* graph output -> graph input: an executable's output is passed straight to the next one (the raw
  action to the controls graph, the clipped action and the RPMs back into the loop, the observation
  and the state operands into the advance graph, ``returns`` / ``alive`` / ``episode_length`` /
  ``end_code`` back into the advance graph, and ``returns`` into the fitness and episode-returns
  graphs).  ``IreeExecutable.run`` passes a same-device payload through with no staging.
* graph output -> drone actuator (``_write_rpm``): the RPMs of ``hover_controls`` are handed
  straight to ``DroneEntity.set_propellers_rpm``, which stages them on the device; see 'The one
  actuator write' for the one build that takes them from host memory instead.
* ``episode_returns`` is the compiled ``(rows,)`` -> ``(pop_size, episodes)`` reshape graph
  (``hover_device_kernels.run_episode_returns``), not a host copy: ``DeviceArray.reshape`` and every
  numpy function on a device array return a HOST ``numpy.ndarray``, so nothing in this module
  reshapes, slices or ``np.*``-es a device value.

Synchronization: IREE's queue is not Quadrants' stream and the DLPack import adds no fence, so this
module fences explicitly -- once per step and once per reset:

* ``_sync()`` after ``scene.step()``, before the advance graph imports the buffers Genesis just wrote
  (Quadrants -> IREE);
* ``_sync()`` once after the generation's reset write, before the reset observation and the first
  step read the state;
* the rotor write needs no fence of its own: the setter stages the graph's output on Quadrants'
  stream, ordered after the graph that produced it and before the kernels of the next
  ``scene.step()``.

Nothing else needs a fence: ``IreeExecutable.run`` blocks on its own queue before returning, so an
exported output is complete by the time it is handed on.

The one actuator write
----------------------
The rotor RPMs of ``hover_controls`` go to ``DroneEntity.set_propellers_rpm``, the torch-free branch
of which stages them into the solver without any host arithmetic and advances the propeller
accumulator on the device.  The one input it cannot stage device to device is a CPU-kind IREE
payload: on such a build the device *is* the host and the payload exports no DLPack (its read path is
``data.to_host()``), so the RPMs are handed over from host memory the graph already owns.

The numpy residency keeps calling ``drone.set_propellers_rpm`` with numpy arrays, unchanged.

What stays on the device
------------------------
The observation, the raw and clipped actions, the rotor RPMs, the current and previous commands,
``last_actions``, ``last_base_pos``, the four state operands, the solver-error mask, ``returns``,
``alive``, ``episode_length``, the latched ``end_code`` and the reset observation.

Host touches that remain (exhaustive)
-------------------------------------
Per step:

* the ``(rows, 3)`` command draw (``x ~ U(-1, 1)``, ``y ~ U(-1, 1)``, ``z`` pinned to 1.0), staged once
  per step: the stream is host RNG by contract, so it is drawn here and uploaded as
  ``resampled_commands``;
* while ``stop_when_done`` is on, the ``(rows,)`` int8 ``end_code`` readback that reproduces the numpy
  loop bound (the "is every row done" test); with ``stop_when_done=False`` that read happens once, at
  the end of the generation.

The solver-error mask is not a host touch: ``scene.get_error_envs_mask()`` hands out the live
``(rows,)`` bool buffer of a torch-free scene, which this loop imports as the ``error_mask`` operand
and then clears with ``clear_errno()`` -- deliberately, every step, so that an error the graph just
latched dead cannot re-raise on the next ``scene.step()``.  ``clear_errno()`` would take that same
mask, but its ``mask_indices`` reads it back to the host, so the bare call (which clears every env
through the solvers' kernels) is the one that keeps the whole exchange on the device.

Per generation: the seeds of the generation (returned device-resident, staged only if a previous
caller handed host arrays), the four genome payload leaves (device-resident under the trainer, host
arrays are staged when this class is called directly), the ``(rows, 3)`` / ``(rows, 4)`` reset pose and
the first command row, the discarded host observation ``HoverNeatEnv.reset_generation`` returns (the
reset observation this loop uses is recomputed on the device), and -- at the end -- exactly
``end_code`` (when ``stop_when_done`` is off), ``commands``, and the ``steps``/``wall_time_s``/
``end_reasons`` bookkeeping.  ``returns``, ``episode_returns``, ``fitness`` and ``seeds`` leave the
generation as device tensors, which is what ``env_base.validate_evaluation_device`` accepts and what
``trainer.NeuroevolutionTrainer`` consumes.

The termination decision is not a host decision: it happens inside ``run_advance``, which owns the
``end_code`` latch, and the host only reads the resulting code.

Semantics
---------
Identical to the numpy residency: the same reset (``HoverNeatEnv.reset_generation``), the same command
stream, the same reward and termination graphs, no mid-episode reset, and the same result mapping
(validated with ``env_base.validate_evaluation_device``).  The only difference is where the operands
live, so both residencies agree bit for bit on ``returns``, ``episode_returns``, ``fitness`` and
``end_reasons``.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import quadrants as qd

import evoxc

from genesis.utils.misc import data_to_array

import hover_kernels  # noqa: F401  (imported first: it owns the shared-layer ``sys.path`` shim)
from hover_device_kernels import run_episode_returns
from hover_kernels import END_REASONS, NUM_ACTIONS

from evox_runtime import POP_OPERANDS, _outputs  # noqa: E402  (path set up above)

logger = logging.getLogger(__name__)

#: Indices of the leaves ``hover_kernels.hover_advance`` returns, by name.  The graph returns a fixed
#: list (obs, the six reward terms, five masks, commands, episode_length, returns, alive, yaw,
#: end_code, base_pos); this module only consumes the carried bookkeeping, so the names are spelled
#: out once here and the count is checked against ``len(_ADVANCE_LEAVES)``.
_ADVANCE_LEAVES = (
    "obs",
    "reward_target",
    "reward_smooth",
    "reward_yaw",
    "reward_angular",
    "reward_crash",
    "reward_total",
    "crash",
    "timeout",
    "done",
    "at_target",
    "commands",
    "episode_length",
    "returns",
    "alive",
    "yaw_deg",
    "end_code",
    "base_pos",
)
_ADVANCE_INDEX = {name: index for index, name in enumerate(_ADVANCE_LEAVES)}

#: The executables one generation drives, in the order ``HoverNeatEnv.executables`` returns them under
#: this residency.
_EXECUTABLE_NAMES = ("controls", "reset_obs", "policy", "advance", "fitness", "episode_returns")

#: The base-state getters the advance graph takes as operands, with the solver buffer each reads.
_STATE_GETTERS = (("base_pos", "get_pos"), ("base_quat", "get_quat"), ("vel", "get_vel"), ("ang", "get_ang"))


class DeviceRollout:
    """Score one hover generation without leaving the device (see the module docstring).

    One instance serves one generation: the live getter views are imported per step and re-derived
    after every reset, so nothing cached here can go stale.

    Args:
        env (hover_neat_env.HoverNeatEnv): The environment this rollout scores for.  Its
            executables (built on ``env.backend`` / ``env.device``), its seeded reset, its command
            generators and its drone are reused as-is; the rollout owns no scene and never destroys
            one.
        executables (tuple | None): ``(controls, reset_obs, policy, advance, fitness,
            episode_returns)``; ``None`` asks the env for them, which is a cache hit after the first
            call (``HoverNeatEnv.executables``).

    Attributes:
        env (HoverNeatEnv): The owning environment.
        rows (int), pop_size (int): Row and policy counts of one generation.
        backend (str), device (evoxc.Device): The graph target every executable lives on.

    The env's public task API is treated as read-only input: ``rows``, ``pop_size``,
    ``episodes_per_policy``, ``horizon``, ``stop_when_done``, ``backend``, ``device``, ``cache``,
    ``drone``, ``scene``, ``*_executable``, ``_seeds_for``, ``_command_generators``,
    ``_draw_commands``, ``reset_generation``, ``executables``, ``_reset_pos`` and ``_reset_quat``.
    """

    def __init__(self, env, executables=None):
        if env.backend != "iree":
            raise ValueError(
                f"residency='device' needs the IREE backend (the zero-copy interop is IREE's "
                f"DLPack bridge), got {env.backend!r}"
            )
        self.env = env
        self.rows = int(env.rows)
        self.pop_size = int(env.pop_size)
        self.backend = str(env.backend)
        self.device = env.device
        members = tuple(env.executables() if executables is None else executables)
        if len(members) != len(_EXECUTABLE_NAMES):
            raise ValueError(
                f"device residency drives the {len(_EXECUTABLE_NAMES)} executables "
                f"{', '.join(_EXECUTABLE_NAMES)}, got {len(members)}"
            )
        (
            self.controls_executable,
            self.reset_obs_executable,
            self.policy_executable,
            self.advance_executable,
            self.fitness_executable,
            self.episode_returns_executable,
        ) = members
        for name, executable in zip(_EXECUTABLE_NAMES, members, strict=True):
            if executable.device != self.device:
                raise ValueError(f"{name} must live on {self.device}, got {executable.device}")
        # The HAL device every imported operand is bound to: the one the advance graph runs on, which
        # is the only graph an imported (DLPack) operand is handed to.  A backend that does not expose
        # it is covered by ``_note_hal``, which discovers it from the first device payload staged.
        self._hal = getattr(getattr(self.advance_executable, "backend_executable", None), "runtime_device", None)
        self._closed = False

    # --- public API -------------------------------------------------------------------

    def evaluate(self, pop_arrays, generation):
        """Simulate one generation on the device; returns the mapping the numpy path returns.

        Args:
            pop_arrays (tuple): The four genome payload leaves, in the shapes the policy graph
                was built with (validated against ``policy_executable.pop_shapes``).
            generation (int): Generation index, selecting the row of the seed table.

        Returns:
            dict: ``seeds`` (3,) int64, ``episode_returns`` (pop_size, 3) float32, ``returns``
            (rows,) float32 and ``fitness`` (pop_size,) float32 as device tensors, plus the host
            bookkeeping ``steps``, ``env_steps``, ``wall_time_s``, ``steps_per_s``,
            ``env_steps_per_s``, ``cache_hits``, ``cache_misses``, ``end_reasons`` and
            ``commands`` -- exactly what ``HoverNeatEnv.evaluate`` returns in numpy mode.
        """
        if self._closed:
            raise RuntimeError("DeviceRollout was closed; build a new one for a new env")
        env = self.env
        generation = int(generation)
        start = time.perf_counter()
        pop_operands = self._pop_operands(pop_arrays)
        seeds = env._seeds_for(generation)
        rngs = env._command_generators(seeds, generation)

        # The reset stays in ONE place: the env writes the state and draws the generation's first
        # command row.  Its host observation is discarded, because the same observation is recomputed
        # on the device below from the very pose the reset wrote.
        commands, _host_obs = env.reset_generation(generation, rngs)
        self._sync()  # the reset landed in Quadrants' stream; the graphs must not read ahead of it
        commands_op = self._stage(commands)
        reset_pos = self._stage(env._reset_pos)
        obs = self._reset_obs(commands_op, reset_pos, self._stage(env._reset_quat))

        returns = self._stage(np.zeros(self.rows, dtype=np.float32))
        alive = self._stage(np.ones(self.rows, dtype=bool))
        episode_length = self._stage(np.zeros(self.rows, dtype=np.int32))
        last_actions = self._stage(np.zeros((self.rows, NUM_ACTIONS), dtype=np.float32))
        end_code = self._stage(np.full(self.rows, END_REASONS["unfinished"], dtype=np.int8))
        # ``hover_advance`` never writes an operand, so the reset pose can be both the reset
        # observation's pose and the first step's ``last_base_pos``.
        last_base_pos = reset_pos

        ending = np.full(self.rows, END_REASONS["unfinished"], dtype=np.int8)
        host_alive = True
        steps = 0
        while steps < env.horizon and (not env.stop_when_done or host_alive):
            raw_actions = self._run_policy(pop_operands, obs)
            clipped, rpm = self._run_controls(raw_actions)
            self._write_rpm(rpm)  # once per scene.step()
            env.scene.step()
            steps += 1
            # The solver-error latch is read before the fence: the kernels filling it are ordered
            # after the step's on Quadrants' stream, so the one fence below covers the state and the
            # mask together.
            error_mask = self._error_mask()
            self._sync()  # Genesis wrote the state buffers; the advance graph reads them next

            # Every row burns its three command draws every step, reached target or not.
            resampled = np.stack([env._draw_commands(rng) for rng in rngs], axis=0).astype(np.float32)
            drone = env.drone
            leaves = self._run_advance(
                commands_op,
                self._live(drone.get_pos()),
                last_base_pos,
                self._live(drone.get_quat()),
                self._live(drone.get_vel()),
                self._live(drone.get_ang()),
                clipped,
                last_actions,
                episode_length,
                self._stage(resampled),
                error_mask,
                alive,
                returns,
                end_code,
            )
            # The graph has consumed the mask, so the solvers' flags can go: a latched error that is
            # still set would raise on the next ``scene.step()`` instead of being absorbed here.
            env.scene.clear_errno()
            obs = leaves[_ADVANCE_INDEX["obs"]]
            commands_op = leaves[_ADVANCE_INDEX["commands"]]
            episode_length = leaves[_ADVANCE_INDEX["episode_length"]]
            returns = leaves[_ADVANCE_INDEX["returns"]]
            alive = leaves[_ADVANCE_INDEX["alive"]]
            end_code = leaves[_ADVANCE_INDEX["end_code"]]
            # The graph's snapshot is a fresh buffer, so the next step's position cannot alias it.
            last_base_pos = leaves[_ADVANCE_INDEX["base_pos"]]
            last_actions = clipped

            if env.stop_when_done:
                # The latched code is the host's only view of "is every row done"; the numpy loop
                # reads ``alive`` for the same decision, and ``alive`` never re-arms.
                ending = np.asarray(self._host(end_code), dtype=np.int8)
                host_alive = bool(np.any(ending == END_REASONS["unfinished"]))
        wall_time_s = time.perf_counter() - start

        if not env.stop_when_done:
            ending = np.asarray(self._host(end_code), dtype=np.int8)
        fitness = self._fitness(returns)
        commands_host = np.asarray(self._host(commands_op), dtype=np.float32)
        env_steps = steps * self.rows
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
            "commands": commands_host,
        }

    def state_parity(self):
        """Max abs difference between the imported state operands and the drone getters.

        Imports the four getter results exactly as a step does -- ``DeviceView.contiguous()`` for a
        strided view, then the DLPack import -- reads each imported operand back once and compares it
        with the host value of the same getter.  It is the proof of the operand mapping (layout,
        compaction and device), and it must be exactly ``0.0``; it is a diagnostic, so its readbacks
        are deliberately outside the rollout path.
        """
        difference = 0.0
        for name, getter_name in _STATE_GETTERS:
            value = getattr(self.env.drone, getter_name)()
            imported = self._live(value)
            staged = np.asarray(self._host(imported), dtype=np.float32)
            expected = np.asarray(data_to_array(value), dtype=np.float32)
            difference = max(difference, float(np.abs(staged - expected).max()))
            logger.debug("device rollout state parity: %s %.3e", name, difference)
        return difference

    def close(self):
        """Forget the HAL device (idempotent; never touches the env's scene)."""
        self._hal = None
        self._closed = True

    # --- DLPack bridge ----------------------------------------------------------------

    def _note_hal(self, tensor):
        """Remember the HAL device of a device payload (the fallback for a backend that hides it)."""
        if self._hal is None:
            array = getattr(getattr(tensor, "data", None), "device_array", None)
            if array is not None:
                self._hal = array._device

    def _operand(self, value):
        """Wrap one live device view as an ``evoxc`` operand sharing its memory (zero-copy)."""
        import evoxc.core
        import iree.runtime as rt
        from evoxc.backends.adapters.iree import IreeDevicePayload

        if self._hal is None:
            raise RuntimeError("the HAL device of the graphs is not known; no executable exposed one")
        view = self._hal.from_dlpack_capsule(value.__dlpack__())
        return evoxc.core.Tensor(IreeDevicePayload(rt.DeviceArray(self._hal, view), self.device))

    def _live(self, value):
        """A live device value as a graph operand (compacted on the device when strided).

        ``drone.get_pos`` / ``get_quat`` / ``get_ang`` return a strided view of the live solver buffer
        in a torch-free scene (the link slice of a batch-last field), which IREE refuses, so it is
        materialized with ``DeviceView.contiguous()`` -- a device-to-device copy, never a host one.
        ``get_vel`` and the scene's error mask are already contiguous, and a torch scene hands out
        contiguous tensors, whose ``is_contiguous`` is a method rather than a property.
        """
        flag = getattr(value, "is_contiguous", None)
        flagged = flag() if callable(flag) else flag
        if flagged is False:
            value = value.contiguous()
        return self._operand(value)

    def _stage(self, array):
        """Copy one host array onto the device (the only host -> device staging of the loop)."""
        # IREE and XLA can both serve one device in one process, and a transfer resolves the provider
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

    # --- the one actuator write -------------------------------------------------------

    def _write_rpm(self, rpm):
        """Write the rotor RPMs of ``hover_controls`` to the drone, through the entity setter."""
        self.env.drone.set_propellers_rpm(self._rpm_source(rpm))

    def _rpm_source(self, rpm):
        """The RPM values the setter takes: the graph output, or its host mapping on a CPU-kind build.

        A CPU-kind IREE payload does not export DLPack -- its own read path is ``data.to_host()`` --
        and the device of a CPU build *is* the host, so the values are handed over from host memory
        rather than copied device to device.
        """
        if self.device.kind == "cpu":
            return self._host(rpm)
        return rpm

    # --- graphs -----------------------------------------------------------------------

    def _pop_operands(self, pop_arrays):
        """Hand the four genome payload leaves over: pass-through when device-resident."""
        from device_runtime import assert_device_resident, is_device_tensor

        shapes = tuple(self.policy_executable.pop_shapes)
        arrays = tuple(pop_arrays)
        if len(arrays) != len(POP_OPERANDS):
            raise ValueError(f"pop_arrays must hold the {len(POP_OPERANDS)} GenomeTensors payloads, got {len(arrays)}")
        operands = []
        for name, array, shape in zip(POP_OPERANDS, arrays, shapes, strict=True):
            if is_device_tensor(array):
                # The trainer path: the leaves never leave the device, so nothing is staged.
                operand = assert_device_resident(array, self.device, "pop_arrays leaf")
            else:
                operand = self._stage(array)
            if tuple(int(dim) for dim in operand.shape) != tuple(shape):
                raise ValueError(f"pop_arrays leaf {name} must keep the built shape {tuple(shape)}, got {operand.shape}")
            operands.append(operand)
        return operands

    def _reset_obs(self, commands, base_init_pos, base_init_quat):
        """The observation of the generation's reset, computed from the reset pose on the device."""
        return _outputs(evoxc.run(self.reset_obs_executable, commands, base_init_pos, base_init_quat))[0]

    def _run_policy(self, pop_operands, obs):
        """One policy forward pass; ``obs`` is the previous graph's output (no staging)."""
        return _outputs(evoxc.run(self.policy_executable, *pop_operands, obs))[0]

    def _run_controls(self, raw_actions):
        """Clip the raw action and build the rotor RPMs (``hover_kernels.hover_controls``)."""
        clipped, rpm = _outputs(evoxc.run(self.controls_executable, raw_actions))[:2]
        return clipped, rpm

    def _run_advance(
        self,
        commands,
        base_pos,
        last_base_pos,
        base_quat,
        base_lin_vel_world,
        base_ang_vel_world,
        actions,
        last_actions,
        episode_length,
        resampled_commands,
        error_mask,
        alive,
        returns,
        end_code,
    ):
        """One advance step, operand for operand in the order ``hover_kernels`` declares them."""
        executable = self.advance_executable
        outputs = evoxc.run(
            executable,
            commands,
            base_pos,
            last_base_pos,
            base_quat,
            base_lin_vel_world,
            base_ang_vel_world,
            actions,
            last_actions,
            episode_length,
            resampled_commands,
            error_mask,
            alive,
            returns,
            end_code,
            # The static specialization of the graph: the same value build_advance baked in.
            int(executable.max_episode_length),
        )
        leaves = _outputs(outputs)
        if len(leaves) != len(_ADVANCE_LEAVES):
            raise RuntimeError(
                f"hover_advance returned {len(leaves)} leaves, expected {len(_ADVANCE_LEAVES)}: "
                "the graph and this rollout disagree about the carried bookkeeping"
            )
        return leaves

    def _fitness(self, returns):
        """The fitness graph on the device-resident returns of the generation."""
        executable = self.fitness_executable
        return _outputs(evoxc.run(executable, returns, executable.n_policies, executable.episodes_per_policy))[0]

    def _episode_returns(self, returns):
        """The policy-major ``(pop_size, episodes)`` view of ``returns``, as a graph."""
        return run_episode_returns(self.episode_returns_executable, returns)

    def _error_mask(self):
        """The solver-error latch as a device bool operand, imported from the scene's live mask.

        ``Scene.get_error_envs_mask`` hands out the ``(rows,)`` bool buffer itself in a torch-free
        scene, and the next call rewrites it, so the operand is consumed within the step that
        imported it.  The kernels filling it sit on Quadrants' stream after the step's own, which is
        why the loop takes the mask before its one fence.
        """
        return self._live(self.env.scene.get_error_envs_mask())

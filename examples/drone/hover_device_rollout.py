"""Opt-in device-resident (zero-copy) rollout of the drone hover evaluation environment.

:class:`DeviceRollout` scores one generation of the TensorNEAT hover task exactly like the
host-staged loop of :meth:`hover_neat_env.HoverNeatEnv.evaluate`, but it never moves the
per-step state to the host: the compiled graphs consume the simulation state straight out of
the live Quadrants buffers, hand their results to each other in device memory, and only the
genome payload leaves, two host draws per step and the small per-generation result ever cross
the PCIe bus.  It is selected with ``HoverNeatEnv(..., residency="device")`` and imported
lazily by that env, so the default numpy residency never imports this module or torch.

Why it exists
-------------
The numpy residency is the correctness-first boundary: every read materializes a host array
(``drone.get_pos`` / ``get_quat`` / ``get_vel`` / ``get_ang`` / ``scene.get_error_envs_mask``)
and every write stages one back (``drone.set_propellers_rpm``).  That is one full state
round-trip per ``scene.step()``, which dominates the wall time at the 24-row batch size of the
TensorNEAT pilot.  This module removes all of it except the parts that are genuinely host-side
(the command draws and the solver-error latch) plus the one actuator write Genesis requires.

Interop recipe
--------------
Everything below was verified on this host (AMD Instinct MI300X VF, ROCm 7.1, torch
2.13.0+rocm7.1, evoxc 0.1.0, iree-base-runtime 3.13.0) and on the CPU/``local-task`` device.

* **Quadrants -> IREE.** ``qd_to_torch(field, copy=False)`` gives a torch tensor aliasing the
  live solver buffer; ``torch.utils.dlpack.to_dlpack`` turns it into a capsule and
  ``runtime_device.from_dlpack_capsule(capsule)`` wraps the SAME memory in a
  ``HalBufferView``, from which ``iree.runtime.DeviceArray(runtime_device, view)`` and
  ``evoxc.Tensor(IreeDevicePayload(array, device))`` complete the operand.  The torch view is
  kept alive here (it owns nothing, but it must outlive the imported buffer), so the import is
  cached for the generation instead of rebuilt per step.
* **The one staging copy.** That import requires the pointer to be aligned (64 bytes for the
  ``local-task`` heap, 256 for HIP) while a *link slice* of the Quadrants state is not (the
  drone's base link starts 12 or 24 bytes into the field on CPU), so each state field is
  copied once per step into a persistent torch staging buffer (``copy_`` on the device, a few
  dozen bytes, no host round trip) and the import is taken over that buffer.  The staged
  values are bit-identical to the buffers; ``--residency-parity`` proves it against the entity
  getters (measured difference exactly ``0.0``).
* **Which solver fields.** ``env.scene.sim.rigid_solver.dyn_state.links``, sliced to
  ``env.drone.base_link_idx`` exactly like the getters:
  ``get_pos`` = ``links.pos``, ``get_quat`` = ``links.quat``,
  ``get_ang`` = ``links.cd_ang`` and
  ``get_vel`` = ``cd_vel + cd_ang x (pos - root_COM)`` (the transport term of
  ``RigidSolver.get_links_vel``; the propellers of ``cf2x.urdf`` are massless, so it is zero
  here, and it is still computed so the mapping stays exact for any model).
* **IREE -> Genesis.** ``IreeDevicePayload.__dlpack__`` exports the same HAL buffer, so
  ``torch.from_dlpack(rpm)`` aliases the graph output on a CUDA/ROCm device.  A CPU-kind
  payload does not export (by design — its read path is ``.numpy()``, which is a zero-copy
  mapping on ``local-task``), so the CPU path takes ``torch.from_numpy(rpm.numpy())`` instead.
  ``IreeExecutable.run`` blocks on its HIP queue before returning, so no wait is needed for
  the graph itself; the single documented ``torch.cuda.synchronize()`` at the export ->
  ``set_propellers_rpm`` hand-off is there because the Quadrants kernel that consumes it runs
  on torch's stream.  It is the only synchronisation in this module and must be measured
  separately from the graph time.
* **Torch.** ``import torch`` happens inside the helpers that stage or export buffers, never at
  module level, because torch is exactly the Genesis boundary here: the graphs themselves are
  ``evoxc`` all the way down.  This module is therefore deliberately outside the
  ``hover_neat_check.py --imports`` torch-free list.
* **One runtime device.** The five executable objects of one env are loaded with HAL devices
  that evoxc creates per load (``_acquire_runtime_device`` caches only Vulkan), so they are not
  the same object; every imported operand therefore uses the single runtime device of the
  advance executable, and evoxc passes buffers between executables because it compares
  ``core.Device`` labels, not HAL object identity (verified by feeding one executable's output
  into another).

Host round-trips left
---------------------
Two per step, both unavoidable and both small:

* ``scene.get_error_envs_mask()`` — the solver-error latch is only exposed as a host query, and
  it is one ``(rows,)`` bool.  It is uploaded again as the ``error_mask`` operand, and
  ``scene.clear_errno()`` runs whenever a row is flagged (the graph latched the row dead, and a
  latched error otherwise re-raises on the next ``scene.step()``).  A device-side error latch
  in Genesis would remove this hop.
* the two command draws ``x ~ U(-1, 1)``, ``y ~ U(-1, 1)`` (plus the pinned ``z``) that the
  numpy loop also burns per row per step: they are host RNG by contract, so the draw is
  uploaded as ``resampled_commands``.  Moving the RNG graph-side would remove this hop but
  would change the command stream.

When ``stop_when_done`` is set the loop also reads the ``(rows,)`` int8 ``end_code`` back each
step (``end_reasons`` and the "is every row done" test are host bookkeeping); with
``stop_when_done=False`` that read happens once, at the end of the generation.  Nothing else is
downloaded before the end of the generation, where exactly ``returns``, ``fitness``,
``end_code`` and ``commands`` are materialized.

Semantics
---------
Identical to the numpy residency: the same reset (``HoverNeatEnv.reset_generation``), the same
command stream, the same reward/termination graphs, no mid-episode reset, and the same result
mapping (validated with ``env_base.validate_evaluation``).  The only difference is where the
operands live.
"""

from __future__ import annotations

import logging
import time

import numpy as np

import evoxc

from genesis.utils.misc import qd_to_torch

# Importing ``hover_kernels`` first: it puts the shared ``examples/neuroevolution`` layer on
# ``sys.path`` (the env contract and the payload-leaf names live there).
from hover_kernels import END_REASONS, NUM_ACTIONS

from env_base import validate_evaluation  # noqa: E402  (path set up above)
from evox_runtime import POP_OPERANDS, _outputs  # noqa: E402

logger = logging.getLogger(__name__)

#: Indices of the leaves ``hover_kernels.hover_advance`` returns, by name.  The graph returns a
#: fixed list (obs, the six reward terms, five masks, commands, episode_length, returns, alive,
#: yaw, end_code, base_pos); this module only consumes the carried bookkeeping, so the names are
#: spelled out once here and the count is checked against ``len(_ADVANCE_LEAVES)``.
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

#: The live solver fields of ``dyn_state.links`` the base-state getters read, by getter name.
_STATE_FIELDS = ("pos", "quat", "cd_vel", "cd_ang", "root_COM")


class DeviceRollout:
    """Score one hover generation without leaving the device (see the module docstring).

    Args:
        env (hover_neat_env.HoverNeatEnv): The environment this rollout scores for.  Its
            executables (built on ``env.backend`` / ``env.device``), its seeded reset, its
            command generators and its drone are reused as-is; the rollout owns no scene and
            never destroys one.

    Attributes:
        env (HoverNeatEnv): The owning environment.
        rows (int), pop_size (int): Row and policy counts of one generation.
        runtime_device: The IREE HAL device every imported operand is bound to.

    The env's public task API is treated as read-only input: ``rows``, ``pop_size``,
    ``episodes_per_policy``, ``horizon``, ``stop_when_done``, ``backend``, ``device``,
    ``cache``, ``drone``, ``scene``, ``policy_executable`` / ``controls_executable`` /
    ``advance_executable`` / ``fitness_executable``, ``_seeds_for``, ``_command_generators``,
    ``_draw_commands``, ``reset_generation``, ``executables`` and ``_reset_pos``.
    """

    def __init__(self, env):
        if env.backend != "iree":
            raise ValueError(
                f"residency='device' needs the IREE backend (the zero-copy interop is IREE's "
                f"DLPack bridge), got {env.backend!r}"
            )
        self.env = env
        self.rows = int(env.rows)
        self.pop_size = int(env.pop_size)
        self.device = env.device
        # Imported here, not at module level: ``iree`` is an optional dependency of ``evoxc``
        # and this module is only imported when the device residency is selected.
        import iree.runtime as iree_runtime
        from evoxc.backends.adapters.iree import IreeDevicePayload

        self._rt = iree_runtime
        self._payload_cls = IreeDevicePayload
        # One HAL device for every imported operand (see the module docstring: the executable
        # objects each own a HAL device, and evoxc passes buffers across them).
        self.runtime_device = env.advance_executable.backend_executable.runtime_device
        self._state = None
        self._state_key = None
        self._closed = False

    # ------------------------------------------------------------------ public API
    def evaluate(self, pop_arrays, generation):
        """Simulate one generation on the device; returns the numpy path's result mapping.

        Args:
            pop_arrays (tuple): The four genome payload leaves, in the shapes the policy graph
                was built with (validated against ``policy_executable.pop_shapes``).
            generation (int): Generation index, selecting the row of the seed table.

        Returns:
            dict: ``seeds`` (3,) int64, ``episode_returns`` (pop_size, 3) float32, ``returns``
            (rows,) float32, ``fitness`` (pop_size,) float32, ``steps``, ``env_steps``,
            ``wall_time_s``, ``steps_per_s``, ``env_steps_per_s``, ``cache_hits``,
            ``cache_misses``, ``end_reasons`` and ``commands`` -- exactly what
            ``HoverNeatEnv.evaluate`` returns in numpy mode.
        """
        if self._closed:
            raise RuntimeError("DeviceRollout was closed; build a new one for a new env")
        env = self.env
        generation = int(generation)
        payloads = self._policy_payloads(pop_arrays)
        seeds = env._seeds_for(generation)
        rngs = env._command_generators(seeds, generation)
        env.executables()  # compile-once proof: every call after ``__init__`` is a cache hit

        start = time.perf_counter()
        # The reset stays in ONE place: the env writes the state and returns the generation's
        # first commands plus the analytic reset observation, both as host arrays.
        commands, obs = env.reset_generation(generation, rngs)
        commands_t = self._host_operand(commands)
        obs_t = self._host_operand(obs)
        last_base_pos_t = self._host_operand(env._reset_pos)
        returns_t = self._host_operand(np.zeros(self.rows, dtype=np.float32))
        episode_length_t = self._host_operand(np.zeros(self.rows, dtype=np.int32))
        alive_t = self._host_operand(np.ones(self.rows, dtype=bool))
        last_actions_t = self._host_operand(np.zeros((self.rows, NUM_ACTIONS), dtype=np.float32))
        end_code_t = self._host_operand(np.full(self.rows, END_REASONS["unfinished"], dtype=np.int8))
        # The state views are imported afresh for every generation: the reset wrote the state
        # the previous generation's views were staged from.
        self.invalidate()

        steps = 0
        alive_any = True
        ending = np.full(self.rows, END_REASONS["unfinished"], dtype=np.int8)
        while steps < env.horizon and (not env.stop_when_done or alive_any):
            raw_actions_t = evoxc.run(env.policy_executable, *payloads, obs_t)
            raw_actions_t = _first(raw_actions_t)
            clipped_t, rpm_t = _outputs(evoxc.run(env.controls_executable, raw_actions_t))[:2]
            env.drone.set_propellers_rpm(self._export(rpm_t))  # once per scene.step(), or it raises
            env.scene.step()
            steps += 1

            state = self._read_state()
            # Every row burns its three command draws every step, reached target or not.
            resampled = np.stack([env._draw_commands(rng) for rng in rngs], axis=0).astype(np.float32)
            error_mask = self._error_mask()
            outputs = evoxc.run(
                env.advance_executable,
                commands_t,
                state["pos"],
                last_base_pos_t,
                state["quat"],
                state["vel"],
                state["ang"],
                clipped_t,
                last_actions_t,
                episode_length_t,
                self._host_operand(resampled),
                self._host_operand(error_mask),
                alive_t,
                returns_t,
                end_code_t,
                # The static specialization of the graph: the same value build_advance baked in.
                int(env.max_episode_length),
            )
            leaves = _outputs(outputs)
            if len(leaves) != len(_ADVANCE_LEAVES):
                raise RuntimeError(
                    f"hover_advance returned {len(leaves)} leaves, expected {len(_ADVANCE_LEAVES)}: "
                    "the graph and this rollout disagree about the carried bookkeeping"
                )
            obs_t = leaves[_ADVANCE_INDEX["obs"]]
            commands_t = leaves[_ADVANCE_INDEX["commands"]]
            episode_length_t = leaves[_ADVANCE_INDEX["episode_length"]]
            returns_t = leaves[_ADVANCE_INDEX["returns"]]
            alive_t = leaves[_ADVANCE_INDEX["alive"]]
            end_code_t = leaves[_ADVANCE_INDEX["end_code"]]
            # The graph's snapshot is a fresh buffer, so the next step's position cannot alias it.
            last_base_pos_t = leaves[_ADVANCE_INDEX["base_pos"]]
            last_actions_t = clipped_t

            if env.stop_when_done:
                # The latched code is the host's only view of "is every row done"; the numpy loop
                # reads ``alive`` for the same decision.
                ending = self._download(end_code_t).astype(np.int8)
                alive_any = bool(np.any(ending == END_REASONS["unfinished"]))
        wall_time_s = time.perf_counter() - start

        if not env.stop_when_done:
            ending = self._download(end_code_t).astype(np.int8)
        returns = self._download(returns_t).astype(np.float32)
        fitness_executable = env.fitness_executable
        fitness = self._download(
            _first(
                evoxc.run(
                    fitness_executable,
                    returns_t,
                    # The graph's static specializations (see ``evolution_graphs.run_fitness``).
                    int(fitness_executable.n_policies),
                    int(fitness_executable.episodes_per_policy),
                )
            )
        ).astype(np.float32)
        commands = self._download(commands_t).astype(np.float32)

        env_steps = steps * self.rows
        result = {
            "seeds": seeds,
            "episode_returns": returns.reshape(self.pop_size, env.episodes_per_policy).copy(),
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
            "commands": commands,
        }
        validate_evaluation(result, self.pop_size, episodes=env.episodes_per_policy)
        return result

    def state_parity(self):
        """Max abs difference between the live-buffer reads and the entity getters.

        Reads the drone's base state both ways -- through the live solver buffers (the exact
        mapping this rollout feeds to ``hover_advance``) and through ``drone.get_pos`` /
        ``get_quat`` / ``get_vel`` / ``get_ang`` -- and returns the largest disagreement, which
        must be exactly ``0.0``.  It is the proof of the field mapping; there is no fallback to
        the getters on the rollout path.
        """
        self._ensure_state()
        state = self._read_state()
        env = self.env
        getters = {
            "pos": env.drone.get_pos(),
            "quat": env.drone.get_quat(),
            "vel": env.drone.get_vel(),
            "ang": env.drone.get_ang(),
        }
        difference = 0.0
        for name, getter in getters.items():
            staged = self._state["staged"][name]
            difference = max(difference, float(np.abs(self._download(staged) - self._download(getter)).max()))
        return difference

    def invalidate(self):
        """Drop the cached live-state views (their buffers are re-read every step anyway)."""
        self._state = None
        self._state_key = None

    def close(self):
        """Drop the views and the imported buffers (idempotent; never touches the env's scene)."""
        self.invalidate()
        self._closed = True

    # ------------------------------------------------------------------ operand staging
    def _policy_payloads(self, pop_arrays):
        """Upload the four genome payload leaves once, validating their shapes as run_policy does."""
        arrays = tuple(np.ascontiguousarray(leaf) for leaf in tuple(pop_arrays))
        if len(arrays) != len(POP_OPERANDS):
            raise ValueError(f"pop_arrays must hold the {len(POP_OPERANDS)} GenomeTensors payloads, got {len(arrays)}")
        shapes = getattr(self.env.policy_executable, "pop_shapes", None)
        if shapes is not None:
            for name, array, shape in zip(POP_OPERANDS, arrays, shapes, strict=True):
                if array.shape != tuple(shape):
                    raise ValueError(
                        f"pop_arrays leaf {name} must keep the built shape {tuple(shape)}, got {array.shape}"
                    )
        return tuple(self._host_operand(array) for array in arrays)

    def _host_operand(self, array):
        """A device tensor over a host array (``evoxc``'s own staging path)."""
        arr = np.asarray(array)
        if arr.ndim > 0:
            arr = np.ascontiguousarray(arr)
        # Re-select this env's backend first: Tensor.to resolves the transfer provider of the
        # last looked-up backend, and one process may hold IREE and XLA (see evox_runtime).
        evoxc.backends.get(self.env.backend)
        return evoxc.from_numpy(arr).to(self.device)

    def _import(self, torch_tensor):
        """An ``evoxc`` tensor aliasing ``torch_tensor`` memory (no copy, no host round trip)."""
        import torch

        capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
        view = self.runtime_device.from_dlpack_capsule(capsule)
        array = self._rt.DeviceArray(self.runtime_device, view)
        return evoxc.Tensor(self._payload_cls(array, self.device))

    def _download(self, tensor):
        """Materialize one operand on the host (the only device -> host path of this module)."""
        if isinstance(tensor, np.ndarray):
            return tensor
        if isinstance(tensor, evoxc.Tensor):
            tensor = tensor.to(evoxc.Device("cpu", 0))
            return tensor.numpy()
        import torch

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        raise TypeError(f"cannot materialize {type(tensor).__name__} on the host")

    def _export(self, tensor):
        """Hand one graph output to Genesis: a torch view over the SAME device memory.

        On a CUDA/ROCm device the ``IreeDevicePayload`` DLPack export shares the HAL buffer; on
        the CPU (``local-task``) the payload's ``.numpy()`` is itself a zero-copy mapping of the
        buffer, so ``torch.from_numpy`` gives the same aliasing.  This is the only
        synchronisation in the module: the Quadrants kernel Genesis launches next reads the
        buffer on torch's stream, while IREE wrote it on its own queue (``IreeExecutable.run``
        has already blocked on that queue, so the buffer is complete).  Measure it separately.
        """
        import torch

        if self.device.kind == "cpu":
            value = torch.from_numpy(tensor.numpy())
        else:
            value = torch.from_dlpack(tensor)
            torch.cuda.synchronize()
        return value

    # ------------------------------------------------------------------ live solver state
    def _ensure_state(self):
        """Build (once per generation) the staging buffers and the imported state operands."""
        import torch

        import genesis as gs

        links = self.env.scene.sim.rigid_solver.dyn_state.links
        base_link_idx = self.env.drone.base_link_idx
        fields = {name: getattr(links, name) for name in _STATE_FIELDS}
        # Keyed by buffer identity: a new set of field wrappers means new buffers.
        key = tuple(id(field) for field in fields.values())
        if self._state is not None and self._state_key == key:
            return self._state
        torch_device = gs.device
        staged = {
            name: torch.empty((self.rows, width), dtype=torch.float32, device=torch_device)
            for name, width in (("pos", 3), ("quat", 4), ("vel", 3), ("ang", 3))
        }
        # The getters read these fields; the slices alias the live solver buffers (no copy).
        views = {name: qd_to_torch(field, copy=False)[base_link_idx] for name, field in fields.items()}
        self._state = {
            "views": views,
            # The staged buffers are the imported operands: the import is alignment-safe and
            # stays valid, while the values are refreshed in place every step (see _read_state).
            "staged": staged,
            "operands": {name: self._import(buffer) for name, buffer in staged.items()},
        }
        self._state_key = key
        logger.debug("device rollout: imported %d live state buffers (rows=%d)", len(staged), self.rows)
        return self._state

    def _read_state(self):
        """Refresh the staging buffers from the live solver state; returns the state operands."""
        import torch

        state = self._ensure_state()
        staged, views = state["staged"], state["views"]
        staged["pos"].copy_(views["pos"])
        staged["quat"].copy_(views["quat"])
        staged["ang"].copy_(views["cd_ang"])
        staged["vel"].copy_(views["cd_vel"])
        # ``RigidSolver.get_links_vel``: cd_vel + cd_ang x (pos - root_COM).  On the device, no
        # host round trip; zero for the massless-propeller cf2x model and exact for any other.
        delta = views["pos"] - views["root_COM"]
        staged["vel"].add_(torch.cross(views["cd_ang"], delta, dim=-1))
        return {name: state["operands"][name] for name in ("pos", "quat", "vel", "ang")}

    def _error_mask(self):
        """The solver-error latch as a host bool array (see the module docstring)."""
        from genesis.utils.misc import tensor_to_array

        mask = tensor_to_array(self.env.scene.get_error_envs_mask()).astype(bool)
        if mask.any():
            # The graph latched the flagged rows dead; a latched error otherwise re-raises on
            # the next ``scene.step()`` (see 'Scene.get_error_envs_mask').
            self.env.scene.clear_errno()
        return mask


def _first(result):
    """The single output leaf of a one-output ``evoxc.run`` call."""
    leaves = _outputs(result)
    return leaves[0]

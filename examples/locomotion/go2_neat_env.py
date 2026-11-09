"""Batched, seeded evaluation environment for the TensorNEAT Go2 walking trainer.

One :class:`Go2NeatEnv` owns a single ``gs.Scene`` with ``EPISODES_PER_POLICY * pop_size`` Go2
rows (policy-major: rows ``3p, 3p+1, 3p+2`` are the episodes of policy ``p``) and scores one
generation per :meth:`Go2NeatEnv.evaluate` call.

What this module owns
---------------------
Genesis owns the physics and the state, the compiled graphs of ``go2_neat_kernels`` own the
observation / reward / termination math, the shared ``examples/neuroevolution`` layer owns the
policy / fitness / tell graphs and the seeding rules, and this module is the boundary in between:
the Go2 scene, the reset and perturbation application, the per-generation ``evaluate`` loop and
the ``END_REASONS`` classification of the rows that ended.

:class:`Go2NeatEnv` declares that boundary by subclassing :class:`env_base.BatchedEpisodeEnv`:
its ``rows`` / ``episodes_per_policy`` / ``max_episode_length`` / ``cache`` attributes and its
``evaluate`` / ``close`` methods are exactly what a ``trainer.NeuroevolutionTrainer`` drives, and
:meth:`Go2NeatEnv.evaluate` validates its own result with
:func:`env_base.validate_evaluation` before returning it.

Seeding
-------
``np.random.SeedSequence(master_seed).spawn(2)`` splits the master seed into an independent
evolution stream (``seeding.evolution_seed``, for the trainer's own RNG) and an env stream
(``seeding.generation_seed_table``).  Every generation owns ``EPISODES_PER_POLICY`` distinct
episode seeds; a row keeps its generation's seed -- and therefore its reset perturbation -- for
the whole generation, and the three seed-derived 19-vector init qpos are replicated to every
policy's rows.

Resets
------
The three init qpos are applied once per generation.  Episodes that restart after a ``done`` are
discarded by design: the episode's reward is already booked, the row keeps stepping its physics
from the same init qpos with zeroed actions, and only the *first* ending of each row is reported
in ``end_reasons``.  The sampled ``base_rpy`` rotation is a world-axis rotation composed on top of
the authored init orientation (``q_pert * q_init``, not ``q_init * q_pert``).

``horizon`` is the number of ``scene.step()`` calls per generation.  ``None`` means
``max_episode_length`` (one full episode, so nothing times out); a horizon larger than
``max_episode_length`` is what makes the timeout fire and the auto-reset run.

Action latency
--------------
``go2_train`` runs with ``simulate_action_latency=True``: step ``t`` executes the *clipped*
actions of step ``t - 1`` (zero at reset), exactly like ``go2_env.step()``.  The graphs see the
raw, unclipped action of step ``t`` and clip it themselves.

Residency
---------
``Go2NeatEnv(residency=...)`` selects where the values that cross a graph boundary live:

* ``"numpy"`` (the default, and the reference): every graph operand and result is a host array.
  The per-step state read, the control-target write and the reset write are the only boundaries,
  and they are torch-free: reads go through ``genesis.utils.misc.tensor_to_array`` and writes
  hand NumPy arrays to the Genesis setters, which take ``array_like``.
* ``"device"``: the same generation with every operand resident on the ``evoxc`` device.  The loop,
  the DLPack bridge and the exhaustive list of remaining host touches live in
  :mod:`go2_device_rollout`; this module only dispatches.  It needs a non-CPU ``evoxc`` device
  (``backend="iree"`` on ``rocm:N`` / ``cuda:N``), because device residency is what the IREE
  runtime provides, and it refuses to be built otherwise.

Both residencies run the same graphs on the same values in the same order, so they must agree
bit-for-bit on seeds, episode returns, fitness, steps and ``end_reasons``.

Initialization
--------------
``gs.init`` is called only when ``genesis`` is not initialized yet (``gs._initialized``), with
CPU/float32; a caller that inits first (e.g. ``gs.init(backend=gs.gpu)``) keeps its backend and
precision.  The ``backend``/``device`` arguments select the ``evoxc`` backend of the compiled
graphs only, via ``go2_evox_kernels.resolve_backend_device``.

Host boundary (``residency="numpy"``)
------------------------------------
Every conversion is confined to :meth:`Go2NeatEnv.read_state` (device -> host) and
:meth:`Go2NeatEnv._write_targets` / :meth:`Go2NeatEnv._write_reset` (host -> device), and none of
them needs torch:

* read with ``genesis.utils.misc.tensor_to_array(...).astype(np.float32)``: ``robot.get_pos()``,
  ``get_quat()``, ``get_vel()``, ``get_ang()``, ``get_dofs_position(motors_dof_idx)``,
  ``get_dofs_velocity(motors_dof_idx)``, the base quaternion relative to the per-row init pose,
  and its roll/pitch/yaw (both computed on the host);
* read with ``tensor_to_array(...).astype(bool)``: ``scene.get_error_envs_mask()``;
* write with a plain NumPy array: the PD targets and the per-row reset qpos, because
  ``control_dofs_position`` and ``set_qpos`` take ``array_like`` and convert internally;
* ``motors_dof_idx`` / ``actions_dof_idx`` are NumPy index arrays built once in ``__init__`` and
  only their values ever reach a setter.

The relative base quaternion is the one place where this path cannot reuse
``genesis.utils.geom.transform_quat_by_quat``: its numba Hamilton product drifts from the torch
one (up to 2.1e-4 on a policy fitness over 21 steps), which would change measured numbers.
:func:`_quat_mul` therefore evaluates the same coefficient form as
``genesis.utils.geom._tc_quat_mul`` -- the Hamilton product
``(u * v)_k = sum_ij H[k, i, j] * u_i * v_j`` followed by a normalization -- with NumPy, which
reproduces the torch reference bit-for-bit.  ``inv_quat`` and ``quat_to_xyz`` keep their NumPy
branches, and ``transform_quat_by_quat`` is still used by :func:`perturbed_init_qpos`.

Nothing else leaves the device: ``run_policy`` / ``run_obs`` / ``run_advance`` / ``run_fitness``
are numpy-facing (the wrappers stage their own operands), so the observations, actions and returns
stay on the host.  Under ``residency="device"`` none of the above happens: the whole boundary is
``go2_device_rollout``'s, which keeps its torch usage and owns its own device-side init quaternion.

The torch this module still pulls in is Genesis's own: ``gs.init`` imports it, every
``robot.get_*`` accessor returns a ``torch`` tensor on ``gs.device``, and ``tensor_to_array`` is
the single call that turns those tensors into host arrays.  No torch import, tensor op or device
transfer is written here, and the compiled graphs run on their own (``evoxc``) backend.

This module has no CLI; like the other ``*_env.py`` modules it is excluded from the example smoke
tests.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from tensorneat import Pipeline, algorithm, genome

import genesis as gs

from genesis.utils.geom import inv_quat, quat_to_xyz, transform_quat_by_quat
from genesis.utils.misc import tensor_to_array

from go2_evox_kernels import resolve_backend_device

# ``go2_neat_kernels`` installs ``examples/neuroevolution`` on ``sys.path`` as its first
# import-time action, so the shared layer below imports by bare module name and this module
# needs no sys.path shim of its own.
from go2_neat_kernels import (
    CLIP_ACTIONS,
    DEFAULT_COMMANDS,
    NUM_ACTIONS,
    NUM_OBS,
    REWARD_SCALES,
    TERMINATION_LIMIT_DEG,
    TRACKING_SIGMA,
    ExecutableCache,
    build_advance,
    build_fitness,
    build_obs,
    build_policy,
    run_advance,
    run_fitness,
    run_obs,
    run_policy,
)
from env_base import BatchedEpisodeEnv, validate_evaluation
from evox_runtime import _device_name
from seeding import EPISODES_PER_POLICY, Perturbation, evolution_seed, generation_seed_table
from seeding import perturbation_from_seed as _perturbation_from_seed
from trainer import TraceOnlyProblem as _TraceOnlyProblem
from trainer import pop_arrays_of

logger = logging.getLogger(__name__)

# Verbatim copy of the values of ``examples/locomotion/go2_train.get_cfgs()``, the authoritative
# config of the Go2 walking task.  ``go2_train`` cannot be imported here because it raises at
# import time when ``rsl-rl-lib`` is missing, so this copy is checked against the values baked
# into the compiled graphs by :func:`_check_cfg` instead.
CFG = {
    "num_actions": 12,
    "joint_names": [  # control / observation order
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ],
    "default_joint_angles": {  # [rad]
        "FL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    },
    "kp": 20.0,
    "kd": 0.5,
    "termination_if_roll_greater_than": 10,  # degree
    "termination_if_pitch_greater_than": 10,  # degree
    "base_init_pos": [0.0, 0.0, 0.42],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],  # w-first
    "episode_length_s": 20.0,
    "action_scale": 0.25,
    "simulate_action_latency": True,
    "clip_actions": 100.0,
    "tracking_sigma": 0.25,
    "base_height_target": 0.3,
    "reward_scales": {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.2,
        "lin_vel_z": -1.0,
        "base_height": -50.0,
        "action_rate": -0.005,
        "similar_to_default": -0.1,
    },
    "command_cfg": {
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    },
}

#: Control period of the real robot (50 Hz), also ``go2_env.dt`` and ``go2_neat_kernels.DT``.
DT = 0.02

# Genome shape the policy graph is traced with; ``pop_arrays`` must keep these payload shapes.
MAX_NODES = 80
MAX_CONNS = 640
SPECIES_SIZE = 3
TRACE_SEED = 0

#: Row ending codes, in reporting order.
END_REASONS = {"timeout": 1, "tilted": 2, "error": 3, "unfinished": 0}


def _check_cfg():
    """Guard the hand-copied config against the values baked into the compiled graphs."""
    if CFG["num_actions"] != NUM_ACTIONS:
        raise ValueError(f"CFG['num_actions'] must be {NUM_ACTIONS}, got {CFG['num_actions']}")
    if len(CFG["joint_names"]) != NUM_ACTIONS or set(CFG["joint_names"]) != set(CFG["default_joint_angles"]):
        raise ValueError("CFG['joint_names'] and CFG['default_joint_angles'] must describe the same 12 joints")
    if CFG["clip_actions"] != CLIP_ACTIONS:
        raise ValueError(f"CFG['clip_actions'] must be {CLIP_ACTIONS}, got {CFG['clip_actions']}")
    if CFG["termination_if_roll_greater_than"] != TERMINATION_LIMIT_DEG:
        raise ValueError(f"CFG['termination_if_roll_greater_than'] must be {TERMINATION_LIMIT_DEG}")
    if CFG["termination_if_pitch_greater_than"] != TERMINATION_LIMIT_DEG:
        raise ValueError(f"CFG['termination_if_pitch_greater_than'] must be {TERMINATION_LIMIT_DEG}")
    if CFG["reward_scales"] != REWARD_SCALES:
        raise ValueError(f"CFG['reward_scales'] must be {REWARD_SCALES}, got {CFG['reward_scales']}")
    if CFG["tracking_sigma"] != TRACKING_SIGMA:
        raise ValueError(f"CFG['tracking_sigma'] must be {TRACKING_SIGMA}, got {CFG['tracking_sigma']}")
    if not CFG["simulate_action_latency"]:
        raise NotImplementedError("Go2NeatEnv implements the 1-step action latency of go2_train only")
    # ``go2_env`` resamples commands from these ranges every 4 s; singleton ranges make that a
    # no-op, which is why the graphs can bake ``DEFAULT_COMMANDS`` in as a static operand.
    for name, value in zip(("lin_vel_x_range", "lin_vel_y_range", "ang_vel_range"), DEFAULT_COMMANDS, strict=True):
        low, high = CFG["command_cfg"][name]
        if low != high or float(low) != float(value):
            raise ValueError(f"CFG['command_cfg'][{name!r}] must be the singleton [{value}, {value}]")


def perturbation_from_seed(seed, perturbation, num_dof=NUM_ACTIONS):
    """Draw one reset perturbation from ``seed``; Go2 wrapper around the shared draw.

    The draw itself is ``examples/neuroevolution/seeding.perturbation_from_seed``, which makes
    ``num_dof`` mandatory because the joint count belongs to the task; this wrapper supplies the
    Go2 action count as the default the callers here use.

    Args:
        seed (int): Episode seed of the generation's seed table.
        perturbation (Perturbation): Maximum absolute offsets (metres / radians / radians).
        num_dof (int): Length of the joint offset vector, the Go2 action count by default.

    Returns:
        dict: ``base_pos`` (3,) float32, ``base_quat`` (4,) float32 w-first, ``joint_pos``
        (``num_dof``,) float32.
    """
    return _perturbation_from_seed(seed, perturbation, num_dof)


def perturbed_init_qpos(init_qpos, seeds, perturbation):
    """Map the generation's ordered episode seeds to their 19-vector init qpos.

    Each row is ``base_pos (3) | base_quat (4) | dof_pos (num_dof)``: the init base position plus
    the sampled ``base_pos`` offset, the init base quaternion rotated by the sampled ``base_rpy``
    (``q_pert * q_init``, so the perturbation acts in world axes after the authored orientation;
    the init quaternion here is the identity but the composition is what generalizes), and the
    init joint positions plus the sampled ``joint_pos`` offsets, in the qpos joint order.

    Args:
        init_qpos (array-like): ``(7 + num_dof,)`` unperturbed init qpos.
        seeds (array-like): Ordered episode seeds, one row each.
        perturbation (Perturbation): Maximum absolute offsets.

    Returns:
        np.ndarray: ``(len(seeds), 7 + num_dof)`` float32.

    Raises:
        ValueError: If ``init_qpos`` does not hold ``7 + NUM_ACTIONS`` values.
    """
    init_qpos = np.asarray(init_qpos, dtype=np.float32).reshape(-1)
    if init_qpos.size != 7 + NUM_ACTIONS:
        raise ValueError(f"init_qpos must hold {7 + NUM_ACTIONS} values, got {init_qpos.size}")
    seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
    init_quat = np.ascontiguousarray(init_qpos[3:7])
    out = np.empty((seeds.size, init_qpos.size), dtype=np.float32)
    for index, seed in enumerate(seeds):
        drawn = perturbation_from_seed(int(seed), perturbation)
        out[index, :3] = init_qpos[:3] + drawn["base_pos"]
        out[index, 3:7] = transform_quat_by_quat(init_quat, drawn["base_quat"])
        out[index, 7:] = init_qpos[7:] + drawn["joint_pos"]
    return out


def _to_host_float(value):
    """Move one device tensor to the host as float32 (read side of the boundary)."""
    return tensor_to_array(value).astype(np.float32)


def _to_host_bool(value):
    """Move one device tensor to the host as bool (read side of the boundary)."""
    return tensor_to_array(value).astype(bool)


#: Hamilton-product coefficients ``H[k, i, j]`` of ``(u * v)_k = sum_ij H[k, i, j] * u_i * v_j``,
#: transcribed from ``genesis.utils.geom._tc_geom_constants`` (``H[0]`` is the conjugate sign
#: diagonal, ``H[k, 0, k] = H[k, k, 0] = 1`` the scalar-vector blocks, the rest the epsilon terms).
_QUAT_MUL_COEFFICIENTS = np.zeros((4, 4, 4), dtype=np.float32)
_QUAT_MUL_COEFFICIENTS[0] = np.diag(np.array((1.0, -1.0, -1.0, -1.0), dtype=np.float32))
_QUAT_MUL_COEFFICIENTS[1, 0, 1] = _QUAT_MUL_COEFFICIENTS[1, 1, 0] = 1.0
_QUAT_MUL_COEFFICIENTS[2, 0, 2] = _QUAT_MUL_COEFFICIENTS[2, 2, 0] = 1.0
_QUAT_MUL_COEFFICIENTS[3, 0, 3] = _QUAT_MUL_COEFFICIENTS[3, 3, 0] = 1.0
_QUAT_MUL_COEFFICIENTS[(3, 1, 2, 3, 1, 2), (1, 2, 3, 2, 3, 1), (2, 3, 1, 1, 2, 3)] = (1.0, 1.0, 1.0, -1.0, -1.0, -1.0)


def _quat_mul(u, v):
    """Hamilton product of w-first quaternions, normalized, in the torch coefficient form.

    Args:
        u (np.ndarray): ``(..., 4)`` w-first quaternions.
        v (np.ndarray): ``(..., 4)`` w-first quaternions, broadcast against ``u``.

    Returns:
        np.ndarray: ``(..., 4)``, ``quatmul(u, v)`` up to the consumed normalization.
    """
    quat = (u[..., None, :, None] * v[..., None, None, :] * _QUAT_MUL_COEFFICIENTS).sum(axis=(-2, -1))
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


class Go2NeatEnv(BatchedEpisodeEnv):
    """Batched, seeded evaluation environment of the TensorNEAT Go2 walking trainer.

    Subclassing :class:`env_base.BatchedEpisodeEnv` is the declaration that this object is a
    task environment of the shared neuroevolution layer; together with the
    ``episodes_per_policy`` class attribute it is what makes the env usable by
    ``trainer.NeuroevolutionTrainer``.

    Args:
        pop_size (int): Policies per generation; the scene has ``EPISODES_PER_POLICY * pop_size``
            rows, rows ``3p, 3p+1, 3p+2`` belonging to policy ``p``.
        master_seed (int): Seed of the run; see ``seeding.generation_seed_table`` and
            ``seeding.evolution_seed``.
        perturbation (Perturbation): Maximum reset offsets; the all-zero default reproduces the
            fixed reset of ``go2_env``.
        horizon (int | None): ``scene.step()`` calls per generation; ``None`` means
            ``max_episode_length``, which never times out.
        episode_length_s (float | None): Episode cap in seconds, ``None`` keeps the 20 s of
            ``go2_train.get_cfgs()``; it fixes ``max_episode_length``.
        backend (str): ``evoxc`` backend of the compiled graphs (``"numpy"`` by default).
        device (str | evoxc.Device | None): ``evoxc`` device of the compiled graphs.
        show_viewer (bool): Open the Genesis viewer.
        seed_table (array-like | None): ``(num_generations, EPISODES_PER_POLICY)`` int table; when
            given it replaces the table ``seeding.generation_seed_table`` would draw.
        stop_when_done (bool): Stop a generation as soon as every row has ended.
        residency (str): ``"numpy"`` (the reference: host arrays at every graph boundary) or
            ``"device"`` (zero-copy: every operand stays on the ``evoxc`` device, see
            ``go2_device_rollout``).  ``"device"`` needs a non-CPU ``evoxc`` device.

    Attributes:
        rows (int): Number of environment rows, ``3 * pop_size`` (also ``n_envs``).
        episodes_per_policy (int): ``seeding.EPISODES_PER_POLICY``, the row tiling factor.
        max_episode_length (int): ``ceil(episode_length_s / dt)``, baked into the advance graph.
        cache (ExecutableCache): The one cache the four executables are built and looked up in.
        executables (callable): Return the four executables, compiling only the missing ones; call it
            once per generation as the compile-once proof (every call after ``__init__`` is a hit).
    """

    episodes_per_policy = EPISODES_PER_POLICY

    def __init__(
        self,
        pop_size,
        master_seed,
        *,
        perturbation=Perturbation(),
        horizon=None,
        episode_length_s=None,
        backend="numpy",
        device=None,
        show_viewer=False,
        seed_table=None,
        stop_when_done=True,
        residency="numpy",
    ):
        _check_cfg()
        self.pop_size = int(pop_size)
        if self.pop_size < 1:
            raise ValueError(f"pop_size must be positive, got {pop_size}")
        self.master_seed = int(master_seed)
        self.perturbation = perturbation
        self.n_envs = self.rows = EPISODES_PER_POLICY * self.pop_size
        self.backend, self.device = resolve_backend_device(backend, device)
        self.device_name = _device_name(self.device)
        self.show_viewer = bool(show_viewer)
        self.stop_when_done = bool(stop_when_done)
        if residency not in ("numpy", "device"):
            raise ValueError(f"residency must be 'numpy' or 'device', got {residency!r}")
        self.residency = residency
        if self.residency == "device" and self.device.kind == "cpu":
            raise ValueError(
                "residency='device' keeps every graph operand on the evoxc device, which needs a "
                f"non-CPU device backed by IREE (e.g. backend='iree', device='rocm:0' or 'cuda:0'); "
                f"got backend={self.backend!r} on {self.device_name} (a host array is the only "
                "operand a cpu-kind device accepts, so use residency='numpy' there)"
            )

        if seed_table is None:
            self.seed_table = None
        else:
            self.seed_table = np.asarray(seed_table, dtype=np.int64)
            if self.seed_table.ndim != 2 or self.seed_table.shape[1] != EPISODES_PER_POLICY:
                raise ValueError(
                    f"seed_table must be (num_generations, {EPISODES_PER_POLICY}), got {self.seed_table.shape}"
                )

        self.dt = DT
        self.action_scale = CFG["action_scale"]
        self.max_episode_length = math.ceil(
            (CFG["episode_length_s"] if episode_length_s is None else float(episode_length_s)) / self.dt
        )
        self.horizon = self.max_episode_length if horizon is None else int(horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")

        if not gs._initialized:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # A locomotion policy keeps fewer than ~20 collision pairs alive; a low cap saves
                # memory and exceeding it raises.
                max_collision_pairs=20,
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=self.show_viewer,
        )
        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=CFG["base_init_pos"],
                quat=CFG["base_init_quat"],
            ),
        )
        self.scene.build(n_envs=self.rows)

        # ``dof_start`` is the scene-level dof index of a joint; the robot is the only entity with
        # dofs here, so these double as the entity-local indices ``go2_env`` uses.
        motors_dof_idx = [self.robot.get_joint(name).dof_start for name in CFG["joint_names"]]
        self.motors_dof_idx = np.asarray(motors_dof_idx, dtype=np.int32)
        self.actions_dof_idx = np.argsort(self.motors_dof_idx)
        self.robot.set_dofs_kp([CFG["kp"]] * NUM_ACTIONS, self.motors_dof_idx)
        self.robot.set_dofs_kv([CFG["kd"]] * NUM_ACTIONS, self.motors_dof_idx)

        self._motors_dof_idx_np = np.asarray(motors_dof_idx, dtype=np.int64)
        # The free joint owns dofs 0..5, so the 12 actuated dofs are contiguous from the smallest
        # motor index; that same offset is the slot of a control joint inside the qpos dof block.
        dof_low = int(self._motors_dof_idx_np.min())
        dof_high = int(self._motors_dof_idx_np.max()) + 1
        self._control_dof_slice = slice(dof_low, dof_high)
        self._qpos_dof_slot = self._motors_dof_idx_np - dof_low
        self.default_dof_pos = np.array(
            [CFG["default_joint_angles"][name] for name in CFG["joint_names"]], dtype=np.float32
        )

        # Initial state, exactly as ``go2_env`` builds it: the dof block of the init qpos follows
        # ``robot.joints[1:]`` (URDF order), the control vector follows ``joint_names``.
        init_dof_qpos = np.array(
            [CFG["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]], dtype=np.float32
        )
        if init_dof_qpos.size != NUM_ACTIONS:
            raise ValueError(f"the robot must have {NUM_ACTIONS} actuated joints, got {init_dof_qpos.size}")
        self.init_qpos = np.concatenate(
            (
                np.asarray(CFG["base_init_pos"], dtype=np.float32),
                np.asarray(CFG["base_init_quat"], dtype=np.float32),
                init_dof_qpos,
            )
        )
        self._init_quat = None
        self._inv_init_quat = None
        self._zero_obs_inputs = {
            "base_ang_vel_world": np.zeros((self.rows, 3), dtype=np.float32),
            "base_quat": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (self.rows, 1)),
            "dof_vel": np.zeros((self.rows, NUM_ACTIONS), dtype=np.float32),
            "actions": np.zeros((self.rows, NUM_ACTIONS), dtype=np.float32),
        }

        # The four executables are compiled once, here, and every generation afterwards reuses
        # them through the same cache (that is what ``cache_hits`` counts).
        self.cache = ExecutableCache()
        self._algo, self._policy_config, self._trace_pop = self._build_policy_trace()
        self._pop_shapes = tuple(np.shape(leaf) for leaf in self._trace_pop)
        self.obs_executable, self.policy_executable, self.advance_executable, self.fitness_executable = (
            self.executables()
        )
        logger.info(
            "Go2NeatEnv: pop_size=%d rows=%d horizon=%d max_episode_length=%d backend=%s device=%s",
            self.pop_size,
            self.rows,
            self.horizon,
            self.max_episode_length,
            self.backend,
            self.device_name,
        )

    def _build_policy_trace(self):
        """Trace the genome pipeline once to obtain ``(algo, config, pop_arrays)``.

        ``build_policy`` needs the algorithm instance and the tensor-free static config of a
        ``Pipeline.setup()``; tracing that pipeline on the reference backend is cheap and gives
        the payload shapes the policy graph must be built with.  The policy executable itself is
        built on ``self.backend`` / ``self.device``.  The problem is only a stub
        (``trainer.TraceOnlyProblem``) carrying the two genome widths, and the payload is read
        with the shared ``trainer.pop_arrays_of``.
        """
        gen = genome.DefaultGenome(
            num_inputs=NUM_OBS,
            num_outputs=NUM_ACTIONS,
            max_nodes=MAX_NODES,
            max_conns=MAX_CONNS,
        )
        # Only the forward pass is traced, so the species structure is irrelevant; it still has to
        # be admissible for a small population (``species_size * min_species_size <= pop_size``).
        algo = algorithm.NEAT(gen, pop_size=self.pop_size, species_size=min(SPECIES_SIZE, self.pop_size))
        trace_problem = _TraceOnlyProblem(NUM_OBS, NUM_ACTIONS)
        pipeline = Pipeline(algo, trace_problem, seed=TRACE_SEED, backend="numpy", is_detailed=False)
        state = pipeline.setup()
        return algo, pipeline._static_config(state), pop_arrays_of(state)

    def executables(self):
        """Return the four executables, compiling only the ones the cache does not hold yet."""
        cache = self.cache
        obs = cache.get(
            ("obs", self.rows, self.backend, self.device_name),
            lambda: build_obs(self.rows, commands=DEFAULT_COMMANDS, backend=self.backend, device=self.device),
        )
        policy = cache.get(
            ("policy", self.rows, *self._pop_shapes, self.backend, self.device_name),
            lambda: build_policy(
                self._algo,
                self._policy_config,
                self._trace_pop,
                self.rows,
                backend=self.backend,
                device=self.device,
            ),
        )
        advance = cache.get(
            ("advance", self.rows, self.max_episode_length, self.backend, self.device_name),
            lambda: build_advance(
                self.rows,
                max_episode_length=self.max_episode_length,
                commands=DEFAULT_COMMANDS,
                dt=self.dt,
                backend=self.backend,
                device=self.device,
            ),
        )
        fitness = cache.get(
            ("fitness", self.pop_size, EPISODES_PER_POLICY, self.backend, self.device_name),
            lambda: build_fitness(self.pop_size, backend=self.backend, device=self.device),
        )
        return obs, policy, advance, fitness

    def _seeds_for(self, generation):
        """Return the ``(EPISODES_PER_POLICY,)`` int64 seeds of one generation."""
        if self.seed_table is not None:
            if not 0 <= generation < self.seed_table.shape[0]:
                raise IndexError(
                    f"generation {generation} is outside the supplied seed table "
                    f"({self.seed_table.shape[0]} generations)"
                )
            return self.seed_table[generation]
        return generation_seed_table(self.master_seed, generation + 1)[generation]

    def read_state(self):
        """Read the simulated state into host arrays; the whole device -> host boundary.

        ``base_quat`` is relative to the per-row init pose and the velocities are world-frame,
        which is what ``go2_neat_kernels.run_advance`` expects; ``base_rpy_deg`` follows the
        ``genesis.utils.geom.quat_to_xyz`` convention (w-first, roll/pitch/yaw in degrees).

        Returns:
            dict: ``base_pos`` (rows, 3), ``base_quat`` (rows, 4), ``base_lin_vel_world``
            (rows, 3), ``base_ang_vel_world`` (rows, 3), ``dof_pos`` (rows, 12),
            ``dof_vel`` (rows, 12), ``base_rpy_deg`` (rows, 3), ``error_mask`` (rows,) -- all
            numpy, all float32 except ``error_mask``.
        """
        if self._inv_init_quat is None:
            raise RuntimeError("read_state() needs the per-row init pose; call evaluate() first")
        base_quat = _to_host_float(self.robot.get_quat())
        base_quat_rel = _quat_mul(base_quat, self._inv_init_quat)
        return {
            "base_pos": _to_host_float(self.robot.get_pos()),
            "base_quat": base_quat_rel,
            "base_lin_vel_world": _to_host_float(self.robot.get_vel()),
            "base_ang_vel_world": _to_host_float(self.robot.get_ang()),
            "dof_pos": _to_host_float(self.robot.get_dofs_position(self.motors_dof_idx)),
            "dof_vel": _to_host_float(self.robot.get_dofs_velocity(self.motors_dof_idx)),
            "base_rpy_deg": quat_to_xyz(base_quat_rel, rpy=True, degrees=True),
            "error_mask": _to_host_bool(self.scene.get_error_envs_mask()),
        }

    def _write_targets(self, targets):
        """Feed the PD controllers with joint-order host targets, ``(rows, 12)`` float32."""
        self.robot.control_dofs_position(
            np.ascontiguousarray(targets, dtype=np.float32)[:, self.actions_dof_idx], self._control_dof_slice
        )

    def _write_reset(self, qpos, envs_idx=None):
        """Teleport rows ``envs_idx`` (all when ``None``) to ``qpos`` and kill their velocities."""
        self.robot.set_qpos(
            np.ascontiguousarray(qpos, dtype=np.float32),
            envs_idx=None if envs_idx is None else np.asarray(envs_idx, dtype=np.int32),
            zero_velocity=True,
            skip_forward=True,
        )

    def _reset_generation(self, row_init_qpos):
        """Apply one generation's init qpos to every row and return the initial observation.

        The observation is computed from the known reset values (zero base angular velocity,
        identity base orientation relative to the init pose, the init joint positions, zero joint
        velocities and zero actions), exactly like ``go2_env.reset()`` -- no simulation step is
        needed to produce it.
        """
        self._init_quat = np.ascontiguousarray(row_init_qpos[:, 3:7], dtype=np.float32)
        self._inv_init_quat = inv_quat(self._init_quat)
        self._write_reset(row_init_qpos)
        self.scene.clear_errno()  # drop any flag the previous generation left behind
        dof_pos = np.ascontiguousarray(row_init_qpos[:, 7:][:, self._qpos_dof_slot])
        return run_obs(
            self.obs_executable,
            base_ang_vel_world=self._zero_obs_inputs["base_ang_vel_world"],
            base_quat=self._zero_obs_inputs["base_quat"],
            dof_pos=dof_pos,
            dof_vel=self._zero_obs_inputs["dof_vel"],
            actions=self._zero_obs_inputs["actions"],
            commands=DEFAULT_COMMANDS,
        )

    def evaluate(self, pop_arrays, generation):
        """Simulate one generation and return its fitness plus the run bookkeeping.

        The returned mapping is checked against ``env_base.validate_evaluation`` (the contract of
        :class:`env_base.BatchedEpisodeEnv`) once per generation before it is handed back, for both
        residencies.

        Args:
            pop_arrays (tuple): The four genome payload arrays (``node_keys``, ``node_attrs``,
                ``conn_keys``, ``conn_attrs``) of the generation, in the shapes the policy graph
                was built with; handed to the policy graph.
            generation (int): Generation index, selecting the row of the seed table.

        Returns:
            dict: ``seeds`` (3,) int64, ``episode_returns`` (pop_size, 3) float32, ``returns``
            (rows,) float32, ``fitness`` (pop_size,) float32, ``steps`` (``scene.step()`` calls of
            this generation), ``env_steps`` (``steps * rows``), ``wall_time_s``,
            ``steps_per_s``, ``env_steps_per_s``, ``cache_hits``, ``cache_misses``,
            ``end_reasons`` (counts of ``timeout`` / ``tilted`` / ``error`` / ``unfinished``) and
            ``init_qpos`` (3, 19).

        Raises:
            IndexError: If ``generation`` is outside the supplied seed table.
        """
        generation = int(generation)
        pop_arrays = tuple(pop_arrays)
        seeds = self._seeds_for(generation)
        init_qpos = perturbed_init_qpos(self.init_qpos, seeds, self.perturbation)
        row_init_qpos = np.tile(init_qpos, (self.pop_size, 1))

        if self.residency == "device":
            # Imported here so the numpy/CPU path never imports the IREE bridge.
            from go2_device_rollout import evaluate_device

            result = evaluate_device(
                self,
                executables=(
                    self.obs_executable,
                    self.policy_executable,
                    self.advance_executable,
                    self.fitness_executable,
                ),
                pop_arrays=pop_arrays,
                seeds=seeds,
                init_qpos=init_qpos,
                row_init_qpos=row_init_qpos,
                horizon=self.horizon,
                stop_when_done=self.stop_when_done,
            )
        else:
            result = self._evaluate_numpy(pop_arrays, seeds, init_qpos, row_init_qpos)
        validate_evaluation(result, self.pop_size, self.episodes_per_policy)
        return result

    def _evaluate_numpy(self, pop_arrays, seeds, init_qpos, row_init_qpos):
        """The host-staged rollout of one generation (the reference implementation)."""
        start = time.perf_counter()
        obs = self._reset_generation(row_init_qpos)
        returns = np.zeros(self.rows, dtype=np.float32)
        alive = np.ones(self.rows, dtype=bool)
        episode_length = np.zeros(self.rows, dtype=np.int32)
        last_actions = np.zeros((self.rows, NUM_ACTIONS), dtype=np.float32)
        ending = np.full(self.rows, END_REASONS["unfinished"], dtype=np.int8)
        steps = 0
        while steps < self.horizon:
            raw_actions = run_policy(self.policy_executable, pop_arrays, obs)
            self._write_targets(last_actions * self.action_scale + self.default_dof_pos)
            self.scene.step()
            steps += 1

            state = self.read_state()
            step = run_advance(
                self.advance_executable,
                base_pos=state["base_pos"],
                base_quat=state["base_quat"],
                base_lin_vel_world=state["base_lin_vel_world"],
                base_ang_vel_world=state["base_ang_vel_world"],
                dof_pos=state["dof_pos"],
                dof_vel=state["dof_vel"],
                actions=raw_actions,
                last_actions=last_actions,
                returns=returns,
                alive=alive,
                episode_length=episode_length,
                error_mask=state["error_mask"],
                commands=DEFAULT_COMMANDS,
            )
            obs = step["obs"]
            returns = step["returns"]
            alive = step["alive"]
            episode_length = step["episode_length"]

            # ``last_actions`` is what the next step executes; the graph used the same clipping,
            # so the control target matches the ``action_rate`` term of the reward.
            last_actions = np.clip(raw_actions, -CLIP_ACTIONS, CLIP_ACTIONS).astype(np.float32)

            done = step["done"]
            if done.any():
                ending = self._classify_endings(ending, done, state["base_rpy_deg"], state["error_mask"])
                reset_idx = np.flatnonzero(done)
                self._write_reset(row_init_qpos[reset_idx], reset_idx)
                last_actions[reset_idx] = 0.0
            if state["error_mask"].any():
                # The graph already latched the error into ``alive``/``returns``, and a row that is dead
                # is never reset again, so the flag has to be dropped here or the next ``scene.step()``
                # raises (see 'Scene.get_error_envs_mask').
                self.scene.clear_errno()
            if self.stop_when_done and not alive.any():
                break
        wall_time_s = time.perf_counter() - start

        fitness = run_fitness(self.fitness_executable, returns)
        env_steps = steps * self.rows
        result = {
            "seeds": seeds,
            "episode_returns": returns.reshape(self.pop_size, self.episodes_per_policy).copy(),
            "returns": returns,
            "fitness": fitness,
            "steps": steps,
            "env_steps": env_steps,
            "wall_time_s": wall_time_s,
            "steps_per_s": steps / wall_time_s,
            "env_steps_per_s": env_steps / wall_time_s,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "end_reasons": {name: int(np.count_nonzero(ending == code)) for name, code in END_REASONS.items()},
            "init_qpos": init_qpos,
        }
        return result

    @staticmethod
    def _classify_endings(ending, done, base_rpy_deg, error_mask):
        """Tag the rows that just ended, for reporting only (the graphs own the decision).

        ``error`` is the externally signalled failure, ``tilted`` is a roll or pitch beyond
        ``termination_if_*_greater_than`` in the ``quat_to_xyz`` convention, and everything else
        that ends is the episode ``timeout``.  A row is tagged once, on its first ending.
        """
        pending = done & (ending == END_REASONS["unfinished"])
        error = pending & error_mask
        tilted = pending & ~error_mask & (np.abs(base_rpy_deg[:, :2]).max(axis=1) > TERMINATION_LIMIT_DEG)
        timeout = pending & ~error_mask & ~tilted
        ending[error] = END_REASONS["error"]
        ending[tilted] = END_REASONS["tilted"]
        ending[timeout] = END_REASONS["timeout"]
        return ending

    def close(self):
        """Destroy the scene and release the compiled executables (idempotent)."""
        if self.scene is not None:
            self.scene.destroy()
            self.scene = None
        self.cache.clear()


"""Batched, seeded evaluation environment for the TensorNEAT Go2 walking trainer.

One :class:`Go2NeatEnv` owns a single ``gs.Scene`` with ``EPISODES_PER_POLICY * pop_size`` Go2
rows (policy-major: rows ``3p, 3p+1, 3p+2`` are the episodes of policy ``p``) and scores one
generation per :meth:`Go2NeatEnv.evaluate` call.  Genesis owns the physics and the state, the
compiled graphs of ``go2_neat_kernels`` own the observation / policy / reward / termination math,
and this module is only the boundary in between.

Seeding
-------
``np.random.SeedSequence(master_seed).spawn(2)`` splits the master seed into an independent
evolution stream (:func:`evolution_seed`, for the trainer's own RNG) and an env stream
(:func:`generation_seed_table`).  Every generation owns ``EPISODES_PER_POLICY`` distinct episode
seeds; a row keeps its generation's seed -- and therefore its reset perturbation -- for the whole
generation, and the three seed-derived 19-vector init qpos are replicated to every policy's rows.

Resets
------
The three init qpos are applied once per generation.  Episodes that restart after a ``done`` are
discarded by design: the episode's reward is already booked, the row keeps stepping its physics
from the same init qpos with zeroed actions, and only the *first* ending of each row is reported
in ``end_reasons``.

``horizon`` is the number of ``scene.step()`` calls per generation.  ``None`` means
``max_episode_length`` (one full episode, so nothing times out); a horizon larger than
``max_episode_length`` is what makes the timeout fire and the auto-reset run.

Action latency
--------------
``go2_train`` runs with ``simulate_action_latency=True``: step ``t`` executes the *clipped*
actions of step ``t - 1`` (zero at reset), exactly like ``go2_env.step()``.  The graphs see the
raw, unclipped action of step ``t`` and clip it themselves.

Initialization
--------------
``gs.init`` is called only when ``genesis`` is not initialized yet (``gs._initialized``), with
CPU/float32; a caller that inits first (e.g. ``gs.init(backend=gs.gpu)``) keeps its backend and
precision.  The ``backend``/``device`` arguments select the ``evoxc`` backend of the compiled
graphs only, via ``go2_evox_kernels.resolve_backend_device``.

Torch <-> host round trips
--------------------------
Every conversion is confined to :meth:`Go2NeatEnv.read_state` (device -> host) and
:meth:`Go2NeatEnv._write_targets` / :meth:`Go2NeatEnv._write_reset` (host -> device):

* read with ``.detach().cpu().numpy().astype(np.float32)``: ``robot.get_pos()``, ``get_quat()``,
  ``get_vel()``, ``get_ang()``, ``get_dofs_position(motors_dof_idx)``,
  ``get_dofs_velocity(motors_dof_idx)``, the base quaternion relative to the per-row init pose,
  and its roll/pitch/yaw (the relative quaternion and its ``quat_to_xyz`` are evaluated on the
  device, in torch);
* read with ``.detach().cpu().numpy().astype(bool)``: ``scene.get_error_envs_mask()``;
* write with ``torch.as_tensor(..., dtype=gs.tc_float, device=gs.device)``: the PD targets and
  the per-row reset qpos;
* ``motors_dof_idx`` / ``actions_dof_idx`` are built once in ``__init__`` (they are static) and
  only their ``numpy`` copies index host arrays.

Nothing else leaves the device: ``run_policy`` / ``run_obs`` / ``run_advance`` / ``run_fitness``
are numpy-facing (the wrappers stage their own operands), so the observations, actions and returns
stay on the host.

This module has no CLI; like the other ``*_env.py`` modules it is excluded from the example smoke
tests.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch
from tensorneat import Pipeline, algorithm, genome, problem
from tensorneat.common.tools import to_numpy

import genesis as gs

from genesis.utils.geom import inv_quat, quat_to_xyz, transform_quat_by_quat, xyz_to_quat

from go2_evox_kernels import resolve_backend_device
from go2_neat_kernels import (
    CLIP_ACTIONS,
    DEFAULT_COMMANDS,
    EPISODES_PER_POLICY,
    NUM_ACTIONS,
    NUM_OBS,
    REWARD_SCALES,
    TERMINATION_LIMIT_DEG,
    TRACKING_SIGMA,
    ExecutableCache,
    Perturbation,
    build_advance,
    build_fitness,
    build_obs,
    build_policy,
    run_advance,
    run_fitness,
    run_obs,
    run_policy,
)

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


def generation_seed_table(master_seed, num_generations, episodes=EPISODES_PER_POLICY):
    """Return the ``(num_generations, episodes)`` int64 table of episode seeds.

    The table is drawn from the env stream of ``SeedSequence(master_seed).spawn(2)`` (the
    evolution stream is :func:`evolution_seed`), one row at a time: ``rng.choice(2**31, size=
    episodes, replace=False)`` per row keeps a row's seeds distinct and, unlike one big
    ``replace=False`` draw, keeps the table **prefix-stable** -- ``table(seed, N)[i]`` is identical
    for every ``N > i``.

    Args:
        master_seed (int): Seed of the whole run.
        num_generations (int): Number of generation rows to materialize.
        episodes (int): Episodes per policy and row length.

    Returns:
        np.ndarray: ``(num_generations, episodes)`` int64, every row strictly distinct.
    """
    num_generations = int(num_generations)
    episodes = int(episodes)
    if num_generations < 0 or episodes < 1:
        raise ValueError(f"num_generations must be >= 0 and episodes >= 1, got {num_generations} and {episodes}")
    _, env_seq = np.random.SeedSequence(master_seed).spawn(2)
    rng = np.random.default_rng(env_seq)
    table = np.empty((num_generations, episodes), dtype=np.int64)
    for generation in range(num_generations):
        table[generation] = rng.choice(2**31, size=episodes, replace=False)
    return table


def evolution_seed(master_seed):
    """Return the unsigned 32-bit seed of the evolution stream of ``master_seed``.

    This is the sibling of the env stream :func:`generation_seed_table` draws from, so the
    trainer's evolution RNG is independent of -- and unaffected by -- the episode seeds.

    Args:
        master_seed (int): Seed of the whole run.

    Returns:
        int: Non-negative seed that fits in ``uint32``.
    """
    evolution_seq, _ = np.random.SeedSequence(master_seed).spawn(2)
    return int(evolution_seq.generate_state(1, dtype=np.uint32)[0])


def _uniform(rng, amplitude, size):
    """Return ``size`` offsets uniform in ``[-amplitude, +amplitude]``, consuming ``size`` draws."""
    amplitude = float(amplitude)
    return rng.uniform(-amplitude, amplitude, size=size)


def perturbation_from_seed(seed, perturbation, num_dof=NUM_ACTIONS):
    """Draw one reset perturbation from ``seed`` alone.

    The draw order is fixed (and independent of the amplitudes, so the stream never shifts):
    first the three ``base_pos`` offsets, then the three ``base_rpy`` angles, then the
    ``num_dof`` ``joint_pos`` offsets, all uniform per component in ``[-p, +p]``.  A zero
    ``Perturbation`` draws nothing and returns exact zeros, with ``base_quat == [1, 0, 0, 0]``.

    ``base_quat`` is the standard w-first roll-pitch-yaw quaternion of ``base_rpy``
    (``genesis.utils.geom.xyz_to_quat(..., rpy=True)``), i.e. the small base rotation in world axes.

    Args:
        seed (int): Episode seed of the generation's seed table.
        perturbation (Perturbation): Maximum absolute offsets (metres / radians / radians).
        num_dof (int): Length of the joint offset vector.

    Returns:
        dict: ``base_pos`` (3,) float32, ``base_quat`` (4,) float32 w-first, ``joint_pos``
        (``num_dof``,) float32.
    """
    num_dof = int(num_dof)
    if perturbation.is_zero:
        return {
            "base_pos": np.zeros(3, dtype=np.float32),
            "base_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "joint_pos": np.zeros(num_dof, dtype=np.float32),
        }
    rng = np.random.default_rng(int(seed))
    base_pos = _uniform(rng, perturbation.base_pos, 3)
    base_rpy = _uniform(rng, perturbation.base_rpy, 3)
    joint_pos = _uniform(rng, perturbation.joint_pos, num_dof)
    return {
        "base_pos": base_pos.astype(np.float32),
        "base_quat": np.asarray(xyz_to_quat(base_rpy, rpy=True), dtype=np.float32),
        "joint_pos": joint_pos.astype(np.float32),
    }


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
        out[index, 3:7] = transform_quat_by_quat(drawn["base_quat"], init_quat)
        out[index, 7:] = init_qpos[7:] + drawn["joint_pos"]
    return out


def _device_name(device):
    """Return the stable ``"kind:index"`` spelling of a resolved ``evoxc`` device."""
    return f"{device.kind}:{device.index}"


def _to_host_float(value):
    """Move one device tensor to the host as float32 (read side of the boundary)."""
    return value.detach().cpu().numpy().astype(np.float32)


def _to_host_bool(value):
    """Move one device tensor to the host as bool (read side of the boundary)."""
    return value.detach().cpu().numpy().astype(bool)


def _to_device_float(array):
    """Move one host array to ``gs.device`` as float32 (write side of the boundary)."""
    return torch.as_tensor(np.ascontiguousarray(array), dtype=gs.tc_float, device=gs.device)


class _TraceOnlyProblem(problem.BaseProblem):
    """Stand-in problem for the throw-away pipeline that only supplies the genome statics.

    ``build_policy`` needs a TensorNEAT ``(algo, config, pop_arrays)`` triple, and the config is
    the tensor-free ``Pipeline._static_config`` of a ``Pipeline.setup()``.  Nothing here is ever
    evaluated: the fitness comes from the simulation, not from the problem.
    """

    input_shape = (NUM_OBS,)
    output_shape = (NUM_ACTIONS,)

    def evaluate(self, state, keys, act_fn, transformed):
        raise NotImplementedError("Go2NeatEnv only traces the genome pipeline, it never evaluates a problem")


class Go2NeatEnv:
    """Batched, seeded evaluation environment of the TensorNEAT Go2 walking trainer.

    Args:
        pop_size (int): Policies per generation; the scene has ``EPISODES_PER_POLICY * pop_size``
            rows, rows ``3p, 3p+1, 3p+2`` belonging to policy ``p``.
        master_seed (int): Seed of the run; see :func:`generation_seed_table` and
            :func:`evolution_seed`.
        perturbation (Perturbation): Maximum reset offsets; the all-zero default reproduces the
            fixed reset of ``go2_env``.
        horizon (int | None): ``scene.step()`` calls per generation; ``None`` means
            ``max_episode_length``, which never times out.
        backend (str): ``evoxc`` backend of the compiled graphs (``"numpy"`` by default).
        device (str | evoxc.Device | None): ``evoxc`` device of the compiled graphs.
        show_viewer (bool): Open the Genesis viewer.
        seed_table (array-like | None): ``(num_generations, EPISODES_PER_POLICY)`` int table; when
            given it replaces the table :func:`generation_seed_table` would draw.
        stop_when_done (bool): Stop a generation as soon as every row has ended.

    Attributes:
        rows (int): Number of environment rows, ``3 * pop_size`` (also ``n_envs``).
        max_episode_length (int): ``ceil(episode_length_s / dt)``, baked into the advance graph.
        cache (ExecutableCache): The one cache the four executables are built and looked up in.
    """

    def __init__(
        self,
        pop_size,
        master_seed,
        *,
        perturbation=Perturbation(),
        horizon=None,
        backend="numpy",
        device=None,
        show_viewer=False,
        seed_table=None,
        stop_when_done=True,
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
        self.max_episode_length = math.ceil(CFG["episode_length_s"] / self.dt)
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
        self.motors_dof_idx = torch.tensor(motors_dof_idx, dtype=gs.tc_int, device=gs.device)
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
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
            self._executables()
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
        built on ``self.backend`` / ``self.device``.
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
        pipeline = Pipeline(algo, _TraceOnlyProblem(), seed=TRACE_SEED, backend="numpy", is_detailed=False)
        state = pipeline.setup()
        pop_arrays = tuple(np.array(to_numpy(leaf)) for leaf in state.pop)
        return algo, pipeline._static_config(state), pop_arrays

    def _executables(self):
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
        base_quat_rel = transform_quat_by_quat(self._inv_init_quat, self.robot.get_quat())
        return {
            "base_pos": _to_host_float(self.robot.get_pos()),
            "base_quat": _to_host_float(base_quat_rel),
            "base_lin_vel_world": _to_host_float(self.robot.get_vel()),
            "base_ang_vel_world": _to_host_float(self.robot.get_ang()),
            "dof_pos": _to_host_float(self.robot.get_dofs_position(self.motors_dof_idx)),
            "dof_vel": _to_host_float(self.robot.get_dofs_velocity(self.motors_dof_idx)),
            "base_rpy_deg": _to_host_float(quat_to_xyz(base_quat_rel, rpy=True, degrees=True)),
            "error_mask": _to_host_bool(self.scene.get_error_envs_mask()),
        }

    def _write_targets(self, targets):
        """Feed the PD controllers with joint-order host targets, ``(rows, 12)`` float32."""
        target = _to_device_float(targets)
        self.robot.control_dofs_position(target[:, self.actions_dof_idx], self._control_dof_slice)

    def _write_reset(self, qpos, envs_idx=None):
        """Teleport rows ``envs_idx`` (all when ``None``) to ``qpos`` and kill their velocities."""
        if envs_idx is not None:
            envs_idx = torch.as_tensor(np.asarray(envs_idx), dtype=gs.tc_int, device=gs.device)
        self.robot.set_qpos(_to_device_float(qpos), envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

    def _reset_generation(self, row_init_qpos):
        """Apply one generation's init qpos to every row and return the initial observation.

        The observation is computed from the known reset values (zero base angular velocity,
        identity base orientation relative to the init pose, the init joint positions, zero joint
        velocities and zero actions), exactly like ``go2_env.reset()`` -- no simulation step is
        needed to produce it.
        """
        self._init_quat = _to_device_float(row_init_qpos[:, 3:7])
        self._inv_init_quat = inv_quat(self._init_quat)
        self._write_reset(row_init_qpos)
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

        Args:
            pop_arrays (tuple): The four genome payload arrays (``node_keys``, ``node_attrs``,
                ``conn_keys``, ``conn_attrs``) of the generation, in the shapes the policy graph
                was built with; passed straight to ``run_policy``.
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
            if self.stop_when_done and not alive.any():
                break
        wall_time_s = time.perf_counter() - start

        fitness = run_fitness(self.fitness_executable, returns)
        env_steps = steps * self.rows
        return {
            "seeds": seeds,
            "episode_returns": returns.reshape(self.pop_size, EPISODES_PER_POLICY).copy(),
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


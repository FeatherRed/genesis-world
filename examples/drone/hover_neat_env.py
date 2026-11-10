"""Batched, seeded evaluation environment for the TensorNEAT drone hover task.

One :class:`HoverNeatEnv` owns a single ``gs.Scene`` with ``EPISODES_PER_POLICY * pop_size``
drone rows (policy-major: rows ``3p, 3p+1, 3p+2`` are the three episodes of policy ``p``) and
scores one generation per :meth:`HoverNeatEnv.evaluate` call.  Genesis owns the physics and the
state, the compiled graphs of ``hover_kernels`` own the policy / observation / reward /
termination math, and this module is only the boundary in between.

One episode per row
-------------------
Every row is a **single episode**: it is reset once per generation to the generation's init pose
(``BASE_INIT_POS`` / ``BASE_INIT_QUAT`` plus the optional perturbation) and then stepped until it
ends, with no in-episode reset.  That is the deliberate divergence from
``examples/drone/hover_env.py``, the PPO baseline this migration replaces:

* the baseline's **reset-step reward pass is not reproduced**.  ``hover_env.step`` calls
  ``reset_idx`` *before* computing the reward, so a reset row's ``episode_sums`` and
  ``last_actions`` are zeroed and the post-reset step's reward lands on the new episode.  Here a
  truncated PPO episode return therefore includes one extra post-reset step that this eval model
  does not, and ``last_actions`` is never zeroed inside an episode (``smooth`` always compares
  against the genuinely previous action);
* the baseline resets a row mid-rollout and keeps stepping it; here a row that ends simply stops
  being credited (``alive`` goes false and never re-arms), so the ``EPISODES_PER_POLICY`` rows of
  one policy are ``EPISODES_PER_POLICY`` independent episodes rather than one long rollout;
* the roll/pitch 180-degree termination terms of the baseline are unreachable (Euler roll lives
  in ``[-180, 180]`` and pitch in ``[-90, 90]``), so the graphs drop them.

Seeding (every rule comes from ``examples/neuroevolution/seeding.py``)
----------------------------------------------------------------------
``np.random.SeedSequence(master_seed).spawn(2)`` splits the master seed into an independent
evolution stream (:func:`seeding.evolution_seed`, for the trainer's own RNG) and an env
stream (:func:`seeding.generation_seed_table`).  Every *generation* owns one ordered triple
of episode seeds, and the **same triple** scores every policy of that generation, so a fitness
comparison is fair; a fresh triple is drawn per generation.  Row ``3p + e`` (episode slot ``e``)
draws its per-step command stream from its own ``np.random.Generator``, seeded from that slot's
episode seed and the generation index: a row's command stream is therefore independent of every
other row's timing and of the step order.

Reset perturbation
------------------
The optional ``perturbation`` (a :class:`seeding.Perturbation`) offsets every row's init pose.
Row ``3p + e`` is reset to ``BASE_INIT_POS + draw["base_pos"]`` and
``normalize(draw["base_quat"] * BASE_INIT_QUAT)``, where the draw comes from
:func:`seeding.perturbation_from_seed` on episode slot ``e``'s seed.  Rows ``3p, 3p+1, 3p+2``
therefore take the three slots of one seed triple, so every policy of a generation is reset under
the same three perturbations -- the paired-comparison property of the episode seeds, extended to
the init pose.  The default ``Perturbation()`` is all-zero and short-circuits to the exact tiled
``BASE_INIT_POS`` / ``BASE_INIT_QUAT`` constants, so an unperturbed run is bit-for-bit the
fixed-reset task.

Command resampling
------------------
The baseline draws commands only for the rows that reached their target (and again inside
``reset_idx``).  Here the host burns **three uniforms per row per step** (``x ~ U(-1, 1)``,
``y ~ U(-1, 1)``, ``z ~ U(1, 1)``, pinned) and the advance graph selects the draw only for the
rows that reached their target.  Same distribution as the baseline, different exact sequence: a
fixed number of draws keeps every row's stream in phase.

Errors
------
A row reported in ``scene.get_error_envs_mask()`` is **killed, never reset again, and its reward
is not credited** (its measured state is meaningless); ``scene.clear_errno()`` is called for
every such row and once per generation reset, or the next ``scene.step()`` re-raises the latched
solver error.

Torch
-----
This module must not ``import torch``, and no torch tensor operation appears on the environment
path.  ``residency="numpy"`` (the default) is the correctness-first boundary: the Genesis reads go
through ``genesis.utils.misc.tensor_to_array`` (``drone.get_pos`` / ``get_quat`` / ``get_vel`` /
``get_ang``, ``scene.get_error_envs_mask``) and the writes pass numpy arrays straight to the setters
(``array_like`` inputs, ``envs_idx`` included), so every read and every write crosses the device
boundary exactly once.  ``residency="device"`` is the opt-in zero-copy device-resident rollout,
implemented in ``hover_device_rollout.py`` and imported only when that residency is selected; it is
torch-free as well, so a whole generation of it runs in a process where ``import torch`` fails.

The module has no CLI; like the other ``*_env.py`` modules it is excluded from the example smoke
tests.  ``hover_neat_check.py`` is its smoke/parity CLI.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
from tensorneat import Pipeline, algorithm, genome
from tensorneat.common.tools import to_numpy

import genesis as gs
from genesis.utils.misc import tensor_to_array

from hover_kernels import (
    BASE_INIT_POS,
    BASE_INIT_QUAT,
    DT,
    END_REASONS,
    NUM_ACTIONS,
    NUM_OBS,
    build_advance,
    build_controls,
    build_reset_obs,
    run_advance,
    run_controls,
    run_reset_obs,
)
from hover_device_kernels import build_episode_returns

# Importing ``hover_kernels`` also puts the shared ``examples/neuroevolution`` layer on
# ``sys.path``; the environment contract, the seeding rules, the executable cache and the
# fitness/policy graphs live there now, not in the task module.
from env_base import (  # noqa: E402  (path set up above)
    BatchedEpisodeEnv,
    validate_evaluation,
    validate_evaluation_device,
)
from evolution_graphs import ExecutableCache, build_fitness, run_fitness  # noqa: E402
from evox_runtime import resolve_backend_device  # noqa: E402
from policy_graphs import build_policy, run_policy  # noqa: E402
from seeding import (  # noqa: E402
    EPISODES_PER_POLICY,
    Perturbation,
    generation_seed_table,
    perturbation_from_seed,
)
from trainer import TraceOnlyProblem  # noqa: E402  (path set up above)

logger = logging.getLogger(__name__)

#: Episode cap of ``hover_train.get_cfgs()`` in seconds; it fixes ``max_episode_length``.
EPISODE_LENGTH_S = 15.0

# Genome shape the policy graph is traced with; ``pop_arrays`` must keep these payload shapes.
MAX_NODES = 80
MAX_CONNS = 640
SPECIES_SIZE = 3
TRACE_SEED = 0

#: Rollout residencies: host staging (default) or the device-resident rollout module.
RESIDENCIES = ("numpy", "device")


class HoverNeatEnv:
    """Batched, seeded evaluation environment of the TensorNEAT drone hover task.

    It implements the shared :class:`env_base.BatchedEpisodeEnv` contract: one
    :meth:`evaluate` call simulates one generation of ``EPISODES_PER_POLICY * pop_size``
    policy-major rows and returns that generation's episode returns, and :meth:`close` releases
    the scene and the executables.

    Two rollout residencies are selectable.  ``residency="numpy"`` (the default) is the
    correctness-first host staging: every state read and every actuator write crosses the device
    boundary once, as a numpy array.  ``residency="device"`` is the opt-in zero-copy
    device-resident rollout implemented in the companion ``hover_device_rollout.py`` module, which
    is torch-free too; it is imported lazily, inside :meth:`evaluate`, so numpy mode never imports it.

    Args:
        pop_size (int): Policies per generation; the scene has ``EPISODES_PER_POLICY * pop_size``
            rows, rows ``3p, 3p+1, 3p+2`` belonging to policy ``p``.
        master_seed (int): Seed of the run; see the module docstring.
        horizon (int | None): ``scene.step()`` calls per generation; ``None`` means
            ``max_episode_length``, so nothing times out.
        episode_length_s (float | None): Episode cap in seconds, ``None`` keeps the 15 s of
            ``hover_train.get_cfgs()``; it fixes ``max_episode_length``.
        backend (str): ``evoxc`` backend of the compiled graphs (``"numpy"`` by default).
        device (str | evoxc.Device | None): ``evoxc`` device of the compiled graphs.
        show_viewer (bool): Open the Genesis viewer.
        seed_table (array-like | None): ``(num_generations, EPISODES_PER_POLICY)`` int table; when
            given it replaces the table :func:`seeding.generation_seed_table` would draw.
        stop_when_done (bool): Stop a generation as soon as every row has ended.
        perturbation (seeding.Perturbation | None): Maximum reset offsets of the init pose;
            ``None`` is the all-zero :class:`seeding.Perturbation`, i.e. the fixed reset.
        residency (str): ``"numpy"`` (default, host-staged rollout) or ``"device"`` (the
            device-resident rollout of ``hover_device_rollout.py``).

    Attributes:
        rows (int): Number of environment rows, ``3 * pop_size`` (also ``n_envs``).
        episodes_per_policy (int): ``EPISODES_PER_POLICY``, the rows one policy owns.
        max_episode_length (int): ``ceil(episode_length_s / dt)``, baked into the advance graph.
        horizon (int): ``scene.step()`` calls of one generation.
        backend (str), device (evoxc.Device), device_name (str): The resolved graph target.
        residency (str): The rollout residency; ``device`` also owns the sixth executable, the
            episode-returns reshape graph (``episode_returns_executable``).
        perturbation (seeding.Perturbation): The reset perturbation amplitudes in force.
        cache (ExecutableCache): The one cache the five (device residency: six) executables are
            built and looked up in.
        executables (callable): Return the executables, compiling only the missing ones; call it
            once per generation as the compile-once proof (every call after ``__init__`` is a hit).
    """

    def __init__(
        self,
        pop_size,
        master_seed,
        *,
        horizon=None,
        episode_length_s=None,
        backend="numpy",
        device=None,
        show_viewer=False,
        seed_table=None,
        stop_when_done=True,
        perturbation=None,
        residency="numpy",
    ):
        self.pop_size = int(pop_size)
        if self.pop_size < 1:
            raise ValueError(f"pop_size must be positive, got {pop_size}")
        self.master_seed = int(master_seed)
        self.episodes_per_policy = EPISODES_PER_POLICY
        self.n_envs = self.rows = EPISODES_PER_POLICY * self.pop_size
        self.backend, self.device = resolve_backend_device(backend, device)
        self.device_name = f"{self.device.kind}:{self.device.index}"
        self.show_viewer = bool(show_viewer)
        self.stop_when_done = bool(stop_when_done)

        self.residency = str(residency)
        if self.residency not in RESIDENCIES:
            raise ValueError(f"residency must be one of {RESIDENCIES}, got {residency!r}")
        if self.residency == "device" and self.backend == "numpy":
            raise ValueError("residency='device' needs a device evoxc backend, not the numpy backend")

        if perturbation is None:
            self.perturbation = Perturbation()
        elif isinstance(perturbation, Perturbation):
            self.perturbation = perturbation
        else:
            raise TypeError(f"perturbation must be a seeding.Perturbation, got {type(perturbation).__name__}")

        if seed_table is None:
            self.seed_table = None
        else:
            self.seed_table = np.asarray(seed_table, dtype=np.int64)
            if self.seed_table.ndim != 2 or self.seed_table.shape[1] != EPISODES_PER_POLICY:
                raise ValueError(
                    f"seed_table must be (num_generations, {EPISODES_PER_POLICY}), got {self.seed_table.shape}"
                )

        self.dt = DT
        self.max_episode_length = math.ceil(
            (EPISODE_LENGTH_S if episode_length_s is None else float(episode_length_s)) / self.dt
        )
        self.horizon = self.max_episode_length if horizon is None else int(horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")

        if not gs._initialized:
            gs.init(backend=gs.cpu, precision="32", logging_level="warning")

        # The baseline's physics, verbatim: 100 Hz control steps over 200 Hz rigid substeps with
        # collisions, joint limits and the Newton constraint solver.
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt / 2,
                enable_collision=True,
                enable_joint_limit=True,
                constraint_solver=gs.constraint_solver.Newton,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            show_viewer=self.show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.drone = self.scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf"))
        self.scene.build(n_envs=self.rows)

        self._reset_pos = np.ascontiguousarray(np.tile(BASE_INIT_POS, (self.rows, 1)))
        self._reset_quat = np.ascontiguousarray(np.tile(BASE_INIT_QUAT, (self.rows, 1)))

        # The executables are compiled once, here, and every generation afterwards reuses them
        # through the same cache (that is what ``cache_hits`` counts).
        self.cache = ExecutableCache()
        # The declaration that this env satisfies the shared contract: every attribute and method
        # the protocol requires exists by now (``cache`` is the last one built), so a drift in the
        # contract fails here instead of in the trainer.
        missing = [
            name
            for name in ("rows", "episodes_per_policy", "max_episode_length", "cache", "evaluate", "close")
            if not hasattr(self, name)
        ]
        if missing or not isinstance(self, BatchedEpisodeEnv):
            raise TypeError(
                "HoverNeatEnv must satisfy the BatchedEpisodeEnv contract; "
                f"missing attributes: {', '.join(missing) if missing else '(none, check the method signatures)'}"
            )
        self._algo, self._policy_config, self._trace_pop = self._build_policy_trace()
        self._pop_shapes = tuple(np.shape(leaf) for leaf in self._trace_pop)
        executables = self.executables()
        (
            self.controls_executable,
            self.reset_obs_executable,
            self.policy_executable,
            self.advance_executable,
            self.fitness_executable,
        ) = executables[:5]
        # The device residency drives one more graph, the ``(rows,)`` -> ``(pop_size, episodes)``
        # reshape of the episode returns (see ``executables``).
        (self.episode_returns_executable,) = executables[5:] if self.residency == "device" else (None,)
        logger.info(
            "HoverNeatEnv: pop_size=%d rows=%d horizon=%d max_episode_length=%d backend=%s device=%s "
            "residency=%s perturbation=%s",
            self.pop_size,
            self.rows,
            self.horizon,
            self.max_episode_length,
            self.backend,
            self.device_name,
            self.residency,
            self.perturbation,
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
        pipeline = Pipeline(
            algo,
            TraceOnlyProblem(NUM_OBS, NUM_ACTIONS),
            seed=TRACE_SEED,
            backend="numpy",
            is_detailed=False,
        )
        state = pipeline.setup()
        pop_arrays = tuple(np.array(to_numpy(leaf)) for leaf in state.pop)
        return algo, pipeline._static_config(state), pop_arrays

    def initial_pop_arrays(self):
        """The untouched population the trace pipeline drew; a CLI can score it as-is."""
        return self._trace_pop

    def executables(self):
        """Return the env-side executables, compiling only the ones the cache does not hold yet.

        The numpy residency drives the five graphs ``hover_kernels`` owns (controls, reset
        observation, policy, advance, fitness).  The device residency additionally drives the
        ``episode_returns`` graph of ``hover_device_kernels`` -- the ``(rows,)`` ->
        ``(pop_size, episodes)`` reshape of the returns, which cannot be a host-side ``reshape``
        without leaving the device -- which is why the tuple grows to six there.
        """
        cache = self.cache
        controls = cache.get(
            ("controls", self.rows, self.backend, self.device_name),
            lambda: build_controls(self.rows, backend=self.backend, device=self.device),
        )
        reset_obs = cache.get(
            ("reset_obs", self.rows, self.backend, self.device_name),
            lambda: build_reset_obs(self.rows, backend=self.backend, device=self.device),
        )
        policy = cache.get(
            ("policy", self.rows, *self._pop_shapes, self.backend, self.device_name),
            lambda: build_policy(
                self._algo,
                self._policy_config,
                self._trace_pop,
                self.rows,
                num_inputs=NUM_OBS,
                num_outputs=NUM_ACTIONS,
                episodes=EPISODES_PER_POLICY,
                backend=self.backend,
                device=self.device,
            ),
        )
        advance = cache.get(
            ("advance", self.rows, self.max_episode_length, self.backend, self.device_name),
            lambda: build_advance(
                self.rows,
                max_episode_length=self.max_episode_length,
                backend=self.backend,
                device=self.device,
            ),
        )
        fitness = cache.get(
            ("fitness", self.pop_size, EPISODES_PER_POLICY, self.backend, self.device_name),
            lambda: build_fitness(self.pop_size, backend=self.backend, device=self.device),
        )
        if self.residency != "device":
            return controls, reset_obs, policy, advance, fitness
        episode_returns = cache.get(
            ("episode_returns", self.pop_size, EPISODES_PER_POLICY, self.backend, self.device_name),
            lambda: build_episode_returns(self.pop_size, backend=self.backend, device=self.device),
        )
        return controls, reset_obs, policy, advance, fitness, episode_returns

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

    @staticmethod
    def _draw_commands(rng):
        """Draw one ``(3,)`` command row, burning the three uniforms of the baseline.

        ``z`` is drawn from its singleton range ``[1, 1]``: the draw is burned anyway, so a row's
        stream never shifts.
        """
        return np.array(
            [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(1.0, 1.0)],
            dtype=np.float32,
        )

    def _command_generators(self, seeds, generation):
        """One ``np.random.Generator`` per row, seeded from its episode slot and the generation."""
        seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
        return [
            np.random.default_rng([int(seeds[row % EPISODES_PER_POLICY]), int(generation)])
            for row in range(self.rows)
        ]

    @staticmethod
    def _compose_quat(draw_quat, init_quat):
        """``normalize(draw_quat * init_quat)``: the seeding rule's ``q_pert * q_init``.

        Both operands are w-first Hamilton quaternions, and the product is normalized the way
        ``genesis.utils.geom.transform_quat_by_quat`` normalizes it, so the composed rotation is a
        unit quaternion by construction.
        """
        w0, x0, y0, z0 = (float(value) for value in np.asarray(draw_quat).reshape(-1))
        w1, x1, y1, z1 = (float(value) for value in np.asarray(init_quat).reshape(-1))
        quat = np.array(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dtype=np.float64,
        )
        norm = np.linalg.norm(quat)
        if norm == 0.0:
            raise ValueError(f"the composed init quaternion is null (draw={draw_quat!r}, init={init_quat!r})")
        return (quat / norm).astype(np.float32)

    def _init_poses(self, seeds):
        """Return the ``(rows, 3)`` / ``(rows, 4)`` init pose of one generation, perturbation applied.

        Row ``r`` takes episode slot ``e = r % EPISODES_PER_POLICY`` and therefore ``seeds[e]``, so
        rows ``3p, 3p+1, 3p+2`` of a policy share the generation's three perturbations.
        An all-zero :class:`seeding.Perturbation` returns the exact tiled ``BASE_INIT_POS`` /
        ``BASE_INIT_QUAT`` constants, with nothing round-tripping through float arithmetic.
        """
        seeds = np.asarray(seeds, dtype=np.int64).reshape(-1)
        if self.perturbation.is_zero:
            return (
                np.ascontiguousarray(np.tile(BASE_INIT_POS, (self.rows, 1))),
                np.ascontiguousarray(np.tile(BASE_INIT_QUAT, (self.rows, 1))),
            )
        positions = np.empty((self.rows, 3), dtype=np.float32)
        quats = np.empty((self.rows, 4), dtype=np.float32)
        for slot in range(EPISODES_PER_POLICY):
            draw = perturbation_from_seed(int(seeds[slot]), self.perturbation, num_dof=0)
            positions[slot::EPISODES_PER_POLICY] = BASE_INIT_POS + draw["base_pos"]
            quats[slot::EPISODES_PER_POLICY] = self._compose_quat(draw["base_quat"], BASE_INIT_QUAT)
        return positions, quats

    def reset_generation(self, generation=0, rngs=None):
        """Reset every row to the init pose and return ``(commands, obs)`` of the generation start.

        The commands are the first draw of each row's generator, taken **before** the first step,
        so the analytic reset observation already shows the episode's first target -- exactly like
        ``hover_env.reset()``.  No simulation step is needed to produce the observation.  Pass the
        generation's already-built ``rngs`` to keep a running step loop in phase (its next draw must
        be the one after this initial command, not the initial command again).

        The per-row init pose is recomputed here from the generation's seeds and this env's
        ``perturbation``, and kept as ``self._reset_pos`` / ``self._reset_quat``.
        """
        seeds = self._seeds_for(int(generation))
        if rngs is None:
            rngs = self._command_generators(seeds, generation)
        commands = np.stack([self._draw_commands(rng) for rng in rngs], axis=0).astype(np.float32)

        self._reset_pos, self._reset_quat = self._init_poses(seeds)
        # The base pose of a free link *is* its qpos, so writing qpos sets exactly the state ``set_pos`` /
        # ``set_quat`` set (measured bit-identical, before and after stepping), without their fixed-link guard:
        # a Torch-free process walks that guard over the sanitized device ``links_idx`` and reads past its end,
        # which refused every perturbed (env-specific) reset. The drone owns no fixed link, so the
        # ``batch_fixed_verts`` morph option cannot avoid the guard.
        self.drone.set_qpos(
            np.concatenate((self._reset_pos, self._reset_quat), axis=1), zero_velocity=True, envs_idx=None
        )
        self.drone.zero_all_dofs_velocity()
        self.scene.clear_errno()  # drop any flag the previous generation left behind
        obs = run_reset_obs(
            self.reset_obs_executable,
            commands=commands,
            base_init_pos=self._reset_pos,
            base_init_quat=self._reset_quat,
        )["obs"]
        return commands, obs

    def read_state(self):
        """Read the simulated base state into host arrays; the whole device -> host boundary.

        Returns:
            dict: ``base_pos`` (rows, 3), ``base_quat`` (rows, 4, the raw absolute
            ``drone.get_quat()``), ``base_lin_vel_world`` (rows, 3), ``base_ang_vel_world``
            (rows, 3), ``error_mask`` (rows,) -- all numpy, float32 except ``error_mask``.
        """
        return {
            "base_pos": tensor_to_array(self.drone.get_pos()),
            "base_quat": tensor_to_array(self.drone.get_quat()),
            "base_lin_vel_world": tensor_to_array(self.drone.get_vel()),
            "base_ang_vel_world": tensor_to_array(self.drone.get_ang()),
            "error_mask": tensor_to_array(self.scene.get_error_envs_mask()).astype(bool),
        }

    def evaluate(self, pop_arrays, generation):
        """Simulate one generation and return its fitness plus the run bookkeeping.

        With ``residency="numpy"`` the rows are stepped on the host-staged loop below, and the
        assembled result is checked against the shared contract with
        :func:`env_base.validate_evaluation`.  With ``residency="device"`` the call is dispatched to
        ``hover_device_rollout.DeviceRollout``, whose four training leaves stay on the device and are
        checked with :func:`env_base.validate_evaluation_device` instead.

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
            ``end_reasons`` (counts of ``timeout`` / ``crash`` / ``error`` / ``unfinished``) and
            ``commands`` (rows, 3) the final targets.  The four training leaves are numpy arrays in
            the numpy residency and device tensors in the device residency.

        Raises:
            IndexError: If ``generation`` is outside the supplied seed table.
            ValueError: If the assembled result violates the shared evaluation contract.
        """
        if self.residency == "device":
            # Imported here, not at module level: the device rollout is the only part of this stack
            # that speaks IREE directly, and numpy mode must never import it.
            from hover_device_rollout import DeviceRollout

            result = DeviceRollout(
                self,
                (
                    self.controls_executable,
                    self.reset_obs_executable,
                    self.policy_executable,
                    self.advance_executable,
                    self.fitness_executable,
                    self.episode_returns_executable,
                ),
            ).evaluate(pop_arrays, generation)
            validate_evaluation_device(result, self.pop_size, self.episodes_per_policy, self.device)
            return result

        generation = int(generation)
        pop_arrays = tuple(pop_arrays)
        seeds = self._seeds_for(generation)
        rngs = self._command_generators(seeds, generation)
        self.executables()  # compile-once proof: every call after ``__init__`` is a cache hit

        start = time.perf_counter()
        commands, obs = self.reset_generation(generation, rngs)
        returns = np.zeros(self.rows, dtype=np.float32)
        alive = np.ones(self.rows, dtype=bool)
        episode_length = np.zeros(self.rows, dtype=np.int32)
        last_actions = np.zeros((self.rows, NUM_ACTIONS), dtype=np.float32)
        last_base_pos = self._reset_pos
        ending = np.full(self.rows, END_REASONS["unfinished"], dtype=np.int8)
        steps = 0
        while steps < self.horizon and (not self.stop_when_done or alive.any()):
            controls = run_controls(self.controls_executable, run_policy(self.policy_executable, pop_arrays, obs))
            self.drone.set_propellers_rpm(controls["rpm"])  # once per scene.step(), or it raises
            self.scene.step()
            steps += 1

            state = self.read_state()
            # Every row burns its three command draws every step, reached target or not.
            resampled_commands = np.stack([self._draw_commands(rng) for rng in rngs], axis=0).astype(np.float32)
            step = run_advance(
                self.advance_executable,
                commands=commands,
                base_pos=state["base_pos"],
                last_base_pos=last_base_pos,
                base_quat=state["base_quat"],
                base_lin_vel_world=state["base_lin_vel_world"],
                base_ang_vel_world=state["base_ang_vel_world"],
                actions=controls["actions"],
                last_actions=last_actions,
                episode_length=episode_length,
                resampled_commands=resampled_commands,
                error_mask=state["error_mask"],
                alive=alive,
                returns=returns,
                end_code=ending,
            )
            obs = step["obs"]
            returns = step["returns"]
            alive = step["alive"]
            episode_length = step["episode_length"]
            commands = step["commands"]
            ending = step["end_code"]  # the graph owns the latch: error > timeout > crash, once
            # The graph's snapshot is a fresh buffer, so the next ``get_pos()`` cannot alias it.
            last_base_pos = step["base_pos"]
            last_actions = controls["actions"]

            if state["error_mask"].any():
                # The graph already latched the error into ``done``/``alive``/``returns``, and a row
                # that is dead is never reset again, so the flag has to be dropped here or the next
                # ``scene.step()`` raises (see 'Scene.get_error_envs_mask').
                self.scene.clear_errno()
        wall_time_s = time.perf_counter() - start

        fitness = run_fitness(self.fitness_executable, returns)
        env_steps = steps * self.rows
        result = {
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
            "commands": commands,
        }
        validate_evaluation(result, self.pop_size, episodes=self.episodes_per_policy)
        return result

    def close(self):
        """Destroy the scene and release the compiled executables (idempotent)."""
        if self.scene is not None:
            self.scene.destroy()
            self.scene = None
        self.cache.clear()

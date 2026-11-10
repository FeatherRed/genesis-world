"""Pilot-level parity of the Torch-free, GPU-resident Go2 TensorNEAT trainer.

``examples/locomotion`` carries two implementations of one Go2 walking generation: the host
(``residency="numpy"``) reference of ``go2_neat_env.Go2NeatEnv``, which stages every graph operand
through numpy, and the device-resident (``residency="device"``) loop of ``go2_device_rollout``, which
keeps every operand on an ``evoxc`` device and runs in a process where ``import torch`` fails.  This
module pins what the two residencies promise, at *pilot* level: the per-step observation, reward,
bookkeeping and termination masks, the documented end-to-end numbers of the trainer, and the first
population update.

What is pinned
--------------
1. **Per-step parity on one population** (``test_per_step_observation_and_reward_parity``): the same
   generation-0 population is fed to both residencies; every advance step's observation, reward,
   ``returns``, ``alive``, ``done`` and ``episode_length`` is compared, and the six
   ``go2_legacy_reference.walking_reward_terms`` are recomputed from *each* path's recorded inputs and
   checked against that path's own graph reward.
2. **Termination mask and timeout** (``test_termination_mask_and_timeout_parity``): with
   ``episode_length_s=0.1`` (``max_episode_length=5``) and a horizon of 40, both residencies must
   report the same step count and ``end_reasons``, the same per-step ``done`` mask, a reset
   ``episode_length`` on every done row, and an ``alive`` mask that never re-arms.
3. **Documented end-to-end numbers** (``test_two_generation_end_to_end_reference_numbers``): each
   residency's *own* generation-0 population (device-first initialization), two generations,
   compared against the seed table and fitness history ``examples/locomotion/CONTEXT.md`` documents.
4. **First population update** (``test_first_population_update_parity``): the post-``tell`` payload,
   species leaves, ``randkey`` and generation counter of a one-generation run.
5. **Full horizon without Torch** (``test_full_horizon_reference_numbers_without_torch``, ``slow``):
   three generations of 1000 steps in a child interpreter where ``torch`` cannot be imported.

Two scenes are never alive at the same time
-------------------------------------------
Each ``Go2NeatEnv`` owns its own 24-row Go2 ``gs.Scene``, so the module never holds two of them: the
module-scoped ``default_campaign`` and ``timeout_campaign`` fixtures build one residency's scene, do
all of that residency's work on it (record one generation, then run the trainer), close it, and only
then build the other residency's scene.  Both residencies still share the *same* Genesis GPU backend:
the numpy env is built on ``gs.gpu`` as well, because the two numbers only compare when the physics
runs on the same device.

Why this module owns ``gs.init``/``gs.destroy``
----------------------------------------------
``pytestmark`` parametrizes ``backend=None``, which makes the suite's autouse ``initialize_genesis``
fixture a no-op: that fixture calls ``gs.destroy()`` after every test, while the envs built here must
outlive a single test (the module-scoped campaigns keep their scenes and executables for every test
that reads them), and ``gs.destroy()`` would pull the solver buffers out from under them.  The
module-scoped ``gpu_genesis`` fixture therefore initializes Genesis once, and destroys it once, at
module teardown.

Tolerances, with the value measured on this host
------------------------------------------------
AMD Instinct MI300X VF (gfx942), ROCm 7.1.0, evoxc 0.1.0, tensorneat 0.2.0, iree 3.13.0, evoxc
device ``rocm:0``, ``pop_size=8`` (24 rows), ``master_seed=0``, the default reset perturbation
(0.05 m / 0.05 rad / 0.05 rad), genesis ``precision="32"``.  Every bound has at least 3x headroom over
the largest value measured in a whole-module run, except the three marked *deliberately tight*:

* ``RESET_OBS_ATOL = 1e-6`` (measured 1.513e-09, 660x): the observation of step 0.  It is computed from
  the same staged values on both paths and only the float32 rounding of the inverse quaternion differs,
  so it is a reset-equivalence claim and not a rollout bound.
* ``OBS_ATOL = 1e-3`` (measured 1.984e-04, 5.0x): the full ``(rows, 45)`` observation of one step.  The
  reset observation starts at 1.513e-09 and the divergence then grows in the trailing 12 columns -- the
  action slot -- because the two residencies' float32 control-target arithmetic differs in the last bit
  and an untrained policy at a contact threshold amplifies it (``CONTEXT.md``: 2.4e-07 in, 2.2e-03 out
  over 21 steps).  The action-free block is bounded separately.
* ``OBS_STATE_ATOL = 2e-5`` (measured 3.7e-06, 5.4x): the 33 action-free observation columns of one step.
* ``REWARD_ATOL = 5e-5`` (measured 1.144e-05, 4.4x): one step's total reward.  Above the graph-level
  1e-6 for the same reason -- the reward is a function of the diverging state.
* ``RETURNS_ATOL = 5e-5`` (measured 7.63e-06, 6.6x): the accumulated episode return of one row.
* ``TERM_ATOL = 2e-5`` (measured 3.81e-06, 5.2x): one ``walking_reward_terms`` value of the five
  state-driven terms, compared across residencies.  ``action_rate`` is the exception:
  ``ACTION_RATE_ATOL = 5e-1`` (measured 9.86e-02, 5.1x), because it is the squared sum of the action
  differences and therefore quadratic in the diverging action vector.  That bound is still a relative
  1e-6 of the term itself, which is 4.8e5 in magnitude on both paths, and the *graph-level* agreement of
  the term is 1e-5, pinned in ``tests/test_go2_neat_kernels.py``.
* ``TERMS_SUM_ATOL = 1e-5`` (measured 3.81e-06): *deliberately tight*.  Each path's six terms summed
  against that same path's graph ``reward`` in the same process.  This is the check that a path's
  reward is the legacy formula, so it is not a cross-path chaos bound and must stay far below the
  cross-path bounds; widening it would let a wrong reward formula hide behind the rollout divergence.
* ``HISTORY_ATOL = 1e-5`` (measured 4.77e-07, 21x): one ``fitness_history`` entry, against the
  documented values and across residencies -- the convention ``tests/test_neuroevolution_device_
  parity.py`` uses.  *Deliberately tight*: it only absorbs the float32 reduction of one episode mean,
  never a rollout.
* ``CROSS_RESIDENCY_FITNESS_ATOL = 3e-3`` (measured 7.02e-04, 4.3x): one per-policy fitness of each
  residency's *own* generation-0 population.  Device-first initialization drifts by <= 2.4e-07 on the
  float genome leaves and that is amplified by the same conditional-path chaos, so no smaller constant
  is a "residency tolerance" -- the bit-exact claim is made in
  :func:`test_first_population_update_parity` instead, on one identical population.
* ``LEAF_ATOL = 1e-5`` (measured 2.384e-07, 42x): one float genome/species leaf after one ``tell``.

Skip reasons
------------
``pytest.importorskip`` skips the module where ``evoxc``/``tensorneat`` are missing (both are optional
and undeclared in ``pyproject.toml``); ``gpu_genesis`` skips it on a CPU-only host
(``tests.torch_block.has_gpu``, which falls back to the ``/dev/kfd`` device node where NVML/AMD SMI are
not installed); ``device_placement`` skips it when no non-CPU ``evoxc`` device accepts a write
(``tests.device_pilot.reachable_device_or_skip``, which can be pointed at a device through
``EVOXC_DEVICE``).  The in-process tier -- every test that builds a Genesis GPU scene here, i.e. both
campaign fixtures -- additionally skips where ``tests.device_pilot.unsafe_hsa_runtime_reason`` reports
that the worker maps PyTorch's bundled HSA runtime, with a reason naming the ``LD_PRELOAD`` remedy (see
the HSA section below); the torch-blocked child test is not gated, because its child repairs the
runtime itself.

One process, one module
-----------------------
The module runs in one process.  Two module-scoped campaigns build four Go2 scenes in total: one numpy
and one device scene for the default configuration (recording plus the two trainer runs share each
scene) and one of each for the ``episode_length_s=0.1`` timeout configuration; only one scene is alive
at a time.  Verified on this host with the system HSA runtime preloaded (see below): three consecutive
serial ``-n 0`` runs pass all five tests (178 s, 220 s and 222 s) and an ``-n 2`` run passes them in
197 s; ``-m required`` collects all five.  CI runs it under the suite's default
``-n auto --dist=worksteal``, which gives each test its own worker and therefore its own Genesis.

HSA runtime, and why the process must use the system one
--------------------------------------------------------
``gs.init`` re-executes a plain interpreter with the system ``libhsa-runtime64.so`` preloaded, because
torch's bundled copy corrupts AQL packets on ROCm rollouts.  Under ``pytest`` that repair is warn-only
(a re-exec would restart the session), so the worker keeps torch's bundled runtime, and then Genesis
GPU work can abort with ``HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`` ("The AQL packet is malformed",
surfacing as ``Queue ... aborting`` and as the ``qd_destroy_data`` abort seen here).  That is a
process-wide constraint, not a scene-count one: bare ``pytest`` aborted the *first* in-process Go2 scene
of a whole-module run, both while another job was training on the same GPU and again after that load had
lightened, and every such run passes once the process preloads the runtime the repair would pick.  The
in-process tier therefore requires the system HSA runtime to be preloaded on a ROCm host; this is the
exact command::

    LD_PRELOAD=/opt/rocm/lib/libhsa-runtime64.so.1 \\
        PYTHONPATH=$PWD:/tmp/gs-pytest-libs /root/evox-robotics/.venv/bin/python -m pytest \\
        tests/test_go2_device_parity.py -q -n 0 --durations=15

Without it the campaign tests skip with a reason naming that ``LD_PRELOAD`` path, instead of aborting
the worker.  ``tests.device_pilot.unsafe_hsa_runtime_reason`` decides, and it only gates the known-bad
case: a process that already preloaded the system runtime, and one whose PyTorch bundles no ROCm HSA
runtime (a CUDA host), run the tier as before.
The 1000-step case does not depend on that: it runs in a torch-blocked child through
``tests.torch_block.run_python_checked``, where torch is not importable at all, so the system runtime is
the one that gets loaded.  Every in-process rollout is short (13 / 6 / 21 steps).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

pytest.importorskip("evoxc")
pytest.importorskip("tensorneat")

#: ``iree`` truncates its ROCm compile of a reduction graph without this flag; ``build_tell`` passes it
#: itself for ``iree`` on cuda/rocm, and the module sets it up front for the graphs built here.
#: Source: ``examples/neuroevolution/CONTEXT.md`` "Device backends".
os.environ.setdefault("EVOXC_IREE_COMPILE_ARGS", "--iree-codegen-llvmgpu-use-reduction-vector-distribution=false")

pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: ``examples/`` are plain scripts, not a package, so the shared layer needs its directory on
#: ``sys.path`` before any bare-name import (the bootstrap ``test_go2_neat_kernels.py`` uses).
EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
LOCOMOTION_DIR = EXAMPLES_DIR / "locomotion"
NEUROEVOLUTION_DIR = EXAMPLES_DIR / "neuroevolution"
for _directory in (NEUROEVOLUTION_DIR, LOCOMOTION_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import go2_legacy_reference as legacy  # noqa: E402
import go2_neat_kernels as kernels  # noqa: E402
from seeding import EPISODES_PER_POLICY, Perturbation, evolution_seed  # noqa: E402
from trainer import NeuroevolutionTrainer  # noqa: E402

from tests.device_pilot import (  # noqa: E402
    as_host,
    assert_allclose,
    reachable_device_or_skip,
    skip_if_unsafe_hsa_runtime,
)
from tests.torch_block import has_gpu, run_python_checked  # noqa: E402


#: Policies per generation; 8 policies are 24 rows, the size ``verify_device.py`` measures at.
POP_SIZE = 8
#: Rows of the scene: ``EPISODES_PER_POLICY`` episodes per policy, policy-major.
ROWS = POP_SIZE * EPISODES_PER_POLICY
#: Seed of both RNG streams; the seed table and the evolution seed are documented for it.
MASTER_SEED = 0
#: ``go2_neat_train.py``'s default reset perturbation (``--perturb-*`` defaults).
PERTURBATION = Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.05)
#: ``scene.step()`` calls per generation: two short generations, tilted rows end them early.
HORIZON = 25
#: The two genome widths of the Go2 walking task (``kernels.NUM_OBS`` / ``kernels.NUM_ACTIONS``).
NUM_INPUTS = kernels.NUM_OBS
NUM_OUTPUTS = kernels.NUM_ACTIONS
#: The ``evoxc`` backend of the device residency (the only one the zero-copy bridge uses).
DEVICE_BACKEND = "iree"
#: The timeout probe: ``episode_length_s=0.1`` is ``max_episode_length=5`` at ``dt=0.02``.
TIMEOUT_EPISODE_LENGTH_S = 0.1
TIMEOUT_MAX_EPISODE_LENGTH = 5
TIMEOUT_HORIZON = 40
#: The full-horizon case: three generations of one whole episode (``CFG["episode_length_s"] == 20``).
FULL_HORIZON = 1000
FULL_HORIZON_GENERATIONS = 3
#: The child runs a HSA re-exec (~12 s), a cold scene build and 3000 steps; the module gives it 900 s.
FULL_HORIZON_TIMEOUT_S = 900

#: ``seeding.evolution_seed(0)``, documented in ``examples/neuroevolution/CONTEXT.md``.
DOCUMENTED_EVOLUTION_SEED = 3757552657
#: ``seeding.generation_seed_table(0, 3)``: the three generation rows, policy-shared.
DOCUMENTED_SEED_TABLE = [
    [1454269175, 757617120, 1421746521],
    [1313752747, 908599966, 53557616],
    [1683657442, 1375977993, 1654802006],
]
#: The two-generation, horizon-25 fitness history of ``examples/locomotion/CONTEXT.md``.
DOCUMENTED_HISTORY_HORIZON_25 = [-7.3784708976745605, -7.318042755126953]
#: The three-generation, full-horizon fitness history of the Torch-free device run.
DOCUMENTED_HISTORY_FULL_HORIZON = [-7.378471374511719, -7.318042755126953, -6.039944171905518]
#: ``steps`` of the two horizon-25 generations: the rows tilt at 21 and at 10 steps.
DOCUMENTED_STEPS_HORIZON_25 = [21, 10]

#: The reset observation of step 0, computed from the same staged values on both paths: measured
#: 1.513e-09 (the host and the graph round the inverse quaternion differently in float32), 660x.
RESET_OBS_ATOL = 1e-6
#: One full observation of one step, both residencies: measured 1.984e-04 (5.0x).
OBS_ATOL = 1e-3
#: The 33 action-free observation columns of one step: measured 3.7e-06 (5.4x).
OBS_STATE_ATOL = 2e-5
#: The action slot of the observation starts here (45 - 12).
OBS_ACTION_COLUMN = NUM_INPUTS - NUM_OUTPUTS
#: One step's total reward: measured 1.144e-05 (4.4x).
REWARD_ATOL = 5e-5
#: One row's accumulated episode return: measured 7.63e-06 (6.6x).
RETURNS_ATOL = 5e-5
#: One state-driven ``walking_reward_terms`` value across residencies: measured 3.81e-06 (5.2x).
TERM_ATOL = 2e-5
#: ``action_rate`` across residencies: measured 9.86e-02 (5.1x), quadratic in the diverging action
#: vector; the term itself is 4.8e5 in magnitude, so this is a relative bound of ~1e-6.
ACTION_RATE_ATOL = 5e-1
#: A path's six terms summed against that path's graph reward: measured 3.81e-06.  Deliberately not
#: widened to 3x: it compares one path with itself inside one process, so it is not a cross-path chaos
#: bound and must stay far below the cross-path ones.
TERMS_SUM_ATOL = 1e-5
#: One ``fitness_history`` entry, against the documented values and across residencies: measured
#: 4.77e-07 (21x).  Deliberately kept at the cross-residency 1e-5: it absorbs one float32 episode-mean
#: reduction, never a rollout.
HISTORY_ATOL = 1e-5
#: One per-policy fitness of each residency's own generation-0 population: measured 7.02e-04 (4.3x).
CROSS_RESIDENCY_FITNESS_ATOL = 3e-3
#: One float genome/species leaf after one ``tell``: measured 2.384e-07 (42x).
LEAF_ATOL = 1e-5

#: The six reward terms plus the ``dt``-scaled total, in ``walking_reward_terms`` order.
TERM_NAMES = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "base_height",
    "action_rate",
    "similar_to_default",
    "total",
)
#: Terms driven by the simulated state only; ``action_rate`` is driven by the action vector.
TERM_STATE_NAMES = tuple(name for name in TERM_NAMES if name != "action_rate")
#: The six leaves of the advance graph, in ``go2_neat_advance``'s return order.
ADVANCE_FIELDS = ("obs", "reward", "returns", "alive", "done", "episode_length")


@pytest.fixture(scope="module")
def gpu_genesis():
    """Initialize Genesis on its GPU backend once, and destroy it once at module teardown.

    ``pytestmark``'s ``parametrize("backend", [None])`` disables the suite's autouse
    ``initialize_genesis`` fixture; this module needs its own, module-scoped one because the envs here
    outlive a single test and ``gs.destroy()`` between tests would invalidate their solver state.
    """
    if not has_gpu():
        pytest.skip(
            "Go2 device-residency parity needs a GPU-backed Genesis: neither NVML nor AMD SMI reports a "
            "backend and no GPU device node (/dev/kfd, /dev/nvidiactl) exists"
        )
    import genesis as gs

    if gs._initialized:
        gs.destroy()
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    try:
        yield gs
    finally:
        gs.destroy()


@pytest.fixture(scope="module")
def safe_hsa_runtime():
    """Skip the in-process tier where Genesis' HSA re-exec repair cannot run (see the module docstring).

    It has to be resolved before ``gpu_genesis`` initializes Genesis and before a scene is built, which
    is why the two campaign fixtures request it first; the torch-blocked child test does not use it.
    """
    skip_if_unsafe_hsa_runtime()


@pytest.fixture(scope="module")
def device_placement():
    """``(backend, device)`` of a reachable non-CPU ``evoxc`` device, or a skip naming every failure."""
    return reachable_device_or_skip(DEVICE_BACKEND)


class RecordedGeneration(NamedTuple):
    """One recorded generation: its result mapping and one entry per advance step."""

    result: dict
    steps: list
    population: tuple
    max_episode_length: int


class TrainerRun(NamedTuple):
    """Host snapshot of one finished ``NeuroevolutionTrainer`` run (the env is already closed)."""

    fitness_history: list
    records: list
    seed_table: np.ndarray
    pop_arrays: tuple
    species_leaves: tuple
    randkey: int
    generation: float


@contextlib.contextmanager
def _open_env(residency, device, *, horizon=HORIZON, **kwargs):
    """Build one ``Go2NeatEnv`` on the shared Genesis GPU backend and always close it."""
    from go2_neat_env import Go2NeatEnv

    env = Go2NeatEnv(
        pop_size=POP_SIZE,
        master_seed=MASTER_SEED,
        perturbation=PERTURBATION,
        horizon=horizon,
        backend="numpy" if residency == "numpy" else DEVICE_BACKEND,
        device=None if residency == "numpy" else device,
        residency=residency,
        **kwargs,
    )
    try:
        yield env
    finally:
        env.close()


def _advance_fields(result) -> dict:
    """One advance result as comparable host arrays, masks and counter in their own dtype.
    The device path's leaves are ``evoxc`` device tensors (``_run_advance`` never hostifies them), so
    every leaf goes through ``as_host`` before it is compared with the numpy path's array.
    """
    fields = {name: np.array(as_host(result[name]), copy=True) for name in ADVANCE_FIELDS}
    fields["alive"] = fields["alive"].astype(bool)
    fields["done"] = fields["done"].astype(bool)
    fields["episode_length"] = fields["episode_length"].astype(np.int32)
    return fields


def _record_numpy_advance(monkeypatch) -> list:
    """Record ``go2_neat_env.run_advance``'s inputs and returned dict, step by step."""
    import go2_neat_env

    original = go2_neat_env.run_advance
    steps = []

    def recording(executable, **kwargs):
        result = original(executable, **kwargs)
        steps.append(
            {
                "inputs": {name: np.array(value, copy=True) for name, value in kwargs.items()},
                "advance": _advance_fields(result),
            }
        )
        return result

    monkeypatch.setattr(go2_neat_env, "run_advance", recording)
    return steps


def _record_device_advance(monkeypatch) -> list:
    """Record the device rollout's advance inputs and its six graph leaves, step by step.

    ``go2_device_rollout._DeviceRollout._run_advance`` receives the state-graph outputs and the
    device-resident bookkeeping and drops the graph's ``reward`` (the host never needs it), so the
    leaves are captured where they are produced: ``go2_device_rollout._outputs`` is the module-global
    normalizer every graph run goes through, and only a six-leaf call is the advance graph.
    """
    import go2_device_rollout

    original_advance = go2_device_rollout._DeviceRollout._run_advance
    original_outputs = go2_device_rollout._outputs
    steps = []
    pending = {}

    def recording_outputs(result):
        outputs = original_outputs(result)
        if len(outputs) == len(ADVANCE_FIELDS):
            pending["advance"] = _advance_fields(dict(zip(ADVANCE_FIELDS, outputs, strict=True)))
        return outputs

    def recording_advance(
        self, base_quat, dof_pos, dof_vel, actions, last_actions, returns, alive, episode_length, error_flag
    ):
        # The three values the numpy path reads from the getters are read here too, before the graph
        # runs; ``base_quat``/``dof_pos``/``dof_vel`` are the state graph's own outputs.
        inputs = {
            "base_pos": np.array(as_host(self.env.robot.get_pos()), copy=True),
            "base_quat": np.array(as_host(base_quat), copy=True),
            "base_lin_vel_world": np.array(as_host(self.env.robot.get_vel()), copy=True),
            "base_ang_vel_world": np.array(as_host(self.env.robot.get_ang()), copy=True),
            "dof_pos": np.array(as_host(dof_pos), copy=True),
            "dof_vel": np.array(as_host(dof_vel), copy=True),
            "actions": np.array(as_host(actions), copy=True),
            "last_actions": np.array(as_host(last_actions), copy=True),
        }
        result = original_advance(
            self, base_quat, dof_pos, dof_vel, actions, last_actions, returns, alive, episode_length, error_flag
        )
        steps.append({"inputs": inputs, "advance": pending.pop("advance")})
        return result

    monkeypatch.setattr(go2_device_rollout._DeviceRollout, "_run_advance", recording_advance)
    monkeypatch.setattr(go2_device_rollout, "_outputs", recording_outputs)
    return steps


def _recorded_generation(monkeypatch, residency, device, *, env=None, population=None, horizon=HORIZON, **kwargs):
    """Evaluate generation 0 of one residency with every advance step recorded.

    Args:
        monkeypatch (pytest.MonkeyPatch): Restores the recording hooks at test teardown.
        residency (str): ``"numpy"`` or ``"device"``.
        device (evoxc.Device | None): The device of the residency (``None`` for ``"numpy"``).
        env (Go2NeatEnv | None): An already open env of this residency to record on; ``None`` builds a
            private one just for this generation.
        population (tuple | None): The four genome payload leaves to evaluate; ``None`` evaluates the
            env's own trace population and returns it, so the other residency can be fed the same one.
        horizon (int): ``scene.step()`` calls of the generation when ``env`` is ``None``.
        **kwargs: Further ``Go2NeatEnv`` arguments (e.g. ``episode_length_s``) when ``env`` is ``None``.

    Returns:
        RecordedGeneration: The result mapping, the recorded steps and the population used.
    """
    if residency == "numpy":
        steps = _record_numpy_advance(monkeypatch)
    else:
        steps = _record_device_advance(monkeypatch)
    if env is None:
        with _open_env(residency, device, horizon=horizon, **kwargs) as opened:
            return _evaluate_recorded(opened, steps, population)
    return _evaluate_recorded(env, steps, population)


def _evaluate_recorded(env, steps, population):
    """Run one recorded generation on an open ``env`` and snapshot it on the host."""
    population = tuple(np.array(leaf, copy=True) for leaf in env._trace_pop) if population is None else population
    result = env.evaluate(population, 0)
    return RecordedGeneration(result, steps, population, int(env.max_episode_length))


def _reward_terms(steps) -> np.ndarray:
    """The legacy reward terms of every recorded step of one path, from that path's own inputs.

    ``action_change`` is the clipped action of the step minus the action the physics executed, exactly
    the quantity ``action_rate`` consumes (``go2_neat_kernels.run_advance`` clips inside the graph).

    Args:
        steps (list): Recorded steps of one residency.

    Returns:
        np.ndarray: ``(len(steps), rows, len(TERM_NAMES))`` float64.
    """
    terms = np.empty((len(steps), ROWS, len(TERM_NAMES)), dtype=np.float64)
    for index, step in enumerate(steps):
        inputs = step["inputs"]
        action_change = np.clip(inputs["actions"], -kernels.CLIP_ACTIONS, kernels.CLIP_ACTIONS) - inputs["last_actions"]
        for row in range(ROWS):
            values = legacy.walking_reward_terms(
                inputs["base_pos"][row],
                inputs["base_quat"][row],
                inputs["base_lin_vel_world"][row],
                inputs["base_ang_vel_world"][row],
                inputs["dof_pos"][row],
                action_change[row],
            )
            terms[index, row] = [float(value.item()) if hasattr(value, "item") else float(value) for value in
                                 (values[name] for name in TERM_NAMES)]
    return terms


def _trainer_run(env, generations) -> TrainerRun:
    """Run ``generations`` on ``env`` and return the host snapshot of the finished run."""
    trainer = NeuroevolutionTrainer(
        env,
        POP_SIZE,
        generations,
        master_seed=MASTER_SEED,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        residency=env.residency,
    )
    try:
        trainer.setup()
        trainer.run()
        return TrainerRun(
            fitness_history=[float(value) for value in trainer.fitness_history],
            records=[
                {
                    "steps": int(record["steps"]),
                    "end_reasons": {name: int(count) for name, count in record["end_reasons"].items()},
                    "seeds": np.asarray(as_host(record["seeds"])),
                    "fitness": np.asarray(as_host(record["fitness"]), dtype=np.float64),
                }
                for record in trainer.records
            ],
            seed_table=np.array(trainer.seed_table, copy=True),
            pop_arrays=tuple(np.array(as_host(leaf), copy=True) for leaf in trainer.pop_arrays),
            species_leaves=tuple(np.array(as_host(leaf), copy=True) for leaf in trainer.species_leaves),
            randkey=int(np.asarray(as_host(trainer.randkey))),
            generation=float(np.asarray(as_host(trainer.generation_counter))),
        )
    finally:
        trainer.close()


def _record_and_train(monkeypatch, residency, device, *, population=None, horizon=HORIZON, **kwargs):
    """One residency's whole default-config work on a single env, closed before the other residency.

    The recorded generation and both trainer runs share one ``Go2NeatEnv`` (and therefore one 24-row
    scene): building the scene once per residency is what keeps the number of Genesis scenes a serial
    run creates at four instead of six, and each residency's work finishes before the other residency's
    scene is built.

    Args:
        monkeypatch (pytest.MonkeyPatch): Restores the recording hooks after the recorded generation.
        residency (str): ``"numpy"`` or ``"device"``.
        device (evoxc.Device | None): The device of the residency (``None`` for ``"numpy"``).
        population (tuple | None): The payload leaves the recorded generation evaluates; ``None`` uses
            the env's own trace population, so the numpy residency hands its population to the device.
        horizon (int): ``scene.step()`` calls of the recorded generation.
        **kwargs: Further ``Go2NeatEnv`` arguments (e.g. ``episode_length_s``).

    Returns:
        tuple: ``(RecordedGeneration, (two_generation_run, one_generation_run))``.
    """
    with _open_env(residency, device, horizon=horizon, **kwargs) as env:
        recorded = _recorded_generation(monkeypatch, residency, device, env=env, population=population)
        monkeypatch.undo()
        runs = (_trainer_run(env, 2), _trainer_run(env, 1))
    return recorded, runs


def _assert_leaf_parity(numpy_leaves, device_leaves, *, label, atol=LEAF_ATOL) -> float:
    """Integer leaves bitwise, float leaves within ``atol``; returns the largest float difference."""
    numpy_leaves = tuple(numpy_leaves)
    device_leaves = tuple(device_leaves)
    assert len(numpy_leaves) == len(device_leaves), (
        f"{label}: {len(numpy_leaves)} numpy leaves vs {len(device_leaves)} device leaves; both "
        "residencies must agree on the leaf layout of the same state"
    )
    largest = 0.0
    for index, (reference, actual) in enumerate(zip(numpy_leaves, device_leaves, strict=True)):
        reference = np.asarray(reference)
        actual = np.asarray(actual)
        assert reference.shape == actual.shape, (
            f"{label}[{index}]: numpy shape {reference.shape} vs device shape {actual.shape}"
        )
        if reference.dtype.kind in "iu":
            differing = int(np.count_nonzero(reference != actual))
            assert differing == 0, (
                f"{label}[{index}] (integer {reference.dtype}): {differing} of {reference.size} entries "
                "differ; the integer leaves are documented as bit-exact across residencies"
            )
            continue
        largest = max(
            largest, assert_allclose(actual, reference, atol=atol, rtol=0, label=f"{label}[{index}]")[0]
        )
    return largest


@pytest.fixture(scope="module")
def default_campaign(safe_hsa_runtime, gpu_genesis, device_placement):
    """Each residency's recorded generation and two trainer runs, one scene at a time.

    ``safe_hsa_runtime`` runs before ``gpu_genesis`` so an unsafe HSA runtime skips the tier before
    Genesis is initialized; ``gpu_genesis`` is requested next so Genesis is initialized before the
    first scene is built; each residency's scene is built, used for all of its work and closed before
    the other residency's scene is built, so a serial run creates two scenes here (and two in
    ``timeout_campaign``).
    """
    backend, device = device_placement
    campaign = {}
    for residency, placement in (("numpy", None), ("device", device)):
        population = campaign["numpy"][0].population if residency == "device" else None
        with pytest.MonkeyPatch.context() as monkeypatch:
            campaign[residency] = _record_and_train(monkeypatch, residency, placement, population=population)
    return campaign


@pytest.fixture(scope="module")
def timeout_campaign(safe_hsa_runtime, gpu_genesis, device_placement):
    """Each residency's recorded timeout generation, one scene at a time.

    ``episode_length_s=0.1`` fixes ``max_episode_length`` at 5 inside the advance graph, so the timeout
    branch needs its own scenes; they are built, used and closed one at a time like the default ones.
    The device residency is fed the numpy residency's generation-0 population.
    """
    backend, device = device_placement
    campaign = {}
    for residency, placement in (("numpy", None), ("device", device)):
        population = campaign["numpy"].population if residency == "device" else None
        with pytest.MonkeyPatch.context() as monkeypatch:
            with _open_env(
                residency, placement, horizon=TIMEOUT_HORIZON, episode_length_s=TIMEOUT_EPISODE_LENGTH_S
            ) as env:
                campaign[residency] = _recorded_generation(
                    monkeypatch, residency, placement, env=env, population=population
                )
    return campaign


def test_per_step_observation_and_reward_parity(default_campaign):
    """One generation on one population: observation, reward, bookkeeping and reward terms.

    The same generation-0 population (the numpy env's trace population, which both envs trace from the
    same ``Pipeline(seed=0)``) is evaluated by both residencies, and every advance step is compared.
    Measured: 13 steps, ``tilted=24`` in both, the first step's observation 1.513e-09 apart, the
    observation up to 1.984e-04 (action slot) / 3.7e-06 (action-free block), the total reward up to
    1.144e-05, ``returns`` up to 7.63e-06, the ``action_rate`` term up to 9.86e-02 on a term of 4.8e5,
    and the masks/counter exactly equal.
    """
    numpy = default_campaign["numpy"][0]
    recorded = default_campaign["device"][0]

    assert len(numpy.steps) == len(recorded.steps) == numpy.result["steps"] == recorded.result["steps"], (
        f"recorded {len(numpy.steps)} numpy steps and {len(recorded.steps)} device steps, reported "
        f"{numpy.result['steps']} and {recorded.result['steps']} steps; both residencies must run the "
        "same generation step for step"
    )
    assert numpy.result["end_reasons"] == recorded.result["end_reasons"], (
        f"end_reasons differ: numpy {numpy.result['end_reasons']} vs device {recorded.result['end_reasons']}"
    )
    # Non-vacuous: rows end inside the horizon, and they end because they tilted -- ``max_episode_length``
    # is 1000 here, so a timeout cannot fire inside 25 steps.
    assert numpy.result["end_reasons"]["tilted"] == ROWS, (
        f"expected all {ROWS} rows to tilt inside the {HORIZON}-step horizon, got "
        f"{numpy.result['end_reasons']}"
    )
    assert any(step["advance"]["done"].any() for step in numpy.steps), (
        f"no row ended inside the {HORIZON}-step horizon, so the termination-mask comparison is vacuous"
    )
    assert not numpy.steps[-1]["advance"]["alive"].any(), (
        "the generation must stop when every row has ended (the numpy path's ``alive.any()`` loop "
        "condition); the last recorded step still reports a live row"
    )

    # The reset observation is computed from the same staged values on both paths, so it only differs
    # by the host-versus-graph float32 rounding of the inverse quaternion: measured 1.513e-09.
    assert_allclose(
        recorded.steps[0]["advance"]["obs"],
        numpy.steps[0]["advance"]["obs"],
        atol=RESET_OBS_ATOL,
        rtol=0,
        label="reset observation (device vs numpy)",
    )
    for index, (numpy_step, device_step) in enumerate(zip(numpy.steps, recorded.steps, strict=True)):
        numpy_advance = numpy_step["advance"]
        device_advance = device_step["advance"]
        assert_allclose(
            device_advance["obs"],
            numpy_advance["obs"],
            atol=OBS_ATOL,
            rtol=0,
            label=f"step {index} observation (device vs numpy)",
        )
        assert_allclose(
            device_advance["obs"][:, :OBS_ACTION_COLUMN],
            numpy_advance["obs"][:, :OBS_ACTION_COLUMN],
            atol=OBS_STATE_ATOL,
            rtol=0,
            label=f"step {index} action-free observation block (device vs numpy)",
        )
        assert_allclose(
            device_advance["reward"],
            numpy_advance["reward"],
            atol=REWARD_ATOL,
            rtol=0,
            label=f"step {index} total reward (device vs numpy)",
        )
        assert_allclose(
            device_advance["returns"],
            numpy_advance["returns"],
            atol=RETURNS_ATOL,
            rtol=0,
            label=f"step {index} returns (device vs numpy)",
        )
        for name in ("alive", "done", "episode_length"):
            assert np.array_equal(device_advance[name], numpy_advance[name]), (
                f"step {index} {name!r} must be identical across residencies; device "
                f"{device_advance[name]!r} vs numpy {numpy_advance[name]!r}"
            )

    numpy_terms = _reward_terms(numpy.steps)
    device_terms = _reward_terms(recorded.steps)
    for name in TERM_STATE_NAMES:
        column = TERM_NAMES.index(name)
        assert_allclose(
            device_terms[:, :, column],
            numpy_terms[:, :, column],
            atol=TERM_ATOL,
            rtol=0,
            label=f"reward term {name!r} over the horizon (device vs numpy)",
        )
    assert_allclose(
        device_terms[:, :, TERM_NAMES.index("action_rate")],
        numpy_terms[:, :, TERM_NAMES.index("action_rate")],
        atol=ACTION_RATE_ATOL,
        rtol=0,
        label="reward term 'action_rate' over the horizon (device vs numpy)",
    )
    assert_allclose(
        device_terms[:, :, TERM_NAMES.index("total")],
        numpy_terms[:, :, TERM_NAMES.index("total")],
        atol=REWARD_ATOL,
        rtol=0,
        label="reward term 'total' over the horizon (device vs numpy)",
    )
    for label, terms, steps in (
        ("numpy", numpy_terms, numpy.steps),
        ("device", device_terms, recorded.steps),
    ):
        rewards = np.array([step["advance"]["reward"] for step in steps])
        assert_allclose(
            terms[:, :, TERM_NAMES.index("total")],
            rewards,
            atol=TERMS_SUM_ATOL,
            rtol=0,
            label=f"{label} reward terms summed vs the graph reward",
        )


def test_termination_mask_and_timeout_parity(timeout_campaign):
    """The timeout branch and the never-re-arming ``alive`` mask, on one population.

    ``episode_length_s=0.1`` makes ``max_episode_length`` 5, so a 40-step horizon must time rows out,
    reset them and let them keep stepping.  Measured on both residencies: 6 steps and
    ``{"timeout": 17, "tilted": 7, "error": 0, "unfinished": 0}``, ``done`` counts per step
    ``[0, 0, 0, 0, 4, 20]``, the ``episode_length`` of every done row reset to 0, and ``alive``
    monotone non-increasing per row.
    """
    numpy = timeout_campaign["numpy"]
    recorded = timeout_campaign["device"]

    assert numpy.max_episode_length == recorded.max_episode_length == TIMEOUT_MAX_EPISODE_LENGTH, (
        f"expected max_episode_length {TIMEOUT_MAX_EPISODE_LENGTH} on both residencies, got "
        f"{numpy.max_episode_length} and {recorded.max_episode_length}"
    )
    assert numpy.result["steps"] == recorded.result["steps"] == 6, (
        f"expected 6 steps on both residencies (five to reach the timeout, one to end the last "
        f"tilted rows), got {numpy.result['steps']} and {recorded.result['steps']}"
    )
    expected_end_reasons = {"timeout": 17, "tilted": 7, "error": 0, "unfinished": 0}
    assert numpy.result["end_reasons"] == recorded.result["end_reasons"] == expected_end_reasons, (
        f"end_reasons differ from the measured {expected_end_reasons}: numpy "
        f"{numpy.result['end_reasons']} vs device {recorded.result['end_reasons']}"
    )

    numpy_done = np.array([step["advance"]["done"] for step in numpy.steps])
    device_done = np.array([step["advance"]["done"] for step in recorded.steps])
    assert np.array_equal(numpy_done, device_done), (
        "the per-step done mask must be identical across residencies; first differing step "
        f"{int(np.argmax((numpy_done != device_done).any(axis=1)))}"
    )
    assert numpy_done.sum(axis=1).tolist() == [0, 0, 0, 0, 4, 20], (
        f"rows ending per step changed: {numpy_done.sum(axis=1).tolist()} (measured [0, 0, 0, 0, 4, 20])"
    )
    for label, steps, done in (("numpy", numpy.steps, numpy_done), ("device", recorded.steps, device_done)):
        episode_length = np.array([step["advance"]["episode_length"] for step in steps])
        alive = np.array([step["advance"]["alive"] for step in steps])
        assert np.array_equal(episode_length[done], np.zeros(int(done.sum()), dtype=np.int32)), (
            f"{label}: every done row must have its episode_length reset to 0, got "
            f"{episode_length[done].tolist()}"
        )
        assert np.all(np.diff(alive.astype(np.int8), axis=0) <= 0), (
            f"{label}: 'alive' must never re-arm (a reset row stays dead for the rest of the "
            f"generation); per-step alive counts {alive.sum(axis=1).tolist()}"
        )
        assert episode_length.max() == TIMEOUT_MAX_EPISODE_LENGTH, (
            f"{label}: the counter must reach max_episode_length {TIMEOUT_MAX_EPISODE_LENGTH} before it "
            f"resets, got {int(episode_length.max())}"
        )


def test_two_generation_end_to_end_reference_numbers(default_campaign):
    """Two generations at each residency's own population, against the documented numbers.

    Each residency's generation-0 population is the trainer's own (device-first initialization under
    the device residency), so the two runs differ by more than the rollout boundary alone -- see
    ``CROSS_RESIDENCY_FITNESS_ATOL``.  Measured: seed table rows ``[1454269175, 757617120, 1421746521]``
    and ``[1313752747, 908599966, 53557616]``, ``steps`` ``[21, 10]``, ``tilted=24`` per generation,
    ``fitness_history`` -7.378471374511719 / -7.318042278289795 (numpy) and -7.378471374511719 /
    -7.318042755126953 (device), 4.77e-07 apart, and a generation-0 per-policy fitness drift of up to
    7.02e-04.
    """
    numpy_run, _ = default_campaign["numpy"][1]
    device_run, _ = default_campaign["device"][1]

    assert evolution_seed(MASTER_SEED) == DOCUMENTED_EVOLUTION_SEED, (
        f"seeding.evolution_seed({MASTER_SEED}) is {evolution_seed(MASTER_SEED)}, documented as "
        f"{DOCUMENTED_EVOLUTION_SEED}; every run below is seeded from it"
    )
    for label, run in (("numpy", numpy_run), ("device", device_run)):
        assert run.seed_table.tolist() == DOCUMENTED_SEED_TABLE[:2], (
            f"{label}: the seed table of the two generations is {run.seed_table.tolist()}, documented as "
            f"{DOCUMENTED_SEED_TABLE[:2]}"
        )
        assert [record["seeds"].tolist() for record in run.records] == DOCUMENTED_SEED_TABLE[:2], (
            f"{label}: the env reported episode seeds "
            f"{[record['seeds'].tolist() for record in run.records]}, which must be the trainer's own "
            f"table {DOCUMENTED_SEED_TABLE[:2]}"
        )
        assert [record["steps"] for record in run.records] == DOCUMENTED_STEPS_HORIZON_25, (
            f"{label}: steps per generation {[record['steps'] for record in run.records]}, documented as "
            f"{DOCUMENTED_STEPS_HORIZON_25}"
        )
        for generation, record in enumerate(run.records):
            assert record["end_reasons"] == {"timeout": 0, "tilted": ROWS, "error": 0, "unfinished": 0}, (
                f"{label} generation {generation}: end_reasons {record['end_reasons']}, expected every "
                f"one of the {ROWS} rows to tilt (max_episode_length is {FULL_HORIZON})"
            )
        assert_allclose(
            run.fitness_history,
            DOCUMENTED_HISTORY_HORIZON_25,
            atol=HISTORY_ATOL,
            rtol=0,
            label=f"{label} fitness_history vs the documented two-generation history",
        )

    assert_allclose(
        device_run.fitness_history,
        numpy_run.fitness_history,
        atol=HISTORY_ATOL,
        rtol=0,
        label="fitness_history (device vs numpy residency)",
    )
    generation_0 = np.abs(device_run.records[0]["fitness"] - numpy_run.records[0]["fitness"])
    assert float(generation_0.max()) <= CROSS_RESIDENCY_FITNESS_ATOL, (
        f"generation-0 per-policy fitness drift {generation_0.tolist()} (max {float(generation_0.max()):.3e}) "
        f"exceeds {CROSS_RESIDENCY_FITNESS_ATOL:.1e}; device-first initialization moves the float genome "
        "leaves by <= 2.4e-07 and an untrained policy at a contact threshold amplifies that, so the "
        "measured 7.02e-04 is conditional-path chaos, not a residency bias"
    )


def test_first_population_update_parity(default_campaign):
    """The post-``tell`` state of one generation: integer leaves/key/counter exact, floats measured.

    Measured: ``randkey`` -5246925798323494445 and ``generation`` 1.0 on both residencies, both integer
    payload leaves and both integer species leaves bit-identical, and every float leaf within 2.384e-07.
    """
    _, numpy_first = default_campaign["numpy"][1]
    _, device_first = default_campaign["device"][1]

    assert numpy_first.generation == device_first.generation == 1.0, (
        f"the generation counter must advance to exactly 1.0 on both residencies, got "
        f"{numpy_first.generation} (numpy) and {device_first.generation} (device)"
    )
    assert numpy_first.randkey == device_first.randkey, (
        f"the RNG key after one tell differs: numpy {numpy_first.randkey} vs device "
        f"{device_first.randkey}; it is documented as bit-exact across residencies"
    )
    payload_diff = _assert_leaf_parity(
        numpy_first.pop_arrays, device_first.pop_arrays, label="next generation payload"
    )
    species_diff = _assert_leaf_parity(
        numpy_first.species_leaves, device_first.species_leaves, label="next generation species"
    )
    assert max(payload_diff, species_diff) <= LEAF_ATOL, (
        f"float leaf drift after one tell: payload {payload_diff:.3e}, species {species_diff:.3e}, "
        f"bound {LEAF_ATOL:.1e}"
    )


CHILD_SCRIPT = """
import json
import sys

import numpy as np

import evoxc
import genesis as gs

sys.path.insert(0, {locomotion!r})
sys.path.insert(0, {neuroevolution!r})

evoxc.backends.get({backend!r})
device = evoxc.Device({kind!r}, {index})
evoxc.from_numpy(np.zeros(1, dtype=np.float32)).to(device)

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)

from go2_neat_env import Go2NeatEnv
from seeding import Perturbation
from trainer import NeuroevolutionTrainer

env = Go2NeatEnv(
    pop_size={pop_size},
    master_seed={master_seed},
    perturbation=Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.05),
    horizon={horizon},
    stop_when_done=False,
    backend={backend!r},
    device=device,
    residency="device",
)
trainer = NeuroevolutionTrainer(
    env,
    {pop_size},
    {generations},
    master_seed={master_seed},
    num_inputs={num_inputs},
    num_outputs={num_outputs},
    residency="device",
)
trainer.setup()
trainer.run()

payload = {{
    "fitness_history": [float(value) for value in trainer.fitness_history],
    "steps": [int(record["steps"]) for record in trainer.records],
    "end_reasons": [dict(record["end_reasons"]) for record in trainer.records],
    "seeds": [np.asarray(record["seeds"]).tolist() for record in trainer.records],
    "devices": [f"{{leaf.device.kind}}:{{leaf.device.index}}" for leaf in trainer.pop_arrays],
    "torch_modules": sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.")),
}}
env.close()
gs.destroy()
print("GO2_FULL_HORIZON " + json.dumps(payload))
"""

MARKER = "GO2_FULL_HORIZON"


@pytest.mark.slow
def test_full_horizon_reference_numbers_without_torch(gpu_genesis, device_placement):
    """Three full-horizon (1000-step) generations with the Torch-free, device-resident trainer.

    The 3000-step rollout cannot run in this process (see the module docstring's HSA section), so it runs
    in a child interpreter through ``tests.torch_block.run_python_checked``: PyTorch is blocked before
    the first import and the harness fails the child if ``torch`` reaches ``sys.modules`` anyway.
    Measured there: ``fitness_history`` -7.378471374511719 / -7.318042755126953 / -6.039944171905518,
    1000 steps per generation, ``tilted=24`` and every other reason 0, and the documented seed triples.

    Marked ``slow``: a cold scene build, three generations of 1000 steps and the HSA re-exec of the
    child add up to ~2 minutes on an idle MI300X, well beyond a fast test's budget.
    """
    backend, device = device_placement
    script = CHILD_SCRIPT.format(
        locomotion=str(LOCOMOTION_DIR),
        neuroevolution=str(NEUROEVOLUTION_DIR),
        backend=DEVICE_BACKEND,
        kind=device.kind,
        index=int(device.index),
        pop_size=POP_SIZE,
        master_seed=MASTER_SEED,
        horizon=FULL_HORIZON,
        generations=FULL_HORIZON_GENERATIONS,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
    )
    completed = run_python_checked(script, timeout=FULL_HORIZON_TIMEOUT_S)
    payload = _marker_payload(completed.stdout)

    assert payload["torch_modules"] == [], (
        f"the torch-free child imported PyTorch: {payload['torch_modules']}; the device residency must "
        "not need it at all"
    )
    assert set(payload["devices"]) == {f"{device.kind}:{device.index}"}, (
        f"the population leaves must stay on {device.kind}:{device.index}, got {payload['devices']}"
    )
    assert payload["steps"] == [FULL_HORIZON] * FULL_HORIZON_GENERATIONS, (
        f"steps per generation {payload['steps']}, expected one {FULL_HORIZON}-step generation each "
        "(``stop_when_done=False`` keeps the generation running to its horizon)"
    )
    for generation, end_reasons in enumerate(payload["end_reasons"]):
        assert end_reasons == {"timeout": 0, "tilted": ROWS, "error": 0, "unfinished": 0}, (
            f"generation {generation}: end_reasons {end_reasons}, expected every one of the {ROWS} rows "
            f"to tilt inside one full episode (``max_episode_length`` is exactly {FULL_HORIZON})"
        )
    assert payload["seeds"] == DOCUMENTED_SEED_TABLE, (
        f"episode seeds {payload['seeds']}, documented as {DOCUMENTED_SEED_TABLE}"
    )
    assert_allclose(
        payload["fitness_history"],
        DOCUMENTED_HISTORY_FULL_HORIZON,
        atol=HISTORY_ATOL,
        rtol=0,
        label="torch-free full-horizon fitness_history vs the documented three-generation history",
    )


def _marker_payload(stdout, marker=MARKER):
    """Parse the JSON payload of the child's single ``<marker> {...}`` line."""
    for line in stdout.splitlines():
        if line.startswith(marker + " "):
            return json.loads(line[len(marker) + 1 :])
    raise AssertionError(f"the child printed no {marker!r} line; its output was:\n{stdout}")

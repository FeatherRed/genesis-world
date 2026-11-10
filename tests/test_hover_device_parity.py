"""Pilot-level parity of the Torch-free, GPU-resident drone (hover) TensorNEAT trainer.

``examples/drone`` carries two implementations of one hover generation: the host (``residency="numpy"``)
reference of ``hover_neat_env.HoverNeatEnv``, which stages every graph operand through numpy, and the
device-resident (``residency="device"``) rollout of ``hover_device_rollout.DeviceRollout``, which keeps
every operand on an ``evoxc`` device.  The Torch-free pilot this module scores -- a
``hover_device_kernels.py`` with a torch-free DLPack bridge, a rewritten torch-free
``hover_device_rollout.py`` and a ``hover_neat_train.py --torch-free`` entry point -- is the drone
counterpart of the Go2 pilot pinned by ``tests/test_go2_device_parity.py``.  This is its test module,
written the same way and sharing the same helpers of ``tests/device_pilot.py``.

Availability (the gate this module ships with)
----------------------------------------------
The drone pilot does not exist in this tree yet, so a module-scoped autouse fixture
(:func:`torch_free_pilot`) skips every test with :func:`_unavailable_reason`: the reason names *every*
pilot module that is missing, imports PyTorch, or (for the CLI) accepts no ``--torch-free`` flag.  The
check is static -- ``tests.device_pilot.torch_import_offenders`` parses the four sources, nothing is
imported -- so the skip stays green while the parallel change lands and the same tests start running
real assertions as soon as all four modules are in place.  Nothing here is a fail-until-updated stub.

What is pinned once the pilot lands
-----------------------------------
1. **Per-step parity on one population** (:func:`test_per_step_observation_and_reward_parity`): the same
   generation-0 population is fed to both residencies; the reset observation, every advance step's
   observation, the five ``dt``-scaled reward terms plus the total, ``returns``, the
   ``crash``/``timeout``/``done``/``at_target`` masks, ``commands``, ``yaw_deg``, ``episode_length``,
   ``alive`` and the latched ``end_code`` are compared, and each path's graph output is checked against
   ``hover_legacy_reference.step_maths`` recomputed from *that path's own recorded operands*.
2. **Termination and timeout** (:func:`test_termination_mask_and_timeout_parity`): with
   ``episode_length_s=0.1`` (``max_episode_length=10`` at ``dt=0.01``) and ``stop_when_done=False``, the
   per-step ``done``/``timeout``/``end_code`` masks must be identical across residencies, the timeout must
   fire exactly on the step whose counter exceeds ``max_episode_length``, ``episode_length`` must never be
   reset (one episode per row), ``end_code`` must latch forever, ``alive`` must never re-arm and a dead
   row's ``returns`` must stop growing.
3. **Documented end-to-end numbers** (:func:`test_two_generation_end_to_end_reference_numbers`): each
   residency's own generation-0 population (device-first initialization under the device residency), two
   generations at ``pop_size=8``, compared against the seed table, the per-generation ``steps``
   ``[160, 58]`` and the cross-residency exactness ``examples/drone/CONTEXT.md`` documents.
4. **Placement and Torch-free training** (``test_device_resident_training_without_torch``, ``slow``): a
   ``tests.torch_block.run_python_checked`` child that initializes Genesis on its GPU, builds
   ``HoverNeatEnv`` device-resident, runs a two-generation trainer budget, asserts every training leaf's
   placement with ``device_runtime.assert_device_resident``/``placement_map`` and fails if ``torch``
   reaches ``sys.modules``.
5. **The real CLI** (``test_torch_free_cli_runs_with_torch_import_blocked``, ``slow``): the same run
   through ``hover_neat_train.py --torch-free`` as a script, with a ``torch`` package that refuses to
   import first on ``PYTHONPATH``, and its ``history.jsonl``/``summary.json`` artifacts checked against the
   documented numbers.

Active-verified vs written-against-the-documentation (current state)
--------------------------------------------------------------------
Nothing below item 1 of the availability gate has ever executed in this tree: the pilot modules are
absent or still import torch, so the gate skips, and the numeric expectations of items 1-5 are taken from
``examples/drone/CONTEXT.md`` rather than measured here.  Concretely, the *gate* (including the offender
report it prints) is active-verified today, and every assertion of items 1-5 is written against the
documented API and numbers: the drone's reproducible measurements are the fixed-reset fingerprint
(``seeds=1454269175,757617120,1421746521``, ``sum=-0.024585718`` at ``pop 2``/``horizon 30``), the
residency parity (``returns_max_abs_diff 0.000e+00``, ``fitness_max_abs_diff 0.000e+00``,
``end_reasons_equal 1``, ``seeds_equal 1``) and the full-length two-generation run whose ``steps`` are
``160`` then ``58`` and whose ``returns``/``fitness`` agree to exactly ``0.0``.  The drone documents *no*
absolute fitness value for that run, so item 3 pins the ``steps``/seed/reason metadata and the
cross-residency agreement only, and :data:`DOCUMENTED_HISTORY` stays ``None`` until the pilot's own
reference numbers are documented.  A follow-up that measured the activated path must replace the
``written against`` tolerances with measured ones, exactly as the Go2 module did.

Tolerances
----------
The Go2 module derives its bounds from measured values; here the bounds are the parity tolerances the
drone already documents for its graphs (``tests/test_hover_evox_parity.py``), widened where a quantity
accumulates or is quadratic in a diverging value:

* :data:`RESET_OBS_ATOL` / :data:`OBS_ATOL` = ``1e-6``: the analytic reset observation and one step's
  ``(rows, 17)`` observation.  ``examples/drone/CONTEXT.md`` measures the reset observation and the
  observation at exactly ``0`` on one backend; across the numpy and ``iree`` backends the last float32 bit
  may move, so the bound is the graph-level parity tolerance rather than ``0``.
* :data:`REWARD_TERM_ATOL` = ``1e-5`` / :data:`REWARD_TOTAL_ATOL` = ``1e-6``: one ``dt``-scaled reward term
  and the total, the documented graph tolerances.
* :data:`RETURNS_ATOL` = :data:`HISTORY_ATOL` = :data:`CROSS_RESIDENCY_ATOL` = ``1e-5``: an accumulated
  episode return and a ``fitness_history`` entry.  The drone's full-length run documents exactly ``0.0``
  cross-residency; the bound leaves room for a reordering of the last float32 bit while still failing any
  real divergence (the convention ``tests/test_neuroevolution_device_parity.py`` uses).
* :data:`YAW_ATOL_DEG` = ``5e-4``: the documented yaw tolerance (a float32 minimax ``atan2`` plus one ulp).
* :data:`TERMS_SUM_ATOL` = ``1e-5``: a path's five terms summed against that path's own graph ``total``.
* masks, ``episode_length``, ``end_code`` and every integer leaf: exact.

Skip reasons
------------
``pytest.importorskip`` skips the module where ``evoxc``/``tensorneat`` are missing (both are optional and
undeclared in ``pyproject.toml``); ``torch_free_pilot`` skips it while the Torch-free drone pilot is not in
place; ``gpu_only`` skips it on a CPU-only host (``tests.torch_block.has_gpu``, which falls back to the
``/dev/kfd`` device node where NVML/AMD SMI are not installed); ``device_placement`` skips it when no
non-CPU ``evoxc`` device accepts a write (``tests.device_pilot.reachable_device_or_skip``, which can be
pointed at a device through ``EVOXC_DEVICE``).

HSA caveat, and why the training cases run in children
------------------------------------------------------
Under pytest the HSA repair in ``gs.init`` is warn-only, so this process keeps torch's bundled HSA runtime
and a long torch-present Genesis rollout can abort with ``HSA_STATUS_ERROR_INVALID_PACKET_FORMAT``.  Every
in-process rollout here is therefore short (25 and 15 steps), and the two full-length (160 + 58 step)
training cases are delegated to child interpreters where PyTorch is blocked and the system runtime is the
one that gets loaded.  Independence of the rollout length is not assumed: ``gpu_genesis`` skips the
in-process tier where ``tests.device_pilot.unsafe_hsa_runtime_reason`` reports that the worker maps
PyTorch's bundled runtime, naming the ``LD_PRELOAD`` remedy; the child scenarios are not gated.

Two scenes are never alive at the same time
-------------------------------------------
Each ``HoverNeatEnv`` owns its own ``3 * pop_size``-row drone ``gs.Scene``, so every test builds one
residency's env, records what it needs on the host, closes it, and only then builds the other.  Both
residencies still share the *same* Genesis GPU backend: the numpy env is built on ``gs.gpu`` as well,
because the two numbers only compare when the physics runs on the same device.

Why this module owns ``gs.init``/``gs.destroy``
----------------------------------------------
``pytestmark`` parametrizes ``backend=None``, which makes the suite's autouse ``initialize_genesis``
fixture a no-op (and that fixture calls ``gs.destroy()`` after every test, while the envs here must
outlive a single test).  The module-scoped ``gpu_genesis`` fixture initializes Genesis once and destroys
it once, at module teardown.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

pytest.importorskip("evoxc")
pytest.importorskip("tensorneat")

#: ``iree`` truncates its ROCm compile of a reduction graph without this flag; the trainer passes it itself
#: for ``iree`` on cuda/rocm, and the graphs built here need the same setting.  Source:
#: ``examples/neuroevolution/CONTEXT.md`` "Device backends".
os.environ.setdefault("EVOXC_IREE_COMPILE_ARGS", "--iree-codegen-llvmgpu-use-reduction-vector-distribution=false")

pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: ``examples/`` are plain scripts, not a package, so the sibling directories need to be on ``sys.path``
#: before any bare-name import (the bootstrap ``tests/test_hover_evox_parity.py`` uses).
EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
DRONE_DIR = EXAMPLES_DIR / "drone"
NEUROEVOLUTION_DIR = EXAMPLES_DIR / "neuroevolution"
for _directory in (NEUROEVOLUTION_DIR, DRONE_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from seeding import EPISODES_PER_POLICY, Perturbation, evolution_seed  # noqa: E402

from tests.device_pilot import (  # noqa: E402
    as_host,
    assert_allclose,
    reachable_device_or_skip,
    skip_if_unsafe_hsa_runtime,
    torch_import_offenders,
)
from tests.torch_block import REPO_ROOT, has_gpu, run_python_checked  # noqa: E402

#: The four modules the Torch-free drone pilot consists of: the planned torch-free device kernels, the
#: rewritten device rollout, the env and the CLI.  A missing module is an offender too, so the gate skips
#: until all four are in place.
PILOT_MODULES = (
    "hover_device_kernels.py",
    "hover_device_rollout.py",
    "hover_neat_env.py",
    "hover_neat_train.py",
)
#: The training entry point and the flag that makes it a Torch-free pilot (``examples/locomotion``
#: precedent: ``go2_neat_train.py --torch-free``).
TORCH_FREE_CLI = "hover_neat_train.py"
TORCH_FREE_FLAG = "--torch-free"

#: Policies per generation; 8 policies are 24 rows, the size the drone's documented runs use.
POP_SIZE = 8
#: Rows of the scene: ``EPISODES_PER_POLICY`` episodes per policy, policy-major.
ROWS = POP_SIZE * EPISODES_PER_POLICY
#: Seed of both RNG streams; the seed table and the evolution seed are documented for it.
MASTER_SEED = 0
#: ``hover_neat_train.py``'s default reset perturbation (``--perturb-*`` defaults; the joint amplitude is
#: inert on a body-only task and therefore 0).
PERTURBATION = Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.0)
#: ``scene.step()`` calls of the short in-process generations (0.25 s of a 15 s episode).
HORIZON = 25
#: ``hover_neat_env.EPISODE_LENGTH_S`` in seconds and the episode cap it implies (``ceil(15 / 0.01)``).
EPISODE_LENGTH_S = 15.0
FULL_HORIZON = 1500
#: The timeout probe: ``episode_length_s=0.1`` is ``max_episode_length=10`` at ``dt=0.01``; the graph
#: times a row out on the step whose counter *exceeds* the cap, i.e. step 11.
TIMEOUT_EPISODE_LENGTH_S = 0.1
TIMEOUT_MAX_EPISODE_LENGTH = 10
TIMEOUT_HORIZON = 15
#: The genome widths of the hover task (``hover_kernels.NUM_OBS`` / ``hover_kernels.NUM_ACTIONS``).
NUM_INPUTS = 17
NUM_OUTPUTS = 4
#: Generations of the documented two-generation run.
GENERATIONS = 2
#: The ``evoxc`` backend of the device residency (the only one the zero-copy bridge uses).
DEVICE_BACKEND = "iree"
#: The child runs an HSA re-exec, a cold scene build, the ``iree`` compiles and ~220 steps; 900 s is ample.
CHILD_TIMEOUT_S = 900
#: The CLI child pays the same cold compiles and writes the artifacts; same budget.
CLI_TIMEOUT_S = 900

#: ``seeding.evolution_seed(0)``, documented in ``examples/neuroevolution/CONTEXT.md``.
DOCUMENTED_EVOLUTION_SEED = 3757552657
#: ``seeding.generation_seed_table(0, 3)``: the three generation rows, policy-shared.
DOCUMENTED_SEED_TABLE = [
    [1454269175, 757617120, 1421746521],
    [1313752747, 908599966, 53557616],
    [1683657442, 1375977993, 1654802006],
]
#: ``steps`` of the two full-length generations of the untrained population (``examples/drone/CONTEXT.md``,
#: ``--pop-size 8 --generations 2``): the rows end by crash after 160 and 58 control steps.
DOCUMENTED_STEPS = [160, 58]
#: The absolute ``fitness_history`` of that run is NOT documented (the drone CONTEXT quotes only the exact
#: cross-residency agreement), so it is left unmeasured instead of being invented.  Replace with the
#: pilot's documented values when they exist, and assert them in
#: :func:`test_two_generation_end_to_end_reference_numbers`.
DOCUMENTED_HISTORY = None

#: One observation of one reset/step, both residencies: documented as ``0`` on one backend.
RESET_OBS_ATOL = 1e-6
OBS_ATOL = 1e-6
#: One ``dt``-scaled reward term / the total: the documented graph tolerances (1e-5 / 1e-6).
REWARD_TERM_ATOL = 1e-5
REWARD_TOTAL_ATOL = 1e-6
#: A path's five terms summed against that path's own graph ``total``.
TERMS_SUM_ATOL = 1e-5
#: One row's accumulated episode return, and one ``fitness_history`` entry: documented as exactly 0.0
#: cross-residency; the bound absorbs float32 reduction-order drift without hiding a divergence.
RETURNS_ATOL = 1e-5
HISTORY_ATOL = 1e-5
CROSS_RESIDENCY_ATOL = 1e-5
#: The next ``commands`` of one step (a host draw selected by the graph).
COMMANDS_ATOL = 1e-6
#: The base-position snapshot of one step (metres); documented as the live buffers being bit-identical.
POSITION_ATOL = 1e-6
#: The documented yaw tolerance: a float32 minimax ``atan2`` plus one ulp of a float32 degree.
YAW_ATOL_DEG = 5e-4

#: The five reward terms plus the ``dt``-scaled total, in ``hover_kernels`` order.
TERM_NAMES = ("target", "smooth", "yaw", "angular", "crash", "total")
#: The graph leaves ``hover_kernels.hover_advance`` returns, in return order (``hover_kernels.run_advance``
#: unpacks exactly this list, and ``hover_device_rollout`` mirrors it).
ADVANCE_LEAVES = (
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
#: The subset of a step compared across residencies, and the masks/integers that must be exact.
ADVANCE_FIELDS = (
    "obs",
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
EXACT_FIELDS = ("crash", "timeout", "done", "at_target", "alive", "episode_length", "end_code")
MASK_FIELDS = ("crash", "timeout", "done", "at_target", "alive")

SKIP_NO_GPU = (
    "drone device-residency parity needs a GPU-backed Genesis: neither NVML nor AMD SMI reports a backend "
    "and no GPU device node (/dev/kfd, /dev/nvidiactl) exists"
)


# ---------------------------------------------------------------------------------------------
# Availability gate: the Torch-free drone pilot
# ---------------------------------------------------------------------------------------------


def _accepted_flags(path) -> list:
    """The flag strings of every ``add_argument`` call in ``path``, from an AST scan.

    A file is parsed, never imported: importing the CLI needs ``genesis`` and a GPU.  An unreadable or
    unparsable file yields an empty list, and :func:`_unavailable_reason` reports the missing flag; a
    *missing* file is reported by :func:`torch_import_offenders` instead.

    Args:
        path (str | pathlib.Path): The module to scan.

    Returns:
        list[str]: Every positional flag string, e.g. ``["--pop-size", "--torch-free/--no-torch-free"]``.
    """
    import ast

    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    flags = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                flags.append(argument.value)
    return flags


def _accepts_flag(flags, flag) -> bool:
    """Whether an ``argparse`` flag list accepts ``flag``, including the ``--x/--no-x`` form."""
    return any(part == flag for entry in flags for part in entry.split("/"))


def _unavailable_reason():
    """Why the Torch-free drone pilot cannot be tested yet; ``None`` when every module is in place.

    A module is an offender when it is missing, when it imports PyTorch anywhere (the AST scan sees the
    lazy ``import torch`` inside the DLPack helpers too, which is exactly what the rewrite removes), or --
    for the CLI -- when it declares no ``--torch-free`` argument.  Every offender is named, so the skip
    reason doubles as the follow-up checklist.

    Returns:
        str | None: The reason to skip with, or ``None`` when the pilot is available.
    """
    problems = torch_import_offenders(*(DRONE_DIR / name for name in PILOT_MODULES))
    cli = DRONE_DIR / TORCH_FREE_CLI
    if cli.exists() and not _accepts_flag(_accepted_flags(cli), TORCH_FREE_FLAG):
        problems.append(f"{cli.name} accepts no {TORCH_FREE_FLAG} flag")
    if not problems:
        return None
    return "the Torch-free drone pilot is not in place yet: " + "; ".join(problems)


def _require_pilot():
    """Skip the calling test unless the Torch-free drone pilot is complete."""
    reason = _unavailable_reason()
    if reason is not None:
        pytest.skip(reason)


# ---------------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def torch_free_pilot():
    """Gate every test of this module on the Torch-free drone pilot being in place (see the module docstring)."""
    _require_pilot()


@pytest.fixture(scope="module")
def gpu_only(torch_free_pilot):
    """Skip on a CPU-only host before anything initializes Genesis or probes an ``evoxc`` device."""
    if not has_gpu():
        pytest.skip(SKIP_NO_GPU)
    return True


@pytest.fixture(scope="module")
def safe_hsa_runtime():
    """Skip the in-process tier where Genesis' HSA re-exec repair cannot run (see the module docstring)."""
    skip_if_unsafe_hsa_runtime()


@pytest.fixture(scope="module")
def gpu_genesis(gpu_only, safe_hsa_runtime):
    """Initialize Genesis on its GPU backend once, and destroy it once at module teardown.

    ``pytestmark``'s ``parametrize("backend", [None])`` disables the suite's autouse ``initialize_genesis``
    fixture; this module needs its own, module-scoped one because the envs here outlive a single test and
    ``gs.destroy()`` between tests would invalidate their solver state.  Only the in-process tests request
    it, so they alone are gated on the HSA runtime being safe to use.
    """
    import genesis as gs

    if gs._initialized:
        gs.destroy()
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    try:
        yield gs
    finally:
        gs.destroy()


@pytest.fixture(scope="module")
def device_placement(gpu_only):
    """``(backend, device)`` of a reachable non-CPU ``evoxc`` device, or a skip naming every failure."""
    return reachable_device_or_skip(DEVICE_BACKEND)


# ---------------------------------------------------------------------------------------------
# Recorded runs
# ---------------------------------------------------------------------------------------------


class RecordedGeneration(NamedTuple):
    """One recorded generation: its result mapping, one entry per advance step and the reset operand."""

    result: dict
    steps: list
    resets: list
    population: tuple
    max_episode_length: int
    rows: int


class TrainerRun(NamedTuple):
    """Host snapshot of one finished ``NeuroevolutionTrainer`` run (the env is already closed)."""

    fitness_history: list
    records: list
    seed_table: np.ndarray
    pop_arrays: tuple
    species_leaves: tuple
    randkey: int
    generation: float
    placement: dict


@contextlib.contextmanager
def _open_env(residency, device, *, horizon=HORIZON, **kwargs):
    """Build one ``HoverNeatEnv`` on the shared Genesis GPU backend and always close it."""
    from hover_neat_env import HoverNeatEnv

    env = HoverNeatEnv(
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


def _advance_operand_names():
    """The tensor operand names of ``hover_kernels.hover_advance``, in call order.

    The trailing ``max_episode_length`` is the graph's static specialization (it is passed as a trailing
    Python int, not as a tensor), so it is dropped.
    """
    import inspect

    import hover_kernels

    names = tuple(inspect.signature(hover_kernels.hover_advance).parameters)
    if not names or names[-1] != "max_episode_length":
        raise AssertionError(
            f"hover_kernels.hover_advance's signature changed ({names}); the advance-operand mapping of "
            "tests/test_hover_device_parity.py assumes max_episode_length is its trailing specialization"
        )
    return names[:-1]


def _cast_fields(fields):
    """Cast one advance step's leaves to their comparison dtypes (masks bool, counters/keys integer)."""
    for name in MASK_FIELDS:
        fields[name] = fields[name].astype(bool)
    fields["episode_length"] = fields["episode_length"].astype(np.int32)
    fields["end_code"] = fields["end_code"].astype(np.int8)
    return fields


def _numpy_fields(result) -> dict:
    """One ``hover_neat_env.run_advance`` result as comparable host arrays."""
    fields = {name: np.array(result[name], copy=True) for name in ADVANCE_FIELDS}
    fields["reward_terms"] = {name: np.array(value, copy=True) for name, value in result["reward_terms"].items()}
    return _cast_fields(fields)


def _device_fields(leaves) -> dict:
    """One ``hover_advance`` run's leaves, keyed like :func:`_numpy_fields`.

    The device path's leaves are ``evoxc`` device tensors, so every leaf goes through ``as_host`` before it
    is compared with the numpy path's array.
    """
    values = dict(zip(ADVANCE_LEAVES, leaves, strict=True))
    fields = {name: np.array(as_host(values[name]), copy=True) for name in ADVANCE_FIELDS}
    fields["reward_terms"] = {
        name: np.array(as_host(values[f"reward_{name}"]), copy=True) for name in TERM_NAMES
    }
    return _cast_fields(fields)


def _record_numpy_advance(monkeypatch) -> list:
    """Record ``hover_neat_env.run_advance``'s operands and returned dict, step by step."""
    import hover_neat_env

    original = hover_neat_env.run_advance
    steps = []

    def recording(executable, **kwargs):
        result = original(executable, **kwargs)
        steps.append(
            {
                "inputs": {name: np.array(as_host(value), copy=True) for name, value in kwargs.items()},
                "fields": _numpy_fields(result),
            }
        )
        return result

    monkeypatch.setattr(hover_neat_env, "run_advance", recording)
    return steps


def _is_advance(executable, env) -> bool:
    """Whether one ``evoxc.run`` call is the advance graph of ``env``."""
    if getattr(executable, "graph", None) == "advance":
        return True
    return executable is getattr(env, "advance_executable", None)


def _record_device_advance(monkeypatch, env) -> list:
    """Record the device rollout's advance operands and its leaves, step by step.

    ``DeviceRollout`` hands the state graph's outputs, the device-resident bookkeeping and the host command
    draw straight to ``evoxc.run(env.advance_executable, ...)``, so the advance call is wrapped where it is
    made: ``hover_device_rollout.evoxc.run`` for an executable whose attached ``graph`` metadata is
    ``"advance"`` (``evox_runtime._attach`` sets it in ``hover_kernels.build_advance``), and the leaves are
    read with the module's own ``_outputs`` normalizer.
    """
    import hover_device_rollout

    evoxc_module = getattr(hover_device_rollout, "evoxc", None)
    if evoxc_module is None or not callable(getattr(evoxc_module, "run", None)):
        raise AssertionError(
            f"{hover_device_rollout.__name__} no longer exposes 'evoxc.run'; this module hooks that call to "
            "record the device advance step by step -- update the hook if the rollout now imports 'run' directly"
        )
    names = _advance_operand_names()
    original = evoxc_module.run
    steps = []

    def recording(executable, *args, **kwargs):
        result = original(executable, *args, **kwargs)
        if _is_advance(executable, env):
            operands = dict(zip(names, args, strict=False))
            operands.update({name: value for name, value in kwargs.items() if name in names})
            leaves = hover_device_rollout._outputs(result)
            if len(leaves) != len(ADVANCE_LEAVES):
                raise AssertionError(
                    f"hover_advance returned {len(leaves)} leaves, this module maps {len(ADVANCE_LEAVES)}; "
                    "update ADVANCE_LEAVES with the graph's documented return order"
                )
            steps.append({"inputs": _device_inputs(operands), "fields": _device_fields(leaves)})
        return result

    monkeypatch.setattr(evoxc_module, "run", recording)
    return steps


def _device_inputs(operands) -> dict:
    """One device advance call's operands as host arrays, with the masks/counters cast like the numpy path."""
    inputs = {}
    for name, value in operands.items():
        array = np.array(as_host(value), copy=True)
        if name in ("error_mask", "alive"):
            array = array.astype(bool)
        elif name == "episode_length":
            array = array.astype(np.int32)
        elif name == "end_code":
            array = array.astype(np.int8)
        inputs[name] = array
    return inputs


def _record_reset(monkeypatch) -> list:
    """Record ``hover_neat_env.run_reset_obs``'s operands and observation, one entry per generation reset.

    The device rollout delegates its reset to ``HoverNeatEnv.reset_generation``, so this hook records the
    analytic reset observation of both residencies at their single source.
    """
    import hover_neat_env

    original = hover_neat_env.run_reset_obs
    resets = []

    def recording(executable, **kwargs):
        result = original(executable, **kwargs)
        resets.append(
            {
                "inputs": {name: np.array(as_host(value), copy=True) for name, value in kwargs.items()},
                "obs": np.array(as_host(result["obs"]), copy=True),
            }
        )
        return result

    monkeypatch.setattr(hover_neat_env, "run_reset_obs", recording)
    return resets


def _recorded_generation(
    monkeypatch, residency, device, *, population=None, horizon=HORIZON, reset_log=None, **kwargs
):
    """Evaluate generation 0 of one residency with every advance step and its reset recorded.

    Args:
        monkeypatch (pytest.MonkeyPatch): Restores the recording hooks at test teardown.
        residency (str): ``"numpy"`` or ``"device"``.
        device (evoxc.Device | None): The device of the residency (``None`` for ``"numpy"``).
        population (tuple | None): The four genome payload leaves to evaluate; ``None`` evaluates the
            env's own trace population and returns it, so the other residency can be fed the same one.
        horizon (int): ``scene.step()`` calls of the generation.
        reset_log (list | None): A reset log shared by several generations of one test (a fresh hook per call
            would nest and record into the previous call's list); a test with one generation may omit it.
        **kwargs: Further ``HoverNeatEnv`` arguments (e.g. ``episode_length_s``).

    Returns:
        RecordedGeneration: The result mapping, the recorded steps and resets, and the population used.
    """
    if reset_log is None:
        reset_log = _record_reset(monkeypatch)
    reset_start = len(reset_log)
    with _open_env(residency, device, horizon=horizon, **kwargs) as env:
        steps = _record_numpy_advance(monkeypatch) if residency == "numpy" else _record_device_advance(
            monkeypatch, env
        )
        population = (
            tuple(np.array(leaf, copy=True) for leaf in env.initial_pop_arrays())
            if population is None
            else population
        )
        result = env.evaluate(population, 0)
        recorded = RecordedGeneration(
            result=result,
            steps=steps,
            resets=reset_log[reset_start:],
            population=population,
            max_episode_length=int(env.max_episode_length),
            rows=int(env.rows),
        )
    if residency != "numpy" and not recorded.steps:
        raise AssertionError(
            "no hover_advance graph run was recorded on the device residency; this module hooks "
            "hover_device_rollout.evoxc.run for executables tagged graph='advance' -- update the hook if "
            "the rewritten rollout runs the advance graph somewhere else"
        )
    return recorded


def _legacy_step(inputs) -> dict:
    """``hover_legacy_reference.step_maths`` of one step's recorded operands (the oracle of that step)."""
    import inspect

    import hover_legacy_reference as legacy

    names = tuple(inspect.signature(legacy.step_maths).parameters)
    return legacy.step_maths(**{name: inputs[name] for name in names if name in inputs})


def _scaled_legacy_terms(reference) -> dict:
    """The oracle's five PRE-scale terms turned into the graph's ``dt``-scaled terms, plus the total.

    ``hover_legacy_reference.reward_terms`` returns the raw terms and the ``dt``-scaled total; the graphs
    return ``dt * scale * term`` for the five terms and the same total.
    """
    import hover_legacy_reference as legacy

    terms = reference["reward_terms"]
    scaled = {
        name: np.asarray(terms[name], dtype=np.float32) * np.float32(legacy.DT * legacy.REWARD_SCALES[name])
        for name in TERM_NAMES
        if name != "total"
    }
    scaled["total"] = np.asarray(terms["total"], dtype=np.float32)
    return scaled


def _assert_path_matches_legacy(label, inputs, fields) -> None:
    """Check one path's graph step against the legacy oracle recomputed from that same path's operands.

    Rewards and the observation are tolerance-bounded (the graph and the oracle both work in float32), the
    masks, ``commands`` and the yaw angle use the documented graph-level bounds.
    """
    reference = _legacy_step(inputs)
    assert_allclose(fields["obs"], reference["obs"], atol=OBS_ATOL, rtol=0, label=f"{label} observation vs oracle")
    scaled = _scaled_legacy_terms(reference)
    for name in TERM_NAMES:
        atol = REWARD_TOTAL_ATOL if name == "total" else REWARD_TERM_ATOL
        assert_allclose(
            fields["reward_terms"][name], scaled[name], atol=atol, rtol=0, label=f"{label} reward term {name!r}"
        )
    assert_allclose(fields["commands"], reference["commands"], atol=COMMANDS_ATOL, rtol=0, label=f"{label} commands")
    assert_allclose(fields["yaw_deg"], reference["yaw_deg"], atol=YAW_ATOL_DEG, rtol=0, label=f"{label} yaw_deg")
    for name in ("crash", "timeout", "done", "at_target"):
        assert np.array_equal(fields[name], reference[name]), (
            f"{label}: mask {name!r} disagrees with the legacy oracle "
            f"({int(np.count_nonzero(fields[name] != reference[name]))} of {fields[name].size} rows)"
        )


def _trainer_run(env, generations) -> TrainerRun:
    """Run ``generations`` on ``env`` and return the host snapshot of the finished run."""
    from trainer import NeuroevolutionTrainer

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
                    "seeds": [int(seed) for seed in record["seeds"]],
                    "returns": np.array(as_host(record["returns"]), dtype=np.float32),
                    "episode_returns": np.array(as_host(record["episode_returns"]), dtype=np.float32),
                    "fitness": np.array(as_host(record["fitness"]), dtype=np.float64),
                }
                for record in trainer.records
            ],
            seed_table=np.array(trainer.seed_table, copy=True),
            pop_arrays=tuple(np.array(as_host(leaf), copy=True) for leaf in trainer.pop_arrays),
            species_leaves=tuple(np.array(as_host(leaf), copy=True) for leaf in trainer.species_leaves),
            randkey=int(np.asarray(as_host(trainer.randkey))),
            generation=float(np.asarray(as_host(trainer.generation_counter))),
            placement=dict(trainer.placement),
        )
    finally:
        trainer.close()


@pytest.fixture(scope="module")
def residency_runs(gpu_genesis, device_placement):
    """Both residencies' two-generation trainer runs, built and closed one after the other.

    ``gpu_genesis`` is requested first so Genesis is initialized before the first scene is built, and only
    one env is alive at a time (each one owns a 24-row drone scene).
    """
    backend, device = device_placement
    runs = {}
    for residency, target in (("numpy", None), ("device", device)):
        with _open_env(residency, target) as env:
            runs[residency] = _trainer_run(env, GENERATIONS)
    return runs


# ---------------------------------------------------------------------------------------------
# Per-step parity
# ---------------------------------------------------------------------------------------------


def test_per_step_observation_and_reward_parity(gpu_genesis, device_placement, monkeypatch):
    """One generation on one population: observation, reward terms, bookkeeping and masks.

    The same generation-0 population (the numpy env's trace population, which both envs trace from the same
    ``Pipeline(seed=0)``) is evaluated by both residencies and every advance step is compared.  Documented
    cross-residency agreement of the drone's ``--residency-parity`` check: ``returns``/``fitness`` exactly
    ``0.0``, the masks and the seed triple equal -- so the bounds here are tolerance-bounded floats and
    exact masks.
    """
    backend, device = device_placement
    reset_log = _record_reset(monkeypatch)
    numpy = _recorded_generation(monkeypatch, "numpy", None, reset_log=reset_log)
    recorded = _recorded_generation(monkeypatch, "device", device, population=numpy.population, reset_log=reset_log)

    assert numpy.rows == recorded.rows == ROWS, (
        f"the scene must have {ROWS} rows (3 episodes x {POP_SIZE} policies), got {numpy.rows} and {recorded.rows}"
    )
    assert len(numpy.steps) == len(recorded.steps) == numpy.result["steps"] == recorded.result["steps"], (
        f"recorded {len(numpy.steps)} numpy steps and {len(recorded.steps)} device steps, reported "
        f"{numpy.result['steps']} and {recorded.result['steps']} steps; both residencies must run the same "
        "generation step for step"
    )
    assert numpy.result["end_reasons"] == recorded.result["end_reasons"], (
        f"end_reasons differ: numpy {numpy.result['end_reasons']} vs device {recorded.result['end_reasons']}"
    )
    # Non-vacuous: the short horizon cannot end an episode, but physics ran and the reward is real.
    assert len(numpy.steps) == HORIZON, (
        f"{len(numpy.steps)} steps of a {HORIZON}-step horizon; nothing may stop a 0.25 s generation "
        "(max_episode_length is 1500 and the drone cannot crash that fast)"
    )
    assert np.any(numpy.steps[-1]["fields"]["returns"] != 0.0), (
        "every row's returns are exactly 0 after 25 steps; the rollouts produced no reward"
    )
    assert not np.array_equal(numpy.steps[0]["fields"]["obs"], numpy.steps[-1]["fields"]["obs"]), (
        "the observation did not change over the horizon; the scene did not step"
    )

    # The reset observation has one owner (``HoverNeatEnv.reset_generation``) and both residencies use it.
    assert len(numpy.resets) == len(recorded.resets) == 1, (
        f"expected one reset per generation, recorded {len(numpy.resets)} numpy and {len(recorded.resets)} device"
    )
    for name in ("commands", "base_init_pos", "base_init_quat"):
        assert np.array_equal(numpy.resets[0]["inputs"][name], recorded.resets[0]["inputs"][name]), (
            f"reset operand {name!r} differs across residencies although both envs are seeded identically"
        )
    assert_allclose(
        recorded.resets[0]["obs"],
        numpy.resets[0]["obs"],
        atol=RESET_OBS_ATOL,
        rtol=0,
        label="reset observation (device vs numpy)",
    )

    # Each path's graph output is the legacy formula of that path's own operands.
    for label, steps in (("numpy", numpy.steps), ("device", recorded.steps)):
        for index, step in enumerate(steps):
            _assert_path_matches_legacy(f"{label} step {index}", step["inputs"], step["fields"])

    for index, (numpy_step, device_step) in enumerate(zip(numpy.steps, recorded.steps, strict=True)):
        numpy_fields = numpy_step["fields"]
        device_fields = device_step["fields"]
        assert_allclose(
            device_fields["obs"], numpy_fields["obs"], atol=OBS_ATOL, rtol=0, label=f"step {index} observation"
        )
        for name in TERM_NAMES:
            atol = REWARD_TOTAL_ATOL if name == "total" else REWARD_TERM_ATOL
            assert_allclose(
                device_fields["reward_terms"][name],
                numpy_fields["reward_terms"][name],
                atol=atol,
                rtol=0,
                label=f"step {index} reward term {name!r}",
            )
        assert_allclose(
            device_fields["returns"], numpy_fields["returns"], atol=RETURNS_ATOL, rtol=0, label=f"step {index} returns"
        )
        assert_allclose(
            device_fields["commands"],
            numpy_fields["commands"],
            atol=COMMANDS_ATOL,
            rtol=0,
            label=f"step {index} next commands",
        )
        assert_allclose(
            device_fields["yaw_deg"],
            numpy_fields["yaw_deg"],
            atol=YAW_ATOL_DEG,
            rtol=0,
            label=f"step {index} yaw_deg",
        )
        assert_allclose(
            device_fields["base_pos"],
            numpy_fields["base_pos"],
            atol=POSITION_ATOL,
            rtol=0,
            label=f"step {index} base_pos snapshot",
        )
        for name in EXACT_FIELDS:
            assert np.array_equal(device_fields[name], numpy_fields[name]), (
                f"step {index} {name!r} must be identical across residencies; device {device_fields[name]!r} "
                f"vs numpy {numpy_fields[name]!r}"
            )

    # Each path's five terms sum to that path's own graph total (the reward is the legacy formula).
    for label, steps in (("numpy", numpy.steps), ("device", recorded.steps)):
        terms = {name: np.array([step["fields"]["reward_terms"][name] for step in steps]) for name in TERM_NAMES}
        summed = np.sum(np.stack([terms[name] for name in TERM_NAMES[:-1]], axis=-1), axis=-1)
        assert_allclose(
            summed, terms["total"], atol=TERMS_SUM_ATOL, rtol=0, label=f"{label} terms summed vs the graph total"
        )


def test_termination_mask_and_timeout_parity(gpu_genesis, device_placement, monkeypatch):
    """The timeout branch, the latched ``end_code`` and the never-re-arming ``alive``, on one population.

    ``episode_length_s=0.1`` makes ``max_episode_length`` 10, and ``stop_when_done=False`` forces the full
    15-step horizon so the rows keep stepping after they end.  Documented graph semantics: the counter is
    incremented (never reset -- one episode per row), the timeout fires on the step whose counter exceeds the
    cap, ``end_code`` latches the first ending forever, ``alive`` never re-arms and a dead row's ``returns``
    stops growing.
    """
    backend, device = device_placement
    kwargs = {
        "horizon": TIMEOUT_HORIZON,
        "episode_length_s": TIMEOUT_EPISODE_LENGTH_S,
        "stop_when_done": False,
    }
    reset_log = _record_reset(monkeypatch)
    numpy = _recorded_generation(monkeypatch, "numpy", None, reset_log=reset_log, **kwargs)
    recorded = _recorded_generation(
        monkeypatch, "device", device, population=numpy.population, reset_log=reset_log, **kwargs
    )
    assert numpy.max_episode_length == recorded.max_episode_length == TIMEOUT_MAX_EPISODE_LENGTH, (
        f"expected max_episode_length {TIMEOUT_MAX_EPISODE_LENGTH}, got {numpy.max_episode_length} and "
        f"{recorded.max_episode_length}"
    )
    assert numpy.result["steps"] == recorded.result["steps"] == TIMEOUT_HORIZON, (
        f"``stop_when_done=False`` must run the whole horizon, got {numpy.result['steps']} and "
        f"{recorded.result['steps']} of {TIMEOUT_HORIZON} steps"
    )
    assert numpy.result["end_reasons"] == recorded.result["end_reasons"], (
        f"end_reasons differ: numpy {numpy.result['end_reasons']} vs device {recorded.result['end_reasons']}"
    )
    end_reasons = numpy.result["end_reasons"]
    assert sum(end_reasons.values()) == ROWS, (
        f"every one of the {ROWS} rows must have ended inside {TIMEOUT_HORIZON} steps (max_episode_length + "
        f"expected slack), got {end_reasons}"
    )
    assert end_reasons["unfinished"] == 0, f"a row is still unfinished: {end_reasons}"
    assert end_reasons["timeout"] > 0, (
        f"the timeout branch never fired ({end_reasons}); the episode cap or the horizon is wrong and the "
        "comparison below would be vacuous"
    )

    import hover_kernels

    numpy_masks = {
        name: np.array([step["fields"][name] for step in numpy.steps])
        for name in ("done", "timeout", "crash", "alive", "episode_length", "end_code", "returns")
    }
    device_masks = {
        name: np.array([step["fields"][name] for step in recorded.steps])
        for name in ("done", "timeout", "crash", "alive", "episode_length", "end_code", "returns")
    }
    for name in ("done", "timeout", "crash", "alive", "episode_length", "end_code"):
        assert np.array_equal(numpy_masks[name], device_masks[name]), (
            f"the per-step {name!r} sequence must be identical across residencies; first differing step "
            f"{int(np.argmax((numpy_masks[name] != device_masks[name]).any(axis=1)))}"
        )

    # The counter increments every step and is never reset (one episode per row).
    expected_length = np.arange(1, TIMEOUT_HORIZON + 1, dtype=np.int32)[:, None] * np.ones((1, ROWS), np.int32)
    assert np.array_equal(numpy_masks["episode_length"], expected_length), (
        f"episode_length must increment by one every step with no reset, got {numpy_masks['episode_length'].tolist()}"
    )
    # ``alive`` never re-arms.
    assert np.all(np.diff(numpy_masks["alive"].astype(np.int8), axis=0) <= 0), (
        f"'alive' must never re-arm; per-step alive counts {numpy_masks['alive'].sum(axis=1).tolist()}"
    )
    # ``end_code`` latches the first ending forever.
    latched = numpy_masks["end_code"].copy()
    for step in range(1, latched.shape[0]):
        previous = latched[step - 1]
        latched[step] = np.where(previous != hover_kernels.END_REASONS["unfinished"], previous, latched[step])
    assert np.array_equal(latched, numpy_masks["end_code"]), (
        "'end_code' must latch the row's first ending and never be re-armed; the sequence was rewritten"
    )
    # The timeout fires exactly on the step whose counter exceeds the cap.
    done = numpy_masks["done"]
    assert bool(done.any(axis=0).all()), "some row never ended in the whole horizon"
    first_done = done.argmax(axis=0)
    timeout_rows = numpy_masks["end_code"][-1] == hover_kernels.END_REASONS["timeout"]
    assert timeout_rows.any() and np.all(first_done[timeout_rows] + 1 == TIMEOUT_MAX_EPISODE_LENGTH + 1), (
        f"a timed-out row must first end on step {TIMEOUT_MAX_EPISODE_LENGTH + 1} (its counter exceeds the cap "
        f"there), got first-done steps {first_done.tolist()}"
    )
    # A dead row's returns stop growing on the step it ends.
    returns = numpy_masks["returns"]
    for row in range(ROWS):
        assert returns[first_done[row], row] == returns[-1, row], (
            f"row {row} kept accumulating returns after it ended on step {int(first_done[row]) + 1}: "
            f"{float(returns[first_done[row], row])!r} -> {float(returns[-1, row])!r}"
        )
    # ``stop_when_done=True`` would have stopped the loop right after the last ending.
    assert not numpy_masks["alive"][TIMEOUT_MAX_EPISODE_LENGTH].any(), (
        "every row must be dead once the timeout step has run, otherwise the default stop_when_done loop "
        "would not stop at the documented step"
    )


# ---------------------------------------------------------------------------------------------
# Documented end-to-end numbers
# ---------------------------------------------------------------------------------------------


def test_two_generation_end_to_end_reference_numbers(gpu_genesis, residency_runs):
    """Two full-length generations at each residency's own population, against the documented numbers.

    Each residency's generation-0 population is the trainer's own (device-first initialization under the
    device residency).  Documented: the seed table rows, ``steps`` ``[160, 58]`` for ``--pop-size 8
    --generations 2``, identical ``end_reasons`` and ``returns``/``fitness`` agreeing to exactly ``0.0``
    across residencies.  The absolute fitness values are not documented, so :data:`DOCUMENTED_HISTORY` is
    ``None`` and only the cross-residency agreement is pinned here.
    """
    numpy_run = residency_runs["numpy"]
    device_run = residency_runs["device"]

    assert evolution_seed(MASTER_SEED) == DOCUMENTED_EVOLUTION_SEED, (
        f"seeding.evolution_seed({MASTER_SEED}) is {evolution_seed(MASTER_SEED)}, documented as "
        f"{DOCUMENTED_EVOLUTION_SEED}; every run below is seeded from it"
    )
    for label, run in (("numpy", numpy_run), ("device", device_run)):
        assert run.seed_table.tolist() == DOCUMENTED_SEED_TABLE[:GENERATIONS], (
            f"{label}: the seed table of the {GENERATIONS} generations is {run.seed_table.tolist()}, "
            f"documented as {DOCUMENTED_SEED_TABLE[:GENERATIONS]}"
        )
        assert [record["seeds"] for record in run.records] == DOCUMENTED_SEED_TABLE[:GENERATIONS], (
            f"{label}: the env reported episode seeds {[record['seeds'] for record in run.records]}, which "
            f"must be the trainer's own table {DOCUMENTED_SEED_TABLE[:GENERATIONS]}"
        )
        assert len(run.records) == GENERATIONS, f"{label}: {len(run.records)} records for {GENERATIONS} generations"
        # The generation-0 population is the trace population both residencies draw, so its step count is
        # the documented one on both paths.
        assert run.records[0]["steps"] == DOCUMENTED_STEPS[0], (
            f"{label}: generation 0 ran {run.records[0]['steps']} steps, documented as {DOCUMENTED_STEPS[0]}"
        )
        for generation, record in enumerate(run.records):
            assert sum(record["end_reasons"].values()) == ROWS, (
                f"{label} generation {generation}: end_reasons {record['end_reasons']} do not account for all "
                f"{ROWS} rows"
            )
            assert record["end_reasons"]["unfinished"] == 0, (
                f"{label} generation {generation}: a row is unfinished ({record['end_reasons']}), but the "
                f"generation stopped after {record['steps']} of {FULL_HORIZON} steps"
            )

    # The documented generation-1 step count belongs to the numpy residency's own run (the device residency
    # tells on its own IREE device, so its generation-1 population is a different float payload).
    assert [record["steps"] for record in numpy_run.records] == DOCUMENTED_STEPS, (
        f"numpy: steps per generation {[record['steps'] for record in numpy_run.records]}, documented as "
        f"{DOCUMENTED_STEPS}"
    )

    # Documented cross-residency exactness.
    assert [record["steps"] for record in device_run.records] == [
        record["steps"] for record in numpy_run.records
    ], (
        "the two residencies must run the same number of steps per generation; documented as identical "
        f"(numpy {[record['steps'] for record in numpy_run.records]}, device "
        f"{[record['steps'] for record in device_run.records]})"
    )
    for generation, (numpy_record, device_record) in enumerate(
        zip(numpy_run.records, device_run.records, strict=True)
    ):
        assert device_record["seeds"] == numpy_record["seeds"], (
            f"generation {generation}: seeds differ ({device_record['seeds']} vs {numpy_record['seeds']})"
        )
        assert device_record["end_reasons"] == numpy_record["end_reasons"], (
            f"generation {generation}: end_reasons differ (device {device_record['end_reasons']} vs numpy "
            f"{numpy_record['end_reasons']})"
        )
        for name in ("returns", "episode_returns", "fitness"):
            assert_allclose(
                device_record[name],
                numpy_record[name],
                atol=CROSS_RESIDENCY_ATOL,
                rtol=0,
                label=f"generation {generation} {name} (device vs numpy)",
            )
    assert_allclose(
        device_run.fitness_history,
        numpy_run.fitness_history,
        atol=HISTORY_ATOL,
        rtol=0,
        label="fitness_history (device vs numpy residency)",
    )
    assert len(numpy_run.fitness_history) == GENERATIONS and np.all(np.isfinite(numpy_run.fitness_history)), (
        f"fitness_history {numpy_run.fitness_history} must hold {GENERATIONS} finite entries"
    )
    if DOCUMENTED_HISTORY is not None:
        assert_allclose(
            numpy_run.fitness_history,
            DOCUMENTED_HISTORY,
            atol=HISTORY_ATOL,
            rtol=0,
            label="fitness_history vs the documented two-generation history",
        )

    # Placement bookkeeping of the two runs, from the trainer itself.
    assert set(numpy_run.placement["pop"]) == {"host"}, (
        f"the numpy residency must keep its payload on the host, got {numpy_run.placement['pop']}"
    )
    assert numpy_run.generation == device_run.generation, (
        f"the generation counter after {GENERATIONS} generations differs: {numpy_run.generation} vs "
        f"{device_run.generation}"
    )
    assert numpy_run.randkey == device_run.randkey, (
        f"the RNG key after {GENERATIONS} generations differs: numpy {numpy_run.randkey} vs device "
        f"{device_run.randkey}; the tell graphs share one TensorNEAT transition"
    )


# ---------------------------------------------------------------------------------------------
# Device placement and the Torch-free training path
# ---------------------------------------------------------------------------------------------

CHILD_SCRIPT = """
import json
import sys

import numpy as np

import evoxc
import genesis as gs

sys.path.insert(0, {drone!r})
sys.path.insert(0, {neuroevolution!r})

evoxc.backends.get({backend!r})
device = evoxc.Device({kind!r}, {index})
evoxc.from_numpy(np.zeros(1, dtype=np.float32)).to(device)

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)

from hover_neat_env import HoverNeatEnv
from seeding import Perturbation
from trainer import NeuroevolutionTrainer
from device_runtime import assert_device_resident

env = HoverNeatEnv(
    pop_size={pop_size},
    master_seed={master_seed},
    perturbation=Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.0),
    episode_length_s={episode_length_s},
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
for leaf in trainer.pop_arrays:
    assert_device_resident(leaf, device, "pop leaf")
for leaf in trainer.species_leaves:
    assert_device_resident(leaf, device, "species leaf")
assert_device_resident(trainer.randkey, device, "randkey")
assert_device_resident(trainer.generation_counter, device, "generation")
trainer.run()

payload = {{
    "fitness_history": [float(value) for value in trainer.fitness_history],
    "steps": [int(record["steps"]) for record in trainer.records],
    "end_reasons": [dict(record["end_reasons"]) for record in trainer.records],
    "seeds": [list(record["seeds"]) for record in trainer.records],
    "placement": {{name: value for name, value in trainer.placement.items()}},
    "devices": [f"{{leaf.device.kind}}:{{leaf.device.index}}" for leaf in trainer.pop_arrays],
    "step_type": type(trainer.records[0]["steps"]).__name__,
    "end_reasons_type": type(trainer.records[0]["end_reasons"]).__name__,
    "torch_modules": sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.")),
}}
env.close()
gs.destroy()
print("HOVER_DEVICE " + json.dumps(payload))
"""

MARKER = "HOVER_DEVICE"


@pytest.mark.slow
def test_device_resident_training_without_torch(gpu_only, device_placement):
    """Two generations of a device-resident trainer in a child where ``import torch`` fails.

    The child builds ``HoverNeatEnv``/``NeuroevolutionTrainer`` device-resident, asserts the placement of
    every leaf with ``device_runtime.assert_device_resident``, trains, and reports the trainer's own
    ``placement`` map plus its records.  The torch-blocked harness fails the child if PyTorch reaches
    ``sys.modules``; the assertions here pin that every training leaf stayed on the device, that
    ``steps``/``end_reasons`` stayed host data, and the documented step counts.

    Marked ``slow``: a cold scene build, the ``iree`` compiles of the policy/controls/advance/fitness/tell
    graphs and ~220 steps of a 24-row drone scene, in a child that also pays the HSA re-exec.
    """
    backend, device = device_placement
    script = CHILD_SCRIPT.format(
        drone=str(DRONE_DIR),
        neuroevolution=str(NEUROEVOLUTION_DIR),
        backend=DEVICE_BACKEND,
        kind=device.kind,
        index=int(device.index),
        pop_size=POP_SIZE,
        master_seed=MASTER_SEED,
        generations=GENERATIONS,
        episode_length_s=EPISODE_LENGTH_S,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
    )
    completed = run_python_checked(script, timeout=CHILD_TIMEOUT_S)
    payload = _marker_payload(completed.stdout)

    assert payload["torch_modules"] == [], (
        f"the torch-free child imported PyTorch: {payload['torch_modules']}; the device residency must not "
        "need it at all"
    )
    target = f"{device.kind}:{device.index}"
    assert set(payload["devices"]) == {target}, (
        f"the population leaves must stay on {target}, got {payload['devices']}"
    )
    placement = payload["placement"]
    assert placement["pop"] == [target] * len(payload["devices"]), (
        f"every payload leaf must be device-resident, got {placement['pop']}"
    )
    assert set(placement["species"]) == {target}, (
        f"every species leaf must be device-resident, got {placement['species']}"
    )
    for name in ("randkey", "generation", "fitness", "episode_returns", "returns", "seeds"):
        assert placement[name] == target, (
            f"{name!r} must be device-resident on {target} (the device evaluation contract's four metrics "
            f"plus the RNG key and the generation counter), got {placement[name]!r}"
        )
    assert payload["step_type"] == "int" and payload["end_reasons_type"] == "dict", (
        f"steps/end_reasons are host logging data, got {payload['step_type']} and "
        f"{payload['end_reasons_type']}"
    )

    assert payload["seeds"] == DOCUMENTED_SEED_TABLE[:GENERATIONS], (
        f"episode seeds {payload['seeds']}, documented as {DOCUMENTED_SEED_TABLE[:GENERATIONS]}"
    )
    assert payload["steps"] == DOCUMENTED_STEPS, f"steps {payload['steps']}, documented as {DOCUMENTED_STEPS}"
    for generation, end_reasons in enumerate(payload["end_reasons"]):
        assert sum(end_reasons.values()) == ROWS and end_reasons.get("unfinished", 0) == 0, (
            f"generation {generation}: end_reasons {end_reasons}, expected every one of the {ROWS} rows to end"
        )
    assert len(payload["fitness_history"]) == GENERATIONS and np.all(np.isfinite(payload["fitness_history"])), (
        f"fitness_history {payload['fitness_history']} must hold {GENERATIONS} finite entries"
    )
    if DOCUMENTED_HISTORY is not None:
        assert_allclose(
            payload["fitness_history"],
            DOCUMENTED_HISTORY,
            atol=HISTORY_ATOL,
            rtol=0,
            label="torch-free device fitness_history vs the documented history",
        )


#: The import the CLI scenario's ``torch`` stand-in raises, so ``import torch`` fails inside the real script.
TORCH_BLOCKER_SOURCE = '''\
raise ImportError("PyTorch is blocked by the torch-free test harness")
'''


def _torch_blocker_dir(tmp_path) -> Path:
    """A directory holding a ``torch`` package that refuses to import, for ``PYTHONPATH``."""
    directory = Path(tmp_path) / "notorch"
    (directory / "torch").mkdir(parents=True)
    (directory / "torch" / "__init__.py").write_text(TORCH_BLOCKER_SOURCE, encoding="utf-8")
    return directory


@pytest.mark.slow
def test_torch_free_cli_runs_with_torch_import_blocked(tmp_path, gpu_only, device_placement):
    """The real ``hover_neat_train.py --torch-free`` CLI, with a raising ``torch`` first on ``PYTHONPATH``.

    The script runs as a script (not through ``run_python``), so the only blocker is the ``torch`` package on
    ``PYTHONPATH`` -- the pilot's own Torch-free claim, not the harness'.  A probe proves the stand-in really
    blocks before the run is trusted, and the artifacts are then checked against the documented numbers.

    Marked ``slow``: the same cold compiles and ~220 steps as the child above, plus a second process.
    """
    backend, device = device_placement
    blocker = _torch_blocker_dir(tmp_path)
    mpl_config_dir = Path(tmp_path) / "mpl"
    mpl_config_dir.mkdir()
    child_env = dict(os.environ)
    inherited = child_env.get("PYTHONPATH", "").strip()
    parts = [str(blocker), str(REPO_ROOT)] + ([inherited] if inherited else [])
    child_env["PYTHONPATH"] = os.pathsep.join(parts)
    child_env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    def run(argv, timeout):
        return subprocess.run(
            argv, cwd=str(REPO_ROOT), env=child_env, capture_output=True, text=True, check=False, timeout=timeout
        )

    # Prove the stand-in blocks before trusting a run that claims no torch and then exits 0 anyway.
    probe = run([sys.executable, "-c", "import torch"], timeout=300)
    assert probe.returncode != 0, (
        f"'import torch' succeeded although the blocker was first on PYTHONPATH: {probe.stdout}"
    )
    assert "blocked by the torch-free test harness" in probe.stderr, probe.stderr

    output_dir = Path(tmp_path) / "out"
    argv = [
        sys.executable,
        str(DRONE_DIR / TORCH_FREE_CLI),
        TORCH_FREE_FLAG,
        "--residency",
        "device",
        "-g",
        "--evoxc-backend",
        DEVICE_BACKEND,
        "--evoxc-device",
        f"{device.kind}:{device.index}",
        "--pop-size",
        str(POP_SIZE),
        "--generations",
        str(GENERATIONS),
        "--horizon",
        str(FULL_HORIZON),
        "--episode-length-s",
        str(EPISODE_LENGTH_S),
        "--output-dir",
        str(output_dir),
    ]
    completed = run(argv, timeout=CLI_TIMEOUT_S)
    assert completed.returncode == 0, (
        f"The --torch-free CLI exited with code {completed.returncode}.\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    assert config["residency"] == "device", config
    assert config["genesis_backend"] == "gpu", config
    assert config["evoxc_backend"] == DEVICE_BACKEND, config
    assert config["evolution_seed"] == DOCUMENTED_EVOLUTION_SEED, config["evolution_seed"]
    # The flag's claim: PyTorch never entered the process.  ``.get`` because the field's spelling is the
    # CLI's own (``examples/locomotion``'s pilot records it as ``torch_in_process``).
    assert config.get("torch_in_process") is not True, config
    assert [record["steps"] for record in summary["generations"]] == DOCUMENTED_STEPS, (
        f"steps {[record['steps'] for record in summary['generations']]}, documented as {DOCUMENTED_STEPS}"
    )
    assert [record["seeds"] for record in summary["generations"]] == DOCUMENTED_SEED_TABLE[:GENERATIONS], (
        f"seeds {[record['seeds'] for record in summary['generations']]}, documented as "
        f"{DOCUMENTED_SEED_TABLE[:GENERATIONS]}"
    )
    history = [json.loads(line) for line in (output_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(history) == GENERATIONS, history
    assert [record["best_fitness"] for record in history] == summary["totals"]["fitness_history"], history


def _marker_payload(stdout, marker=MARKER):
    """Parse the JSON payload of the child's single ``<marker> {...}`` line."""
    for line in stdout.splitlines():
        if line.startswith(marker + " "):
            return json.loads(line[len(marker) + 1 :])
    raise AssertionError(f"the child printed no {marker!r} line; its output was:\n{stdout}")

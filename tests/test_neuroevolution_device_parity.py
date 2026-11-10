"""Cross-residency parity of the shared neuroevolution loop: ``numpy`` reference vs ``device``.

``examples/neuroevolution`` carries two implementations of one loop: the host (``residency="numpy"``)
reference, which stages every operand through numpy, and the device-resident (``residency="device"``)
path, which keeps the genome payload, the species pytree, the RNG key, the generation counter and the
fitness on one ``evoxc`` device and never round-trips them.  This module pins their numeric agreement
with fixed seeds and the tolerances the shared layer documents.

What is promised
----------------
Exactness across residencies holds **only** for the integer genome leaves, the RNG ``randkey`` and the
``generation`` counter; every float leaf is a tolerance check, because the device graph is a float32
reduction order of the same arithmetic that differs from numpy's.
Source: ``examples/neuroevolution/CONTEXT.md`` ("Device-residency loop") and ``verify_device.py``'s
``initialization_parity``/``residency_parity``.

Two tiers
---------
* **CPU plumbing** (``backend="numpy"``, ``device="cpu:0"``, the layer's plumbing-only device mode):
  both sides are the same code path, so every comparison is bitwise.  If a float leaf ever stops being
  bitwise equal on ``cpu:0``, the documented ``CPU_FLOAT_ATOL`` (1e-5) bound applies instead and the
  measured value is in the failure message.
* **Accelerator** (``backend="iree"``, ``device="rocm:0"``, pop 8, 2 generations, every test
  ``@pytest.mark.slow``): skipped cleanly when no non-CPU ``evoxc`` device is reachable.  Nothing here
  needs a ``genesis`` scene, a viewer or a network.

Tolerances (module constants, each naming its source)
-----------------------------------------------------
* ``DEVICE_FLOAT_ATOL = 1e-5`` -- one float *leaf*: ``verify_device.py``'s ``--tolerance`` default.
* ``DEVICE_AGGREGATE_ATOL = DEVICE_AGGREGATE_RTOL = 1e-5`` -- one aggregated return/fitness value: the
  ``numpy.allclose(atol, rtol)`` convention ``CONTEXT.md`` documents for cross-residency numbers.  The
  Go2 parity style's ``1e-6`` on a reward *total* is the host-side contract (``trainer._aggregate``'s
  own check) and cannot hold here: the synthetic return sums 320 float32 genome leaves up to ~99, where
  one float32 ulp is already 7.6e-06.
* ``HISTORY_ATOL = HISTORY_RTOL = 1e-5`` -- the per-generation ``fitness_history``: ``residency_parity``
  compares it exactly this way.
* ``CPU_FLOAT_ATOL = 1e-5`` -- the CPU tier's fallback bound (same source as ``DEVICE_FLOAT_ATOL``).

Measured on an AMD Instinct MI300X VF (gfx942), ROCm 7.1.0, iree 3.13.0, pop 8, master seed 0; every
number is also asserted in the test that measures it, and lives in that test's docstring:

* ``cpu:0``: every float leaf, record and history 0.0 (bitwise on both sides);
* ``device_setup`` vs ``Pipeline.setup()``: 4.768e-07 max abs on the float leaves, integer leaves /
  ``randkey`` / ``generation`` exact;
* policy forward 45 -> 12, numpy graph vs accelerator graph: 0.0 max abs (actions up to ~18);
* synthetic ``episode_returns``/``returns``: 2.289e-05 max abs / 2.510e-07 rel (values ~99, ~3 float32
  ulp); ``fitness`` 1.526e-05 abs / 1.542e-07 rel; records ``episode_returns`` 2.289e-05, ``fitness``
  3.052e-05 abs / 3.242e-07 rel; ``fitness_history`` over 2 generations 7.629e-06 abs / 7.695e-08 rel;
* first ``run_tell``: 2.384e-07 max abs on the payload/species leaves, ``generation`` exactly +1,
  ``randkey`` exact.

``SyntheticDeviceEnv`` has no termination: its ``end_reasons`` are the host constant
``{"timeout": rows}``, so this module's ``done`` check can only pin the host metadata both residencies
report (``end_reasons``/``steps``/``env_steps``); termination-*mask* parity is pilot-level.

Not covered yet: the Torch-free Go2 pilot
-----------------------------------------
``examples/locomotion/go2_neat_env.py``'s ``residency="device"`` dispatch still goes through
``go2_device_rollout.py``, which imports ``torch`` (and ``genesis``) for its DLPack bridge, so the Go2
pilot's device path cannot run without torch today; ``test_go2_torch_free_pilot_parity`` detects that
with an AST scan and skips.  A follow-up must add, against the torch/numpy reference, at pilot level
(Go2 walking, 45 -> 12, ``max_episode_length=1000``, pop 8 / 24 rows):

1. **observations**: the per-row observation the device rollout builds equals
   ``go2_env._update_observation``'s (base position/quaternion/velocities, dof position/velocity, last
   action) to the reduction tolerance;
2. **reward terms**: every term of ``go2_legacy_reference.walking_reward_terms`` plus the total, per
   term at 1e-5 and 1e-6 on the total (the profile ``test_go2_evox_parity.py`` uses);
3. **roll/pitch termination mask**: identical ``done`` flags, including the ``max_episode_length``
   timeout and the never-re-arming ``alive`` mask (the profile ``test_go2_neat_kernels.py`` pins at
   graph level);
4. **episode returns**: the ``(pop, 3)`` generation matrix, elementwise;
5. **fitness**: the policy-major episode mean, i.e. the trainer's one aggregation rule;
6. **first population update**: one ``run_tell`` from the same generation-0 state -- integer leaves /
   ``randkey`` / ``generation`` exact, float leaves within 1e-5.

Example::

    PYTHONPATH=$PWD:/tmp/gs-pytest-libs /root/evox-robotics/.venv/bin/python -m pytest \
        tests/test_neuroevolution_device_parity.py -q -n 0

``evoxc`` and ``tensorneat`` are not declared dependencies of ``genesis-world``, so the module skips
where they are not installed; the accelerator tier additionally needs a reachable non-CPU ``evoxc``
device and is marked ``slow``.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

pytest.importorskip("evoxc")
pytest.importorskip("tensorneat")

#: ``iree`` truncates its ROCm compile of the tell graph without this flag (``iree-compile`` never
#: returns); ``evolution_graphs.build_tell`` passes it itself for ``iree`` on cuda/rocm, but the
#: policy and synthetic-returns graphs this module builds do not, so the module sets it up front.
#: Source: ``examples/neuroevolution/CONTEXT.md`` "Device backends".
os.environ.setdefault("EVOXC_IREE_COMPILE_ARGS", "--iree-codegen-llvmgpu-use-reduction-vector-distribution=false")

pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: ``examples/`` are plain scripts, not a package, so the shared layer needs its directory on
#: ``sys.path`` before any bare-name import (the bootstrap ``test_neuroevolution_trainer.py`` uses).
EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
NEUROEVOLUTION_DIR = EXAMPLES_DIR / "neuroevolution"
LOCOMOTION_DIR = EXAMPLES_DIR / "locomotion"
for _directory in (NEUROEVOLUTION_DIR, LOCOMOTION_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import evoxc  # noqa: E402

from tensorneat import Pipeline, algorithm, genome  # noqa: E402

from device_env import SyntheticDeviceEnv  # noqa: E402
from device_runtime import device_payload, device_setup, device_species_leaves, is_device_tensor  # noqa: E402
from evolution_graphs import build_tell, run_tell  # noqa: E402
from evox_runtime import _to_numpy, resolve_backend_device  # noqa: E402
from policy_graphs import build_policy, run_policy  # noqa: E402
from seeding import EPISODES_PER_POLICY, evolution_seed, generation_seed_table  # noqa: E402
from trainer import NeuroevolutionTrainer, TraceOnlyProblem, pop_arrays_of, species_leaves_of  # noqa: E402

#: The two real task widths, so every graph below traces the shapes the pilots use (Go2 walking).
NUM_INPUTS = 45
NUM_OUTPUTS = 12
#: The genome budget of both pilots (``trainer.py``'s defaults), so the payload shapes match.
MAX_NODES = 80
MAX_CONNS = 640
#: NEAT species size, clamped to ``pop_size`` like the trainer does.
SPECIES_SIZE = 3
#: The master seed both residencies are run at.
MASTER_SEED = 0
#: Policies per generation: 8 (24 rows), the smallest size ``verify_device.py`` measures at.
POP_SIZE = 8
#: Generations the trainer parity runs: 2 is one ``tell`` beyond the initial population.
GENERATIONS = 2
#: ``SyntheticDeviceEnv``'s default simulated steps per episode, reported as ``steps``.
SYNTHETIC_HORIZON = 25
#: Seed of the policy-graph observation tile; any fixed tile works, only the agreement matters.
OBSERVATION_SEED = 7
#: The episode-seed table both residencies are handed: the env stream of the master seed.
SEED_TABLE = generation_seed_table(MASTER_SEED, GENERATIONS, episodes=EPISODES_PER_POLICY)
#: The plumbing-only device mode of the layer (``backend="numpy"`` on ``cpu:0``).
CPU_BACKEND = "numpy"
CPU_DEVICE = "cpu:0"
#: The backend the device residency serves ROCm with (``iree`` is the only one the loop uses there).
DEVICE_BACKEND = "iree"

#: One float genome/species *leaf*: ``verify_device.py``'s ``--tolerance`` default (measured
#: 4.768e-07 on ``iree@rocm:0``, 0.0 on ``cpu:0``).
DEVICE_FLOAT_ATOL = 1e-5
#: One aggregated return/fitness value: the ``atol = rtol = 1e-5`` convention ``CONTEXT.md``
#: documents for cross-residency numbers; the measured drift is 2-3 float32 ulp of the ~99
#: magnitude (see ``DEVICE_AGGREGATE_RTOL``'s comment and the module docstring).
DEVICE_AGGREGATE_ATOL = 1e-5
#: Relative half of the aggregate bound: the same convention (``numpy.allclose``'s signature).
DEVICE_AGGREGATE_RTOL = 1e-5
#: The per-generation ``fitness_history``: exactly how ``verify_device.residency_parity``
#: compares it (``np.allclose(atol=tolerance, rtol=tolerance)``).
HISTORY_ATOL = 1e-5
HISTORY_RTOL = 1e-5
#: Fallback bound of the CPU tier: the documented device tolerance, used only if a ``cpu:0``
#: float leaf stops being bitwise equal (measured 0.0 for every leaf today).
CPU_FLOAT_ATOL = 1e-5
#: The aggregation tolerance of the *host* path (``trainer._aggregate``'s own check, the Go2
#: parity style's "1e-6 on the total").  Stated here for the record; no cross-residency comparison
#: can hold it at the synthetic env's ~99 magnitude (see the module docstring).
FITNESS_CONTRACT_ATOL = 1e-6

#: Record fields derived from wall-clock time, stripped before two runs are compared
#: (the tuple of ``tests/test_neuroevolution_trainer.py``).
TIMING_FIELDS = ("wall_time_s", "steps_per_s", "env_steps_per_s", "executable_build_time_s", "tell_build_time_s")
#: Cache counters, stripped as well: a device trainer compiles two extra graphs (fitness rule and
#: verification), so its hits/misses legitimately differ from the numpy path
#: (``CONTEXT.md`` "Cross-residency comparisons must strip the cache counters ...").
CACHE_FIELDS = ("cache_hits", "cache_misses", "cache_hits_delta", "cache_misses_delta")
#: The record fields that carry float training data; every other record field is host
#: metadata/integer data and is compared exactly across residencies.
RECORD_FLOAT_FIELDS = ("episode_returns", "returns", "fitness", "best_fitness", "median_fitness")


# --- Helpers ------------------------------------------------------------------------


class LeafComparison(NamedTuple):
    """Evidence of one leaf-sequence comparison."""

    #: Whether every float leaf is bitwise equal (NaN padding of unused slots counts as equal).
    bitwise: bool
    #: Largest finite absolute difference over the float leaves.
    float_max_abs: float
    #: Number of integer leaves compared (always bitwise-equal, or the helper failed).
    integer_leaves: int
    #: Number of float leaves compared.
    float_leaves: int


def _as_host(value) -> np.ndarray:
    """Materialize an ``evoxc`` tensor as numpy; pass a host array/scalar through unchanged.

    ``numpy`` 2.x exposes ``ndarray.device`` as the *string* ``"cpu"``, so ``is_device_tensor``
    (not ``hasattr(value, "device")``) is the tensor predicate.
    """
    if isinstance(value, np.ndarray):
        return value
    if is_device_tensor(value):
        return _to_numpy(value)
    return np.asarray(value)


def _max_abs_diff(left, right) -> float:
    """Largest finite absolute difference; NaN padding is ignored, a shape mismatch is ``inf``.

    The convention of ``verify_device.float_max_abs_diff``.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    with np.errstate(invalid="ignore"):
        differences = np.abs(left - right)
    differences = differences[np.isfinite(differences)]
    return float(differences.max()) if differences.size else 0.0


def _max_rel_diff(left, right) -> float:
    """Largest relative difference against the magnitude of ``right``; ``inf`` on a shape mismatch.

    Where the reference is zero the absolute difference is reported instead; the convention of
    ``verify_device.float_max_rel_diff``.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    with np.errstate(invalid="ignore", divide="ignore"):
        differences = np.abs(left - right)
        scale = np.abs(right)
        relative = np.where(scale > 0.0, differences / np.where(scale > 0.0, scale, 1.0), differences)
    relative = relative[np.isfinite(relative)]
    return float(relative.max()) if relative.size else 0.0


def _assert_allclose(device_value, reference_value, *, atol, rtol, label) -> tuple[float, float]:
    """Assert ``device_value`` is within ``atol + rtol * |reference|``; returns the measured diffs.

    The measured absolute and relative differences are in the message, so a regression reports how
    far it moved instead of only that it moved.
    """
    device = np.asarray(_as_host(device_value), dtype=np.float64)
    reference = np.asarray(_as_host(reference_value), dtype=np.float64)
    abs_diff = _max_abs_diff(device, reference)
    rel_diff = _max_rel_diff(device, reference)
    assert np.allclose(device, reference, atol=atol, rtol=rtol, equal_nan=True), (
        f"{label}: device vs numpy max |diff| {abs_diff:.3e}, max rel {rel_diff:.3e}, outside "
        f"atol={atol:.1e} + rtol={rtol:.1e} x |reference| (largest |reference| "
        f"{float(np.max(np.abs(reference))) if reference.size else 0.0:.4g}; shapes "
        f"{device.shape} vs {reference.shape})"
    )
    return abs_diff, rel_diff


def _compare_leaves(host_leaves, device_leaves, *, label) -> LeafComparison:
    """Compare leaf sequences in order: integer leaves bitwise, float leaves measured.

    A shape mismatch or a non-bitwise integer leaf fails here -- neither is a tolerance question;
    the float evidence is returned for the caller to assert with its tier's bound.
    """
    host_leaves = tuple(host_leaves)
    device_leaves = tuple(device_leaves)
    assert len(host_leaves) == len(device_leaves), (
        f"{label}: {len(host_leaves)} host leaves vs {len(device_leaves)} device leaves; the two "
        "residencies must agree on the leaf layout of the same state"
    )
    bitwise = True
    float_max_abs = 0.0
    integer_leaves = 0
    float_leaves = 0
    problems = []
    for index, (host_leaf, device_leaf) in enumerate(zip(host_leaves, device_leaves, strict=True)):
        host = np.asarray(_as_host(host_leaf))
        device = np.asarray(_as_host(device_leaf))
        if host.shape != device.shape:
            problems.append(f"{label}[{index}]: host shape {host.shape} vs device shape {device.shape}")
            continue
        if np.issubdtype(host.dtype, np.integer):
            integer_leaves += 1
            differing = int(np.count_nonzero(host != device))
            if differing:
                problems.append(f"{label}[{index}] (integer {host.dtype}): {differing} entries differ")
        else:
            float_leaves += 1
            bitwise = bitwise and bool(np.array_equal(host, device, equal_nan=True))
            float_max_abs = max(float_max_abs, _max_abs_diff(host, device))
    assert not problems, (
        f"{label}: integer leaves/leaf shapes must be bitwise identical across residencies "
        f"(only float leaves are a tolerance check); measured float max |diff| {float_max_abs:.3e}: "
        f"{problems}"
    )
    return LeafComparison(
        bitwise=bitwise,
        float_max_abs=float_max_abs,
        integer_leaves=integer_leaves,
        float_leaves=float_leaves,
    )


def _assert_leaf_parity(host_leaves, device_leaves, *, label, exact=False, atol=DEVICE_FLOAT_ATOL) -> LeafComparison:
    """Leaf parity of one tier: integers exact, floats bitwise (``exact``) or within ``atol``.

    ``exact=True`` is the CPU plumbing tier, where both sides are one code path: a non-bitwise float
    leaf must at least stay inside ``atol`` (measured 0.0 for every CPU leaf today), and the measured
    value is in the message either way.
    """
    comparison = _compare_leaves(host_leaves, device_leaves, label=label)
    if exact:
        assert comparison.bitwise or comparison.float_max_abs <= atol, (
            f"{label}: both sides run the same code path, so every float leaf must be bitwise equal; "
            f"measured max |diff| {comparison.float_max_abs:.3e} over {comparison.float_leaves} float "
            f"leaves (documented bound {atol:.1e})"
        )
    else:
        assert comparison.float_max_abs <= atol, (
            f"{label}: {comparison.float_leaves} float leaves differ by max |diff| "
            f"{comparison.float_max_abs:.3e} across residencies, outside the documented {atol:.1e} "
            f"({comparison.integer_leaves} integer leaves bitwise equal)"
        )
    return comparison


def _algo(pop_size, num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS):
    """A fresh ``NEAT`` instance on the pilots' genome budget; neither side may share one."""
    generator = genome.DefaultGenome(
        num_inputs=num_inputs, num_outputs=num_outputs, max_nodes=MAX_NODES, max_conns=MAX_CONNS
    )
    return algorithm.NEAT(generator, pop_size=pop_size, species_size=min(SPECIES_SIZE, pop_size))


def _host_state(pop_size, master_seed=MASTER_SEED):
    """Build the host reference state exactly as the trainer's numpy residency does.

    Returns the algorithm/problem, the tensor-free ``Pipeline._static_config`` the compiled graphs
    are traced with, and the payload/species leaves, key and generation counter as host values.
    """
    algo = _algo(pop_size)
    problem = TraceOnlyProblem(NUM_INPUTS, NUM_OUTPUTS)
    pipeline = Pipeline(algo, problem, seed=evolution_seed(master_seed), backend="numpy", is_detailed=False)
    state = pipeline.setup()
    extractor = getattr(pipeline, "_static_config", None)
    assert callable(extractor), (
        "TensorNEAT's private Pipeline._static_config is missing; the trainer and these parity "
        "tests both trace the compiled graphs with it (pinned TensorNEAT version changed?)"
    )
    return {
        "algo": algo,
        "problem": problem,
        "config": extractor(state),
        "pop": pop_arrays_of(state),
        "species": tuple(_as_host(leaf) for leaf in species_leaves_of(state.species)),
        "randkey": _as_host(state.randkey),
        "generation": _as_host(state.generation),
    }


def _device_state(pop_size, backend, device, master_seed=MASTER_SEED):
    """Build the generation-0 state through ``device_runtime.device_setup`` (device-first init).

    The same dict as :func:`_host_state`, with every runtime leaf still a device tensor.
    """
    algo = _algo(pop_size)
    problem = TraceOnlyProblem(NUM_INPUTS, NUM_OUTPUTS)
    config, state = device_setup(
        algo, problem, seed=evolution_seed(master_seed), backend=backend, device=device
    )
    return {
        "algo": algo,
        "problem": problem,
        "config": config,
        "pop": device_payload(state),
        "species": device_species_leaves(state),
        "randkey": state.randkey,
        "generation": state.generation,
    }


def _to_device(array, backend, device):
    """Stage one host array onto ``device``, re-selecting the backend first (as ``device_env._from_host`` does).

    ``Tensor.to`` resolves the transfer provider of the backend looked up last, hence the re-selection.
    """
    resolved_device = resolve_backend_device(backend, device)[1]
    if resolved_device.kind != "cpu":
        evoxc.backends.get(backend)
    return evoxc.from_numpy(np.ascontiguousarray(np.asarray(array))).to(resolved_device)


def _observation_tile(rows, num_inputs=NUM_INPUTS):
    """A deterministic ``(rows, num_inputs)`` float32 observation tile."""
    return np.random.default_rng(OBSERVATION_SEED).normal(size=(rows, num_inputs)).astype(np.float32)


def _fitness_vector(pop_size):
    """A deterministic, decidedly non-uniform fitness vector for the tell comparisons."""
    return np.linspace(-0.5, 1.5, pop_size, dtype=np.float32)


def _policy_executable(algo, config, pop_arrays, *, backend, device=None):
    """Compile the shared policy graph at this module's widths (45 -> 12, 3 episodes per policy)."""
    return build_policy(
        algo,
        config,
        pop_arrays,
        POP_SIZE * EPISODES_PER_POLICY,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        episodes=EPISODES_PER_POLICY,
        backend=backend,
        device=device,
    )


def _tell_executable(algo, config, pop_arrays, species_leaves, *, backend, device=None):
    """Compile the tell graph of one generation at this module's population size."""
    return build_tell(algo, config, pop_arrays, species_leaves, POP_SIZE, backend=backend, device=device)


def _evaluate_generation(pop_arrays, residency, *, backend, device, generation=1):
    """Run one synthetic generation in one residency; the env is created and closed here."""
    env = SyntheticDeviceEnv(
        POP_SIZE,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        episodes=EPISODES_PER_POLICY,
        backend=backend,
        device=device,
        residency=residency,
        seed=MASTER_SEED,
        seed_table=SEED_TABLE,
    )
    try:
        return env.evaluate(pop_arrays, generation)
    finally:
        env.close()


def _one_tell(state, *, backend, device, residency):
    """Build the tell graph of one side and run one generation transition from its state."""
    executable = _tell_executable(
        state["algo"], state["config"], state["pop"], state["species"], backend=backend, device=device
    )
    fitness = _fitness_vector(POP_SIZE)
    if residency == "device":
        fitness = _to_device(fitness, backend, device)
    return run_tell(
        executable,
        randkey=state["randkey"],
        pop_arrays=state["pop"],
        generation=state["generation"],
        fitness=fitness,
        species_leaves=state["species"],
        residency=residency,
    )


def _assert_next_key_and_generation(reference, actual):
    """The RNG key and the generation counter are the exact half of the cross-residency contract."""
    assert int(np.asarray(reference["randkey"])) == int(_as_host(actual["randkey"])), (
        f"next randkey {int(_as_host(actual['randkey']))} (device) != "
        f"{int(np.asarray(reference['randkey']))} (numpy); the key is documented as exact"
    )
    reference_generation = float(np.asarray(reference["generation"]))
    device_generation = float(_as_host(actual["generation"]))
    assert reference_generation == device_generation == 1.0, (
        f"the first update must advance generation by exactly +1: numpy {reference_generation}, device "
        f"{device_generation} (both start from the same generation-0 state)"
    )


def _strip_run_fields(value):
    """Return ``value`` without the wall-clock fields and the cache counters, recursively."""
    if isinstance(value, dict):
        return {
            key: _strip_run_fields(item)
            for key, item in value.items()
            if key not in TIMING_FIELDS and key not in CACHE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_run_fields(item) for item in value]
    return value


def _trainer(pop_size, residency, *, backend, device, generations=GENERATIONS, master_seed=MASTER_SEED):
    """Build ``(trainer, env)`` of one residency at one seed; the caller runs and closes them.

    Env and trainer are handed the same explicit seed table, so the residency is the only difference.
    """
    seed_table = generation_seed_table(master_seed, generations, episodes=EPISODES_PER_POLICY)
    env = SyntheticDeviceEnv(
        pop_size,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        episodes=EPISODES_PER_POLICY,
        backend=backend,
        device=device,
        residency=residency,
        seed=master_seed,
        seed_table=seed_table,
    )
    trainer = NeuroevolutionTrainer(
        env,
        pop_size,
        generations,
        master_seed=master_seed,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,
        species_size=SPECIES_SIZE,
        residency=residency,
        seed_table=seed_table,
    )
    return trainer, env


_DEVICE_CACHE: dict = {}


def _device_or_skip():
    """Return ``(backend, device)`` of a reachable non-CPU device, or skip the calling test.

    ``evoxc.devices()`` lists every runtime the wheel was built with, so a ROCm-only host also
    enumerates ``cuda:0``; the only honest reachability test is moving one float32 element there
    through the backend's placement provider (as ``verify_device.reachable_devices`` does).  The
    probe is cached: the first enumeration costs ~7.5 s.
    """
    if "placement" in _DEVICE_CACHE:
        return _DEVICE_CACHE["placement"]
    evoxc.backends.get(DEVICE_BACKEND)
    unusable = []
    for device in evoxc.devices():
        if device.kind == "cpu":
            continue
        try:
            evoxc.from_numpy(np.zeros(1, dtype=np.float32)).to(device)
        except Exception as exc:  # noqa: BLE001 - an unreachable runtime is an expected outcome to skip on
            unusable.append(f"{device.kind}:{device.index} ({type(exc).__name__})")
            continue
        _DEVICE_CACHE["placement"] = (DEVICE_BACKEND, device)
        return _DEVICE_CACHE["placement"]
    pytest.skip(
        f"no reachable non-CPU evoxc device for backend {DEVICE_BACKEND!r}; probed "
        f"{', '.join(unusable) or 'no non-CPU device'}"
    )


def _assert_record_parity(accelerated, reference, *, exact):
    """Compare two runs' records field by field, after the wall-clock/cache fields are stripped.

    ``exact=True`` is the CPU tier (same code path, bitwise records); otherwise the float metric
    fields are compared with the aggregate ``atol = rtol = 1e-5`` convention and every other field
    exactly (the record's ``seeds``, ``generation``, ``steps``, ``env_steps``, ``end_reasons`` and
    ``diagnostics`` are host/integer data and must not depend on the residency).
    """
    device_records = _strip_run_fields(accelerated.records)
    reference_records = _strip_run_fields(reference.records)
    assert [sorted(record) for record in device_records] == [
        sorted(record) for record in reference_records
    ], (
        "record schema differs across residencies after stripping the wall-clock fields "
        f"{TIMING_FIELDS} and the cache counters {CACHE_FIELDS}; device keys "
        f"{sorted(device_records[0])} vs numpy {sorted(reference_records[0])}"
    )
    for generation, (device_record, reference_record) in enumerate(
        zip(device_records, reference_records, strict=True)
    ):
        for key, reference_value in reference_record.items():
            if key in RECORD_FLOAT_FIELDS and not exact:
                _assert_allclose(
                    device_record[key],
                    reference_value,
                    atol=DEVICE_AGGREGATE_ATOL,
                    rtol=DEVICE_AGGREGATE_RTOL,
                    label=f"record[{generation}] {key!r} (device vs numpy)",
                )
            else:
                assert device_record[key] == reference_value, (
                    f"record[{generation}] {key!r} must be identical across residencies; device "
                    f"{device_record[key]!r} vs numpy {reference_value!r}"
                )


def _assert_cache_counters_differ(accelerated, reference):
    """The cache counters must legitimately differ (and therefore be stripped before comparing)."""
    assert accelerated.records[0]["cache_hits"] != reference.records[0]["cache_hits"] or (
        accelerated.records[0]["cache_misses"] != reference.records[0]["cache_misses"]
    ), (
        f"the cache counters are expected to differ between the residencies (device "
        f"{[(r['cache_hits'], r['cache_misses']) for r in accelerated.records]}, numpy "
        f"{[(r['cache_hits'], r['cache_misses']) for r in reference.records]}); the device trainer "
        "compiles the fitness-rule and verification graphs on top of the tell graph, and stripping "
        "them is what keeps the records comparable"
    )


def _assert_initialization_parity(host, accelerated, *, exact):
    """Compare the two generation-0 states; returns the ``(payload, species)`` leaf evidence.

    Integer leaves and leaf shapes must match everywhere (``_compare_leaves`` fails on those); the
    float leaves are the tier's tolerance check.
    """
    payload = _assert_leaf_parity(
        host["pop"], accelerated["pop"], label="payload", exact=exact, atol=CPU_FLOAT_ATOL
    )
    species = _assert_leaf_parity(
        host["species"], accelerated["species"], label="species", exact=exact, atol=CPU_FLOAT_ATOL
    )
    assert int(np.asarray(host["randkey"])) == int(_as_host(accelerated["randkey"])), (
        f"randkey {int(_as_host(accelerated['randkey']))} (device_setup) != "
        f"{int(np.asarray(host['randkey']))} (Pipeline.setup()); the RNG key is documented as bit-exact"
    )
    assert float(np.asarray(host["generation"])) == float(_as_host(accelerated["generation"])) == 0.0, (
        f"generation {float(_as_host(accelerated['generation']))} (device_setup) != 0.0; the counter "
        "is documented as bit-exact"
    )
    assert (payload.integer_leaves, species.integer_leaves) == (2, 2), (
        f"expected 2 integer payload leaves (node_keys, conn_keys) and 2 integer species leaves "
        f"(center_node_keys, center_conn_keys), got {payload.integer_leaves} and "
        f"{species.integer_leaves}; the leaf layout of the state changed"
    )
    return payload, species


def _policy_actions(host, *, backend, device=None):
    """Run the shared policy graph on one side with the same payload and observation tile."""
    obs = _observation_tile(POP_SIZE * EPISODES_PER_POLICY)
    executable = _policy_executable(host["algo"], host["config"], host["pop"], backend=backend, device=device)
    actions = run_policy(executable, host["pop"], obs)
    assert actions.shape == (POP_SIZE * EPISODES_PER_POLICY, NUM_OUTPUTS), (
        f"policy output shape {actions.shape} != {(POP_SIZE * EPISODES_PER_POLICY, NUM_OUTPUTS)} "
        "(rows = pop * episodes)"
    )
    return actions


# --- A. CPU plumbing tier (always runs, no skip) -------------------------------------


def test_cpu_initialization_parity():
    """``device_setup`` vs ``Pipeline(...).setup()`` on ``cpu:0``: every leaf bitwise equal.

    The CPU equivalent of the device-first initialization is documented as bitwise identical to
    ``Pipeline.setup()``; measured 0.0 on every float leaf, integer leaves/``randkey``/``generation``
    exact.
    """
    host = _host_state(POP_SIZE)
    device = _device_state(POP_SIZE, CPU_BACKEND, CPU_DEVICE)
    payload, species = _assert_initialization_parity(host, device, exact=True)
    assert payload.bitwise and species.bitwise, (
        f"the CPU plumbing path must reproduce Pipeline.setup() bitwise; measured payload max |diff| "
        f"{payload.float_max_abs:.3e}, species max |diff| {species.float_max_abs:.3e}"
    )


def test_cpu_policy_forward_parity():
    """Policy forward 45 -> 12 on ``cpu:0``: the plumbing path reproduces the reference bitwise."""
    host = _host_state(POP_SIZE)
    reference = _policy_actions(host, backend=CPU_BACKEND)
    # The plumbing device is the same numpy backend on cpu:0, so this is the same code path once
    # more; the accelerator tier's policy test is where the numeric check has teeth.
    actual = _policy_actions(host, backend=CPU_BACKEND, device=CPU_DEVICE)
    _assert_leaf_parity([reference], [actual], label="actions", exact=True)


def test_cpu_synthetic_returns_and_fitness_parity():
    """``SyntheticDeviceEnv`` returns/fitness on ``cpu:0``: bitwise equal across residencies.

    Both envs run the same compiled graph with the same payload and the same seed row, so the
    episode returns, their policy-major flattening (`returns`) and the fitness rule agree exactly.
    """
    host = _host_state(POP_SIZE)
    device = _device_state(POP_SIZE, CPU_BACKEND, CPU_DEVICE)

    reference = _evaluate_generation(host["pop"], "numpy", backend=CPU_BACKEND, device=CPU_DEVICE)
    actual = _evaluate_generation(device["pop"], "device", backend=CPU_BACKEND, device=CPU_DEVICE)
    for key in ("episode_returns", "returns", "fitness"):
        _assert_leaf_parity([reference[key]], [actual[key]], label=key, exact=True)
    assert np.array_equal(np.asarray(reference["seeds"]), _as_host(actual["seeds"])), (
        f"seeds {np.asarray(reference['seeds']).tolist()} (numpy) != "
        f"{_as_host(actual['seeds']).tolist()} (device); the seed triple is integer metadata"
    )
    assert np.asarray(reference["episode_returns"]).shape == (POP_SIZE, EPISODES_PER_POLICY), (
        f"episode_returns shape {np.asarray(reference['episode_returns']).shape} != "
        f"{(POP_SIZE, EPISODES_PER_POLICY)}"
    )


def test_cpu_tell_parity():
    """One ``run_tell`` on ``cpu:0``: payload/species leaves, ``randkey`` and ``generation`` exact."""
    host = _host_state(POP_SIZE)
    device = _device_state(POP_SIZE, CPU_BACKEND, CPU_DEVICE)

    reference = _one_tell(host, backend=CPU_BACKEND, device=CPU_DEVICE, residency="numpy")
    actual = _one_tell(device, backend=CPU_BACKEND, device=CPU_DEVICE, residency="device")

    _assert_leaf_parity(reference["pop"], actual["pop"], label="next payload", exact=True)
    _assert_leaf_parity(
        species_leaves_of(reference["species"]),
        evoxc.tree_leaves(actual["species"]),
        label="next species",
        exact=True,
    )
    _assert_next_key_and_generation(reference, actual)


def test_cpu_two_generation_trainer_parity():
    """Two generations on ``cpu:0``: identical seed table, history and records (cache stripped)."""
    reference, reference_env = _trainer(POP_SIZE, "numpy", backend=CPU_BACKEND, device=CPU_DEVICE)
    device, device_env = _trainer(POP_SIZE, "device", backend=CPU_BACKEND, device=CPU_DEVICE)
    try:
        reference.run()
        device.run()
    finally:
        reference.close()
        reference_env.close()
        device.close()
        device_env.close()

    assert np.array_equal(reference.seed_table, device.seed_table), (
        f"seed table {device.seed_table.tolist()} (device) != {reference.seed_table.tolist()} (numpy); "
        "the episode stream comes from the master seed alone"
    )
    assert reference.fitness_history == device.fitness_history, (
        f"fitness history {device.fitness_history} (device) != {reference.fitness_history} (numpy): "
        "the numpy backend is the same code path on cpu:0, so the histories must be bitwise equal"
    )
    _assert_record_parity(device, reference, exact=True)
    _assert_cache_counters_differ(device, reference)
    assert device.placement["pop"] == ["cpu:0"] * 4, (
        f"device residency on cpu:0 must report cpu:0 for every payload leaf, got {device.placement['pop']}"
    )


# --- B. Accelerator tier (skipped without a device, slow) ----------------------------


@pytest.mark.slow
def test_device_initialization_parity():
    """``device_setup`` vs ``Pipeline(...).setup()`` on the accelerator: integers exact, floats 1e-5.

    Measured 4.768e-07 on ``iree@rocm:0``: the float leaves are a different float32 reduction order
    of the same initialization, which is why only the integer leaves and the RNG key/generation
    counter are promised exactly.
    """
    backend, device = _device_or_skip()
    host = _host_state(POP_SIZE)
    accelerated = _device_state(POP_SIZE, backend, device)
    _assert_initialization_parity(host, accelerated, exact=False)


@pytest.mark.slow
def test_device_policy_forward_parity():
    """Policy forward 45 -> 12, numpy graph vs accelerator graph, same payload and observation tile.

    ``run_policy`` stages the host payload and the observation tile onto the executable's device, so
    this compares the two compiled graphs (the same arithmetic, two reduction orders), not the
    staging path.  Measured 0.0 max abs, actions up to ~18.
    """
    backend, device = _device_or_skip()
    host = _host_state(POP_SIZE)
    reference = _policy_actions(host, backend=CPU_BACKEND)
    actual = _policy_actions(host, backend=backend, device=device)
    _assert_allclose(
        actual,
        reference,
        atol=DEVICE_FLOAT_ATOL,
        rtol=0.0,
        label="policy actions (numpy graph vs accelerator graph)",
    )


def _synthetic_env_pair(backend, device, pop_size=POP_SIZE, generation=1):
    """Run one synthetic generation in both residencies from the SAME payload and seed row.

    The host payload is placed on ``device`` for the device env, so the comparison isolates the
    graph arithmetic instead of mixing in the 4.8e-07 initialization drift.
    """
    host = _host_state(pop_size)
    reference = _evaluate_generation(
        host["pop"], "numpy", backend=CPU_BACKEND, device=CPU_DEVICE, generation=generation
    )
    device_leaves = tuple(_to_device(leaf, backend, device) for leaf in host["pop"])
    actual = _evaluate_generation(device_leaves, "device", backend=backend, device=device, generation=generation)
    return reference, actual


@pytest.mark.slow
def test_device_synthetic_rewards_parity():
    """Synthetic rewards across residencies: episode returns, their flattening and the fitness.

    ``episode_returns``/``returns`` measure 2.289e-05 max abs (2.510e-07 rel) at values of ~99, i.e.
    ~3 float32 ulp of the graph's 320-element reduction, and ``fitness`` 1.526e-05 (1.542e-07 rel);
    hence the ``atol = rtol = 1e-5`` convention instead of a bare absolute bound -- at this magnitude
    one ulp is already 7.6e-06, and ``FITNESS_CONTRACT_ATOL`` (1e-6) holds only on the host path.
    """
    backend, device = _device_or_skip()
    reference, actual = _synthetic_env_pair(backend, device)

    for key in ("episode_returns", "returns", "fitness"):
        _assert_allclose(
            actual[key],
            reference[key],
            atol=DEVICE_AGGREGATE_ATOL,
            rtol=DEVICE_AGGREGATE_RTOL,
            label=f"synthetic {key} (numpy env vs device env)",
        )
    assert np.asarray(reference["episode_returns"]).shape == (POP_SIZE, EPISODES_PER_POLICY), (
        f"episode_returns shape {np.asarray(reference['episode_returns']).shape} != "
        f"{(POP_SIZE, EPISODES_PER_POLICY)}"
    )
    assert np.asarray(reference["fitness"]).shape == (POP_SIZE,), (
        f"fitness shape {np.asarray(reference['fitness']).shape} != {(POP_SIZE,)}"
    )
    assert np.array_equal(np.asarray(reference["seeds"]), _as_host(actual["seeds"])), (
        f"seeds {np.asarray(reference['seeds']).tolist()} (numpy) != {_as_host(actual['seeds']).tolist()} "
        "(device): the seed triple is integer metadata and must be bit-exact"
    )


@pytest.mark.slow
def test_device_done_and_host_metadata_parity():
    """The host metadata both residencies report is identical, and both report the same seeds.

    ``SyntheticDeviceEnv`` does not terminate (``end_reasons`` is the constant ``{"timeout": rows}``,
    ``steps``/``env_steps`` the horizon), so this pins exactly that.  Termination-*mask* parity -- the
    roll/pitch ``done`` flags, the timeout and the never-re-arming ``alive`` mask -- is pilot-level.
    """
    backend, device = _device_or_skip()
    reference, actual = _synthetic_env_pair(backend, device)
    rows = POP_SIZE * EPISODES_PER_POLICY

    for key in ("steps", "env_steps", "end_reasons"):
        assert reference[key] == actual[key], (
            f"{key}: device residency reports {actual[key]!r}, numpy residency {reference[key]!r}; the "
            "logging values are plain host values and must not depend on the residency"
        )
    assert actual["end_reasons"] == {"timeout": rows}, (
        f"the simulator-free env has no termination, so its end_reasons must stay {{'timeout': {rows}}}, "
        f"got {actual['end_reasons']!r}"
    )
    assert (actual["steps"], actual["env_steps"]) == (SYNTHETIC_HORIZON, SYNTHETIC_HORIZON * rows), (
        f"steps/env_steps {actual['steps']}/{actual['env_steps']} != "
        f"{SYNTHETIC_HORIZON}/{SYNTHETIC_HORIZON * rows} (the env's horizon and horizon x rows)"
    )
    assert np.array_equal(np.asarray(reference["seeds"]), _as_host(actual["seeds"])), (
        f"seeds differ: {np.asarray(reference['seeds']).tolist()} (numpy) vs "
        f"{_as_host(actual['seeds']).tolist()} (device)"
    )


@pytest.mark.slow
def test_device_two_generation_trainer_parity():
    """Two generations, numpy residency vs device residency at master seed 0.

    The seed table is bit-exact; ``fitness_history`` matches to ``atol = rtol = 1e-5`` (measured
    7.629e-06 abs / 7.695e-08 rel); the records match after stripping the wall-clock fields AND the
    cache counters (the device trainer compiles two extra graphs): float metrics to ``atol = rtol =
    1e-5``, every other field identically.
    """
    device_backend, device_device = _device_or_skip()
    reference, reference_env = _trainer(POP_SIZE, "numpy", backend=CPU_BACKEND, device=CPU_DEVICE)
    accelerated, accelerated_env = _trainer(POP_SIZE, "device", backend=device_backend, device=device_device)
    try:
        reference.run()
        accelerated.run()
    finally:
        reference.close()
        reference_env.close()
        accelerated.close()
        accelerated_env.close()

    assert np.array_equal(reference.seed_table, accelerated.seed_table), (
        f"seed table {accelerated.seed_table.tolist()} (device) != {reference.seed_table.tolist()} "
        "(numpy); the episode stream is drawn from the master seed, not from the residency"
    )
    device_history = np.asarray(accelerated.fitness_history, dtype=np.float64)
    numpy_history = np.asarray(reference.fitness_history, dtype=np.float64)
    history_abs, history_rel = _assert_allclose(
        device_history,
        numpy_history,
        atol=HISTORY_ATOL,
        rtol=HISTORY_RTOL,
        label="fitness_history (device vs numpy residency)",
    )
    assert len(accelerated.records) == len(reference.records) == GENERATIONS, (
        f"generation count {len(accelerated.records)} (device) / {len(reference.records)} (numpy) != "
        f"{GENERATIONS}"
    )

    _assert_record_parity(accelerated, reference, exact=False)
    _assert_cache_counters_differ(accelerated, reference)
    assert set(accelerated.placement["pop"]) == {f"{device_device.kind}:{device_device.index}"}, (
        f"the device trainer must report every payload leaf on {device_device.kind}:{device_device.index}, "
        f"got {accelerated.placement['pop']}"
    )
    assert accelerated.placement["fitness"] == f"{device_device.kind}:{device_device.index}", (
        f"the device fitness must stay on {device_device.kind}:{device_device.index}, "
        f"got {accelerated.placement['fitness']}"
    )


@pytest.mark.slow
def test_device_first_population_update_parity():
    """One ``run_tell`` from the same generation-0 state, numpy residency vs device residency.

    Integer leaves, ``randkey`` and ``generation`` are exact; the float leaves measure 2.384e-07 max
    abs (the initialization drift propagating through one update).  ``generation`` must advance by
    exactly +1 on both sides.
    """
    backend, device = _device_or_skip()
    host = _host_state(POP_SIZE)
    accelerated = _device_state(POP_SIZE, backend, device)

    reference = _one_tell(host, backend=CPU_BACKEND, device=CPU_DEVICE, residency="numpy")
    actual = _one_tell(accelerated, backend=backend, device=device, residency="device")

    _assert_leaf_parity(reference["pop"], actual["pop"], label="next generation payload")
    _assert_leaf_parity(
        species_leaves_of(reference["species"]),
        evoxc.tree_leaves(actual["species"]),
        label="next generation species",
    )
    _assert_next_key_and_generation(reference, actual)


# --- D. Pilot-level parity: not available yet ---------------------------------------

#: The two modules the Go2 pilot's ``residency="device"`` dispatch goes through.
PILOT_MODULES = (
    LOCOMOTION_DIR / "go2_device_rollout.py",
    LOCOMOTION_DIR / "go2_neat_env.py",
)


def _torch_imports(path):
    """Return the ``torch`` modules ``path`` imports, from an AST scan; empty when absent/unparsable.

    The file is parsed, never imported (importing it needs ``torch``, ``genesis`` and a GPU); a
    missing or unparsable file yields an empty list, so the detector cannot fail the suite itself.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return []
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(
                alias.name for alias in node.names if alias.name == "torch" or alias.name.startswith("torch.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "torch" or node.module.startswith("torch."):
                modules.append(node.module)
    return modules


def test_go2_torch_free_pilot_parity():
    """Placeholder: the Torch-free Go2 pilot does not exist yet, so pilot parity cannot be pinned.

    ``go2_neat_env.py``'s ``residency="device"`` dispatch routes through ``go2_device_rollout.py``,
    whose DLPack bridge imports ``torch`` (and ``genesis``); until that bridge is Torch-free there is
    no device-resident Go2 rollout to compare against the torch/numpy reference, so this test skips
    naming the offending modules instead of faking a comparison.  Once neither module imports torch,
    this placeholder fails and must be replaced by the pilot-level cases listed in the module
    docstring (observations, reward terms, roll/pitch termination mask, episode returns, fitness and
    the first population update).
    """
    offenders = {}
    for path in PILOT_MODULES:
        imports = _torch_imports(path)
        if imports:
            offenders[path.name] = sorted(set(imports))
    if offenders:
        pytest.skip(
            "Torch-free Go2 pilot not available yet: "
            + "; ".join(f"{name} imports {', '.join(imports)}" for name, imports in offenders.items())
        )
    pytest.fail(
        "the Go2 pilot no longer imports torch, so the pilot-level parity cases are now expected: "
        "replace this placeholder with observations, reward terms, roll/pitch termination mask, "
        "episode returns, fitness and first-population-update parity against the torch/numpy "
        "reference (see this module's docstring), using fixed seeds and the documented tolerances."
    )

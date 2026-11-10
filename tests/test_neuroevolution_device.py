"""Placement tests for the device-resident neuroevolution layer.

``examples/neuroevolution`` ships two residencies of the same loop: ``"numpy"`` keeps every
training value on the host, ``"device"`` keeps the four ``GenomeTensors`` payload leaves, the
``SpeciesState`` leaves, the RNG key and the generation counter on **one** ``evoxc`` device and
never stages them again.  These tests pin that placement contract on a real accelerator; the
numeric agreement of the same flow belongs to the residency-parity module, so nothing here
compares numbers across residencies.

What is checked, in order:

1. **Initialization** -- ``NeuroevolutionTrainer.setup()`` at pop 8 leaves the four payload
   leaves, every species leaf, ``randkey`` and ``generation_counter`` on the trainer's device
   (``assert_payload_resident`` plus ``placement_map`` evidence), and ``trainer.placement``
   names that device for ``pop``/``species``/``randkey``/``generation`` and never ``"host"``.
2. **Policy forward output** -- the shared width-generic policy graph (45 -> 12) run with the
   device payload and a device-resident observation tile returns a device tensor of shape
   ``(pop * 3, 12)`` float32; the same graph through ``run_policy`` on the host numpy residency
   returns a host ``numpy.ndarray``, so the placement assertion is not vacuous.
3. **Rewards / returns / fitness** -- ``env.evaluate`` under device residency returns
   ``seeds``/``episode_returns``/``returns``/``fitness`` as device tensors with the documented
   dtypes and shapes (``(3,)``, ``(pop, 3)``, ``(pop * 3,)``, ``(pop,)``), accepted by
   ``env_base.validate_evaluation_device``; that validator rejects a numpy payload.
4. **``tell`` outputs** -- ``run_tell(..., residency="device")`` returns ``randkey``,
   ``generation``, the four payload leaves and every species leaf on the device,
   ``generation`` advanced by exactly 1, and ``assert_payload_resident`` passes on the
   returned generation.
5. **Population entering the next generation** -- a two-generation run at pop 8: the state
   entering generation 1 (i.e. after generation 0) and the final state after generation 1 both
   keep the payload, the species leaves, the key and the counter on the device, and
   ``trainer.placement`` reports the device for all eight runtime keys.
6. **Negative contract** -- a host array is not device-resident, a device tensor is not a host
   value, and a tensor on another device is rejected.  Cheap and accelerator-free.
7. **GPU-free plumbing mode** -- the same placement plumbing on ``numpy@cpu:0`` (device
   tensors on ``cpu:0``, not host arrays), so the contract is still exercised on a host
   without an accelerator rather than skipped.
8. **A larger population** -- pop 32 and pop 64, one generation each, ``@pytest.mark.slow``:
   the same invariants and a measured wall clock, skipped cleanly without a device.

The accelerator tests resolve their target exactly like ``verify_device.py`` does: backend
``iree`` (the default; ``EVOXC_BACKEND`` overrides) on the first non-CPU ``evoxc`` device the
backend can actually reach (``EVOXC_DEVICE`` overrides), skipping with a reason otherwise.
The measured host is an AMD Instinct MI300X VF (gfx942, ROCm 7.1), where ``evoxc.devices()``
is ``cpu:0``/``cuda:0``/``rocm:0`` and the run is ``iree@rocm:0``.  Measured runtimes are
recorded at the bottom of this docstring.

Run it from the repository root::

    PYTHONPATH=$PWD:/tmp/gs-pytest-libs /root/evox-robotics/.venv/bin/python -m pytest \\
        tests/test_neuroevolution_device.py -q -n 0

``evoxc`` and ``tensorneat`` are not declared dependencies of ``genesis-world``, so the module
skips where they are absent; ``pytestmark`` parametrizes ``backend`` to ``None`` so the autouse
Genesis fixture never initializes a simulator.

Device-first initialization compiles TensorNEAT's own graphs on the device and ``build_tell``
passes the ROCm ``--iree-codegen-llvmgpu-use-reduction-vector-distribution=false`` flag itself
in device residency, so no test here needs ``EVOXC_IREE_COMPILE_ARGS``: every graph of this
module built and ran without it on the measured host.

Measured runtimes (MI300X VF, ``iree@rocm:0``, ``-n 0``): ``9 passed in 40.3 s``.  The module
shares one ``ExecutableCache`` across its trainers, so only the first device test pays the
cold compiles (9.20 s initialization; the tell graph is 7.6 s of it); the ones after it reuse
them -- policy forward 3.13 s (its own 45 -> 12 graph), evaluation contract 1.00 s, tell
1.08 s, two generations 1.24 s, ``cpu:0`` plumbing 0.29 s, the accelerator-free predicates
< 0.01 s -- while the slow sizes pay their own compile: pop 32 9.20 s and pop 64 9.38 s
(setup 9.03/9.08 s plus one generation of 0.173/0.292 s for 96/192 rows).  With
``EVOXC_DEVICE=cpu:0`` the module reports ``2 passed, 7 skipped`` (the negative predicates and
the plumbing mode still run) instead of failing.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("evoxc")
pytest.importorskip("tensorneat")

pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: ``examples/`` is a set of plain scripts rather than a package, so the shared neuroevolution
#: layer imports its siblings by bare module name; put the directory on ``sys.path`` first.
EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
NEUROEVOLUTION_DIR = EXAMPLES_DIR / "neuroevolution"
if str(NEUROEVOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(NEUROEVOLUTION_DIR))

import evoxc
from device_env import SyntheticDeviceEnv
from device_runtime import (
    assert_device_resident,
    assert_host_value,
    assert_payload_resident,
    device_placement,
    is_device_tensor,
    placement_map,
    same_device,
)
from env_base import DEVICE_KEYS, validate_evaluation_device
from evolution_graphs import ExecutableCache, run_tell
from evox_runtime import POP_OPERANDS, _device_name, _outputs, _to_numpy, resolve_backend_device
from policy_graphs import build_policy, run_policy
from seeding import EPISODES_PER_POLICY
from trainer import NeuroevolutionTrainer
from verify_device import DEFAULT_BACKEND, ensure_pjrt_plugin, reachable_devices

#: Policies per generation of the placement checks: the Go2 pilot's population.
POP_SIZE = 8
#: Episodes per policy, the documented triple every generation shares.
EPISODES = EPISODES_PER_POLICY
#: Rows one generation simulates, policy-major.
ROWS = POP_SIZE * EPISODES
#: Genome widths of the widest task the layer serves (Go2 walking), the real policy graph shape.
NUM_INPUTS, NUM_OUTPUTS = 45, 12
#: Generations of the "population entering the next generation" check.
GENERATIONS = 2
#: The master seed every placement run uses (placement is seed-independent, the value is not).
MASTER_SEED = 0
#: Population of the GPU-free plumbing check, the same size the self-check's cpu mode uses.
PLUMBING_POP_SIZE = 3
#: Population sizes of the slow, larger-population check.
LARGE_POP_SIZES = (32, 64)
#: The four metric keys a device ``evaluate`` result must keep on the device.
RESULT_KEYS = ("fitness", "episode_returns", "returns", "seeds")
#: Every key ``trainer.placement`` reports once a generation has run.
ALL_PLACEMENT_KEYS = ("pop", "species", "randkey", "generation", *RESULT_KEYS)


# --- Helpers ------------------------------------------------------------------------


def _device_or_skip():
    """Return ``(backend, device, plugin_path)`` of the first reachable non-CPU device, or skip.

    Mirrors ``verify_device.resolve_target`` without its CLI: the backend is ``iree`` unless
    ``EVOXC_BACKEND`` names another one, the device is ``EVOXC_DEVICE`` when set and otherwise
    the first non-CPU device the backend can actually place a tensor on.  ``evoxc.devices()``
    lists every runtime the wheel knows, so reachability is probed by a real one-element
    transfer (``verify_device.reachable_devices``), which is what makes the skip honest.
    """
    backend = os.environ.get("EVOXC_BACKEND") or DEFAULT_BACKEND
    plugin_path = ensure_pjrt_plugin(backend)
    if backend == "xla" and plugin_path is None:
        pytest.skip(
            "the xla adapter needs a discoverable PJRT plugin for device-first initialization; "
            "set EVOXC_PJRT_PLUGIN to the xla_<kind>_plugin.so"
        )
    devices = evoxc.devices()
    available = ", ".join(_device_name(entry) for entry in devices)
    usable, skipped = reachable_devices(devices, backend)
    device_spec = (os.environ.get("EVOXC_DEVICE") or "").strip()
    if device_spec:
        device = resolve_backend_device(backend, device_spec)[1]
    elif usable:
        device = usable[0]
    else:
        reasons = "; ".join(f"{entry['device']} unreachable ({entry['error']})" for entry in skipped)
        pytest.skip(
            f"no non-CPU evoxc device is reachable through {backend} (available: {available}"
            f"{'; ' + reasons if reasons else ''})"
        )
    if device.kind == "cpu":
        pytest.skip(
            f"EVOXC_DEVICE={device_spec} names a cpu-kind device; "
            "test_device_placement_on_a_cpu_only_host covers the plumbing-only mode"
        )
    resolved_backend, resolved_device = resolve_backend_device(backend, device)
    # The trainer forwards the plugin into device-first initialization on xla only; a path that
    # is not a plugin for this backend is never read, so it is handed over unconditionally.
    return resolved_backend, resolved_device, plugin_path


@pytest.fixture(scope="module")
def target():
    """``(backend, device, plugin_path)`` of the accelerator every hardware test runs on."""
    return _device_or_skip()


@pytest.fixture(scope="module")
def executable_cache():
    """One executable cache per module, so identical graphs compile once instead of per test.

    The cache key already carries the population size, the backend and the device, so a hit can
    only reuse a graph compiled for exactly the same contract, and the module hands it to every
    env it builds; each env then owns only its own lifecycle (``SyntheticDeviceEnv.close``
    leaves a caller-owned cache alone).
    """
    cache = ExecutableCache()
    yield cache
    cache.clear()


class _PlacementProbe:
    """Delegating env wrapper that snapshots ``trainer.placement`` as each generation starts.

    The trainer refreshes ``placement`` at the end of a generation, so the state *entering*
    generation ``g`` is the state after generation ``g - 1``: that is how "after each
    generation" is observed without reaching into the trainer's private loop.  ``env`` and
    ``trainer`` are assigned by :func:`_trainer`, which owns the construction order.
    """

    def __init__(self, env=None):
        self.env = env
        self.trainer = None
        self.entries = []

    def __getattr__(self, name):
        """Delegate everything the trainer reads (``cache``, ``residency``, ``device``, …)."""
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def evaluate(self, pop_arrays, generation):
        """Record the runtime placement entering ``generation``, then run the wrapped env."""
        if self.trainer is not None:
            self.entries.append(
                {
                    "generation": int(generation),
                    "counter": float(_to_numpy(self.trainer.generation_counter)),
                    "payload": [device_placement(leaf) for leaf in self.trainer.pop_arrays],
                    "species": [device_placement(leaf) for leaf in self.trainer.species_leaves],
                    "randkey": device_placement(self.trainer.randkey),
                    "placement": {
                        key: list(value) if isinstance(value, list) else value
                        for key, value in self.trainer.placement.items()
                    },
                }
            )
        return self.env.evaluate(pop_arrays, generation)

    def close(self):
        """Close the wrapped env."""
        self.env.close()


@contextlib.contextmanager
def _trainer(pop_size, target, *, cache, generations=GENERATIONS, probe=None):
    """Build a device-resident trainer for ``pop_size`` policies and yield ``(env, trainer)``.

    The env is the simulator-free ``SyntheticDeviceEnv`` (a cpu-kind device is its plumbing-only
    mode), the trainer inherits the device residency from it, and both are released on the way
    out -- the trainer never closes the env it was handed, so the test does.

    Args:
        pop_size (int): Policies per generation.
        target (tuple): ``(backend, device, plugin_path)`` of the run, as :func:`_device_or_skip`
            returns it; ``plugin_path`` is forwarded to device-first initialization.
        cache (ExecutableCache): The module's shared executable cache.
        generations (int): Generations the trainer is configured for.
        probe (_PlacementProbe | None): When given, it wraps the env and records placement.
    """
    backend, device, plugin_path = target
    env = SyntheticDeviceEnv(
        pop_size,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        episodes=EPISODES,
        backend=backend,
        device=device,
        residency="device",
        seed=MASTER_SEED,
        cache=cache,
    )
    wrapped = env
    if probe is not None:
        probe.env = env
        wrapped = probe
    trainer = NeuroevolutionTrainer(
        wrapped,
        pop_size,
        generations,
        master_seed=MASTER_SEED,
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        residency="device",
        plugin_path=plugin_path,
    )
    if probe is not None:
        probe.trainer = trainer
    try:
        trainer.setup()
        yield env, trainer
    finally:
        trainer.close()
        env.close()


def _device_tensor(array, backend, device):
    """Stage one host array as a tensor on ``device``, selecting its serving backend first.

    The transfer provider behind ``Tensor.to`` belongs to the backend that was looked up last,
    so a mixed numpy/iree process must re-select the backend that serves the target device.
    """
    evoxc.backends.get(backend)
    return evoxc.from_numpy(np.ascontiguousarray(np.asarray(array))).to(device)


def _run_policy_device(executable, pop_arrays, obs, backend, device):
    """Run the policy graph with every operand already device-resident.

    ``policy_graphs.run_policy`` is the host-staging entry point; the device call is the very
    same compiled graph through ``evoxc.run``, with the four payload leaves and a
    ``(rows, num_inputs)`` float32 observation tile as the only operands.

    Args:
        executable: The graph from :func:`policy_graphs.build_policy`.
        pop_arrays: The four genome payload leaves, device tensors on ``device``.
        obs (numpy.ndarray): The ``(rows, num_inputs)`` observation tile, staged here.
        backend (str): The ``evoxc`` backend the executable was built with.
        device (evoxc.Device): The executable's device.

    Returns:
        evoxc.Tensor: The raw ``(rows, num_outputs)`` float32 actions, never materialized.
    """
    observation = _device_tensor(obs, backend, device)
    return _outputs(evoxc.run(executable, *pop_arrays, observation))[0]


def _assert_placement(placement, device_name, keys=ALL_PLACEMENT_KEYS):
    """Assert ``placement`` names ``device_name`` for every ``key``, one leaf or a list of them."""
    for key in keys:
        assert key in placement, f"trainer.placement is missing {key!r}: {placement}"
        value = placement[key]
        values = list(value) if isinstance(value, list) else [value]
        assert values, f"{key} must report at least one runtime leaf, got {value!r}"
        assert set(values) == {device_name}, f"{key} must live on {device_name}, got {value!r}"


def _assert_runtime_placement(trainer, device, *, result=False):
    """Assert every runtime leaf of ``trainer`` is resident on ``device``; return the evidence.

    ``assert_payload_resident`` is the metadata-only check of the four payload leaves, the
    species leaves, the key and the generation counter; ``placement_map`` is the per-leaf
    evidence this helper returns.  ``trainer.placement`` must agree, and with ``result=True``
    it must also name the device for the four metrics of the last generation.
    """
    device_name = _device_name(device)
    assert trainer.residency == "device"
    assert_payload_resident(
        trainer.pop_arrays,
        trainer.species_leaves,
        trainer.randkey,
        trainer.generation_counter,
        device,
    )
    evidence = placement_map(
        {
            **{operand: leaf for operand, leaf in zip(POP_OPERANDS, trainer.pop_arrays, strict=True)},
            **{f"species_{index}": leaf for index, leaf in enumerate(trainer.species_leaves)},
            "randkey": trainer.randkey,
            "generation": trainer.generation_counter,
        }
    )
    assert set(evidence.values()) == {device_name}, evidence
    _assert_placement(trainer.placement, device_name, ("pop", "species", "randkey", "generation"))
    if result:
        _assert_placement(trainer.placement, device_name, RESULT_KEYS)
    else:
        # No generation has run yet, so the result metrics cannot be device evidence.
        _assert_placement(trainer.placement, "host", RESULT_KEYS)
    return evidence


# --- 1. Initialization ---------------------------------------------------------------


def test_initialization_is_device_resident(target, executable_cache):
    """``setup()`` builds generation 0 on the device, so no leaf ever crosses the host."""
    backend, device, _ = target
    device_name = _device_name(device)
    with _trainer(POP_SIZE, target, cache=executable_cache) as (_env, trainer):
        evidence = _assert_runtime_placement(trainer, device)
        assert "host" not in evidence.values()
        # The evolution graphs run on the env's own backend and device (single-backend residency).
        assert (trainer.tell_backend, trainer.tell_device_name) == (backend, device_name)
        # The payload is the four GenomeTensors leaves, each resident on its own.
        assert len(trainer.pop_arrays) == len(POP_OPERANDS) == 4
        assert all(is_device_tensor(leaf) for leaf in trainer.pop_arrays)
        assert trainer.species_leaves and all(is_device_tensor(leaf) for leaf in trainer.species_leaves)
        # The counter starts at 0 on the device, so generation 0 is the state that was built.
        assert tuple(trainer.generation_counter.shape) == ()
        assert np.dtype(trainer.generation_counter.dtype) == np.float32
        assert float(_to_numpy(trainer.generation_counter)) == 0.0


# --- 2. Policy forward outputs --------------------------------------------------------


def test_policy_forward_outputs_stay_on_the_device(target, executable_cache):
    """The 45 -> 12 policy graph returns a device tensor, and a host ndarray on the host path."""
    backend, device, _ = target
    device_name = _device_name(device)
    with _trainer(POP_SIZE, target, cache=executable_cache) as (_env, trainer):
        payload = tuple(trainer.pop_arrays)
        # One observation tile for every row; each row carries its own observation by contract.
        obs = np.tile(np.linspace(-1.0, 1.0, NUM_INPUTS, dtype=np.float32), (ROWS, 1))
        executable = build_policy(
            trainer.algo,
            trainer.config,
            payload,
            ROWS,
            num_inputs=NUM_INPUTS,
            num_outputs=NUM_OUTPUTS,
            episodes=EPISODES,
            backend=backend,
            device=device,
        )
        actions = _run_policy_device(executable, payload, obs, backend, device)
        assert is_device_tensor(actions)
        assert device_placement(actions) == device_name
        assert tuple(actions.shape) == (ROWS, NUM_OUTPUTS)
        assert np.dtype(actions.dtype) == np.float32
        # The same graph on the host residency stays host: the placement assertion is not vacuous.
        host_payload = tuple(_to_numpy(leaf) for leaf in payload)
        host_executable = build_policy(
            trainer.algo,
            trainer.config,
            host_payload,
            ROWS,
            num_inputs=NUM_INPUTS,
            num_outputs=NUM_OUTPUTS,
            episodes=EPISODES,
            backend="numpy",
            device="cpu:0",
        )
        host_actions = run_policy(host_executable, host_payload, obs)
        assert isinstance(host_actions, np.ndarray)
        assert not is_device_tensor(host_actions)
        assert device_placement(host_actions) == "host"
        assert host_actions.shape == (ROWS, NUM_OUTPUTS)
        assert host_actions.dtype == np.float32


# --- 3. Rewards / returns / fitness ---------------------------------------------------


def test_evaluation_metrics_are_device_resident(target, executable_cache):
    """``env.evaluate`` returns the four metrics on the device with the contract's dtypes."""
    _, device, _ = target
    device_name = _device_name(device)
    with _trainer(POP_SIZE, target, cache=executable_cache) as (env, trainer):
        result = env.evaluate(trainer.pop_arrays, 0)
        for key in DEVICE_KEYS:
            value = result[key]
            assert is_device_tensor(value), f"{key} must be a device tensor, got {type(value).__name__}"
            assert device_placement(value) == device_name, f"{key} is on {device_placement(value)}"
        assert tuple(result["seeds"].shape) == (EPISODES,)
        assert np.issubdtype(np.dtype(result["seeds"].dtype), np.integer)
        assert tuple(result["episode_returns"].shape) == (POP_SIZE, EPISODES)
        assert np.dtype(result["episode_returns"].dtype) == np.float32
        assert tuple(result["returns"].shape) == (ROWS,)
        assert np.dtype(result["returns"].dtype) == np.float32
        assert tuple(result["fitness"].shape) == (POP_SIZE,)
        assert np.dtype(result["fitness"].dtype) == np.float32
        # Logging values stay host values: they are never training data.
        for key in ("steps", "env_steps", "wall_time_s"):
            assert not is_device_tensor(result[key]), f"{key} must stay a host value"
        # The metadata-only device validator accepts the result ...
        assert validate_evaluation_device(result, POP_SIZE, EPISODES, device) is None
        # ... and rejects the same mapping once its four metrics are host arrays.
        host_result = {key: _to_numpy(result[key]) for key in DEVICE_KEYS}
        host_result.update({key: value for key, value in result.items() if key not in DEVICE_KEYS})
        with pytest.raises(ValueError):
            validate_evaluation_device(host_result, POP_SIZE, EPISODES, device)


# --- 4. tell() outputs ----------------------------------------------------------------


def test_tell_outputs_stay_on_the_device(target, executable_cache):
    """One ``tell`` transition keeps the key, the counter, the payload and the species resident."""
    _, device, _ = target
    device_name = _device_name(device)
    with _trainer(POP_SIZE, target, cache=executable_cache) as (env, trainer):
        result = env.evaluate(trainer.pop_arrays, 0)
        before = float(_to_numpy(trainer.generation_counter))
        # The trainer's own cached tell executable, so this is the graph a generation runs.
        tell = run_tell(
            trainer._tell_executable(),
            randkey=trainer.randkey,
            pop_arrays=trainer.pop_arrays,
            generation=trainer.generation_counter,
            fitness=result["fitness"],
            species_leaves=trainer.species_leaves,
            residency="device",
        )
        assert set(tell) == {"randkey", "generation", "pop", "species"}
        species_leaves = tuple(evoxc.tree_leaves(tell["species"]))
        # The whole returned generation is device-resident, checked by metadata only.
        assert_payload_resident(tell["pop"], species_leaves, tell["randkey"], tell["generation"], device)
        evidence = placement_map(
            {
                **{operand: leaf for operand, leaf in zip(POP_OPERANDS, tell["pop"], strict=True)},
                **{f"species_{index}": leaf for index, leaf in enumerate(species_leaves)},
                "randkey": tell["randkey"],
                "generation": tell["generation"],
            }
        )
        assert set(evidence.values()) == {device_name}, evidence
        # The key keeps its documented form, and the counter advanced by exactly one generation.
        assert tuple(tell["randkey"].shape) == ()
        assert np.issubdtype(np.dtype(tell["randkey"].dtype), np.integer)
        assert tuple(tell["generation"].shape) == ()
        assert np.dtype(tell["generation"].dtype) == np.float32
        assert float(_to_numpy(tell["generation"])) == before + 1.0
        assert len(tuple(tell["pop"])) == len(POP_OPERANDS)
        assert len(species_leaves) == len(trainer.species_leaves)


# --- 5. The population entering the next generation -----------------------------------


def test_two_generations_keep_the_population_on_the_device(target, executable_cache):
    """After every generation the payload, species, key and counter are still on the device."""
    _, device, _ = target
    device_name = _device_name(device)
    probe = _PlacementProbe()
    with _trainer(POP_SIZE, target, cache=executable_cache, probe=probe) as (_env, trainer):
        # Generation 0 is the device-first initialization.
        _assert_runtime_placement(trainer, device)
        trainer.run()
        # One snapshot per generation, at its start: entry g is the state after generation g - 1.
        assert [entry["generation"] for entry in probe.entries] == list(range(GENERATIONS))
        assert [entry["counter"] for entry in probe.entries] == [float(index) for index in range(GENERATIONS)]
        after_first = probe.entries[1]
        assert set(after_first["payload"]) == {device_name}
        assert set(after_first["species"]) == {device_name}
        assert after_first["randkey"] == device_name
        # Everything is device evidence once a generation has run: all eight placement keys.
        _assert_placement(after_first["placement"], device_name)
        # The final state, after the last generation, is the one a third generation would enter.
        evidence = _assert_runtime_placement(trainer, device, result=True)
        assert "host" not in evidence.values()
        assert float(_to_numpy(trainer.generation_counter)) == float(GENERATIONS)
        assert len(trainer.records) == GENERATIONS
        assert len(trainer.fitness_history) == GENERATIONS


# --- 6. Negative contract (accelerator-free) ------------------------------------------


def test_placement_predicates_reject_host_values():
    """A host array and a device tensor are never interchangeable, and a wrong device is rejected."""
    device = evoxc.Device("cpu", 0)
    host = np.zeros((2, 2), dtype=np.float32)
    device_tensor = _device_tensor(host, "numpy", device)
    assert device_placement(host) == "host"
    assert not is_device_tensor(host)
    assert not same_device(host, device)
    with pytest.raises(TypeError):
        assert_device_resident(host, device, "payload")
    # The mirror direction: a device tensor must not pass as a host value ...
    assert device_placement(device_tensor) == "cpu:0"
    assert same_device(device_tensor, device)
    assert assert_device_resident(device_tensor, device, "payload") is device_tensor
    with pytest.raises(TypeError):
        assert_host_value(device_tensor, "payload")
    # ... and a device tensor placed elsewhere is not accepted for this device.
    other = evoxc.Device("cpu", 1)
    assert not same_device(device_tensor, other)
    with pytest.raises(TypeError):
        assert_device_resident(device_tensor, other, "payload")


# --- 7. GPU-free plumbing mode --------------------------------------------------------


def test_device_placement_on_a_cpu_only_host(executable_cache):
    """The same plumbing holds on ``numpy@cpu:0``: device tensors, not host arrays."""
    backend, device = resolve_backend_device("numpy", "cpu:0")
    plumbing_target = (backend, device, None)
    with _trainer(PLUMBING_POP_SIZE, plumbing_target, cache=executable_cache, generations=1) as (env, trainer):
        evidence = _assert_runtime_placement(trainer, device)
        assert set(evidence.values()) == {"cpu:0"}
        # A cpu-kind device is still a device: the leaves are tensors, never host arrays.
        for leaf in (*trainer.pop_arrays, *trainer.species_leaves, trainer.randkey, trainer.generation_counter):
            assert not isinstance(leaf, np.ndarray)
            assert is_device_tensor(leaf)
        result = env.evaluate(trainer.pop_arrays, 0)
        for key in DEVICE_KEYS:
            assert is_device_tensor(result[key])
            assert device_placement(result[key]) == "cpu:0"
        assert validate_evaluation_device(result, PLUMBING_POP_SIZE, EPISODES, device) is None
        trainer.run()
        _assert_runtime_placement(trainer, device, result=True)
        assert float(_to_numpy(trainer.generation_counter)) == 1.0


# --- 8. A larger population -----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("pop_size", LARGE_POP_SIZES)
def test_large_population_stays_device_resident(pop_size, target, executable_cache):
    """A 32- and 64-policy generation keeps the same placement invariants; wall time is reported."""
    backend, device, _ = target
    device_name = _device_name(device)
    setup_start = time.perf_counter()
    with _trainer(pop_size, target, cache=executable_cache, generations=1) as (_env, trainer):
        setup_s = time.perf_counter() - setup_start
        _assert_runtime_placement(trainer, device)
        run_start = time.perf_counter()
        trainer.run()
        run_s = time.perf_counter() - run_start
        evidence = _assert_runtime_placement(trainer, device, result=True)
        assert "host" not in evidence.values()
        assert float(_to_numpy(trainer.generation_counter)) == 1.0
        record = trainer.records[0]
        assert len(record["episode_returns"]) == pop_size
        assert len(record["fitness"]) == pop_size
        print(
            f"\npop {pop_size} on {device_name} ({backend}): setup {setup_s:.2f} s, "
            f"1 generation {run_s:.3f} s, {pop_size * EPISODES} rows, placement {device_name}"
        )

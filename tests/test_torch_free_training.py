"""Torch-free training tests: Genesis must import, initialize, step and be driven without PyTorch.

Genesis is being made usable where PyTorch is not installed, so that a training stack can run on Quadrants and
NumPy alone.  Proving that requires the *absence* of PyTorch, which the parent pytest process cannot provide (its
conftest imports torch).  Every scenario therefore runs in a child interpreter built by ``tests/torch_block.py``,
whose meta-path blocker refuses ``torch`` and every ``torch.*`` name before anything can import it:

* the blocker itself works, and importing Genesis under it leaves ``HAS_TORCH`` False and ``torch`` out of
  ``sys.modules``;
* a real GPU scene (plane + box, 2 environments, 10 steps) initializes, steps and returns device-resident state
  through the torch-free ``DeviceView`` bridge, with no torch in the child -- on whichever GPU the host exposes,
  since ``gs.gpu`` is the vendor-agnostic enum member that resolves to the AMD, CUDA or Metal backend;
* the static half: nothing on the torch-free training path imports PyTorch, which now includes the pilots' device
  residency modules (``go2_device_*.py``, ``hover_device_*.py``), except the sanctioned ``genesis/`` files;
* the state boundary agrees bit for bit between a torch-free and a torch-present child running the very same scene,
  including one control write with a host NumPy array, which is what a training loop does every step;
* the Go2 NEAT walking pilot itself trains, in a torch-free child, two generations of eight policies to the numbers
  the locomotive stack documents (fitness history, steps, ``tilted`` endings, seed triples) and leaves every training
  leaf -- the population, the species, the RNG key, the generation counter, the fitness and the returns -- on the
  same non-CPU device, with only the logging values on the host;
* the pilot's real CLI (``examples/locomotion/go2_neat_train.py --torch-free``), not a reimplementation of it, runs to
  completion as a script while a ``torch`` package that refuses to import sits first on ``PYTHONPATH``, and reports
  ``torch_in_process=False`` both on stdout and in its ``summary.json``.

The GPU scenarios are guarded on the machine rather than on a marker (the suite has no ``gpu`` marker) and skip
cleanly when no GPU is visible.  Run the module from the repository root::

    PYTHONPATH=$PWD:/tmp/gs-pytest-libs /root/evox-robotics/.venv/bin/python -m pytest \
        tests/test_torch_free_training.py -q -n 0

The four GPU tests are ``slow`` because each child pays its own first Quadrants compile of the scene plus the IREE
compiles of the training graphs, which takes about a minute; the other two are CPU-only and take a few seconds.

Note that ``--torch-free`` is *not* runnable on a host that has PyTorch installed unless PyTorch is blocked:
``gs.init(backend=gs.gpu)`` imports the installed ``torch`` itself, and the flag then refuses to continue (that is its
whole point).  The CLI scenario therefore blocks it the way the locomotion documentation documents the probe, with a
``torch/__init__.py`` that raises on import, instead of relying on the suite's meta-path blocker, which only reaches
children spawned through ``tests/torch_block.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tests.device_pilot import non_cpu_device_or_skip
from tests.torch_block import (
    EXAMPLES_DIR,
    GENESIS_DIR,
    REPO_ROOT,
    assert_no_torch_imports,
    has_gpu,
    run_python_checked,
)

pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: Canonical skip reason of the GPU scenarios, resolved by the session's skip-reason registry
SKIP_NO_GPU = "No usable GPU detected; the torch-free Genesis run needs one."

#: Skip reason of the pilot scenarios when the accelerated compiler stack they need is not installed here
SKIP_NO_EVOXC = "Neither 'evoxc' nor 'tensorneat' is installed; the torch-free Go2 pilot needs both."

#: The torch-free training path: every shared neuroevolution module, the Go2 and drone pilots -- the numpy-facing CLI
#: and env stack plus the zero-copy device residency (whose globs tolerate a file a parallel change has not written
#: yet) -- and the whole Genesis package.
#: 'examples/neuroevolution/*.py' is deliberately single-level: the 'benchmarks/' subtree measures PyTorch itself
#: (import and residency probes) and is not part of the training path.
#: The Go2 *reference* side is deliberately outside the scan rather than allowlisted.  'go2_env.py' imports torch at
#: module scope -- it is the behaviour the pilots reproduce -- and 'go2_legacy_reference.py' is the torch oracle the
#: Go2 parity tests compare the compiled graphs against; both are reached through the compiled-graph layer of
#: 'go2_evox_kernels.py', which is torch-free but is an oracle's own dependency rather than a pilot module, so no
#: pilot glob matches it either.  Torch is those files' subject, not an accident: they are never part of the
#: contracted surface, and the scan itself is never weakened to accommodate them.
TRAINING_PATH = (
    EXAMPLES_DIR / "neuroevolution" / "*.py",
    EXAMPLES_DIR / "locomotion" / "go2_neat_*.py",
    EXAMPLES_DIR / "locomotion" / "go2_device_*.py",
    EXAMPLES_DIR / "drone" / "hover_neat_*.py",
    EXAMPLES_DIR / "drone" / "hover_device_*.py",
    GENESIS_DIR,
)

#: The Go2 torch-free pilot's own modules, which the scan must really reach: an empty or mistyped glob has to fail
#: instead of passing with a reassuring empty violation list.  The guard pins the Go2 files because this module's
#: assertions are about that pilot; the drone pilot's device modules are reached by the 'hover_device_*.py' and
#: 'hover_neat_*.py' globs above, and 'test_hover_device_parity.py' names the four files of that pilot itself.
GO2_PILOT_FILES = (
    EXAMPLES_DIR / "locomotion" / "go2_device_kernels.py",
    EXAMPLES_DIR / "locomotion" / "go2_device_rollout.py",
    EXAMPLES_DIR / "locomotion" / "go2_neat_env.py",
    EXAMPLES_DIR / "locomotion" / "go2_neat_train.py",
)

#: The only files allowed to import PyTorch although the scan reaches them, with the reason for each exemption.
TORCH_IMPORT_ALLOWLIST = (
    # The single sanctioned import site of the optional-PyTorch compatibility layer: it re-exports the real module,
    # or a module-like placeholder when it is missing, and every other module imports torch from here.
    GENESIS_DIR / "utils" / "torch_compat.py",
    # The differentiable-simulation bridge, which imports 'torch.autograd' on use only: both call sites are below a
    # 'require_torch' guard, so nothing is imported until a gradient-enabled scene asks for it.
    GENESIS_DIR / "grad" / "bridge.py",
)

#: Vendored third-party code, which is not part of the torch-free contract.
TORCH_IMPORT_EXCLUDES = (GENESIS_DIR / "ext",)

#: Child script of 'test_torch_blocker_blocks_torch_in_a_child'.
_BLOCKER_SCRIPT = """
import sys

for statement in ("import torch", "import torch.nn", "from torch import Tensor"):
    try:
        exec(statement, {})
    except ImportError as exc:
        print(f"blocked: {statement} ({exc})")
    else:
        raise AssertionError(f"'{statement}' succeeded although PyTorch is blocked")

leaked = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch."))
assert not leaked, leaked

import genesis

from genesis.utils.torch_compat import HAS_TORCH

print(f"genesis {genesis.__version__} imported, HAS_TORCH={HAS_TORCH}")
assert HAS_TORCH is False, HAS_TORCH
"""

#: Child script of 'test_genesis_initializes_and_steps_without_torch': the smallest GPU scene that exercises the
#: torch-free device bridge end to end (build, step, read a device-resident getter).
_GPU_STEP_SCRIPT = """
import sys

import genesis as gs

gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)))
scene.build(n_envs=2)

for _ in range(10):
    scene.step()

pos = box.get_pos()
quat = box.get_quat()
print(f"type={type(pos).__name__} device={pos.device} pos={pos.to_numpy().tolist()}")
assert tuple(pos.shape) == (2, 3), pos.shape
assert tuple(quat.shape) == (2, 4), quat.shape
assert type(pos).__name__ == "DeviceView", type(pos).__name__
# Both the ROCm and the CUDA path expose a torch-free 'DeviceView' whose device type is "cuda"
assert pos.device.type == "cuda", pos.device
assert float(pos.to_numpy()[0, 2]) < 0.5, pos.to_numpy()
leaked = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch."))
assert not leaked, leaked
"""

#: Child script of 'test_torch_free_and_torch_present_genesis_agree_on_the_state_boundary', run twice: once with
#: PyTorch blocked and once with it installed. '__OUT_DIR__' and '__EXPECT_TORCH__' are substituted per run.
_STATE_BOUNDARY_SCRIPT = """
import os

import numpy as np

import genesis as gs

from genesis.utils.torch_compat import HAS_TORCH

OUT_DIR = __OUT_DIR__
EXPECT_TORCH = __EXPECT_TORCH__

assert HAS_TORCH is EXPECT_TORCH, (HAS_TORCH, EXPECT_TORCH)

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)))
scene.build(n_envs=2)


def host(value):
    # The getters hand back a torch tensor where PyTorch is installed, and a 'DeviceView' everywhere else
    if hasattr(value, "to_numpy"):
        return np.array(value.to_numpy(), copy=True)
    return np.array(value.cpu().numpy(), copy=True)


# One write on the action boundary, with a host array: 3 linear then 3 angular velocities of the free joint
velocity = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
box.set_dofs_velocity(velocity)

for _ in range(10):
    scene.step()

arrays = {
    "pos": host(box.get_pos()),
    "quat": host(box.get_quat()),
    "vel": host(box.get_dofs_velocity()),
}
for name, array in arrays.items():
    np.save(os.path.join(OUT_DIR, name + ".npy"), array)
print(f"HAS_TORCH={HAS_TORCH} pos={arrays['pos'].tolist()} vel={arrays['vel'].tolist()}")
"""


def _state_boundary_script(out_dir: Path, *, expect_torch: bool) -> str:
    return _STATE_BOUNDARY_SCRIPT.replace("__OUT_DIR__", repr(str(out_dir))).replace(
        "__EXPECT_TORCH__", repr(expect_torch)
    )


#: Population size, generation count and horizon of the pilot run the two scenarios below reproduce
GO2_PILOT_POP_SIZE = 8
GO2_PILOT_GENERATIONS = 2
GO2_PILOT_HORIZON = 25

#: Documented fitness history of that run ('examples/locomotion/CONTEXT.md', torch-free and numpy smoke alike): the
#: device residency's exact values, which are the numpy residency's -7.3785 / -7.3180 up to one float32 ulp.
GO2_PILOT_FITNESS = (-7.378471374511719, -7.318042755126953)
#: Bound of the fitness comparison: the residency's own documented drift is below 5e-07, so 1e-5 is generous
GO2_PILOT_FITNESS_ATOL = 1e-5
#: ``scene.step()`` calls per generation: '--stop-when-done' truncates both, at 21 and then at 10 steps
GO2_PILOT_STEPS = (21, 10)
#: Rows that end tilted in each generation: all 24 of them, because an untrained policy falls
GO2_PILOT_TILTED = 24
#: The documented environment-stream seed triples of master seed 0, one row per generation
GO2_PILOT_SEEDS = ((1454269175, 757617120, 1421746521), (1313752747, 908599966, 53557616))
#: The documented evolution-stream seed of master seed 0, independent of the triples above
GO2_PILOT_EVOLUTION_SEED = 3757552657

#: Child script of 'test_go2_pilot_trains_without_torch_and_keeps_results_device_resident': the pilot's own objects,
#: driven the way its CLI drives them, with every placement assertion the device residency promises and one
#: machine-readable marker line for the parent.  Nothing here imports PyTorch, and the harness' epilogue re-checks it.
_GO2_PILOT_SCRIPT = """
import json
import os
import sys

import numpy as np

import genesis as gs

REPO_ROOT = os.getcwd()
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "locomotion"))
sys.path.insert(0, os.path.join(REPO_ROOT, "examples", "neuroevolution"))

import evoxc

from device_runtime import assert_device_resident, assert_host_value, assert_payload_resident, device_placement
from go2_evox_kernels import resolve_backend_device
from go2_neat_env import MAX_CONNS, MAX_NODES, Go2NeatEnv
from go2_neat_kernels import NUM_ACTIONS, NUM_OBS
from seeding import EPISODES_PER_POLICY, Perturbation, generation_seed_table
from trainer import NeuroevolutionTrainer

POP_SIZE = 8
GENERATIONS = 2
HORIZON = 25
MASTER_SEED = 0
# The CLI's documented perturbation amplitudes ('--perturb-*' defaults), which 'Perturbation()' does not carry
PERTURBATION = Perturbation(base_pos=0.05, base_rpy=0.05, joint_pos=0.05)
# The logging values of a generation, which stay on the host under the device residency
HOST_KEYS = ("steps", "env_steps", "wall_time_s", "steps_per_s", "env_steps_per_s", "end_reasons", "init_qpos")
# The payload shapes the policy graph is traced with
POP_SHAPES = (
    (POP_SIZE, MAX_NODES),
    (POP_SIZE, MAX_NODES, 4),
    (POP_SIZE, MAX_CONNS, 2),
    (POP_SIZE, MAX_CONNS, 1),
)


def resolve_device():
    # 'EVOXC_DEVICE' names the target explicitly; without it, probe the first non-CPU device the backend reaches
    backend, device = resolve_backend_device("iree", None)
    if os.environ.get("EVOXC_DEVICE", "").strip():
        return backend, device
    evoxc.backends.get(backend)
    for candidate in evoxc.devices():
        if candidate.kind == "cpu":
            continue
        try:
            evoxc.from_numpy(np.zeros(1, dtype=np.float32)).to(candidate)
        except Exception as exc:
            # A wheel lists every runtime it was built with; the only honest reachability test is a placement
            print(f"unreachable {candidate.kind}:{candidate.index}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            continue
        return backend, candidate
    return backend, None


def flatten_placement(trainer, result, device_name):
    evidence = {}
    for index, leaf in enumerate(trainer.pop_arrays):
        evidence[f"pop_{index}"] = device_placement(leaf)
    for index, leaf in enumerate(trainer.species_leaves):
        evidence[f"species_{index}"] = device_placement(leaf)
    evidence["randkey"] = device_placement(trainer.randkey)
    evidence["generation"] = device_placement(trainer.generation_counter)
    for name in ("seeds", "episode_returns", "returns", "fitness"):
        evidence[f"result_{name}"] = device_placement(result[name])
    for key, value in trainer.placement.items():
        if isinstance(value, (list, tuple)):
            for index, entry in enumerate(value):
                evidence[f"placement_{key}_{index}"] = entry
        else:
            evidence[f"placement_{key}"] = value
    mismatched = {key: value for key, value in evidence.items() if value != device_name}
    assert not mismatched, f"expected every training leaf on {device_name}, got {mismatched}"
    return evidence


def main():
    backend, device = resolve_device()
    if device is None or device.kind == "cpu":
        print(f"GO2_TORCH_FREE_SKIP no reachable non-CPU evoxc device for backend {backend!r}")
        return
    device_name = f"{device.kind}:{device.index}"
    print(f"device={device_name} backend={backend}")

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)

    seed_table = generation_seed_table(MASTER_SEED, GENERATIONS)
    env = Go2NeatEnv(
        pop_size=POP_SIZE,
        master_seed=MASTER_SEED,
        horizon=HORIZON,
        perturbation=PERTURBATION,
        backend=backend,
        device=device,
        residency="device",
        seed_table=seed_table,
    )
    trainer = NeuroevolutionTrainer(
        env,
        POP_SIZE,
        GENERATIONS,
        master_seed=MASTER_SEED,
        num_inputs=NUM_OBS,
        num_outputs=NUM_ACTIONS,
        residency="device",
        seed_table=seed_table,
    )
    trainer.setup()
    # Initialization is device-first: no payload, species, key or counter leaf is ever a host value
    assert_payload_resident(
        trainer.pop_arrays, trainer.species_leaves, trainer.randkey, trainer.generation_counter, device
    )
    for leaf, shape in zip(trainer.pop_arrays, POP_SHAPES):
        assert tuple(leaf.shape) == shape, (leaf.shape, shape)

    summary = trainer.run()

    # The tell graph's output is the next generation's payload: it must still be resident on the same device
    assert_payload_resident(
        trainer.pop_arrays, trainer.species_leaves, trainer.randkey, trainer.generation_counter, device
    )

    # One more generation for the result-level evidence: 'evaluate' is where the four metrics are produced
    result = env.evaluate(trainer.pop_arrays, GENERATIONS - 1)
    for name, shape in (
        ("seeds", (EPISODES_PER_POLICY,)),
        ("episode_returns", (POP_SIZE, EPISODES_PER_POLICY)),
        ("returns", (POP_SIZE * EPISODES_PER_POLICY,)),
        ("fitness", (POP_SIZE,)),
    ):
        assert_device_resident(result[name], device, name)
        assert tuple(result[name].shape) == shape, (name, result[name].shape, shape)
    # The bookkeeping of the generation is host data; only the training values are device-resident
    for name in HOST_KEYS:
        assert_host_value(result[name], name)

    payload = {
        "device": device_name,
        "backend": backend,
        "evolution_seed": int(trainer.evolution_seed),
        "fitness_history": [float(value) for value in trainer.fitness_history],
        "steps": [int(record["steps"]) for record in trainer.records],
        "end_reasons": [record["end_reasons"] for record in trainer.records],
        "seeds": [[int(seed) for seed in row] for row in seed_table],
        "placement": flatten_placement(trainer, result, device_name),
        "total_steps": int(summary["totals"]["steps"]),
        "torch_in_process": "torch" in sys.modules,
    }
    print("GO2_TORCH_FREE " + json.dumps(payload))
    assert payload["torch_in_process"] is False, payload["torch_in_process"]
    leaked = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch."))
    assert not leaked, leaked
    env.close()


main()
"""

#: The import the CLI scenario's ``torch`` stand-in raises, so that ``import torch`` fails inside the real script
_TORCH_BLOCKER_SOURCE = '''\
raise ImportError("PyTorch is blocked by the torch-free test harness")
'''


def _torch_blocker_dir(tmp_path: Path) -> Path:
    """A directory holding a ``torch`` package that refuses to import, for ``PYTHONPATH``."""
    directory = tmp_path / "notorch"
    (directory / "torch").mkdir(parents=True)
    (directory / "torch" / "__init__.py").write_text(_TORCH_BLOCKER_SOURCE, encoding="utf-8")
    return directory


def _go2_pilot_payload(stdout: str) -> dict:
    """The parsed ``GO2_TORCH_FREE {json}`` marker line of the pilot child."""
    marker = "GO2_TORCH_FREE "
    for line in stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise AssertionError(f"The child printed no '{marker.strip()}' marker line:\n{stdout}")


def _assert_documented_pilot_run(*, fitness_history, steps, end_reasons, seeds, context: str) -> None:
    """Check a pilot run against the documented numbers of ``examples/locomotion/CONTEXT.md``.

    A regression shows up as a named field instead of one opaque dict comparison, and the fitness deltas are reported
    so a drift that stays inside the documented float32 noise is visible rather than merely tolerated.
    """
    assert len(fitness_history) == len(GO2_PILOT_FITNESS), (
        f"{context}: expected {len(GO2_PILOT_FITNESS)} generations, got {list(fitness_history)}"
    )
    deltas = [float(value) - expected for value, expected in zip(fitness_history, GO2_PILOT_FITNESS)]
    assert all(abs(delta) <= GO2_PILOT_FITNESS_ATOL for delta in deltas), (
        f"{context}: fitness history {list(fitness_history)} deviates from the documented {list(GO2_PILOT_FITNESS)} "
        f"by more than {GO2_PILOT_FITNESS_ATOL} (deltas {deltas})"
    )
    assert [int(value) for value in steps] == list(GO2_PILOT_STEPS), f"{context}: steps {list(steps)}"
    assert [list(int(seed) for seed in row) for row in seeds] == [list(row) for row in GO2_PILOT_SEEDS], (
        f"{context}: seed table {list(seeds)}"
    )
    tilted = [int(reasons.get("tilted", 0)) for reasons in end_reasons]
    assert tilted == [GO2_PILOT_TILTED] * len(GO2_PILOT_FITNESS), f"{context}: tilted endings {tilted}"


def test_torch_blocker_blocks_torch_in_a_child():
    # CPU-only and fast: proves the harness itself instead of trusting it in the slower scenarios below
    result = run_python_checked(_BLOCKER_SCRIPT, timeout=300)
    assert "HAS_TORCH=False" in result.stdout, result.stdout


@pytest.mark.slow
def test_genesis_initializes_and_steps_without_torch():
    if not has_gpu():
        pytest.skip(SKIP_NO_GPU)
    result = run_python_checked(_GPU_STEP_SCRIPT)
    assert "type=DeviceView device=device(type='cuda'" in result.stdout, result.stdout


@pytest.mark.slow
def test_torch_free_and_torch_present_genesis_agree_on_the_state_boundary(tmp_path):
    if not has_gpu():
        pytest.skip(SKIP_NO_GPU)

    torch_free_dir = tmp_path / "torch_free"
    torch_present_dir = tmp_path / "torch_present"
    torch_free_dir.mkdir()
    torch_present_dir.mkdir()

    torch_free = run_python_checked(_state_boundary_script(torch_free_dir, expect_torch=False))
    torch_present = run_python_checked(
        _state_boundary_script(torch_present_dir, expect_torch=True), block_torch=False
    )

    for name in ("pos", "quat", "vel"):
        candidate = np.load(torch_free_dir / f"{name}.npy")
        reference = np.load(torch_present_dir / f"{name}.npy")
        assert candidate.shape == reference.shape, (
            f"Shape mismatch for '{name}': torch-free={candidate.shape}, torch-present={reference.shape}.\n"
            f"torch-free: {torch_free.stdout}\ntorch-present: {torch_present.stdout}"
        )
        assert np.array_equal(candidate, reference), (
            f"'{name}' differs between the torch-free and the torch-present run: "
            f"max|diff|={float(np.max(np.abs(candidate.astype(np.float64) - reference.astype(np.float64))))}, "
            f"torch-free={candidate.tolist()}, torch-present={reference.tolist()}.\n"
            f"torch-free: {torch_free.stdout}\ntorch-present: {torch_present.stdout}"
        )

    # A state boundary that trivially agrees (both runs frozen at the initial pose) would prove nothing
    pos = np.load(torch_free_dir / "pos.npy")
    assert np.all(pos[:, 2] < 0.5), pos.tolist()
    assert np.any(np.abs(pos[:, 1]) > 1e-4), pos.tolist()


@pytest.mark.slow
def test_go2_pilot_trains_without_torch_and_keeps_results_device_resident():
    if not has_gpu():
        pytest.skip(SKIP_NO_GPU)
    # The pilot trains on a non-CPU device, so a cpu-kind 'EVOXC_DEVICE' override skips here, naming the override,
    # instead of starting a child that can only report that it found no usable device
    non_cpu_device_or_skip("iree", feature="the Go2 device residency")
    result = run_python_checked(_GO2_PILOT_SCRIPT, timeout=900)
    if "GO2_TORCH_FREE_SKIP" in result.stdout:
        pytest.skip(f"No reachable non-CPU 'evoxc' device; the pilot's device residency needs one.\n{result.stdout}")
    payload = _go2_pilot_payload(result.stdout)

    device = payload["device"]
    assert not device.startswith("cpu"), f"The pilot must train on a non-CPU device, got {device}.\n{result.stdout}"
    # The population, the species, the key, the counter and the four metrics all name that one device
    assert set(payload["placement"].values()) == {device}, payload["placement"]
    assert payload["torch_in_process"] is False, result.stdout
    assert payload["evolution_seed"] == GO2_PILOT_EVOLUTION_SEED, payload["evolution_seed"]
    _assert_documented_pilot_run(
        fitness_history=payload["fitness_history"],
        steps=payload["steps"],
        end_reasons=payload["end_reasons"],
        seeds=payload["seeds"],
        context=f"torch-free pilot run on {device}",
    )


@pytest.mark.slow
def test_go2_train_cli_runs_with_torch_import_blocked(tmp_path):
    if not has_gpu():
        pytest.skip(SKIP_NO_GPU)
    # The parent stays torch-only: only the child needs the compiler stack, so it is probed, never imported
    missing = [name for name in ("evoxc", "tensorneat") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"{SKIP_NO_EVOXC} Missing: {', '.join(missing)}.")
    # The CLI runs the pilot's device residency too, so the same non-CPU gate applies before any child starts
    non_cpu_device_or_skip("iree", feature="the Go2 --torch-free CLI")

    blocker = _torch_blocker_dir(tmp_path)
    mpl_config_dir = tmp_path / "mpl"
    mpl_config_dir.mkdir()
    child_env = dict(os.environ)
    inherited = child_env.get("PYTHONPATH", "").strip()
    parts = [str(blocker), str(REPO_ROOT)] + ([inherited] if inherited else [])
    child_env["PYTHONPATH"] = os.pathsep.join(parts)
    child_env.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    def run(argv, timeout):
        return subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

    # Prove the stand-in really blocks before trusting a run that reports no torch: an unblocked run refuses anyway
    probe = run([sys.executable, "-c", "import torch"], timeout=300)
    assert probe.returncode != 0, (
        f"'import torch' succeeded although the blocker was first on PYTHONPATH: {probe.stdout}"
    )
    assert "blocked by the torch-free test harness" in probe.stderr, probe.stderr

    output_dir = tmp_path / "out"
    argv = [
        sys.executable,
        "examples/locomotion/go2_neat_train.py",
        "--torch-free",
        "--pop-size",
        str(GO2_PILOT_POP_SIZE),
        "--generations",
        str(GO2_PILOT_GENERATIONS),
        "--horizon",
        str(GO2_PILOT_HORIZON),
        "--output-dir",
        str(output_dir),
    ]
    # '--torch-free' defaults its rollout device to the ROCm one ('rocm:0'); an explicit non-CPU override, the same
    # one the other device modules honour, is the only way the very same command runs on a CUDA host.  A cpu-kind
    # override is not forwarded: there is no host-resident torch-free pilot.
    device_override = os.environ.get("EVOXC_DEVICE", "").strip()
    if device_override and not device_override.startswith("cpu"):
        argv += ["--evoxc-device", device_override]
    completed = run(argv, timeout=900)
    assert completed.returncode == 0, (
        f"The --torch-free CLI exited with code {completed.returncode}.\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    # The header line and the closing line both report the claim the flag makes
    assert "torch_free=True torch_in_process=False" in completed.stdout, completed.stdout
    done_lines = [line for line in completed.stdout.splitlines() if line.startswith("done:")]
    assert done_lines, completed.stdout
    assert "torch_in_process=False" in done_lines[-1], done_lines[-1]

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    assert config["torch_free"] is True, config
    assert config["torch_in_process"] is False, config
    assert config["residency"] == "device", config
    assert config["genesis_backend"] == "gpu", config
    assert config["evoxc_backend"] == "iree", config
    assert config["evolution_seed"] == GO2_PILOT_EVOLUTION_SEED, config["evolution_seed"]
    _assert_documented_pilot_run(
        fitness_history=summary["totals"]["fitness_history"],
        steps=[record["steps"] for record in summary["generations"]],
        end_reasons=[record["end_reasons"] for record in summary["generations"]],
        seeds=[record["seeds"] for record in summary["generations"]],
        context=f"--torch-free CLI run on {config['evoxc_device']}",
    )
    # Both artifacts of the run exist, and the history is the same run as the summary
    history = [json.loads(line) for line in (output_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["best_fitness"] for record in history] == summary["totals"]["fitness_history"], history


def _is_scanned(path: Path) -> bool:
    """Whether ``path`` is matched by one of :data:`TRAINING_PATH`'s entries (a directory or a glob)."""
    for entry in TRAINING_PATH:
        if entry.is_dir():
            if path.is_relative_to(entry):
                return True
        elif path.parent == entry.parent and path.match(entry.name):
            return True
    return False


def test_no_torch_imports_on_the_training_path():
    # Guards the scan itself: an all-missing path list must not pass with a reassuring empty violation list
    for required in (
        GENESIS_DIR / "utils" / "torch_compat.py",
        GENESIS_DIR / "grad" / "bridge.py",
        EXAMPLES_DIR / "neuroevolution" / "trainer.py",
        *GO2_PILOT_FILES,
    ):
        assert required.is_file(), required

    # The pilot's own modules must really be reached by the globs, not merely exist next to them
    for required in GO2_PILOT_FILES:
        assert _is_scanned(required), f"{required} is not matched by TRAINING_PATH"

    assert_no_torch_imports(
        TRAINING_PATH,
        allowlist=TORCH_IMPORT_ALLOWLIST,
        exclude=TORCH_IMPORT_EXCLUDES,
    )

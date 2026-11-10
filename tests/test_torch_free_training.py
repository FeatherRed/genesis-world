"""Torch-free training tests: Genesis must import, initialize, step and be driven without PyTorch.

Genesis is being made usable where PyTorch is not installed, so that a training stack can run on Quadrants and
NumPy alone.  Proving that requires the *absence* of PyTorch, which the parent pytest process cannot provide (its
conftest imports torch).  Every scenario therefore runs in a child interpreter built by ``tests/torch_block.py``,
whose meta-path blocker refuses ``torch`` and every ``torch.*`` name before anything can import it:

* the blocker itself works, and importing Genesis under it leaves ``HAS_TORCH`` False and ``torch`` out of
  ``sys.modules``;
* a real AMD GPU scene (plane + box, 2 environments, 10 steps) initializes, steps and returns device-resident state
  through the torch-free ``DeviceView`` bridge, with no torch in the child;
* the static half: nothing on the torch-free training path imports PyTorch, except the two sanctioned ``genesis/``
  files;
* the state boundary agrees bit for bit between a torch-free and a torch-present child running the very same scene,
  including one control write with a host NumPy array, which is what a training loop does every step.

The two GPU scenarios are guarded on the machine rather than on a marker (the suite has no ``gpu`` marker) and skip
cleanly when no GPU is visible.  Run the module from the repository root::

    PYTHONPATH=$PWD:/tmp/gs-pytest-libs /root/evox-robotics/.venv/bin/python -m pytest \
        tests/test_torch_free_training.py -q -n 0

The two GPU tests are ``slow`` because the first Quadrants compile of the scene takes about a minute each; the other
two are CPU-only and take a few seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.torch_block import (
    EXAMPLES_DIR,
    GENESIS_DIR,
    assert_no_torch_imports,
    has_gpu,
    run_python_checked,
)

pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: Canonical skip reason of the GPU scenarios, resolved by the session's skip-reason registry
SKIP_NO_GPU = "No usable GPU detected; the torch-free Genesis run needs one."

#: The torch-free training path: every shared neuroevolution module, the Go2 and drone pilots (whose glob tolerates a
#: file that is not written yet), and the whole Genesis package.
#: 'examples/neuroevolution/*.py' is deliberately single-level: the 'benchmarks/' subtree measures PyTorch itself
#: (import and residency probes) and is not part of the training path.
TRAINING_PATH = (
    EXAMPLES_DIR / "neuroevolution" / "*.py",
    EXAMPLES_DIR / "locomotion" / "go2_neat_*.py",
    EXAMPLES_DIR / "drone" / "hover_neat_*.py",
    GENESIS_DIR,
)

#: The only files of 'genesis/' allowed to import PyTorch, with the reason for each exemption.
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

gs.init(backend=gs.amdgpu, logging_level="warning")
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

gs.init(backend=gs.amdgpu, precision="32", logging_level="warning", seed=0)
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


def test_no_torch_imports_on_the_training_path():
    # Guards the scan itself: an all-missing path list must not pass with a reassuring empty violation list
    for required in (
        GENESIS_DIR / "utils" / "torch_compat.py",
        GENESIS_DIR / "grad" / "bridge.py",
        EXAMPLES_DIR / "neuroevolution" / "trainer.py",
    ):
        assert required.is_file(), required

    assert_no_torch_imports(
        TRAINING_PATH,
        allowlist=TORCH_IMPORT_ALLOWLIST,
        exclude=TORCH_IMPORT_EXCLUDES,
    )

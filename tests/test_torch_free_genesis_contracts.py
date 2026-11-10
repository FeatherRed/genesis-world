"""Genesis contracts under a torch-free interpreter: DLPack writes, field-backed buffers and the error mask.

Three Genesis behaviours are observable only in a process without PyTorch, and the parent pytest process cannot
provide one (``tests/conftest.py`` imports torch).  Every scenario therefore runs in a child interpreter built by
``tests/torch_block.py``, whose meta-path blocker refuses ``torch`` and every ``torch.*`` name before anything can
import it, and whose epilogue fails the child if torch reached ``sys.modules`` anyway:

* a strides-less DLPack producer -- an ``evoxc``/IREE ``core.Tensor``, which exposes ``shape``, ``dtype`` and
  ``__dlpack__`` but no Python-visible ``.strides`` -- writes a batch-first ``(n_envs, 6)`` dofs-velocity block into a
  scene.  The read-back must match the written values exactly, the box must really move, and ``gs.utils.dlpack_info``
  must describe the producer as compact row-major: the write path reads the DLPack capsule, never a Python attribute.
* ``performance_mode=True`` builds and steps on field-backed buffers (``gs.use_ndarray is False``): the getters hand
  back device-resident ``DeviceView``s, ``scene.get_error_envs_mask()`` is a boolean ``(n_envs,)`` view, and the masked
  ``set_qpos`` writes land on exactly the selected rows, in both accepted ``envs_idx`` forms.
* ``scene.get_error_envs_mask()`` is boolean in both residencies: the very same child script runs once torch-blocked
  and once torch-present, flags two environments through ``Solver.flag_errno``, clears them again, and reports the same
  value sequence, with ``np.bool_``/``kDLBool`` without PyTorch and ``torch.bool`` with it.

Each child asserts its own contract and prints one machine-readable marker line, which the parent re-checks; every
child is bounded by a 900 s timeout, one child per ``gs.init`` configuration.  The GPU scenarios are guarded on the
machine rather than on a marker (the suite has no ``gpu`` marker) and skip cleanly when no GPU is visible; the first
one also needs a reachable non-CPU ``evoxc`` device, which :func:`tests.device_pilot.reachable_device_or_skip` names
in its skip reason.  None of them calls ``skip_if_unsafe_hsa_runtime()``: that gate covers Genesis scenes built inside
the pytest process, while every scene here lives in a child, where ``genesis/utils/hsa.py`` repairs normally.

Run the module from the repository root::

    PYTHONPATH=$PWD:/tmp/gs-pytest-libs /root/evox-robotics/.venv/bin/python -m pytest \
        tests/test_torch_free_genesis_contracts.py -q -n 0

The three tests are ``slow`` because each child pays its own cold Quadrants compile of the scene plus, for the first
one, the IREE compile of the producer, which takes 25-40 seconds per child.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.device_pilot import reachable_device_or_skip
from tests.torch_block import has_gpu, run_python_checked

pytestmark = [pytest.mark.required, pytest.mark.parametrize("backend", [None])]  # Disable genesis initialization

#: Canonical skip reason of the GPU scenarios, resolved by the session's skip-reason registry
SKIP_NO_GPU = "No usable GPU detected; the torch-free Genesis run needs one."

#: Skip reason of the producer scenario when the optional compiler stack it needs is not installed here
SKIP_NO_EVOXC = "The strides-less DLPack producer is an 'evoxc' device tensor, and 'evoxc' is not installed."

#: Child script of 'test_strides_less_device_producer_write_lands'. '__DEVICE__' is substituted with the
#: '(kind, index)' pair of the 'evoxc' device the parent probed, so both processes agree on the producer's placement.
_STRIDES_LESS_PRODUCER_SCRIPT = """
import json
import sys

import numpy as np

import genesis as gs

import evoxc

DEVICE = __DEVICE__
# Batch-first (n_envs, 6) dofs velocities of the free joint: 3 linear then 3 angular entries per environment
WRITTEN = np.array(
    [
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    ],
    dtype=np.float32,
)


def strides_less(value):
    # The frontend tensor is the producer; its backend payload is the raw 'evoxc.core' tensor that a frontend grown a
    # '.strides' property would leave usable.  A producer that exposes '.strides' would let a copyability check read a
    # Python attribute instead of the capsule, which is the very bug this scenario reproduces.
    if not hasattr(value, "strides"):
        return value
    raw = getattr(value, "data", None)
    if raw is not None and not hasattr(raw, "strides"):
        return raw
    return None


def main():
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    evoxc.backends.get("iree")
    producer = strides_less(evoxc.from_numpy(WRITTEN).to(evoxc.Device(*DEVICE)))
    if producer is None:
        print(
            "GENESIS_CONTRACT_PRODUCER_SKIP no 'evoxc' object can be produced without a Python-visible '.strides'"
        )
        return

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)))
    scene.build(n_envs=len(WRITTEN))

    # The producer is described from its DLPack capsule alone: a NULL strides pointer means compact row-major
    info = gs.utils.dlpack_info(producer)
    assert tuple(info.shape) == WRITTEN.shape, (info.shape, WRITTEN.shape)
    assert info.dtype == np.dtype(np.float32), info.dtype
    assert info.itemsize == WRITTEN.itemsize, info.itemsize
    assert tuple(info.strides) == (WRITTEN.shape[1], 1), info.strides

    box.set_dofs_velocity(producer)

    readback = box.get_dofs_velocity()
    assert type(readback).__name__ == "DeviceView", type(readback).__name__
    assert tuple(readback.shape) == WRITTEN.shape, readback.shape
    host = np.array(readback.to_numpy(), copy=True)
    assert np.array_equal(host, WRITTEN), (host.tolist(), WRITTEN.tolist())

    start = np.array(box.get_pos().to_numpy(), copy=True)
    for _ in range(10):
        scene.step()
    end = np.array(box.get_pos().to_numpy(), copy=True)
    lateral = float(np.abs(end[:, :2] - start[:, :2]).max())
    # A silently dropped write would leave every box falling straight down from its initial pose
    assert np.all(end[:, 2] < start[:, 2]), (start.tolist(), end.tolist())
    assert lateral > 1e-4, lateral

    payload = {
        "device": f"{DEVICE[0]}:{DEVICE[1]}",
        "producer_type": f"{type(producer).__module__}.{type(producer).__name__}",
        "producer_has_strides": hasattr(producer, "strides"),
        "written": WRITTEN.tolist(),
        "readback": host.tolist(),
        "dlpack": {"shape": list(info.shape), "strides": list(info.strides), "dtype": info.dtype.name},
        "start_pos": start.tolist(),
        "end_pos": end.tolist(),
        "lateral_drift": lateral,
        "torch_in_process": "torch" in sys.modules,
    }
    print("GENESIS_CONTRACT_PRODUCER " + json.dumps(payload))
    assert payload["torch_in_process"] is False, payload["torch_in_process"]
    # An 'evoxc' device tensor still alive when the Quadrants scene is torn down at interpreter exit segfaults the
    # ROCm teardown (measured: exit code -11, twice in a row, and clean once the tensor is dropped before the exit)
    del producer


main()
"""

#: Child script of 'test_performance_mode_build_and_masked_reset'.
_PERFORMANCE_MODE_SCRIPT = """
import json
import sys

import numpy as np

import genesis as gs

N_ENVS = 4
INITIAL_POS = (0.0, 0.0, 0.5)
# The free joint's 7 qpos values: position then (w, x, y, z) quaternion
RESET_QPOS = np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
ROW_QPOS = np.array([0.1, 0.1, 0.6, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def main():
    gs.init(backend=gs.gpu, precision="32", performance_mode=True, logging_level="warning", seed=0)
    # 'performance_mode=True' is supported without PyTorch because it backs every solver buffer with a Quadrants field
    assert gs.use_ndarray is False, gs.use_ndarray

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=INITIAL_POS))
    scene.build(n_envs=N_ENVS)

    for _ in range(10):
        scene.step()

    pos = box.get_pos()
    assert type(pos).__name__ == "DeviceView", type(pos).__name__
    assert tuple(pos.shape) == (N_ENVS, 3), pos.shape
    assert pos.device.type == "cuda", pos.device

    mask = scene.get_error_envs_mask()
    assert type(mask).__name__ == "DeviceView", type(mask).__name__
    assert tuple(mask.shape) == (N_ENVS,), mask.shape
    assert mask.dtype == np.bool_, mask.dtype
    assert tuple(mask.strides) == (1,), mask.strides
    assert not mask.to_numpy().any(), mask.to_numpy()

    fallen = np.array(pos.to_numpy(), copy=True)
    assert np.all(fallen[:, 2] < INITIAL_POS[2]), fallen.tolist()

    # The documented masked-reset form: one row per selected environment, in the order of the selection
    box.set_qpos(np.tile(RESET_QPOS, (2, 1)), envs_idx=[0, 2])
    after_reset = np.array(box.get_pos().to_numpy(), copy=True)
    assert np.array_equal(after_reset[[0, 2]], np.tile(RESET_QPOS[:3], (2, 1))), after_reset.tolist()
    assert np.array_equal(after_reset[[1, 3]], fallen[[1, 3]]), after_reset.tolist()

    scene.step()
    after_step = np.array(box.get_pos().to_numpy(), copy=True)
    assert np.abs(after_step[[0, 2], 2] - RESET_QPOS[2]).max() < 1e-2, after_step.tolist()
    assert np.all(after_step[[1, 3], 2] < fallen[[1, 3], 2]), after_step.tolist()
    assert scene.get_error_envs_mask().dtype == np.bool_, scene.get_error_envs_mask().dtype

    # The other accepted form of the same field-backed buffer: one '(1, 7)' row broadcast over a one-env selection
    box.set_qpos(ROW_QPOS.reshape(1, -1), envs_idx=[1])
    after_row = np.array(box.get_pos().to_numpy(), copy=True)
    assert np.array_equal(after_row[1], ROW_QPOS[:3]), after_row.tolist()
    mask = scene.get_error_envs_mask()
    assert mask.dtype == np.bool_, mask.dtype
    assert tuple(mask.strides) == (1,), mask.strides

    payload = {
        "use_ndarray": bool(gs.use_ndarray),
        "dtype_name": mask.dtype.name,
        "mask_shape": list(mask.shape),
        "mask_strides": list(mask.strides),
        "mask_values": [bool(value) for value in mask.to_numpy()],
        "fallen_pos": fallen.tolist(),
        "reset_pos": after_reset[[0, 2], :].tolist(),
        "untouched_pos": after_reset[[1, 3], :].tolist(),
        "after_step_pos": after_step[[0, 2], :].tolist(),
        "row_pos": after_row[1].tolist(),
        "torch_in_process": "torch" in sys.modules,
    }
    print("GENESIS_CONTRACT_PERFORMANCE_MODE " + json.dumps(payload))
    assert payload["torch_in_process"] is False, payload["torch_in_process"]


main()
"""

#: Child script of 'test_error_envs_mask_is_boolean_in_both_residencies', run twice: once with PyTorch blocked and
#: once with it installed. '__EXPECT_TORCH__' is substituted per run.
_ERROR_MASK_SCRIPT = """
import ctypes
import json
import sys

import numpy as np

import genesis as gs
import quadrants as qd

from genesis.utils.torch_compat import HAS_TORCH

EXPECT_TORCH = __EXPECT_TORCH__
N_ENVS = 4
FLAGGED_ENVS = [1, 3]
ERROR_CODE = 1
# DLPack v0 data type code of a boolean, as 'genesis.utils.device_buffer' names it
KDLBOOL = 6


# The DLPack v0 ABI, spelled out so that the capsule can be read without any private helper
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_is_valid.restype = ctypes.c_int
_capsule_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def dlpack_dtype(value):
    # A capsule carries a DLManagedTensor whose first field is the DLTensor this reads, without consuming the capsule
    capsule = value.__dlpack__()
    assert _capsule_is_valid(capsule, b"dltensor"), "the mask exports no DLPack v0 capsule"
    tensor = _DLTensor.from_address(_capsule_pointer(capsule, b"dltensor"))
    return {
        "code": int(tensor.dtype.code),
        "bits": int(tensor.dtype.bits),
        "lanes": int(tensor.dtype.lanes),
        "shape": [int(tensor.shape[axis]) for axis in range(tensor.ndim)],
        "strides": [int(tensor.strides[axis]) for axis in range(tensor.ndim)] if tensor.strides else None,
    }


def roundtrip(value):
    # 'np.from_dlpack' accepts kDLCPU only, so a device-resident mask is imported through the bridge's own
    # device-to-device DLPack copy, which is what a torch-free loop does with a producer
    buffer = qd.ndarray(gs.qd_bool, shape=tuple(value.shape))
    gs.utils.copy_from_dlpack(buffer, value)
    return buffer.to_numpy()


def main():
    assert HAS_TORCH is EXPECT_TORCH, (HAS_TORCH, EXPECT_TORCH)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.5)))
    scene.build(n_envs=N_ENVS)

    def host(mask):
        # The getters hand back a torch tensor where PyTorch is installed, and a 'DeviceView' everywhere else
        if HAS_TORCH:
            return mask.cpu().numpy().copy()
        return np.array(mask.to_numpy(), copy=True)

    mask = scene.get_error_envs_mask()
    before = host(mask)
    if HAS_TORCH:
        import torch

        is_boolean = bool(mask.dtype == torch.bool)
    else:
        is_boolean = bool(mask.dtype == np.bool_)

    # Flag two environments: the mask only tests whether a latched flag is positive
    scene.rigid_solver.flag_errno(ERROR_CODE, envs_idx=FLAGGED_ENVS)
    flagged_mask = scene.get_error_envs_mask()
    flagged = host(flagged_mask)

    dlpack = {"code": None, "bits": None, "lanes": None, "shape": None, "strides": None}
    roundtrip_values = None
    numpy_from_dlpack = {"values": None, "error": None}
    if not HAS_TORCH:
        # The export is read off the capsule, and imported through the sanctioned torch-free DLPack write path
        dlpack = dlpack_dtype(flagged_mask)
        roundtrip_values = [bool(value) for value in roundtrip(flagged_mask)]
        try:
            numpy_from_dlpack["values"] = [bool(value) for value in np.from_dlpack(flagged_mask)]
        except Exception as exc:  # noqa: BLE001 - recorded, never trusted: NumPy imports kDLCPU tensors only
            numpy_from_dlpack["error"] = f"{type(exc).__name__}: {exc}"

    scene.clear_errno()
    cleared = host(scene.get_error_envs_mask())

    payload = {
        "has_torch": HAS_TORCH,
        "dtype_name": type(mask).__name__,
        "dtype": str(mask.dtype),
        "is_boolean": is_boolean,
        "strides": [int(stride) for stride in (mask.stride() if HAS_TORCH else mask.strides)],
        "before": [bool(value) for value in before],
        "flagged": [bool(value) for value in flagged],
        "cleared": [bool(value) for value in cleared],
        "dlpack": dlpack,
        "roundtrip": roundtrip_values,
        "numpy_from_dlpack": numpy_from_dlpack,
        "torch_in_process": "torch" in sys.modules,
    }
    print("GENESIS_CONTRACT_ERROR_MASK " + json.dumps(payload))
    assert payload["torch_in_process"] is EXPECT_TORCH, payload["torch_in_process"]
    if HAS_TORCH:
        assert payload["dtype"] == "torch.bool", payload["dtype"]
    else:
        assert payload["dtype"] == "bool", payload["dtype"]
        assert dlpack["code"] == KDLBOOL, dlpack
        assert dlpack["bits"] == 8, dlpack
        assert dlpack["lanes"] == 1, dlpack


main()
"""


def _marker_payload(stdout: str, marker: str) -> dict:
    """The parsed ``<marker> {json}`` line of a child."""
    prefix = marker + " "
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise AssertionError(f"The child printed no '{marker}' marker line:\n{stdout}")


def _render(script: str, **substitutions) -> str:
    """``script`` with every ``__NAME__`` placeholder replaced by the ``repr`` of its value."""
    for name, value in substitutions.items():
        script = script.replace(f"__{name}__", repr(value))
    return script


@pytest.mark.slow
def test_strides_less_device_producer_write_lands():
    if not has_gpu():
        pytest.skip(SKIP_NO_GPU)
    # The strides-less producer is an 'evoxc' device tensor, and 'evoxc' is an optional, undeclared dependency
    pytest.importorskip("evoxc", reason=SKIP_NO_EVOXC)
    _, device = reachable_device_or_skip("iree")

    result = run_python_checked(
        _render(_STRIDES_LESS_PRODUCER_SCRIPT, DEVICE=(device.kind, device.index)), timeout=900
    )
    if "GENESIS_CONTRACT_PRODUCER_SKIP" in result.stdout:
        pytest.skip(
            "No 'evoxc' object exposes a DLPack producer without a Python-visible '.strides', so the copyability "
            f"check this scenario reproduces cannot be exercised.\n{result.stdout}"
        )
    payload = _marker_payload(result.stdout, "GENESIS_CONTRACT_PRODUCER")

    # The producer really is the strides-less kind: the write path must read the capsule, not a Python attribute
    assert payload["producer_type"].startswith("evoxc."), payload["producer_type"]
    assert payload["producer_has_strides"] is False, payload["producer_type"]
    assert payload["device"] == f"{device.kind}:{device.index}", payload["device"]

    # 'dlpack_info' describes a compact row-major '(n_envs, 6)' float32 block, read off that capsule
    assert payload["dlpack"]["shape"] == [3, 6], payload["dlpack"]
    assert payload["dlpack"]["strides"] == [6, 1], payload["dlpack"]
    assert payload["dlpack"]["dtype"] == "float32", payload["dlpack"]

    written = np.asarray(payload["written"], dtype=np.float32)
    readback = np.asarray(payload["readback"], dtype=np.float32)
    assert written.shape == readback.shape == (3, 6), (written.shape, readback.shape)
    assert np.array_equal(readback, written), (readback.tolist(), written.tolist())
    # A write of zeros, or a dropped one, would pass an equality check against zeros alone
    assert float(np.abs(written).max()) > 0.0, written.tolist()

    start = np.asarray(payload["start_pos"], dtype=np.float32)
    end = np.asarray(payload["end_pos"], dtype=np.float32)
    assert np.all(end[:, 2] < start[:, 2]), (start.tolist(), end.tolist())
    assert max(float(row[2]) for row in payload["end_pos"]) < 0.5, payload["end_pos"]
    assert float(payload["lateral_drift"]) > 1e-4, payload["lateral_drift"]
    assert payload["torch_in_process"] is False, result.stdout


@pytest.mark.slow
def test_performance_mode_build_and_masked_reset():
    if not has_gpu():
        pytest.skip(SKIP_NO_GPU)

    result = run_python_checked(_PERFORMANCE_MODE_SCRIPT, timeout=900)
    payload = _marker_payload(result.stdout, "GENESIS_CONTRACT_PERFORMANCE_MODE")

    # Field-backed buffers: 'V()' allocated every solver buffer as a Quadrants field, never a host ndarray
    assert payload["use_ndarray"] is False, payload["use_ndarray"]
    assert payload["dtype_name"] == "bool", payload["dtype_name"]
    assert payload["mask_shape"] == [4], payload["mask_shape"]
    assert payload["mask_strides"] == [1], payload["mask_strides"]
    assert payload["mask_values"] == [False, False, False, False], payload["mask_values"]

    # The two reset rows are exactly at the reset pose, and the two others were left where they fell
    reset = np.asarray(payload["reset_pos"], dtype=np.float32)
    assert np.array_equal(reset, np.tile(np.array([0.0, 0.0, 0.5], dtype=np.float32), (2, 1))), reset.tolist()
    fallen = np.asarray(payload["fallen_pos"], dtype=np.float32)
    untouched = np.asarray(payload["untouched_pos"], dtype=np.float32)
    assert np.array_equal(untouched, fallen[[1, 3]]), (untouched.tolist(), fallen[[1, 3]].tolist())
    assert np.all(fallen[:, 2] < 0.5), fallen.tolist()

    # One step later the reset rows are still near the reset pose while the others keep falling
    after_step = np.asarray(payload["after_step_pos"], dtype=np.float32)
    assert np.abs(after_step[:, 2] - 0.5).max() < 1e-2, after_step.tolist()
    assert np.all(after_step[:, 2] < 0.5), after_step.tolist()

    # The single-row broadcast form landed on exactly the selected environment
    assert np.array_equal(
        np.asarray(payload["row_pos"], dtype=np.float32), np.array([0.1, 0.1, 0.6], dtype=np.float32)
    ), payload["row_pos"]
    assert payload["torch_in_process"] is False, result.stdout


@pytest.mark.slow
def test_error_envs_mask_is_boolean_in_both_residencies():
    if not has_gpu():
        pytest.skip(SKIP_NO_GPU)

    torch_free = run_python_checked(_render(_ERROR_MASK_SCRIPT, EXPECT_TORCH=False), timeout=900)
    torch_present = run_python_checked(
        _render(_ERROR_MASK_SCRIPT, EXPECT_TORCH=True), block_torch=False, timeout=900
    )
    payloads = {
        "torch-free": _marker_payload(torch_free.stdout, "GENESIS_CONTRACT_ERROR_MASK"),
        "torch-present": _marker_payload(torch_present.stdout, "GENESIS_CONTRACT_ERROR_MASK"),
    }
    expected = {"before": [False] * 4, "flagged": [False, True, False, True], "cleared": [False] * 4}

    for residency, payload in payloads.items():
        # Both residencies latch exactly the flag 'flag_errno' set, and 'clear_errno' clears it again
        for field, values in expected.items():
            assert payload[field] == values, f"{residency}: {field} {payload[field]}"
        assert payload["is_boolean"] is True, f"{residency}: dtype {payload['dtype']}"
        assert payload["strides"] == [1], f"{residency}: strides {payload['strides']}"
        assert payload["torch_in_process"] is payload["has_torch"], f"{residency}: {payload}"

    torch_free_payload = payloads["torch-free"]
    torch_present_payload = payloads["torch-present"]

    # The very same value sequence on both sides, and each side's documented dtype
    for field in expected:
        assert torch_free_payload[field] == torch_present_payload[field], (field, payloads)
    assert torch_free_payload["has_torch"] is False, torch_free.stdout
    assert torch_free_payload["torch_in_process"] is False, torch_free.stdout
    assert torch_free_payload["dtype_name"] == "DeviceView", torch_free_payload["dtype_name"]
    assert torch_free_payload["dtype"] == "bool", torch_free_payload["dtype"]
    assert torch_present_payload["has_torch"] is True, torch_present.stdout
    assert torch_present_payload["torch_in_process"] is True, torch_present.stdout
    assert torch_present_payload["dtype_name"] == "Tensor", torch_present_payload["dtype_name"]
    assert torch_present_payload["dtype"] == "torch.bool", torch_present_payload["dtype"]

    # The torch-free mask exports a kDLBool capsule (code 6, 8 bits, one lane) and round-trips through DLPack
    dlpack = torch_free_payload["dlpack"]
    assert dlpack["code"] == 6, dlpack
    assert dlpack["bits"] == 8, dlpack
    assert dlpack["lanes"] == 1, dlpack
    assert dlpack["shape"] == [4], dlpack
    assert dlpack["strides"] == [1], dlpack
    assert torch_free_payload["roundtrip"] == expected["flagged"], torch_free_payload["roundtrip"]
    numpy_values = torch_free_payload["numpy_from_dlpack"]["values"]
    if numpy_values is not None:
        assert numpy_values == expected["flagged"], numpy_values

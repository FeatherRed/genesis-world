"""Torch-free write direction of the Quadrants <-> DLPack bridge, backed by the HIP runtime.

Quadrants exports every field, ndarray and tensor as a DLPack v0 capsule but implements no importer, so a producer
cannot hand its buffer over through DLPack alone. This module writes into the device memory of the destination
through a guarded HIP device-to-device copy, and falls back to NumPy on a CPU backend::

    copy_from_dlpack(field, producer, transpose=True)   # producer implements '__dlpack__', e.g. an evoxc output

The read direction, plus every layout primitive this module relies on, lives in :mod:`genesis.utils.device_buffer`.
The dependency is one-directional (this module imports the read side, never the other way around, apart from the
re-export line at the end of that module), and nothing here imports PyTorch.
The HIP runtime is loaded lazily, so the module imports cleanly on a machine with no GPU at all.
"""

import ctypes
import functools
from typing import Any

import numpy as np
import quadrants as qd

from .device_buffer import (
    DeviceView,
    _contiguous_strides,
    _host_index_array,
    _kDLCPU,
    _qd_program,
    _runtime_backend,
    _unwrap_qd,
    dlpack_info,
    view_of,
)
from .torch_compat import _raise_genesis_error

# HIP copy kind for a device-to-device transfer.
_HIP_MEMCPY_DEVICE_TO_DEVICE = 4


# ------------------------------------------------------------------------------------------------
# Write direction
# ------------------------------------------------------------------------------------------------


def _selected_indices(selection: Any, num_items: int) -> tuple[int, ...] | None:
    """Indices selected along one axis, or None when the whole axis is selected.

    Unlike a view mask, a write selection may be any collection of indices: the copy is split into one transfer per
    contiguous run, so no gather and no temporary buffer is needed. The selection is read back to the host, which is
    fine because it is small and the alternative would be a silent out-of-bound write.
    """
    if selection is None:
        return None
    if isinstance(selection, slice):
        indices = tuple(range(*selection.indices(num_items)))
    elif isinstance(selection, range):
        indices = tuple(index + num_items if index < 0 else index for index in selection)
    elif type(selection) is int:
        indices = ((selection + num_items if selection < 0 else selection),)
    else:
        values = _host_index_array(selection)
        if values.dtype == np.bool_:
            return tuple(int(index) for index in np.nonzero(values)[0])
        indices = tuple(int(index) + num_items if int(index) < 0 else int(index) for index in values.reshape(-1))
    for index in indices:
        if not 0 <= index < num_items:
            _raise_genesis_error(f"Index {index} is out of bound for an axis of size {num_items}.")
    return indices


def _index_runs(indices: tuple[int, ...] | None, num_items: int) -> tuple[tuple[int, int], ...]:
    """Runs of consecutive ascending indices, one contiguous transfer being enough for each run."""
    if indices is None:
        return ((0, num_items),)
    runs: list[list[int]] = []
    for index in indices:
        if runs and index == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([index, 1])
    return tuple((start, count) for start, count in runs)


class _HipRuntime:
    """The handful of HIP entry points needed to write into a Quadrants device buffer."""

    def __init__(self):
        try:
            self._library = ctypes.CDLL("libamdhip64.so")
        except OSError as e:
            _raise_genesis_error(
                "The HIP runtime 'libamdhip64.so' is required to write into a device buffer, but it could not be "
                "loaded.",
                cause=e,
            )
        self._library.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._library.hipMemcpy.restype = ctypes.c_int
        self._library.hipMemcpy2D.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._library.hipMemcpy2D.restype = ctypes.c_int
        self._library.hipDeviceSynchronize.argtypes = []
        self._library.hipDeviceSynchronize.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, function_name: str) -> None:
        if status != 0:
            _raise_genesis_error(f"The HIP call '{function_name}' failed with the error code {status}.")

    def copy(self, dst_ptr: int, src_ptr: int, nbytes: int) -> None:
        self._check(
            self._library.hipMemcpy(dst_ptr, src_ptr, nbytes, _HIP_MEMCPY_DEVICE_TO_DEVICE),
            "hipMemcpy",
        )

    def copy_2d(
        self, dst_ptr: int, dst_pitch: int, src_ptr: int, src_pitch: int, width: int, height: int
    ) -> None:
        self._check(
            self._library.hipMemcpy2D(
                dst_ptr, dst_pitch, src_ptr, src_pitch, width, height, _HIP_MEMCPY_DEVICE_TO_DEVICE
            ),
            "hipMemcpy2D",
        )

    def synchronize(self) -> None:
        self._check(self._library.hipDeviceSynchronize(), "hipDeviceSynchronize")


@functools.lru_cache(maxsize=1)
def _hip_runtime() -> _HipRuntime:
    """HIP runtime of the process, loaded on first use so that no HIP library is touched on a non-ROCm backend."""
    return _HipRuntime()


def _hip_copy_transposed(dst_ptr: int, src_ptr: int, num_values: int, num_envs: int, itemsize: int) -> None:
    """Transpose-copy a contiguous '(num_envs, num_values)' source into a contiguous '(num_values, num_envs)' target.

    Each of the 'min(num_values, num_envs)' calls transfers one destination row or column in a single 2D copy, so no
    intermediate buffer nor host roundtrip is ever allocated.
    """
    runtime = _hip_runtime()
    if num_values <= num_envs:
        # One call per destination row: the destination row is contiguous while the matching source column is strided.
        for i_v in range(num_values):
            runtime.copy_2d(
                dst_ptr + i_v * num_envs * itemsize,
                itemsize,
                src_ptr + i_v * itemsize,
                num_values * itemsize,
                itemsize,
                num_envs,
            )
    else:
        # One call per destination column: the source row is contiguous while the destination column is strided.
        for i_b in range(num_envs):
            runtime.copy_2d(
                dst_ptr + i_b * itemsize,
                num_envs * itemsize,
                src_ptr + i_b * num_values * itemsize,
                itemsize,
                itemsize,
                num_values,
            )


def _hip_copy_selection(
    dst_ptr: int,
    src_ptr: int,
    num_values: int,
    num_envs: int,
    itemsize: int,
    value_runs: tuple[tuple[int, int], ...],
    env_runs: tuple[tuple[int, int], ...],
    is_transposed: bool,
) -> None:
    """Copy the selected block of a contiguous source into a contiguous target, one transfer per contiguous run."""
    runtime = _hip_runtime()
    for value_start, value_count in value_runs:
        for env_start, env_count in env_runs:
            for i_v in range(value_start, value_start + value_count):
                if is_transposed:
                    runtime.copy_2d(
                        dst_ptr + (i_v * num_envs + env_start) * itemsize,
                        itemsize,
                        src_ptr + (env_start * num_values + i_v) * itemsize,
                        num_values * itemsize,
                        itemsize,
                        env_count,
                    )
                else:
                    offset = (i_v * num_envs + env_start) * itemsize
                    runtime.copy(dst_ptr + offset, src_ptr + offset, env_count * itemsize)


def _write_qd_object(value: Any, array: np.ndarray) -> None:
    _unwrap_qd(value).from_numpy(array)


def _host_source_array(src: Any) -> np.ndarray:
    if isinstance(src, (qd.Field, qd.Ndarray, qd.Tensor)):
        return src.to_numpy()
    if isinstance(src, np.ndarray):
        return src
    return np.from_dlpack(src)


def _copy_host(
    dst: Any,
    dst_view: DeviceView,
    src: Any,
    value_runs: tuple[tuple[int, int], ...],
    env_runs: tuple[tuple[int, int], ...],
    is_transposed: bool,
    is_whole: bool,
) -> None:
    """Host fallback of the write direction, for a CPU backend."""
    patch = _host_source_array(src)
    if is_transposed:
        patch = patch.T
    patch = patch.reshape(dst_view.shape)
    if is_whole:
        _write_qd_object(dst, np.ascontiguousarray(patch))
        return
    array = dst_view.to_numpy()
    for value_start, value_count in value_runs:
        for env_start, env_count in env_runs:
            selection = (
                slice(value_start, value_start + value_count),
                slice(env_start, env_start + env_count),
            )
            array[selection] = patch[selection]
    _write_qd_object(dst, array)


def copy_from_dlpack(
    dst: Any,
    src: Any,
    *,
    transpose: bool = False,
    envs_idx: Any = None,
    idx: Any = None,
    sync: bool = True,
) -> None:
    """Copy the DLPack producer 'src' into the Quadrants buffer 'dst', which is batch-last.

    'dst' is a field, ndarray or tensor of shape '(num_values, num_envs)', or '(num_envs,)' with a single value axis.
    With 'transpose=True', 'src' is laid out '(num_envs, num_values)', as most machine-learning runtimes emit it;
    without it, 'src' has the layout of 'dst'. Both sides must agree on dtype and device: no cast and no conversion is
    performed. 'envs_idx' selects along the batch axis and 'idx' along the value axis; a mask that is not a plain
    slice or a contiguous ascending run is refused rather than silently gathered.

    On a ROCm backend the transfer is a device-to-device HIP copy: one 'hipMemcpy' for a contiguous same-layout copy,
    'min(num_values, num_envs)' 'hipMemcpy2D' calls for a transposed copy, and one contiguous run per selected outer
    index for a masked copy. No host roundtrip is performed on this path, and every HIP call is checked. On a CPU
    backend the same selection semantics are applied through NumPy and written back with 'from_numpy'.

    With 'sync=True', the Quadrants stream is drained with 'qd.sync()' and the HIP device is synchronized before the
    copy, because the producer may still have writes pending on its own HIP queue (IREE and evoxc use a separate
    stream); the HIP device is synchronized once more after the copy, so that the next Quadrants kernel sees the data.
    Without it, the caller owns this ordering.

    Args:
        dst (qd.Field | qd.Ndarray | qd.Tensor): Batch-last destination buffer.
        src: DLPack producer, or a NumPy array.
        transpose (bool): Whether 'src' is laid out with the batch axis first.
        envs_idx (optional): Batch-axis selection.
        idx (optional): Value-axis selection.
        sync (bool): Whether to synchronize the Quadrants stream and the HIP device around the copy.
    """
    dst_view = view_of(dst)
    if dst_view.ndim > 2:
        _raise_genesis_error(f"Only 1D and 2D buffers can be written from DLPack, got shape {dst_view.shape}.")
    num_envs = dst_view.shape[-1]
    num_values = dst_view.shape[0] if dst_view.ndim == 2 else 1
    is_transposed = transpose and dst_view.ndim == 2
    src_info = dlpack_info(src)
    expected_shape = (num_envs, num_values) if is_transposed else dst_view.shape
    if src_info.shape != expected_shape:
        _raise_genesis_error(
            f"The DLPack source of shape {src_info.shape} cannot be copied into a buffer of shape {dst_view.shape} "
            f"with 'transpose={transpose}'."
        )
    if src_info.dtype != dst_view.dtype:
        _raise_genesis_error(
            f"The DLPack source dtype '{src_info.dtype}' does not match the buffer dtype '{dst_view.dtype}'."
        )
    if not dst_view.is_contiguous:
        _raise_genesis_error(f"Cannot write into the non-contiguous device view of shape {dst_view.shape}.")
    if src_info.strides != _contiguous_strides(src_info.shape):
        _raise_genesis_error(
            f"Only a contiguous DLPack source can be copied, but it has the strides {src_info.strides}."
        )
    backend_name = _runtime_backend()
    if backend_name == "metal":
        _raise_genesis_error("Writing into a Metal buffer from DLPack is not supported by the torch-free bridge.")

    value_runs = _index_runs(_selected_indices(idx, num_values), num_values)
    env_runs = _index_runs(_selected_indices(envs_idx, num_envs), num_envs)
    is_whole = value_runs == ((0, num_values),) and env_runs == ((0, num_envs),)

    if backend_name == "cpu":
        if src_info.device_type != _kDLCPU:
            _raise_genesis_error(
                f"Only a host DLPack source can be copied into a CPU buffer, but the source is on device "
                f"({src_info.device_type}, {src_info.device_id})."
            )
        _copy_host(dst, dst_view, src, value_runs, env_runs, is_transposed, is_whole)
        return

    device = dst_view.device
    if (src_info.device_type, src_info.device_id) != (device.dlpack_device_type, device.index):
        _raise_genesis_error(
            f"The DLPack source on device ({src_info.device_type}, {src_info.device_id}) cannot be copied into "
            f"'{device}'."
        )
    runtime = _hip_runtime()
    dst_ptr = dst_view.data_ptr
    src_ptr = src_info.data_ptr + src_info.byte_offset
    if sync:
        qd.sync()
        runtime.synchronize()
    if is_whole:
        if is_transposed:
            _hip_copy_transposed(dst_ptr, src_ptr, num_values, num_envs, dst_view.itemsize)
        else:
            runtime.copy(dst_ptr, src_ptr, dst_view.nbytes)
    else:
        _hip_copy_selection(
            dst_ptr,
            src_ptr,
            num_values,
            num_envs,
            dst_view.itemsize,
            value_runs,
            env_runs,
            is_transposed,
        )
    if sync:
        runtime.synchronize()


def qd_ptr(value: Any) -> int:
    """Device pointer of a Quadrants field, ndarray or tensor, or of a :class:`DeviceView`."""
    if isinstance(value, DeviceView):
        return value.data_ptr
    if isinstance(value, qd.Ndarray):
        return _qd_program().get_ndarray_data_ptr_as_int(value.arr)
    if isinstance(value, qd.Tensor):
        value = value._unwrap()
        if isinstance(value, qd.Ndarray):
            return _qd_program().get_ndarray_data_ptr_as_int(value.arr)
    info = dlpack_info(value)
    return info.data_ptr + info.byte_offset

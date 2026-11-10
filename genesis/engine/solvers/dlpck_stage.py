"""Staging of a DLPack producer into the device buffers the torch-free kernels read.

A state setter of a torch-free scene receives the values of a state from a machine-learning runtime, which hands them
over as a DLPack producer laid out batch-first, and writes them into Quadrants buffers laid out batch-last, through
kernels that take 'qd.types.ndarray()' arguments. This module is the half of that bridge which turns a producer into
such an argument, so that no solver repeats any of it:

- 'stage_kernel_values', which gives a setter the contiguous device buffer of the dtype and shape its kernel reads.
- 'DlpackStager', the staging buffers a staged write scatters into the state from.
- '_HipAsyncCopies', '_hip_async_copies', '_device_copy_plan', '_copy_after_pending_work' and
  '_copy_transposed_after_pending_work', the stream-ordered device-to-device transfers those two issue, which cost the
  caller no device synchronization.
- '_read_selection', '_contiguous_strides', '_device_copy_shape' and '_host_values', the read-side selection and the
  layout queries the copies are planned from.

The read and sanitize helpers these functions build on ('numpy_dtype', 'to_numpy_values', 'device_tensor_from_numpy',
'read_view', 'is_dlpack_source', 'is_full_env_selection', ...) live in 'genesis.engine.solvers.torch_free', which
imports this module at module scope to re-export its names.
The module-level dependency therefore runs one way only, and the few functions here that need one of those helpers
import it inside the call, which is the only cycle break there is.
Nothing here imports PyTorch.
"""

import ctypes
import functools

import numpy as np

import quadrants as qd

import genesis as gs
from genesis.utils.device_buffer import (
    _kDLCPU,
    _index_runs,
    _selected_indices,
    copy_from_dlpack,
    dlpack_info,
    view_of,
)

__all__ = [
    "DlpackStager",
    "stage_kernel_values",
]

# HIP copy kind of a transfer whose two sides are device pointers.
_HIP_MEMCPY_DEVICE_TO_DEVICE = 3

# The stream Quadrants launches its kernels on: the default stream of the process, which is what its runtime is given
# when a kernel is launched without an explicit stream, as every kernel of the engine is.
_QUADRANTS_STREAM = None


# ------------------------------------------------------------------------------------------------
# Value layouts and read selections
# ------------------------------------------------------------------------------------------------


def _read_selection(index):
    """`index` as 'read_view' reads it: a selection naming no entry selects the empty slice instead of raising.

    A read selecting no entry at all, as the empty index array of a getter asked for zero entries does, names the empty
    slice, which the view reads as the empty array of the shape the torch path returns for that selection. The
    selection is inspected without reading it back from the device unless it lives there, which '_normalize_mask'
    reads anyway.
    """
    # Local import: 'torch_free' re-exports this module at module scope, so the dependency stays one-way.
    from .torch_free import _is_device_selection, to_numpy_values

    if index is None or isinstance(index, (slice, range, int, np.integer)):
        return index
    if isinstance(index, np.ndarray):
        return slice(0, 0) if index.size == 0 else index
    if isinstance(index, (list, tuple)):
        return slice(0, 0) if not len(index) else index
    if _is_device_selection(index):
        return slice(0, 0) if not to_numpy_values(index).size else index
    return index


def _contiguous_strides(shape) -> tuple[int, ...]:
    """Row-major strides, in elements, of a contiguous buffer of the given shape."""
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def _device_copy_shape(value, dtype) -> tuple[int, ...] | None:
    """Shape of the device buffer the bridge copies `value` into without a host roundtrip, None when it takes the host.

    It writes a buffer of at most two dimensions only, which is why the third dimension of a state layout, the
    components of a vector, takes the host path ('stage_kernel_values').
    Both sides share a dtype and no cast is performed, the source must be compact, and a Metal backend has no device
    copy at all.
    A producer that keeps its layout to itself, as a compiled-graph runtime hands its output over, is described by its
    own DLPack capsule: the DLTensor of the capsule holds the dtype, the shape and the strides, a NULL 'strides' field
    meaning the compact row-major layout of that shape, which is the reading 'dlpack_info' gives it.
    A capsule that describes something else than the shape and dtype a device copy takes, which includes a host
    pointer (the host path reads it whatever the backend), and an object whose layout cannot be read at all, are left
    to the host path.
    """
    # Local import: 'torch_free' re-exports this module at module scope, so the dependency stays one-way.
    from .torch_free import numpy_dtype

    if gs.backend == gs.metal:
        return None
    shape = getattr(value, "shape", None)
    strides = getattr(value, "strides", None)
    value_dtype = getattr(value, "dtype", None)
    if isinstance(shape, (tuple, list)) and isinstance(strides, (tuple, list)) and value_dtype is not None:
        shape = tuple(shape)
        if len(shape) > 2 or numpy_dtype(value_dtype) != numpy_dtype(dtype):
            return None
        return shape if tuple(strides) == _contiguous_strides(shape) else None
    # A producer exposing no layout of its own, as a compiled-graph output does, describes it in its capsule alone
    try:
        info = dlpack_info(value)
    except gs.GenesisException:
        return None
    if len(info.shape) > 2 or info.dtype != numpy_dtype(dtype) or info.device_type == _kDLCPU:
        return None
    return info.shape if info.strides == _contiguous_strides(info.shape) else None


def _host_values(value) -> np.ndarray:
    """Host values of `value`, a DLPack producer that no device copy takes, laid out as the kernel reads them.

    A zero-copy device view reads itself back through its own 'to_numpy', which is a single device-to-host transfer.
    Anything else is a DLPack capsule of some runtime, which only NumPy can read, and which a device-resident producer
    refuses: the error then names the forms the setter accepts.
    """
    to_numpy = getattr(value, "to_numpy", None)
    if to_numpy is not None:
        return np.asarray(to_numpy())
    try:
        return np.from_dlpack(value)
    except (BufferError, RuntimeError, TypeError, ValueError) as err:
        gs.raise_exception_from(
            "The values come from a device-resident DLPack producer, which a state setter can only take in the two "
            "layouts it copies on the device, the batch axis leading and at most two dimensions. Pass them as a NumPy "
            "array, a Quadrants buffer or a zero-copy device view instead.",
            err,
        )


def stage_kernel_values(value, dtype):
    """Return `value` as the contiguous device buffer of `dtype` that a kernel takes as an ndarray argument.

    A state setter writes the arrays of a state, which a torch-free getter returns as zero-copy device views
    ('read_view'), and the kernels writing the state take 'qd.types.ndarray()' arguments, which a device view is not.
    Every array therefore reaches those kernels as the device buffer of the dtype and shape the kernel reads:

    - a Quadrants buffer of that dtype already is one, and passes through untouched;
    - a NumPy array, a plain sequence and a 'qd.Field' hold their values on the host, and are uploaded from there (the
      field through 'read_view', its batch axis being the last one);
    - a zero-copy device view and any other DLPack producer are copied device to device when the bridge of
      'genesis.utils.device_buffer' can write them, which it does for at most two dimensions and without casting, and
      through the host otherwise, the state layouts holding the components of a vector as a third dimension. A producer
      that exposes no layout of its own is read off its own DLPack capsule, so a compiled-graph output takes the device
      copy as any other ('_device_copy_shape').

    The device-to-device copy is issued on the stream Quadrants launches its kernels on and is ordered by that stream
    after the kernels launched before it, so staging costs no device synchronization at all (see
    '_copy_after_pending_work'); the copy of the bridge stays the fallback of a layout, a dtype or a backend it cannot
    take, and reports a producer it refuses with its own message.
    Every call stages into a buffer of its own rather than into a cached one, because the arrays of one state share their
    shapes: the velocities and the accelerations of a solver have the same shape and dtype, and a buffer cached per shape
    would make the kernel write the accelerations into the velocities as well.
    """
    # Local import: 'torch_free' re-exports this module at module scope, so the dependency stays one-way.
    from .torch_free import (
        _qd_type,
        device_tensor_from_numpy,
        is_dlpack_source,
        numpy_dtype,
        read_view,
        to_numpy_values,
    )

    if isinstance(value, (qd.Tensor, qd.Ndarray)):
        if numpy_dtype(value.dtype) == numpy_dtype(dtype):
            return value
        return device_tensor_from_numpy(value.to_numpy(), dtype)
    if isinstance(value, qd.Field):
        return device_tensor_from_numpy(to_numpy_values(read_view(value)), dtype)
    if is_dlpack_source(value):
        shape = _device_copy_shape(value, dtype)
        if shape is not None:
            buffer = qd.ndarray(_qd_type(dtype), shape=shape)
            if not _copy_after_pending_work(value, buffer):
                copy_from_dlpack(buffer, value, transpose=False)
            return buffer
        return device_tensor_from_numpy(_host_values(value), dtype)
    return device_tensor_from_numpy(to_numpy_values(value), dtype)


# ------------------------------------------------------------------------------------------------
# Stream-ordered copies
# ------------------------------------------------------------------------------------------------


class _HipAsyncCopies:
    """The asynchronous device-to-device HIP transfer a stream-ordered staging copy issues.

    'genesis.utils.device_buffer' owns the synchronous transfers of a write into a device buffer, and this class binds
    the two asynchronous counterparts, with the same conventions: the library is loaded on first use only, so no HIP
    runtime is touched on a backend that has none, and every call is checked.
    """

    def __init__(self) -> None:
        try:
            self._library = ctypes.CDLL("libamdhip64.so")
        except OSError as e:
            gs.raise_exception_from(
                "The HIP runtime 'libamdhip64.so' is required to stage the values of a DLPack producer on the device, "
                "but it could not be loaded.",
                e,
            )
        self._library.hipMemcpy2DAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._library.hipMemcpy2DAsync.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, function_name: str) -> None:
        if status != 0:
            gs.raise_exception(f"The HIP call '{function_name}' failed with the error code {status}.")

    def copy_2d(
        self, dst_ptr: int, dst_pitch: int, src_ptr: int, src_pitch: int, width: int, height: int
    ) -> None:
        self._check(
            self._library.hipMemcpy2DAsync(
                dst_ptr,
                dst_pitch,
                src_ptr,
                src_pitch,
                width,
                height,
                _HIP_MEMCPY_DEVICE_TO_DEVICE,
                _QUADRANTS_STREAM,
            ),
            "hipMemcpy2DAsync",
        )


@functools.lru_cache(maxsize=1)
def _hip_async_copies() -> _HipAsyncCopies:
    """Asynchronous HIP transfers of the process, bound on first use so that a CPU backend never loads a HIP library."""
    return _HipAsyncCopies()


def _device_copy_plan(src, dst):
    """The two ends of a device-to-device copy, or None when the stream cannot issue it.

    `dst` is the contiguous device buffer of at most two dimensions the copy writes, and the DLPack producer `src` must
    hold the same dtype with a contiguous layout of its own, on the device of `dst`. A producer exposing no layout of
    its own, as a raw capsule does, a backend without the asynchronous transfer, as the CPU and Metal ones, and a buffer
    holding no byte at all are refused, and the caller falls back on 'copy_from_dlpack'.
    """
    if gs.backend != gs.amdgpu:
        return None
    dst_view = view_of(dst)
    if not dst_view.is_contiguous or dst_view.ndim > 2 or not dst_view.nbytes:
        return None
    src_info = dlpack_info(src)
    if src_info.dtype != dst_view.dtype or src_info.strides != _contiguous_strides(src_info.shape):
        return None
    device = dst_view.device
    if (src_info.device_type, src_info.device_id) != (device.dlpack_device_type, device.index):
        return None
    return dst_view, src_info


def _copy_after_pending_work(src, dst) -> bool:
    """Copy the DLPack producer `src` into the contiguous device buffer `dst` of the same shape, without any host wait.

    This is the copy a state setter stages its values with, which it hands to a kernel as the 'qd.types.ndarray()'
    argument filling the state (see 'stage_kernel_values'). The two buffers hold the same values in the same order, so a
    single transfer covers the whole payload, and neither the shape nor the values reach the host.

    Like the transposed copy of the staged setters, the transfer is issued on the stream Quadrants launches its kernels
    on, so it is ordered after every kernel launched before it and before every kernel launched after it: the kernel
    reading the buffer sees the values without any device synchronization, and staging a state drains nothing (see
    '_copy_transposed_after_pending_work').

    Returns False, having issued nothing, when the copy cannot be taken this way: another backend, or a producer the
    bridge does not copy device to device (a layout, a dtype or a device mismatch), so that the caller falls back on
    'copy_from_dlpack', which reports a producer it refuses with its own message.
    """
    plan = _device_copy_plan(src, dst)
    if plan is None:
        return False
    dst_view, src_info = plan
    if src_info.shape != dst_view.shape:
        return False
    shape = dst_view.shape
    itemsize = dst_view.itemsize
    # One row of 'num_cols' values per row of the payload, a one-dimensional payload being a single row of its values
    num_cols = shape[-1]
    num_rows = shape[0] if len(shape) == 2 else 1
    _hip_async_copies().copy_2d(
        dst_view.data_ptr,
        num_cols * itemsize,
        src_info.data_ptr + src_info.byte_offset,
        num_cols * itemsize,
        num_cols * itemsize,
        num_rows,
    )
    return True


def _copy_transposed_after_pending_work(src, dst, envs_idx) -> bool:
    """Copy the DLPack producer `src` of shape (num_envs, num_values) into the batch-last `dst`, without any host wait.

    The producer holds one row per environment and one column per value written, the batch axis first as a
    machine-learning runtime emits it, and `dst` is the contiguous '(num_values, num_envs)' staging buffer a staged
    write kernel reads; `envs_idx` selects the environments to write, and the environments it does not name are left
    untouched. Each destination row is one 'hipMemcpy2DAsync' transfer, so the copy stays on the device and no host
    roundtrip of the payload takes place.

    The transfers are issued on the stream Quadrants launches its kernels on, which is the default stream of the
    process: 'Kernel.__call__' passes no stream to the runtime unless the caller gives one, and no caller in Genesis
    ever does. A transfer of that stream is ordered after every Quadrants kernel launched before it, and every kernel
    launched after it is ordered after the transfer, so the kernels reading the staging buffer see the values without
    any device synchronization, and the transfer of the previous step no longer drains the stream.

    Returns False, having issued nothing, when the copy cannot be taken this way: another backend, a producer the
    bridge does not copy device to device (a layout, a dtype or a device mismatch), so that the caller falls back on
    'copy_from_dlpack', which reports a producer it refuses with its own message.
    """
    plan = _device_copy_plan(src, dst)
    if plan is None:
        return False
    dst_view, src_info = plan
    if dst_view.ndim != 2:
        return False
    num_values, num_envs = dst_view.shape
    if src_info.shape != (num_envs, num_values):
        return False
    copies = _hip_async_copies()
    dst_ptr = dst_view.data_ptr
    src_ptr = src_info.data_ptr + src_info.byte_offset
    itemsize = dst_view.itemsize
    for i_v in range(num_values):
        for env_start, env_count in _index_runs(_selected_indices(envs_idx, num_envs), num_envs):
            copies.copy_2d(
                dst_ptr + (i_v * num_envs + env_start) * itemsize,
                itemsize,
                src_ptr + (env_start * num_values + i_v) * itemsize,
                num_values * itemsize,
                itemsize,
                env_count,
            )
    return True


# ------------------------------------------------------------------------------------------------
# Staging buffers
# ------------------------------------------------------------------------------------------------


class DlpackStager:
    """Staging buffers that turn a DLPack producer into the contiguous batch-last values a kernel scatters into a field.

    A Quadrants field is contiguous along its first axis, so its batch-last view is not contiguous, and the bridge of
    'genesis.utils.device_buffer' refuses to write into it.
    The stager copies the producer into a contiguous ndarray instead, device to device, which the staged kernels of
    'genesis.engine.solvers.rigid.abd.torch_free_kernels' then scatter into the state.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[int, int], "qd.Ndarray"] = {}

    def stage(self, src, num_values: int, num_envs: int, *, envs_idx=None) -> "qd.Ndarray":
        """Write the DLPack producer `src` of shape (num_envs, num_values) into the staging buffer of that shape.

        The buffer is allocated on first use and cached per shape, so a setter stages on every step without allocating.
        `src` holds one row per environment of the solver, in environment order, and `envs_idx` only selects the columns
        of the buffer that receive the values of those environments, in place, as the write into a state leaves them.
        The columns `envs_idx` does not select hold whatever the buffer held before, and must never be read.
        The copy runs device to device and is stream-ordered: it is issued on the stream Quadrants launches its kernels
        on, so the staged write kernel that reads the buffer sees the values without any synchronization, and a write
        costs no host wait at all (see '_copy_transposed_after_pending_work'). It moves no data to the host.
        The dtype of `src` must be that of the buffer, 'gs.qd_float' for a float state, since the bridge casts nothing
        and reports a mismatch with its own message.
        A selection covering every environment in order is dropped, so that the copy takes the whole buffer and no
        selection reaches the host.
        A masked write still needs its selection on the host, where the copy splits into contiguous runs, so a selection
        given on the device costs one host read of its indices, and a backend without a stream-ordered device copy, as
        the CPU one, reads the buffer back once more to patch it. A device-resident selection holding one 0/1 flag per
        environment is the mask form of a torch-free scene, which becomes the indices it flags (see
        'sanitize_index_values').
        """
        # Local import: 'torch_free' re-exports this module at module scope, so the dependency stays one-way.
        from .torch_free import (
            _is_device_selection,
            _resolve_flag_mask,
            is_full_env_selection,
            to_numpy_values,
        )

        buffer = self._buffers.get((num_values, num_envs))
        if buffer is None:
            buffer = qd.ndarray(gs.qd_float, shape=(num_values, num_envs))
            self._buffers[(num_values, num_envs)] = buffer
        selection = envs_idx
        if _is_device_selection(selection):
            selection = _resolve_flag_mask(to_numpy_values(selection), num_envs)
        if is_full_env_selection(selection, num_envs):
            selection = None
        if not _copy_transposed_after_pending_work(src, buffer, selection):
            copy_from_dlpack(buffer, src, transpose=True, envs_idx=selection)
        return buffer

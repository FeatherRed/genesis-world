"""Helpers the solver base layer uses in place of PyTorch.

The solvers only need Quadrants and NumPy to run the physics: PyTorch enters the picture for the differentiable
simulation and for the zero-copy views, neither of which exists without it (see 'genesis.utils.torch_compat').
This module holds what replaces it on the torch-free path, so that no solver repeats any of it:

- 'broadcast_values' and 'broadcast_tensor', the numpy and device counterparts of 'genesis.utils.misc.broadcast_tensor'.
- 'sanitize_index_values' and 'sanitize_io_variables', the numpy counterpart of the index sanitizing of a getter.
- 'device_tensor_from_numpy', 'index_field' and 'qd_field', the allocation of a kernel argument and of a solver buffer.
- 'read_view', 'owned_copy', 'owned_view', 'is_dlpack_source', 'stage_kernel_values' and 'DlpackStager', the
  device-buffer bridge of 'genesis.utils.device_buffer'.
- 'envs_selection' and 'mps_synchronize', which read the environments of a call and flush the stream after a write.

Nothing here imports PyTorch except through 'torch_compat', and only 'mps_synchronize' touches it at all.
The batch (environment) dimension is the LAST one of a Quadrants buffer, as everywhere in the engine.
"""

from itertools import combinations

import numpy as np

import quadrants as qd

from quadrants.lang.util import to_numpy_type

import genesis as gs
from genesis.utils.device_buffer import (
    _DLPACK_CAPSULE_NAME,
    _DLPACK_VERSIONED_CAPSULE_NAME,
    DeviceView,
    _normalize_mask,
    _py_capsule_is_valid,
    copy_from_dlpack,
    view_of,
)
from genesis.utils.torch_compat import HAS_TORCH, torch

__all__ = [
    "DlpackStager",
    "broadcast_tensor",
    "broadcast_values",
    "device_tensor_from_numpy",
    "drop_repeated_values",
    "envs_flag_mask",
    "envs_index_tensor",
    "envs_selection",
    "index_field",
    "is_dlpack_source",
    "is_full_env_selection",
    "kernel_copy_values",
    "last_occurrence",
    "mps_synchronize",
    "numpy_dtype",
    "owned_copy",
    "owned_view",
    "qd_field",
    "read_view",
    "sanitize_index_values",
    "sanitize_io_variables",
    "stage_kernel_values",
    "to_numpy_values",
    "write_from_dlpack",
]

# The Quadrants primitive type each numpy dtype an accessor uses is uploaded as, Quadrants having no ndarray of its own
# for anything else.
_QD_TYPES = {
    np.dtype(np.float32): qd.f32,
    np.dtype(np.float64): qd.f64,
    np.dtype(np.int32): qd.i32,
    np.dtype(np.int64): qd.i64,
    np.dtype(np.bool_): qd.u1,
}


# ------------------------------------------------------------------------------------------------
# Dtypes and host values
# ------------------------------------------------------------------------------------------------


def numpy_dtype(dtype) -> np.dtype:
    """Numpy dtype of `dtype`, given as a numpy dtype, a PyTorch dtype or a Quadrants primitive type."""
    if isinstance(dtype, np.dtype):
        return dtype
    try:
        return np.dtype(dtype)
    except TypeError:
        # The configured dtypes 'gs.tc_*' hold a torch dtype even on a torch-free path, so the class of the dtype
        # decides, not the flag of this module, which an emulation switches off while PyTorch stays importable
        if isinstance(dtype, torch.dtype):
            return np.dtype(str(dtype).rsplit(".", 1)[-1])
        return np.dtype(to_numpy_type(dtype))


def to_numpy_values(value) -> np.ndarray:
    """Host numpy view of `value`, a numpy array, a Quadrants buffer or a zero-copy device view."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, DeviceView):
        return np.asarray(value.to_numpy())
    if isinstance(value, (qd.Tensor, qd.Field, qd.Ndarray)):
        # A host copy, which the index selections of an accessor can afford: none of them is large
        return np.asarray(value.to_numpy())
    return np.asarray(value)


# ------------------------------------------------------------------------------------------------
# Allocation
# ------------------------------------------------------------------------------------------------


def qd_field(dtype, shape):
    """Allocate a zero-initialized Quadrants field of `dtype` and `shape`, which a solver keeps as a device buffer."""
    return qd.field(dtype, shape=tuple(shape))


def device_tensor_from_numpy(values, dtype):
    """Upload `values` as the C-contiguous Quadrants ndarray of `dtype` that a kernel reads as 'qd.types.ndarray()'.

    A buffer holding no entry at all stays a host array: Quadrants has no zero-length ndarray, and a kernel takes a
    NumPy array as it takes an ndarray, so the loops over a selection of no entry run none of their iterations. The
    empty selection of an environment or of an entry of the state is thus a well-defined no-op rather than an error.
    """
    array = np.ascontiguousarray(values, dtype=numpy_dtype(dtype))
    if not array.size:
        return array
    tensor = qd.ndarray(_qd_type(dtype), shape=array.shape)
    tensor.from_numpy(array)
    return tensor


def _qd_type(dtype):
    """Quadrants primitive type matching the numpy dtype `dtype` normalizes to."""
    array_dtype = numpy_dtype(dtype)
    if array_dtype not in _QD_TYPES:
        gs.raise_exception(f"Cannot allocate a device buffer holding values of dtype '{array_dtype}'.")
    return _QD_TYPES[array_dtype]


# ------------------------------------------------------------------------------------------------
# Value broadcasting
# ------------------------------------------------------------------------------------------------


def broadcast_values(values, dtype, shape, dim_names=None) -> np.ndarray:
    """Broadcast `values` to a C-contiguous numpy array of `dtype` and `shape`, as 'broadcast_tensor' does.

    The expansion rules are those of 'genesis.utils.misc.broadcast_tensor', of which this is the numpy analogue: a
    scalar gains the trailing dimensions, a value whose trailing dimensions match those of `shape` gains the leading
    ones, otherwise its dimensions spread over the dimensions of `shape` that accept them (the last combination
    winning), a dimension of size 1 repeats, and a `-1` in `shape` stands for the size it already has. The values are
    cast to `dtype` first, which is a numpy dtype, a PyTorch dtype or a Quadrants primitive type.
    """
    shape = tuple(shape)
    if dim_names is None:
        dim_names = ("",) * len(shape)

    if values is None:
        if any(size == -1 for size in shape):
            gs.raise_exception(
                "Tensor not pre-allocated and expected shape not fully specified but allocation is not skipped."
            )
        return np.empty(shape, dtype=numpy_dtype(dtype))

    values_ = np.asarray(to_numpy_values(values), dtype=numpy_dtype(dtype))
    values_shape = values_.shape
    values_ndim = len(values_shape)
    shape_ndim = len(shape)

    # Expand current values shape with extra dims of size 1 if necessary before expanding to the expected shape
    if values_ndim == 0:
        values_ = values_[None]
    elif values_ndim < shape_ndim and not all(
        [size == expected or expected == -1 for size, expected in zip(values_shape, shape[-values_ndim:])]
    ):
        # Try expanding first dimensions if priority
        for dims_valid in tuple(combinations(range(shape_ndim), values_ndim))[::-1]:
            curr_idx = 0
            expanded_shape = []
            for i in range(shape_ndim):
                if i in dims_valid:
                    dim, size = values_.shape[curr_idx], shape[i]
                    if dim == size or dim == 1 or size == -1:
                        expanded_shape.append(dim)
                        curr_idx += 1
                    else:
                        break
                else:
                    expanded_shape.append(1)
            else:
                if curr_idx == values_ndim:
                    values_ = values_.reshape(expanded_shape)
                    break
    elif values_ndim > shape_ndim:
        gs.raise_exception(f"Invalid input shape: {values_shape}. Expecting at most {shape_ndim}D tensor.")

    if values_.ndim > shape_ndim:
        gs.raise_exception(f"Invalid input shape: {values_shape}. Expecting at most {shape_ndim}D tensor.")
    # A '-1' stands for the size the value already has in that dimension, which numpy has no notation for, and names no
    # dimension of the value when the value has fewer of them, as torch refuses it too
    n_missing_dims = shape_ndim - values_.ndim
    expanded_shape = []
    for i, size in enumerate(shape):
        if size != -1:
            expanded_shape.append(size)
        elif i < n_missing_dims:
            _raise_invalid_shape(values_, shape, dim_names)
        else:
            expanded_shape.append(values_.shape[i - n_missing_dims])
    try:
        return np.ascontiguousarray(np.broadcast_to(values_, tuple(expanded_shape)))
    except ValueError:
        _raise_invalid_shape(values_, shape, dim_names)


def _raise_invalid_shape(values: np.ndarray, shape: tuple[int, ...], dim_names):
    """Raise the error message 'broadcast_tensor' gives for values that fit none of the expansions of `shape`."""
    msg_err = f"Invalid input shape: {values.shape}."
    msg_infos: list[str] = []
    for i, name in enumerate(dim_names):
        size = shape[i]
        if size > 0 and i < values.ndim and (dim := values.shape[i]) != 1 and dim != size:
            if name:
                msg_infos.append(f"Dimension {i} consistent with len({name})={size}")
            else:
                msg_infos.append(f"Dimension {i} consistent with required size {size}")
    if msg_infos:
        msg_err += f" {' & '.join(msg_infos)}."
    else:
        msg_err += f" Expected shape: {tuple(shape)}."
    gs.raise_exception(msg_err)


def broadcast_tensor(values, dtype, shape, dim_names=None):
    """Broadcast `values` to `shape` and upload them as the device tensor a kernel reads as an argument."""
    return device_tensor_from_numpy(broadcast_values(values, dtype, shape, dim_names), dtype)


# ------------------------------------------------------------------------------------------------
# Index sanitizing
# ------------------------------------------------------------------------------------------------


def _is_device_selection(index) -> bool:
    """Whether the entries of `index` live in device memory, so that only a host read can inspect them."""
    return isinstance(index, (DeviceView, qd.Tensor, qd.Field, qd.Ndarray))


def _resolve_flag_mask(values: np.ndarray, max_size: int) -> np.ndarray:
    """Indices a device-resident selection of one flag per entry flags, as 'mask_indices' reads such a mask.

    A torch-free scene states the mask of the entries a call selects as a device buffer holding one flag for each of
    them, which 'Solver.get_error_envs_mask' returns and 'Simulator.clear_errno' reads back. A non-zero flag selects
    its entry, exactly as a true entry of a boolean mask does. A selection of another shape, or holding a value other
    than 0 and 1, names its entries itself and is returned unchanged.
    """
    if values.ndim != 1 or values.size != max_size or values.dtype.kind not in "iu":
        return values
    if not np.isin(values, (0, 1)).all():
        return values
    return np.nonzero(values)[0]


def envs_flag_mask(envs_idx, num_envs: int) -> "np.ndarray | None":
    """The boolean mask over every environment of the scene that `envs_idx` states, or None if it names its entries.

    A boolean mask states one flag per environment, be it on the host or on the device, and so does a device-resident
    buffer of one 0/1 flag per environment, which is the mask form of a torch-free scene (see '_resolve_flag_mask').
    An index collection names the environments it selects whatever its values, as PyTorch reads it.
    """
    if not isinstance(envs_idx, np.ndarray) and not _is_device_selection(envs_idx):
        return None
    values = to_numpy_values(envs_idx)
    if values.ndim != 1 or values.size != num_envs:
        return None
    if values.dtype == np.bool_:
        return values
    if _is_device_selection(envs_idx) and values.dtype.kind in "iu" and np.isin(values, (0, 1)).all():
        return values.astype(bool)
    return None


def sanitize_index_values(index, expected_size: int, max_size: int, dim: int, name: str) -> np.ndarray:
    """Sanitize one index selection into the C-contiguous int32 values a kernel iterates over.

    The accepted forms and the errors raised are those of 'genesis.utils.misc.sanitize_index', of which this is the
    numpy analogue: a negative entry wraps around `max_size`, a boolean mask must have shape '(max_size,)', and the
    selection ends up one-dimensional, of length `expected_size` when that is not -1. A device-resident selection of
    one 0/1 flag per entry is the mask a torch-free scene gives, and becomes the indices it flags.
    """
    is_bool_mask = False
    is_negative_wrap_required = False
    if index is None:
        values = np.arange(max_size, dtype=np.int64)
    elif isinstance(index, slice):
        values = np.arange(*index.indices(max_size), dtype=np.int64)
    elif isinstance(index, (int, np.integer)):
        values = np.array([index + max_size if -max_size <= index < 0 else index], dtype=np.int64)
    elif isinstance(index, range):
        if not len(index):
            values = np.empty((0,), dtype=np.int64)
        elif -max_size <= index[0] < 0 and -max_size <= index[-1] < 0:
            values = np.arange(index.start + max_size, index.stop + max_size, index.step, dtype=np.int64)
        else:
            values = np.asarray(list(index), dtype=np.int64)
            is_negative_wrap_required = True
    else:
        is_device_index = _is_device_selection(index)
        values = to_numpy_values(index)
        if is_device_index:
            # The mask form of a torch-free scene holds one flag per entry, which a kernel cannot read as such
            values = _resolve_flag_mask(values, max_size)
        # See 'broadcast_values' for arrays with negative strides
        if any(stride < 0 for stride in values.strides):
            values = values.copy()
        # A sequence of booleans holds integers, as it does for 'sanitize_index', which only a boolean array masks with
        if values.dtype == np.bool_ and not isinstance(index, (list, tuple)):
            is_bool_mask = True
        else:
            is_negative_wrap_required = True

    if is_bool_mask:
        if values.ndim != 1 or len(values) != max_size:
            gs.raise_exception(f"Boolean masks for `{name}` must have shape ({max_size},).")
        values = np.nonzero(values)[0]
    else:
        try:
            values = np.asarray(values, dtype=np.int64)
        except (TypeError, ValueError) as err:
            gs.raise_exception_from(f"Expecting integer indices for `{name}`.", err)
        if values.ndim == 0:
            values = values[None]
        elif values.ndim > 1:
            dim_info = f" `{name}`" if name else ""
            gs.raise_exception(f"Invalid shape: {values.shape}. Expecting 0D or 1D tensor for {dim}-th index{dim_info}.")
        if expected_size != -1 and expected_size != len(values):
            dim_info = f" `{name}`" if name else ""
            gs.raise_exception(
                f"Invalid shape: {values.shape}. Expecting 1D tensor of length {expected_size} for {dim}-th "
                f"index{dim_info}."
            )
        if is_negative_wrap_required:
            values = np.where((values < 0) & (values >= -max_size), values + max_size, values)

    return np.ascontiguousarray(values, dtype=numpy_dtype(gs.tc_int))


def index_field(index, size: int, name: str = "index"):
    """Sanitize the `index` selection of `size` entries and upload it as the device tensor a kernel iterates over."""
    return device_tensor_from_numpy(sanitize_index_values(index, -1, size, 0, name), gs.tc_int)


def envs_selection(scene, envs_idx) -> np.ndarray:
    """Host values of the environments a call selects, read back from the scene only when the caller names some."""
    if envs_idx is None:
        return np.arange(scene._B, dtype=numpy_dtype(gs.tc_int))
    return np.ascontiguousarray(to_numpy_values(scene._sanitize_envs_idx(envs_idx)), dtype=numpy_dtype(gs.tc_int))


def envs_index_tensor(scene, envs_idx):
    """Device tensor of the environments a call selects, as the kernel iterating over them reads it.

    None is returned when the selection names no environment at all: Quadrants has no zero-length buffer, and a write
    over no environment leaves every one of them as it is, so the caller returns without launching anything.
    """
    values = sanitize_index_values(envs_idx, -1, scene._B, 0, "envs_idx")
    if not values.size:
        return None
    return device_tensor_from_numpy(values, gs.tc_int)


# ------------------------------------------------------------------------------------------------
# Repeated indices
# ------------------------------------------------------------------------------------------------


def last_occurrence(index: np.ndarray) -> np.ndarray:
    """Position of the last entry holding the same index as each entry, along the last axis of `index`.

    The positions are those of 'genesis.utils.misc.get_last_occurrence', which is the torch counterpart computed on
    the device; this one runs on the host, where the accessors of a torch-free scene have their values anyway.
    """
    n_entries = index.shape[-1]
    order = np.argsort(index, axis=-1, kind="stable")
    index_sorted = np.take_along_axis(index, order, axis=-1)
    is_last_sorted = np.ones_like(index_sorted, dtype=bool)
    is_last_sorted[..., :-1] = index_sorted[..., 1:] != index_sorted[..., :-1]
    # The sort is stable, so the last entry of each run of equal indices is their last occurrence
    positions = np.broadcast_to(np.arange(n_entries), order.shape)
    run_last = np.where(is_last_sorted, positions, n_entries)
    run_last = np.minimum.accumulate(run_last[..., ::-1], axis=-1)[..., ::-1]
    positions_of_last_occurrence = np.empty_like(order)
    np.put_along_axis(positions_of_last_occurrence, order, np.take_along_axis(order, run_last, axis=-1), axis=-1)
    return positions_of_last_occurrence


def drop_repeated_values(index_input, index: np.ndarray, tensors: tuple, axis: int, max_size: int):
    """Keep the last occurrence of every index given several times, as a write leaves it, and the matching values.

    The counterpart of 'genesis.utils.misc.drop_repeated_index': an index given one row per environment keeps every
    entry, the earlier occurrences of a row becoming -1, which the kernels skip; an index given as a sequence is
    compacted along its last axis, and so are the values along `axis`. A boolean mask, on the host or on the device,
    names each entry at most once and is left as it is, its positions in the mask saying nothing of the rows of the
    values, which hold one row per selected entry.
    """
    tensors = tuple(tensors)
    if _is_device_selection(index_input):
        # The repeated entries of a device-resident selection are found on the host, where a mask is read back anyway
        index_input = _resolve_flag_mask(to_numpy_values(index_input), max_size)
    if not isinstance(index_input, (list, tuple, np.ndarray)):
        return index, tensors
    n_dims = np.ndim(index_input)
    if n_dims == 0:
        # A scalar names one entry, which cannot repeat
        return index, tensors
    if n_dims > 1:
        is_last = last_occurrence(index) == np.arange(index.shape[-1])
        return np.where(is_last, index, -1), tensors
    index_given = np.asarray(to_numpy_values(index_input))
    if index_given.dtype == np.bool_:
        return index, tensors
    index_row = np.mod(index_given.reshape(-1), max_size)
    is_last = last_occurrence(index_row) == np.arange(index_row.shape[-1])
    positions = np.nonzero(is_last)[0]
    if len(positions) == len(is_last):
        return index, tensors
    # The positions are those of the entries of the selection, one row of the values per entry: a selection holding
    # fewer rows than it names entries, as one covering a subset of the environments does, has no entry to compact
    if index.shape[-1] != len(is_last) or any(
        tensor is not None and tensor.shape[axis] != len(is_last) for tensor in tensors
    ):
        gs.raise_exception(
            f"Cannot keep the last occurrence of the repeated indices {index_given.reshape(-1).tolist()}, which name "
            f"{len(is_last)} entries where the selection and its values hold {index.shape[-1]} of them."
        )
    values_kept = tuple([None if tensor is None else np.take(tensor, positions, axis=axis) for tensor in tensors])
    return np.take(index, positions, axis=-1), values_kept


# ------------------------------------------------------------------------------------------------
# Accessor state I/O
# ------------------------------------------------------------------------------------------------


def _values_shape(values) -> "tuple[int, ...] | None":
    """Shape of `values` as far as it is declared, read without touching their entries."""
    shape = getattr(values, "shape", None)
    if shape is not None:
        return tuple(shape)
    if isinstance(values, (list, tuple)):
        return tuple(np.shape(values))
    return None


def select_envs_values(values, envs_mask, num_selected: int, expected_shape, dim_names):
    """Values written to the environments a mask selects, given one row per selected or per existing environment.

    A masked write takes its values one row per selected environment, as 'set_qpos' does on the torch path, and one row
    per environment of the scene, out of which the rows the mask flags are the ones written, as 'masked_scatter_'
    reads them. A single row, and the values of a form without a leading row at all, spread over every selected
    environment; a batch of rows of another length is refused here with the forms a masked write accepts.
    """
    if values is None or envs_mask is None or num_selected == len(envs_mask):
        return values
    values_shape = _values_shape(values)
    if values_shape is None or not values_shape:
        # A scalar, or a container NumPy reads one of: the broadcast handles its shape
        return values
    if values_shape[0] in (1, num_selected) or len(values_shape) != len(expected_shape):
        # A single row, or values stated without a leading row per environment, which the broadcast spreads
        return values
    if values_shape[0] != len(envs_mask):
        gs.raise_exception(
            f"Invalid input shape: {values_shape}. A masked write takes values of shape "
            f"(len(envs_idx), *element_shape) = {tuple(expected_shape)}, one row per environment of the scene "
            f"({len(envs_mask)}, *element_shape) out of which the mask selects the rows written, (1, *element_shape), "
            f"or (*element_shape)."
        )
    return np.ascontiguousarray(np.asarray(to_numpy_values(values))[envs_mask])


def sanitize_io_variables(
    scene,
    tensor,
    inputs_idx,
    input_size,
    idx_name,
    envs_idx=None,
    element_shape=(),
    *,
    dtype=None,
    batched=True,
    batched_idx=False,
    skip_allocation=False,
    accumulate=False,
):
    """Torch-free counterpart of 'Solver._sanitize_io_variables', with the same arguments and semantics.

    The values are broadcast to the numpy array of the shape the kernel filling or reading them expects, and uploaded
    as the device tensor that kernel takes. The indices are uploaded as int32 device tensors, except `envs_idx_` of a
    call that is not batched: no environment is selected then, Quadrants having no zero-length buffer, so it is None,
    which every such caller discards already. A selection naming no environment or no entry at all is a no-op: its
    buffers hold no entry, which the loops of the kernels skip (see 'device_tensor_from_numpy').

    The values a masked write takes are those of the torch path: one row per selected environment, one row per
    environment of the scene (out of which the mask flags the rows written), a single row, or none at all, which the
    last three spread over every selected environment (see 'select_envs_values').
    """
    if dtype is None:
        dtype = gs.tc_float
    n_envs = scene.n_envs
    is_overwrite = tensor is not None and not accumulate
    envs_idx_ = envs_selection(scene, envs_idx) if batched else None
    # The mask a selection of environments states, which the values of a masked write are reduced by
    envs_mask = envs_flag_mask(envs_idx, n_envs) if batched and n_envs > 0 else None

    if batched and batched_idx and np.ndim(inputs_idx) == 2:
        dim_names = ("envs_idx", idx_name, *["" for _ in element_shape])
        inputs_values = broadcast_values(inputs_idx, gs.tc_int, (len(envs_idx_), -1), dim_names[:2])
        # Negative indices wrap as in sanitize_index, before the repeated ones are dropped
        inputs_values = np.where(
            (inputs_values < 0) & (inputs_values >= -input_size), inputs_values + input_size, inputs_values
        )
        tensor_values = None
        if tensor is not None or not skip_allocation:
            tensor_values = broadcast_values(
                select_envs_values(tensor, envs_mask, len(envs_idx_), (*inputs_values.shape, *element_shape), dim_names),
                dtype,
                (*inputs_values.shape, *element_shape),
                dim_names,
            )
        if is_overwrite:
            envs_idx_, (tensor_values, inputs_values) = drop_repeated_values(
                envs_idx, envs_idx_, (tensor_values, inputs_values), axis=0, max_size=n_envs
            )
            inputs_values, (tensor_values,) = drop_repeated_values(
                inputs_idx, inputs_values, (tensor_values,), 1, input_size
            )
        return (
            _device_values(tensor_values, dtype),
            device_tensor_from_numpy(inputs_values, gs.tc_int),
            device_tensor_from_numpy(envs_idx_, gs.tc_int),
        )

    if n_envs == 0 or not batched:
        inputs_values = sanitize_index_values(inputs_idx, -1, input_size, 0, idx_name)
        tensor_values = None
        if tensor is not None or not skip_allocation:
            tensor_values = broadcast_values(
                tensor, dtype, (len(inputs_values), *element_shape), (idx_name, *["" for _ in element_shape])
            )
    else:
        inputs_values = sanitize_index_values(inputs_idx, -1, input_size, 1, idx_name)
        tensor_values = None
        if tensor is not None or not skip_allocation:
            tensor_values = broadcast_values(
                select_envs_values(
                    tensor,
                    envs_mask,
                    len(envs_idx_),
                    (len(envs_idx_), len(inputs_values), *element_shape),
                    ("envs_idx", idx_name, *["" for _ in element_shape]),
                ),
                dtype,
                (len(envs_idx_), len(inputs_values), *element_shape),
                ("envs_idx", idx_name, *["" for _ in element_shape]),
            )
        if is_overwrite:
            envs_idx_, (tensor_values,) = drop_repeated_values(
                envs_idx, envs_idx_, (tensor_values,), axis=0, max_size=n_envs
            )
    if is_overwrite:
        axis = 1 if n_envs > 0 and batched else 0
        inputs_values, (tensor_values,) = drop_repeated_values(
            inputs_idx, inputs_values, (tensor_values,), axis, input_size
        )

    if batched_idx:
        # The kernels indexing the entries per environment read one row of indices for each of them
        inputs_values = np.ascontiguousarray(np.broadcast_to(inputs_values[None], (len(envs_idx_), *inputs_values.shape)))
        if tensor_values is not None and n_envs == 0:
            tensor_values = tensor_values[None]

    return (
        _device_values(tensor_values, dtype),
        device_tensor_from_numpy(inputs_values, gs.tc_int),
        None if envs_idx_ is None else device_tensor_from_numpy(envs_idx_, gs.tc_int),
    )


def _device_values(values, dtype):
    """Upload the sanitized values, or keep them unallocated when the caller asked for no allocation."""
    return None if values is None else device_tensor_from_numpy(values, dtype)


# ------------------------------------------------------------------------------------------------
# Device-buffer bridge
# ------------------------------------------------------------------------------------------------


def mps_synchronize() -> None:
    """Synchronize the torch stream after a zero-copy write, which only the Metal backend needs."""
    if HAS_TORCH and gs.backend == gs.metal:
        torch.mps.synchronize()


def is_full_env_selection(selection, num_envs: int) -> bool:
    """Whether `selection` names every environment of the solver, in their own order, as no selection at all does.

    `None` selects every environment, and so does any selection that sanitizes to 'range(num_envs)', whatever the form
    it is given in: a slice, a range, an integer, a sequence, an array, a boolean mask, on the host or on the device.
    The order matters, because a copy maps the row of the source of an environment onto the column of that same
    environment: a permutation of every environment is not a full selection.
    A selection the sanitizing of a write refuses is not full, and the copy raises that error itself.
    """
    if selection is None:
        return True
    if isinstance(selection, slice):
        return selection.indices(num_envs) == (0, num_envs, 1)
    try:
        values = sanitize_index_values(selection, -1, num_envs, 0, "envs_idx")
    except gs.GenesisException:
        return False
    return bool(np.array_equal(values, np.arange(num_envs, dtype=values.dtype)))


def is_dlpack_source(value) -> bool:
    """Whether `value` is a DLPack producer, which a copy takes device to device rather than through the host.

    A producer implements the DLPack protocol itself, as a torch tensor and a zero-copy device view do, or carries a
    DLPack capsule, which is what a compiled graph hands over.
    The 'qd.Tensor' and 'qd.Ndarray' buffers export such a capsule through 'to_dlpack', without implementing the
    protocol's '__dlpack__'.
    A 'qd.Field' stays a host source, its batch-last layout being the one a copy writes into rather than reads from.
    A NumPy array and a plain sequence are host sources as well, which the accessors broadcast and upload instead.
    """
    if isinstance(value, np.ndarray):
        return False
    if isinstance(value, (qd.Tensor, qd.Ndarray)):
        return True
    if _py_capsule_is_valid(value, _DLPACK_CAPSULE_NAME):
        return True
    if _py_capsule_is_valid(value, _DLPACK_VERSIONED_CAPSULE_NAME):
        return True
    return hasattr(value, "__dlpack__")


@qd.kernel
def kernel_copy_values(dst: qd.types.ndarray(), src: qd.types.ndarray()):
    """Copy every value of one device buffer into another of the same layout, without leaving the device."""
    for I in qd.grouped(dst):
        dst[I] = src[I]


def owned_copy(buffer) -> "qd.Ndarray":
    """Copy every value of `buffer` into a device buffer this call allocates, and return that buffer.

    The copy runs in a kernel, device to device, on every backend: no value ever reaches the host, and the source may be
    a field, a member of a struct, an ndarray or a tensor. The destination mirrors the element layout of the source, a
    vector or a matrix element included, so that a view of it reads as a view of the source (see 'read_view').
    """
    raw = buffer._unwrap() if isinstance(buffer, qd.Tensor) else buffer
    dtype = raw.dtype
    element_shape = tuple(getattr(raw, "element_shape", ()) or ())
    if len(element_shape) == 1:
        dtype = qd.types.vector(element_shape[0], dtype)
    elif len(element_shape) == 2:
        dtype = qd.types.matrix(*element_shape, dtype)
    elif element_shape:
        gs.raise_exception(
            f"A torch-free state cannot copy a buffer of elements shaped {element_shape} without PyTorch."
        )
    copy = qd.ndarray(dtype, shape=tuple(raw.shape))
    kernel_copy_values(copy, raw)
    return copy


def owned_view(buffer, envs_idx=None, idx=None):
    """Zero-copy view of an owned device copy of `buffer`, in the layout 'read_view' gives.

    Unlike 'read_view', the view is not a window on the live buffer: it follows the copy 'owned_copy' made of it, so it
    keeps the values it was read from while the buffer goes on changing. A state snapshot reads through it, while the
    getters of a torch-free scene read through 'read_view', which is what its hot paths can afford.
    """
    return read_view(owned_copy(buffer), envs_idx, idx)


def read_view(field, envs_idx=None, idx=None):
    """Zero-copy device view of `field`, with the batch axis leading and the given selections applied.

    The batch axis is the last one of the shape the field declares, which precedes the dimensions of a vector element, so
    a buffer of vectors is transposed on that axis rather than on its last physical one, as 'qd_to_torch' does.
    'envs_idx' selects the batch axis and 'idx' the element axis, both after the transposition.
    The selection names every axis of the transposed view, since a partial one would drop the axes it leaves out of the
    shape and the strides of the view, which a DLPack consumer reads.
    """
    if len(field.shape) <= 1:
        return view_of(field, row_mask=_read_selection(idx), col_mask=_read_selection(envs_idx))
    view = DeviceView(field).movedim(len(field.shape) - 1, 0)
    selection = _normalize_mask((_read_selection(envs_idx), _read_selection(idx)), True)
    return view[selection + (slice(None),) * (view.ndim - len(selection))]


def _read_selection(index):
    """`index` as 'read_view' reads it: a selection naming no entry selects the empty slice instead of raising.

    A read selecting no entry at all, as the empty index array of a getter asked for zero entries does, names the empty
    slice, which the view reads as the empty array of the shape the torch path returns for that selection. The
    selection is inspected without reading it back from the device unless it lives there, which '_normalize_mask'
    reads anyway.
    """
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


def _is_copyable_on_device(value, dtype) -> bool:
    """Whether the bridge can copy `value` into a device buffer of `dtype` without a host roundtrip.

    It writes a buffer of at most two dimensions only, which is why the third dimension of a state layout, the
    components of a vector, takes the host path ('stage_kernel_values').
    Both sides share a dtype and no cast is performed, the source must be contiguous, and a Metal backend has no device
    copy at all.
    A producer exposing no layout of its own, as a raw DLPack capsule does, is left to the host path as well.
    """
    shape = getattr(value, "shape", None)
    strides = getattr(value, "strides", None)
    value_dtype = getattr(value, "dtype", None)
    if not isinstance(shape, (tuple, list)) or not isinstance(strides, (tuple, list)) or len(shape) > 2:
        return False
    if value_dtype is None or gs.backend == gs.metal:
        return False
    return numpy_dtype(value_dtype) == numpy_dtype(dtype) and tuple(strides) == _contiguous_strides(tuple(shape))


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
      through the host otherwise, the state layouts holding the components of a vector as a third dimension.

    Every call stages into a buffer of its own rather than into a cached one, because the arrays of one state share their
    shapes: the velocities and the accelerations of a solver have the same shape and dtype, and a buffer cached per shape
    would make the kernel write the accelerations into the velocities as well.
    """
    if isinstance(value, (qd.Tensor, qd.Ndarray)):
        if numpy_dtype(value.dtype) == numpy_dtype(dtype):
            return value
        return device_tensor_from_numpy(value.to_numpy(), dtype)
    if isinstance(value, qd.Field):
        return device_tensor_from_numpy(to_numpy_values(read_view(value)), dtype)
    if is_dlpack_source(value):
        if _is_copyable_on_device(value, dtype):
            buffer = qd.ndarray(_qd_type(dtype), shape=tuple(value.shape))
            copy_from_dlpack(buffer, value, transpose=False)
            return buffer
        return device_tensor_from_numpy(_host_values(value), dtype)
    return device_tensor_from_numpy(to_numpy_values(value), dtype)


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
        The copy runs device to device and synchronizes the HIP device, which costs one device synchronize per write
        because the producer may still have writes pending on its own queue, and it moves no data to the host.
        The dtype of `src` must be that of the buffer, 'gs.qd_float' for a float state, since the bridge casts nothing
        and reports a mismatch with its own message.
        A selection covering every environment in order is dropped, so that the copy takes the whole buffer and no
        selection reaches the host.
        A masked write still needs its selection on the host, where the bridge splits the copy into contiguous runs, so
        a selection given on the device costs one host read of its indices, and a backend without a device copy, as the
        CPU one, reads the buffer back once more to patch it. A device-resident selection holding one 0/1 flag per
        environment is the mask form of a torch-free scene, which becomes the indices it flags (see
        'sanitize_index_values').
        """
        buffer = self._buffers.get((num_values, num_envs))
        if buffer is None:
            buffer = qd.ndarray(gs.qd_float, shape=(num_values, num_envs))
            self._buffers[(num_values, num_envs)] = buffer
        selection = envs_idx
        if _is_device_selection(selection):
            selection = _resolve_flag_mask(to_numpy_values(selection), num_envs)
        if is_full_env_selection(selection, num_envs):
            selection = None
        copy_from_dlpack(buffer, src, transpose=True, envs_idx=selection)
        return buffer


def write_from_dlpack(dst, src, *, envs_idx=None, idx=None) -> None:
    """Write the DLPack producer `src` into `dst`, which must be a visibly contiguous batch-last buffer.

    A Quadrants field is never one, so a write into the state goes through 'DlpackStager' and the staged kernels
    instead; the bridge reports the non-contiguity of a field with its own message.
    """
    copy_from_dlpack(dst, src, transpose=True, envs_idx=envs_idx, idx=idx)

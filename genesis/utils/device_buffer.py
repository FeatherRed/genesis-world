"""Torch-free zero-copy interop between Quadrants buffers and DLPack producers, read direction.

Quadrants exports every field, ndarray and tensor as a DLPack v0 capsule, and DLPack is a plain C ABI, so the read
direction needs neither PyTorch nor a per-step host roundtrip. This module turns such a capsule into a
:class:`DeviceView`, the torch-free analogue of 'qd_to_torch'::

    view = view_of(field, transpose=True)   # (num_envs, num_values), batch axis last in the field
    view.shape, view.strides, view.dtype, view.device
    view.to_dlpack()                        # a fresh v0 capsule, consumable by any DLPack runtime
    view.to_numpy()                         # explicit HOST copy, for logging and diagnostics only

The write direction, which Quadrants lacks an importer for and which therefore goes through a guarded HIP
device-to-device copy, lives in :mod:`genesis.utils.hip_copy`::

    copy_from_dlpack(field, producer, transpose=True)   # producer implements '__dlpack__', e.g. an evoxc output

That module imports the layout primitives defined here, so this one never imports it at module scope; instead the two
public entry points it owns are re-exported lazily at the very end of this file, which keeps the public surface of this
module unchanged and keeps both import orders free of cycles.

Nothing here imports PyTorch: 'to_torch' is the only entry point that requires it, and it asks 'torch_compat' first.
The C++ DLPack export of Quadrants does import it, for every buffer at a nonzero byte offset, so a buffer whose export
fails for that reason is described from the buffer itself (see '_qd_buffer_info') and still gets a capsule of its own.
"""

from __future__ import annotations

import ctypes
import dataclasses
from typing import Any

import numpy as np
import quadrants as qd

from .torch_compat import HAS_TORCH, _raise_genesis_error, require_torch, torch

__all__ = [
    "DLTensorInfo",
    "DeviceInfo",
    "DeviceView",
    "copy_from_dlpack",
    "device_of",
    "dlpack_info",
    "qd_ptr",
    "view_of",
    "zero_copy_supported",
]

# ------------------------------------------------------------------------------------------------
# DLPack v0 C ABI
# ------------------------------------------------------------------------------------------------

# DLPack device types, named after their 'DLDeviceType' enum counterparts.
_kDLCPU = 1
_kDLCUDA = 2
_kDLMetal = 8
_kDLROCM = 10

# DLPack data type codes, named after their 'DLDataTypeCode' enum counterparts.
_kDLINT = 0
_kDLUINT = 1
_kDLFLOAT = 2
_kDLBOOL = 6

_DLPACK_CAPSULE_NAME = b"dltensor"
_DLPACK_VERSIONED_CAPSULE_NAME = b"dltensor_versioned"


class _DLDevice(ctypes.Structure):
    """DLPack v0 device descriptor."""

    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class _DLDataType(ctypes.Structure):
    """DLPack v0 data type descriptor."""

    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    """DLPack v0 tensor descriptor, whose shape and strides are in elements and whose offset is in bytes."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    """DLPack v0 managed tensor, whose deleter the consumer calls exactly once."""


_DL_MANAGED_TENSOR_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DL_MANAGED_TENSOR_DELETER),
]

_py_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_py_capsule_is_valid.restype = ctypes.c_int
_py_capsule_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]

_py_capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_py_capsule_get_pointer.restype = ctypes.c_void_p
_py_capsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

_py_capsule_set_context = ctypes.pythonapi.PyCapsule_SetContext
_py_capsule_set_context.restype = ctypes.c_int
_py_capsule_set_context.argtypes = [ctypes.py_object, ctypes.c_void_p]

_py_capsule_new = ctypes.pythonapi.PyCapsule_New
_py_capsule_new.restype = ctypes.py_object
# The destructor of an exported capsule is a NULL pointer rather than a Python callback, see the note below.
_py_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# Capsule storage is owned by Python, since the C side only ever holds a pointer to it. An owner keeps a strong
# reference to its capsule, so a capsule never dies halfway through an exception: a consumer releases both by calling
# the deleter of the managed tensor it carries, and '_release_capsule_owners' releases the ones no consumer ever took.
_DLPACK_CAPSULE_OWNERS: dict[int, "_CapsuleOwner"] = {}


@_DL_MANAGED_TENSOR_DELETER
def _release_managed_tensor(managed_tensor, _owners=_DLPACK_CAPSULE_OWNERS):
    """Release the storage of a managed tensor once its consumer is done with it.

    The address of the owner travels in 'manager_ctx', which DLPack reserves for the producer, so that this callback
    reads nothing but the structure it is given. It never propagates anything into the consumer that called it, and
    everything it needs is captured as a default argument, so that it keeps working while the interpreter shuts down
    and clears the module globals.
    """
    try:
        address = int(managed_tensor.contents.manager_ctx or 0)
        if address:
            _owners.pop(address, None)
    except BaseException:
        pass


def _release_capsule_owners(_owners=_DLPACK_CAPSULE_OWNERS) -> None:
    """Release every exported capsule still owned, for the buffers it describes to disappear with the program.

    A capsule that a consumer never took keeps its owner, and hence the view it describes, alive until this sweep.
    """
    while _owners:
        _owners.popitem()


# An exported capsule carries no destructor callback, which is deliberate: CPython destroys a capsule as soon as its
# last reference dies, and that happens while an exception unwinds whenever the consumer refuses the tensor it was
# given, as IREE does with 'Unsupported strided tensor'. ctypes reports a callback that returns while an exception is
# pending as 'SystemError: <...> returned a result with an exception set', and clears that exception with it, so the
# refusal would surface as 'SystemError: error return without exception set' instead of the real error, on stderr
# first and on every later attempt too. No callback can run then, so none is registered: the owner keeps its capsule
# alive, and the only paths that pop its entry run outside of any exception, namely the deleter of the managed tensor
# that a consumer calls once it owns the tensor, and '_release_capsule_owners' when the program is reset.
qd.lang.impl.on_reset(_release_capsule_owners)


# The deleter outlives the module globals that reference it, since a managed tensor may still be alive at interpreter
# shutdown, after those globals are cleared. An extra reference keeps its C thunk mapped for the whole process, which
# leaks one tiny object and nothing else.
_py_inc_ref = ctypes.pythonapi.Py_IncRef
_py_inc_ref.argtypes = [ctypes.py_object]
_py_inc_ref.restype = None
_py_inc_ref(_release_managed_tensor)


_DLPACK_DTYPES = {
    (_kDLINT, 8): np.dtype(np.int8),
    (_kDLINT, 16): np.dtype(np.int16),
    (_kDLINT, 32): np.dtype(np.int32),
    (_kDLINT, 64): np.dtype(np.int64),
    (_kDLUINT, 8): np.dtype(np.uint8),
    (_kDLUINT, 16): np.dtype(np.uint16),
    (_kDLUINT, 32): np.dtype(np.uint32),
    (_kDLUINT, 64): np.dtype(np.uint64),
    (_kDLFLOAT, 16): np.dtype(np.float16),
    (_kDLFLOAT, 32): np.dtype(np.float32),
    (_kDLFLOAT, 64): np.dtype(np.float64),
    (_kDLBOOL, 8): np.dtype(np.bool_),
}

_DLPACK_DEVICE_TYPES = {
    _kDLCPU: "cpu",
    _kDLCUDA: "cuda",
    _kDLMetal: "mps",
    # ROCm buffers must look like CUDA ones to every consumer, so that 'gs.device.type == "cuda"' keeps holding.
    _kDLROCM: "cuda",
}

_QD_BACKEND_NAMES = {
    qd.cpu: "cpu",
    qd.cuda: "cuda",
    qd.amdgpu: "amdgpu",
    qd.metal: "metal",
}


@dataclasses.dataclass(frozen=True)
class DLTensorInfo:
    """Parsed description of one DLPack tensor, borrowed from its capsule and never owning its memory."""

    data_ptr: int
    device_type: int
    device_id: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: np.dtype
    itemsize: int
    nbytes: int
    byte_offset: int


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides, in elements, of a contiguous tensor of the given shape."""
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def _nbytes(shape: tuple[int, ...], itemsize: int) -> int:
    return itemsize * int(np.prod(shape, dtype=np.int64))


def _dtype_from_dlpack(data_type: _DLDataType) -> np.dtype:
    if data_type.lanes != 1:
        _raise_genesis_error(f"DLPack tensors with {data_type.lanes} lanes per element are not supported.")
    dtype = _DLPACK_DTYPES.get((data_type.code, data_type.bits))
    if dtype is None:
        _raise_genesis_error(
            f"The DLPack data type (code={data_type.code}, bits={data_type.bits}) is not supported."
        )
    return dtype


def _dlpack_data_type(dtype: np.dtype) -> tuple[int, int]:
    for (code, bits), candidate in _DLPACK_DTYPES.items():
        if candidate == dtype:
            return code, bits
    _raise_genesis_error(f"The dtype '{dtype}' cannot be exported as a DLPack tensor.")


def _capsule_of(value):
    """DLPack capsule carried by 'value', whether 'value' is a capsule already or a DLPack producer."""
    if _py_capsule_is_valid(value, _DLPACK_CAPSULE_NAME):
        return value
    if _py_capsule_is_valid(value, _DLPACK_VERSIONED_CAPSULE_NAME):
        _raise_genesis_error(
            "DLPack v1 ('dltensor_versioned') capsules are not supported by the torch-free device-buffer bridge."
        )
    if isinstance(value, (qd.Field, qd.Ndarray, qd.Tensor)):
        return value.to_dlpack()
    if isinstance(value, np.ndarray):
        return value.__dlpack__()
    try:
        export = value.__dlpack__
    except AttributeError:
        _raise_genesis_error(
            f"Object of type '{type(value).__name__}' does not export its buffer through DLPack."
        )
    return export(stream=None)


def dlpack_info(value: Any) -> DLTensorInfo:
    """Parse the DLPack description of 'value', without consuming nor renaming the capsule it comes from.

    'value' is either a 'dltensor' PyCapsule or any object exporting 'DLPack', including a Quadrants field, ndarray or
    tensor and a NumPy array. The returned record borrows the description of the producer: it stays valid only as long
    as 'value' is alive, and it says nothing about who owns the buffer or how long it lives.
    A Quadrants buffer whose own DLPack export requires PyTorch, which the C++ layer demands for every buffer at a
    nonzero byte offset, is described from the buffer itself instead (see '_qd_buffer_info').

    Args:
        value: DLPack capsule or DLPack-capable object.

    Returns:
        The parsed tensor description, with strides in elements and the byte offset of the data pointer.
    """
    if isinstance(value, (qd.Field, qd.Ndarray, qd.Tensor)):
        try:
            capsule = _capsule_of(value)
        except ImportError:
            # Only the C++ DLPack export of Quadrants needs PyTorch, and only for a nonzero byte offset, so every other
            # failure, such as an unsupported dtype, keeps surfacing as itself.
            return _qd_buffer_info(value)
        return _capsule_info(capsule)
    return _capsule_info(_capsule_of(value))


def _dlpack_of(value: Any) -> tuple[DLTensorInfo, Any]:
    """Describe 'value', together with the one DLPack capsule the description was read from, 'None' if none was made.

    A capsule is a one-shot handshake and its export is not free, so a caller that needs both the description and the
    capsule exports exactly one, whether it then reads the description or hands the capsule to a consumer.
    """
    if isinstance(value, (qd.Field, qd.Ndarray, qd.Tensor)):
        try:
            capsule = _capsule_of(value)
        except ImportError:
            # The description of such a buffer is read off the buffer itself, which needs no capsule at all.
            return _qd_buffer_info(value), None
        return _capsule_info(capsule), capsule
    capsule = _capsule_of(value)
    return _capsule_info(capsule), capsule


def _capsule_info(capsule) -> DLTensorInfo:
    """Description carried by a DLPack v0 capsule, borrowed and never consumed."""
    pointer = _py_capsule_get_pointer(capsule, _DLPACK_CAPSULE_NAME)
    if not pointer:
        _raise_genesis_error("Failed to read the DLPack description carried by a DLPack capsule.")
    tensor = _DLTensor.from_address(pointer)
    shape = tuple(int(tensor.shape[axis]) for axis in range(tensor.ndim))
    if tensor.strides:
        strides = tuple(int(tensor.strides[axis]) for axis in range(tensor.ndim))
    else:
        strides = _contiguous_strides(shape)
    dtype = _dtype_from_dlpack(tensor.dtype)
    return DLTensorInfo(
        data_ptr=int(tensor.data),
        device_type=int(tensor.device.device_type),
        device_id=int(tensor.device.device_id),
        shape=shape,
        strides=strides,
        dtype=dtype,
        itemsize=dtype.itemsize,
        nbytes=_nbytes(shape, dtype.itemsize),
        byte_offset=int(tensor.byte_offset),
    )


# ------------------------------------------------------------------------------------------------
# Devices
# ------------------------------------------------------------------------------------------------


def _runtime_backend() -> str:
    """Name of the active backend, resolved without importing PyTorch.

    'gs.backend' is authoritative whenever 'genesis' is importable. The torch-free mode has to work while
    'genesis/__init__.py' still imports PyTorch eagerly, so the Quadrants architecture is the fallback.
    """
    try:
        import genesis as gs
    except ImportError:
        gs = None
    if gs is not None and gs.backend is not None:
        return gs.backend.name
    try:
        arch = qd.lang.impl.current_cfg().arch
    except qd.QuadrantsRuntimeError:
        return "cpu"
    backend_name = _QD_BACKEND_NAMES.get(arch)
    if backend_name is None:
        _raise_genesis_error(f"The Quadrants architecture '{arch}' has no torch-free DLPack counterpart.")
    return backend_name


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    """Torch-free descriptor of the device backing a buffer, mirroring the 'torch.device' subset Genesis relies on."""

    type: str
    index: int

    def __repr__(self) -> str:
        return f"device(type='{self.type}', index={self.index})"

    @property
    def dlpack_device_type(self) -> int:
        """DLPack device type of this device, resolved from the active backend so that ROCm reports 'kDLROCM'."""
        if self.type == "cpu":
            return _kDLCPU
        if self.type == "mps":
            _raise_genesis_error("Apple Metal (MPS) is not supported by the torch-free device-buffer bridge.")
        backend_name = _runtime_backend()
        if backend_name == "metal":
            _raise_genesis_error("Apple Metal (MPS) is not supported by the torch-free device-buffer bridge.")
        return _kDLROCM if backend_name == "amdgpu" else _kDLCUDA


def _device_from_dlpack(device_type: int, device_id: int) -> DeviceInfo:
    kind = _DLPACK_DEVICE_TYPES.get(device_type)
    if kind is None:
        _raise_genesis_error(f"The DLPack device type {device_type} is not supported.")
    return DeviceInfo(type=kind, index=device_id)


def _unwrap_qd(value):
    """Underlying Quadrants object of 'value', unwrapping the 'qd.Tensor' view."""
    if isinstance(value, qd.Tensor):
        return value._unwrap()
    return value


def _qd_program():
    program = qd.lang.impl.get_runtime().prog
    if program is None:
        _raise_genesis_error("Quadrants has not been initialized yet: call 'gs.init' or 'qd.init' first.")
    return program


# ------------------------------------------------------------------------------------------------
# Torch-free buffer description
# ------------------------------------------------------------------------------------------------


# Scalars this module can describe, keyed by the name Quadrants prints for the dtype. Genesis allocates floats and
# integers as 'f32'/'f64'/'i32'/'i64' and booleans as 'i8', so this covers every buffer its solvers hold.
_QD_DTYPES = {
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "u1": np.dtype(np.bool_),
    "u8": np.dtype(np.uint8),
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "u64": np.dtype(np.uint64),
}

# The Quadrants primitive type each of those dtypes is allocated as, for the buffers '_materialize' creates.
_QD_PRIMITIVES = {dtype: getattr(qd, name) for name, dtype in _QD_DTYPES.items()}


def _qd_element_shape(value) -> tuple[int, ...]:
    """Trailing element dimensions of a Quadrants buffer: the vector or matrix of a compound one, none otherwise."""
    if isinstance(value, qd.Ndarray):
        return tuple(int(size) for size in value.element_shape)
    if isinstance(value, qd.MatrixField):
        return (int(value.n), int(value.m))[: value.ndim]
    if isinstance(value, qd.ScalarField):
        return ()
    _raise_genesis_error(
        f"The Quadrants buffer of type '{type(value).__name__}' has no DLPack counterpart without PyTorch: only "
        "scalar, vector and matrix fields and ndarrays are supported."
    )


def _qd_dtype(value) -> np.dtype:
    """NumPy dtype of one value of a Quadrants buffer, refused rather than guessed when unknown."""
    name = str(value.dtype)
    dtype = _QD_DTYPES.get(name)
    if dtype is None:
        _raise_genesis_error(
            f"The Quadrants dtype '{name}' of '{type(value).__name__}' has no DLPack counterpart without PyTorch."
        )
    return dtype


def _qd_strides(shape: tuple[int, ...], layout: tuple[int, ...] | None) -> tuple[int, ...]:
    """Strides of a Quadrants buffer, in elements, as its own DLPack export reports them.

    'layout' lists the canonical axes in physical order, so a tagged buffer is contiguous over its permuted physical
    shape; the canonical strides are that row-major pattern permuted back, and the trailing element dimensions of a
    compound buffer, which are innermost in both orders, keep theirs.
    """
    if layout is None:
        return _contiguous_strides(shape)
    num_axes = len(layout)
    physical_strides = _contiguous_strides(tuple(shape[axis] for axis in layout) + shape[num_axes:])
    positions = [0] * num_axes
    for position, axis in enumerate(layout):
        positions[axis] = position
    return tuple(physical_strides[positions[axis]] for axis in range(num_axes)) + physical_strides[num_axes:]


def _qd_layout(value) -> tuple[int, ...] | None:
    """Axis permutation a Quadrants buffer is stored with, or None when it is stored in canonical order."""
    layout = value.layout
    return None if layout is None else tuple(int(axis) for axis in layout)


def _qd_snode_offset(node) -> int:
    """Byte offset of an SNode in the buffer of its tree, the sum of its offsets in the cells of its ancestors."""
    offset = 0
    while node is not None:
        offset += int(node.offset_bytes_in_parent_cell)
        node = node.parent
    return offset


def _qd_snode_leaf(node):
    """First SNode of a subtree carrying a concrete dtype, below the 'gen' nodes holding its children."""
    while str(node.data_type()) == "gen":
        if node.get_num_ch() == 0:
            _raise_genesis_error("A Quadrants SNode tree carries no leaf, so its buffer cannot be described.")
        node = node.get_ch(0)
    return node


def _qd_field_base(value) -> tuple[int, int, int]:
    """Base address of the tree buffer a field lives in, and the DLPack device that buffer sits on.

    Quadrants allocates the buffer of a tree once and reports it as the data pointer of every field of the tree, the
    byte offset of the field separating them. Its export reaches the PyTorch version check only for a field at a nonzero
    offset, so the leading node of the tree, which sits at offset zero, still exports without PyTorch, and its capsule
    carries the address of the whole buffer. That address is re-read on every call rather than cached, so a tree whose
    buffer is replaced by a later materialization can never be described by a stale address.
    """
    node = _qd_snode_leaf(_qd_program().get_snode_root(value._snode._snode_tree_id))
    try:
        capsule = _qd_program().field_to_dlpack(node, 0, 0, 0)
    except RuntimeError as e:
        _raise_genesis_error(
            f"The Quadrants buffer of '{type(value).__name__}' cannot be described without PyTorch: the buffer of its "
            "SNode tree is not exportable either.",
            cause=e,
        )
    tensor = _DLTensor.from_address(_py_capsule_get_pointer(capsule, _DLPACK_CAPSULE_NAME))
    return (
        int(tensor.data) + int(tensor.byte_offset) - _qd_snode_offset(node),
        int(tensor.device.device_type),
        int(tensor.device.device_id),
    )


def _qd_device() -> tuple[int, int]:
    """DLPack device of the active Quadrants backend, for the buffers no capsule describes."""
    backend = _runtime_backend()
    device = DeviceInfo(type="cpu" if backend == "cpu" else "cuda", index=0)
    return device.dlpack_device_type, device.index


def _qd_buffer_info(value) -> DLTensorInfo:
    """Description of a Quadrants buffer read off the buffer itself, for the ones DLPack only describes with PyTorch."""
    qd.lang.impl.get_runtime().materialize()
    source = _unwrap_qd(value)
    shape = tuple(int(size) for size in source.shape) + _qd_element_shape(source)
    dtype = _qd_dtype(source)
    if isinstance(source, qd.Ndarray):
        data_ptr = _qd_program().get_ndarray_data_ptr_as_int(source.arr)
        byte_offset = 0
        device_type, device_id = _qd_device()
    else:
        data_ptr, device_type, device_id = _qd_field_base(source)
        byte_offset = _qd_snode_offset(source._snode.ptr)
    return DLTensorInfo(
        data_ptr=data_ptr,
        device_type=device_type,
        device_id=device_id,
        shape=shape,
        strides=_qd_strides(shape, _qd_layout(source)),
        dtype=dtype,
        itemsize=dtype.itemsize,
        nbytes=_nbytes(shape, dtype.itemsize),
        byte_offset=byte_offset,
    )


def device_of(value: Any) -> DeviceInfo:
    """Device of a Quadrants field, ndarray or tensor, of a :class:`DeviceView`, or of any DLPack-capable object."""
    if isinstance(value, DeviceView):
        return value.device
    try:
        device_type, device_id = value.__dlpack_device__()
    except AttributeError:
        info = dlpack_info(value)
        return _device_from_dlpack(info.device_type, info.device_id)
    return _device_from_dlpack(int(device_type), int(device_id))


def zero_copy_supported() -> bool:
    """Whether buffers of the active backend can be shared without going through a host copy."""
    return _runtime_backend() in ("cpu", "cuda", "amdgpu")


# ------------------------------------------------------------------------------------------------
# Views
# ------------------------------------------------------------------------------------------------


def _moved(values: tuple[int, ...], source: int, destination: int) -> tuple[int, ...]:
    """'values' with the axis 'source' moved to position 'destination', as 'numpy.moveaxis' does."""
    num_dims = len(values)
    source %= num_dims
    destination %= num_dims
    order = list(range(num_dims))
    order.insert(destination, order.pop(source))
    return tuple(values[axis] for axis in order)


def _derive_layout(
    shape: tuple[int, ...], strides: tuple[int, ...], operations: tuple[tuple[str, Any], ...]
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Element offset, shape and strides of a view, from the layout of its base buffer and the operations applied."""
    offset = 0
    for operation, argument in operations:
        if operation == "transpose":
            axes = argument
            shape = tuple(shape[axis] for axis in axes)
            strides = tuple(strides[axis] for axis in axes)
        elif operation == "moveaxis":
            source, destination = argument
            shape = _moved(shape, source, destination)
            strides = _moved(strides, source, destination)
        elif operation == "reshape":
            shape = argument
            strides = _contiguous_strides(shape)
        else:  # 'index'
            # As in NumPy indexing, the axes left out of the selection keep all their elements, which is how the masks
            # of a vector buffer leave the element axis untouched instead of dropping it.
            selections = argument + (slice(None),) * (len(shape) - len(argument))
            next_shape, next_strides = [], []
            for axis, selection in enumerate(selections):
                if isinstance(selection, int):
                    index = selection + shape[axis] if selection < 0 else selection
                    if not 0 <= index < shape[axis]:
                        _raise_genesis_error(
                            f"Index {selection} is out of bound for axis {axis} of size {shape[axis]}."
                        )
                    offset += index * strides[axis]
                else:
                    start, stop, step = selection.indices(shape[axis])
                    next_shape.append(len(range(start, stop, step)))
                    next_strides.append(strides[axis] * step)
                    offset += start * strides[axis]
            shape, strides = tuple(next_shape), tuple(next_strides)
    return offset, shape, strides


def _element_ndim(value) -> int:
    """Number of trailing axes of a Quadrants buffer holding the vector or matrix of one element, none otherwise.

    The grid of a buffer and the element it holds are exported as one flat tuple of axes, so this count is what tells
    the batch axis, the last one of the grid, from the element axes following it.
    """
    if isinstance(value, (qd.Field, qd.Ndarray)):
        return len(_qd_element_shape(value))
    return 0


class _CapsuleOwner:
    """Python-side storage backing one exported DLPack capsule, alive until the consumer releases it."""

    def __init__(self, view: "DeviceView"):
        self.view = view
        # Strong reference to the capsule this owner backs, which is what keeps the pair released at a single point:
        # a capsule can only die once its entry is popped, which only ever happens outside of an exception unwinding.
        self.capsule = None
        self.shape = (ctypes.c_int64 * view.ndim)(*view.shape)
        self.strides = (ctypes.c_int64 * view.ndim)(*view.strides)
        code, bits = _dlpack_data_type(view.dtype)
        self.managed = _DLManagedTensor()
        self.managed.dl_tensor.data = ctypes.c_void_p(view.data_ptr)
        self.managed.dl_tensor.device.device_type = view.device.dlpack_device_type
        self.managed.dl_tensor.device.device_id = view.device.index
        self.managed.dl_tensor.ndim = view.ndim
        self.managed.dl_tensor.dtype.code = code
        self.managed.dl_tensor.dtype.bits = bits
        self.managed.dl_tensor.dtype.lanes = 1
        self.managed.dl_tensor.shape = self.shape
        self.managed.dl_tensor.strides = self.strides
        self.managed.dl_tensor.byte_offset = 0
        self.address = ctypes.addressof(self.managed)
        self.managed.manager_ctx = ctypes.c_void_p(self.address)
        self.managed.deleter = _release_managed_tensor


def _export_capsule(view: "DeviceView"):
    """Fresh DLPack capsule of a view that cannot reuse the capsule of its base buffer."""
    owner = _CapsuleOwner(view)
    _DLPACK_CAPSULE_OWNERS[owner.address] = owner
    try:
        capsule = _py_capsule_new(ctypes.c_void_p(owner.address), _DLPACK_CAPSULE_NAME, None)
    except BaseException:
        _DLPACK_CAPSULE_OWNERS.pop(owner.address, None)
        raise
    if _py_capsule_set_context(capsule, ctypes.c_void_p(owner.address)) != 0:
        _DLPACK_CAPSULE_OWNERS.pop(owner.address, None)
        _raise_genesis_error("Failed to export a torch-free device view as a DLPack capsule.")
    owner.capsule = capsule
    return capsule


class DeviceView:
    """Zero-copy view over a Quadrants field, ndarray or tensor, or over any DLPack-capable producer.

    The view carries the element offset, shape and strides (in elements) of the selected region, so that it can be
    exported as a DLPack capsule without touching the data. It is the torch-free counterpart of the tensor returned by
    'qd_to_torch', and it supports the same layout semantics through :func:`view_of`.
    """

    def __init__(self, value: Any):
        source = _unwrap_qd(value)
        info = dlpack_info(source)
        self._source = source
        self._base_info = info
        self._element_ndim = _element_ndim(source)
        self._dtype = info.dtype
        self._device = _device_from_dlpack(info.device_type, info.device_id)
        self._operations: tuple[tuple[str, Any], ...] = ()
        # Quadrants buffer this view owns, which only a materialized view has: it keeps the device memory alive for
        # exactly as long as the view, and its derived views, are alive.
        self._buffer = None
        self._offset, self._shape, self._strides = _derive_layout(info.shape, info.strides, ())

    @classmethod
    def _derive(cls, base: "DeviceView", operations: tuple[tuple[str, Any], ...]) -> "DeviceView":
        view = cls.__new__(cls)
        view._source = base._source
        view._base_info = base._base_info
        view._element_ndim = base._element_ndim
        view._dtype = base._dtype
        view._device = base._device
        view._buffer = base._buffer
        view._operations = base._operations + operations
        offset, shape, strides = _derive_layout(base._base_info.shape, base._base_info.strides, view._operations)
        view._offset = offset
        view._shape = shape
        view._strides = strides
        return view

    def __repr__(self) -> str:
        return (
            f"DeviceView(shape={self.shape}, strides={self.strides}, dtype={self.dtype}, device={self.device})"
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Strides of the view, in elements, as DLPack reports them."""
        return self._strides

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def itemsize(self) -> int:
        return self._dtype.itemsize

    @property
    def device(self) -> DeviceInfo:
        return self._device

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def element_ndim(self) -> int:
        """Number of trailing axes holding the vector or matrix of one element, none for a plain buffer."""
        return self._element_ndim

    @property
    def grid_ndim(self) -> int:
        """Number of leading axes that index elements, i.e. every axis but the element ones."""
        return len(self._shape) - self._element_ndim

    @property
    def nbytes(self) -> int:
        return _nbytes(self._shape, self.itemsize)

    @property
    def data_ptr(self) -> int:
        """Device pointer of the first element of the view."""
        return self._base_info.data_ptr + self._base_info.byte_offset + self._offset * self.itemsize

    @property
    def is_contiguous(self) -> bool:
        return self._strides == _contiguous_strides(self._shape)

    @property
    def is_base_view(self) -> bool:
        """Whether the view covers exactly the buffer of its source, without any selection nor permutation."""
        return not self._operations

    def to_dlpack(self, *, versioned: bool = False):
        """Export the view as a fresh DLPack capsule, which the caller must hand over to a single consumer."""
        if self.is_base_view and isinstance(self._source, (qd.Field, qd.Ndarray, qd.Tensor)):
            try:
                return self._source.to_dlpack(versioned=versioned)
            except ImportError:
                # The C++ export of Quadrants needs PyTorch for any buffer at a nonzero byte offset, while the capsule
                # built from the description of this view needs nothing: only a v1 capsule has no torch-free builder.
                if versioned:
                    _raise_genesis_error(
                        "A DLPack v1 capsule of a Quadrants buffer at a nonzero byte offset requires PyTorch."
                    )
                return _export_capsule(self)
        if versioned:
            _raise_genesis_error(
                "A DLPack v1 capsule can only be exported for a whole Quadrants buffer, not for a partial view."
            )
        return _export_capsule(self)

    def __dlpack__(self, stream=None):
        return self.to_dlpack()

    def __dlpack_device__(self):
        return self._device.dlpack_device_type, self._device.index

    def _base_host_array(self) -> np.ndarray:
        if isinstance(self._source, (qd.Field, qd.Ndarray, qd.Tensor)):
            return self._source.to_numpy()
        if self._device.type != "cpu":
            _raise_genesis_error(
                f"A host copy of a '{self._device}' DLPack producer is not supported; only Quadrants buffers and CPU "
                "producers can be read back without PyTorch."
            )
        return np.from_dlpack(self)

    def to_numpy(self) -> np.ndarray:
        """Host copy of the view, for logging and diagnostics only: never call it on a hot path."""
        array = self._base_host_array()
        for operation, argument in self._operations:
            if operation == "transpose":
                array = np.transpose(array, argument)
            elif operation == "moveaxis":
                array = np.moveaxis(array, argument[0], argument[1])
            elif operation == "reshape":
                array = array.reshape(argument)
            else:  # 'index'
                array = array[argument]
        return np.array(array, copy=True)

    def to_torch(self, copy: bool = False):
        """Torch tensor sharing this view, or an independent contiguous copy of it if 'copy' is True."""
        require_torch("genesis.utils.device_buffer.DeviceView.to_torch")
        tensor = torch.from_dlpack(self)
        if copy:
            tensor = tensor.clone(memory_format=torch.contiguous_format)
        return tensor

    def contiguous(self) -> "DeviceView":
        """Device-resident copy of this view with contiguous strides, or this very view when it is contiguous already.

        Materializing a view is what a DLPack consumer requiring a contiguous tensor needs: IREE refuses a strided
        one with 'Unsupported strided tensor', which a batch-last solver buffer viewed with 'transpose=True' always
        is, its strides being '(1, num_envs)'. The returned view owns a freshly allocated Quadrants buffer holding the
        same values, so the copy is device-to-device on a ROCm backend, never a host roundtrip, and its layout no
        longer depends on the buffer it was read from.

        The copy runs once per call: 'qd.sync()' drains the Quadrants stream first, so that every write already
        launched is part of the snapshot, and the transfers themselves are synchronous, so the values are complete
        when the method returns. The returned view is therefore safe to hand to a DLPack consumer of any stream.

        Raises:
            GenesisException: On a Metal backend, which the torch-free bridge cannot copy on.
        """
        if self.is_contiguous:
            return self
        if self._device.type == "metal":
            _raise_genesis_error("Materializing a device view is not supported on Metal by the torch-free bridge.")
        num_elements = int(np.prod(self._shape, dtype=np.int64))
        buffer = qd.ndarray(dtype=_QD_PRIMITIVES[self._dtype], shape=(num_elements,))
        view = DeviceView(buffer).reshape(self._shape)
        view._buffer = buffer
        qd.sync()
        if self._device.type == "cpu":
            buffer.from_numpy(np.ascontiguousarray(self.to_numpy().reshape(-1)))
            return view
        # Local import: the HIP runtime lives in the write direction, which imports this module at module scope.
        from genesis.utils.hip_copy import _materialize_view

        _materialize_view(view, self)
        qd.sync()
        return view

    def reshape(self, *shape) -> "DeviceView":
        """View of the same elements with another shape, materializing a non-contiguous view first as NumPy would."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(size) for size in shape)
        if not self.is_contiguous:
            # NumPy reshapes a non-contiguous array by copying it, and so does this view, on the device itself.
            return self.contiguous().reshape(*shape)
        if -1 in shape:
            if shape.count(-1) > 1:
                _raise_genesis_error(f"At most one dimension of a reshape can be -1, got {shape}.")
            known = int(np.prod([size for size in shape if size != -1], dtype=np.int64))
            num_elements = int(np.prod(self._shape, dtype=np.int64))
            if known == 0 or num_elements % known:
                _raise_genesis_error(f"Cannot reshape a device view of shape {self.shape} to {shape}.")
            shape = tuple(num_elements // known if size == -1 else size for size in shape)
        return DeviceView._derive(self, (("reshape", shape),))

    def transpose(self, *axes) -> "DeviceView":
        """View of the same elements with permuted axes, reversing them when 'axes' is empty as in NumPy."""
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = tuple(int(axis) % self.ndim for axis in axes) if self.ndim else ()
        if sorted(axes) != list(range(self.ndim)):
            _raise_genesis_error(f"Axes {axes} are not a permutation of a device view with {self.ndim} axes.")
        return DeviceView._derive(self, (("transpose", axes),))

    def movedim(self, source: int, destination: int) -> "DeviceView":
        """View of the same elements with one axis moved, as 'numpy.moveaxis' does."""
        return DeviceView._derive(self, (("moveaxis", (source, destination)),))

    def __getitem__(self, key) -> "DeviceView":
        """View of the selected slice, or of the single element a plain integer index designates."""
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.ndim:
            _raise_genesis_error(f"Too many indices for a device view with {self.ndim} axes.")
        selection = []
        for item in key:
            if isinstance(item, (int, slice)):
                selection.append(item)
            else:
                _raise_genesis_error(
                    "torch-free device views support only slice/contiguous-index selection, but got "
                    f"'{type(item).__name__}' as an index."
                )
        selection += [slice(None)] * (self.ndim - len(key))
        return DeviceView._derive(self, (("index", tuple(selection)),))


def _host_index_array(index: Any) -> np.ndarray:
    """Host copy of an index collection, which is deliberate: masks are small and cannot be uploaded without torch."""
    if isinstance(index, DeviceView):
        return np.asarray(index.to_numpy())
    if HAS_TORCH and isinstance(index, torch.Tensor):
        from .misc import tensor_to_array  # local import: 'misc' imports torch at module scope

        return np.asarray(tensor_to_array(index))
    return np.asarray(index)


def _normalize_mask(indices: tuple[Any, ...], keepdim: bool) -> tuple[slice | int, ...]:
    """Normalize masks the way 'indices_to_mask(..., keepdim=keepdim, to_torch=False)' does.

    An integer keeps its dimension as a length-1 slice when 'keepdim' is True. A collection is accepted only when it
    selects a contiguous ascending run, which a plain slice describes exactly and a strided copy would not.
    """
    selection = []
    for index in indices:
        if index is None:
            selection.append(slice(None))
        elif isinstance(index, slice):
            selection.append(index)
        elif isinstance(index, range):
            selection.append(slice(index.start, index.stop, index.step))
        elif type(index) is int:
            selection.append(slice(index, index + 1 if index != -1 else None) if keepdim else index)
        else:
            values = _host_index_array(index)
            if values.dtype == np.bool_:
                values = np.nonzero(values)[0]
            if values.size == 1:
                value = int(values[0])
                selection.append(slice(value, value + 1 if value != -1 else None) if keepdim else value)
                continue
            first, last = int(values[0]), int(values[-1])
            if not np.array_equal(values, np.arange(first, last + 1)):
                _raise_genesis_error(
                    "torch-free device views support only slice/contiguous-index selection, but got the "
                    f"non-contiguous indices {values.tolist()}."
                )
            selection.append(slice(first, last + 1))
    return tuple(selection)


def view_of(
    value: Any,
    *,
    transpose: bool = False,
    row_mask: Any = None,
    col_mask: Any = None,
    keepdim: bool = True,
) -> DeviceView:
    """Torch-free analogue of 'qd_to_torch', with the same layout semantics and no host roundtrip.

    The batch axis is the last axis of the grid: 'transpose=True' moves it to the front, exactly as 'qd_to_torch' does,
    so the element axes of a vector or matrix buffer stay the trailing ones ('(m, n)' grid of a vector buffer holding
    '(n, m, 3)', never '(3, m, n)'). The masks are applied after the transposition, over the grid axes only, and an
    integer keeps its dimension as a length-1 slice when 'keepdim' is True. Only slices and contiguous ascending index
    collections are supported: a strided selection would require a copy, which this function refuses instead of
    silently materializing a buffer.

    Args:
        value (qd.Tensor | qd.Field | qd.Ndarray | DeviceView): Buffer or view to select from.
        transpose (bool): Whether move to front the first non-batch dimension.
        row_mask (optional): Rows to extract from batch dimension after transpose if requested.
        col_mask (optional): Columns to extract from batch dimension after transpose if requested.
        keepdim (bool): Whether to keep all dimensions even if masks are integers.

    Returns:
        The zero-copy view.
    """
    view = value if isinstance(value, DeviceView) else DeviceView(value)
    grid_ndim = view.grid_ndim
    if transpose and grid_ndim >= 2:
        view = view.movedim(grid_ndim - 1, 0)
    if row_mask is None and col_mask is None:
        return view
    if grid_ndim < 2:
        if row_mask is not None and col_mask is not None:
            _raise_genesis_error("Cannot specify both row and column masks for a device view with 1D batch.")
        indices = (row_mask if col_mask is None else col_mask,)
    else:
        indices = (row_mask, col_mask)
    return DeviceView._derive(view, (("index", _normalize_mask(indices, keepdim)),))


# ------------------------------------------------------------------------------------------------
# Write direction (re-exported)
# ------------------------------------------------------------------------------------------------

# The write direction lives in 'genesis.utils.hip_copy', which imports the read-side primitives defined here, so a
# module-level import of it would be circular: importing 'hip_copy' first would find this module's last line while
# 'hip_copy' is still only partially initialized. The entry points of the write direction are therefore resolved on
# first access, which works whichever of the two modules gets imported first.
# The selection helpers below are part of that write direction: they live in 'hip_copy' because a write selection may be
# any collection of indices, unlike the contiguous runs the read direction requires, and they are re-exported here
# because callers ask for them next to 'copy_from_dlpack'.
_LAZY_REEXPORTS = ("_index_runs", "_selected_indices", "copy_from_dlpack", "qd_ptr")


def __getattr__(name: str):
    """Resolve the re-exported entry points of 'genesis.utils.hip_copy', importing that module on first access."""
    if name in _LAZY_REEXPORTS:
        from genesis.utils import hip_copy

        return hip_copy.__dict__[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List the public surface, so that the lazy re-exports show up as the plain attributes they behave like."""
    return sorted(set(globals()) | set(_LAZY_REEXPORTS))

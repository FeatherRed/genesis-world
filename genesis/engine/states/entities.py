import numpy as np

from genesis.repr_base import RBC
from genesis.utils.torch_compat import is_torch_tensor


def _copy_of(value):
    """Copy of the buffer 'value' holds, so that editing this state leaves the solver state it was read from untouched.

    A PyTorch tensor is cloned where it lives and a NumPy array copied on the host. Without PyTorch the buffer is a
    read-only zero-copy device view of the solver state, which nothing can be written back through, so it is kept as it
    is rather than materialized on the host, a roundtrip the PyTorch path does not pay either.
    """
    if is_torch_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


class MPMEntityState(RBC):
    """
    Dynamic state queried from a genesis MPMEntity.
    """

    def __init__(self, entity, s_global, pos, vel, C, F, Jp, active):
        self._entity = entity
        self._s_global = s_global
        self._pos = pos
        self._vel = vel
        self._C = C
        self._F = F
        self._Jp = Jp
        self._active = active

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def Jp(self):
        return self._Jp

    @property
    def active(self):
        return self._active


class SPHEntityState(RBC):
    """
    Dynamic state queried from a genesis SPHEntity.
    """

    def __init__(self, entity, s_global, pos, vel):
        self._entity = entity
        self._s_global = s_global
        self._pos = pos
        self._vel = vel

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel


class FEMEntityState:
    """
    Dynamic state queried from a genesis FEMEntity.
    """

    def __init__(self, entity, s_global, pos, vel):
        self._entity = entity
        self._s_global = s_global
        self._pos = pos
        self._vel = vel

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel


class RigidEntityState(RBC):
    """
    Dynamic state queried from a genesis RigidEntity.
    """

    def __init__(self, entity, s_global, pos, quat):
        self._entity = entity
        self._s_global = s_global
        # Copies, so that editing this state leaves the solver state it was read from untouched
        self._pos = _copy_of(pos)
        self._quat = _copy_of(quat)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def quat(self):
        return self._quat

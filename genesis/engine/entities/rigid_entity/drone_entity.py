import numpy as np

import genesis as gs
from genesis.utils.misc import qd_to_torch
from genesis.utils.torch_compat import HAS_TORCH, is_torch_tensor, torch

from .description import DroneEntityDescription
from .rigid_entity import RigidEntity


def _is_opaque_producer(value) -> bool:
    """Whether `value` is an opaque DLPack producer a setter stages on the device instead of reading it on the host.

    Such a producer exposes its 'shape', its 'dtype' and the DLPack protocol alone, without element access, length or
    iteration, so it can neither be sanitized nor transposed the way a host array and a torch tensor are (see
    'genesis.engine.solvers.torch_free.is_dlpack_source'). A host array and a torch tensor are DLPack sources as well,
    and the setters of a scene with PyTorch keep their own path for both of them.
    """
    # Local import: the solvers import the entities, so importing them at module scope would close the cycle
    from genesis.engine.solvers.torch_free import is_dlpack_source

    return is_dlpack_source(value) and not is_torch_tensor(value)


class DroneEntity(RigidEntity):
    _description_cls = DroneEntityDescription

    def _load_model(self):
        super()._load_model()

        morph = self._morph
        self._n_propellers = len(morph.propellers_link_name)

        propellers_link = gs.List([self.get_link(name) for name in morph.propellers_link_name])
        propellers_link_idx = [link.idx for link in propellers_link]
        if HAS_TORCH:
            self._propellers_link_idx = torch.tensor(propellers_link_idx, dtype=gs.tc_int, device=gs.device)
        else:
            self._propellers_link_idx = np.asarray(propellers_link_idx, dtype=gs.np_int)
        try:
            propellers_vgeom_idxs = [link.vgeoms[0].idx for link in propellers_link]
            if HAS_TORCH:
                self._propellers_vgeom_idxs = torch.tensor(propellers_vgeom_idxs, dtype=gs.tc_int, device=gs.device)
            else:
                self._propellers_vgeom_idxs = np.asarray(propellers_vgeom_idxs, dtype=gs.np_int)
            self._animate_propellers = True
        except Exception:
            gs.logger.warning("No visual geometry found for propellers. Skipping propeller animation.")
            self._animate_propellers = False

        if HAS_TORCH:
            self._propellers_spin = torch.tensor(morph.propellers_spin, dtype=gs.tc_float, device=gs.device)
        else:
            self._propellers_spin = np.asarray(morph.propellers_spin, dtype=gs.np_float)
        self._model = morph.model

    def _build(self):
        super()._build()

        if HAS_TORCH:
            self._propellers_revs = torch.zeros(
                (self._n_propellers, self.solver._B), dtype=gs.tc_float, device=gs.device
            )
        else:
            from genesis.engine.solvers.torch_free import device_tensor_from_numpy  # local: the solvers import this

            # A torch-free scene keeps the accumulator on the device, where the kernel advancing it reads and writes it
            self._propellers_revs = device_tensor_from_numpy(
                np.zeros((self._n_propellers, self.solver._B), dtype=gs.np_float), gs.qd_float
            )
        self._prev_prop_t = None

    def set_propellers_rpm(self, propellers_rpm):
        """
        Set the RPM (revolutions per minute) for each propeller in the drone.

        Parameters
        ----------
        propellers_rpm : array-like or torch.Tensor or DLPack producer
            A tensor or array of shape (n_propellers,) or (n_envs, n_propellers) specifying
            the desired RPM values for each propeller. Must be non-negative.
            A DLPack producer, as the output of a compiled graph, is accepted as well, opaque or not: it is copied to
            the device without leaving it and never indexed on the host.

        Raises
        ------
        RuntimeError
            If the method is called more than once per simulation step, or if the input shape
            does not match the number of propellers, or contains negative values.
        """
        if self._prev_prop_t == self.sim.cur_step_global:
            gs.raise_exception("`set_propellers_rpm` can only be called once per step.")
        self._prev_prop_t = self.sim.cur_step_global

        if not HAS_TORCH:
            # Both the accumulator and the staged RPM values live on the device, so the solver advances the one and
            # applies the other without any host arithmetic
            if propellers_rpm is None:
                gs.raise_exception("'propellers_rpm' cannot be None.")
            self.solver.set_drone_propellers_rpm(
                self._propellers_revs,
                self._propellers_link_idx,
                self.KF,
                self.KM,
                propellers_rpm,
                self._propellers_spin,
                self._model == "RACE",
            )
            return

        if _is_opaque_producer(propellers_rpm):
            # An opaque producer is opaque all the same with PyTorch installed, where indexing it on the host would
            # raise: it takes the staged device path of a torch-free scene, and the solver expands a single row to
            # every environment. The accumulator of a scene with PyTorch stays a torch tensor, which advances here on
            # the staged values, in the layout they hold, exactly as it does on the host path below.
            values = self.solver._stage_drone_rpm(propellers_rpm, self._n_propellers)
            self._propellers_revs = (self._propellers_revs + qd_to_torch(values).T) % (60 / self.solver.dt)
            self.solver.set_drone_rpm(
                self._propellers_link_idx, self.KF, self.KM, values, self._propellers_spin, self._model == "RACE"
            )
            return

        assert propellers_rpm is not None
        propellers_rpm, *_ = self._solver._sanitize_io_variables(
            propellers_rpm, self._propellers_link_idx, self._n_propellers, "propellers_link_idx"
        )
        if self._scene.n_envs == 0:
            propellers_rpm = propellers_rpm[None]

        # FIXME: This check is too expensive
        # if (propellers_rpm < 0.0).any():
        #     gs.raise_exception("`propellers_rpm` cannot be negative.")

        self._propellers_revs = (self._propellers_revs + propellers_rpm.T) % (60 / self.solver.dt)

        self.solver.set_drone_rpm(
            self._propellers_link_idx, self.KF, self.KM, propellers_rpm, self._propellers_spin, self._model == "RACE"
        )

    def update_propeller_vgeoms(self):
        """
        Update the visual geometry of the propellers for animation based on their current rotation.

        This method is a no-op if animation is disabled due to missing visual geometry.
        """
        if self._animate_propellers:
            self.solver.update_drone_propeller_vgeoms(
                self._propellers_vgeom_idxs, self._propellers_revs, self._propellers_spin
            )

    @property
    def model(self):
        """The model type of the drone."""
        return self._model

    @property
    def KF(self):
        """The drone's thrust coefficient."""
        return self._desc.kf

    @property
    def KM(self):
        """The drone's moment coefficient."""
        return self._desc.km

    @property
    def n_propellers(self):
        """The number of propellers on the drone."""
        return self._n_propellers

    @property
    def COM_link_idx(self):
        """The index of the center-of-mass (COM) link of the drone."""
        return self._COM_link_idx

    @property
    def propellers_idx(self):
        """The indices of the drone's propeller links."""
        return self._propellers_link_idx

    @property
    def propellers_spin(self):
        """The spin direction for each propeller."""
        return self._propellers_spin

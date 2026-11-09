# genesis/engine/states

## Intent

Holds the serializable physics state of a `Scene`: read-only state value objects (`*State`) and the checkpoint records (`SolverCheckpoint` / `SimulatorCheckpoint`) used for save/load and trajectory recording.
This directory owns only data holders — no physics logic, no kernels, no serialization code.
The `*State` classes are returned by `get_state()` calls; the `*Checkpoint` dataclasses are returned by `__getstate__`/`save_checkpoint` and consumed by the trajectory writer/reader.

## API Surface

`__init__.py` re-exports `solvers` only (`from .solvers import *`), so the solver-state and checkpoint classes are the package surface.
Entity-state classes are NOT re-exported: entity modules import them directly from `genesis.engine.states.entities`.

### `entities.py` — per-entity views (`Entity.get_state()`)
- `MPMEntityState(entity, s_global, pos, vel, C, F, Jp, active)` — RBC, read-only properties.
- `SPHEntityState(entity, s_global, pos, vel)` — RBC.
- `FEMEntityState(entity, s_global, pos, vel)` — plain class (no RBC).
- `RigidEntityState(entity, s_global, pos, quat)` — RBC; clones `pos`/`quat`.
All hold a back-reference to the owning entity and `s_global` (the step index at capture).

### `solvers.py` — checkpoint records + per-solver states
- `SolverCheckpoint(arrays, configs, kinds)` — dataclass; `arrays` = named arrays under `Solver.data` names, `configs` = static provenance (never written back), `kinds` = `frozenset[DataKind]`.
- `SimulatorCheckpoint(steps, solvers)` — dataclass; `steps` = per-env step counts `[B]`, `solvers` = dict of solver class name → `SolverCheckpoint`.
- `SimState(scene, s_global, f_local, solvers)` — RBC; builds `solvers_state` = `[solver.get_state(f_local) for solver in solvers]`; iterable over that list.
- `KinematicSolverState` — qpos, dofs_vel, links_pos, links_quat (position-only: no velocity/accel/mass/friction).
- `RigidSolverState` — qpos, dofs_vel, dofs_acc, links_pos, links_quat, friction_ratio.
- `MPMSolverState` — pos, vel, C, F, Jp, actu, active.
- `SPHSolverState` — pos, vel, active, pressure (`None` under WCSPH, a tensor under DFSPH).
- `PBDSolverState` — pos, vel, free.
- `FEMSolverState` — pos, vel.

## Design Decisions

Entity state vs solver state are separated: solver state is the whole `[B, ...]` batch owned by one solver, entity state is a per-entity slice/view plus a back-reference and step index.
Solver state carries fields the entity view omits (MPM `actu`, SPH `active`/`pressure`, rigid `friction_ratio`), so `SimState`/reset round-trip the full solver while `entity.get_state()` returns only that entity's slice.

State vs checkpoint: `get_state()` returns live "current values" (SimState / solver state / entity state) for reading and `Simulator.reset`.
`__getstate__`/`save_checkpoint` return the serializable "boundary/initial values" (`SolverCheckpoint`/`SimulatorCheckpoint`) used by scene save/load, trajectory frames, and the differentiable backward replay.

Checkpoint selection is by `DataKind` (see `genesis/utils/data.py`): `CHECKPOINT_KINDS = {INFO, STATE, DERIVED}` for a full save; `STATE_KINDS = {STATE}` (defined in `simulator.py`) for a boundary checkpoint of the backward pass.
Trajectory frames use "exact" = {STATE, DERIVED} and "minimal" = {STATE}; a checkpoint file holds every kind but CONSTANT (scratch included).

`boundary_only=True` means: no gradients, frame 0 of framed arrays, on-device regardless of backend — used by `restore_checkpoints` to rewind the backward pass and by `StepOperation.forward` to capture the start of a differentiable step.

Zero-copy design: checkpoint arrays are copied where they live — torch tensors on device when zero-copy views exist, numpy arrays otherwise — so an in-process save+restore costs one on-device copy.
`restore_checkpoints` flushes Metal zero-copy writes with `torch.mps.synchronize()` after `load_checkpoint`.

## Serialization Pipeline

- `Scene.__getstate__` → `SceneCheckpoint(scene=desc, digest, layout, sim=SimulatorCheckpoint)`.
- `Simulator.__getstate__` → `SimulatorCheckpoint(steps, solvers={ClassName: solver.__getstate__()})`; `__setstate__` restarts (coupler/grad/rollout/sensors) then restores each solver and the steps.
- `Solver.__getstate__` = `save_checkpoint(CHECKPOINT_KINDS)`; `__setstate__` = `load_checkpoint` (rejects a record from another build via the scene digest).
- `Scene.save_checkpoint(path)` writes a `.gstraj` file (a trajectory of one frame holding every array, scratch included); `load_checkpoint`/`load_trajectory` open it and seek to the last frame.
- `.gscene` (`SCENE_FORMAT`) is the scene-description-only export; `.gstraj` (`TRAJECTORY_FORMAT`, magic `GSTRAJ4`) carries the description plus the frame arrays.

## Gotchas

- `RigidEntityState` clones `pos`/`quat` so editing it leaves the solver untouched; MPM/SPH/FEM entity states return slices/views.
  Copy semantics differ across solvers (MPM/rigid/FEM `get_state` copies via `_read_data(copy=True)`; SPH/PBD allocate fresh tensors and kernel-fill) — do not assume uniformity when mutating a returned state.
- `SimState.solvers_state` has one entry per solver in fixed order (rigid, kinematic, mpm, sph, pbd, fem), `None` for inactive solvers; `Simulator.reset` zips over `self._solvers` in the same order and restores only solvers with `n_entities > 0`.
- DERIVED arrays may be recomputed instead of restored: `KinematicSolver.load_checkpoint` runs forward kinematics when the record lacks `DataKind.DERIVED`; recomputed DERIVED can change results "at the last bits".
- Boundary checkpoints carry no gradients and no static configs (they never leave the process); a full `__getstate__` copies gradients so the autograd state round-trips.
- IPC-coupled scenes cannot be checkpointed (`Simulator.__getstate__` and `Simulator.data` raise).
- `FEMEntityState`, `KinematicSolverState`, `RigidSolverState`, `PBDSolverState`, `FEMSolverState` skip `RBC` (no pretty `__repr__`) — cosmetic only.
- No fp narrowing happens here: state arrays use the configured `gs.tc_float`/device and checkpoints copy what lives on device; the "last bits" caveat applies only to DERIVED recomputation, not to STATE copies.

## Constraints

- Data holders only: no kernels, no `get_state`/`set_state`/`save_checkpoint`/`load_checkpoint` implementations live here (they are on the solvers/entities/simulator).
- Keep imports light to avoid cycles: `entities.py` imports only `genesis.repr_base`; `solvers.py` imports `torch`/`numpy`/`dataclasses` and `genesis.utils.data.DataKind`.

## See Also

- `../solvers/base_solver.py` → `Solver.save_checkpoint`/`load_checkpoint`/`__getstate__`/`__setstate__` and `data` (read-only)
- `../simulator.py` → `SimState`/`SimulatorCheckpoint` consumers: `get_state`, `__getstate__`/`__setstate__`, `reset`, `restore_checkpoints`, `STATE_KINDS` (read-only)
- `../scene.py` → `Scene.get_state`, `__getstate__`→`SceneCheckpoint`, `save_checkpoint`/`load_checkpoint`/`load_trajectory` (read-only)
- `../entities/` → `Entity.get_state()` returns the entity-state views (read-only)
- `../../utils/data.py` → `DataKind`, `CHECKPOINT_KINDS`, `ArrayItem`/`ScalarItem`/`ConfigItem`, `fill_data`, `data_to_array` (read-only)
- `../../recorders/trajectory.py` → `.gstraj` format; frame recording reads these checkpoints (read-only)

## Status

Complete — this flat module is fully documented.

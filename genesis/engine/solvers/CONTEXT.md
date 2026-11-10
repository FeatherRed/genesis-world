## Intent
`genesis/engine/solvers/` is the physics-solver layer of the Genesis engine.
It defines the common `Solver` base class (lifecycle, state I/O, error/errno handling, autodiff hooks, checkpointing) and six concrete solvers — rigid, kinematic, MPM, SPH, PBD, FEM — plus the abstract `ParticleSolver` base.
`genesis/engine/simulator.py` owns exactly one instance of each solver and steps every active one once per substep, in a fixed order, with a coupler exchanging state between them in the middle of each substep.
`rigid/` is the project's REFERENCE implementation: the largest, most mature solver, whose `func_*`/`kernel_*` naming and Quadrants kernel structure the other solvers follow.

## File Map
- `base_solver.py` (923 lines) — `TimeBasedMixin`, `merge_errno`, `BoundaryItems`, `GravityMixin`, `Solver`, `kernel_set_gravity`, the errno kernels and `ERRNO_CHECK_INTERVAL`.
- `torch_free.py` (797 lines) — the shared read/sanitize helpers every solver uses (`numpy_dtype`, `to_numpy_values`, `qd_field`, `device_tensor_from_numpy`, `broadcast_values`, masks, `sanitize_*`, `index_field`, `index_tensor`, `envs_*`, `read_view`, `owned_view`, `owned_copy`, `is_dlpack_source`, `write_from_dlpack`, ...), plus the re-export of `dlpck_stage.py`; see "Torch-free mode" below.
- `dlpck_stage.py` (397 lines) — the DLPack staging machinery: `DlpackStager`, `stage_kernel_values` and the stream-ordered HIP copies and layout queries they issue (`_copy_after_pending_work`, `_copy_transposed_after_pending_work`, `_device_copy_plan`, `_HipAsyncCopies`, `_read_selection`, `_is_copyable_on_device`, `_host_values`, `_contiguous_strides`).
- `kinematic_solver.py` — `KinematicSolver`: forward kinematics, rendering, state get/set, Jacobians/IK for articulated entities; base of `RigidSolver`.
- `particle_solver.py` — `ParticleSolver`: base of MPM/SPH/PBD; particle state accessors + the `add_vverts` render-skinning kernel. NOT exported from `__init__.py`.
- `rigid/` — `RigidSolver` + the `abd/`, `collider/`, `constraint/` kernel subpackages (see routing table).
- `fem_solver.py` — `FEMSolver` (implicit Newton/PCG or explicit integration).
- `mpm_solver.py` — `MPMSolver` (+ module-level `signmax`, `backward_svd`).
- `sph_solver.py` — `SPHSolver` (WCSPH and DFSPH pressure solvers).
- `pbd_solver.py` — `PBDSolver` (cloth/elastic/liquid/particle materials).
- `__init__.py` — exports `FEMSolver`, `KinematicSolver`, `MPMSolver`, `PBDSolver`, `RigidSolver`, `SPHSolver` (NOT `ParticleSolver`).

## Solver Class Hierarchy
`Solver(RBC)` is the root; `RBC` is the repr base in `genesis/repr_base.py`.
- `KinematicSolver(Solver)` — `material_cls = Kinematic`, `is_differentiable = True`, no frames, no gravity, no integration rate of its own.
- `RigidSolver(GravityMixin, TimeBasedMixin, KinematicSolver)` — `material_cls = Rigid`, differentiable, no frames.
- `ParticleSolver(GravityMixin, TimeBasedMixin, Solver)` — abstract particle base (accessors + `add_vverts`).
- `MPMSolver(ParticleSolver)` — `has_frames = True`; `is_differentiable` is a property returning `not _enable_CPIC`.
- `SPHSolver(ParticleSolver)` — no frames, not differentiable (defaults).
- `PBDSolver(ParticleSolver)` — no frames, not differentiable (defaults).
- `FEMSolver(GravityMixin, TimeBasedMixin, Solver)` — `has_frames = True`, `is_differentiable = True`.

## Solver Base API & Lifecycle
Constructor chain: `__init__(scene, sim, options)` sets uid, scene/sim/options refs, `_requires_grad = sim.options.requires_grad`, `_rollout_graph`, subscribers, and the errno buffers.
`TimeBasedMixin.__init__` computes `_substeps = round(sim.dt / options.dt)` (or `sim.substeps` when `dt` unset) and `_substep_dt`; `Simulator.build` then reconciles every active `TimeBasedMixin` solver to one shared substep rate.
`GravityMixin._build_gravity()` allocates the per-env gravity buffer (reused as `RigidInfo.gravity` by the rigid solver) and provides `set_gravity`/`get_gravity`.
`build()` allocates `_errno_latched` (and a pinned host copy + CUDA event for non-blocking checks) then delegates to the subclass to build fields/entities.
`is_active` is a subclass property (`n_links > 0` for kinematic/rigid, `n_particles > 0` for MPM/SPH/PBD, `n_elements > 0` for FEM); the simulator collects `_active_solvers` from it.
Stepping is NOT a `step()` method on the solver: the simulator drives `substep_pre_coupling(f)` and `substep_post_coupling(f)` per active solver, where `f` is the local substep index (`0..substeps-1`).
`reset` is not a solver method either — `Simulator.reset` calls each solver's `set_state(0, state, envs_idx)`; solvers expose `get_state`/`set_state` and a `reset_grad()` for the backward pass.
Framed solvers (MPM, FEM) additionally expose `copy_frame(source, target)`; `TimeBasedMixin.n_frames` is `substeps + 1` under `requires_grad` else `2`.

### Error / errno handling
`register_errors(errno, messages)` declares one bit per error and its message; `update_errno()` merges the device flags into `_errno_latched` once per step (a numba-jitted `merge_errno` runs the CPU merge).
`check_errno(blocking, context)` raises the first flagged error; non-blocking CUDA checks read a pinned copy enqueued one call earlier.
`flag_errno`/`clear_errno`/`get_error_envs_mask` set/clear/read per-env flags, and the mask is boolean-valued `(n_envs,)` in both modes: `_errno_latched > 0` as a fresh `torch.bool` tensor with PyTorch, a live zero-copy `DeviceView` of the `gs.qd_bool` `_errno_mask` field without it; `ErrorCode` bits live in `genesis/utils/array_class.py`.
Without PyTorch the latch stays a `gs.tc_int` qd field, the kernels `kernel_merge_errno`/`kernel_flag_errno`/`kernel_clear_errno`/`kernel_clear_errno_flags` maintain the boolean `_errno_mask` beside it (every write casts to `gs.qd_bool`, which Apple Metal stores as `int32`), `get_error_envs_mask()` returns a `DeviceView` of it, and a non-blocking check reads the host only once every `ERRNO_CHECK_INTERVAL` (64) steps (see "Torch-free mode").
`build()` registers `ErrorCode.INVALID_GRAD_NAN` for differentiable scenes (NaN/inf adjoint of the state).

### Autodiff hooks
`requires_grad`, `is_recording`, `record_read`, `_read_data`, `add_grad_from_state`, `reset_grad` form the differentiable-simulation protocol a solver implements to join a differentiable scene.
The simulator's `RolloutGraph` records each step as one operation and reverses it by replaying kernels (`genesis/grad/bridge.py`).
Solver-specific differentiable bits: FEM wraps its implicit solve in `custom_backward(self._implicit_solve_grad, f)`; MPM uses `register_custom_backward(svd_grad)` for a closed-form SVD reverse; the rigid constraint solve uses an adjoint solve (see `rigid/`).

### Checkpointing
`data` yields the solver's arrays/scalars/configs as `DataItem`s; `save_checkpoint`/`load_checkpoint` (full and `boundary_only`) and `__getstate__`/`__setstate__` implement solver checkpoint/restore.
`BoundaryItems` caches the zero-copy views and per-item copies of a boundary checkpoint after the first one, so it never re-walks the structs.

## Driving by the Simulator
`Simulator.__init__` builds `rigid, kinematic, mpm, sph, pbd, fem` solvers (in that order in `_solvers`) and a coupler: `SAPCoupler` / `LegacyCoupler` / `IPCCoupler` (from `genesis/engine/couplers/`).
`Simulator.build` settles the shared substep rate from the active `TimeBasedMixin` solvers, builds each solver, and (for `requires_grad`) requires every active solver and the coupler to be differentiable.
Per substep, `Simulator.substep(f)` runs `coupler.preprocess(f)` -> all active `substep_pre_coupling(f)` -> `coupler.couple(f)` -> all active `substep_post_coupling(f)`.
`Simulator.step` checks errno (non-blocking), runs `_substeps` substeps, merges errno via `update_errno`, then steps sensors under `torch.no_grad`.
`_rigid_only` fast path: when only the rigid solver is active with a non-SAP/non-IPC coupler and gradients are off, the simulator skips the coupler and calls `rigid_solver.substep` directly.
The `f` frame index is `sim.cur_substep_local`; the rigid/MPM/FEM solvers read the coupler's rigid solver through `self.sim.coupler.rigid_solver`.

## Design Decisions
Unified scene/state model: every solver reads one `Scene`/`Simulator` and exposes its state as Quadrants structs (`DynState`/`DynInfo`/`RigidInfo`, `particles`, `elements_v`, ...) defined in `genesis/utils/array_class.py`.
Batch (environment) dimension is LAST in Quadrants fields (`[i_x, i_b]`), and torch-boundary views transpose it first (`qd_to_torch(..., transpose=True)` / `qd_from_torch` in `genesis/utils/misc.py`).
`gs.use_zerocopy` enables zero-copy torch views over Quadrants arrays for setters/getters and checkpoint copies (with a Metal `torch.mps.synchronize()` flush after writes).
Kernels are Quadrants DSL (`@qd.func` inlined subroutines vs `@qd.kernel` launched entrypoints); `func_*`/`kernel_*` naming is the convention established by `rigid/`.
Solver kernels reach gravity through the solver object, so gravity must be a `qd` field for them (see the `FIXME` in `GravityMixin._build_gravity`).
Particle-based solvers (`ParticleSolver`) share particle accessors and reorder particles via a spatial hash; MPM/PBD/SPH each define their own grid/hash layout.
The torch-free helper layer splits by what it does, not by caller: `torch_free.py` holds the read/sanitize side (dtypes, broadcasting, index selections and their cache, zero-copy views) and `dlpck_stage.py` the write side (DLPack staging buffers and the stream-ordered HIP copies), with a one-way module dependency `torch_free` -> `dlpck_stage` and a re-export so that callers never depend on the split.
Differentiable solvers store state in frames (`has_frames`) or use per-substep boundary checkpoints (rigid) so the backward pass can replay each substep.

## Torch-free mode
Every solver module imports PyTorch through `genesis/utils/torch_compat.py` (`HAS_TORCH`, the `torch` placeholder, `require_torch(feature)`), never as `import torch`: under `HAS_TORCH is False` the `torch` name is a placeholder, so annotations and `isinstance` checks stay valid while every executed torch call sits inside `if HAS_TORCH:` or after a `require_torch` statement.
`torch_free.py` holds the read/sanitize helpers the torch-free path shares (`numpy_dtype`, `to_numpy_values`, `device_tensor_from_numpy`, `broadcast_values`, `index_tensor`/`index_field`, `sanitize_index_values`/`sanitize_io_variables`, `envs_selection`, `read_view`, `owned_view`, `is_dlpack_source`, `write_from_dlpack`), and `dlpck_stage.py` holds the DLPack staging machinery (`DlpackStager`, `stage_kernel_values` and the stream-ordered copies they issue), so no solver repeats any of it.
The split is layered one way: `torch_free.py` reads and sanitizes what a caller hands over, `dlpck_stage.py` turns a DLPack producer into the contiguous device buffer a kernel reads, and the module-level dependency runs from the former to the latter (which re-exports every name, so `from .torch_free import DlpackStager` still works); the staging functions import the read helpers inside the call, which is the only cycle break.
Reads: `Solver._read_data` keeps `qd_to_torch(..., copy=True)` when torch is present and returns `view_of(field, transpose=True, row_mask=envs_idx, col_mask=idx)` otherwise — a zero-copy `DeviceView` (batch axis LAST in the qd buffer, `transpose=True` moves it first, never a host copy); getters that are one read return that view, and `get_links_vel`/`get_links_acc` gather through a kernel into a qd ndarray as before.
State snapshots: `Solver._read_state_values` (one array or several) and `KinematicSolver._read_state_value` (one array) read what a state snapshot restores later, so a torch-free scene reads `owned_view(field, envs_idx, idx)` — a `DeviceView` over a `qd.ndarray` filled by the `kernel_copy_values` device-to-device kernel (`owned_copy`) — while a torch scene keeps its copies; the entity accessors keep reading the live buffers through `read_view`.
Writes from a numpy source: `Solver._sanitize_io_variables` has a numpy branch returning the values as a qd tensor plus int32 index arrays, so the existing `kernel_set_*`/`kernel_control_*` entrypoints accept them unchanged; `Solver._broadcast_values` is the numpy counterpart of `broadcast_tensor`.
Writes from a DLPack producer: a qd field cannot be a `copy_from_dlpack` destination, because fields are contiguous along their first axis and so not contiguous in the batch-last layout, hence `DlpackStager.stage(src, num_values, num_envs, envs_idx=...)` copies the `(n_envs, n_values)` producer device-to-device into a cached contiguous batch-last buffer and the `kernel_*_staged` kernels of `rigid/abd/torch_free_kernels.py` scatter it into the field (including the `ctrl_mode`/`ctrl_vel` bookkeeping); it moves no data to the host, and a masked write needs the selection on the host because the copy splits into contiguous runs.
The staging copy is stream-ordered, not synchronized: `dlpck_stage._copy_transposed_after_pending_work` issues one `hipMemcpy2DAsync` per value on the stream Quadrants launches its kernels on (the process default stream, since no caller passes a `qd_stream`), so the staged write kernel is ordered after the copy by the stream itself and a write costs no host wait; `genesis.utils.device_buffer.copy_from_dlpack` stays the fallback for another backend or a producer it refuses, and its `sync=True` default is what a write must not use per step.
Index tensors are cached, not re-uploaded: `index_tensor` (`torch_free.py`) hands back the device tensor already uploaded for a selection, and `index_field`, `envs_index_tensor`, `sanitize_io_variables` (the entry and environment indices) and the `is_dof_set` masks of `RigidSolver`/`KinematicSolver` all go through it, so a steady-state write uploads no index at all; a numpy write still uploads its values, one h2d copy per `control_dofs_*`/`set_dofs_*` call, since they change at every step and nothing can cache them.
Cache rules: the key is the exact content of the sanitized int32 index values (shape + bytes), never the identity of the object holding them, so a changed selection — or an index array mutated in place — misses; the cache is an `OrderedDict` bounded to `_INDEX_TENSOR_CACHE_MAX_ENTRIES` (32) LRU entries, so a scene handing random masks to the setters cannot grow it; the tensors are read-only (kernel index arguments are always read, never written); and `_clear_index_tensors` is registered with both `qd_impl.on_reset` and `genesis.utils.cache.register_cache_clear`, so every entry dies with the Quadrants program that owns its device memory (`gs.destroy()` and a bare `qd.reset()` alike).
State staging is stream-ordered too: `dlpck_stage.stage_kernel_values` copies a DLPack source of at most two dimensions through `_copy_after_pending_work`, one `hipMemcpy2DAsync` on the same stream (`_device_copy_plan` holds the backend/dtype/layout/device checks it shares with `_copy_transposed_after_pending_work`), so `set_state`/`reset`/the particle, kinematic and FEM state setters no longer pay the `qd.sync` + device synchronizes of `copy_from_dlpack`; a third-dimension (vector) layout, another backend or a producer the bridge refuses still takes `copy_from_dlpack`, and a device-resident producer no runtime can read still gets the actionable error.
The staging copy no longer drains the device, so the ordering of a producer that writes on a queue of its own now rests on the DLPack protocol alone: the bridge asks a producer for its buffer with `__dlpack__(stream=None)`, the default stream, and a producer that ignores that request must make its writes visible itself, where the former device synchronize covered it.
`Solver.build` raises `require_torch("differentiable simulation")` for a torch-free `requires_grad` build, and the checkpoint protocol raises `require_torch("differentiable simulation checkpoint")`: frames, recording, boundary checkpoints and gradients are torch-only.
Reusable helper names: `read_view`, `broadcast_values`, `broadcast_tensor`, `device_tensor_from_numpy`, `index_field`, `index_tensor`, `envs_index_tensor`, `qd_field`, `is_dlpack_source`, `is_full_env_selection`, `envs_selection`, `mps_synchronize`, `sanitize_io_variables`, `DlpackStager`, `stage_kernel_values`, `_device_copy_plan`, `_copy_after_pending_work`, `_copy_transposed_after_pending_work`, `_clear_index_tensors`, plus `Solver._read_data`/`Solver._sanitize_io_variables`/`Solver._broadcast_values`.
Still torch-only, each raising `require_torch` with a precise feature name: `get_links_jacobian`, inverse kinematics, terrain height query, the device-arithmetic getters (`get_dofs_actuator_force`, `get_dofs_kp`/`get_dofs_kv`, the energy getters, `get_AABB`, `get_mass_mat` with a dof selection), the COM frame and the relative-pose variants with a non-identity morph offset, contact and equality-constraint introspection, `control_dofs_position_velocity` from two distinct DLPack producers, FEM vertex constraints and MPM state readout.

## Known Issues / Gotchas
FEM implicit vs explicit paths: `FEMOptions.use_implicit_solver` selects them, and under `IPCCoupler` the FEM substep is skipped entirely (IPC handles FEM).
FEM's implicit path solves to round-off in differentiable scenes (`pcg_threshold_rel` differs for f32/f64); its backward pass is the adjoint of the solve root.
MPM's `is_differentiable` is `not CPIC`: CPIC (SDF particle-cell separation) is not supported with `requires_grad` and raises at build.
MPM skips SVD on the forward pass when no registered material reads U/V/S (uses `det(F_tmp)`), but always runs SVD under `requires_grad`.
MPM/SVD backward is a hand-written `backward_svd` (PyTorch-derived) registered via `register_custom_backward` — autodiff cannot derive it.
`_sanitize_sol_params` (rigid) treats solver params as a 7-tuple with strict positive `dampratio` (actually stiffness) and unsupported negative `timeconst`.
Elliptic friction cone / box-box detection / hibernation are disabled under `requires_grad` (rigid).
Batch dim moved to the LAST index in solver data; torch-boundary transposes are the only place the layout flips.
fp64 vs fp32: tolerances and thresholds are dtype-dependent (e.g. rigid solver tolerance `1e-5` f32 / `1e-8` f64; FEM pcg threshold squared).
`check_errno` non-blocking mode raises an error one call later on CUDA (pinned-copy pipeline) — see the docstring.
The `FIXME` in `GravityMixin._build_gravity`: Quadrants cannot resolve an ndarray attribute of a `@qd.data_oriented` class in kernel scope, so gravity buffers are only allocated for solvers with entities.

## Routing Table
- `rigid/` -> `RigidSolver` and the ABD compute kernels; the reference implementation. Delegates further to `rigid/abd/`, `rigid/collider/`, `rigid/constraint/` (see its own CONTEXT.md).
- `torch_free.py` -> the shared read/sanitize helpers (`read_view`, `owned_view`, `broadcast_values`, `index_field`, `index_tensor`, `qd_field`, `sanitize_io_variables`, `is_dlpack_source`) that every solver and every entity caller routes through, and the re-export of the staging machinery (this level).
- `dlpck_stage.py` -> `DlpackStager`, `stage_kernel_values` and the stream-ordered HIP copies of a DLPack producer into a kernel argument; reached through `torch_free.py` or imported directly (this level).
- `base_solver.py` -> common solver base, mixins, errno, checkpointing (this level).
- `kinematic_solver.py` -> articulated FK/state accessors shared with the rigid solver (this level).
- `particle_solver.py` -> shared particle accessors/skinning for MPM/SPH/PBD (this level).
- `mpm_solver.py`, `sph_solver.py`, `pbd_solver.py`, `fem_solver.py` -> the four other concrete solvers (this level).
- `../simulator.py` -> owns the solvers and coupler and drives the substep loop (sibling — read-only).
- `../couplers/` -> `SAPCoupler`, `LegacyCoupler`, `IPCCoupler` inter-solver coupling (sibling — read-only).
- `../entities/` -> entity classes each solver builds (`RigidEntity`, `MPMEntity`, `SPHEntity`, `FEMEntity`, `PBD*Entity`) (sibling — read-only).
- `../../utils/array_class.py` -> state structs (`DynState`, `RigidInfo`, `ErrorCode`, particle/element fields) (read-only reference).
- `../../utils/misc.py`, `../../utils/data.py` -> `qd_to_torch`/`qd_from_torch`/`qd_to_numpy` transpose + zero-copy conventions, `fill_data` (read-only reference).
- `../../grad/bridge.py` -> `RolloutGraph`, `custom_backward`, `register_custom_backward`, `no_grad` (read-only reference).
- `../../options/solvers.py` -> `*Options` user-facing solver configuration (read-only reference).
- `../states/solvers.py` -> `*SolverState` classes returned by `get_state` (read-only reference).

## Notes for Agents
In this sandbox `rg` sometimes returns only one match per file and plain `git grep -n` prints nothing; use `git grep -o -n -E "pattern"` to list definitions with line numbers, and the `read` tool (raise `max_bytes` to 131072 for large files; long lines are clipped).
Treat the `rigid/CONTEXT.md` (and its `abd/`, `collider/`, `constraint/` children) as authoritative for that subtree; keep this file a routing table plus cross-cutting lifecycle/layout/convention notes.

## Test Strategy
No unit tests live under `solvers/`; physics and differentiable-simulation correctness is exercised by end-to-end scene/example tests elsewhere in the repo.
Relevant regression themes (see git log): batch dim moved to last index, zero gradient through joints at rest in diff sim, fp64 vs fp32 tolerance differences, implicit FEM solve round-off.

## See Also
`genesis/engine/simulator.py` (substep/coupler order), `genesis/engine/couplers/` (SAP/Legacy/IPC), `genesis/grad/bridge.py` (autodiff), `genesis/utils/array_class.py` (state structs), `genesis/options/solvers.py` (user surface).

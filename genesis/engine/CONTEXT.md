# genesis/engine/ — Core Physics Engine

## Intent
Unified multi-physics simulation engine of Genesis World (`genesis-world` v1.4.6).
It runs rigid-body, FEM, MPM, PBD and SPH (particle) solvers over one shared scene and one shared physics state, couples them between solvers with SAP/Legacy/IPC couplers, and supports end-to-end autodiff through Quadrants kernels plus a recorded rollout graph.
This node is the engine library: seven top-level orchestration modules plus seven subpackages.
`__init__.py` is empty; public names are re-exported by the package (`genesis/__init__.py`) and by each subpackage's own `__init__.py`.

## API Surface (top-level modules)
- `scene.py` — `Scene` (user-facing API: `build`, `add_entity`, `add_force_field`, `add_sensor`, `add_recorder`, `add_camera`, `add_emitter`, `add_light`, `reset`, `step`, `backward`, `get_state`, `export`/`load`, `save_checkpoint`/`load_checkpoint`/`load_trajectory`, plus `draw_debug_*`) and the dataclasses `SceneDescription`, `EnvironmentLayout`, `SceneCheckpoint`, `TrajectorySource` and `description_digest`.
- `simulator.py` — `Simulator` (owns exactly one solver of each kind plus the coupler and the sensor manager, and drives the substep loop), `StepOperation` (the differentiable step operation of the rollout), `restore_checkpoints` (zero-copy boundary restore helper).
- `interactive_scene.py` — `InteractiveScene` (thread-safe async editing wrapper that applies pause/step/rebuild intent on the stepping thread and rebuilds a Scene in place), `InteractiveFeature` (editing capabilities enum).
- `state_change.py` — `StateChange`, `MutatedLinks`, `Subscriber`, `mutates` (the mutation-tagging decorator that drives both subscriber notifications and differentiable rollout recording).
- `force_fields.py` — `ForceField` base plus `Constant`, `Wind`, `Point`, `Drag`, `Noise`, `Vortex`, `Turbulence`, `Custom`, `PerlinNoiseField` (acceleration fields evaluated inside solver kernels).
- `bvh.py` — linear BVH (LBVH) construction and box query over AABBs (Karras 2012), `BVHData`; consumed by rigid broadphase, raycaster and tactile sensors.
- `mesh.py` — `Mesh` (Genesis' own `trimesh.Trimesh` wrapper with convexify/decimate/UV support), `InertialProperties`, `SDFGrid`, `SupportTable`, and the `cached_by_tmesh`/`mutates_tmesh` decorators.

## Routing Table
- `entities/` → live entity layer: rigid, FEM, MPM, PBD, SPH, particle, emitter and hybrid entities plus their `*Description` dataclasses.
- `solvers/` → solver implementations and kernels; `solvers/rigid/` is the reference implementation (abd / collider / constraint split).
- `materials/` → constitutive models per solver family (FEM/, MPM/, PBD/, SPH/).
- `sensors/` → camera, depth, IMU, contact/joint force, raycaster, tactile, temperature and probe sensors plus `SensorManager`.
- `couplers/` → SAP, Legacy and IPC inter-solver couplers.
- `boundaries/` → `CubeBoundary` (particle/MPM domain box).
- `states/` → serializable entity/solver state and checkpoint dataclasses.

## Design Decisions
A scene is authored as a `SceneDescription` (options + a list of `EntityDescription`); `Scene.build` resolves descriptions once, instantiates live entities, and each entity only indexes into its solver's buffers.
Each entity is simulated by exactly one solver, chosen in `Simulator._add_entity` from the solver whose `material_cls` is the most-derived superclass of the entity's material (`Rigid` derives from `Kinematic`, so `RigidSolver` wins over `KinematicSolver`).
`Simulator` owns a fixed solver list in order rigid, kinematic, mpm, sph, pbd, fem, plus one coupler; the per-substep pipeline is `coupler.preprocess(f)` → every active solver's `substep_pre_coupling(f)` → `coupler.couple(f)` → every active solver's `substep_post_coupling(f)`.
Substep rates are reconciled at `Simulator.build`: a solver may ask for its own `dt`/substep count, but all active solvers must integrate at the same rate.
The batch (environment) dimension is the LAST index in Quadrants fields, with a torch-boundary transpose; `gs.use_zerocopy` zero-copy writes are flushed on Metal after restore.
Autodiff replays a recorded rollout: `state_change.mutates` tags state-mutating methods, `RolloutGraph` records them as operations, and `Simulator._reverse`/`StepOperation.backward` rewind substeps via boundary checkpoints and `LaunchRecorder`.
The rigid solver is the project's reference implementation for kernel structure and naming: `func_*` are `@qd.func` inlined device subroutines, `kernel_*` are `@qd.kernel` launched entrypoints; the other solvers follow this convention.
Force fields are acceleration fields, not forces (no notion of spatial density); they are activated/deactivated and evaluated per-particle/per-entity inside solver kernels.

## GPU-Resident Stepping (host/device traffic)
`Scene.get_error_envs_mask` (`scene.py:1111`) delegates to `Simulator.get_error_envs_mask` (`simulator.py:380`), which allocates a fresh `torch.zeros((B,), dtype=torch.bool, device=gs.device)` and ORs each `Solver.get_error_envs_mask()` (`base_solver.py:323`, itself a fresh `_errno_latched > 0`); the result is always a new device bool tensor, never a live view.
The live per-env buffer behind it is `Solver._errno_latched` (`base_solver.py:471`, `gs.tc_int` on `gs.device`), merged once per step by `Solver.update_errno` at the end of `Simulator.step`; `Scene.clear_errno`/`Solver.clear_errno` zero it per env.
On CUDA every `Simulator.step` opens with `check_errno(blocking=False)` (`simulator.py:400`, `base_solver.py:363-373`): `_errno_event.synchronize()` plus a read of the pinned `_errno_host` copy enqueued on the previous step, which keeps the host one step behind the device; only solvers with registered errors (rigid, FEM, MPM) pay it, CPU merges numpy views (`_errno_views`), and pbd/sph/kinematic-only scenes skip it.
The engine exposes no done/termination/reset mask and no episode counter; training loops derive termination from entity/sensor reads — only `Simulator.steps` (`simulator.py:556`, the live `_steps` int tensor `[B]` on `gs.device`, incremented in place per step and zeroed per env by `reset`) and `get_time` (`simulator.py:629`, fresh `tc_float` device tensor, scalar when `n_envs == 0`) are env bookkeeping.
There is no `Scene.sync()`/`Simulator.sync()`; synchronization is implicit — `qd_to_torch(copy=True)`, the errno event, and `torch.mps.synchronize()`/`qd.sync()` on Metal only (CUDA is never synchronized explicitly).
Entity state getters pass `copy=True` into `Solver._read_data` (`base_solver.py:520`, e.g. `kinematic_solver.py:1253`), so a per-step read returns a fresh device tensor (an allocation, no host transfer); control setters are device-side too — with `gs.use_zerocopy` (default on CUDA) they are in-place torch writes on zero-copy views (`rigid_solver.py:2620`), otherwise they launch `kernel_control_dofs_position`.
Per-step order: errno check → `_rigid_only` fast path (direct `rigid_solver.substep`) or `_step()` → `substep()` = `coupler.preprocess` → active solvers' `substep_pre_coupling` (rigid, kinematic, mpm, sph, pbd, fem order) → `coupler.couple` → `substep_post_coupling` → `clear_external_force` → errno merge → `SensorManager.step`; `Scene.step` wraps that with pre-step callbacks (which can veto), the recorder read and the viewer update.

## Torch-Free Operation
The whole engine imports PyTorch through `genesis/utils/torch_compat.py` (`HAS_TORCH`, placeholder `torch`, `require_torch`); no engine module does `import torch`.
Torch-free mode is `HAS_TORCH is False`: `gs.use_zerocopy` is `ZeroCopyMode(True)`, `gs.tc_float`/`tc_int`/`tc_bool` are Quadrants primitive types, and `gs.device` is a `DeviceInfo` whose `.type` reports `"cuda"` on ROCm.
Reads: `Solver._read_data` returns `view_of`/`read_view` `DeviceView`s (batch axis moved first), so entity getters hand back zero-copy device views.
State snapshots (`get_state`) do not alias: they copy through `torch_free.owned_copy`, a device-to-device `@qd.kernel` copy, which is what lets `scene.reset()` and `set_state` restore real values.
Writes: numpy sources take the numpy branch of `Solver._sanitize_io_variables` into the existing `kernel_*` entrypoints; any DLPack producer (`__dlpack__`/`__dlpack_device__`, e.g. a compiled-graph output shaped `(B, n)`) is staged device-to-device by `DlpackStager` with stream-ordered `hipMemcpy2DAsync` on the default HIP stream, which is the stream Quadrants launches on, so a write costs no host wait.
`set_state`-style sources go through `torch_free.stage_kernel_values`, which still synchronizes once per array.
errno stays device-resident: `_errno_latched` and its mask are qd fields, `get_error_envs_mask()` returns a `DeviceView` of the flag buffer (int32, nonzero = flagged), and `check_errno(blocking=False)` reads the host once every `ERRNO_CHECK_INTERVAL = 64` steps instead of every step.
Helpers live in `engine/torch_free.py` (counters, errno-mask kernels, index fields), `engine/solvers/torch_free.py` (read, sanitize, staging) and `engine/entities/torch_free.py` (index/mask).
Torch-only by design, raising `require_torch("<feature>")`: differentiable scenes/recording/checkpoints, the viewer, cameras and lights, recorder/export paths, all sensors, Jacobian/IK/energy/terrain queries, FEM vertex constraints, MPM readout and Metal zero-copy.
Measured torch-free on ROCm (MI300X, fp32): Go2 plane scene at B=16 runs 1000 steps in ~3.6 ms/step and at B=4096 ~7.2 ms/step, bit-exact against the torch path, with zero device synchronizations per step for numpy and DLPack control writes.

## Constraints
Solver/entity state setters must be tagged with `@mutates` (with the right `StateChange` category, `links` reach and `differentiable`/`derived` flags), or sensors and the rollout graph will miss mutations.
Kernel code targets both CPU and GPU via Quadrants; device-specific branches use `gs.backend` (cpu/cuda/metal) and `gs.use_ndarray` (performance mode).
Precision must follow `gs.qd_float`/`gs.tc_float` (fp32/fp64), not hardcoded float types.
Cross-package writes belong to the owning directory; engine modules may read `genesis/options`, `genesis/utils`, `genesis/grad`, `genesis/vis`, `genesis/recorders` but not modify them.

## Known Issues
IPC-coupled scenes cannot be checkpointed yet (`Simulator.data`/`__getstate__` raise): the external `libuipc` world holds its own state.
`SAPCoupler` requires `precision='64'` and the implicit FEM solver, and is not differentiable.
Muscle actuation exists only for `FEM.Muscle`/`MPM.Muscle` materials; entity `assert_muscle` raises otherwise.
`stable_neohookean` (FEM) has no Hessian, so implicit FEM is unsupported for it.
`MPM.Base._default_Jp` stores solver state on the material object (tech debt).
`boundaries/boundaries.py` leaks imported names (`np`, `qd`, `gs`, ...) because `__init__.py` does `from .boundaries import *` with no `__all__`.

## Dependencies
In-repo: `genesis/options` (Options groups), `genesis/utils` (array_class, data, misc), `genesis/grad` (bridge: RolloutGraph/LaunchRecorder), `genesis/vis` (renderers/viewer), `genesis/recorders` (RecorderManager/Trajectory), `genesis/repr_base` (RBC base), `genesis/typing`.
External (beyond pip): `quadrants` (kernel/data-oriented framework, `qd`), `trimesh`, `numpy`, `torch`.
Optional/backend-specific: `pyuipc` + `libuipc` (CUDA) for the IPC coupler; a renderer backend for camera sensors.

## Test Strategy
Engine behavior is covered by the sibling `tests/` tree, not inside this node.
Relevant suites: `tests/rigid/` (collision, constraints, control, islands, kinematics, serialization, mujoco parity), `tests/particles/` (mpm/pbd/sph), `tests/deformable/` (fem/muscle), `tests/coupling/`, `tests/sensors/`, `tests/grad/` (rollout and rigid grad), `tests/ipc/`, `tests/core/test_bvh.py`, `tests/benchmarks/test_rigid.py`, `tests/integration/`.
Gradient tests gate `requires_grad=True` support per solver/coupler via `is_differentiable`.

## Notes for Agents
This sandbox's `rg` tool may return only one match per file; use `git grep -o -n -E "pattern"` to list top-level definitions with line numbers, and the `read` tool to read files.
The rigid solver (`solvers/rigid/`) is the place to learn kernel/naming conventions before touching other solvers.
Each child directory's CONTEXT.md holds the detailed per-module API; drill down before re-investigating a module.

## See Also
- `../` (`genesis/` package root) — options, utils, grad bridge, vis, recorders that the engine consumes.
- `../../tests/` — the test suites mapping to this engine (sibling, read-only).
- `../grad/` — the rollout graph and launch recorder driving engine autodiff.
- `../ext/LuisaRender` — external renderer submodule (outside this node's scope).

## Status
Complete: the context tree below this node (entities, solvers, materials, sensors, couplers, boundaries, states and their children) is established.

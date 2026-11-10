# genesis/ — The `genesis` Package (user-facing API)

## Intent
The Python package behind Genesis World (imported as `import genesis as gs`, package name `genesis-world` v1.4.6).
It exposes the user-facing simulation interface and assembles the three backend layers — physics (`engine/`), render (`vis/` + `ext/`), compiler (external `quadrants`) — behind one `gs.Scene` object.
This node documents the package root (top-level modules + `logging/` + `assets/`); the heavy subsystems live in the child directories below.

## API Surface (top-level modules)
- `__init__.py` — the public namespace and lifecycle: `gs.init(...)` / `gs.destroy()`, `gs.register_external_module(init_fun, destroy_fun)` / `unregister_external_module`, `gs.GenesisException`.
  It configures Quadrants + torch, sets backend-dependent dtypes (`gs.qd_float`/`gs.np_float`/`gs.tc_float`, `gs.qd_vec3`, `gs.qd_mat3`, int/bool equivalents), zero-copy mode, and precision (32/64).
  PyTorch is optional: it is imported through `utils/torch_compat.py` (`gs.HAS_TORCH`, `gs.require_torch`, the annotation-safe `gs.torch` placeholder), and torch-free mode sets `gs.tc_*` to their Quadrants counterparts and enables zero-copy on every backend but Metal (see the PyTorch note below).
  It re-exports: `gs.Scene`, `gs.Mesh`, `gs.states`, `gs.materials`, `gs.force_fields`, `gs.morphs`, `gs.sensors`, `gs.renderers`, `gs.surfaces`, `gs.textures`, `gs.recorders`, `gs.tools`, `gs.UID`, `gs.List`, `gs.utils.geom.*`, misc helpers (`raise_exception`, `assert_built`, ...), and the constants enums (`gs.JOINT_TYPE`, `gs.GEOM_TYPE`, `gs.CTRL_MODE`, `gs.integrator`, `gs.constraint_solver`, `gs.friction_cone`, `gs.contact_resolution`, `gs.broadphase_traversal`, `gs.link_ref_frame`, `gs.backend` with members `cpu/gpu/cuda/amdgpu/metal`).
  It also re-exports the ROCm HSA runtime helpers `gs.ensure_system_hsa_runtime` / `gs.inspect_hsa_runtime`, `gs.init` calling the former before any GPU work (see `gs.utils.HsaRuntimeStatus` and the ROCm HSA note below).
- `_main.py` — the `gs` CLI (pyproject `[project.scripts] gs = "genesis._main:main"`): `gs launch` / `gs view` (deprecated alias) to inspect an asset in the viewer, `gs play` (ImGui joint controls), `gs replay` (`.gstraj` trajectory), `gs animate` (images → video).
- `constants.py` — file-format constants (`URDF_FORMAT`, `MJCF_FORMAT`, `MESH_FORMATS`, `USD_FORMATS`, `XACRO_FORMAT`, `GLTF_FORMATS`) and the enums above (rigid solver geometry/joint/equality/control types, integrator/constraint/friction/contact-resolution/broadphase options, `backend`, `IMAGE_TYPE`, `PARA_LEVEL`, `ACTIVE`/`INACTIVE`).
- `datatypes.py` — `gs.List[T]`: a pydantic-compatible, richly-repr'd list used everywhere for entity/link/geom collections.
- `repr_base.py` — `RBC` (REPR Base Class): every Genesis class inherits it to get the colorized, property-listing `__repr__` and `__repr_name__` (`<gs.engine.entities.RigidEntity>` style).
- `styles.py` — log theming: `THEME` (`dark`/`light`/`raw`), the `~kind<...>~` log markup parser, `COLORS`/`FORMATS`, and the module-level `style`/`colors`/`formats` singletons.
- `typing.py` — pydantic `Annotated` type aliases used by `options/` (`Vec3FType`, `UnitVec3FType`, `RotationMatrixType`, `FrozenDictType`, ...) with strict validators (vector normalization, int coercion rejecting bool/float); `_to_float_grid` accepts a nested sequence, an array, a torch tensor or a `DeviceView`, so the grid aliases work torch-free.
- `version.py` — `__version__ = "1.4.6"`.
- `logging/` → `Logger`, `GenesisFormatter`, `TimeElapser` (see its CONTEXT.md).

## Routing Table
- `engine/` → core physics engine: Scene/Simulator, entities, solvers, materials, sensors, couplers, boundaries, states.
- `utils/` → shared helpers: geometry math, mesh/asset parsers (URDF/MJCF/SDF/USD/GLTF), array/cache/tools; `misc.py`/`geom.py` are the canonical helpers every module must use.
- `vis/` → visualization: Visualizer, Viewer, Camera, Rasterizer, Raytracer, BatchRenderer, viewer plugins.
- `ext/` → vendored third-party code: pyrender fork, urdfpy fork, isaacgym terrain utils, `_trimesh_patch`, and the `LuisaRender/` gitlink submodule (render backends).
- `options/` → pydantic option classes (SceneOptions, solver/coupler/sensor/renderer/morph/surface/texture options) consumed by `engine/` at build time.
- `recorders/` → recording/export: video/CSV/NPZ writers, `.gstraj` trajectories, plotters, RecorderManager.
- `grad/` → differentiable-simulation bridge: RolloutGraph, LaunchRecorder, kernel-backward decorators.
- `logging/` → logger + elapsed-time helper (tiny, documented here).
- `assets/` → bundled static data: meshes, textures, URDF/MJCF robot models, sample XML scenes (data only, no Python).

## Design Decisions
Scenes are authored declaratively (`Scene` + morph/mesh/material/surface/sensor/recorder options) and resolved into solver buffers once at `Scene.build` — descriptions (options) are plain pydantic dataclasses, live entities only index into solver state.
One `Simulator` owns exactly one solver of each kind plus one coupler, so rigid/FEM/MPM/PBD/SPH share one scene and one state; substeps are reconciled to a single rate.
All hot-path computation runs as Quadrants kernels (Python lowered to CUDA/ROCm/Metal/Vulkan/x86/ARM); the `gs.backend` enum abstracts the device and `gs.use_ndarray` toggles static (performance) vs dynamic (fast-compile) array mode.
Autodiff replays a recorded rollout (`grad/`) rather than in-place autograd through the C++ solver.
`RBC`/`styles` give Genesis its distinctive colorized, markup-tagged console output (`~<value>~`, `~~<name>~~`, `~~~<uid>~~~`, `~~~~<title>~~~~`).

## Constraints
Coding conventions are enforced project-wide (see `../CLAUDE.md`, `../CODING_GUIDELINES.md`, `../.github/contributing/`): the rigid solver (`engine/solvers/rigid/`) is the reference implementation for naming and kernel structure.
No `getattr`/`hasattr`, no on-the-fly attributes, no plain dicts for attribute packing, no commented-out prints, no `detach().cpu().numpy()` (use `utils.misc.tensor_to_array`), no local imports unless required to break cycles.
All state-mutating engine methods must be tagged `@mutates` so sensors and the rollout graph see mutations.
External code in `ext/` does not follow these conventions (vendored).

## Dependencies
Beyond pip (see `../pyproject.toml`): `quadrants==1.4.0` (compiler/kernels), `torch` (install separately, per-backend), `trimesh>=5.0.0`, `mujoco>=3.2.5`, `numpy`.
Optional backends: `pyuipc`/`libuipc` (IPC coupler), `usd-core`/`pxr` (USD), LuisaRenderPy (ray tracer, built from `ext/LuisaRender` submodule), `gs-madrona` (Linux batch renderer), `rsl-rl-lib` (RL examples).

## Test Strategy
Package behavior is covered by the sibling `../tests/` tree, not inside this node (`tests/rigid/`, `tests/particles/`, `tests/deformable/`, `tests/sensors/`, `tests/grad/`, `tests/parsers/`, `tests/rendering/`, ...).
Run `pytest -m required` (minimal merge set); default addopts exclude `benchmarks`/`examples` markers and run `--numprocesses=auto --dist=worksteal --import-mode=importlib`.

## Notes for Agents
`genesis/ext/LuisaRender` and `doc/` are git submodules that arrive as EMPTY placeholders in worktrees; populate with `git submodule update --init` if their content is needed (never delete the placeholder or commit inside them).
`assets/` is ~hundreds of binary files (meshes/textures/robot models) — treat as static data; do not refactor or split it.
Each child directory's CONTEXT.md holds the detailed per-module API; drill down before re-investigating.

### PyTorch is optional
`import genesis` and `gs.init`/`gs.destroy` work with no PyTorch installed; every module imports PyTorch through `genesis/utils/torch_compat.py` (`HAS_TORCH`, `torch`, `require_torch`, `no_grad`, `is_torch_tensor`) instead of `import torch`.
Torch-free, `gs.device` is a `genesis.utils.device_buffer.DeviceInfo` (`type`/`index`) rather than a `torch.device`, the accessors return zero-copy `DeviceView`s instead of torch tensors, and `gs.tc_float`/`gs.tc_int`/`gs.tc_bool` are the Quadrants dtypes (`gs.qd_float`/`gs.qd_int`/`gs.qd_bool`).
Zero-copy is enabled by default on `cpu`/`cuda`/`amdgpu` because the DLPack device views exist there, and `gs.init` raises a GenesisException on `metal`, unless `GS_ENABLE_ZEROCOPY=0` disables it.
What stays torch-only: debug mode (`torch.use_deterministic_algorithms`, asked for with `require_torch("deterministic algorithms in debug mode")`), the `GS_TORCH_FORCE_CPU_DEVICE` fallback, the `torch<2.8`/`torch<2.9.1` version warnings (their flags `_IS_OLD_TORCH`/`_TORCH_MPS_SUPPORT_DLPACK_FIELD` default to `False`), and the gfx12 HIP mempool check, which queries `torch.cuda.get_device_properties()`.
### Device-buffer interop for compiled graphs (torch-free)
Torch-free, the entity getters `get_pos`/`get_quat`/`get_vel`/`get_ang`/`get_dofs_position`/`get_dofs_velocity` and `scene.get_error_envs_mask()` return `gs.utils.DeviceView` objects instead of torch tensors, while the setters (`control_dofs_position`, `set_qpos`, `set_dofs_*`) accept any object exposing `__dlpack__`/`__dlpack_device__`.
The cross-residency contracts below are the SAME with and without PyTorch, so a caller never has to know which residency it runs in (see `../tests/test_torch_free_training.py` for the bit-identical state-boundary check).
- `set_qpos(qpos, qs_idx=None, envs_idx=None, *, skip_forward=False)` takes, in precedence order, a single `(1, *element_shape)` row broadcast to every selected environment, one row per selected environment of shape `(len(envs_idx), *element_shape)` in the order of `envs_idx` (the reading that wins when `len(envs_idx) == n_envs`), or the whole `(n_envs, *element_shape)` tile under a partial selection, where row `e` is the value written into environment `e` and only the environments named by `envs_idx` are written.
  `envs_idx` accepts None, an int index array (host or device), a slice, and a boolean mask (host or device) in both residencies, and any other row count raises with one message naming the three forms.
  The torch path gathers the named rows out of the value and scatters them, while the torch-free path stages the whole batch-first tile on the device with no host roundtrip and `kernel_set_qpos_rows_staged` picks each environment's row; zeroing the velocity, skipping or running the forward kinematics and clearing the contacts then behave identically.
- `set_dofs_*` through `_set_dofs_info` accept the same `envs_idx` contract in both residencies: with `RigidOptions(batch_dofs_info=False)` (the default) the value is written to the named dofs of every environment and the selection is accepted and ignored, exactly as torch does, while with `batch_dofs_info=True` each environment carries its own values and a masked write selects the environments it names.
  A torch-free scene hands its kernel `index_field(None, n_envs, "envs_idx")` where the selection is None, because a `qd.types.ndarray()` parameter takes no None, so no caller-side `batch_dofs_info=not HAS_TORCH` workaround is needed.
- `scene.get_error_envs_mask()` (and the solver-level `get_error_envs_mask()`) return a boolean-valued `(n_envs,)` result: `torch.bool` with PyTorch, and without it a `DeviceView` of a `gs.qd_bool` field whose `dtype` is `np.bool_`, whose `to_numpy()` is a host `bool` array and whose DLPack capsule reports `kDLBool` (8-bit, one lane).
  The mask is latched per environment since its last reset, a flagged environment raises at the start of the next step, and `scene.clear_errno()` accepts the returned mask or an index selection.
Torch-free getters return a `DeviceView` with the batch axis leading and the element axes trailing: `(n_envs, n_items)` with strides `(1, n_envs)` for a scalar element, `(n_envs, n_items, k)` with strides `(k, n_envs * k, 1)` for a vector element.
`get_pos`/`get_quat`/`get_ang`/`get_dofs_position`/`get_dofs_velocity` and the link getters are live windows on the solver buffer, while `get_vel`, `get_links_vel` and `get_links_acc` return a freshly allocated per-call snapshot (~0.45 ms/call) whose values do not follow a later step.
Read selections accept None, an int, a slice, a range and a contiguous ascending index collection; the setters additionally accept permutations and boolean masks.
`qd.sync()` is required before a consumer on a stream of its own (IREE/evoxc) reads a `DeviceView` after `scene.step()`: Quadrants launches kernels asynchronously on the default stream, a DLPack export enqueues nothing, and `DeviceView.__dlpack__(stream=...)` ignores the stream.
Measured, unfenced post-step reads returned stale values 39/40 times at 2 envs and 40/40 at 4096 envs, `qd.sync()` costs 0.11-0.17 ms per step, and `to_numpy()`/`contiguous()` fetch on their own.
`gs.utils.view_of(buffer, transpose=True, row_mask=..., col_mask=...)` builds the view — the torch-free analogue of `qd_to_torch`, with the same layout semantics: the batch axis is the last grid axis, `transpose=True` moves it to the front and leaves the element axes trailing; masks accept only slices and contiguous ascending indices.
`DeviceView.to_dlpack()`/`__dlpack__()`/`__dlpack_device__()` export a zero-copy DLPack v0 capsule at the exact source pointer (`device_type=10` on ROCm), including for buffers that do not start at their allocation origin and whose own `Field.to_dlpack()` would need PyTorch; `to_numpy()` is an explicit host copy for logging and diagnostics only.
`gs.utils.copy_from_dlpack(dst, src, *, transpose=False, envs_idx=None, idx=None, sync=True)` is the graph→Genesis direction: no cast, no host roundtrip, one `hipMemcpy` for a whole contiguous run and `min(num_values, num_envs)` `hipMemcpy2D` calls for a transposed batch-last destination; `sync=True` (the default) drains the Quadrants stream before and synchronizes the HIP device after, so the copy is complete on return.
Quadrants 1.4.0 has no DLPack importer, so the device-to-device write is `genesis/utils/hip_copy.py`, a guarded ctypes shim over `libamdhip64.so` (`hipMemcpy`/`hipMemcpy2D`, `hipMemcpyDeviceToDevice`) that is only used for a source whose device matches `gs.device`; a mismatch raises instead of silently round-tripping through the host, and a CPU backend applies the same selection semantics through NumPy.
`stage_kernel_values` copies a DLPack producer device to device when its DLTensor is device-resident, compact, at most 2-D and of the destination dtype, reading the layout out of the producer's own capsule when the Python object hides it (an evoxc/IREE `core.Tensor` exposes `shape`/`dtype`/`__dlpack__` but no `strides`); anything else takes the host path, which refuses a device-resident producer with a message naming the forms a setter accepts.
`copy_from_dlpack(sync=True)` spends ~10-13 us of fence per copy: the trailing `hipDeviceSynchronize()` is redundant for a Quadrants consumer, the pre-copy `hipDeviceSynchronize()` is what covers a producer writing on a foreign stream (a lone `qd.sync()` does not), so the copies of one step may share one `qd.sync()` plus one `hipDeviceSynchronize()` and run with `sync=False`.
IREE — and therefore any compiled-graph runtime — refuses a non-contiguous DLTensor with `Unsupported strided tensor`, so a transposed or masked solver view must be materialized with `DeviceView.contiguous()` (one allocation plus a device-to-device copy per call) or copied into a caller-owned contiguous buffer with `copy_from_dlpack(scratch, view, transpose=True, envs_idx=...)`, which is the lower-overhead pattern for a hot loop.
Compiled-graph runtimes take their operands in their own tensor type and the DLPack protocol is the only contract: `genesis/` never imports `evoxc`, `tensorneat` or `iree`, and the environment layer owns the wrapping (for evoxc/IREE: `capsule = view.__dlpack__(stream=None)`, `hal_view = exe.backend_executable.runtime_device.from_dlpack_capsule(capsule)`, then `evoxc.core.Tensor(IreeDevicePayload(iree.runtime.DeviceArray(hal, hal_view), exe.device))`).
The interop is gated on `gs.use_zerocopy` (on by default for `cpu`/`cuda`/`amdgpu`, with `metal` refused at `gs.init`), and a recording (autodiff) scene disables zero-copy, so the torch-free read direction is unavailable there.
`performance_mode=True` is supported without PyTorch: it sets `gs.use_ndarray = False`, so `V()` allocates every solver buffer as a `qd.Backend.FIELD`, and the getters, the DLPack setters and `scene.get_error_envs_mask()` work on field-backed buffers (`solvers.torch_free.owned_copy` dispatches its device-to-device copy on the kind of the buffer).
A field-backed scene and an ndarray-backed one are each deterministic but not bit-identical (Go2, 1000 steps: qpos 4.6e-06, dof velocity 5.5e-05 between array modes, the same order with or without PyTorch) because a struct-for over a field ignores `serialize` on the GPU (quadrants#1082), which reorders the floating-point reductions.
Measured on the MI300X in this configuration: 352 steps/s torch-free at B=16 (2.84 ms/step, `scene.step` 2.38 ms, DLPack control write 0.277 ms vs 0.599 ms for a numpy host write) and a 200 steps/s closed graph→setter→step loop at B=64 with bit-exact target landing.

### Device-memory interop (Quadrants ↔ torch)
Solver state lives in Quadrants device buffers and reaches torch only through DLPack; there is no `__cuda_array_interface__` and no Python-level `data_ptr` on `qd.Tensor`.
`genesis.utils.misc.qd_to_torch(field, copy=False)` returns a torch tensor that aliases the Quadrants buffer and caches it on the qd impl as `_tc`/`_T_tc`; entity/scene getters call it with `copy=True` (a fresh device clone — never a host copy).
`Field`/`Ndarray` export DLPack via `to_dlpack()` / `to_torch(copy=False)`; the `qd.Tensor` wrapper does not itself implement `__dlpack__`, so call `to_dlpack()` or use the cached torch view.
On ROCm (`qd.init(arch=qd.amdgpu)`) Quadrants memory appears to torch as `torch.device("cuda", 0)` and its DLPack device is `(kDLROCM=10, 0)`; `gs.device` is `torch.device("cuda", 0)` and `gs.use_zerocopy.is_supported` is True.
`gs.tc_float`/`gs.tc_int`/`gs.tc_bool` are float32/int32/bool (Quadrants f32/i32/bool), while `gs.use_ndarray` defaults to True; recording (autodiff) disables zero-copy, so `copy=False` views are unavailable in a recording scene.

### ROCm HSA runtime (AMD Instinct MI300X VF)
On this host torch 2.13.0+rocm7.1 bundles `torch/lib/libhsa-runtime64.so` (version 1.18.70101, sha256 `10ac797047ff6f153551295adda65c2ab5ab83fcaf7e2a7699d1ea81427c33c8`), which corrupts AQL packets after roughly 50-600 steps, surfacing as `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` -> `hipErrorLaunchFailure` in Quadrants' `hipStreamSynchronize`, and sometimes as a SIGSEGV.
The system `/opt/rocm/lib/libhsa-runtime64.so.1.18.70100` (ROCm 7.1.0, sha256 `e45cf51a79392a6033623e95a705408bb72a3b66998c80f2e400e0e8716926e3`) does not have this defect.
`genesis/utils/hsa.py` detects the mismatch (`inspect_hsa_runtime` / `system_hsa_runtime_path`, pure detection that never imports torch beyond `find_spec`, never raises and never re-executes) and `ensure_system_hsa_runtime` repairs it.
`gs.init` on a GPU/ROCm backend calls `ensure_system_hsa_runtime` right after the logger exists and strictly before any `get_device` probing or `qd.init`, so the repair precedes all torch/quadrants GPU work; a user scene therefore needs no `LD_PRELOAD` prefix and no example-local shim.
A `ctypes.CDLL(..., RTLD_GLOBAL)` preload is NOT equivalent: the symbols bind but the abort persists, because `LD_PRELOAD` must be in place before the dynamic loader maps torch's HIP/HSA libraries, which is why a one-shot `os.execve` of the current interpreter is required.
`GS_HSA_REEXEC=0` opts out (with one warning naming both libraries) and `GS_HSA_REEXEC_DONE=1` is the internal one-shot sentinel that keeps the re-exec from ever looping.
The re-exec is warn-only under pytest (`PYTEST_VERSION` or `pytest` in `sys.modules`), `python -m`, interactive/IPython and embedded interpreters, and never happens for `gs.cpu`/`gs.metal`, for a missing torch ROCm wheel, or once the system runtime is already preloaded.
The cost is one extra interpreter start on the affected host only, since the parent already paid `import genesis` (measured baseline 13.0 s / 3.2 GB for `import genesis` on this host).

## See Also
- `../README.md`, `../pyproject.toml`, `../CLAUDE.md`, `../CODING_GUIDELINES.md` — project overview and conventions.
- `../tests/` — test suites (sibling, read-only).
- `../examples/` — runnable demos (sibling, read-only).

## Status
Complete: package root and all child subpackages (`engine/`, `utils/`, `vis/`, `ext/`, `options/`, `recorders/`, `grad/`, `logging/`, `assets/`) are documented.

# genesis/utils — Intent

Shared utility layer of the Genesis physics engine (`genesis-world`).
It provides the canonical tensor/array conversion helpers, geometry math in three backend variants, declarative solver data structs, asset parsers (URDF/MJCF/glTF/mesh/SDF/terrain/USD), on-disk + in-memory caches, and assorted tooling (media, path planning, particle sampling, serialization).

`genesis/utils/__init__.py` is `from .misc import *`, so the `misc` helpers are the public face of `gs.utils` and are re-exported package-wide.

# Routing Table

- `usd/` → USD stage parser (Pixar `pxr` binding): `parse_usd_stage` / `parse_usd_rigid_entity`, `UsdContext`, material baking, collision filtering. Own CONTEXT.md.
- `../engine/` → consumer of `data.py` (the declarative struct layer) and owner of the concrete solver structs (`solvers/rigid/data.py`, `solvers/rigid/collider/data.py`, `solvers/rigid/constraint/data.py`) — sibling, read-only.
- `../options/` → `morphs` (Mesh/Terrain/USD, etc.) and `Options` consumed by parsers — sibling, read-only.
- `../typing.py`, `../constants.py`, `../repr_base.py` → shared types/constants/repr base imported throughout — siblings, read-only.
- `../ext/` → vendored `urdfpy`, IsaacGym terrain utils, etc. — sibling, read-only.
- `../../tests/` → test suites (parsers under `tests/parsers/`) — sibling, read-only.

# API Surface (flat modules)

Canonical helpers & misc:
- `misc.py` — the canonical helpers every module must use: `tensor_to_array`, `tensor_to_cpu`, `data_to_array` (torch→numpy without `detach().cpu().numpy()`), `qd_to_torch`/`qd_to_numpy`/`qd_from_torch`/`qd_destroy_data` (Quadrants↔torch/numpy zero-copy), `raise_exception`/`raise_exception_from`, `redirect_libc_stderr`, `ZeroCopyMode`, `set_environ`, `assert_initialized`/`assert_built`/`assert_unbuilt`/`with_lock`, `set_random_seed`, `get_device`, `get_assets_dir`/`get_src_dir`/`get_gen_log_dir`/`get_entry_point_name`, `unique_rows` (numba), `make_tensor_field`/`concat_with_tensor`, `indices_to_mask` (it reads a boolean selection as one only when it is a torch tensor: a host NumPy or list mask is cast to an `int64` collection of its 0/1 flags), `geometric_mean`/`harmonic_mean`, screen helpers (`get_default_screen`, `has_display`), Quadrants monkey-patches (`_stringify_obj_type_with_grad`, `_kernel_impl_with_adjoint_callable`, `_try_load_fastcache_by_autodiff_mode`, `_kernel_reset_with_template_args`, `_release_dispatcher_args`, `_perf_dispatcher_init_released_on_reset`), and the indexed-tensor helpers (`sanitize_index`, `get_last_occurrence`, `sanitize_indices`, `broadcast_tensor`, `sanitize_indexed_tensor`, `get_indexed_shape`, `assign_indexed_tensor`).
  `misc` also re-exports the device-buffer bridge (`DeviceView`, `view_of`, `copy_from_dlpack`, `DeviceInfo`, `device_of`, `dlpack_info`, `qd_ptr`, `zero_copy_supported`), which `genesis/utils/__init__.py` (`from .misc import *`, no `__all__`) turns into `gs.utils.*`.
- `torch_compat.py` — the single import site for PyTorch: `HAS_TORCH`, `torch` (module-like placeholder whose attributes are `_Unavailable` sentinel types, so annotations, unions, subscriptions and `isinstance` keep working while calling one raises an actionable Genesis error), `require_torch(feature)`, `no_grad()` (context manager and decorator), `is_torch_tensor(x)`.
- `device_buffer.py` — the read half of the torch-free zero-copy bridge: `DLTensorInfo`/`dlpack_info`, `DeviceInfo`/`device_of`, `DeviceView`/`view_of` (with `contiguous()`/`reshape` materialization), `zero_copy_supported`, the DLPack capsule owners, and re-exported `copy_from_dlpack`/`qd_ptr`.
- `hip_copy.py` — the write half: `copy_from_dlpack`, `qd_ptr`, `_materialize_view` (copying a strided view into a contiguous buffer), `_host_array` (reading a producer's description in place), the HIP runtime shim (`hipMemcpy`/`hipMemcpy2D`, device synchronize) and the NumPy host fallback (see the write-path notes below).
- `warnings.py` — `warn_once` (deduplicated logger warning).
- `emoji.py` — `random_scene`, `get_clock` (CLI/progress decoration).
- `deprecated_module_wrapper.py` — `_DeprecatedModuleWrapper` / `create_virtual_deprecated_module` to alias renamed modules with a warning.
- `repr.py` — `brief` / `__repr_name__` used by the `RBC` repr system for compact nested reprs.
- `tools.py` — `animate` (moviepy video), `save_img_arr`, and timing utilities `Timer`/`create_timer`/`Rate`/`RunningMean`/`StepTimer`.

Math & geometry:
- `geom.py` — the canonical geometry module, with three parallel families: `qd_*` (`@qd.func`, device kernels incl. gradients), numpy `_np_*` / torch `_tc_*` backends, and dispatchers (`xyz_to_quat`, `quat_to_R`, `transform_by_quat`, `slerp`, `polar`, etc.) that accept a `DeviceView` on their numpy path. Also `SpatialHasher`, `qd_hash01` (deterministic counter hash), `cubic_spline_1d`, `orthogonals`, `default_solver_params`, `default_friction*`, `default_dofs_kp/kv`, `GeomConstants`/`_tc_geom_constants` (precomputed torch constant tensors per device/dtype).
- `linalg.py` — `@qd.func` small-matrix kernels: `mat_mul`, `mat_mul_vec`, `mat_inverse` (LU), `mat_add`, `mat_transpose`, `mat_add_eye`, `mat_mask`.
- `simt.py` — `@qd.func` SIMT block/segment primitives (`qd_block_sum/scan/min/max/shuffle/ballot/all/any`, lane-slot ops, `qd_segmented_sum/min`, `qd_segment_add/min`).

Declarative solver data:
- `data.py` — the declarative struct layer: `Data`, `StaticConfig`, `DataKind`/`CHECKPOINT_KINDS`, `field`, `opt_in`/`is_opted_in`, `Size`/`SizeExpr`/`_SizeArithmetic`, `FieldRef`, `Namespace`/`namespace`, `V` (qd.tensor allocator honoring `gs.use_ndarray`), `eoa` (env-major axis order), `ENVS`/`WORKERS`, `iter_data`/`iter_state_arrays`/`check_data`/`fill_data`, `ArrayItem`/`ScalarItem`/`ConfigItem`/`DataItem`.
- There is no `array_class.py`: the concrete solver structs (`RigidInfo`/`RigidData`, `DynInfo`/`DynState`, `DofsInfo`/`DofsState`, `LinksInfo`/`LinksState`, `ErrorCode` subclasses, collider/constraint structs) live with their components under `../engine/solvers/rigid/`.
- `ring_buffer.py` — `TensorRingBuffer` (preallocated circular buffer of torch tensors, view-returning `at`, `set`/`rotate`/`get`/`__getitem__`, host-resident index for perf); refusing without PyTorch.

Caching:
- `cache.py` — cache-directory layout under `~/.cache/genesis` (override via `GS_CACHE_FILE_PATH` / `XDG_CACHE_HOME`); per-domain `get_*_cache_dir`/`get_*_cache`/`get_*_path` helpers (gsd/gnd/cvx/ptc/fps_pc/tet/gel/rm/wt/wth/exr/usd); in-memory `SizeCappedCache` (byte-bounded LRU), `OwnedCache` (entries live while owners do), `MeshCache` (hash-keyed pickle files), `register_cache_clear`/`clear_caches` teardown hooks, `xxh64`/`get_hashkey`, cache-version constants (`WT_CACHE_VERSION`, etc.).

Mesh & asset geometry:
- `mesh.py` — trimesh/igl/coacd/tetgen wrapper: color/roughness converters, `get_signed_volume`, `convex_hull`, `get_wall_thickness`, `MeshInfo`/`MeshInfoGroup`, `load_trimesh`/`load_mesh`/`parse_mesh_trimesh`/`trimesh_to_mesh`, `compute_sdf_data`, `surface_uvs_to_trimesh_visual`, `convex_decompose`, `postprocess_collision_geoms`, texture helpers (`create_texture`, `PIL_to_array`, `tonemapped`, `adjust_alpha_cutoff`), primitive creators (`create_sphere/cylinder/cone/arrow/line/box/plane/frame/camera_frustum`), `create_tets_mesh`/`tetrahedralize_mesh`/`visualize_tet`, `apply_transform`, `check_exr_compression`.
- `watertighten.py` — `watertighten_mesh`: triangle-soup → watertight wrap via SDF dual contouring + feature-preserving quadric-error decimation; also `decimate_mesh`, numba inner loops, adaptive pitch params.
- `gltf.py` — glTF/GLB loader (`parse_mesh_glb`, `parse_glb_tree`, `parse_glb_material`, `get_glb_data_from_accessor`, Draco decode via `DracoPy`, `pygltflib`), with an `OwnedCache` of decoded GLB images.
- `sdf.py` — `SDF` grid builder for rigid geoms + `@qd.kernel`/`@qd.func` SDF query kernels (`sdf_kernel_init_geom_fields`, `sdf_func_*` world/local/grad/normal/coarse lower-bound).
- `terrain.py` — `parse_terrain` (IsaacGym-style heightfield generator), `fractal_terrain`, `convert_heightfield_to_watertight_trimesh`, `mesh_to_heightfield`.
- `point_cloud.py` — furthest-point sampling (`furthest_point_sample`, `sample_mesh_point_cloud` with disk cache + normals).
- `element.py` — `mesh_to_elements` (tetrahedralize with `.tet` cache), `split_all_surface_tets` (hydroelastic contact).
- `particle.py` — particle count/placement helpers, `trimesh_to_particles_simple`/`trimesh_to_particles_pbs`, `box/cylinder/sphere/shell_to_particles`, `particles_to_mesh`, `FoamGenerator`/`filter_surface` (SPH-ish foam), `_cubic_spline_kernel` helpers.
- `hybrid.py` — skeletonization via `pygel3d` (`load_hmesh`, `trimesh_to_gelmesh`, `skeletonization`), `networkx` graph reduce/attributes, `Arrow3D`/`plot_nxgraph` visualization.

Robot parsers (return `(l_infos, links_j_infos, links_g_infos, eqs_info)` for the rigid-entity description):
- `urdf.py` — `get_robot_name`, `load_xacro`, `order_links_depth_first`, `parse_urdf`, `parse_equalities`, `merge_fixed_links`, inertia transforms (`translate/rotate_inertia`, `principal_axes_rot`, `compose_inertial_properties`, `merge_inertia`, `transform_inertial`), `update_subtree`.
- `mjcf.py` — `get_model_name`, `build_model` (mujoco compile), `parse_xml`, `parse_link`/`parse_links`, `parse_geom`/`parse_geoms`, `parse_equalities`; builds on `urdf.py` for xacro/URDF inputs.

Raycasting:
- `raycast.py` — host-side `Ray`/`RayHit` NamedTuples and `plane_raycast`.
- `raycast_qd.py` — `@qd.func`/`@qd.kernel` BVH ray-cast kernels (`bvh_ray_cast`, `kernel_cast_rays`, `kernel_cast_rays_visual`, triangle/ray/AABB primitives, `kernel_refresh_*_bvh`); see Known Issues for the duplicated collision/visual paths.

Serialization & media:
- `serialization.py` — scene archive format (JSON `scene.json` + `arrays.npz`): `export`/`load`, `SerializationMixin`, `register`/`_class_codec`, `Exporting`/`Exported`/`Loading`, `schema`/`declared_schema`, `source_digest`, `pixel_less_textures`; files carry no code/module paths (type-driven).
- `image_exporter.py` — `as_grayscale_image` and `FrameImageExporter` (parallel multi-camera/env export).
- `video_encoder.py` — threaded `VideoEncoder` with `_probe_h264_codec` (hardware-first codec selection: videotoolbox/nvenc/vaapi/qsv/libx264) and a bounded frame queue.

Other:
- `path_planning.py` — `PathPlanner` (ABC), `RRT`, `RRTConnect`, `align_waypoints_length`, `rrt_valid_mask`, `rrt_connect_valid_mask` (Quadrants kernels driven from torch tensors, so they refuse without PyTorch; FREE/SPHERICAL joints unsupported).
- `uid.py` — `UID` (`full`/`short`/`match`, uuid4-hex based).
- `hsa.py` — ROCm HSA runtime repair: `HsaRuntimeStatus`, `inspect_hsa_runtime` / `system_hsa_runtime_path` (pure detection of torch's bundled `libhsa-runtime64.so` vs. a differing system `libhsa-runtime64.so.1.*`, with versions, sha256 candidates from `$ROCM_PATH`/`$ROCM_HOME`/`/opt/rocm`/`/opt/rocm-*`), and `ensure_system_hsa_runtime` (one-shot `os.execve` with `LD_PRELOAD`, gated by `GS_HSA_REEXEC` opt-out and the `GS_HSA_REEXEC_DONE` one-shot sentinel).

### Torch-free mode
`torch_compat`, `device_buffer` and `hip_copy` are the foundation of running Genesis without PyTorch: none of them imports torch or `genesis` at module scope.
Torch-free mode is `HAS_TORCH is False`: `gs.use_zerocopy` is then `ZeroCopyMode(True)`, `gs.tc_float`/`gs.tc_int`/`gs.tc_bool` are Quadrants primitive types, `gs.device` is a `DeviceInfo` and `gs.use_ndarray` is True.
`torch_compat.torch` is the real module when PyTorch is installed, and a placeholder whose attributes are `_Unavailable` sentinel classes otherwise.
Every module of this node imports PyTorch through `torch_compat`; `torch` is never imported directly, so a module-level `import` costs nothing when PyTorch is absent and annotations/isinstance checks stay legal.
A helper either has a torch-free implementation (`qd_to_numpy`, `tensor_to_array`, `tensor_to_cpu`, `data_to_array` without `to_torch`, `make_tensor_field` (a `qd.field`), `set_random_seed` (python/numpy only), `indices_to_mask(to_torch=False)`, `data.fill_data`, `geom` dispatchers given numpy or a `DeviceView`, `get_device`, `get_gpu_cores_per_unit`/`get_gpu_core_count`/`get_gpu_shared_tile_sizes`) or refuses with `require_torch("<feature>")`: `qd_to_torch`, `qd_from_torch`, `qd_add_grad`, `concat_with_tensor`, `broadcast_tensor`, `sanitize_index`/`sanitize_indices`/`get_last_occurrence`, `sanitize_indexed_tensor`/`get_indexed_shape`/`assign_indexed_tensor`, `TensorRingBuffer`, the RRT planner, `DeviceView.to_torch`, and `data_to_array(..., to_torch=True)`.
`get_device` keeps returning a `torch.device` when PyTorch is present; torch-free it returns a `DeviceInfo` for `gs.cpu` and for `gs.amdgpu`/`gs.gpu` (name from `hipDeviceGetName`, total memory from `hipMemGetInfo`, sysfs and then `0.0` with one warning), and raises for `gs.cuda`/`gs.metal`.
The torch-free GPU helpers query HIP through `hipDeviceGetAttribute` (compute unit count, threads per unit, warp size, shared memory per unit and per opt-in block) with plausibility bounds and the documented fallbacks (64 cores per AMD compute unit, 256 compute units), which only feed tile-size heuristics.
`device_buffer` reads Quadrants buffers through DLPack capsules (`dlpack_info`/`DeviceView`) and writes them through the HIP runtime or NumPy, since Quadrants has no DLPack importer.
The public names are `gs.utils.DeviceView`, `gs.utils.DeviceInfo` (`type`/`index`/`dlpack_device_type`), `view_of(value, *, transpose=False, row_mask=None, col_mask=None, keepdim=True) -> DeviceView`, `copy_from_dlpack(dst, src, *, transpose=False, envs_idx=None, idx=None, sync=True) -> None`, `dlpack_info(value) -> DLTensorInfo`, `device_of(value) -> DeviceInfo`, `zero_copy_supported() -> bool` and `qd_ptr(value) -> int`.
`DeviceView` exposes `shape`, `strides`, `dtype`, `device`, `ndim`, `grid_ndim`, `element_ndim`, `to_dlpack(versioned=False)`, `__dlpack__(stream=None)`, `__dlpack_device__()`, `to_numpy()` (host copy, diagnostics only), `to_torch(copy=False)` (requires PyTorch), `contiguous()`, `reshape`, `transpose`, `movedim` and `__getitem__`.
`DeviceInfo` reports type `"cuda"` for both CUDA and ROCm, and `dlpack_device_type` distinguishes them (`2` vs `10`) by resolving the active backend from `gs.backend`, falling back to the Quadrants architecture when `genesis` is not importable.
`_raise_genesis_error` raises `gs.GenesisException` whenever `genesis` is importable and an `ImportError` carrying the same message otherwise, because `genesis/__init__.py` still imports PyTorch eagerly.
`DeviceView.to_numpy()` is an explicit host copy for logging and diagnostics only; solver getters must use `DeviceView`/`view_of` on the hot path.
`copy_from_dlpack` refuses dtype, device, shape and non-contiguous mismatches with a `gs.GenesisException` instead of converting or gathering silently.
The write path lives in `hip_copy.py`: `copy_from_dlpack(dst, src, *, transpose=False, envs_idx=None, idx=None, sync=True)` takes a batch-last Quadrants destination and a DLPack producer (or NumPy array), uses one `hipMemcpy` for a contiguous same-layout copy, `min(num_values, num_envs)` `hipMemcpy2D` calls for a transposed copy and one contiguous transfer per selected outer index for a masked copy, with no host roundtrip on a GPU backend; on the CPU backend the same selection semantics run through NumPy and are written back with `from_numpy`.
`sync=True` (the default) does three HIP calls around the transfer — `qd.sync()`, `hipDeviceSynchronize()`, the copy, then `hipDeviceSynchronize()` again — because the producer may still have writes pending on its own queue; `sync=False` leaves that ordering to the caller (see the measured fence costs below).
A `DeviceView` carries the element offset, shape and strides of its region (never the data), so `view_of(value, *, transpose=False, row_mask=None, col_mask=None, keepdim=True) -> DeviceView` mirrors the `qd_to_torch` layout semantics without a copy; selections are limited to slices and contiguous ascending index runs, and anything else is refused rather than silently materialized.
`DeviceView.element_ndim` counts the trailing axes holding the vector or matrix of one element and `grid_ndim` the rest, so `transpose=True` moves the batch axis (`grid_ndim - 1`) to the front and the masks select over grid axes only, leaving a `(m, n)` grid of a vector buffer as `(n, m, 3)` rather than `(3, m, n)`.
`device_buffer` re-exports `copy_from_dlpack`, `qd_ptr`, `_index_runs` and `_selected_indices` from `hip_copy` through `_LAZY_REEXPORTS`/`__getattr__`/`__dir__`, so a caller may import the write direction from either module.
Torch-free `geom` dispatchers accept a `DeviceView` and take their numpy path through `to_numpy()`, since a `DeviceView` is a host-side helper argument, not a hot-path buffer.
`data.fill_data` writes numpy/`DeviceView`/Quadrants values through `from_numpy` or `copy_from_dlpack` (device-resident contiguous views matching the destination dtype) and refuses an unknown value type with a `GenesisException`; the torch branch keeps its `to_torch`/`from_torch` zero-copy path.
`dlpack_info` describes a Quadrants buffer whose own DLPack export requires PyTorch, which the C++ layer demands for every buffer at a nonzero byte offset, from the buffer itself: the base address of its SNode tree comes from the zero-offset leading node of that tree, plus the SNode byte offset, the canonical shape and strides, the dtype and the device.
`DeviceView.to_dlpack()` builds its own v0 capsule for such a buffer, so `copy_from_dlpack`, `DeviceView.to_torch` and `np.from_dlpack` keep working torch-free, while a v1 export request is refused.
A DLPack consumer that requires contiguity refuses a strided tensor outright — IREE says 'Unsupported strided tensor' — and a batch-last solver buffer read with `transpose=True` is always strided, its strides being `(1, num_envs)`.
`DeviceView.contiguous()` satisfies that requirement on the device: it returns `self` when the view is contiguous already, and otherwise a NEW `DeviceView` over a fresh `qd.ndarray` it owns in `_buffer`, filled by `hip_copy._materialize_view` on ROCm and by NumPy on the CPU, with `qd.sync()` before the copy (and after it on ROCm) so that every write already launched is part of the snapshot and the values are complete for a consumer on any stream.
A materialized view owns its Quadrants buffer, which lives as long as the view and the views derived from it and is freed with them or at `qd.reset`, so a consumer may hold the capsule for the rest of its own call.
The copy is a single `hipMemcpy` when the source is contiguous, `min(n, B)` 2-D `hipMemcpy2D` calls when it is the `(B, n)` -> `(n, B)` transpose of a batch-last buffer, and a per-row or per-element loop only for a layout neither case describes; a Metal backend raises instead of copying.
The hot-loop pattern keeps one preallocated contiguous `qd` destination buffer and calls `copy_from_dlpack(scratch, view, transpose=True, envs_idx=...)` on it every step, instead of materializing the view with `contiguous()`, which allocates a fresh buffer and syncs on every call.
`DeviceView.reshape` materializes a non-contiguous view first, as NumPy copies when a reshape needs it, so reshaping a transposed view gives the contiguous flattening of the transposed values.
An exported capsule carries a NULL destructor by design: CPython destroys a capsule as soon as its last reference dies, which happens while a consumer's exception unwinds, and a ctypes callback cannot run then — it would report the refusal as 'SystemError: error return without exception set' and swallow the real 'ValueError'.
The owner keeps a strong reference to its capsule instead, the managed-tensor deleter pops the owner entry through `manager_ctx`, and `qd.lang.impl.on_reset(_release_capsule_owners)` sweeps the capsules no consumer ever took.
`copy_from_dlpack` parses the producer capsule once: `device_buffer._dlpack_of` returns the description plus the capsule that pins the buffer, and the host path builds its array in place with `_host_array(info)` instead of a second `np.from_dlpack(src)` export, so a producer's `__dlpack__` runs exactly once per copy.

### Verified torch-free accessor contract (MI300X VF, ROCm 7.1, quadrants 1.4.0)
Measured on a torch-free plane+box scene (`PYTHONPATH=/tmp/notorch:$PWD python`, `"torch" not in sys.modules`, `gs.init(backend=gs.amdgpu)`): n_links=2, n_dofs=6, f32, `gs.use_ndarray=True`.
The batch axis is the last axis of the solver field, whose shape is `(n_items, n_envs)`, so 'read_view' transposes it to the front; a vector element keeps its components trailing: `(n_envs, n_items, k)` with strides `(k, n_envs*k, 1)`.
`RigidEntity.get_pos`/`get_quat`/`get_vel`/`get_ang` return `(n_envs, 3|4)`, and `get_dofs_position`/`get_dofs_velocity` return `(n_envs, n_dofs)`; `envs_idx` shrinks only the leading axis and leaves the strides untouched.
Measured `n_envs=2` (identical at 4096 apart from sizes): `get_pos` `(n_envs, 3)` strides `(3, 1)` contiguous, `get_quat` `(n_envs, 4)` strides `(4, 1)`, `get_ang` `(n_envs, 3)` strides `(3, 1)`, `get_vel` `(n_envs, 3)` strides `(3, 1)`, `get_dofs_position`/`get_dofs_velocity` `(n_envs, 6)` strides `(1, n_envs)` and therefore NOT contiguous.
A selection gives `get_pos([1])` `(1, 3)` strides `(3, 1)` and `get_dofs_position(envs_idx=[1])` `(1, 6)` strides `(1, n_envs)`; at `n_envs=1` the dofs view is `(1, 6)` with strides `(1, 1)`, which `is_contiguous` calls False (the row-major check counts the size-1 axis) even though NumPy would call it contiguous.
The read path accepts `None`, an int, a slice, a `range` and a contiguous ascending index collection (a boolean mask only when the flagged environments are contiguous) and refuses `[2, 1]`, `[0, 2]` or `[True, False, True, False]` with a `GenesisException`; the write path (`copy_from_dlpack(envs_idx=...)`) accepts every one of those forms, splitting a permutation or a strided selection into one transfer per contiguous run.
Entity getters always return a derived view (`is_base_view == False`, the `first_row` selection being an operation); `solver.get_links_pos()` with no link selection returns `(n_envs, n_links, 3)` strides `(3, n_envs*3, 1)` with `element_ndim == 1`.
`get_pos`, `get_quat`, `get_ang`, `get_dofs_position` and `get_dofs_velocity` are LIVE windows on solver state (`read_view`): the same `data_ptr` on every call, and new values after `scene.step()`.
`get_vel`, `get_links_vel` and `get_links_acc` are OWNED snapshots: a fresh device buffer per call filled by `kernel_get_links_vel`/`kernel_get_links_acc`, unchanged afterwards, at ~430-480 us/call versus ~24-40 us for the live getters at both 2 and 4096 envs.
The velocity views report `element_ndim == 0` (a plain f32 ndarray, not a vector field) while the pose/ang views report `element_ndim == 1`, so `view_of(vel_view, transpose=True)` would move the component axis.
State snapshots read through `owned_view` (`owned_copy` by `kernel_copy_values`, then `read_view`), so `solver.get_state().links_pos` is a `DeviceView` over a copy that keeps its values while the solver steps on.
`scene.get_error_envs_mask()` returns a `DeviceView` of dtype bool, shape `(n_envs,)`, strides `(1,)`.

### Read-after-write ordering (measured)
Quadrants 1.4.0 launches every `qd.kernel` on the default/legacy stream (handle 0) and returns immediately: 500 launches cost 13.5 ms of Python, and a 10M-iteration serial kernel returns from the call in 0.05 ms with `qd.sync()` then blocking 190 ms.
`qd.sync()` drains that default stream only (`quadrants.lang.runtime_ops.sync` states that streams from `qd.create_stream()` are not waited on); it costs ~1.2 us idle and 0.11-0.17 ms after a `scene.step()`.
DLPack export enqueues nothing: `DeviceView.to_dlpack()` takes ~17-24 us while a 190 ms kernel is in flight, and `DeviceView.__dlpack__(stream=...)` ignores the stream, so no cross-stream event handshake happens.
Verdict: `qd.sync()` (the only fence Quadrants 1.4.0 exposes — there is no `qd.synchronize()`) is REQUIRED before a DLPack consumer reads a `DeviceView` on its own non-blocking stream; it is redundant only for a consumer issuing a blocking `hipMemcpy` on the default stream, which legacy-stream rules already order.
Evidence: a `hipMemcpyAsync` D2H on a `hipStreamNonBlocking` stream issued right after an in-flight 190 ms kernel read the previous value (stale) in 0.06 ms, while the same read through a blocking `hipMemcpy` on the default stream blocked 190 ms and read the fresh value.
Evidence at scene level: after `scene.step()`, unfenced non-blocking-stream reads of `get_pos()` were stale 20/50 at n_envs=1, 39/40 at n_envs=2 and 40/40 at n_envs=4096 once the state changes every step.
`DeviceView.to_numpy()` is self-fencing (the Quadrants host copy blocks: 190 ms of a 190 ms kernel) and `contiguous()` calls `qd.sync()` before its copy, so both are safe to read without any caller-side fence.

### `copy_from_dlpack` sync points (measured)
With 6 values, `sync=True` costs 44.5-66.8 us per call and `sync=False` 34.6-54.0 us from n_envs=1 to 4096, so the fence set is ~10-13 us per copy.
Dropping the trailing `hipDeviceSynchronize()` is safe: with `sync=False` the next Quadrants kernel read the copied values 100/100 times, because `hipMemcpy`/`hipMemcpy2D` block on the host and run on the default stream.
Dropping the leading `qd.sync()` is safe for a Quadrants-produced source (fresh 100/100, a blocking `hipMemcpy` being ordered after the default stream already) but it is what protects a destination that pending Quadrants kernels still read or write.
Dropping the leading `hipDeviceSynchronize()` is unsafe for a producer on a foreign stream: a producer kernel on a `qd.create_stream()` stream left the destination stale 13/20 with `sync=False` and 13/20 with an explicit `qd.sync()` alone, and 0/20 with `sync=True`.
One `qd.sync()` plus one `hipDeviceSynchronize()` per step followed by `sync=False` copies is therefore safe and cheaper: 8 copies cost 443 us instead of 524 us at n_envs=16.
`genesis/engine/solvers/dlpck_stage.py` already works that way from the other side — it issues `hipMemcpy2DAsync` on the default stream and never synchronizes — which is ordered for the kernels around it but carries no fence for a producer writing on a foreign stream.

### `DeviceView.contiguous()` cost (measured)
`contiguous()` allocates on every non-contiguous call: ~110-120 us for a `(n_envs, 6)` transposed view at n_envs=1/2/64/4096, of which ~61-65 us is the `qd.ndarray` allocation plus `reshape`, 7-23 us `_materialize_view` and ~1.4 us `qd.sync()`; a contiguous view returns `self` in ~4 us.
The cost follows the layout rather than the size until the 3-D fallback: a `(n_envs, n_links, 3)` transposed vector view (the shape `get_links_pos()` returns) costs 176 us at n_envs=2 and 38.3 ms at n_envs=4096, because `_materialize_view` then copies one row per `hipMemcpy` call (8192 calls).

# Constraints

- Never import `torch` directly in this node: import `HAS_TORCH`, `torch` and `require_torch` from `genesis.utils.torch_compat`, and give every executed torch call an `HAS_TORCH` branch or a `require_torch` refusal.
- `torch_compat`, `device_buffer` and `hip_copy` must import with torch blocked: no module-scope torch import, and no module-scope `genesis` import beyond the deferred error path.
- The device-buffer bridge must never convert, gather or cast silently: dtype, device, shape and contiguity mismatches raise a `GenesisException`.
- Never use `tensor.detach().cpu().numpy()` in Genesis code — use `misc.tensor_to_array`/`tensor_to_cpu`/`data_to_array`.
- Torch/Quadrants conversions go through `misc.qd_to_torch`/`qd_to_numpy` (zero-copy) and `qd_from_torch`; `qd_destroy_data` pairs with them.
- Flat modules here are one level: only `usd/` is a child subdirectory; do not add subpackages to this node without routing them here.
- Solver struct field order is load-bearing: `Data` allocates arrays in declaration order and kernels consume the structs by name, so do not reorder `genesis/engine/solvers/rigid/**/data.py` fields without a solver/kernel audit.
- The Quadrants monkey-patches in `misc.py` are load-bearing for autodiff/fast-cache/dispatcher behaviour — do not remove without understanding the `quadrants` internals they patch.

# Design Decisions

- One geometry API, three backends: `geom.py` keeps numpy (`_np_*`), torch (`_tc_*`), and Quadrants (`qd_*`) implementations in sync behind a single public name, so host-side parsing and device-side simulation share the same conventions.
- Solvers are data-driven: `data.py` provides the declarative layer and each component declares its structs/fields once under `../engine/solvers/rigid/`; kernels take those structs and are cached/compiled on equality of `StaticConfig`.
- `Data` allocates every leaf array through `V()` on the current Quadrants device, resolves `float`/`int`/`bool` to `gs.qd_float`/`gs.qd_int`/`gs.qd_bool` (set in `genesis/__init__.py` at `gs.init`), turns an element shape such as `(float, 3)` into `qd.types.vector(3, gs.qd_float)`, and derives `needs_grad` from the field's `grad=True` and the constructor's `requires_grad`; the `ENVS` dim is declared last but stored env-first in memory on CPU by default.
- Caches are two-tier: content-addressed on-disk caches (hash-keyed, version-bumped on algorithm change) plus owner/byte-bounded in-memory caches released on teardown via `register_cache_clear`.
- Parsers converge on a common rigid-entity output tuple rather than a shared IR, keeping each format's quirks local.
- `misc.py` re-exported via `__init__.py` makes conversion helpers available everywhere as `gs.utils.*`.
- PyTorch is optional, not required: every module of this node reaches it through `torch_compat`, and the simulation core plus the device-buffer bridge need only Quadrants and NumPy, so a helper is either given a torch-free implementation or a `require_torch` refusal, never an implicit dependency.
- The device bridge splits by direction: `device_buffer.py` reads (DLPack `DeviceView`, no copy) and `hip_copy.py` writes (HIP runtime on a GPU backend, NumPy on the CPU), which keeps each half small and lets the read path stay usable without any runtime to load.
- Quadrants has no DLPack importer, so the write path goes through a HIP copy or `from_numpy`, and the read path exports a view rather than materializing a buffer.

# Known Issues

- `raycast_qd.py` carries duplicated collision vs visual BVH traversal kernels (`bvh_ray_cast` vs `bvh_ray_cast_visual`, etc.); a FIXME documents that Quadrants cannot accept either of two dataclass shapes for one `qd.func` arg, forcing parallel copies.
- `data.py` works around upstream Quadrants limitations documented in FIXMEs: a kernel reads an ndarray's layout through private `_qd_layout`; a struct-for over a field ignores `serialize` on GPU; `qd.ndarray` takes no memory layout (so `V()` allocates ndarrays through `qd.tensor`); a kernel cannot take `None` for a struct argument (unused arrays become 0-d placeholders); Metal silently mis-addresses tensors above 2 GiB.
- `mjcf.py` importing `mujoco` with `MUJOCO_GL=EGL` forcibly defines `PYOPENGL_PLATFORM=egl` (documented side effect).
- `quat_to_xyz(..., rpy=True)` exists only in the torch/numpy backends (`qd_quat_to_xyz` implements the non-`rpy` intrinsic x-y-z convention only); `rpy=True` angles of `q` equal the negated non-`rpy` angles of `inv_quat(q)`.
- `_tc_quat_to_xyz` applies the `gs.EPS` gimbal-lock test to the `|q|^2`-scaled cosine, while `_np_quat_to_xyz`/`qd_quat_to_xyz` test the normalized cosine, so non-unit quaternions near pitch ±90° can take different branches per backend.
- Heavy optional dependencies are imported lazily inside functions (e.g. `pygel3d` in `hybrid.py`, `moviepy` in `tools.animate`), so module import stays cheap but the deps are still required at call time.
- `hsa.py` must re-exec the interpreter rather than preload with `ctypes.CDLL(..., RTLD_GLOBAL)`, because `LD_PRELOAD` has to be in place before the dynamic loader maps torch's HIP/HSA libraries; a runtime `ctypes` preload is provably insufficient.
- `ensure_system_hsa_runtime` may only run once from a fresh interpreter before any torch/quadrants GPU work; `GS_HSA_REEXEC=0` is the opt-out and `GS_HSA_REEXEC_DONE=1` is the one-shot sentinel that stops the child from looping.
- `misc.py` is deliberately one large module (~1900 lines) because `genesis/utils/__init__.py` exposes it wholesale as `gs.utils`, and its Quadrants monkey-patches, torch-free GPU queries and conversion helpers are imported by everything; split it only with a re-export plan that keeps `gs.utils.*` stable.
- Quadrants' own `field.to_dlpack`/`ndarray.to_dlpack` runs a C++ `import torch` whenever the buffer sits at a nonzero byte offset, so every buffer after the first of a process needs the torch-free description fallback in `dlpack_info`; that is why the fallback exists and must not be removed.
- `DeviceView.is_contiguous` compares strides with a plain row-major pattern, so it ignores size-1 axes: a `(1, n)` dofs view with strides `(1, 1)` reports False and `contiguous()` copies it, where NumPy calls that layout contiguous.
- `_materialize_view` copies a 3-D transposed view one row per `hipMemcpy` (the `(n_envs, n_links, 3)` shape of `get_links_pos()`), which costs 38.3 ms at n_envs=4096; a consumer needing contiguity should use the preallocated `copy_from_dlpack` destination instead.
- The velocity getters' views carry `element_ndim == 0` while the pose ones carry 1, because the kernel-filled buffer is a plain ndarray: `view_of` on such a view treats the component axis as a grid axis.
- A torch-free scene with `n_envs=0` cannot be built: `RigidSolver.get_state` calls `update_forward_pos`, which passes `scene._envs_idx` (a NumPy array, not a Quadrants buffer) into `Scene._sanitize_envs_idx`, and that refuses any selection on a non-parallelized scene; the batched layouts above are the whole documented contract.
- `device_buffer._kDLBOOL` is `6`, the DLPack `kDLBool` code Quadrants emits for `u1` buffers, so a `u1` field round-trips through `dlpack_info`/`DeviceView`/`np.from_dlpack`; Genesis still allocates solver state as `i8`.
- Derived link velocities are the one intentional cross-path rounding difference: the torch `cross` path and `kernel_get_links_vel` agree within ~1.1e-8, while every raw state buffer and the other accessors are bit-exact across the two paths.
- `utils/serialization.py` and `utils/sdf.py` take part in pre-existing, torch-independent import cycles with `../options/textures.py` and `../engine/solvers/rigid/collider/`, so importing them before `genesis.options`/`genesis.engine.solvers` fails; a normal `import genesis` order avoids it.

# Dependencies (beyond the package manager)

- `quadrants` (core kernel language, vendored runtime expectations), `numba`, `torch`, `numpy`, `pydantic`.
- Geometry/parsing: `trimesh`, `igl` (libigl), `coacd`, `tetgen`, `pygel3d`, `OpenEXR`/`Imath`, `DracoPy`, `pygltflib`, `skimage`, `xacro`, `mujoco`, `urdfpy` (in `genesis/ext`), `fast_simplification`, `IsaacGym` terrain utils (in `genesis/ext`).
- Media: `av` (PyAV), `moviepy`, `PIL`.
- System/display: `cpuinfo`, `psutil`, `pyglet`, `networkx`, `matplotlib`.

# Test Strategy

- Tests live outside this node under `../../tests/`; parser parity/behaviour is covered in `tests/parsers/` (URDF/MJCF/glTF/USD) with format-availability skips (e.g. `pxr` for USD, `SKIP_NO_OMNIVERSE_KIT` for Omniverse material baking).
- Geometry/math helpers are indirectly covered by solver and scene tests; there is no dedicated unit suite for `geom.py`/`misc.py` in this node.
- On-disk cache correctness depends on the version constants in `cache.py`; bump them whenever an algorithm's output changes.
- There is no automated torch-free test in the repository: the torch-free path is checked by running an interpreter where `import torch` raises (a `sitecustomize.py` on `PYTHONPATH` that blocks `torch`), importing every module of this node and exercising `get_device`, `qd_to_numpy`/`copy_from_dlpack`/`make_tensor_field` on both the CPU and the ROCm backend.

# genesis/utils — Intent

Shared utility layer of the Genesis physics engine (`genesis-world`).
It provides the canonical tensor/array conversion helpers, geometry math in three backend variants, declarative solver data structs, asset parsers (URDF/MJCF/glTF/mesh/SDF/terrain/USD), on-disk + in-memory caches, and assorted tooling (media, path planning, particle sampling, serialization).

`genesis/utils/__init__.py` is `from .misc import *`, so the `misc` helpers are the public face of `gs.utils` and are re-exported package-wide.

# Routing Table

- `usd/` → USD stage parser (Pixar `pxr` binding): `parse_usd_stage` / `parse_usd_rigid_entity`, `UsdContext`, material baking, collision filtering. Own CONTEXT.md.
- `../engine/` → consumer of most kernels/structs here (solvers, BVH, FK) — sibling, read-only.
- `../options/` → `morphs` (Mesh/Terrain/USD, etc.) and `Options` consumed by parsers — sibling, read-only.
- `../typing.py`, `../constants.py`, `../repr_base.py` → shared types/constants/repr base imported throughout — siblings, read-only.
- `../ext/` → vendored `urdfpy`, IsaacGym terrain utils, etc. — sibling, read-only.
- `../../tests/` → test suites (parsers under `tests/parsers/`) — sibling, read-only.

# API Surface (flat modules)

Canonical helpers & misc:
- `misc.py` — the canonical helpers every module must use: `tensor_to_array`, `tensor_to_cpu`, `data_to_array` (torch→numpy without `detach().cpu().numpy()`), `qd_to_torch`/`qd_to_numpy`/`qd_from_torch`/`qd_destroy_data` (Quadrants↔torch/numpy zero-copy), `raise_exception`/`raise_exception_from`, `redirect_libc_stderr`, `ZeroCopyMode`, `set_environ`, `assert_initialized`/`assert_built`/`assert_unbuilt`/`with_lock`, `set_random_seed`, `get_device`, `get_assets_dir`/`get_src_dir`/`get_gen_log_dir`/`get_entry_point_name`, `unique_rows` (numba), `make_tensor_field`/`concat_with_tensor`, `indices_to_mask`, `geometric_mean`/`harmonic_mean`, screen helpers (`get_default_screen`, `has_display`), Quadrants monkey-patches (`_stringify_obj_type_with_grad`, `_kernel_impl_with_adjoint_callable`, `_try_load_fastcache_by_autodiff_mode`, `_kernel_reset_with_template_args`, `_release_dispatcher_args`, `_perf_dispatcher_init_released_on_reset`), and the indexed-tensor helpers (`sanitize_index`, `get_last_occurrence`, `sanitize_indices`, `broadcast_tensor`, `sanitize_indexed_tensor`, `get_indexed_shape`, `assign_indexed_tensor`).
- `warnings.py` — `warn_once` (deduplicated logger warning).
- `emoji.py` — `random_scene`, `get_clock` (CLI/progress decoration).
- `deprecated_module_wrapper.py` — `_DeprecatedModuleWrapper` / `create_virtual_deprecated_module` to alias renamed modules with a warning.
- `repr.py` — `brief` / `__repr_name__` used by the `RBC` repr system for compact nested reprs.
- `tools.py` — `animate` (moviepy video), `save_img_arr`, and timing utilities `Timer`/`create_timer`/`Rate`/`RunningMean`/`StepTimer`.

Math & geometry:
- `geom.py` — the canonical geometry module, with three parallel families: `qd_*` (`@qd.func`, device kernels incl. gradients), numpy `_np_*` / torch `_tc_*` backends, and dispatchers (`xyz_to_quat`, `quat_to_R`, `transform_by_quat`, `slerp`, `polar`, etc.). Also `SpatialHasher`, `qd_hash01` (deterministic counter hash), `cubic_spline_1d`, `orthogonals`, `default_solver_params`, `default_friction*`, `default_dofs_kp/kv`, `GeomConstants`/`_tc_geom_constants` (precomputed torch constant tensors per device/dtype).
- `linalg.py` — `@qd.func` small-matrix kernels: `mat_mul`, `mat_mul_vec`, `mat_inverse` (LU), `mat_add`, `mat_transpose`, `mat_add_eye`, `mat_mask`.
- `simt.py` — `@qd.func` SIMT block/segment primitives (`qd_block_sum/scan/min/max/shuffle/ballot/all/any`, lane-slot ops, `qd_segmented_sum/min`, `qd_segment_add/min`).

Declarative solver data:
- `data.py` — the declarative struct layer: `Data`, `StaticConfig`, `DataKind`, `field`, `requires`, `opt_in`/`is_opted_in`, `Size`/`SizeExpr`/`_SizeArithmetic`, `FieldRef`, `Namespace`/`namespace`, `V` (qd.tensor allocator honoring `gs.use_ndarray`), `eoa` (env-major axis order), `ENVS`/`WORKERS`, `iter_data`/`iter_state_arrays`/`check_data`/`fill_data`, `ArrayItem`/`ScalarItem`/`ConfigItem`.
- `array_class.py` — the concrete solver struct dataclasses built on `data.py`: `RigidInfo`, `RigidData`, `Island*`, `IK*`, `ConstraintState`, `ContactData`/`DiffContactInput`/`SortBuffer`/`ContactCache`, `NarrowphaseWorkQueue`, `ColliderState`/`ColliderInfo`/`ColliderStaticConfig`, MPR/GJK/EPA states+scratch, `SDFInfo`/`SDFGeomInfo`, `DofsInfo`/`LinksInfo`/`JointsInfo`/`GeomsInfo`/`VertsInfo`/`FacesInfo`/`EdgesInfo` (+`State` counterparts, visual `V*` variants, `Entities*`/`EqualitiesInfo`), `DynInfo`/`DynState`, `RigidSimStaticConfig`, `RaycastResult`, `BVH*` structs/enums, `SAPContactQueriesState`, `ErrorCode` (IntEnum), and tile-size helpers `cholesky_tile_size_for`/`cholesky_packed_tile_shape`/`island_tile_caps`.
- `ring_buffer.py` — `TensorRingBuffer` (preallocated circular buffer of tensors, view-returning `at`, `set`/`rotate`/`get`/`__getitem__`, host-resident index for perf).

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
- `path_planning.py` — `PathPlanner` (ABC), `RRT`, `RRTConnect`, `align_waypoints_length`, `rrt_valid_mask`, `rrt_connect_valid_mask` (Quadrants-based planners; FREE/SPHERICAL joints unsupported).
- `uid.py` — `UID` (`full`/`short`/`match`, uuid4-hex based).

# Constraints

- Never use `tensor.detach().cpu().numpy()` in Genesis code — use `misc.tensor_to_array`/`tensor_to_cpu`/`data_to_array`.
- Torch/Quadrants conversions go through `misc.qd_to_torch`/`qd_to_numpy` (zero-copy) and `qd_from_torch`; `qd_destroy_data` pairs with them.
- Flat modules here are one level: only `usd/` is a child subdirectory; do not add subpackages to this node without routing them here.
- `array_class.py` structs are consumed by `genesis/engine` kernels; their field layout/size ordering is load-bearing (do not reorder without a solver/kernel audit).
- The Quadrants monkey-patches in `misc.py` are load-bearing for autodiff/fast-cache/dispatcher behaviour — do not remove without understanding the `quadrants` internals they patch.

# Design Decisions

- One geometry API, three backends: `geom.py` keeps numpy (`_np_*`), torch (`_tc_*`), and Quadrants (`qd_*`) implementations in sync behind a single public name, so host-side parsing and device-side simulation share the same conventions.
- Solvers are data-driven: `data.py` + `array_class.py` declare structs/fields once; kernels take those structs and are cached/compiled on equality of `StaticConfig`.
- Caches are two-tier: content-addressed on-disk caches (hash-keyed, version-bumped on algorithm change) plus owner/byte-bounded in-memory caches released on teardown via `register_cache_clear`.
- Parsers converge on a common rigid-entity output tuple rather than a shared IR, keeping each format's quirks local.
- `misc.py` re-exported via `__init__.py` makes conversion helpers available everywhere as `gs.utils.*`.

# Known Issues

- `raycast_qd.py` carries duplicated collision vs visual BVH traversal kernels (`bvh_ray_cast` vs `bvh_ray_cast_visual`, etc.); a FIXME documents that Quadrants cannot accept either of two dataclass shapes for one `qd.func` arg, forcing parallel copies.
- `data.py` has FIXMEs referencing upstream Quadrants limitations (kernel reads `_qd_layout` privately; struct-for over a field ignores `serialize` on GPU; `qd.ndarray` takes no memory layout).
- `mjcf.py` importing `mujoco` with `MUJOCO_GL=EGL` forcibly defines `PYOPENGL_PLATFORM=egl` (documented side effect).
- Heavy optional dependencies are imported lazily inside functions (e.g. `pygel3d` in `hybrid.py`, `moviepy` in `tools.animate`), so module import stays cheap but the deps are still required at call time.

# Dependencies (beyond the package manager)

- `quadrants` (core kernel language, vendored runtime expectations), `numba`, `torch`, `numpy`, `pydantic`.
- Geometry/parsing: `trimesh`, `igl` (libigl), `coacd`, `tetgen`, `pygel3d`, `OpenEXR`/`Imath`, `DracoPy`, `pygltflib`, `skimage`, `xacro`, `mujoco`, `urdfpy` (in `genesis/ext`), `fast_simplification`, `IsaacGym` terrain utils (in `genesis/ext`).
- Media: `av` (PyAV), `moviepy`, `PIL`.
- System/display: `cpuinfo`, `psutil`, `pyglet`, `networkx`, `matplotlib`.

# Test Strategy

- Tests live outside this node under `../../tests/`; parser parity/behaviour is covered in `tests/parsers/` (URDF/MJCF/glTF/USD) with format-availability skips (e.g. `pxr` for USD, `SKIP_NO_OMNIVERSE_KIT` for Omniverse material baking).
- Geometry/math helpers are indirectly covered by solver and scene tests; there is no dedicated unit suite for `geom.py`/`misc.py` in this node.
- On-disk cache correctness depends on the version constants in `cache.py`; bump them whenever an algorithm's output changes.

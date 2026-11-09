# genesis/utils/usd — USD Stage Parser

## Intent

This is the Universal Scene Description (USD) loader for Genesis.
It converts Pixar USD stages (`.usd`, `.usda`, `.usdc`, and packaged `.usdz`) into Genesis rigid entities — articulations and standalone rigid bodies — along with their geometry, materials, and physics properties.

It is the USD counterpart of the flat `genesis/utils` parsers (`urdf.py`, `mjcf.py`, `gltf.py`) and produces the same `(l_infos, links_j_infos, links_g_infos, eqs_info)` data shape those parsers emit, so the downstream rigid-entity builder is format-agnostic.
USD adds two entry points the other formats do not: `Scene.add_stage` (split one stage into many entities) and `Scene.add_entity(gs.morphs.USD)` (parse one entity).

The mathematical contract for transform/scale handling is documented in `UsdParserSpec.md` in this directory (read it before changing `extract_scale` or the joint-anchor math).

## API Surface

- `__init__.py` — guards the `pxr` import (`gs.raise_exception_from` with `pip install genesis-world[usd]` hint if missing); re-exports `parse_usd_stage`, `parse_usd_rigid_entity`, `UsdContext`, `HAS_OMNIVERSE_KIT_SUPPORT`.
- `usd_context.py` — `UsdContext` (the central context: stage open, transforms, material binding, entity discovery, USDZ decompression, bake orchestration); `usd_context(stage_file)` contextmanager + `get_current_usd_context()` (a `contextvars.ContextVar`, so one stage serves every entity parse inside a block); `PhysicsMaterial` NamedTuple; module-level `find_joints_in_range`, `find_rigid_bodies_in_range`, `resolve_rigid_body_link_path`, `extract_links_referenced_by_joints`, `decompress_usdz`.
- `usd_stage.py` — `parse_usd_stage(morph)` contextmanager; helpers `_find_common_ancestor_path`, `_find_connected_components`.
- `usd_rigid_entity.py` — `parse_usd_rigid_entity(morph, surface)`; helpers `_LinkJoint`, `_detect_generic_joint_type`, `_parse_joint_axis_pos`, `_parse_link`, `_parse_articulation_structure`, `_parse_geoms`, `_parse_links`, `_compute_joint_prim_paths`; `DRIVE_NAMES` maps Genesis joint types to USD DriveAPI drive components.
- `usd_geometry.py` — `parse_prim_geoms` (recursive Gprim walk); `get_triangle_ids`, `geom_exception`; `_APPROXIMATION_OVERRIDES` maps `UsdPhysics.MeshCollisionAPI` approximation tokens to per-geom post-processing.
- `usd_material.py` — `parse_material_preview_surface`, `parse_preview_surface`, `parse_component`, `get_shader`, `get_input_attribute_value`, `CS_ENCODE`.
- `usd_utils.py` — axis/transform helpers `AXES_VECTOR`, `AXES_T`, `extract_scale`, `get_attr_value_by_candidates`, `UnionFind`; USD-attribute converters `usd_pos_to_numpy`, `usd_quat_to_numpy`, `usd_center_of_mass_to_numpy`, `usd_principal_axes_to_numpy`, `usd_inertia_to_numpy`, `usd_diagonal_inertia_to_numpy`, `usd_mass_to_float`, `usd_density_to_float`, `usd_attr_array_to_numpy`, `usd_primvar_array_to_numpy`.
- `usd_bake.py` — `omni_bootstrap`, `bake_usd_material`; `__main__` CLI is the subprocess target for Omniverse-Kit material baking.
- `usd_collision.py` — `apply_collision_filtering`, `_geoms_under`.

Note: `decompress_usdz` lives in `usd_context.py`, not `usd_utils.py`.

## Data Flow

1. `Scene.add_stage(morphs.USD)` calls `parse_usd_stage(morph)`.
   It opens one `usd_context` (one `UsdContext`), finds all joints and rigid bodies, and builds connected components over the joint graph with `UnionFind` (joints = edges, rigid-body links = nodes).
   Each articulation component and each pure (unreferenced) rigid body becomes one `gs.morphs.USD` copy whose `prim_path` is the component's topmost common ancestor (or the rigid body path).
   The morph list is yielded inside the context block; `add_stage` calls `add_entity` per morph while the stage stays open.
2. `Scene.add_entity(morph)` → rigid-entity description calls `parse_usd_rigid_entity(morph, surface)`.
   It reads `get_current_usd_context()` (reusing the open stage) or opens a fresh `UsdContext`; calls `find_all_materials()`; resolves `entity_prim` from `morph.prim_path` (or the default prim); computes joint paths via `_compute_joint_prim_paths`; builds links/joints via `_parse_articulation_structure`; walks geometry via `parse_prim_geoms`; parses each link via `_parse_link`; then `apply_collision_filtering`, density-override, and `urdf_utils.order_links_depth_first`.
3. Materials are parsed lazily in `UsdContext.find_all_materials()`: bound `UsdShade.Material` preview surfaces → Genesis textures; unsupported (complex) shaders are baked to a flat USD via the `usd_bake.py` subprocess, then re-read; Gprims without a bound material fall back to `displayColor`.
4. Geometry: each Gprim's USD transform is split into rigid `geom_Q` + scale `geom_S` (reflection included); `geom_S` (times `morph.scale`) is baked into the trimesh (`geom_ST`), `geom_Q` becomes the Genesis pos/quat; primitives (plane/sphere/capsule/cube/cylinder) produce typed `data` (e.g. radius/height) plus a trimesh visual.

## Design Decisions

- Official `pxr` (usd-core) binding instead of a custom parser: full layer/reference/payload/instancing semantics via `Usd.PrimRange`, `Usd.TraverseInstanceProxies`, `UsdShade.MaterialBindingAPI`, and the asset resolver (needed for `.usdz` package paths like `mesh.usdz[texture.png]`).
- Genesis links are rigid-only, so USD scale (including negative/reflection scale) is separated via polar decomposition and baked into mesh vertices; see `UsdParserSpec.md` and `extract_scale`.
- Coordinate normalization: `UsdContext.compute_transform` multiplies Y-up stages by `mu.Y_UP_TRANSFORM` and applies `metersPerUnit`, so everything downstream is Z-up meters.
- Omniverse-Kit baking runs in a subprocess (`-X faulthandler`) to isolate Kit's global conversion registry (in-process import segfaults), serialized across processes with a `filelock`.
- `UsdContext` caches `XformCache`, material dicts, and physics materials; `get_prim_id` gives layer-stable IDs for materials across baked/unbaked stages.
- USD joint graph → connected components mirrors MJCF's grouping so `add_stage` returns one entity per independent mechanism.

## Constraints

- `pxr` (usd-core) is a hard runtime dependency; importing the package without it raises.
- Omniverse Kit (`omni.kit_app`) + CUDA is optional and only required for baking complex materials; CI must set `OMNI_KIT_ACCEPT_EULA=yes`.
- `morph.file_meshes_are_zup` is rejected for USD morphs (up axis comes from stage metadata).
- USD does not express equality constraints; `parse_usd_rigid_entity` always returns `eqs_info = []`.

## Known Issues / Gotchas

- USD physics-material `restitution` is parsed but dropped (no rigid-rigid restitution in the solver); a one-time warning is emitted via `note_unsupported_restitution`.
- Collision filtering that spans entities (stage split by `add_stage`) cannot be applied; detected and reported once via `note_unsupported_cross_entity_filtering`.
- Spherical joint cone/pyramid limits are parsed but not enforced by the Genesis solver.
- A `morph.prim_path` subtree containing both joints and unreferenced rigid bodies raises ("mixed entity") — use `add_stage`.
- Non-uniform scale on a primitive collision uses the first axis and warns.
- `DriveAPI` target position/velocity are not implemented (only stiffness/damping/maxForce → `dofs_act_gain`/`dofs_act_bias`/`dofs_force_range`).
- Joint `init_qpos` is zeroed (IsaacSim `state:...:physics:position` attribute is a TODO).
- `UsdContext.__init__` parameter is `use_bake_cache`, but the class docstring still calls it `usd_bake_cache` (doc drift).
- Baking subprocess needs `OMNI_KIT_ALLOW_ROOT=1`, runs under a 600 s file lock, and its first launch may require EULA acceptance/extra installs.

## Test Strategy

- `tests/parsers/test_usd.py` is the main suite: parity tests that build the same scene from MJCF/URDF and from USD and compare them (`build_usd_scene`/`compare_scene` fixtures in `tests/parsers/conftest.py`).
- Module-level `pytest.skip` when `pxr` is unavailable; `SKIP_NO_OMNIVERSE_KIT` marks bake-dependent tests.
- Primitive, joint, and mesh parity cases; scale is a parametrized dimension (`1.0`, `2.0`).

## See Also (siblings — read-only, escalate writes to parent)

- `./UsdParserSpec.md` — parser transform/scale/joint-anchor specification (same directory).
- `genesis/constants.py` — `USD_FORMATS = (".usd", ".usda", ".usdc", ".usdz")`.
- `genesis/options/morphs.py` — `gs.morphs.USD` (parser options: joint attr candidates, visual/collision prim regex patterns, `prim_path`, `fixed`).
- `genesis/utils/cache.py` — `get_usd_cache_dir`, `get_usd_bake_path`, `get_usd_zip_path` (cache layout under `<cache>/usd/{zip,bake}/<hash>`).
- `genesis/utils/geom.py` — `polar`, `transform_by_*`, `R_to_quat`, `zero_pos`, `identity_quat`, `default_solver_params`, `default_friction`.
- `genesis/utils/mesh.py` — `Y_UP_TRANSFORM`, `create_texture`, `PIL_to_array`.
- `genesis/utils/urdf.py` — `order_links_depth_first` (shared link reordering).
- `genesis/engine/scene.py` — `Scene.add_stage` (consumer of `parse_usd_stage`).
- `genesis/engine/entities/rigid_entity/description.py` — consumer of `parse_usd_rigid_entity`.

## Routing Table

No child directories — this is a leaf module.
- `usd_stage.py` → stage-to-entity splitting (`add_stage` path).
- `usd_rigid_entity.py` → single-entity parse (links/joints).
- `usd_geometry.py` → Gprim → Genesis mesh/primitive geometry.
- `usd_material.py` → preview-surface shader → Genesis texture.
- `usd_context.py` → shared stage/material/transform context + bake orchestration.
- `usd_utils.py` → attribute conversion, scale decomposition, UnionFind.
- `usd_collision.py` → USD collision filtering → link contact exclusions.
- `usd_bake.py` → Omniverse-Kit material baking subprocess.

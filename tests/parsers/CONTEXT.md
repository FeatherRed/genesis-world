# Intent
Tests for Genesis asset parsers: mesh formats (glTF/GLB, OBJ, STL, MJCF mesh/texture, URDF mesh) and USD (via pxr/usd-core, with an optional Omniverse-Kit baking path).
They verify that `gs.morphs.Mesh`/`URDF`/`MJCF`/`USD` and `genesis.utils.gltf`/`genesis.utils.usd` parse authored assets correctly, either against a `trimesh` reference load or against an equivalent scene parsed from another format (MJCF vs USD).

# API Surface
- `conftest.py` — local fixtures, comparison helpers, and in-session asset generators (~1800 lines, the bulk of the directory).
- `test_mesh.py` — mesh/material/texture parsing tests (scale, Y-up, glTF, MJCF, surface reconstruction, convex-decompose cache).
- `test_usd.py` — USD parsing tests (primitives, joints, visuals, mass/inertia, collision APIs, capsule axis).
- `__init__.py` — empty.

# conftest.py
Module tolerances (USD round-trips through .glb/.usda lose precision, so they are not held to the session `tol`): `USD_COLOR_TOL=1e-7`, `USD_NORMALS_TOL=1e-2`, `USD_PHYSICS_TOL=5e-7` (UsdPhysics authors joint attributes in single precision, so anchors/limits/drive gains are always rounded by that amount).
Comparison helpers: `extract_mesh` (canonical vertex ordering via `np.lexsort` so unordered meshes compare), `check_gs_meshes`, `check_gs_tm_meshes` (gs.Mesh vs trimesh.Trimesh), `check_gs_tm_textures`, `check_gs_textures`, `check_gs_surfaces`, `compare_links`/`compare_joints`/`compare_geoms`/`compare_vgeoms`/`compare_scene`/`compare_mesh_scene`.
Scene builders keep collision detection disabled (`enable_collision=False`) so parser comparison stays fast: `build_mjcf_scene`, `build_mesh_scene`, `build_usd_scene`.
`build_usd_scene` monkeypatches `UsdContext.__init__` with `use_bake_cache=False` so a stage is read as authored rather than through baked assets a prior run left on disk, and uses `add_stage` for stages vs `add_entity` for plain assets.
Fixtures: `mesh_path` (resolves `mesh_file` from `get_assets_dir()` or falls back to `get_hf_dataset`), `mesh_urdf` (wraps the mesh as the visual of a one-link URDF), `xml_path` (writes a MJCF element to a temp `.xml`), `scale`/`fixed` defaults, `usd_scene` (builds from a fixture named `model_name`).
Session-scoped generators build files in `asset_tmp_path` via `pygltflib`/`trimesh`/`pxr`: glTF accessor-order fixtures (`normal_accessor_zero_glb`, `texcoord_0_accessor_zero_glb`, `texcoord_1_accessor_zero_glb`, `triangle_strip_nodes_glb`, `non_uniform_node_scale_glb`, `material_variants_glb`); MJCF models (`material_mjcf`, `normals_mjcf`, `textured_mjcf`, `all_primitives_mjcf`, `all_joints_mjcf`); USD equivalents (`all_primitives_usd`, `all_joints_usd` — `indirect` param toggles `ArticulationRootAPI`, `pure_rigid_usd`, `collision_only_rigid_usd`, `visual_collision_usd`, `negative_scale_rigid_usd`, `nested_collision_joint_usd`, `physics_material_usd`, `density_align_usd`, `collision_approximation_usd`, `collision_filtering_usd`, `cross_entity_filtering_usd`, `oriented_capsule_usd`, `usdz_packaged_texture_usd`).
All heavy imports (`numpy`, `trimesh`, `pxr`, `genesis`) are wrapped in `lazy_imports()` from `tests/conftest.py`, so they resolve on first use.

# test_mesh.py
Scale: `test_morph_scale` (non-isotropic URDF `scale` raises `gs.GenesisException`; isotropic scale matches a manually transformed reference), `test_urdf_scale`.
Y-up: `test_mesh_yup`, `test_urdf_yup` — exercise `file_meshes_are_zup` and `vmesh.metadata["imported_as_zup"]` across glTF/STL/OBJ y/z-up cannon assets from HuggingFace.
Geometry: `test_urdf_mesh_processing` (visual vs collision vertex counts; OBJ color from material library, STL default white), `test_glb_parse_geometry` (gs vs trimesh, precision=32, node transforms with authored normals mapped through the inverse transpose), `test_glb_draco_missing_normals_texcoord` (Draco-compressed GLBs may lack normals/texcoords), `test_glb_material_variants` (alpha modes MASK/BLEND/OPAQUE, KHR_materials_unlit, per-texCoord UV sets, normalized UNSIGNED_SHORT UV decoding).
Material/texture: `test_glb_parse_material` (chopper.glb vs trimesh PBRMaterial), `test_glb_shared_texture_not_duplicated` (16 submeshes share one memoized RGBA array; `GenesisGeomRetriever` sees one texture), `test_glb_uv_set_and_unlit_albedo_resolution`, `test_glb_multi_primitive_distinct_materials` (per-primitive vgeoms but one merged collision geom per node), `test_float_texture_conversion` (in-memory `MeshSet` float image → uint8), `test_2_channels_luminance_alpha_textures`, `test_plane_texture_path_preservation`, `test_mjcf_parse_material`, `test_mjcf_parse_mesh_normals` (normal block at a different offset than the vertex block), `test_mjcf_2d_texture_mapping` (object-relative/fitted/spatial UV conventions, `texrepeat`/`texuniform`, explicit `texcoord`, shared RGBA with read-only `image_array`).
Surface reconstruction: `test_surface_reconstruction` (SPH/PBD liquids, splashsurf/levelset, segmentation pixel-count comparison) — solver behavior, not a parser test, but lives here.
Mesh processing/caching: `test_convex_decompose_cache` (one coacd decomposition serves scaled/re-exported copies; `@pytest.mark.cache(False)`).

# test_usd.py
Module-level `pytest.skip` when `find_spec("pxr") is None`; `test_bake` is additionally skipped unless `HAS_OMNIVERSE_KIT_SUPPORT`.
Primitives/joints: `test_primitives_mjcf_vs_usd`, `test_joints_mjcf_vs_usd` (both compare a built MJCF scene against a built USD scene via `compare_scene`).
Visuals: `test_visual_parse` (glb vs usdz with `vis_mode="visual"`, `is_stage=False`, tol 5e-6), `test_parse_nodegraph`, `test_usdz_packaged_texture_resolution` (texture resolved from a `.usdz` package-internal path), `test_bake` (cuda-only; asserts `vmesh.metadata["bake_success"]`).
Mass/inertia: `test_massapi_invalid_defaults_mjcf_vs_usd` (UsdPhysics MassAPI sentinel defaults treated as unset), `test_physics_material_friction_and_density`, `test_align_anchor_with_geom_densities`, `test_align_requires_all_or_none_geom_densities`.
Robustness/body structure: `test_uv_size_mismatch_no_crash` (mismatched UV counts warn and drop UVs), `test_pure_rigid_body_fixed`, `test_collision_only_fixed_override` (no RigidBodyAPI → `fixed` controls `n_dofs`), `test_visual_collision_parsing` (purpose=guide colliders vs visuals vs invisible sites; export round-trip excludes the asset directory from the manifest), `test_ur10_visual_fallback`, `test_humanoid_generic_joint_detection`, `test_negative_scale_reflection`, `test_nested_collision_joint_targets`.
Collision: `test_collision_approximations` (MeshCollisionAPI boundingCube/boundingSphere/convexHull/sdf/none vs `decimate`), `test_collision_filtering` (FilteredPairsAPI + CollisionGroups, partial-filter warning), `test_cross_entity_filtering_not_applied`, `test_oriented_capsule` and `test_ant_capsule_axis_collision` (capsule axis orientation, settle without penetration).

# Constraints
Do not modify source; this directory only contains tests and their CONTEXT.md.
`pxr` (usd-core) is required for `test_usd`; Omniverse Kit is optional and only needed by `test_bake`.
`test_bake` is parametrized to run on `gs.cuda` only.
Parser comparisons go through the `build_*_scene` builders with collision detection off; a few collision/filtering tests build their own scenes with collision enabled.
Tolerances are defined once as module constants (`USD_*_TOL`) rather than derived from the session `tol`.

# Design Decisions
MJCF↔USD equivalence: author equivalent MJCF and USD files and compare the two built scenes, instead of hand-computing expected values.
Mesh/glTF reference is `trimesh` (`process=False`), with `extract_mesh` sorting vertices into a canonical order before comparison.
`mesh_file`/`model_name` are parametrized test arguments reused as fixtures through `request.getfixturevalue`, so each test only states an asset path or a generator-fixture name.
Session-scoped generator fixtures author once and share files across parametrizations, keeping `pxr`/`pygltflib` authoring out of the hot test loop.
USD stages are always read with the bake cache disabled to make parse results deterministic across runs.

# Known Issues
`@pytest.mark.cache(False)` in `test_mesh.py` is not a registered marker (unknown-mark warnings are suppressed in `pyproject.toml`).
`test_convex_decompose_cache` has its `# @pytest.mark.required` commented out and a FIXME: it can take ~1200s on some platforms.
`test_massapi_invalid_defaults_mjcf_vs_usd` carries a FIXME: the tolerance is inexplicably lax (`1e-5`).
`test_mesh_yup` has a commented-out bounding-box equality assertion (FIXME: the STL assets differ from the glTF).
`test_oriented_capsule` deliberately asserts a quat is NOT identity via `pytest.raises(AssertionError)`.
USD→GLB conversion lowers precision, which is why USD_* tolerances exist separately from `tol`.

# Test Strategy
Run with `pytest tests/parsers` (root addopts: worksteal parallel, 1200s timeout, benchmarks/examples deselected).
Most tests are `@pytest.mark.required`; slow ones: `test_mesh_yup` (~200s), `test_bake` (~400s), `test_oriented_capsule` (~250s), `test_ant_capsule_axis_collision` (~200s).
Assets come from HuggingFace `Genesis-Intelligence/assets` via `tests/utils/assets.py::get_hf_dataset`, or are generated in-session into `asset_tmp_path`.
Optional-dependency skips: `test_usd` module skips without `pxr`; `test_bake` skips without Omniverse Kit.

# See Also
- `../../genesis/utils/gltf.py`, `../../genesis/utils/usd.py`, `../../genesis/utils/mesh.py` — parsers under test.
- `../conftest.py` — shared fixtures (`backend`/`tol`/`show_viewer`/`precision`/`asset_tmp_path`), markers, `lazy_imports`.
- `../utils/assets.py`, `../utils/assertions.py` — asset fetching and assertion helpers (read-only).

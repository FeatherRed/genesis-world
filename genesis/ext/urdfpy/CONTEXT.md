# genesis/ext/urdfpy

## Intent
Vendored fork of `urdfpy` (upstream mmatl/urdfpy, version `0.0.22`) — a pure-Python URDF object model + parser with forward kinematics.
Embedded to load URDF/XACRO/MJCF assets and to synthesize URDFs from networkx graphs in the hybrid-entity pipeline.
This is third-party code kept in upstream style, not Genesis-native style.

## API Surface
Flat module — four files, no child directories.

- `__init__.py` — re-exports the full object model plus matrix helpers and `__version__` (see `__all__`).
- `urdf.py` — the object model, XML parsing, and forward kinematics.
- `utils.py` — matrix/pose helpers and mesh/asset loading.
- `version.py` — `__version__ = "0.0.22"`.

Object model (all subclass `URDFType`):
- `URDFType` — abstract base; declarative `_ATTRIBS` / `_ELEMENTS` / `_TAG` class attributes drive ElementTree (un)parsing.
- `Box`, `Cylinder`, `Capsule`, `Sphere` — primitive geometries; each has `.meshes` (lazily built trimesh) and `copy()`.
- `Mesh` — `.filename`, `.scale`, `.meshes` (list of `trimesh.Trimesh`).
- `Geometry` — wrapper holding exactly one of box/cylinder/capsule/sphere/mesh; `.geometry` returns the set one, `.meshes` delegates.
- `Texture` — `.filename`, `.image` (PIL image).
- `Material` — `.name`, `.color` (RGBA), `.texture`.
- `Collision` — `.geometry`, `.name`, `.origin`.
- `Visual` — `.geometry`, `.name`, `.origin`, `.material`.
- `Inertial` — `.mass`, `.inertia` (3x3), `.origin`.
- `JointCalibration` (rising/falling), `JointDynamics` (damping/friction), `JointLimit` (effort/velocity/lower/upper), `JointMimic` (joint/multiplier/offset), `SafetyController` (k_position/k_velocity/soft limits).
- `Actuator`, `TransmissionJoint`, `Transmission` — ROS transmission elements (actuator/hardware-interface metadata).
- `Joint` — `.name`, `.joint_type`, `.parent`, `.child`, `.axis`, `.origin`, `.limit`, `.dynamics`, `.safety_controller`, `.calibration`, `.mimic`; `is_valid()`, `get_child_pose(cfg)`, `get_child_poses(cfgs)`.
- `Link` — `.name`, `.inertial`, `.visuals`, `.collisions`, `collision_mesh()`.
- `URDF` — root (`<robot>`); builds a networkx link graph, validates it, and caches FK paths.

`URDF` key methods (public surface used by Genesis):
- Parsing/build: `load(file_obj)`, `_from_xml(node, root, path)`, constructor `URDF(name, links, joints=..., transmissions=..., materials=...)`, `from_nxgraph(G)`.
- Properties: `links`, `link_map`, `joints`, `joint_map`, `transmissions`, `transmission_map`, `materials`, `material_map`, `actuated_joints`, `actuated_joint_names`, `base_link`, `end_links`, `joint_limits`, `joint_limit_cfgs`.
- FK: `link_fk`, `link_fk_batch`, `visual_geometry_fk(_batch)`, `visual_trimesh_fk(_batch)`, `collision_geometry_fk(_batch)`, `collision_trimesh_fk(_batch)`.
- Misc: `cfg_to_vector`, `copy`, `to_xml`, `join`, `animate`, `show`.

`utils.py` public functions: `rpy_to_matrix`, `matrix_to_rpy`, `matrix_to_xyz_rpy`, `xyz_rpy_to_matrix`, `parse_origin`, `unparse_origin`, `configure_origin`, `get_filename`, `load_meshes`.

## Constraints
Vendored fork — keep upstream object model and ElementTree parsing intact; changes should stay minimal and integration-focused.
No schema validation is performed (stdlib `xml.etree.ElementTree`); malformed/conflicting URDFs are detected by constructor-time `ValueError`s.
No writes to source files here — this node only documents the module.

## Dependencies
- External: `numpy`, `trimesh`, `networkx`, `PIL` (Pillow).
- Internal: `genesis.utils.mesh.load_trimesh` (mesh loading), `genesis.ext.pyrender` (lazy import for `show`/`animate` only), `genesis as gs` (`gs.raise_exception` on missing assets).
- `xacro` is used by the consumer `genesis/utils/urdf.py`, not by urdfpy itself.

## Design Decisions (local modifications vs upstream)
- `utils.get_filename` resolves `package://` URIs by stripping the scheme and, when not found under the base path, walking the base path upward to the package-name segment; it calls `gs.raise_exception` on any missing asset.
- `utils.load_meshes` delegates to `genesis.utils.mesh.load_trimesh(..., process=False)`; it flattens `trimesh.Scene` nodes into a list of `Trimesh` using `copy(include_cache=True)` to preserve vertex normals, and always returns a list.
- `urdf.py` threads the XML `root` node through `_parse` / `_parse_simple_elements` / `_from_xml` (stdlib ElementTree lacks parent pointers) so `Mesh._from_xml` can inspect ancestors.
- `Mesh._from_xml` resolves the filename via `get_filename`, loads meshes, and merges all submeshes into one `Trimesh` when the geometry sits under a `<collision>` (stripping visuals); visual-geometry meshes stay separate to preserve colors/textures.
- `Texture._from_xml` resolves the filename via `get_filename` before opening with PIL.
- `show()` and `animate()` do `from genesis.ext import pyrender` inside the method (lazy), so the vendored renderer is used and the external `pyrender` package is not required for CI/headless import.
- Added `URDF.from_nxgraph` plus module-level helpers `norm_vec`, `rotation_matrix_from_vectors`, `xyz_rotmat_to_matrix`, `transform_by_T`, `trans_R_to_T` to synthesize a URDF (box links + continuous joints) from a networkx tree of 3D points.
- Upstream's `if __name__ == "__main__": main()` demo is removed.

## Known Issues / Gotchas
- `show()` / `animate()` require an OpenGL context and the vendored pyrender viewer; they are not import-safe for headless CI without an active context.
- `load_meshes` returns a list of `Trimesh` even for single-mesh files (and raises if a file yields zero meshes).
- `get_filename`'s `package://` resolution handles the common "package name appears in the base path" case only; it raises (not warns) on missing files.
- FK results are poses relative to `base_link`; `cfg` may be a joint→value dict (keys can be `Joint` or name), or a vector ordered as `actuated_joints` (topological order, mimic/fixed excluded).
- `base_link` is exactly one root link; disconnected or cyclic link graphs raise at construction.
- Collision-geometry meshes are concatenated into a single `Trimesh`, so per-submesh collision identities are lost.

## Consumers (read-only cross-references — escalate writes to the owning agent)
- `genesis/utils/urdf.py` — primary consumer: builds Genesis link/joint/geom info from a parsed URDF (uses `URDF._from_xml`/`load`/constructor, `Mesh`/`Box`/`Capsule`/`Cylinder`/`Sphere`/`Visual`/`Inertial`, `utils.get_filename`, and link/joint/geom attributes).
- `genesis/utils/mjcf.py` — `URDF._from_xml`, `robot.to_xml()`, `robot.links`.
- `genesis/engine/entities/hybrid_entity.py` — `from genesis.ext.urdfpy.urdf import URDF`; uses `URDF.from_nxgraph`.
- `genesis/options/morphs.py` — `urdfpy.URDF`, `robot.to_xml()`.

## Test Strategy
No dedicated tests live under this node; parsing correctness is exercised indirectly via the `genesis/utils/urdf.py` and `genesis/utils/mjcf.py` consumers and their test suites.

# genesis/vis — Visualization & Rendering Layer

## Intent
Built-in visualization and rendering for Genesis World.
Provides the interactive viewer plus three render backends (rasterizer, Luisa ray tracer, Madrona batch renderer).
Cameras are the only render entry point; they are created by `Scene.add_camera(...)` and wrapped by camera sensors.

## API Surface
Top-level modules (flat package, no sub-package except `viewer_plugins/`):

- `__init__.py` — exports `Visualizer` only.
- `visualizer.py` — `Visualizer`: the facade/orchestrator owned by `Scene` (as `scene.visualizer`).
  Owns the shared `RasterizerContext`, optional `Viewer`, always-on `Rasterizer`, optional `Raytracer`/`BatchRenderer`, and the `Camera` list.
  API: `add_camera`, `add_mesh_light`, `add_light`, `build`, `reset`, `sync`, `update`, `update_on_sim_step`, `update_visual_states`, `colorize_seg_idxc_arr`, `destroy`; properties `viewer`, `rasterizer`, `raytracer`, `batch_renderer`, `renderer`, `context`, `cameras`, `is_software`, `segmentation_idx_dict`, `has_display`.
- `viewer.py` — `Viewer`: wraps `genesis.ext.pyrender.Viewer` (pyglet window).
  API: `build`, `run`, `stop`, `is_alive`, `update`, `update_on_sim_step`, `sleep`, `set_camera_pose`, `follow_entity`, `render_offscreen`, `close_offscreen`, `register_keybinds`/`remap_keybind`/`remove_keybind`, `add_plugin`/`remove_plugin`, `toggle_recording`; properties `recording`, `plugins`, `res`, `refresh_rate`, `realtime_factor` (settable), `camera_pos`/`camera_lookat`/`camera_pose`/`camera_up`/`camera_fov`.
  `ViewerLock` wraps the pyrender viewer's re-entrant render lock as a context manager.
- `camera.py` — `Camera`: the render entry point and sensor-facing API.
  Models `pinhole`, `thinlens`, `fisheye`; renders RGB/depth/segmentation/normal.
  API: `render`, `render_pointcloud`, `set_pose`, `get_pos`/`get_lookat`/`get_up`/`get_quat`/`get_transform`, `attach`/`detach`/`move_to_attach`, `follow_entity`/`unfollow_entity`/`update_following`, `start_recording`/`pause_recording`/`stop_recording`, `distance_center_to_plane`; properties `idx`, `uid`, `model`, `res`, `fov`, `aperture`, `focal_len`, `focus_dist`, `near`, `far`, `aspect_ratio`, `intrinsics`, `extrinsics`, `projection_matrix`, `f`/`cx`/`cy`, `env_idx`, `debug`, `GUI` (settable).
- `rasterizer.py` — `Rasterizer`: rasterizer render backend.
  In offscreen mode (no viewer) it owns a `pyrender.OffscreenRenderer`; with a viewer it delegates to `Viewer.render_offscreen`.
  API: `add_camera`, `update_camera`, `remove_camera`, `render_camera`, `update_scene`, `destroy`.
  Sets `PYOPENGL_PLATFORM=osmesa` at import time on headless Linux lacking EGL.
- `rasterizer_context.py` — `RasterizerContext`: the pyrender scene graph and node bookkeeping shared by viewer and rasterizer.
  Also `TrackedNodeList`, `RigidNodeBatch`, `SegmentationColorMap`, `POSE_MAX_ABS`.
  API (main): `build`, `update`, `reset`, `destroy`, `add_node`/`remove_node`, `add_rigid_node`/`add_geom_node`/`remove_rigid_node`, `add_static_node`/`add_dynamic_node`/`add_external_node`, `clear_dynamic_nodes`/`clear_external_node(s)`, `rigid_geoms_T`, per-solver `on_*`/`update_*`/`draw_*`, `make_debug_*`/`add_debug_node`/`update_debug_objects`/`clear_debug_object(s)`, `add_light`, `create_node_seg`/`remove_node_seg`, `colorize_seg_idxc_arr`.
- `raytracer.py` — `Raytracer`: LuisaRender backend (`LuisaRenderPy`).
  Also `EnvironmentSphere`, `ShapeLight`/`SphereLight`/`MeshLight`.
  API: `add_mesh_light`, `build`, `add_camera`/`update_camera`, `reset`, `update_scene`, `render_camera`, `destroy`.
  Renders RGB only; depth/segmentation/normal always fall back to the rasterizer.
- `batch_renderer.py` — `BatchRenderer`: Madrona GPU batch renderer for many cameras at once.
  Also `GenesisGeomRetriever` and `Light`.
  API: `add_light`, `build`, `render` (returns one image tuple per camera), `update_scene`, `colorize_seg_idxc_arr`, `reset`, `destroy`.
  Linux x86-64 + CUDA only; all cameras must share one resolution.
- `keybindings.py` — pyglet-compatible key handling: `Key`, `KeyMod`, `KeyAction`, `MouseButton` (labeled `IntEnum`s), `Keybind`, `Keybindings`, `get_key_hash`.

## Architecture / Design Decisions
`Visualizer` is the single facade: `Scene` constructs it with `options.vis`, `options.viewer`, `options.renderer`, and everything under `genesis/vis` is reached through it.
The rasterizer is created unconditionally, even when `renderer` is a `RayTracer`, because depth/segmentation/normal always render through the rasterizer.
One `RasterizerContext` (one pyrender scene graph) is shared by the interactive viewer and the offscreen rasterizer; the viewer draws whatever the context uploads, synchronized via `viewer_lock`.
`Camera.render` dispatches by active backend: `BatchRenderer` → `_batch_render`; `RayTracer` → RGB from raytracer, other channels from rasterizer; otherwise rasterizer for all channels.
Debug cameras (`debug=True`) always render through the plain rasterizer, bypassing raytracer/batch, so they can record arbitrary views without affecting robot perception.
Camera poses are stored in float64 regardless of simulation precision, so rendered views do not depend on sim precision; the public getters return the simulation precision.
`Visualizer.update_visual_states` updates visualization-only quantities (vgeoms, render fields, propellers) and early-returns once `_t >= sim.cur_step_global` unless forced.
`TrackedNodeList` redraws ephemeral nodes (contact-force arrows, sensor debug overlays) at every sync and invalidates them on subscribed `StateChange` events.
Segmentation uses an index-color-code (`idxc`) packed into RGB; `SegmentationColorMap` maps `seg_key` → `idxc` → color.
Node categories in the context: rigid nodes (pose-tracked, batched per solver+vis_mode), static nodes (particle spheres), dynamic nodes (per-step, cleared when outdated), external nodes (user debug objects), tracked nodes.
Interactive viewer platforms are probed in order on Linux (`native`/`egl`/`glx`, or `egl` only when pyglet headless), reloading `OpenGL.platform` per attempt.
`run_in_thread` defaults to `True` except on macOS, where a viewer must run on the main thread.

## Constraints
Do not import `genesis.ext.pyrender` before settling `PYOPENGL_PLATFORM` (see `rasterizer.py`); PyOpenGL binds all GL functions at first pyrender import.
`RasterizerContext` node mutations must hold `scene._visualizer.viewer_lock` when a viewer is active.
`BatchRenderer` cameras must be non-debug and identical in resolution; the backend requires CUDA and Linux x86-64.
`RayTracer` does not support multi-environment rendering.
Only CONTEXT.md files may be edited here; never modify source under this node during context extraction.

## Known Issues / Gotchas
`Visualizer.is_software` reaches into private `_pyrender_viewer._is_software` / `_rasterizer._renderer._is_software` and asserts the relevant renderer exists.
`rasterizer.py` falls back to OSMesa (software) only when EGL is missing on a headless Linux host; a machine with a display keeps its window-system platform.
macOS interactive viewer cannot run in a background thread; forcing it raises an exception.
On `PYOPENGL_PLATFORM=osmesa`, the interactive viewer is unsupported and raises.
The pyrender viewer's `render_lock` is re-entrant; `ViewerLock` relies on that re-entrancy when add/remove-plugin is called from a render callback.
`RayTracer` shape management is stateless w.r.t. surfaces/entities added after `build()`; those may be absent from the Luisa scene.
`Raytracer.render_camera` always returns a BGRA→RGB 8-bit image regardless of backend precision.
`genesis/engine/sensors/camera.py` (camera sensors) depends on this package and reuses `visualizer.add_camera`; keep that call signature stable.

## Dependencies
- `pyglet` (windowing), `PyOpenGL`, `OpenGL`, `trimesh`, `numpy`, `torch`.
- PyTorch is imported through `genesis.utils.torch_compat` (`HAS_TORCH`/`torch`/`require_torch`), so every module here imports without PyTorch installed; rendering stays torch-only and `Camera(...)`, `Camera.render`, `Raycaster(...)` and `BatchRenderer(...)` raise a GenesisException naming the missing feature when it is absent.
- `genesis.ext.pyrender` — vendored pyrender with JIT renderer, ImGui overlay, and pyglet viewer.
- `LuisaRenderPy` (from the `genesis/ext/LuisaRender` git submodule build) — raytracer backend.
- `gs_madrona` — batch renderer backend, optional (Linux x86-64 CUDA only).
- Optional runtime: `cv2` (camera GUI display), `imgui_bundle` (save-image / record-video prompts).

## Test Strategy
No dedicated tests live in this node; behavior is exercised through the example scripts and sensor/render tests elsewhere in the repo.
Manually smoke-test the interactive viewer on each OS (Linux/X11+EGL+headless-OSMesa, Windows/wgl, macOS/metal).
Verify the OSMesa fallback on a headless Linux CI box without EGL.
Camera recordings (`start_recording`) and `split_envs` multi-video output need explicit coverage.

## Routing Table
- `viewer_plugins/` → pyglet event-handling plugins (base class, BVH raycasting, default controls, mouse interaction).
- `../engine/sensors/camera.py` → camera sensor wrapper that consumes this package (sibling — read-only, escalate writes to parent).
- `../engine/scene.py` → constructs and owns the `Visualizer` (sibling — read-only).
- `../options/vis.py` → `VisOptions`/`ViewerOptions` consumed here (sibling — read-only).
- `../options/renderers.py` → `RayTracer`/`BatchRenderer`/`Rasterizer` renderer options dispatched in `Visualizer` (sibling — read-only).
- `../options/sensors/camera.py` → camera sensor options for the sensors that wrap `genesis.vis.Camera` (sibling — read-only).
- `../ext/pyrender` → vendored renderer/viewer implementation (sibling — read-only).
- `../ext/LuisaRender` → git submodule providing `LuisaRenderPy` (sibling — read-only).

## See Also
- `genesis/repr_base.py` — `RBC` base class used by most classes here.
- `genesis/utils/geom.py`, `genesis/utils/mesh.py`, `genesis/utils/particle.py` — mesh/geom/particle helpers used heavily by `rasterizer_context.py` and `raytracer.py`.

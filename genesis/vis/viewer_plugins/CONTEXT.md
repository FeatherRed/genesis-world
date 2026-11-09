# genesis/vis/viewer_plugins — Viewer Event Plugins

## Intent
Pluggable pyglet window event handling for the Genesis interactive viewer.
Plugins are registered on `Viewer` (built-in ones from here; the ImGui overlay comes from `genesis.ext.pyrender.overlay`).

## API Surface
- `__init__.py` — re-exports `ViewerPlugin`, `EVENT_HANDLED`, `EVENT_HANDLE_STATE`, `RaycasterViewerPlugin`, and `from .plugins import *` (`DefaultControlsPlugin`, `MouseInteractionPlugin`).
- `base.py` — `ViewerPlugin`: base class with no-op pyglet event hooks.
  Hooks: `build(viewer, camera, scene)`, `on_mouse_motion/drag/press/release/scroll`, `on_key_press/release`, `on_resize`, `update_on_sim_step`, `on_draw`, `on_close`.
  An event handler returns `True` (`EVENT_HANDLED`) to stop propagation; `None` to let the next plugin handle it.
- `raycast.py` — `RaycasterViewerPlugin` and its `Raycaster` engine.
  `Raycaster.cast(ray_origin, ray_direction, max_range, envs_idx)` returns a `RayHit | None` (closest hit across envs and solvers).
  `RaycasterViewerPlugin.build` wires a `Raycaster`; `update_on_sim_step` refreshes its BVHs.
- `plugins/default_controls.py` — `DefaultControlsPlugin`: standard keyboard shortcuts (record/save, reset camera, shadow, normals, world/link frame, wireframe, perspective/orthographic, camera frustum, fullscreen, reload shader).
  Register via `ViewerOptions.enable_default_keybinds`.
- `plugins/mouse_interaction.py` — `MouseInteractionPlugin(RaycasterViewerPlugin)`: drag rigid bodies with the mouse via a spring attached where the cursor grabbed them.
  Constructor options: `use_force`, `spring_slack`, `color`, `use_visual_geom`.
  Registers a default keybind (`MouseButton` grab/release) and manages transient debug nodes drawn while dragging.

## Design Decisions
The raycaster builds one BVH per solver target (`RaycastTarget`) and reduces across environments on the viewer side (in torch), deliberately not in the kernel, to keep parallel envs isolated.
`use_visual_geom=True` casts against visual meshes restricted by `material.use_visual_raycasting` (same opt-in as raycaster sensors), so picking follows what is drawn; `False` casts collision geometry only.
`MouseInteractionPlugin` defaults to a spring force (`use_force=True`) so dragging never violates physics; `use_force=False` moves the link onto the cursor directly.
Plugin lifecycle is managed by `Viewer`; `viewer_plugins` only defines plugin classes, never instantiates them.

## Constraints
Plugins must call `super().build(...)` when overriding `build` (e.g. `MouseInteractionPlugin` needs the base `viewer`/`camera`/`scene` fields).
Return `EVENT_HANDLED` (`True`) from an event hook to consume the event; return `None` to keep dispatching.
Do not create files here except CONTEXT.md (read-only extraction).

## Known Issues / Gotchas
`Raycaster.__init__` warms up with a zero-ray cast (FIXME `quadrants#887`) because a host-side read would otherwise race kernel compilation during the first interactive cast.
The screen-to-ray conversion ignores pixel aspect ratio and uses the current camera node matrix, so it can go stale after projection changes (e.g. the `O` key).
`Raycaster` result buffers are shared across callers and guarded by a lock; returned normals are copied so they survive the next cast.
`MouseInteractionPlugin` is active only while the simulation advances; `_sim_running` is derived from the per-step callback.

## Routing Table
- `plugins/` → concrete plugins (`DefaultControlsPlugin`, `MouseInteractionPlugin`).
- `../viewer.py` → registers and dispatches to these plugins (parent — read-only).
- `../keybindings.py` → `Key`/`KeyMod`/`MouseButton` types used by plugins (parent — read-only).
- `../../engine/bvh.py`, `../../utils/raycast.py`, `../../utils/raycast_qd.py` → BVH data and raycast kernels consumed by `raycast.py` (siblings — read-only).
- `../../utils/mesh.py` → `create_cylinder`/`create_plane` helpers used by mouse interaction (sibling — read-only).

## See Also
- `genesis/ext/pyrender/viewer.py` and `genesis/ext/pyrender/overlay.py` — the viewer these plugins attach to, and the built-in ImGui overlay plugin.

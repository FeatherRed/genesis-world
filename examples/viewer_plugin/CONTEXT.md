# Intent

Demonstrates the interactive viewer's plugin system: keybindings, raycasting, and mouse-driven entity manipulation.

# API Surface

- `keyboard_teleop.py` — Franka gripper teleoperation via keyboard keybindings (`genesis.vis.keybindings.Keybind`), including a "press to grip" spacebar.
- `mouse_interaction.py` — Uses the built-in `gs.vis.viewer_plugins.MouseInteractionPlugin` to drag entities; `--use-force` applies spring forces instead of setting position, `--use-visual-geom` grabs the visual mesh (requires `gs.materials.Rigid(use_visual_raycasting=True)`).
- `mesh_point_selector.py` — Subclasses `RaycasterViewerPlugin` to select surface points on rigid meshes with mouse clicks, stores them in link-local coordinates, and writes `out/selected_points.csv` on close.

# Constraints

- Interactive demos; all use CPU backend and `show_viewer=True`.
- When `PYTEST_VERSION` is present in `os.environ`, the run loop breaks after one step, so they smoke-test plugin construction without a real window.

# Dependencies

- Core package (pyrender/pyglet viewer stack is a core dependency); no extra optional backend.

# Gotchas

- Plugins must be attached via `scene.viewer.add_plugin(...)` before `scene.build()`.
- Only entities opting into `use_visual_raycasting=True` can be picked through their visual mesh; otherwise collision geometry is used.
- `mesh_point_selector.py` writes to `out/` (git-ignored), which is created on close.

# See Also

- `../gui/` — the higher-level ImGui overlay built on the same viewer.
- `../usd/kitchen.py` — another consumer of `MouseInteractionPlugin`.

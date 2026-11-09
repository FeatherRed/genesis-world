## Intent
ImGui control-panel overlay for the Genesis viewer (`ImGuiOverlayPlugin`), a Genesis-specific feature living inside the vendored pyrender tree rather than in upstream pyrender.
Re-exported as `from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin` and registered by `genesis/_main.py`.

## API Surface
`ImGuiOverlayPlugin` (plugin.py): a `genesis.vis.viewer_plugins.ViewerPlugin` providing simulation controls (play/pause/step/reset), per-entity joint sliders, an FPS display, multi-step support, and a custom panel registration API.
`style.apply_dark_theme(imgui)`: applies the Genesis control-panel dark theme.
`types`: `EntityJointData`/`EntityCacheEntry` NamedTuples and `build_entity_joint_data(entity, free_joint_pos_limit)` for joint metadata.

## Constraints
Requires the `render` optional extras (`pip install 'genesis-world[render]'`) and an imgui backend wired through the viewer.
Deeply coupled to Genesis (`genesis.engine`, `genesis.vis`, `genesis.utils`); not usable outside genesis.
Only controls environment 0 in batched simulations.

## Design Decisions
Wraps the scene in an `InteractiveScene` so play/pause and scene rebuild take effect on the stepping thread while the loop just calls `scene.step()`.
`on_close` is idempotent (viewer dispatches close on both window-close and scene teardown).
`TOGGLEABLE_PLUGINS` lists only plugins safe to attach/detach on a live viewer; always-on plugins (default keybinds, the overlay itself) are excluded.

# Intent

Demonstrates the built-in ImGui viewer overlay and how to drive simulation interactively from GUI panels.

# API Surface

- `imgui_joint_control.py` — Single script; sets `ViewerOptions(enable_gui=True)` on a plain `gs.Scene` (no manual `InteractiveScene`), which attaches the ImGui overlay (`genesis.ext.pyrender.overlay.ImGuiOverlayPlugin`) and gives play/pause/step/reset, an entity browser with joint sliders, visualization toggles, camera controls, custom panels via `register_panel()`, and in-step scene rebuild.

# Constraints

- Interactive-only demo; requires a working onscreen/offscreen viewer (pyglet/pyrender core dependencies).

# Dependencies

- Core package only (`gs.init(backend=gs.cpu)`); ImGui overlay ships with Genesis (`imgui-bundle` is in the `render`/`dev` extras).

# Gotchas

- `enable_gui=True` lets the overlay manage scene editing internally; the "Rebuild Scene" action is handled inside `step()`, so no manual scene management code is needed.

# See Also

- `../viewer_plugin/` — lower-level custom viewer plugins and keybindings.
- `../tutorials/visualization.py` — basic visualization options.

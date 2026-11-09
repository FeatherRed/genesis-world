# Intent

Rendering integration tests for the Genesis renderers, cameras, interactive viewer, and debug-draw / ImGui overlay.
This directory exercises the pixel-level render output (RGB / depth / segmentation / normal / point cloud), the offscreen `camera.render` API, and the interactive viewer paths (keybindings, mouse interaction, recording, trajectory replay).
It does not test simulation physics beyond what a render needs; those live in sibling `tests/` directories.

# API Surface

## Local fixtures (conftest.py)

- `RENDERER_TYPE` (IntEnum): `RASTERIZER=0`, `RAYTRACER=1`, `BATCHRENDER_RASTERIZER=2`, `BATCHRENDER_RAYTRACER=3`.
- `renderer_type` (function, defaults to `RASTERIZER`) — the single knob parametrized by renderer-dependent tests; the default keeps viewer/overlay tests resolving.
- `renderer` (function) — builds the `gs.renderers` instance for `renderer_type`; RayTracer gets a preset indoor HDR environment + light, BatchRenderer gets `use_rasterizer=` per type.
- `backend` (function, overrides the parent session `backend`) — `gs.cuda` for both batch types, `gs.gpu` for RayTracer, otherwise `--backend` option resolved via `gs.constants.backend` or `gs.cpu`.
- `skip_if_not_installed` (function, autouse) — skips via `SKIP_NO_MADRONA` when `gs_madrona` is missing and `SKIP_NO_LUISA` when `LuisaRenderPy` is missing.
- `base_plus_emissive_glb` (session) — builds a trimesh GLB quad pairing a red base atlas with a blue emissive atlas (used by `test_emissive_composites`).

## Inherited fixtures (tests/conftest.py — see See Also)

`initialize_genesis` (autouse gs.init), `show_viewer`, `png_snapshot`, `tol`, `asset_tmp_path`, `load_trajectory_snapshot` (unused here), `lazy_imports`, and the skip constants `SKIP_NO_MADRONA` / `SKIP_NO_LUISA` / `SKIP_NO_VIEWER` / `SKIP_NO_IMGUI_BUNDLE`, `IS_INTERACTIVE_VIEWER_AVAILABLE`, `is_imgui_bundle_supported`.

## Test files

- `test_offscreen.py` (17 tests) — offscreen camera render API, determinism, env splitting/heterogeneous entities, segmentation maps, point clouds, planes, video recording, GL context isolation, transparency compositing.
- `test_batch_render.py` (3 tests) — BatchRenderer snapshot tests: lights, BatchTexture, fisheye camera.
- `test_debug_draw.py` (3 tests) — debug arrows/lines/spheres/frames, sensor debug visuals, frustum and trajectory markers.
- `test_dynamic_meshes.py` (4 tests) — deformable mesh rendering: batch deformables, UV textures on PBD/FEM, `set_vverts` semantics.
- `test_imgui_overlay.py` (5 tests) — `ImGuiOverlayPlugin` control panel / editing controls snapshots, plugin toggle+pause, scene rebuild, entity capture.
- `test_interactive_viewer.py` (8 tests) — default/custom keybindings, mouse-interaction plugin + raycasting, add-camera consistency, camera sensor context sharing, camera resolution honor, viewer-thread crash traceback, trajectory record/replay pacing.

# Routing Table

- `conftest.py` → renderer/backend/skip plumbing for this folder.
- `test_offscreen.py` → offscreen camera render surface and correctness invariants.
- `test_batch_render.py` → Madrona BatchRenderer snapshot coverage.
- `test_debug_draw.py` → debug draw markers and sensor debug visuals.
- `test_dynamic_meshes.py` → dynamic/deformable mesh and custom vertex rendering.
- `test_imgui_overlay.py` → ImGui overlay plugin and interactive scene editing.
- `test_interactive_viewer.py` → interactive viewer behavior (input, recording, threading).

# Constraints

- Snapshot tests must compare via `rgb_array_to_png_bytes(...) == png_snapshot` (or raw bytes `f.read() == png_snapshot`), never `assert_equal` on RGB for snapshot ground truth.
- New renderer-dependent tests should parametrize `renderer_type` rather than hard-coding a renderer; viewer/overlay-only tests may rely on the default `renderer_type`.
- Geometry must stay inside the camera frustum: the Apple Software Renderer misrasterizes out-of-frustum vertices.
- Interactive-viewer tests gate on `IS_INTERACTIVE_VIEWER_AVAILABLE` (`has_display or has_egl`); overlay tests additionally gate on `is_imgui_bundle_supported`.
- Keep `run_in_thread=(sys.platform == "linux")`: Windows cannot run the viewer in a background thread (pyglet `EventLoop.run` must match the importing thread).
- `test_batch_render.py`/batch paths add explicit lights because the BatchRenderer has no built-in lighting.

# Design Decisions

- One `renderer_type` enum drives `renderer`, `backend`, and `skip_if_not_installed` from a single parametrization instead of per-test branching.
- Optional renderer skips are centralized in the autouse `skip_if_not_installed` fixture (Madrona/Luisa presence), while platform/viewer skips stay as `skipif` markers on each test.
- Snapshots are fetched at runtime from HuggingFace `Genesis-Intelligence/snapshots` (pinned revision) into the tests tree, not committed; `--snapshot-update` regenerates them serially.
- Per-test hardware tolerance is expressed by mutating `png_snapshot.extension._std_err_threshold` / `_ratio_err_threshold` / `_blurred_kernel_size` rather than duplicating comparison code.
- ImGui overlay snapshot determinism is achieved by monkeypatching the plugin (`_apply_deterministic_imgui_overrides`): park the mouse off-panel, load ProggyClean bitmap font, disable AA / baked thick lines, pin `display_framebuffer_scale=(1,1)`.
- The viewer/overlay capture path drives `pyrender_viewer.on_draw()` synchronously and reads the window framebuffer, since `render_offscreen` deliberately skips the plugin loop.

# Known Issues

- RayTracer output is non-deterministic across GPUs (e.g. RTX6000 vs H100), so its snapshots relax thresholds (`_std_err_threshold=3.0`, `_blurred_kernel_size=3`).
- macOS CI falls back to the Apple Software Renderer: it misrasterizes out-of-frustum geometry and is flaky/deterministic-variance prone; many tests `pytest.xfail` under `sys.platform == "darwin" and scene.visualizer.is_software`.
- Old CPU Mesa (`llvmpipe`, LLVM < 20) renders text blurry and makes offscreen depth blits impractically slow; `test_key_press` xfails there.
- `render_offscreen` does not run the viewer plugin loop, so the ImGui overlay never appears in it; overlay capture requires `run_in_thread=False` and the synchronous `on_draw` + framebuffer read.
- BatchRenderer returns one image per environment, while the rasterizer composites all environments into one image; tests branch on this (`per_env`).
- The virtualized Apple Metal GPU on macOS CI is wedged by the FEM implicit PCG solve; `test_deformable_uv_textures` skips when `darwin` + non-CPU backend + `QD_ENABLE_METAL != "0"`.
- `--failure-artifacts-dir` (handled in the parent `png_snapshot`) writes each mismatched frame as `<nodeid>-<sha1>.png` for debugging.

# Test Strategy

- Run: `pytest tests/rendering` (optionally `--backend cuda/gpu`, `-m "required"`, `--snapshot-update` to regenerate).
- `required` marks the tests that must pass in CI; `slow` marks the multi-minute ones (each ~250–300 s).
- Renderer matrix: Rasterizer (CPU or GPU), RayTracer (GPU, Luisa), BatchRenderer rasterizer/raytracer (CUDA, Madrona); optional renderers auto-skip when not installed.
- Snapshot ground truth comes from HF (pinned revision); regeneration must run serially via `--snapshot-update`.

# See Also

- `../conftest.py` → shared fixtures (`initialize_genesis`, `png_snapshot`/`PixelMatchSnapshotExtension`, `show_viewer`, `tol`, `asset_tmp_path`, `load_trajectory_snapshot`, skip constants, EGL/headless detection).
- `../utils/assertions.py` → `assert_pixel_match`, `rgb_array_to_png_bytes`, `assert_allclose`, `assert_equal`.
- `../utils/assets.py` → `get_hf_dataset` (snapshot/asset fetch from HuggingFace).
- `../pyproject.toml` → pytest addopts, xfail_strict, markers.

## Intent
Showcases genesis rendering features through five flat, runnable scripts.
Demonstrates the photorealistic LuisaRender-based `RayTracer`, camera control (follow / animated), a decoupled simulation-render loop, and a renderer speed benchmark.

## API Surface
- `demo.py` — photorealistic showcase using `gs.renderers.RayTracer` with an emissive `ImageTexture` environment, an area light, and a gallery of `gs.surfaces` (Aluminium, Rough, Smooth, Iron, Gold, Glass, opacity) plus `gs.textures.ImageTexture`/`ColorTexture`.
- `follow_entity.py` — attaches a camera to a moving entity via `Camera.follow_entity(cube, fix_orientation=...)` so it tracks it each step.
- `moving_camera.py` — animates the camera pose manually with `Camera.set_pose` on every step while rendering.
- `render_async.py` — decouples stepping and rendering: a background thread runs `scene.step(refresh_visualizer=False)` while the main thread runs `scene.viewer.run()` (viewer configured with `run_in_thread=False`).
- `speed_test.py` — measures renderer FPS by timing `Camera.render(rgb=True, depth=True)` over a fixed horizon.

## Constraints
- Standard genesis example pattern: `import genesis as gs` -> `gs.init(backend=gs.cpu)` -> `gs.Scene(...)` -> `add_entity`/`add_camera` -> `build` -> `step` + `render`.
- Every script defines `main()` and is guarded by `if __name__ == "__main__": main()`, so each is directly runnable and importable.
- Scripts are executed as pytest examples by `tests/test_examples.py` (`pytest.mark.examples`), which sets the `PYTEST_VERSION` environment variable so each script shortens its horizon for the test run.
- No subdirectories; all five scripts are flat files.

## Dependencies
- `demo.py` requires the LuisaRender extension (`LuisaRenderPy` module) for the `RayTracer`; that extension lives in the `genesis/ext/LuisaRender` git submodule.
- `tests/test_examples.py` maps `demo.py` -> `["LuisaRenderPy"]` and skips it via `pytest.importorskip` when the module is absent, so the other four scripts still run without LuisaRender.
- Referenced assets resolve under `genesis/assets/` (not this directory): `textures/indoor_bright.png`, `textures/checker.png`, `meshes/sphere.obj`, `meshes/wooden_sphere_OBJ/wooden_sphere.obj`, `xml/franka_emika_panda/panda.xml`.

## Gotchas
- `demo.py` steps only 10 times by default; pass `-s/--steps` for more and `-v/--vis` to show the GUI viewer.
- The `RayTracer` has no CPU fallback; if `LuisaRenderPy` is unavailable, `demo.py` fails at runtime and only the pytest skip avoids it.
- `render_async.py` relies on `ViewerOptions(run_in_thread=False)` so the viewer loop owns the main thread while the simulation thread steps; `refresh_visualizer=False` prevents the step from also driving the viewer.
- Under pytest (`PYTEST_VERSION` set), `moving_camera.py` and `speed_test.py` drop to 5 iterations and `render_async.py` breaks out of its simulation loop early; `follow_entity.py` still runs its full 100 steps.
- `follow_entity.py` uses `vis_options` (not `viewer_options`) with `show_viewer=False`; its visual output comes from `cam.render()`, not a GUI window.

## See Also
- `../../tests/test_examples.py` — pytest harness that runs every example under `examples/` (sibling — read-only, escalate writes to parent).
- `../sensors/` — camera/render usage in a sensor context (sibling — read-only).
- `genesis/ext/LuisaRender` — the LuisaRender submodule backing the `RayTracer`.

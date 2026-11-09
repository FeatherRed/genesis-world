## Intent
Vendored fork of the `pyrender` rasterizer (upstream mmatl/pyrender, `__version__ = "0.1.45"`) embedded as Genesis's offscreen OpenGL renderer.
Upstream code here is heavily modified for Genesis: a numba-JIT render path, batched multi-environment rendering, segmentation, shadow mapping, debug-marker/x-ray overlays, reflective floors, a per-OS offscreen-context abstraction, and an ImGui overlay.
Consumed elsewhere as `from genesis.ext import pyrender` (see `genesis/vis/rasterizer.py`, `genesis/vis/rasterizer_context.py`) plus direct submodule imports (`genesis.ext.pyrender.jit_render`, `genesis.ext.pyrender.mesh`, `genesis.ext.pyrender.overlay`).

## API Surface
`__init__.py` re-exports the glTF-style scene-graph types and Genesis flags: `Camera`, `PerspectiveCamera`, `OrthographicCamera`, `IntrinsicsCamera`, `Light`, `PointLight`, `DirectionalLight`, `SpotLight`, `Sampler`, `Texture`, `Material`, `MetallicRoughnessMaterial`, `Primitive`, `Mesh`, `Node`, `Scene`, `Renderer`, `Viewer`, `OffscreenRenderer`, `RenderFlags`, `TextAlign`, `GLTF`, `__version__`.
Genesis's actual consumer surface: `OffscreenRenderer(seg_node_map=...)` with `make_current`/`make_uncurrent`/`save_current_context`/`render(...)`/`delete`; `Renderer(res_w, res_h, jit_renderer_instance)` with `render(scene, flags, seg_node_map, is_first_pass, force_skip_shadows)` and `delete`; `JITRenderer(scene, node_list, primitive_list, envs_offset)`; scene-graph constructors `Scene(ambient_light=, bg_color=, n_envs=)`, `Mesh.from_trimesh(...)/from_points(...)`, `Node`, `PerspectiveCamera(...)`, `PointLight(...)`.
`RenderFlags` extends upstream with Genesis-only bits: `SEG`, `RET_DEPTH`, `SKIP_FLOOR`, `REFLECTIVE_FLOOR`, `ENV_SEPARATE`, `SKIP_MARKERS`, `MARKER_XRAY` (see `constants.py`).

## Constraints
Vendored third-party code: do not edit source here in Genesis house style; changes are only documented, never "fixed" in CONTEXT.md (use git for history).
The fork is NOT standalone: `offscreen.py`, `numba_gl_wrapper.py`, `platforms/*`, and `overlay/*` do `import genesis as gs` and call `gs.raise_exception`/`gs.logger`, so importing pyrender requires the genesis package.
The GL platform is committed at first PyOpenGL import: `PYOPENGL_PLATFORM` (and, for OSMesa, the whole platform choice) must be set before importing genesis/pyrender.

## Dependencies
Python: `numpy`, `numba` (JIT + `cache=True`), `Pillow` (texture), `trimesh` (mesh conversion), `networkx` (scene graph), `pyglet` (window/context and viewer), `OpenGL`/PyOpenGL with platform plugins (`OpenGL.EGL`, `OpenGL.osmesa`, CGL bindings), `scipy` (overlay quaternion math), `imgui` (via the overlay plugin).
System: EGL (libEGL) on Linux, OSMesa (libOSMesa) for software rendering, CGL/OpenGL.framework on macOS; an OpenGL 3.3+ (ideally 4.2) driver.
Env vars: `PYOPENGL_PLATFORM` ∈ `egl|cgl|osmesa|pyglet` (else OS default: egl on linux, cgl on darwin, pyglet fallback); `EGL_DEVICE_ID` (int index into EGL devices); `GS_DISABLE_OFFSCREEN_MARKERS`.

## Design Decisions
Two cooperating render paths: `Renderer` (renderer.py) owns GL state, viewports, and framebuffers, then delegates every per-primitive draw to `JITRenderer` (jit_render.py); the normal-visualization pass and text/font rendering stay on the legacy Python path.
`JITRenderer` flattens the scene graph into contiguous numpy arrays (vao_id, pose, pbr_mat, spec_mat, render_flags, mode, n_instances, n_indices, light, shadow_map, light_matrix, ...) and numba-JIT-compiles the forward/shadow/point-shadow/read-buffer kernels once (`gen_func_ptr`, `cache=True`).
`numba_gl_wrapper.py`'s `GLWrapper` builds numba-callable ctypes wrappers around OpenGL entry points and unboxes them as a single `GLFunc` struct so kernels avoid per-call Python attribute lookups; unavailable GL functions (e.g. OpenGL<4.2 instanced-base-instance) are skipped at load and the kernel falls back.
`OffscreenRenderer` selects a `platforms.Platform` implementation: EGL (surfaceless, Linux), CGL (window-less, macOS), OSMesa (software), or pyglet hidden 1x1 window; all render into renderer-owned framebuffer objects, so the platform only supplies a current context.
The GL context is process/thread-global and shared by all renderers: `save_current_context` returns a self-contained restore callable so tearing one renderer down does not strand another that is mid-render.
Batched multi-environment rendering: `Scene.n_envs`, `Primitive.envs`/`is_env_instanced`/`env_idx`, and `JITRenderer.envs_offset`/`env_offset_buffer`; `ENV_SEPARATE` renders each environment at its true pose for its camera, otherwise all are laid out side by side on a grid.
Segmentation uses `RenderFlags.SEG` + `seg_node_map` (Node → color) with the segmentation shaders.
Debug markers: `Mesh.is_marker`; `SKIP_MARKERS` hides them, `MARKER_XRAY` renders occluded markers as dimmed ghosts via a `GL_GREATER` depth pass.
Reflective floor: `Mesh.from_trimesh(is_floor=...)`, `REFLECTIVE_FLOOR`/`SKIP_FLOOR` flags, a floor framebuffer pass, and `JITRenderer.reflection_mat` mirror the floor plane.
Shadow mapping: `light._generate_shadow_texture`, per-light `shadow_map_revision` cache, and separate shadow/point-shadow passes; `SHADOW_TEX_SZ = 8192`.

## Known Issues
Software renderers (`llvmpipe`, `Apple Software Renderer`) are detected and force shadows/plane-reflection off; rendering is extremely slow.
OSMesa needs `PYOPENGL_PLATFORM=osmesa` set before importing genesis (rasterizer.py auto-selects it on headless Linux lacking EGL and a display).
EGL teardown deliberately skips `eglTerminate` (NVIDIA driver returns EGL_BAD_ACCESS on terminate/re-init cycles); contexts compare handles by underlying address, not pointer identity.
numba `cache=True` writes to the numba cache dir; the first render compiles kernels and is slow.
GL_BLEND gives non-deterministic results on Nvidia GPUs (FIXME in jit_render.py) — enabled only when a material is actually transparent.

## Routing Table
- `platforms/` → offscreen GL-context backends (`Platform` ABC, EGL/CGL/OSMesa/pyglet).
- `overlay/` → ImGui simulation/joint control panel (`ImGuiOverlayPlugin`).
- `shaders/` → GLSL (mesh, depth, normal, shadow, segmentation, flat, text, debug_quad).
- `fonts/` → bundled TTFs (OpenSans, SpaceMono, UbuntuMono) for text rendering.
- `jit_render.py`, `numba_gl_wrapper.py` → numba-JIT draw path and GL function wrapper.
- `renderer.py`, `offscreen.py`, `viewer.py` → render orchestration, offscreen context, on-screen viewer.
- `scene.py`, `node.py`, `mesh.py`, `primitive.py` → scene graph + Genesis multi-env/marker/floor extensions.
- `light.py`, `camera.py`, `material.py`, `texture.py`, `sampler.py` → light/camera/material/texture types (shadow-map helpers in light.py).
- `shader_program.py`, `font.py`, `trackball.py`, `utils.py`, `constants.py`, `version.py` → GL program/font caches, camera controls, shared helpers, constants.
- Siblings (read-only, escalate writes to parent): `../vis/rasterizer.py`, `../vis/rasterizer_context.py`, `../../engine/sensors/camera.py`, `../../_main.py` are the consumers.

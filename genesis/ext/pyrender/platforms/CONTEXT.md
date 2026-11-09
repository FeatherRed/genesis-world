## Intent
Offscreen OpenGL-context backends for `pyrender.OffscreenRenderer`, refactored from upstream's single `platform.py` into a `Platform` ABC with one implementation per OS/windowing strategy.
Rendering targets framebuffer objects allocated by the renderer, so a platform only has to provide a current context (plus a 1x1 default framebuffer for OSMesa).

## API Surface
`base.Platform` (ABC): `init_context()`, `make_current()`, `make_uncurrent()`, `delete_context()`, and `save_current_context()` (returns a zero-arg restore callable or None).
Implementations (all subclass `Platform`): `EGLPlatform(device_id=None)` in `egl.py`, `CGLPlatform` in `cgl.py`, `OSMesaPlatform` in `osmesa.py`, `PygletPlatform` in `pyglet_platform.py`.
`egl.py` also exposes `query_devices()`, `get_default_device()`, `get_device_by_index(i)`, and `EGLDevice`.

## Constraints
Do not import OpenGL before the platform is chosen; PyOpenGL binds every GL function to the library of the platform selected at first import, so the choice is process-wide.
`egl.py`, `cgl.py`, `osmesa.py` import `genesis as gs` for `gs.raise_exception`/`gs.logger` — the package is not standalone.
`osmesa.py` raises unless `PYOPENGL_PLATFORM=osmesa` was set before importing genesis.

## Design Decisions
Context is process/thread-global and shared across all renderers, so teardown restores whatever was current before (`save_current_context` / address-based ownership checks) instead of blindly rebinding.
EGL contexts are surfaceless (`EGL_NO_SURFACE`); `EGL_DEVICE_ID` picks a specific device via the `EGL_EXT_device_enumeration`/`EGL_EXT_platform_device` extensions.
CGL uses a private `ctypes.CDLL` handle to the OpenGL framework (PyOpenGL ships no CGL bindings) and clears pyglet's current-context bookkeeping when it takes over.
PygletPlatform tries 4.x then 3.3 configs with a hidden 1x1 window, disabling pyglet's shadow window.

## Known Issues
EGL `delete_context` intentionally never calls `eglTerminate` (NVIDIA returns EGL_BAD_ACCESS on terminate/re-init cycles).
CGL requests a hardware-accelerated pixel format first, falling back to the software renderer only if none is found (software renderer requires a 3.2 core profile to compile shaders).

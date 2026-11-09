# genesis/ext

## Intent
Houses vendored / third-party code that the rest of Genesis imports rather than re-implementing: a forked `pyrender` rasterizer, a forked `urdfpy` URDF parser, IsaacGym terrain generators, a monkey-patch for the `trimesh` OBJ loader, and a `LuisaRender` git submodule for ray tracing.
These are external libraries adapted to Genesis conventions — they are not written in the project's own style, so do not "refactor" them toward Genesis idioms; treat them as upstream snapshots plus targeted local patches.

## API Surface
`genesis.ext` and `genesis.ext.isaacgym` are PEP 420 namespace packages (no `__init__.py`); `pyrender` and `urdfpy` are regular packages with `__init__.py`.
Entry points used by the rest of Genesis (read-only consumers, do not modify from here):
- `from genesis.ext import pyrender` — `genesis/vis/rasterizer.py`, `genesis/vis/rasterizer_context.py` (rasterizer render path).
- `from genesis.ext.pyrender.jit_render import JITRenderer` — `genesis/vis/rasterizer_context.py`.
- `from genesis.ext.pyrender.mesh import Mesh` — `genesis/engine/sensors/camera.py` (TYPE_CHECKING only).
- `from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin` — `genesis/_main.py` (`gs launch`/`gs view` interactive overlays).
- `from genesis.ext import urdfpy` — `genesis/utils/urdf.py`, `genesis/utils/mjcf.py`.
- `from genesis.ext.urdfpy.urdf import URDF` — `genesis/engine/entities/hybrid_entity.py`.
- `urdfpy.URDF` + `robot.to_xml()` — `genesis/options/morphs.py`.
- `from genesis.ext.isaacgym import terrain_utils` — `genesis/utils/terrain.py`.
- `from genesis.ext import _trimesh_patch` — `genesis/__init__.py` (imported for side effects, very early in package init).
- `LuisaRenderPy` (built from the submodule) — loaded from `ext/LuisaRender/build/bin` in `genesis/vis/raytracer.py`.

## Routing Table
- `pyrender/` → vendored fork of `mmatl/pyrender` 0.1.45 (the OpenGL rasterizer render path).
- `urdfpy/` → vendored fork of `mmatl/urdfpy` 0.0.22 (URDF XML parsing and forward kinematics).
- `isaacgym/` → `terrain_utils.py` heightfield terrain generators (from NVIDIA IsaacGym).
- `_trimesh_patch.py` → monkey-patch replacing `trimesh.exchange.obj.load_obj` on the pip-installed `trimesh`.
- `LuisaRender/` → git submodule (gitlink) pointing at `https://github.com/Alif-01/LuisaRender`; ray-tracing backend.

## Vendored Components (summary)
- `pyrender/` — the default rasterization backend.
  Fork of upstream pyrender with Genesis-specific additions: a numba-JIT render path (`jit_render.py`, `numba_gl_wrapper.py`), a per-OS offscreen-context abstraction (`platforms/`: EGL/CGL/OSMesa/pyglet), an `overlay/` ImGui plugin, and a rewritten `offscreen.py`.
  See `pyrender/CONTEXT.md`.
- `urdfpy/` — XML-to-trimesh URDF parsing used for URDF/XACRO/MJCF imports and rigid/hybrid entity morphs.
  Upstream parser plus local patches (mesh/texture loading through `genesis.utils.mesh.load_trimesh`, `package://` path resolution, `gs.raise_exception` error reporting).
  See `urdfpy/CONTEXT.md`.
- `isaacgym/terrain_utils.py` — `SubTerrain` heightfield container and terrain generators (`random_uniform_terrain`, `sloped_terrain`, `pyramid_sloped_terrain`, `discrete_obstacles_terrain`, `wave_terrain`, `stairs_terrain`, `pyramid_stairs_terrain`, `stepping_stones_terrain`, `fractal_terrain`, `convert_heightfield_to_trimesh`).
  Adapted to use `genesis as gs`, `gs.np_float`, and `genesis.utils.geom` (cubic spline).
  `genesis/utils/terrain.py` re-implements `fractal_terrain` and `convert_heightfield_to_trimesh` locally (watertight SDF-friendly variant), so only the other generators are actually consumed from here.
- `_trimesh_patch.py` — vendored copy of trimesh's `exchange/obj.py` (`load_obj` + `export_obj`) plus, at module import, `trimesh.exchange.obj.load_obj = load_obj` and `trimesh.exchange.load.mesh_loaders.update({"obj": load_obj})`.
  The `load_obj` replacement loads OBJ materials/vertex-colors/texcoords and groups faces by material.
  `export_obj` is defined but not monkey-patched (leftover from the upstream copy).
  Imported for side effects from `genesis/__init__.py`.
- `LuisaRender/` — empty placeholder gitlink in worktrees; populate with `git submodule update --init` if its content is needed.
  Built to produce the `LuisaRenderPy` Python binding used by `genesis/vis/raytracer.py`.
  Changes inside it belong to the submodule repo, not this superproject.

## Design Decisions
Genesis previously vendored the entire `trimesh` package under `genesis/ext/trimesh/`; commit `f1ee64b5` ("Get rid of internally managed trimesh deps", #918) deleted it (~50k lines) and kept only the OBJ loader as `_trimesh_patch.py`, monkey-patching the externally installed `trimesh`.
The `genesis.ext` namespace package lets vendored code be imported as `genesis.ext.*` without any dispatcher module.

## Constraints
Do not treat these files as first-party Genesis code: preserve upstream structure and add only targeted, clearly-visible local patches.
`genesis.ext` and `genesis.ext.isaacgym` must remain namespace packages unless intentionally converting them (they rely on PEP 420 discovery).
Never create files inside `LuisaRender/` to "fill in" content; it is a tracked gitlink.

## Dependencies
- `pyrender/`: `PyOpenGL`, `pyglet`, `numba`, `PIL`/Pillow; system GL (EGL on Linux, CGL/OpenGL framework on macOS, OSMesa for software) and optionally `ImGui` (overlay).
- `urdfpy/`: `numpy`, `trimesh`, `networkx`, `PIL`; internally `genesis.utils.mesh.load_trimesh` and `genesis.ext.pyrender`.
- `isaacgym/terrain_utils.py`: `numpy`, `genesis.utils.geom`.
- `_trimesh_patch.py`: pip-installed `trimesh`, `PIL`.
- `LuisaRender/`: built from the submodule (C++/CUDA renderer) before the raytracer path is usable.

## Known Issues
`_trimesh_patch.py` has a latent broken fallback: if `PIL` import fails it does `from ..exceptions import ExceptionWrapper`, but `genesis.exceptions` does not exist (leftover from the upstream trimesh package layout); harmless in practice because Pillow is a hard dependency.
`PYOPENGL_PLATFORM` must be chosen before the first `pyrender`/OpenGL import (see `genesis/vis/rasterizer.py`); OSMesa requires `PYOPENGL_PLATFORM=osmesa` before importing `genesis`.

## See Also
- `../vis/rasterizer.py`, `../vis/rasterizer_context.py`, `../vis/raytracer.py` — the three render paths that consume this directory (siblings — read-only).
- `../utils/urdf.py`, `../utils/mjcf.py`, `../utils/terrain.py`, `../options/morphs.py`, `../engine/entities/hybrid_entity.py` — asset-loading consumers (siblings — read-only).

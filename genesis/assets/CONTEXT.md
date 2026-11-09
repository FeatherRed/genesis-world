# genesis/assets/ — Bundled Static Data

## Intent
Static data shipped with the package (declared in `pyproject.toml` `[tool.setuptools.package-data]`): meshes, textures, robot/asset URDF and MJCF XML models, and sample scenes used by examples, tests, and the `gs play`/`gs launch` CLI defaults.

## Contents
- `meshes/` — OBJ/GLB/STL/DAE meshes (duck, dragon, bunny, sphere, terrain, worm, bolt_nut, env_sphere, ...) plus textures/MTL.
- `textures/` — checker, indoor HDRI.
- `urdf/` — robot/asset URDF models (franka/panda_bullet, go2, anymal_c, shadow_hand, kuka_iiwa, drones, wheel, plane, simple, 3763 dataset) with collision/visual meshes.
- `xml/` — MJCF scenes (franka_emika_panda, franka_sim, universal_robots_ur5e, humanoid, walker, dominos, tet/rope/ball primitives, ...).
- `tower/` — generated stacking-tower GLB/URDF assets (collision + visual pairs).

## Constraints
Pure data — no Python modules, no `__init__.py`.
Do not refactor, split, or "improve" these files; they are load-bearing fixtures referenced by examples and tests via `gs.morphs.*` and `gs.morphs.MJCF(file="xml/...")` (path relative to `gs.utils.misc.get_src_dir()`).

## Notes for Agents
This directory is large (~hundreds of binary files); treat as static assets, not source.
Some models carry their own `generate_*.py` scripts (bolt_nut, tower, plane) documenting how they were produced.

## See Also
- `../constants.py` — the format constants used to load these assets.
- `../utils/urdf.py`, `../utils/mjcf.py`, `../utils/mesh.py` — the parsers that consume them.
- `../../examples/` and `../../tests/` — the consumers (siblings, read-only).

## Status
Complete (data-only node; no Python API to document).

# tests/core — Core Engine Internals Tests

## Intent
Unit tests for the core engine internals of Genesis: backend/array-mode switching, BVH construction and query, Quadrants (field vs ndarray) kernel-cache behavior, scene/solver misc behaviors, recorders, surface shortcuts, and low-level geometry/utils helpers.
The directory is flat (no subdirectories); each file below is a self-contained test module.

## API Surface
- `test_backend.py` — backend selection, `GS_ENABLE_NDARRAY` field-vs-ndarray switching, memory-release-on-destroy, and subprocess scene-run/termination.
- `test_bvh.py` — `genesis.engine.bvh` (`build_bvh`, `func_bvh_query_leaves`, `get_bvh_data`) correctness for tree build and query.
- `test_misc.py` — scene lifecycle (destroy/rebuild), entity naming, morph offset resolution, index sanitization, furthest-point sampling, solver `StateChange` subscribers, per-solver/per-env gravity and time, derived substeps, solver-stability warnings.
- `test_quadrants.py` — Quadrants array backend (`qd_to_torch`), and white-box kernel source/front-end cache (`src_ll_cache`/`fe_ll_cache`) observances under `QD_OFFLINE_CACHE`; `linear_to_lower_tri` index math.
- `test_recorders.py` — MPL vector-field/line plotters, CSVFile/NPZFile writers, recorder error handling, and `VideoFile` frame counting.
- `test_surface.py` — `gs.surfaces` shortcut→texture resolution, batched packed-RGBA emissive fallback, and UV handling in `genesis.utils.mesh`.
- `test_utils.py` — geometry helpers (NumPy/Torch/Quadrants consistency, slerp, polar decomposition), inertial-property composition, `unique_rows`, mesh inertia, genuine interpenetration depth, grayscale export, `RunningMean`, logger, `warn_once`.

## Constraints
- Source code under test lives in `genesis/`; these tests may import engine internals (`genesis.engine.*`, `genesis.utils.*`, solver kernels) directly — that is the point of this directory.
- Subprocess-based tests rely on the trailing `if __name__ == "__main__": globals()[sys.argv[1]](sys.argv[2:])` dispatch; do not remove it.
- Child processes communicate results via sentinel exit codes `RET_SUCCESS = 42` and `RET_SKIP = 43`.
- Shared fixtures, markers, and pytest config are defined at `tests/` root (`tests/conftest.py`, `pyproject.toml`) — do not re-document or duplicate them here.

## Design Decisions
- Tests that manage their own `gs.init`/env use `@pytest.mark.parametrize("backend", [None])` to disable the autouse `initialize_genesis` fixture at the worker level; the parent then spawns `python -m tests.core.<module> <child_fn>` with an explicit backend and env.
- `GS_ENABLE_NDARRAY` toggles the field vs ndarray array backend per cycle/child; cache tests also set `QD_OFFLINE_CACHE=1`, `QD_OFFLINE_CACHE_FILE_PATH=<tmp_path>`, and pop `GS_ENABLE_ZEROCOPY`.
- Quadrants kernel-cache behavior is asserted by inspecting the internal `kernel._primal.src_ll_cache_observations` / `fe_ll_cache_observations` counters (white-box).
- Recorder golden images use the syrupy `png_snapshot` fixture; video correctness is verified by decoding frame count with PyAV (`av`), not pixel snapshots.

## Known Issues
- `test_linear_to_lower_tri` exists because Metal GPU `sqrt` can return slightly-below-exact results for perfect squares, requiring a post-correction in the triangular-index formula.
- Windows xfails/skips: `test_process_terminates_on_signal` is skipped (no signal exit status); `test_redirect_libc_stderr` xfails on `quadrants#1075` (Quadrants native logger keeps its own stderr handle).
- `test_plotter` monkeypatches `genesis.recorders.file_writers.VideoFileWriter.process` to observe frames; recorder thread failures surface on the stepping thread via `sync()` or the next step.
- `test_genuine_interpenetration` and `test_ndarray_no_compile` are `slow` (≈300 s and ≈200 s); both are also `required`.

## Test Strategy
- Markers used here: `required` (nearly all), `slow` (`test_ndarray_no_compile`, `test_genuine_interpenetration`), custom `precision("32")` (`test_warn_solver_numerical_stability`) and `cache(False)` (`test_fps_cache_round_trip`), both resolved in `tests/conftest.py`.
- CPU/GPU and field/ndarray coverage is exercised via subprocess parametrization rather than the session `backend` fixture for the init-managing tests; the remaining tests rely on the autouse `initialize_genesis` fixture.
- Run a single module with `pytest tests/core/test_<name>.py`; snapshot regeneration requires serial execution (`--snapshot-update`, `-n 0`).
- Coverage gaps: recorder pixel output is snapshot-only for plotters; `VideoFile` is verified by frame count only.

## See Also
- `../conftest.py` — shared fixtures (`initialize_genesis`, `backend`, `tol`, `show_viewer`, `png_snapshot`) and marker resolution (sibling — read-only, escalate writes to `tests/`).
- `../utils/assertions.py` — `assert_allclose`/`assert_equal`/`rgb_array_to_png_bytes` (sibling — read-only).
- `../utils/collision.py`, `../utils/assets.py` — `get_genuine_interpenetration`, `display_collision_pairs`, `get_hf_dataset` (sibling — read-only).
- `genesis/engine/bvh.py`, `genesis/engine/solvers/rigid/*`, `genesis/utils/*`, `genesis/options/surfaces.py`, `genesis/recorders/*` — code under test.

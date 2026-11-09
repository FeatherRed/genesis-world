## Intent

Benchmark suite for Genesis rigid-body simulation performance, not correctness.
Each case builds a representative scene, measures compile time and steady-state solver throughput, and writes one pipe-delimited report line.
The reports feed `tests/upload_benchmarks_table_to_wandb.py` and `tests/monitor_test_mem.py`.

## API Surface

- `test_rigid.py` — the only test module (1160 lines).
- `test_speed` — single parametrized test; selects a scene fixture via `request.getfixturevalue(runnable)` and logs its metrics.
- Scene factories `make_*` — `make_go2`, `make_anymal`, `make_g1_fall`, `make_double_smplx`, `make_shadow_hand_cubes`, `make_dex_hand`, `make_convexify`, `make_nonconvex_spacecraft`, `make_table_bussing`; each returns `(scene, step_fn, SceneMeta)`.
- `SceneMeta` namedtuple — `compile_time`, `step_dt`, `duration_warmup`, `duration_record` (defaults `STEP_DT=0.01`, warmup 45s, record 15s).
- `get_rigid_solver_options` / `get_file_morph_options` — pin solver and mesh-morph options to the 0.2.1 release baseline keyed by commit timestamp.
- `run_benchmark(step_fn, *, n_envs, meta)` — runs the warmup/record loop, returns `dict(compile_time, runtime_fps, realtime_factor)`.
- Fixtures — `stream_writers` (session: per-worker `speed_test-{worker}.txt`), `factory_logger` (function), and one thin runnable wrapper fixture per scene (`go2`, `anymal_random`, `g1_fall`, `g1_fall_accessors`, `double_smplx`, `shadow_hand`, `dex_hand`, `convexify`, `nonconvex_spacecraft`, `table_bussing`).
- `__init__.py` — empty.

## Constraints

- Module `pytestmark` sets `benchmarks`, `cache(False)`, `debug(False)`, `use_deterministic_algorithms(False)`.
- `benchmarks` is an exclusive, opt-in marker: run only via `-m benchmarks` alone; any combination is rejected in `tests/conftest.py`.
- Benchmarks are GPU-only: a `--backend cpu` run is rejected and the session backend is coerced to `gpu`.
- Excluded from default runs by `pyproject.toml` addopts `-m "not (benchmarks or examples)"`.
- Heavy scenes override the warmup/record window via `SceneMeta` (typically 20s warmup / 5s record) instead of the 45s/15s defaults.

## Design Decisions

- Throughput is wall-clock: `runtime_fps = num_steps * max(n_envs, 1) / record_time`, and `realtime_factor = runtime_fps * step_dt`.
- A single `qd.sync()`-bracketed pre-warmup step forces CUDA-context init and kernel JIT before timing, so the warmup budget is spent on steady-state steps rather than cold-start.
- `qd.sync()` brackets the warmup→record boundary and the final step to bound wall-clock error.
- Solver/morph options are pinned by commit timestamp (`get_git_commit_timestamp`) so current numbers stay comparable to official release 0.2.1 despite later default changes.
- `use_deterministic_algorithms(False)` lets the autotuner choose the solve implementation the benchmark measures.
- Each xdist worker writes its own `{stem}-{worker_id}.txt` report; worker `gw0` prunes stale worker files.
- The dtype is chosen at collection time via `GS_ENABLE_NDARRAY`: `'0'` → field dtype (full 12-case `BENCHMARKS_FIELD`), otherwise ndarray (reduced 3-case `BENCHMARKS_NDARRAY`).

## Known Issues

- `test_speed` and its fixtures request `solver`, `n_envs`, `gjk`, and `printer_session`, which are not defined in this directory, `tests/conftest.py`, or the `genesis` package at this commit; they resolve from external test tooling, so verify the fixture providers before running the suite standalone.
- The `solver` and `gjk` parametrize columns are always `None` today (Genesis defaults); they are sweep hooks, not active sweeps.
- The per-row `backend` label (`gs.cpu`/`gs.gpu`) is not requested by `test_speed`; the actual runtime backend is the coerced session `gpu` backend for every case.
- `n_envs == 0` means an unbatched scene, whose control targets are unbatched rather than `(n_envs, ...)`.
- `make_double_smplx` caps `max_collision_pairs=40` so the 318-DOF Jacobian stays within int32 at `n_envs=4096`.
- `make_table_bussing` re-arms the PD position hold after `scene.reset` because `set_state` flips `ctrl_mode` to FORCE on reset envs.
- Assets come from HF datasets (`get_hf_dataset`) and bundled `urdf/`/`meshes/` paths; first run needs network/cache access.
- The `g1_fall_accessors` variant carries a TODO noting whole-scene reset is still slow (`partial=False` default).

## Test Strategy

- Run explicitly: `pytest tests/benchmarks -m benchmarks`, optionally with `--speed-test-filepath` to redirect reports.
- Pass `--mem-monitoring-filepath` to launch `tests/monitor_test_mem.py` (CUDA + Linux only; enforced in `tests/conftest.py`).
- There are no assertions; the report files are the deliverable and the reported numbers are the validation signal.
- Tune the `STEP_DT`/`DURATION_WARMUP`/`DURATION_RECORD` constants or per-scene `SceneMeta` overrides to change timing windows.

## See Also

- `../conftest.py` — benchmark marker enforcement, backend coercion, `initialize_genesis`, mem-monitor launch (parent scope).
- `../upload_benchmarks_table_to_wandb.py` — consumes the report files.
- `../monitor_test_mem.py` — memory monitoring consumer.
- `../utils/misc.py` — `pprint_oneline` (report formatting).
- `../utils/assets.py` — `get_git_commit_timestamp`, `get_hf_dataset` (baseline pinning and asset fetching).

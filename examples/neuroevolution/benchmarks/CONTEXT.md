# examples/neuroevolution/benchmarks

## Intent
The Stage-3 measurement harness of the PyTorch -> EvoX Compiler migration: it answers, per pilot, whether the migrated evaluation environment still computes the pre-migration maths and how fast it runs.
Every measurement is a committed JSON record under `results/`, reproducible from the exact `command` it stores.
Phase 1 (this directory's current state) implements `replay` only; `throughput`, `residency` and `batch` are the later phases and are documented in the CLI's help text and module docstrings instead of being stubbed.
Nothing here is imported by the trainer or by the pilot envs: the dependency arrow points one way, from the harness into the siblings.

## API Surface
- `common.py` — record schema and provenance (`new_record`, `write_record`, `git_info`, `packages`, `host_info`, `gpu_info`), `stats`, `Timer`, `wrap_function`/`wrap_method`, `sync`, `run_guarded`, `setup_paths`, `BASELINE_COMMIT = "ad2f07e5"`.
- `oracle.py` — independent reimplementations: `go2_observations` (transcribed from `tests/test_go2_neat_kernels.py`), `go2_reward` (via `go2_legacy_reference.walking_reward_terms`), `go2_done`, `hover_oracle_step`/`hover_graph_terms`/`hover_reset_obs` (via `hover_legacy_reference`).
- `bench_envs.py` — the CLI: `replay --task {go2,hover} --pop-size 8 --horizon 100 --master-seed 0 --repeat 0 --process-role only [--generations 2] [--out PATH]`.
- `results/` — the committed records (`env_<task>_replay_seed<seed>[_r<repeat>].json`).

## Constraints
- No module imports `genesis`, `torch`, `tensordict` or `evoxc` at module level; `common.py` imports none of them at all, and the oracles import their siblings inside the functions.
- The pilot environments, kernels, oracles and the tests are read-only siblings: the harness instruments them by runtime monkey-patching (`wrap_function`/`wrap_method`) and restores every patch when a generation loop ends.
- Records are the deliverable: a failed run still writes `status="failed"` plus the error block, and `contended` records a busy GPU or a 1-minute load average above 4.
- Timings are host wall clock with `common.sync()` around each generation; a record states which regions were synchronized and what the unmeasured `residual` is.
- Files stay well under ~1000 lines, one module per concern, and never write outside this directory.

## Routing Table
- `common.py` -> record schema, host/git/package provenance, statistics and timing wrappers.
- `oracle.py` -> the independent per-step maths the parity check compares against.
- `bench_envs.py` -> the `replay` measurement and the CLI entry point.
- `results/` -> committed raw records, one file per measurement.
- `../../locomotion/go2_neat_env.py`, `../../drone/hover_neat_env.py` -> the instrumented pilot envs (siblings — read-only, escalate writes to parent).
- `../trainer.py`, `../seeding.py` -> the shared layer the pilots are built on (siblings — read-only).

## Gotchas
- `go2_neat_kernels.run_advance` returns only the aggregate reward, so a Go2 per-term deviation cannot be measured; the record sets `reward.per_term_available = false` with the reason and compares the total instead.
- `hover_legacy_reference` reports reward terms pre-scale while `hover_kernels` reports them `dt`-scaled, so `oracle.hover_graph_terms` applies the one documented rescale (`total` agrees on both sides).
- `Go2NeatEnv.evaluate` draws the per-generation seed perturbation outside its own `wall_time_s`, so `generation_total` and the env's counter differ; `phases.generation_total_scope` says so, and that draw has been observed to cost seconds once (torch geometry warm-up) before settling at ~10 ms, so compare `host_rng` across records instead of assuming it is free.
- The pilot builds dominate a run (`build_s` 5-40 s, quadrants disk-cache dependent), which is why every record reports `phases.build_s` separately from the generations.
- A bare `bench_envs.py` prints usage and exits 0 so the `examples/**/*.py` smoke harness does not treat the CLI as a failing example; naming a subcommand parses strictly.

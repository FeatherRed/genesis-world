# examples/neuroevolution/benchmarks

## Intent
The Stage-3 measurement harness of the PyTorch -> EvoX Compiler migration: it answers, per pilot, whether the migrated evaluation environment still computes the pre-migration maths and how fast it runs.
Every measurement is a committed JSON record under `results/`, reproducible from the exact `command` it stores.
Phase 1 (`bench_envs.py`) implements `replay`; Phase 2 (`bench_runtime.py`) implements `throughput` and `residency-parity`, which add the steps-per-second matrix and the NumPy-vs-device residency comparison (parity, per-step host traffic, per-graph compile times).
Only the `batch` pop-size/horizon sweep remains a later phase, documented in the CLI help text instead of being stubbed.
Nothing here is imported by the trainer or by the pilot envs: the dependency arrow points one way, from the harness into the siblings.

## API Surface
- `common.py` — record schema and provenance (`new_record`, `write_record`, `git_info`, `packages`, `host_info`, `gpu_info`), `stats`, `Timer`, `wrap_function`/`wrap_method`, `wrap_traffic`/`Counter` (per-call and per-byte traffic aggregation), `sync`, `run_guarded`, `setup_paths`, `BASELINE_COMMIT = "ad2f07e5"`.
- `oracle.py` — independent reimplementations: `go2_observations` (transcribed from `tests/test_go2_neat_kernels.py`), `go2_reward` (via `go2_legacy_reference.walking_reward_terms`), `go2_done`, `hover_oracle_step`/`hover_graph_terms`/`hover_reset_obs` (via `hover_legacy_reference`).
- `bench_envs.py` — the Phase-1 CLI `replay --task {go2,hover} --pop-size 8 --horizon 100 --master-seed 0 --repeat 0 --process-role only [--generations 2] [--out PATH]` plus the reusable helpers Phase 2 imports: `_load_task`, `_build_env(env_cls, pop_size, master_seed, horizon, *, residency, backend, device, stop_when_done, perturbation)`, `_instrument`, `_go2_parity`/`_hover_parity`, `_generation_entry`, `_merge_parity`, `_fingerprint`, `StepCapture`, `CallCapture`.
- `runtime_probes.py` — the measurement vocabulary (timer groups, traffic tags, region descriptions, `DEVICE_RESIDENT_INTERMEDIATES`) and the patches that produce the numbers: per-graph build timer, host/device transfer and rollout-entry wrappers for both residencies, the `RigidOptions` runtime override, the getter-vs-live-buffer state probe; every probe returns its own `restore` callable.
- `runtime_measure.py` — `_measure_env` (build, warm up on generation 0, time generations 1-3, restore every patch), the per-generation record entry, per-step means, traffic totals, GPU-init/provenance helpers (`_ensure_gs_init`, `_filter_own_gpu_pids`, `_process_tree`) and `_consistency_checks`.
- `bench_runtime.py` — the Phase-2 CLI: `throughput --task {go2,hover} --pop-size {8,32} --horizon 200 --residency {numpy,device} --backend numpy|iree --device cpu:0|rocm:0 [--gpu] [--repeat N] [--process-role cold|warm|only] [--collision-cap N] [--out PATH]` and `residency-parity --task ... --pop-size 8 --horizon 200 [--gpu]`, plus the record builders, parity comparison, workload description and summary printer; both subcommands build once, run generation 0 as an untimed warm-up, then generations 1-3 as the timed set.
- `results/` — the committed records (`env_<task>_replay_seed<seed>[_r<repeat>].json`, `env_<task>_throughput_<residency>_<physics>_pop<n>[_r<repeat>].json`, `residency_<task>_<gpu|device_cpu>[_r<repeat>].json`).

## Constraints
- No module imports `genesis`, `torch`, `tensordict` or `evoxc` at module level; `common.py` imports none of them at all, the oracles import their siblings inside the functions, and `bench_runtime.py` imports genesis lazily inside its GPU path.
- The pilot environments, kernels, oracles and the tests are read-only siblings: the harness instruments them by runtime monkey-patching (`wrap_function`/`wrap_method`/`wrap_traffic`, class-level for the lazily built device rollouts) and restores every patch when a generation loop ends, so a `--collision-cap` override is a runtime patch of `gs.options.RigidOptions`, never a source edit.
- Genesis must be initialised before any `torch.cuda`/HIP call: `bench_runtime._ensure_gs_init` runs `gs.init(backend=gs.gpu, precision="32", logging_level="warning")` BEFORE the provenance block touches `torch.cuda`, otherwise the HSA re-exec repair is skipped and the process dies with `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`.
- Records are the deliverable: a failed run still writes `status="failed"` plus the error block, and `contended` records a busy GPU or a 1-minute load average above 4 (this process' own `rocm-smi` PIDs are filtered out into `host.rocm_pids_all`/`owned_gpu_pids`).
- Timings are host wall clock with `common.sync()` before and after each generation; a record states which regions were synchronized (a no-op on CPU-resident runs), what the unmeasured `residual` is, and reports median plus spread over the three timed generations rather than a single number.
- Traffic counters stay mutually exclusive (bytes are counted only on the primitive transport functions, outer wrappers count calls only) so `residual_s` cannot go negative.
- Files stay well under ~1000 lines, one module per concern, and never write outside this directory.

## Routing Table
- `common.py` -> record schema, host/git/package provenance, statistics and timing wrappers.
- `oracle.py` -> the independent per-step maths the parity check compares against.
- `bench_envs.py` -> the `replay` measurement, the CLI entry point and the shared env/instrumentation helpers.
- `runtime_probes.py` -> timer groups, traffic tags and the monkey-patches that measure one environment.
- `runtime_measure.py` -> `_measure_env` and the per-generation record entries it produces.
- `bench_runtime.py` -> the `throughput` and `residency-parity` measurements (timed generations, per-graph builds, host traffic, NumPy-vs-device parity) and their CLI.
- `results/` -> committed raw records, one file per measurement.
- `../../locomotion/go2_neat_env.py`, `../../drone/hover_neat_env.py` -> the instrumented pilot envs (siblings — read-only, escalate writes to parent).
- `../trainer.py`, `../seeding.py` -> the shared layer the pilots are built on (siblings — read-only).

## Gotchas
- `go2_neat_kernels.run_advance` returns only the aggregate reward, so a Go2 per-term deviation cannot be measured; the record sets `reward.per_term_available = false` with the reason and compares the total instead.
- `hover_legacy_reference` reports reward terms pre-scale while `hover_kernels` reports them `dt`-scaled, so `oracle.hover_graph_terms` applies the one documented rescale (`total` agrees on both sides).
- `Go2NeatEnv.evaluate` draws the per-generation seed perturbation outside its own `wall_time_s`, so `generation_total` and the env's counter differ; `phases.generation_total_scope` says so, and that draw has been observed to cost seconds once (torch geometry warm-up) before settling at ~10 ms, so compare `host_rng` across records instead of assuming it is free.
- The pilot builds dominate a run (`build_s` 5-40 s, quadrants disk-cache dependent), which is why every record reports `phases.build_s` separately from the generations.
- Go2 residency parity is NOT bit-identical by construction: NumPy residency reads state through the entity getters while device residency reads the live solver `rigid_info.qpos`, so `base_quat` differs by 1.19e-07 and horizon-200 returns by 3.05e-05 (`state_read_parity` in the record measures exactly that getter-vs-buffer gap); Hover parity IS bit-identical because both sides read `links.quat`.
- The generation-0 warm-up is untimed for a reason: a Go2 GPU NumPy-side first generation runs at 23.7 steps/s (62.6% residual, torch geometry warm-up) against a steady 122, and the device-side Hover run degrades 168 -> 160 steps/s across generations, so never quote a single generation.
- `go2_neat_env.__init__` hardcodes `RigidOptions(max_collision_pairs=20)`; pop 32 (96 rows) does NOT exceed it, so `--collision-cap` was not needed at any measured size and no `variant="collision_cap_200"` record exists.
- A bare `bench_envs.py` or `bench_runtime.py` prints usage and exits 0 so the `examples/**/*.py` smoke harness does not treat the CLI as a failing example; naming a subcommand parses strictly.

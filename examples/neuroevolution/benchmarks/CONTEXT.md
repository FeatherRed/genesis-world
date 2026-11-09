# examples/neuroevolution/benchmarks

## Intent
The measurement harness of the PyTorch -> EvoX Compiler migration: it answers, per pilot, whether the migrated evaluation environment still computes the pre-migration maths, how fast it runs, and how the trainer behaves end to end.
Every measurement is a committed JSON record under `results/`, reproducible from the exact `command` it stores.
`summarize.py` validates and aggregates those records into `results/summary.json` plus `results/tables.md`, and `RESULTS.md` is the published Stage-3 report that embeds those tables.
Nothing here is imported by the trainer or by the pilot envs: the dependency arrow points one way, from the harness into the siblings.

## API Surface
- `common.py` — record schema and provenance (`new_record`, `write_record`, `git_info`, `packages`, `host_info`, `gpu_info`), `stats`, `Timer`, `wrap_function`/`wrap_method`, `wrap_traffic`/`Counter`, `sync`, `run_guarded`, `setup_paths`, `BASELINE_COMMIT = "ad2f07e5"`.
- `oracle.py` — independent reimplementations: `go2_observations`, `go2_reward` (via `go2_legacy_reference.walking_reward_terms`), `go2_done`, `hover_oracle_step`/`hover_graph_terms`/`hover_reset_obs` (via `hover_legacy_reference`).
- `bench_envs.py` — Phase 1 CLI `replay --task {go2,hover} --pop-size 8 --horizon 100 --master-seed 0 --repeat 0 --process-role only [--generations 2] [--out PATH]` plus the reusable helpers (`_load_task`, `_build_env`, `_instrument`, `_go2_parity`/`_hover_parity`, `_generation_entry`, `_merge_parity`, `_fingerprint`, `StepCapture`, `CallCapture`).
- `runtime_probes.py` — the measurement vocabulary (timer groups, traffic tags, `DEVICE_RESIDENT_INTERMEDIATES`) and the monkey-patches that produce the numbers, each returning its own `restore` callable.
- `runtime_measure.py` — `_measure_env` (build, untimed generation 0, timed generations 1-3, restore), per-step means, traffic totals and the GPU/provenance helpers.
- `bench_runtime.py` — Phase 2 CLI: `throughput --task {go2,hover} --pop-size {8,32} --horizon 200 --residency {numpy,device} --backend numpy|iree --device cpu:0|rocm:0 [--gpu] [--repeat N] [--process-role cold|warm|only] [--collision-cap N] [--out PATH]` and `residency-parity --task … --pop-size 8 --horizon 200 [--backend …] [--device …] [--gpu]`.
- `bench_neat.py` — Phase 3 CLI: `train --task --master-seed --pop-size 8 --generations 3 --horizon 200 --residency {numpy,device} --backend {numpy,iree} --device cpu:0|rocm:0 [--tell-backend {numpy,xla}] [--pjrt-plugin PATH] [--gpu] [--repeat N] [--process-role …] [--out PATH]` and `check-determinism A.json B.json [--out PATH]`.
- `bench_neat_measure.py` — one training run: `measure`, the region probes, `holdout`, `generation_entry`, `fingerprint`/`fingerprint_payload`, `perturbation_for`, `genome_of`, `PopulationCapture`.
- `bench_probe.py` — Phase 4 CLI: `unmatched`, `tell-backend --all`, `torch-cpu`, each writing one record and a printed summary.
- `probe_sources.py` — AST import/function scanning, per-baseline `interface_mismatch`, `torch_scan`, and the extraction of committed boundary blocks with source file and git head.
- `probe_runtime.py` — hard-timeout session subprocesses (`run_capture`), `import_case`, the one `install_attempt`, `run_tell_entry`, and the GPU cleanup snapshot.
- `_tell_worker.py` — the phase-marked tell-graph worker (45 -> 12, pop 4) that proves which backends build and which hang.
- `summarize.py` — Phase 5 CLI: `summarize.py [build|check] [--results DIR] [--out PATH] [--markdown PATH] [--print-tables] [--quiet]`; validates every record (`BASELINE_SHA`, `REQUIRED_KEYS`), extracts per-record entries, aggregates by (kind, task, workload signature), writes `results/summary.json` (`serialize_summary`, compact) and `results/tables.md` (`markdown_document`), and `check` runs its own `_selfcheck`.
- `summary_tables.py` — the presentation half of `summarize.py`: `CHUNK_COLUMNS`, `_entries`, `_table_block`, one builder per family (`_inventory_table` T1, `_replay_tables` T2-T3, `_throughput_tables` T4-T7, `_residency_tables` T8-T11, `_neat_tables` T12-T15, `_probe_tables` T16-T20) and `markdown_document`.
- `RESULTS.md` — the published report: scope/verdict, pins, method, the two comparisons, residency, batches, unmatched baselines, the torch/CPU boundary, limitations, reproduction commands and the record index.
- `results/` — 37 committed records; `results/summary.json` and `results/tables.md` are generated, never hand-edited.

## Record schema and families
Every record carries `schema="neuroevolution-benchmark/1"`, `kind`, `command`, `git`, `packages`, `host`, `gpu`, `contended`, `task`, `seed`, `repeat`, `process_role`, `workload`, `phases`, `timings`, `measurements`, `status`, `error`.
Families: `env_replay` (4: oracle parity), `env_throughput` (8: pop 8/32, cold/warm), `env_residency` (6: NumPy vs device, incl. the `iree@cpu:0` noise pair), `neat_train` (15: 3-generation trainer runs), `neat_determinism` (1), `probe_unmatched_baselines` (1), `probe_tell_backend` (1), `probe_torch_cpu_boundary` (1).
Record naming: `env_<task>_replay_seed<seed>[_r<repeat>].json`, `env_<task>_throughput_<residency>_<physics>_pop<n>[_r<n>].json`, `residency_<task>_<gpu|device_cpu>[_r<n>].json`, `neat_<task>_<variant>[_r<n>].json`, `neat_determinism_go2.json`, `unmatched_baselines.json`, `tell_backend_rocm.json`, `torch_cpu_boundary.json`.

## How to re-run
Put `examples/neuroevolution` on `PYTHONPATH` (bare-name sibling imports), set `MPLCONFIGDIR=$TMPDIR/mpl`, and use the project venv.
`--process-role cold` = first process of a command (fresh interpreter, first HIP context, HSA re-exec); `warm` = the second process; `only` = a single process with no pair. "Cold" is NOT cache-cold.
Every run builds once, runs generation 0 as an untimed warm-up, then times generations 1-3; pass `--repeat 1` with `--process-role warm` for the second process of a pair.
GPU runs need `--gpu` (and `gs.init` before any `torch.cuda` call, which `bench_runtime._ensure_gs_init` handles); `xla` additionally needs `--pjrt-plugin <xla_rocm_plugin.so>`.
After any measurement run: `python examples/neuroevolution/benchmarks/summarize.py --markdown examples/neuroevolution/benchmarks/results/tables.md`, which rewrites `results/summary.json` and `results/tables.md`.

## What is comparable vs not
NumPy vs device residency is comparable because both sides run the same physics backend with the same seed and perturbation; only the location of the graph operands differs.
CPU vs GPU *physics* is a different physics backend and must not be read as a residency win.
Go2 replay parity is aggregate-reward-only (`run_advance` returns the sum); hover parity is oracle-complete (terms, total, masks, reset obs, commands, yaw).
The torch environments were never measured (they cannot be imported), so there is no torch-vs-migrated number anywhere in `results/`.
`residency_hover_device_cpu*` is noise (both sides CPU) and `iree@rocm:0` tells time out rather than running slowly.

## Constraints
Nothing may be edited outside this directory; the pilots, kernels, oracles and tests are read-only siblings instrumented only by runtime monkey-patching that restores itself.
No venv installs and no network: the Phase-4 install attempt must stay an `unavailable` outcome.
Every foreign-process measurement runs in its own session with a hard timeout and a process-group kill; a timeout is recorded as `status="timeout"` with its wall time and last phase, never replaced by a CPU result.
One shared GPU: no parallel measurement, and a run is flagged `contended` when the GPU is busy or the 1-minute load average exceeds `common.CONTENTION_LOADAVG = 4.0`.
Files stay well under ~1000 lines, one module per concern, and every module imports `genesis`/`torch`/`evoxc` lazily or not at all.
`summary.json` is written compact on purpose (`MAX_SUMMARY_BYTES = 200_000`) and `tables.md` is the readable rendering of the same payload.

## Routing Table
- `common.py` -> record schema, provenance, statistics and timing wrappers.
- `oracle.py` -> the independent per-step maths the parity check compares against.
- `bench_envs.py` -> the `replay` measurement and the shared env/instrumentation helpers.
- `runtime_probes.py`, `runtime_measure.py` -> timer groups, traffic tags, patches, and one measured environment.
- `bench_runtime.py` -> `throughput` and `residency-parity`.
- `bench_neat.py`, `bench_neat_measure.py` -> `train`, `check-determinism`, the held-out score and the fingerprint.
- `bench_probe.py`, `probe_sources.py`, `probe_runtime.py`, `_tell_worker.py` -> the Phase-4 probes.
- `summarize.py`, `summary_tables.py` -> validation, aggregation, `summary.json`, `tables.md`.
- `RESULTS.md` -> the published report.
- `results/` -> the raw records and the two generated artifacts.
- `../../locomotion/go2_neat_env.py`, `../../drone/hover_neat_env.py`, `../trainer.py`, `../seeding.py` -> the pilots and the shared layer (siblings — read-only, escalate writes to parent).

## Known Issues / Gotchas
"Cold" is not cache-cold: the quadrants/IREE disk caches were warm, so cold and warm build times differ by well under a second (Go2 26.72 vs 26.53 s, hover 24.00 vs 23.87 s).
Go2 residency is not bit-identical by construction (getter vs live-solver-buffer read: `base_quat` 1.19e-07, `returns` 3.05e-05, `fitness` 6.10e-05); hover is bit-identical because both sides read `links.quat`.
`residency_hover_device_cpu*` ratios (0.951x / 1.154x) are noise, and a GPU process' generation 0 is first-touch dominated (Go2 8.23 s against 1.35 s steady).
`iree@rocm:0` tell compilation does not terminate (killed at 240 s inside `build_tell`), so ROCm tells use `xla` with a PJRT plugin, which costs an ~11.8 s build and changes the fingerprint.
`import genesis` pulls `torch` into the process (`torch 2.13.0+rocm7.1`) even though `examples/neuroevolution/*` imports none.
`tests/test_examples.py` collects every `examples/**/*.py`; the CLIs print usage and exit 0 when invoked bare, and the shared helper modules (`common.py`, `oracle.py`, `summary_tables.py`, `runtime_*.py`, `probe_*.py`, `bench_*_measure.py`, `_tell_worker.py`) should be added to `IGNORE_SCRIPT_PATTERNS` by the tests node.
Three records point at other workers' absolute paths in `command` (the harness ran in per-task worktrees); the commands are otherwise identical to the documented invocations.

## Test Strategy
`summarize.py check` is the node's executable check: it asserts the record count equals the number of sources, every group is non-empty, the aggregate counts match, the payload round-trips and is deterministic, it is under `MAX_SUMMARY_BYTES` (194 KB), `tables.md` has >= 18 tables each with a `Sources:` footnote and cites every record, and it pins three numbers by hand-checked value (Go2 replay observation 4.76837158203e-07, Go2 residency speedup 1.33579939608, Go2 seed-0 final best fitness -6.03994607925415).
`bench_neat.py check-determinism` is the reproducibility test of the trainer: two processes of one command must agree exactly on seed table, seeds, fitness, held-out fitness and fingerprint.
`results/tables.md` is regenerated by `summarize.py --markdown` and diffs cleanly, so a stale aggregate is visible in review.
No `tests/` file lives here; the natural follow-up owned by the tests node is a `tests/test_neuroevolution_bench_*.py` covering `summarize.py` and the `IGNORE_SCRIPT_PATTERNS` entries.

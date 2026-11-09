# Stage-3: PyTorch → EvoX Compiler migration of two neuroevolution pilots

Every table below is copied verbatim from `results/tables.md`, which is **generated from `results/summary.json` by `summarize.py`**; no number here is hand-transcribed from a record, and every row names the record file it came from.
The raw records are the deliverable: `results/*.json`, 37 benchmark records (38 files in `results/`, the 38th being `results/CONTEXT.md`).

## 1. Scope and verdict

Two pilots were migrated and measured end to end:

| pilot | environment | observation → action | episode length | trainer |
|---|---|---|---|---|
| Go2 walking | `examples/locomotion/go2_neat_env.py` | 45 → 12 | 20 s (1000 steps, `dt=0.02`) | `trainer.NeuroevolutionTrainer` |
| Drone hovering | `examples/drone/hover_neat_env.py` | 17 → 4 | 15 s (1500 steps, `dt=0.01`) | `trainer.NeuroevolutionTrainer` |

What was measured: (1) per-step numerical replay of the migrated evaluation environment against an independent pre-migration oracle, (2) steps/s, env-steps/s and a per-step time split at pop 8 and pop 32, (3) full 3-generation TensorNEAT training runs with per-generation curves and a held-out score, (4) NumPy vs device residency of the graph operands (parity, host traffic, per-graph compile cost). 37 records, all `status=ok`, none `contended`.

**The one-paragraph answer.** The migration is numerically sound and, at these batch sizes, a qualified win. The migrated environment reproduces the pre-migration arithmetic to float32 noise on both pilots (observation max abs diff 4.77e-07 on Go2 and exactly 0 on hover; no termination, reset, command or reward-total mismatch in 100 compared steps per pilot), and it runs the whole train loop with no episode-level divergence. Moving the graph operands from NumPy arrays to device-resident buffers raises the Go2 rollout from 120.73 to 161.27 steps/s (1.336x) and hover from 105.33 to 167.57 steps/s (1.591x) on the GPU while cutting the per-step host traffic from 95.5 KB to 396 B (Go2) and 85.5 KB to 1.1 KB (hover) — but end to end the accelerator does **not** pay off at 24 rows: the best steady generation is still the CPU NumPy-residency run (Go2 1.26 s/generation, hover 0.92 s/generation against 1.33 s and 1.24 s on the GPU), because physics stays on the CPU for the device runs and the compile bill (4.03-4.49 s of graph builds, 23.9-27.4 s of process build) is paid up front. What cannot be claimed: a comparison against the torch environments, which never imported.

## 2. Pins and environment

Baseline: `ad2f07e5` = `ad2f07e5aec1ba931f5a6704253f70e6fa6370b2`. Every one of the 37 records stores `git.baseline_env_diff_empty=true` with an empty `git.baseline_env_diff`, i.e. the two torch pilots (`go2_env.py`, `hover_env.py`) and their kernels were byte-identical to that commit in every run; `summarize.py` validates the field and reports `baseline_env_diff_empty_all: true`.

The harness itself evolved across commits, so the record families were produced at different heads (`summary.json` → `validation.heads`):

| family | head | note |
|---|---|---|
| `env_replay` (4 records) | `7c567d49519f5e155ed69413083659a0254ae456` | Phase 1 |
| `env_throughput`, `env_residency` (14) | `cdda51acf1ba470fa8eded55b832b0ddadbd6ecf` | Phase 2 |
| `neat_train`, `neat_determinism` (16) | `223049e1e5de14d62511adcc260b810a6186d7ea` | Phase 3 |
| `probe_tell_backend` (1) | `250f77fcd575b30aeb06db6248984f1052ffc78c` | Phase 4 |
| `probe_torch_cpu_boundary` (1) | `42230b382ddbe614e4e50e362c3852623e039113` | Phase 4 |
| `probe_unmatched_baselines` (1) | `1f2994a248a16ba30f163603b73e5d4c60d56f76` | Phase 4 |

Package set recorded in every record: `genesis-world 1.4.6`, `quadrants 1.4.0`, `evoxc 0.1.0`, `tensorneat 0.2.0`, `iree-base-compiler 3.13.0`, `iree-base-runtime 3.13.0`, `numpy 2.5.4`, `torch 2.13.0+rocm7.1`.
Host: one `AMD Instinct MI300X VF` (gfx942, SKU M3000100, 1 device), `torch.version.hip 7.1.52802`, `rocm-smi` "Driver version: 6.16.6" (every record also carries its low-power-state warning), 20 CPUs, Python 3.12.3, Linux.
1-minute load average over the 37 records: 0.76-2.16 (`summary.json` → `environment.loadavg_1m`); `contended=false` and `host.gpu_busy=false` in all 37, so no run is flagged for contention (the threshold is `common.CONTENTION_LOADAVG = 4.0`).

## 3. Method

**Cold vs warm process.** `--process-role cold` is the first process of a run: fresh interpreter, first HIP context, the HSA re-exec path in `bench_runtime._ensure_gs_init` (Genesis is initialised before anything calls `torch.cuda`, otherwise the run dies with `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`). `--process-role warm` is the second process of the same command. **"Cold" is not cache-cold**: the quadrants/IREE disk caches were already populated from earlier work, and the measured build times are identical — Go2 GPU trainer build 26.72 s (cold, `neat_go2_gpu_device_numpy_tell`) vs 26.53 s (warm, `_r1`), hover 24.00 s vs 23.87 s (`neat_hover_gpu_device` / `_r1`), Go2 CPU 6.58 s vs 6.45 s (`neat_go2_seed0` / `_r1`). The pair therefore bounds process-level variance only; it does not measure a cold compile cache.

**Synchronisation.** `common.sync()` brackets every timed region (`torch.cuda.synchronize()` when a CUDA/HIP device is in play, a no-op on CPU); each record states its scope in `workload.sync` ("common.sync() before and after every generation" / "… each timed region").

**Timing protocol.** Every run builds once, runs **generation 0 as an untimed warm-up**, then times generations 1-3 (3 timed generations per process, 2 processes per configuration). Headline medians exclude generation 0 everywhere (`generations[*].warmup`), and the records report median plus spread rather than a single number. The `env_replay` family is the exception: it is a 2-generation parity run and gen 0 is reported separately in T3.

**Timed regions** (verbatim from `workload.timer_groups`): `physics` = `genesis.Scene.step` (class-level patch: rigid solver + state/material update); `graph_policy` = one `run_policy`; `graph_advance` = one `run_advance` (clip → observation → reward → termination → bookkeeping); `graph_obs`, `graph_controls`, `graph_fitness`, `graph_all` = one `run_*` call each; `state_read` = `env.read_state`, the device→host boundary (torch reads, rpy, error mask); `write_targets` / `write_reset` = the actuator write and the reset-write share; `host_rng` = the host draws (`perturbed_init_qpos` on Go2, `_draw_commands` on hover); `download` / `stage_upload` / `sync` = one graph-output readback, one operand staging, one device sync. `residual` is the unmeasured remainder of the generation wall clock.

**Held-out rule.** `bench_neat_measure.holdout` re-evaluates the generation-0 population and the last-generation population on `seeding.generation_seed_table(master_seed + 1000, 1)` — one triple of 3 episode seeds, asserted disjoint from every training seed (`holdout.disjoint_from_training=true`). `median delta` is the median over policies of (trained − initial) on that triple; `best_policy.train_fitness` is that policy's fitness on the training seeds, `heldout_fitness` its fitness on the unseen triple, and `delta` its own trained−initial improvement on the unseen triple.

**Determinism check.** `bench_neat.py check-determinism results/neat_go2_seed0.json results/neat_go2_seed0_r1.json` compares two processes of the same command exactly: seed table, per-generation seeds, per-generation fitness, held-out fitness of both sides, and a fingerprint over all of them (`neat_determinism_go2.json`, T15). It matches bit-for-bit — fingerprint `cd6f6b8e9174…` on both sides, max abs diff 0.0 in every block, 0 mismatched generations.

## 4. Comparison 1: environment replay against the pre-migration oracle

Oracle provenance (`oracle.py`): for hover, `oracle.hover_oracle_step` / `hover_reset_obs` call `hover_legacy_reference`, a full transcription of `hover_env.py`'s step and reset maths; for Go2, the reward comes from `go2_legacy_reference.walking_reward_terms`, the observation from a NumPy transcription of `go2_env._update_observation` (`oracle.go2_observations`), and termination from `genesis.utils.geom.quat_to_xyz` plus the same tilt thresholds (`oracle.go2_done`). The oracle consumes only host arrays, so it is independent of the compiled graphs.

The torch environments themselves could not be imported (see §9), so this is a **maths-to-maths** comparison, not a torch-to-EvoxComp comparison: 100 steps per run, pop 8 (24 rows), zero perturbation, `stop_when_done=false`.

### T2 Environment replay parity against the pre-migration oracle

Tolerances (record `workload.tolerances`): observation 1e-06, reward term and total 1e-05, step return 1e-04, accumulated return 1e-03, masks exact, hover yaw 1e-03 deg. Go2 reward is aggregate-only because `run_advance` returns the sum, not the six terms.

| metric | Go2 (pop 8, horizon 100) | hover (pop 8, horizon 100) |
|---|---|---|
| steps compared | 100 | 100 |
| observation max abs diff | 4.768372e-07 | 0 |
| observation mismatches | 0 | 0 |
| reward per-term max abs diff | not measured (aggregate only) | n/a |
| reward total max abs diff | 0.000008 | 2.980232e-08 |
| reward total mismatches | 0 | 0 |
| step return max abs diff | 0.000009 | 4.842877e-08 |
| accumulated return max abs diff | 0.000029 | 1.544104e-07 |
| fitness (from oracle) max abs diff | 0.000061 | 5.960464e-08 |
| done mismatch count | 0 | 0 |
| tilted / crash mismatch count | 0 | 0 |
| timeout mismatch count | 0 | 0 |
| episode-length mismatch count | 0 | n/a (mask-based) |
| reset observation max abs diff | n/a (Go2 resets from a host qpos) | 0 |
| command max abs diff | n/a (Go2 samples commands inside the graph) | 0 |
| yaw max abs diff (deg) | n/a | 0.000198364 |
| fingerprint (generation 0) | 3259cd4edf5a | b7b87d048098 |

Sources: `env_go2_replay_seed0.json`, `env_go2_replay_seed0_r1.json`, `env_hover_replay_seed0.json`, `env_hover_replay_seed0_r1.json`


The Go2 per-term comparison is impossible by construction, not by omission: `go2_neat_kernels.run_advance` returns only the aggregate reward, so `env_go2_replay_seed0.json` sets `reward.per_term_available=false` with that reason and compares the total (8e-06) instead. Hover's oracle does expose the terms; the record compares them at the 1e-05 tolerance and the total at 2.98e-08.

Repeatability: the `_r1` records of both pilots reproduce the same parity figures and differ only in the state fingerprint under `measurements.fingerprint.per_generation`; both generations of both pilots are in T3 (`env_go2_replay_seed0.json`, `env_hover_replay_seed0.json`).

What installing `tensordict` would add: `go2_env` / `hover_env` would import (they fail at `from tensordict import TensorDict`), which would allow a live torch-vs-migrated replay and a trainer-drivable adapter. Until then the Go2 reward comparison stays aggregate-only and the hover comparison is oracle-only.

## 5. Environment throughput and time split (CPU, NumPy residency)

### T4 CPU NumPy-residency environment throughput (timed generations 1-3, horizon 200)

Generation 0 is excluded everywhere (first-step JIT, empty caches). `residual` is the unmeasured share of the generation wall clock.

| task | pop | rows | process | repeat | steps/s (median ± spread) | env-steps/s | generation wall s | first gen s | residual | record |
|---|---|---|---|---|---|---|---|---|---|---|
| go2 | 32 | 96 | cold | 0 | 87.638 ± 2.192 | 8413.230 ± 210.432 | n/a | 2.545 | 0.64% | env_go2_throughput_numpy_cpu_pop32 |
| go2 | 32 | 96 | warm | 1 | 85.312 ± 1.112 | 8189.995 ± 106.751 | n/a | 2.619 | 0.62% | env_go2_throughput_numpy_cpu_pop32_r1 |
| go2 | 8 | 24 | cold | 0 | 177.785 ± 1.819 | 4266.832 ± 43.667 | n/a | 1.387 | 1.19% | env_go2_throughput_numpy_cpu_pop8 |
| go2 | 8 | 24 | warm | 1 | 176.062 ± 1.437 | 4225.481 ± 34.485 | n/a | 1.384 | 1.20% | env_go2_throughput_numpy_cpu_pop8_r1 |
| hover | 32 | 96 | cold | 0 | 151.953 ± 1.349 | 14587.456 ± 129.531 | n/a | 1.393 | 2.42% | env_hover_throughput_numpy_cpu_pop32 |
| hover | 32 | 96 | warm | 1 | 154.621 ± 1.100 | 14843.653 ± 105.620 | n/a | 1.358 | 2.41% | env_hover_throughput_numpy_cpu_pop32_r1 |
| hover | 8 | 24 | cold | 0 | 234.457 ± 4.308 | 5626.976 ± 103.404 | n/a | 0.893 | 2.11% | env_hover_throughput_numpy_cpu_pop8 |
| hover | 8 | 24 | warm | 1 | 237.065 ± 0.532 | 5689.569 ± 12.763 | n/a | 1.008 | 2.16% | env_hover_throughput_numpy_cpu_pop8_r1 |

Sources: `env_go2_throughput_numpy_cpu_pop32.json`, `env_go2_throughput_numpy_cpu_pop32_r1.json`, `env_go2_throughput_numpy_cpu_pop8.json`, `env_go2_throughput_numpy_cpu_pop8_r1.json`, `env_hover_throughput_numpy_cpu_pop32.json`, `env_hover_throughput_numpy_cpu_pop32_r1.json`, `env_hover_throughput_numpy_cpu_pop8.json`, `env_hover_throughput_numpy_cpu_pop8_r1.json`


Go2 at pop 8 runs 177.785 ± 1.819 steps/s (first process, `env_go2_throughput_numpy_cpu_pop8.json`) and 176.062 ± 1.437 (second, `_r1`); at pop 32 it drops to 87.638 ± 2.192 / 85.312 ± 1.112 while env-steps/s rises from ~4.25 K to ~8.2-8.4 K. Hover goes 234.457 ± 4.308 → 151.953 ± 1.349 steps/s between pop 8 and pop 32 with env-steps/s 5.6 K → 14.6-14.8 K. Individual timed generations are in `summary.json` → `families.env_throughput.<task>.<signature>.entries[*].generations`; every generation is also in `env_*_throughput_*.json`.

Per-step split (T5 in `results/tables.md`, ms/step over the three timed generations) at pop 8: Go2 spends 1730.20 ms physics, 2448.72 ms `graph_policy`, 798.07 ms `graph_advance`, 452.78 ms `state_read`, 102.29 ms actuator write, 17.42 ms reset write, 0.21 ms host RNG, residual 0.067 ms (`env_go2_throughput_numpy_cpu_pop8.json`). Hover at pop 8 spends 679.17 ms physics, 1668.18 ms `graph_policy`, 1020.09 ms `graph_advance`, 204.34 ms `state_read`, 429.83 ms actuator write, 4.97 ms reset write, 115.58 ms host RNG, residual 0.090 ms (`env_hover_throughput_numpy_cpu_pop8.json`). T6 gives the same as shares: `graph_policy` is the largest single component on both pilots (43.6% Go2, 39.1% hover at pop 8), physics 30.8% / 15.9%, and the measured groups leave only 1.19% / 2.11% unexplained.

Not separable: the Go2 reward terms inside `run_advance` (aggregate only), and — on CPU — the graph-operand staging inside each `run_*` call, which is zero-copy NumPy and therefore folded into the graph timer. Hover's `_draw_commands` host RNG *is* separable (`host_rng`, 2.71-2.75% at pop 8, 6.57-6.62% at pop 32).

## 6. Comparison 2: the TensorNEAT trainer

Three generations, pop 8 (24 rows), horizon 200, CPU NumPy residency; seeds 0, 1, 2. Per-generation curves are T12 in `results/tables.md` (45 rows, every record).

### T13 Held-out score: generation-0 population vs last-generation population on one unseen seed triple

Held-out seeds are `seeding.generation_seed_table(master_seed + 1000, 1)`, never used by a training generation. `median delta` is the median over policies of (trained − initial) on that triple; a negative value means the trained population scored worse. `best policy train`/`held-out` are that policy's fitness on the training seeds and on the unseen triple, and the last column is its held-out improvement, not a train-vs-held-out gap.

| task | seed | initial median | trained median | median delta | best policy train | best policy held-out | best policy held-out improvement | separates | record |
|---|---|---|---|---|---|---|---|---|---|
| go2 | 0 | -47.96391 | -10.93208 | 3.08654 | -6.03995 | -7.36767 | -0.00800 | True | neat_go2_gpu_device_numpy_tell |
| go2 | 0 | -47.96391 | -10.93208 | 3.08654 | -6.03995 | -7.36767 | -0.00800 | True | neat_go2_gpu_device_numpy_tell_r1 |
| go2 | 0 | -47.96391 | -10.93208 | 3.08654 | -6.03995 | -7.36767 | -0.00800 | True | neat_go2_gpu_device_xla_tell |
| go2 | 0 | -47.96391 | -10.93208 | 3.08654 | -6.03995 | -7.36767 | -0.00800 | True | neat_go2_gpu_device_xla_tell_r1 |
| go2 | 0 | -47.96391 | -10.93208 | 3.08654 | -6.03995 | -7.36767 | -0.00800 | True | neat_go2_gpu_numpy_residency |
| go2 | 0 | -47.96391 | -10.93208 | 3.08654 | -6.03995 | -7.36767 | -0.00800 | True | neat_go2_gpu_numpy_residency_r1 |
| go2 | 0 | -47.96391 | -10.93208 | 3.08654 | -6.03995 | -7.36767 | -0.00800 | True | neat_go2_seed0 |
| go2 | 0 | -47.96391 | -10.93208 | 3.08654 | -6.03995 | -7.36767 | -0.00800 | True | neat_go2_seed0_r1 |
| go2 | 1 | -68.50865 | -11.73538 | 55.78234 | -5.06396 | -4.19744 | 4.27967 | True | neat_go2_seed1 |
| go2 | 2 | -98.16809 | -15.72710 | 83.33858 | -8.19573 | -16.18274 | 16.94943 | True | neat_go2_seed2 |
| hover | 0 | -0.37818 | -0.18983 | 0.15596 | -0.18867 | -0.18782 | 0.07433 | True | neat_hover_gpu_device |
| hover | 0 | -0.37818 | -0.18983 | 0.15596 | -0.18867 | -0.18782 | 0.07433 | True | neat_hover_gpu_device_r1 |
| hover | 0 | -0.37818 | -0.18983 | 0.15596 | -0.18867 | -0.18782 | 0.07433 | True | neat_hover_seed0 |
| hover | 1 | -0.42423 | -0.17063 | 0.24797 | -0.21210 | -0.17084 | 0.82559 | True | neat_hover_seed1 |
| hover | 2 | -0.17666 | -0.18184 | -0.00721 | -0.18740 | -0.17257 | 0.08430 | True | neat_hover_seed2 |

Sources: `neat_go2_gpu_device_numpy_tell.json`, `neat_go2_gpu_device_numpy_tell_r1.json`, `neat_go2_gpu_device_xla_tell.json`, `neat_go2_gpu_device_xla_tell_r1.json`, `neat_go2_gpu_numpy_residency.json`, `neat_go2_gpu_numpy_residency_r1.json`, `neat_go2_seed0.json`, `neat_go2_seed0_r1.json`, `neat_go2_seed1.json`, `neat_go2_seed2.json`, `neat_hover_gpu_device.json`, `neat_hover_gpu_device_r1.json`, `neat_hover_seed0.json`, `neat_hover_seed1.json`, `neat_hover_seed2.json`


Learning curves against both axes are in T12 (`cumulative env steps` = 4800/9600/14400 for every run; `cumulative loop s` from 0.89 s to 12.6 s depending on configuration). CPU Go2 totals (`neat_go2_seed0.json`): `generation_s` 3.8744 s for 3 generations, made of `evaluate_s` 1.3389 + 1.2333 + 1.2800 s, `tell_s` 0.00912 + 0.00613 + 0.00616 s, `loop_s` 3.8745 s, `best_fitness` −7.37847 → −6.03995 (`neat_go2_seed0.json`). Seed 1 improves −7.06997 → −5.06396 and seed 2 −24.60638 → −8.19573 (`neat_go2_seed1.json`, `neat_go2_seed2.json`), so 3 generations already separate a good from a bad initial population.

Held-out: Go2 seed 0 medians −47.96391 → −10.93208 (median delta +3.08654); the best policy scores −6.03995 on the training seeds and −7.36767 on the unseen triple, i.e. 1.32772 worse on held-out (`neat_go2_seed0.json`). Seed 1's best policy is 0.86652 *better* held-out (−5.06396 → −4.19744) and seed 2's is 7.98701 worse (−8.19573 → −16.18274). Hover: seed 0 −0.37818 → −0.18983 (delta +0.15596; best policy −0.18867 train vs −0.18782 held-out), seed 1 −0.42423 → −0.17063 (+0.24797; −0.21210 vs −0.17084), seed 2 −0.17666 → −0.18184 (−0.00721, i.e. the trained population scored slightly worse on the unseen triple; best policy −0.18740 vs −0.17257). At three generations the held-out gap is a handful of percent on both pilots and changes sign per seed — one triple is not enough to claim generalisation either way.

Determinism: bit-identical across processes (T15, `neat_determinism_go2.json`), including both held-out evaluations.

**GPU device residency, end to end** (`neat_go2_gpu_device_numpy_tell*`, `neat_hover_gpu_device*`, T14): Go2 steady generation 1.3467 / 1.3284 s, 147.53 / 149.91 steps/s, `generation_s` total 10.9272 / 10.6878 s; hover 1.2400 / 1.2585 s, 161.96 / 160.03 steps/s, total 4.4872 / 4.3872 s. The same command with NumPy residency on the same GPU process (`neat_go2_gpu_numpy_residency*`) runs 1.7163 / 1.7179 s per generation and 115.67 / 115.96 steps/s — so device residency is worth 1.33x *inside* the GPU process too, but the GPU process as a whole is still slower than the CPU one (1.27 s/generation, 156.29 steps/s, `neat_go2_seed0.json`).

**xla vs numpy tell** (`neat_go2_gpu_device_xla_tell*`): the tell build costs 11.791 / 11.794 s against 0.396 / 0.379 s for numpy, and the first generation 9.928 / 9.889 s against 8.234 / 8.031 s; steady-state tell time is the same (~6 ms) and the steady generation is 1.3543 / 1.3424 s against 1.3467 / 1.3284 s. The fingerprint differs (`0d9c7a433bc6…` vs `6d71d28b8b1e…`) while the recorded fitness and held-out medians agree to the printed precision in T13, so the two tells are comparable in trend, not bit-identical.

**The trainer comparison against the torch environment is unmatched**, and no adapter exists. The exact interface mismatch (`unmatched_baselines.json` → `measurements.interface_mismatch`): both envs fail at import (`ModuleNotFoundError: tensordict`); the constructor needs `env_cfg/obs_cfg/reward_cfg/command_cfg` from `go2_train.get_cfgs()` / `hover_train.get_cfgs()` (line 64), which are themselves unimportable because they enforce `rsl-rl-lib>=5.0.0`; `Go2Env.step` (line 155) returns `(TensorDict, rew_buf, reset_buf, extras)` with no episode-return array; auto-reset is internal (`Go2Env._reset_idx`, line 208 / `HoverEnv.reset_idx`, line 214) and takes no episode seed, so the row/seed contract cannot be honoured; and neither has `evaluate(pop_arrays, generation)` (`has_evaluate: false`). A thin adapter was therefore not attempted beyond the documented failure — it is blocked at import, not at the boundary.

## 7. NumPy residency (initial port) vs device residency (optimised)

### T8 Residency comparison: NumPy residency vs device residency (same physics backend, same seed)

`residency_hover_device_cpu*` is the `iree@cpu:0` device run: both sides are CPU, so its ratio is noise-dominated and must not be quoted as a win.

| task | record | numpy steps/s | device steps/s | speedup | numpy env-steps/s | device env-steps/s | numpy B/step | device B/step | numpy calls/step | device calls/step | numpy sync/step | device sync/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| go2 | residency_go2_gpu | 120.73 | 161.27 | 1.336x | 2897.6 | 3870.6 | 95553.2 | 395.7 | 4.085 | 4.065 | 0 | 2.005 |
| go2 | residency_go2_gpu_r1 | 119.92 | 159.41 | 1.329x | 2878.2 | 3825.8 | 95553.2 | 395.7 | 4.085 | 4.065 | 0 | 2.005 |
| hover | residency_hover_device_cpu | 219.80 | 209.03 | 0.951x | 5275.3 | 5016.6 | 85450.6 | 1109.5 | 5.015 | 8.115 | 0 | 1.000 |
| hover | residency_hover_device_cpu_r1 | 211.06 | 243.62 | 1.154x | 5065.5 | 5846.8 | 85450.6 | 1109.5 | 5.015 | 8.115 | 0 | 1.000 |
| hover | residency_hover_gpu | 105.33 | 167.57 | 1.591x | 2527.9 | 4021.8 | 85450.6 | 1109.5 | 5.015 | 8.115 | 0 | 1.000 |
| hover | residency_hover_gpu_r1 | 103.69 | 168.11 | 1.621x | 2488.6 | 4034.7 | 85450.6 | 1109.5 | 5.015 | 8.115 | 0 | 1.000 |

Sources: `residency_go2_gpu.json`, `residency_go2_gpu_r1.json`, `residency_hover_device_cpu.json`, `residency_hover_device_cpu_r1.json`, `residency_hover_gpu.json`, `residency_hover_gpu_r1.json`


Headline speedups at pop 8 on the GPU: **Go2 1.336x / 1.329x**, **hover 1.591x / 1.621x**. The third residency record pair (`residency_hover_device_cpu*`) runs the device path on `iree@cpu:0`: 0.951x and 1.154x with spreads larger than the ratio itself — both sides are CPU, so it is **noise-dominated and must not be quoted as a win**.

### T9 Residency parity (generation 0, identical seed and perturbation)

Every column is a max abs diff. `base_quat` is `state_read_parity`: NumPy residency reads the entity getter, device residency the live solver buffer, which is the only arithmetic difference between the two paths.

| task | record | seeds | init_qpos | returns | episode_returns | fitness | commands | fingerprint equal | base_quat (getter vs buffer) | all pass |
|---|---|---|---|---|---|---|---|---|---|---|
| go2 | residency_go2_gpu | 0 | 0 | 0.000031 | 0.000031 | 0.000061 | n/a | False | 1.192093e-07 | False |
| go2 | residency_go2_gpu_r1 | 0 | 0 | 0.000031 | 0.000031 | 0.000061 | n/a | False | 1.192093e-07 | False |
| hover | residency_hover_device_cpu | 0 | n/a | 0 | 0 | 0 | 0 | True | n/a | True |
| hover | residency_hover_device_cpu_r1 | 0 | n/a | 0 | 0 | 0 | 0 | True | n/a | True |
| hover | residency_hover_gpu | 0 | n/a | 0 | 0 | 0 | 0 | True | n/a | True |
| hover | residency_hover_gpu_r1 | 0 | n/a | 0 | 0 | 0 | 0 | True | n/a | True |

Sources: `residency_go2_gpu.json`, `residency_go2_gpu_r1.json`, `residency_hover_device_cpu.json`, `residency_hover_device_cpu_r1.json`, `residency_hover_gpu.json`, `residency_hover_gpu_r1.json`


Go2 residency is **not** bit-identical, and the record explains why: NumPy residency reads the state through the entity getters, device residency reads the live solver buffer, so the two feed the graphs a last-bit different orientation — `sides.numpy.state_read_parity.base_quat` max abs diff 1.19209289551e-07 — which propagates to `returns` 3.0517578125e-05, `episode_returns` 3.0517578125e-05 and `fitness` 6.103515625e-05, while `seeds` and `init_qpos` are exactly 0 (`residency_go2_gpu.json`, `_r1`). That is a read-path difference, not a semantic change. Hover is bit-identical on every array because both sides read `links.quat` (T9, `fp equal = True`, `all pass = True`).

Compile cost and warm-up (T10, T11 in `results/tables.md`): Go2 side build 26.82 s (NumPy) vs 6.50 s (device); graph-build total 4.0569 vs 4.0267 s, split over advance 1.0676 / 1.0726, policy 2.0910 / 2.1195, obs 0.6021 / 0.5567, fitness 0.2961 / 0.2780; first generation 8.543 vs 1.327 s; steady generation 1.6566 vs 1.2401 s (`residency_go2_gpu.json`). Hover on the GPU: build 23.73 vs 4.51 s, first generation 2.519 vs 1.167 s, steady 1.8988 vs 1.1935 s (`residency_hover_gpu.json`). The device side builds cheaper only because the NumPy side additionally pays the torch geometry/material warm-up on its first generation.

Compile-once proof: `cache.hits/misses` 16/4 (Go2, four graphs × four generations + four builds) and 40/5 (hover, five graphs), with `cache.per_generation` showing `hits=4, misses=0` for every generation after the build and `checks.pass=true` on both sides ("one state read per step", "one actuator write per step", observed 600 of 600).

Host traffic per step (T8, from the same-physics comparison block): Go2 95,553.2 B → 395.7 B, calls 4.085 → 4.065, syncs 0 → 2.005; hover 85,450.6 B → 1,109.5 B, calls 5.015 → 8.115, syncs 0 → 1.000. Bytes are an estimate (`byte_estimate`: tensor numel × element size, a graph call counting operands and result), not a PCIe counter. Note the device path pays 2 syncs/step on Go2 and 1 on hover, which the numpy path does not need.

What stays on the device (record `sides.device.device_resident_intermediates.items`): on Go2, observations, raw and clipped actions, `last_actions`, returns/alive/episode_length, PD joint targets, the per-generation init-qpos tile and reset rows, and the zero-valued reset operands; on hover, the staged base-state buffers, observation/clipped actions/propeller RPM, returns/alive/episode_length/end_code/commands, `last_actions`, `last_base_pos` and the four genome leaves (staged once per generation). What still costs a host touch per step is listed in `host_touches_per_step`: the `(rows,)` done mask plus the solver-error/base-rpy rows on a step that ended (Go2), and `scene.get_error_envs_mask()` plus the two host command draws (hover).

## 8. Batches

pop 8 (24 rows) vs pop 32 (96 rows) is T7 in `results/tables.md`. Throughput does not scale linearly: Go2 loses 49.4% of its steps/s for 4x the rows (177.78 → 87.64, `env_go2_throughput_numpy_cpu_pop8.json` / `env_go2_throughput_numpy_cpu_pop32.json`) while env-steps/s grows 1.97x (4266.8 → 8413.2); hover loses 35.2% of steps/s (234.46 → 151.95) and gains 2.59x env-steps/s (5627.0 → 14587.5). The generation wall time rises from 1.1250 to 2.2821 s (Go2) and from 0.8530 to 1.3162 s (hover).

The Go2 environment hardcodes `RigidOptions(max_collision_pairs=20)`; 96 rows at horizon 200 ran to `status=ok` in both processes with no exception (`env_go2_throughput_numpy_cpu_pop32*.json`), so the constraint did not break at the largest measured batch. For larger batches `bench_runtime.py throughput --collision-cap N` overrides that option at runtime (a patch of `gs.options.RigidOptions`, never a source edit); no `variant="collision_cap_200"` record exists because it was never needed.

## 9. Unmatched baselines and unavailable backends

### T16 Unmatched baselines: every import, adapter and install attempt

Each case ran as its own subprocess with a hard timeout. Neither torch environment could be imported, so no torch-vs-migrated environment comparison exists.

| case kind | case | verdict | elapsed s | exception | returncode |
|---|---|---|---|---|---|
| import | go2_env | failed | 7.326 | ModuleNotFoundError: No module named 'tensordict' | 1 |
| import | go2_train | failed | 0.064 | ImportError: Please install 'rsl-rl-lib>=5.0.0'. | 1 |
| import | hover_env | failed | 7.580 | ModuleNotFoundError: No module named 'tensordict' | 1 |
| import | hover_train | failed | 0.064 | ImportError: Please install 'rsl-rl-lib>=5.0.0'. | 1 |
| import | rsl_rl | failed | 0.032 | ModuleNotFoundError: No module named 'rsl_rl' | 1 |
| import | tensordict | failed | 0.032 | ModuleNotFoundError: No module named 'tensordict' | 1 |
| thin adapter | go2 | failed | 7.330 | ModuleNotFoundError: No module named 'tensordict' | 1 |
| thin adapter | hover | failed | 7.331 | ModuleNotFoundError: No module named 'tensordict' | 1 |
| offline install | pip --no-deps --target $TMPDIR | unavailable | 0.032 | No module named pip; uv not on PATH | 1 |
| PPO baseline | rsl-rl PPO | unmatched | n/a | ImportError: Please install 'rsl-rl-lib>=5.0.0'. | n/a |

Sources: `unmatched_baselines.json`


`unmatched_baselines.json` → `measurements.verdict`: `measured_environments: []`, `unmatched_environments: ["go2","hover"]`, `unmatched_algorithms: ["ppo"]`. The rule is explicit — an environment counts as measured only when the trainer drove it through `env.evaluate(pop_arrays, generation)`.

The offline install attempt could not start: `install_attempt.status="unavailable"`, `pip` module absent (`No module named pip`, returncode 1, 0.032 s) and `uv` not on PATH (`uv.available=false`), with `target_is_outside_venv=true` and nothing written outside `$TMPDIR`. That is why the torch baselines stayed unmatched rather than being installed on the spot.

PPO is an unmatched *algorithm* baseline: `rsl_rl` is absent (`ModuleNotFoundError`), `go2_train`/`hover_train` refuse to import without `rsl-rl-lib>=5.0.0`, and `required` lists `rsl-rl-lib>=5.0.0`, `tensordict`, `tensorboard`; no PPO rollout was run (`measurements.ppo.status="unmatched"`).

### T17 tell-graph backend matrix on ROCm (`_tell_worker.py`, pop 4, genome 45 -> 12)

`iree@rocm:0` is killed at its hard timeout inside `build_tell` (`returncode -9`); the phases it reached are in the raw record.

| backend@device | status | elapsed s | last phase | build s | run s | total s | timeout s | PJRT plugin |
|---|---|---|---|---|---|---|---|---|
| iree@rocm:0 | timeout | 240.016 | build_tell_start | n/a | n/a | n/a | 240.0 | n/a |
| numpy@cpu:0 | ok | 0.719 | run_tell_done | 0.271 | 0.009 | 0.590 | n/a | n/a |
| xla@rocm:0 | ok | 11.691 | run_tell_done | 4.724 | 6.161 | 11.269 | n/a | /tmp/xla_rocm_whl/x/jax_plugins/xla_rocm7/xla_rocm_plugin.so |

Sources: `tell_backend_rocm.json`


The IREE/ROCm tell hangs exactly where the migration notes said it would: `iree@rocm:0` is killed by the harness at 240.0 s inside `build_tell` (`returncode -9`, `last_phase="build_tell_start"`, phases reached up to `pipeline_ready`), never substituted by a CPU result. `numpy@cpu:0` builds in 0.271 s and runs in 0.009 s (0.719 s end to end) as the control; `xla@rocm:0` needs the PJRT plugin (`--pjrt-plugin /tmp/xla_rocm_whl/x/jax_plugins/xla_rocm7/xla_rocm_plugin.so`), builds in 4.724 s and runs in 6.161 s (11.691 s end to end), including the first-run XLA compile. `verdict` records `matrix=true`, `matches_expectation=true` for all three, `gpu_free_after=true`, `no_worker_left=true`.

## 10. What still touches Torch or the CPU

AST facts (`torch_cpu_boundary.json` → T18 in `results/tables.md`): 14 files scanned, 11 import no torch at all, and the only module-scope torch imports are the two `*_device_rollout.py`/`*_neat_env.py` Genesis write boundaries — `examples/locomotion/go2_neat_env.py` and `examples/locomotion/go2_device_rollout.py`; `hover_device_rollout.py` defers all 5 of its torch imports into methods. None of `examples/neuroevolution/*.py` (env_base, evolution_graphs, evox_runtime, policy_graphs, seeding, selfcheck, trainer) imports torch.

But the process gets torch anyway: a fresh `import genesis` takes 11.769 s and leaves `torch` in `sys.modules` (`before=False`, `after=True`), with `torch 2.13.0+rocm7.1`, HIP 7.1.52802, device `AMD Instinct MI300X VF` (T19). `find_spec` shows `tensordict`/`rsl_rl` absent and `evoxc`/`tensorneat` present.

Boundary numbers the probe cites rather than re-measures (T20): the NumPy-residency `read_state` boundary costs 0.4738 ms/step = 5.81% of the generation on Go2 (`env_go2_replay_seed0.json`) and 0.2044 ms/step = 4.38% on hover (`env_hover_replay_seed0.json`); the device path's host traffic is 396 B/step (Go2) and 1,109 B/step (hover) against 95,553.2 B and 85,450.6 B for NumPy, with 2.005 and 1.000 syncs/step (`residency_go2_gpu.json`, `residency_hover_gpu.json`).

CPU-resident work that remains on the device path: the hover `_draw_commands` host RNG (2.71-2.75% of the generation at pop 8, `env_hover_throughput_numpy_cpu_pop8.json`), the `(rows,)` done-mask readback and `np.flatnonzero`, end-reason classification, the Go2 solver-error/base-rpy reads (only on steps that end), and the per-generation seed draws that `env.evaluate` performs outside its own `wall_time_s` counter.

## 11. Reliability and limitations

- **Generation 0 is excluded** from every headline median, because it is first-touch dominated: Go2 on the GPU runs its first generation in 8.234 s against a 1.347 s steady generation (`neat_go2_gpu_device_numpy_tell.json`), and the Go2 NumPy-residency side 8.543 s against 1.6566 s (`residency_go2_gpu.json`). Never quote a single generation.
- **"Cold" ≠ cache-cold** (§3): the quadrants/IREE disk caches were warm, so cold/warm build times differ by 0.13-0.19 s and the pair measures process variance only.
- **Hover device-on-CPU is noise** (0.951x / 1.154x, T8). Do not quote it.
- **One held-out triple** of 3 episode seeds per task, and a **3-generation** learning window: the held-out gap changes sign between seeds (§6), so no generalisation claim is supported.
- **One shared GPU, no parallel measurement**: nothing ran concurrently, `contended=false` everywhere, but the GPU is not exclusive to this machine either (`gpu_after.foreign_pids` empty only proves no compile/rollout was left running).
- **Records were taken with `git.dirty=true` in 33 of 37 cases** (the exceptions are `neat_go2_seed0.json`, `unmatched_baselines.json`, `tell_backend_rocm.json`, `torch_cpu_boundary.json`). The dirty paths are always harness files and other records — `M examples/neuroevolution/benchmarks/bench_runtime.py`, `?? examples/neuroevolution/benchmarks/{bench_envs,common,oracle}.py`, `?? examples/neuroevolution/benchmarks/results/` — never a pilot or a kernel; and every record independently proves the pilots were untouched via `git.baseline_env_diff_empty=true`.
- **Byte counts are estimates**, not a PCIe counter (§7).
- **Not measured, and not estimated**: the Go2 reward terms (aggregate-only `run_advance`), the torch-env performance and reward comparison, any PPO baseline, the `iree@rocm:0` tell (timeout, not a slow result), and `xla` without its PJRT plugin.
- **Record count**: 37 records, not the 38 files in `results/` (the 38th is `results/CONTEXT.md`); `summarize.py` validates and reports the real number.
- Known record inconsistency: none found. One table label was corrected instead — the T13 `best policy held-out improvement` column is the policy's trained−initial improvement on the unseen triple, **not** a train-vs-held-out gap; the gap has to be computed from the two recorded fitnesses (as done in §6).

## 12. Reproduction

Environment for every command: `PYTHONPATH=$PWD/examples/neuroevolution` (the harness imports `common`/`oracle` by bare name) and `MPLCONFIGDIR=$TMPDIR/mpl`; the interpreter is the project venv (`/root/evox-robotics/.venv/bin/python`).

| record(s) | command |
|---|---|
| `env_go2_replay_seed0[_r1].json` | `python examples/neuroevolution/benchmarks/bench_envs.py replay --task go2 --pop-size 8 --horizon 100 --master-seed 0 --repeat 0 --process-role only` (then `--repeat 1`) |
| `env_hover_replay_seed0[_r1].json` | same with `--task hover` |
| `env_go2_throughput_numpy_cpu_pop8[_r1].json` | `python …/bench_runtime.py throughput --task go2 --pop-size 8 --horizon 200 --residency numpy --backend numpy --device cpu:0 --repeat 0 --process-role cold` (then `--repeat 1 --process-role warm`) |
| `env_go2_throughput_numpy_cpu_pop32[_r1].json` | same with `--pop-size 32` |
| `env_hover_throughput_numpy_cpu_pop8[_r1].json`, `…_pop32[_r1].json` | same with `--task hover` |
| `residency_go2_gpu[_r1].json`, `residency_hover_gpu[_r1].json` | `python …/bench_runtime.py residency-parity --task {go2,hover} --pop-size 8 --horizon 200 --gpu --repeat 0 --process-role cold` (then `--repeat 1 --process-role warm`) |
| `residency_hover_device_cpu[_r1].json` | `python …/bench_runtime.py residency-parity --task hover --pop-size 8 --horizon 200 --backend iree --device cpu:0 --repeat 0 --process-role only` (then `--repeat 1 --process-role warm`) |
| `neat_go2_seed{0,1,2}.json`, `neat_hover_seed{0,1,2}.json` | `python …/bench_neat.py train --task {go2,hover} --master-seed {0,1,2} --pop-size 8 --generations 3 --horizon 200 --residency numpy --backend numpy --device cpu:0 --tell-backend numpy --process-role only` |
| `neat_go2_seed0_r1.json` | add `--repeat 1 --process-role warm` |
| `neat_go2_gpu_{device_numpy_tell,numpy_residency}[_r1].json`, `neat_hover_gpu_device[_r1].json` | `python …/bench_neat.py train --task {go2,hover} --master-seed 0 --pop-size 8 --generations 3 --horizon 200 --residency {device,numpy} --backend iree --device rocm:0 --tell-backend numpy --gpu --process-role {cold,warm} --repeat {0,1} --out …` |
| `neat_go2_gpu_device_xla_tell[_r1].json` | same with `--tell-backend xla --pjrt-plugin /tmp/xla_rocm_whl/x/jax_plugins/xla_rocm7/xla_rocm_plugin.so` |
| `neat_determinism_go2.json` | `python …/bench_neat.py check-determinism examples/neuroevolution/benchmarks/results/neat_go2_seed0.json examples/neuroevolution/benchmarks/results/neat_go2_seed0_r1.json --out …/neat_determinism_go2.json` |
| `unmatched_baselines.json` | `python …/bench_probe.py unmatched --out …/unmatched_baselines.json` |
| `tell_backend_rocm.json` | `python …/bench_probe.py tell-backend --all --out …/tell_backend_rocm.json` (≈4.5 min, the IREE timeout dominates) |
| `torch_cpu_boundary.json` | `python …/bench_probe.py torch-cpu --out …/torch_cpu_boundary.json` |
| `SUMMARY`/`tables.md` | `python examples/neuroevolution/benchmarks/summarize.py --markdown examples/neuroevolution/benchmarks/results/tables.md` (`summarize.py check` runs the self-check alone) |

Expected runtime: a CPU training run is ~12 s wall for 3 generations plus ~6.5 s of build (`neat_go2_seed0.json` `phases.total_s`), a GPU trainer run ~40 s plus the 24-27 s build, a throughput/residency pair ~1-2 min, the probes 1-5 min. Every record stores its own `command`, so a record can always be reproduced from itself.

## 13. Record index

37 records (T1 in `results/tables.md` carries the same list with head/status/contended). All `status=ok`, all `contended=false`.

| record | kind | task | seed | repeat | process | short description |
|---|---|---|---|---|---|---|
| `env_go2_replay_seed0` | env_replay | go2 | 0 | 0 | only | oracle parity, 100 steps, pop 8 |
| `env_go2_replay_seed0_r1` | env_replay | go2 | 0 | 1 | only | repeat of the above |
| `env_hover_replay_seed0` | env_replay | hover | 0 | 0 | only | oracle parity, 100 steps, pop 8 |
| `env_hover_replay_seed0_r1` | env_replay | hover | 0 | 1 | only | repeat of the above |
| `env_go2_throughput_numpy_cpu_pop8` | env_throughput | go2 | 0 | 0 | cold | CPU NumPy residency, 24 rows, horizon 200 |
| `env_go2_throughput_numpy_cpu_pop8_r1` | env_throughput | go2 | 0 | 1 | warm | repeat |
| `env_go2_throughput_numpy_cpu_pop32` | env_throughput | go2 | 0 | 0 | cold | CPU NumPy residency, 96 rows |
| `env_go2_throughput_numpy_cpu_pop32_r1` | env_throughput | go2 | 0 | 1 | warm | repeat |
| `env_hover_throughput_numpy_cpu_pop8` | env_throughput | hover | 0 | 0 | cold | CPU NumPy residency, 24 rows |
| `env_hover_throughput_numpy_cpu_pop8_r1` | env_throughput | hover | 0 | 1 | warm | repeat |
| `env_hover_throughput_numpy_cpu_pop32` | env_throughput | hover | 0 | 0 | cold | CPU NumPy residency, 96 rows |
| `env_hover_throughput_numpy_cpu_pop32_r1` | env_throughput | hover | 0 | 1 | warm | repeat |
| `residency_go2_gpu` | env_residency | go2 | 0 | 0 | cold | NumPy vs device residency on the GPU |
| `residency_go2_gpu_r1` | env_residency | go2 | 0 | 1 | warm | repeat |
| `residency_hover_device_cpu` | env_residency | hover | 0 | 0 | only | device residency on `iree@cpu:0` (noise) |
| `residency_hover_device_cpu_r1` | env_residency | hover | 0 | 1 | warm | repeat |
| `residency_hover_gpu` | env_residency | hover | 0 | 0 | cold | NumPy vs device residency on the GPU |
| `residency_hover_gpu_r1` | env_residency | hover | 0 | 1 | warm | repeat |
| `neat_go2_seed0` | neat_train | go2 | 0 | 0 | only | full trainer run, CPU NumPy residency |
| `neat_go2_seed0_r1` | neat_train | go2 | 0 | 1 | warm | determinism partner of the above |
| `neat_go2_seed1` | neat_train | go2 | 1 | 0 | only | full trainer run, seed 1 |
| `neat_go2_seed2` | neat_train | go2 | 2 | 0 | only | full trainer run, seed 2 |
| `neat_go2_gpu_device_numpy_tell` | neat_train | go2 | 0 | 0 | cold | GPU device residency, NumPy tell |
| `neat_go2_gpu_device_numpy_tell_r1` | neat_train | go2 | 0 | 1 | warm | repeat |
| `neat_go2_gpu_device_xla_tell` | neat_train | go2 | 0 | 0 | cold | GPU device residency, XLA tell + PJRT plugin |
| `neat_go2_gpu_device_xla_tell_r1` | neat_train | go2 | 0 | 1 | warm | repeat |
| `neat_go2_gpu_numpy_residency` | neat_train | go2 | 0 | 0 | cold | GPU process, NumPy residency |
| `neat_go2_gpu_numpy_residency_r1` | neat_train | go2 | 0 | 1 | warm | repeat |
| `neat_hover_seed0` | neat_train | hover | 0 | 0 | only | full trainer run, CPU NumPy residency |
| `neat_hover_seed1` | neat_train | hover | 1 | 0 | only | full trainer run, seed 1 |
| `neat_hover_seed2` | neat_train | hover | 2 | 0 | only | full trainer run, seed 2 |
| `neat_hover_gpu_device` | neat_train | hover | 0 | 0 | cold | GPU device residency, NumPy tell |
| `neat_hover_gpu_device_r1` | neat_train | hover | 0 | 1 | warm | repeat |
| `neat_determinism_go2` | neat_determinism | go2 | 0 | 1 | only | cross-process determinism verdict |
| `unmatched_baselines` | probe_unmatched_baselines | go2+hover | 0 | 0 | only | torch envs, thin adapter, offline install, PPO |
| `tell_backend_rocm` | probe_tell_backend | go2 | 0 | 0 | only | `numpy@cpu:0` / `xla@rocm:0` / `iree@rocm:0` tell matrix |
| `torch_cpu_boundary` | probe_torch_cpu_boundary | go2+hover | 0 | 0 | only | AST scan, `import genesis` probe, cited boundary numbers |

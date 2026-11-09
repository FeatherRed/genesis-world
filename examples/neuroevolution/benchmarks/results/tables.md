# Stage-3 benchmark tables

Generated from `results/summary.json` by `summarize.py`; do not edit by hand.
Every table names the record files it was derived from, and every record is validated for schema, baseline provenance and status before it is aggregated.

Records: 37 (37 ok, 0 failed, 0 contended). Baseline `ad2f07e5aec1ba931f5a6704253f70e6fa6370b2`; `git.baseline_env_diff_empty` in every record: true.
Host: AMD Instinct MI300X VF, ROCm driver 6.16.6, 1-minute load average 0.76-2.16 over 37 records.

Validation warnings: none.

### T1 Record inventory

Head shas differ per record family because the harness evolved across commits; the pilot sources are byte-identical to the baseline in every record (`git.baseline_env_diff_empty`).

| record | kind | task | seed | repeat | process | head | status | contended |
|---|---|---|---|---|---|---|---|---|
| env_go2_replay_seed0 | env_replay | go2 | 0 | 0 | only | 7c567d49 | ok | false |
| env_go2_replay_seed0_r1 | env_replay | go2 | 0 | 1 | only | 7c567d49 | ok | false |
| env_hover_replay_seed0 | env_replay | hover | 0 | 0 | only | 7c567d49 | ok | false |
| env_hover_replay_seed0_r1 | env_replay | hover | 0 | 1 | only | 7c567d49 | ok | false |
| env_go2_throughput_numpy_cpu_pop32 | env_throughput | go2 | 0 | 0 | cold | cdda51ac | ok | false |
| env_go2_throughput_numpy_cpu_pop32_r1 | env_throughput | go2 | 0 | 1 | warm | cdda51ac | ok | false |
| env_go2_throughput_numpy_cpu_pop8 | env_throughput | go2 | 0 | 0 | cold | cdda51ac | ok | false |
| env_go2_throughput_numpy_cpu_pop8_r1 | env_throughput | go2 | 0 | 1 | warm | cdda51ac | ok | false |
| env_hover_throughput_numpy_cpu_pop32 | env_throughput | hover | 0 | 0 | cold | cdda51ac | ok | false |
| env_hover_throughput_numpy_cpu_pop32_r1 | env_throughput | hover | 0 | 1 | warm | cdda51ac | ok | false |
| env_hover_throughput_numpy_cpu_pop8 | env_throughput | hover | 0 | 0 | cold | cdda51ac | ok | false |
| env_hover_throughput_numpy_cpu_pop8_r1 | env_throughput | hover | 0 | 1 | warm | cdda51ac | ok | false |
| residency_go2_gpu | env_residency | go2 | 0 | 0 | cold | cdda51ac | ok | false |
| residency_go2_gpu_r1 | env_residency | go2 | 0 | 1 | warm | cdda51ac | ok | false |
| residency_hover_device_cpu | env_residency | hover | 0 | 0 | only | cdda51ac | ok | false |
| residency_hover_device_cpu_r1 | env_residency | hover | 0 | 1 | warm | cdda51ac | ok | false |
| residency_hover_gpu | env_residency | hover | 0 | 0 | cold | cdda51ac | ok | false |
| residency_hover_gpu_r1 | env_residency | hover | 0 | 1 | warm | cdda51ac | ok | false |
| neat_go2_gpu_device_numpy_tell | neat_train | go2 | 0 | 0 | cold | 223049e1 | ok | false |
| neat_go2_gpu_device_numpy_tell_r1 | neat_train | go2 | 0 | 1 | warm | 223049e1 | ok | false |
| neat_go2_gpu_device_xla_tell | neat_train | go2 | 0 | 0 | cold | 223049e1 | ok | false |
| neat_go2_gpu_device_xla_tell_r1 | neat_train | go2 | 0 | 1 | warm | 223049e1 | ok | false |
| neat_go2_gpu_numpy_residency | neat_train | go2 | 0 | 0 | cold | 223049e1 | ok | false |
| neat_go2_gpu_numpy_residency_r1 | neat_train | go2 | 0 | 1 | warm | 223049e1 | ok | false |
| neat_go2_seed0 | neat_train | go2 | 0 | 0 | only | 223049e1 | ok | false |
| neat_go2_seed0_r1 | neat_train | go2 | 0 | 1 | warm | 223049e1 | ok | false |
| neat_go2_seed1 | neat_train | go2 | 1 | 0 | only | 223049e1 | ok | false |
| neat_go2_seed2 | neat_train | go2 | 2 | 0 | only | 223049e1 | ok | false |
| neat_hover_gpu_device | neat_train | hover | 0 | 0 | cold | 223049e1 | ok | false |
| neat_hover_gpu_device_r1 | neat_train | hover | 0 | 1 | warm | 223049e1 | ok | false |
| neat_hover_seed0 | neat_train | hover | 0 | 0 | only | 223049e1 | ok | false |
| neat_hover_seed1 | neat_train | hover | 1 | 0 | only | 223049e1 | ok | false |
| neat_hover_seed2 | neat_train | hover | 2 | 0 | only | 223049e1 | ok | false |
| neat_determinism_go2 | neat_determinism | go2 | 0 | 1 | only | 223049e1 | ok | false |
| unmatched_baselines | probe_unmatched_baselines | go2+hover | 0 | 0 | only | 1f2994a2 | ok | false |
| tell_backend_rocm | probe_tell_backend | go2 | 0 | 0 | only | 250f77fc | ok | false |
| torch_cpu_boundary | probe_torch_cpu_boundary | go2+hover | 0 | 0 | only | 42230b38 | ok | false |

Sources: `env_go2_replay_seed0.json`, `env_go2_replay_seed0_r1.json`, `env_go2_throughput_numpy_cpu_pop32.json`, `env_go2_throughput_numpy_cpu_pop32_r1.json`, `env_go2_throughput_numpy_cpu_pop8.json`, `env_go2_throughput_numpy_cpu_pop8_r1.json`, `env_hover_replay_seed0.json`, `env_hover_replay_seed0_r1.json`, `env_hover_throughput_numpy_cpu_pop32.json`, `env_hover_throughput_numpy_cpu_pop32_r1.json`, `env_hover_throughput_numpy_cpu_pop8.json`, `env_hover_throughput_numpy_cpu_pop8_r1.json`, `neat_determinism_go2.json`, `neat_go2_gpu_device_numpy_tell.json`, `neat_go2_gpu_device_numpy_tell_r1.json`, `neat_go2_gpu_device_xla_tell.json`, `neat_go2_gpu_device_xla_tell_r1.json`, `neat_go2_gpu_numpy_residency.json`, `neat_go2_gpu_numpy_residency_r1.json`, `neat_go2_seed0.json`, `neat_go2_seed0_r1.json`, `neat_go2_seed1.json`, `neat_go2_seed2.json`, `neat_hover_gpu_device.json`, `neat_hover_gpu_device_r1.json`, `neat_hover_seed0.json`, `neat_hover_seed1.json`, `neat_hover_seed2.json`, `residency_go2_gpu.json`, `residency_go2_gpu_r1.json`, `residency_hover_device_cpu.json`, `residency_hover_device_cpu_r1.json`, `residency_hover_gpu.json`, `residency_hover_gpu_r1.json`, `tell_backend_rocm.json`, `torch_cpu_boundary.json`, `unmatched_baselines.json`

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

### T3 Environment replay throughput per generation

`gen 0` is the first generation in a fresh process; `gen 1` is steady state for this 2-generation replay.

| task | record | gen | steps | steps/s | env-steps/s | wall s |
|---|---|---|---|---|---|---|
| go2 | env_go2_replay_seed0 | 0 | 100 | 123.64 | 2967.4 | 0.8088 |
| go2 | env_go2_replay_seed0 | 1 | 100 | 181.31 | 4351.4 | 0.5515 |
| go2 | env_go2_replay_seed0_r1 | 0 | 100 | 121.72 | 2921.3 | 0.8216 |
| go2 | env_go2_replay_seed0_r1 | 1 | 100 | 176.10 | 4226.3 | 0.5679 |
| hover | env_hover_replay_seed0 | 0 | 100 | 214.63 | 5151.2 | 0.4659 |
| hover | env_hover_replay_seed0 | 1 | 100 | 237.44 | 5698.7 | 0.4212 |
| hover | env_hover_replay_seed0_r1 | 0 | 100 | 215.68 | 5176.4 | 0.4636 |
| hover | env_hover_replay_seed0_r1 | 1 | 100 | 237.06 | 5689.5 | 0.4218 |

Sources: `env_go2_replay_seed0.json`, `env_go2_replay_seed0_r1.json`, `env_hover_replay_seed0.json`, `env_hover_replay_seed0_r1.json`

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

### T5 Per-step time split, milliseconds per step (timed generations)

`other` = graph_obs + graph_controls + graph_fitness + download + stage_upload + sync.

| task | pop | record | physics | graph_policy | graph_advance | state_read | write_targets | write_reset | host_rng | other | residual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| go2 | 32 | env_go2_throughput_numpy_cpu_pop32 | 4535.0036 | 5361.0610 | 813.7174 | 491.1241 | 105.8692 | 22.1712 | 0.2150 | 0.0019 | 0.0725 |
| go2 | 32 | env_go2_throughput_numpy_cpu_pop32_r1 | 4605.9630 | 5560.4425 | 832.5719 | 501.6673 | 109.3403 | 22.2989 | 0.2047 | 0.0020 | 0.0724 |
| go2 | 8 | env_go2_throughput_numpy_cpu_pop8 | 1730.1976 | 2448.7202 | 798.0671 | 452.7794 | 102.2876 | 17.4221 | 0.2139 | 0.0018 | 0.0670 |
| go2 | 8 | env_go2_throughput_numpy_cpu_pop8_r1 | 1750.8568 | 2469.0772 | 803.3368 | 457.2536 | 102.7689 | 17.5083 | 0.2130 | 0.0019 | 0.0681 |
| hover | 32 | env_hover_throughput_numpy_cpu_pop32 | 1138.0966 | 3096.9732 | 1035.3428 | 221.5570 | 436.8566 | 5.3759 | 435.6385 | 0.0597 | 0.1595 |
| hover | 32 | env_hover_throughput_numpy_cpu_pop32_r1 | 1114.3506 | 3060.3652 | 1017.4079 | 214.5378 | 425.5608 | 5.7324 | 425.0929 | 0.0578 | 0.1556 |
| hover | 8 | env_hover_throughput_numpy_cpu_pop8 | 679.1731 | 1668.1774 | 1020.0917 | 204.3373 | 429.8336 | 4.9681 | 115.5765 | 0.0539 | 0.0898 |
| hover | 8 | env_hover_throughput_numpy_cpu_pop8_r1 | 685.4218 | 1645.1215 | 992.7145 | 204.6851 | 423.7789 | 4.9859 | 116.0746 | 0.0544 | 0.0911 |

Sources: `env_go2_throughput_numpy_cpu_pop32.json`, `env_go2_throughput_numpy_cpu_pop32_r1.json`, `env_go2_throughput_numpy_cpu_pop8.json`, `env_go2_throughput_numpy_cpu_pop8_r1.json`, `env_hover_throughput_numpy_cpu_pop32.json`, `env_hover_throughput_numpy_cpu_pop32_r1.json`, `env_hover_throughput_numpy_cpu_pop8.json`, `env_hover_throughput_numpy_cpu_pop8_r1.json`

### T6 Per-step time share of the measured generation wall clock

Shares are exact sums over the three timed generations (`per_step_means[*].seconds` / timed wall clock); they add to 100% minus the residual.

| task | pop | record | physics | graph_policy | graph_advance | state_read | write_targets | write_reset | host_rng | other | residual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| go2 | 32 | env_go2_throughput_numpy_cpu_pop32 | 39.77% | 47.02% | 7.14% | 4.31% | 0.93% | 0.19% | 1.89e-03% | 0.02% | 0.64% |
| go2 | 32 | env_go2_throughput_numpy_cpu_pop32_r1 | 39.35% | 47.51% | 7.11% | 4.29% | 0.93% | 0.19% | 1.75e-03% | 0.02% | 0.62% |
| go2 | 8 | env_go2_throughput_numpy_cpu_pop8 | 30.80% | 43.60% | 14.21% | 8.06% | 1.82% | 0.31% | 3.81e-03% | 0.03% | 1.19% |
| go2 | 8 | env_go2_throughput_numpy_cpu_pop8_r1 | 30.88% | 43.55% | 14.17% | 8.07% | 1.81% | 0.31% | 3.76e-03% | 0.03% | 1.20% |
| hover | 32 | env_hover_throughput_numpy_cpu_pop32 | 17.29% | 47.04% | 15.73% | 3.37% | 6.64% | 0.08% | 6.62% | 0.91% | 2.42% |
| hover | 32 | env_hover_throughput_numpy_cpu_pop32_r1 | 17.22% | 47.30% | 15.72% | 3.32% | 6.58% | 0.09% | 6.57% | 0.89% | 2.41% |
| hover | 8 | env_hover_throughput_numpy_cpu_pop8 | 15.93% | 39.13% | 23.93% | 4.79% | 10.08% | 0.12% | 2.71% | 1.26% | 2.11% |
| hover | 8 | env_hover_throughput_numpy_cpu_pop8_r1 | 16.26% | 39.03% | 23.55% | 4.86% | 10.05% | 0.12% | 2.75% | 1.29% | 2.16% |

Sources: `env_go2_throughput_numpy_cpu_pop32.json`, `env_go2_throughput_numpy_cpu_pop32_r1.json`, `env_go2_throughput_numpy_cpu_pop8.json`, `env_go2_throughput_numpy_cpu_pop8_r1.json`, `env_hover_throughput_numpy_cpu_pop32.json`, `env_hover_throughput_numpy_cpu_pop32_r1.json`, `env_hover_throughput_numpy_cpu_pop8.json`, `env_hover_throughput_numpy_cpu_pop8_r1.json`

### T7 Batch scaling: pop 8 (24 rows) vs pop 32 (96 rows), horizon 200

Rows are policy-major (3 episodes per policy), so `rows = pop_size * 3`. Throughput does not scale linearly with the batch.

| task | pop | rows | steps/s median | env-steps/s median | generation wall s | first gen s | record |
|---|---|---|---|---|---|---|---|
| go2 | 32 | 96 | 87.64 | 8413.2 | 2.2821 | 2.545 | env_go2_throughput_numpy_cpu_pop32 |
| go2 | 32 | 96 | 85.31 | 8190.0 | 2.3443 | 2.619 | env_go2_throughput_numpy_cpu_pop32_r1 |
| go2 | 8 | 24 | 177.78 | 4266.8 | 1.1250 | 1.387 | env_go2_throughput_numpy_cpu_pop8 |
| go2 | 8 | 24 | 176.06 | 4225.5 | 1.1360 | 1.384 | env_go2_throughput_numpy_cpu_pop8_r1 |
| hover | 32 | 96 | 151.95 | 14587.5 | 1.3162 | 1.393 | env_hover_throughput_numpy_cpu_pop32 |
| hover | 32 | 96 | 154.62 | 14843.7 | 1.2935 | 1.358 | env_hover_throughput_numpy_cpu_pop32_r1 |
| hover | 8 | 24 | 234.46 | 5627.0 | 0.8530 | 0.893 | env_hover_throughput_numpy_cpu_pop8 |
| hover | 8 | 24 | 237.07 | 5689.6 | 0.8436 | 1.008 | env_hover_throughput_numpy_cpu_pop8_r1 |

Sources: `env_go2_throughput_numpy_cpu_pop32.json`, `env_go2_throughput_numpy_cpu_pop32_r1.json`, `env_go2_throughput_numpy_cpu_pop8.json`, `env_go2_throughput_numpy_cpu_pop8_r1.json`, `env_hover_throughput_numpy_cpu_pop32.json`, `env_hover_throughput_numpy_cpu_pop32_r1.json`, `env_hover_throughput_numpy_cpu_pop8.json`, `env_hover_throughput_numpy_cpu_pop8_r1.json`

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

### T10 Residency build, first generation and steady state per side

`build s` is the whole side's construction (scene + every graph); `graph build total s` is the sum of the per-graph compile times only.

| task | record | side | build s | graph build total s | first gen s | steady gen s | steps/s median | env-steps/s median | cache hits/misses | checks pass |
|---|---|---|---|---|---|---|---|---|---|---|
| go2 | residency_go2_gpu | numpy | 26.82 | 4.0569 | 8.543 | 1.6566 | 120.73 | 2897.6 | 16/4 | True |
| go2 | residency_go2_gpu | device | 6.50 | 4.0267 | 1.327 | 1.2401 | 161.27 | 3870.6 | 16/4 | True |
| go2 | residency_go2_gpu_r1 | numpy | 27.43 | 4.0888 | 8.427 | 1.6677 | 119.92 | 2878.2 | 16/4 | True |
| go2 | residency_go2_gpu_r1 | device | 6.51 | 4.0363 | 1.346 | 1.2546 | 159.41 | 3825.8 | 16/4 | True |
| hover | residency_hover_device_cpu | numpy | 7.34 | 3.3378 | 1.101 | 0.9099 | 219.80 | 5275.3 | 40/5 | True |
| hover | residency_hover_device_cpu | device | 3.49 | 3.2124 | 1.075 | 0.9568 | 209.03 | 5016.6 | 40/5 | True |
| hover | residency_hover_device_cpu_r1 | numpy | 7.28 | 3.3026 | 1.063 | 0.9476 | 211.06 | 5065.5 | 40/5 | True |
| hover | residency_hover_device_cpu_r1 | device | 3.48 | 3.2063 | 0.824 | 0.8210 | 243.62 | 5846.8 | 40/5 | True |
| hover | residency_hover_gpu | numpy | 23.73 | 4.0791 | 2.519 | 1.8988 | 105.33 | 2527.9 | 40/5 | True |
| hover | residency_hover_gpu | device | 4.51 | 4.1421 | 1.167 | 1.1935 | 167.57 | 4021.8 | 40/5 | True |
| hover | residency_hover_gpu_r1 | numpy | 24.08 | 4.1103 | 2.585 | 1.9288 | 103.69 | 2488.6 | 40/5 | True |
| hover | residency_hover_gpu_r1 | device | 4.43 | 4.0635 | 1.176 | 1.1897 | 168.11 | 4034.7 | 40/5 | True |

Sources: `residency_go2_gpu.json`, `residency_go2_gpu_r1.json`, `residency_hover_device_cpu.json`, `residency_hover_device_cpu_r1.json`, `residency_hover_gpu.json`, `residency_hover_gpu_r1.json`

### T11 Per-graph compile times per residency side (seconds)

`n/a` means the task has no such graph (Go2 has no controls; no residency side builds a tell graph).

| task | record | side | advance | policy | obs | fitness | controls | reset_obs | tell | total |
|---|---|---|---|---|---|---|---|---|---|---|
| go2 | residency_go2_gpu | numpy | 1.0676 | 2.0910 | 0.6021 | 0.2961 | n/a | n/a | n/a | 4.0569 |
| go2 | residency_go2_gpu | device | 1.0726 | 2.1195 | 0.5567 | 0.2780 | n/a | n/a | n/a | 4.0267 |
| go2 | residency_go2_gpu_r1 | numpy | 1.0973 | 2.1087 | 0.5996 | 0.2832 | n/a | n/a | n/a | 4.0888 |
| go2 | residency_go2_gpu_r1 | device | 1.0874 | 2.0851 | 0.5709 | 0.2929 | n/a | n/a | n/a | 4.0363 |
| hover | residency_hover_device_cpu | numpy | 1.1901 | 1.4431 | n/a | 0.1972 | 0.2718 | 0.2355 | n/a | 3.3378 |
| hover | residency_hover_device_cpu | device | 1.1697 | 1.4002 | n/a | 0.1977 | 0.2086 | 0.2363 | n/a | 3.2124 |
| hover | residency_hover_device_cpu_r1 | numpy | 1.1868 | 1.4016 | n/a | 0.2033 | 0.2732 | 0.2376 | n/a | 3.3026 |
| hover | residency_hover_device_cpu_r1 | device | 1.1811 | 1.3917 | n/a | 0.1962 | 0.2020 | 0.2352 | n/a | 3.2063 |
| hover | residency_hover_gpu | numpy | 1.3015 | 1.8059 | n/a | 0.2838 | 0.3484 | 0.3394 | n/a | 4.0791 |
| hover | residency_hover_gpu | device | 1.3531 | 1.8685 | n/a | 0.2906 | 0.2963 | 0.3336 | n/a | 4.1421 |
| hover | residency_hover_gpu_r1 | numpy | 1.3110 | 1.8191 | n/a | 0.2907 | 0.3470 | 0.3426 | n/a | 4.1103 |
| hover | residency_hover_gpu_r1 | device | 1.3076 | 1.8288 | n/a | 0.2885 | 0.3059 | 0.3325 | n/a | 4.0635 |

Sources: `residency_go2_gpu.json`, `residency_go2_gpu_r1.json`, `residency_hover_device_cpu.json`, `residency_hover_device_cpu_r1.json`, `residency_hover_gpu.json`, `residency_hover_gpu_r1.json`

### T12 TensorNEAT trainer: per-generation learning curves (pop 8, 3 generations, horizon 200)

Rows include the CPU NumPy-residency and the GPU runs; T14 states which is which. `evaluate_s` is `env.evaluate` (seed draw + reset + steps + fitness graph), `generation_s` the whole generation including `tell_s`.

| task | seed | gen | best fitness | median fitness | cumulative env steps | cumulative loop s | evaluate s | tell s | generation s | steps/s | record |
|---|---|---|---|---|---|---|---|---|---|---|---|
| go2 | 0 | 0 | -7.37847 | -51.77536 | 4800 | 8.234 | 8.2230 | 0.01078 | 8.2339 | 24.35 | neat_go2_gpu_device_numpy_tell |
| go2 | 0 | 1 | -7.31804 | -19.02516 | 9600 | 9.564 | 1.3233 | 0.00646 | 1.3301 | 151.24 | neat_go2_gpu_device_numpy_tell |
| go2 | 0 | 2 | -6.03995 | -11.12154 | 14400 | 10.927 | 1.3565 | 0.00636 | 1.3632 | 147.53 | neat_go2_gpu_device_numpy_tell |
| go2 | 0 | 0 | -7.37847 | -51.77536 | 4800 | 8.031 | 8.0200 | 0.01083 | 8.0310 | 24.96 | neat_go2_gpu_device_numpy_tell_r1 |
| go2 | 0 | 1 | -7.31804 | -19.02516 | 9600 | 9.347 | 1.3089 | 0.00632 | 1.3155 | 152.90 | neat_go2_gpu_device_numpy_tell_r1 |
| go2 | 0 | 2 | -6.03995 | -11.12154 | 14400 | 10.688 | 1.3349 | 0.00610 | 1.3413 | 149.91 | neat_go2_gpu_device_numpy_tell_r1 |
| go2 | 0 | 0 | -7.37847 | -51.77536 | 4800 | 9.928 | 8.1173 | 1.81013 | 9.9277 | 24.67 | neat_go2_gpu_device_xla_tell |
| go2 | 0 | 1 | -7.31804 | -19.02516 | 9600 | 11.275 | 1.3407 | 0.00598 | 1.3471 | 149.28 | neat_go2_gpu_device_xla_tell |
| go2 | 0 | 2 | -6.03995 | -11.12154 | 14400 | 12.636 | 1.3553 | 0.00595 | 1.3615 | 147.68 | neat_go2_gpu_device_xla_tell |
| go2 | 0 | 0 | -7.37847 | -51.77536 | 4800 | 9.889 | 8.0838 | 1.80493 | 9.8890 | 24.77 | neat_go2_gpu_device_xla_tell_r1 |
| go2 | 0 | 1 | -7.31804 | -19.02516 | 9600 | 11.222 | 1.3270 | 0.00601 | 1.3334 | 150.85 | neat_go2_gpu_device_xla_tell_r1 |
| go2 | 0 | 2 | -6.03995 | -11.12154 | 14400 | 12.574 | 1.3453 | 0.00587 | 1.3515 | 148.79 | neat_go2_gpu_device_xla_tell_r1 |
| go2 | 0 | 0 | -7.37847 | -51.77536 | 4800 | 8.461 | 8.4491 | 0.01127 | 8.4605 | 23.69 | neat_go2_gpu_numpy_residency |
| go2 | 0 | 1 | -7.31804 | -19.02516 | 9600 | 10.156 | 1.6892 | 0.00637 | 1.6959 | 118.47 | neat_go2_gpu_numpy_residency |
| go2 | 0 | 2 | -6.03995 | -11.12154 | 14400 | 11.893 | 1.7301 | 0.00638 | 1.7367 | 115.67 | neat_go2_gpu_numpy_residency |
| go2 | 0 | 0 | -7.37847 | -51.77536 | 4800 | 8.516 | 8.5053 | 0.01086 | 8.5163 | 23.53 | neat_go2_gpu_numpy_residency_r1 |
| go2 | 0 | 1 | -7.31804 | -19.02516 | 9600 | 10.220 | 1.6968 | 0.00648 | 1.7036 | 117.93 | neat_go2_gpu_numpy_residency_r1 |
| go2 | 0 | 2 | -6.03995 | -11.12154 | 14400 | 11.952 | 1.7256 | 0.00628 | 1.7322 | 115.96 | neat_go2_gpu_numpy_residency_r1 |
| go2 | 0 | 0 | -7.37847 | -51.77536 | 4800 | 1.348 | 1.3389 | 0.00912 | 1.3482 | 150.00 | neat_go2_seed0 |
| go2 | 0 | 1 | -7.31804 | -19.02516 | 9600 | 2.588 | 1.2333 | 0.00613 | 1.2397 | 162.23 | neat_go2_seed0 |
| go2 | 0 | 2 | -6.03995 | -11.12154 | 14400 | 3.874 | 1.2800 | 0.00616 | 1.2865 | 156.29 | neat_go2_seed0 |
| go2 | 0 | 0 | -7.37847 | -51.77536 | 4800 | 1.359 | 1.3492 | 0.00949 | 1.3589 | 148.92 | neat_go2_seed0_r1 |
| go2 | 0 | 1 | -7.31804 | -19.02516 | 9600 | 2.600 | 1.2346 | 0.00610 | 1.2410 | 162.05 | neat_go2_seed0_r1 |
| go2 | 0 | 2 | -6.03995 | -11.12154 | 14400 | 3.914 | 1.3076 | 0.00610 | 1.3140 | 153.00 | neat_go2_seed0_r1 |
| go2 | 1 | 0 | -7.06997 | -51.51889 | 4800 | 1.398 | 1.3877 | 0.00994 | 1.3978 | 144.79 | neat_go2_seed1 |
| go2 | 1 | 1 | -5.28129 | -9.12720 | 9600 | 2.684 | 1.2795 | 0.00684 | 1.2866 | 156.37 | neat_go2_seed1 |
| go2 | 1 | 2 | -5.06396 | -15.85009 | 14400 | 3.948 | 1.2568 | 0.00602 | 1.2631 | 159.19 | neat_go2_seed1 |
| go2 | 2 | 0 | -24.60638 | -102.84394 | 4800 | 1.354 | 1.3445 | 0.00903 | 1.3537 | 149.45 | neat_go2_seed2 |
| go2 | 2 | 1 | -14.29369 | -29.48216 | 9600 | 2.629 | 1.2691 | 0.00587 | 1.2752 | 157.64 | neat_go2_seed2 |
| go2 | 2 | 2 | -8.19573 | -12.13832 | 14400 | 3.882 | 1.2473 | 0.00576 | 1.2533 | 160.40 | neat_go2_seed2 |
| hover | 0 | 0 | -0.14863 | -0.39987 | 4800 | 2.007 | 1.9966 | 0.01040 | 2.0072 | 102.62 | neat_hover_gpu_device |
| hover | 0 | 1 | -0.16914 | -0.19751 | 9600 | 3.244 | 1.2302 | 0.00632 | 1.2368 | 162.87 | neat_hover_gpu_device |
| hover | 0 | 2 | -0.18867 | -0.20551 | 14400 | 4.487 | 1.2368 | 0.00610 | 1.2432 | 161.96 | neat_hover_gpu_device |
| hover | 0 | 0 | -0.14863 | -0.39987 | 4800 | 1.870 | 1.8589 | 0.01111 | 1.8702 | 110.28 | neat_hover_gpu_device_r1 |
| hover | 0 | 1 | -0.16914 | -0.19751 | 9600 | 3.130 | 1.2520 | 0.00779 | 1.2601 | 160.03 | neat_hover_gpu_device_r1 |
| hover | 0 | 2 | -0.18867 | -0.20551 | 14400 | 4.387 | 1.2501 | 0.00646 | 1.2569 | 160.25 | neat_hover_gpu_device_r1 |
| hover | 0 | 0 | -0.14863 | -0.39967 | 4800 | 0.896 | 0.8861 | 0.00948 | 0.8958 | 225.85 | neat_hover_seed0 |
| hover | 0 | 1 | -0.16914 | -0.19751 | 9600 | 1.808 | 0.9065 | 0.00578 | 0.9126 | 220.76 | neat_hover_seed0 |
| hover | 0 | 2 | -0.18867 | -0.20551 | 14400 | 2.732 | 0.9175 | 0.00614 | 0.9239 | 218.10 | neat_hover_seed0 |
| hover | 1 | 0 | -0.17996 | -0.40829 | 4800 | 0.945 | 0.9350 | 0.01000 | 0.9451 | 214.06 | neat_hover_seed1 |
| hover | 1 | 1 | -0.19197 | -0.22464 | 9600 | 1.914 | 0.9625 | 0.00636 | 0.9692 | 207.93 | neat_hover_seed1 |
| hover | 1 | 2 | -0.21210 | -0.22250 | 14400 | 2.813 | 0.8915 | 0.00648 | 0.8982 | 224.47 | neat_hover_seed1 |
| hover | 2 | 0 | -0.19966 | -0.21922 | 4800 | 0.888 | 0.8782 | 0.00927 | 0.8876 | 227.91 | neat_hover_seed2 |
| hover | 2 | 1 | -0.16525 | -0.20354 | 9600 | 1.796 | 0.9018 | 0.00609 | 0.9082 | 221.89 | neat_hover_seed2 |
| hover | 2 | 2 | -0.18740 | -0.20248 | 14400 | 2.645 | 0.8437 | 0.00535 | 0.8493 | 237.19 | neat_hover_seed2 |

Sources: `neat_go2_gpu_device_numpy_tell.json`, `neat_go2_gpu_device_numpy_tell_r1.json`, `neat_go2_gpu_device_xla_tell.json`, `neat_go2_gpu_device_xla_tell_r1.json`, `neat_go2_gpu_numpy_residency.json`, `neat_go2_gpu_numpy_residency_r1.json`, `neat_go2_seed0.json`, `neat_go2_seed0_r1.json`, `neat_go2_seed1.json`, `neat_go2_seed2.json`, `neat_hover_gpu_device.json`, `neat_hover_gpu_device_r1.json`, `neat_hover_seed0.json`, `neat_hover_seed1.json`, `neat_hover_seed2.json`

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

### T14 Trainer configurations: NumPy vs device residency, NumPy vs XLA tell, CPU vs GPU physics

`residency` is where the rollout operands live; `tell backend` is where the NEAT generation transition runs. `neat_*_seed1/2` are master seeds 1 and 2; every other run is seed 0.

| task | record | residency | tell backend | process | repeat | steady gen s | steps/s median | env-steps/s median | tell build s | first gen s | generation_s total | held-out trained median | fingerprint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| go2 | neat_go2_gpu_device_numpy_tell | device | numpy | cold | 0 | 1.3467 | 147.53 | 3540.7 | 0.396 | 8.234 | 10.9272 | -10.93208 | 6d71d28b8b1e |
| go2 | neat_go2_gpu_device_numpy_tell_r1 | device | numpy | warm | 1 | 1.3284 | 149.91 | 3597.9 | 0.379 | 8.031 | 10.6878 | -10.93208 | 6d71d28b8b1e |
| go2 | neat_go2_gpu_device_xla_tell | device | xla | cold | 0 | 1.3543 | 147.68 | 3544.4 | 11.791 | 9.928 | 12.6363 | -10.93208 | 0d9c7a433bc6 |
| go2 | neat_go2_gpu_device_xla_tell_r1 | device | xla | warm | 1 | 1.3424 | 148.79 | 3571.1 | 11.794 | 9.889 | 12.5738 | -10.93208 | 0d9c7a433bc6 |
| go2 | neat_go2_gpu_numpy_residency | numpy | numpy | cold | 0 | 1.7163 | 115.67 | 2776.1 | 0.375 | 8.461 | 11.8932 | -10.93208 | 6d71d28b8b1e |
| go2 | neat_go2_gpu_numpy_residency_r1 | numpy | numpy | warm | 1 | 1.7179 | 115.96 | 2783.1 | 0.393 | 8.516 | 11.9521 | -10.93208 | 6d71d28b8b1e |
| go2 | neat_go2_seed0 | numpy | numpy | only | 0 | 1.2631 | 156.29 | 3751.0 | 0.257 | 1.348 | 3.8744 | -10.93208 | cd6f6b8e9174 |
| go2 | neat_go2_seed0_r1 | numpy | numpy | warm | 1 | 1.2775 | 153.00 | 3671.9 | 0.258 | 1.359 | 3.9139 | -10.93208 | cd6f6b8e9174 |
| go2 | neat_go2_seed1 | numpy | numpy | only | 0 | 1.2749 | 156.37 | 3752.8 | 0.263 | 1.398 | 3.9476 | -11.73538 | f9fb76ab0ff4 |
| go2 | neat_go2_seed2 | numpy | numpy | only | 0 | 1.2643 | 157.64 | 3783.5 | 0.259 | 1.354 | 3.8822 | -15.72710 | e153cdae5a94 |
| hover | neat_hover_gpu_device | device | numpy | cold | 0 | 1.2400 | 161.96 | 3887.1 | 0.371 | 2.007 | 4.4872 | -0.18983 | 7bfa207c13df |
| hover | neat_hover_gpu_device_r1 | device | numpy | warm | 1 | 1.2585 | 160.03 | 3840.7 | 0.381 | 1.870 | 4.3872 | -0.18983 | 7bfa207c13df |
| hover | neat_hover_seed0 | numpy | numpy | only | 0 | 0.9182 | 220.76 | 5298.2 | 0.258 | 0.896 | 2.7322 | -0.18983 | 1a13b28ad1e2 |
| hover | neat_hover_seed1 | numpy | numpy | only | 0 | 0.9337 | 214.06 | 5137.3 | 0.271 | 0.945 | 2.8125 | -0.17063 | fe300d8a29ed |
| hover | neat_hover_seed2 | numpy | numpy | only | 0 | 0.8788 | 227.91 | 5469.9 | 0.364 | 0.888 | 2.6451 | -0.18184 | 2a55c890e965 |

Sources: `neat_go2_gpu_device_numpy_tell.json`, `neat_go2_gpu_device_numpy_tell_r1.json`, `neat_go2_gpu_device_xla_tell.json`, `neat_go2_gpu_device_xla_tell_r1.json`, `neat_go2_gpu_numpy_residency.json`, `neat_go2_gpu_numpy_residency_r1.json`, `neat_go2_seed0.json`, `neat_go2_seed0_r1.json`, `neat_go2_seed1.json`, `neat_go2_seed2.json`, `neat_hover_gpu_device.json`, `neat_hover_gpu_device_r1.json`, `neat_hover_seed0.json`, `neat_hover_seed1.json`, `neat_hover_seed2.json`

### T15 Determinism cross-check: two processes of the same command must agree bit-for-bit

The compared pair is `neat_go2_seed0.json` (process `only`) and `neat_go2_seed0_r1.json` (process `warm`).

| record | fingerprint A | fingerprint B | match | seed table max diff | fitness max diff | held-out initial max diff | held-out trained max diff | mismatched generations | verdict ok |
|---|---|---|---|---|---|---|---|---|---|
| neat_determinism_go2 | cd6f6b8e917423a4d78f184a518aedf4fa54ebf46a866604e1d68a217638ec67 | cd6f6b8e917423a4d78f184a518aedf4fa54ebf46a866604e1d68a217638ec67 | True | 0 | 0 | 0 | 0 | 0 | True |

Sources: `neat_determinism_go2.json`

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

### T17 tell-graph backend matrix on ROCm (`_tell_worker.py`, pop 4, genome 45 -> 12)

`iree@rocm:0` is killed at its hard timeout inside `build_tell` (`returncode -9`); the phases it reached are in the raw record.

| backend@device | status | elapsed s | last phase | build s | run s | total s | timeout s | PJRT plugin |
|---|---|---|---|---|---|---|---|---|
| iree@rocm:0 | timeout | 240.016 | build_tell_start | n/a | n/a | n/a | 240.0 | n/a |
| numpy@cpu:0 | ok | 0.719 | run_tell_done | 0.271 | 0.009 | 0.590 | n/a | n/a |
| xla@rocm:0 | ok | 11.691 | run_tell_done | 4.724 | 6.161 | 11.269 | n/a | /tmp/xla_rocm_whl/x/jax_plugins/xla_rocm7/xla_rocm_plugin.so |

Sources: `tell_backend_rocm.json`

### T18 Torch/CPU boundary: AST import scan

`torch_at_import_time` = examples/locomotion/go2_neat_env.py, examples/locomotion/go2_device_rollout.py; `no_torch` = 11 of 14 scanned files.

| file | torch at import time | torch inside functions | genesis mentions | tensordict mentions | rsl_rl mentions |
|---|---|---|---|---|---|
| examples/drone/hover_device_rollout.py | no | 5 | 3 | 0 | 0 |
| examples/drone/hover_kernels.py | no | 0 | 0 | 0 | 0 |
| examples/drone/hover_neat_env.py | no | 0 | 2 | 0 | 0 |
| examples/locomotion/go2_device_rollout.py | yes | 0 | 3 | 0 | 0 |
| examples/locomotion/go2_evox_kernels.py | no | 0 | 0 | 0 | 0 |
| examples/locomotion/go2_neat_env.py | yes | 0 | 2 | 0 | 0 |
| examples/locomotion/go2_neat_kernels.py | no | 0 | 0 | 0 | 0 |
| examples/neuroevolution/env_base.py | no | 0 | 0 | 0 | 0 |
| examples/neuroevolution/evolution_graphs.py | no | 0 | 0 | 0 | 0 |
| examples/neuroevolution/evox_runtime.py | no | 0 | 0 | 0 | 0 |
| examples/neuroevolution/policy_graphs.py | no | 0 | 0 | 0 | 0 |
| examples/neuroevolution/seeding.py | no | 0 | 0 | 0 | 0 |
| examples/neuroevolution/selfcheck.py | no | 0 | 0 | 0 | 0 |
| examples/neuroevolution/trainer.py | no | 0 | 0 | 0 | 0 |

Sources: `torch_cpu_boundary.json`

### T19 Torch/CPU boundary: fresh-subprocess `import genesis` probe

`import genesis` pulls `torch` into the process even though the neuroevolution layer does not import it.

| fact | value |
|---|---|
| genesis import time (fresh process) | 11.769 s |
| torch in sys.modules before `import genesis` | False |
| torch in sys.modules after `import genesis` | True |
| torch version | 2.13.0+rocm7.1 |
| torch HIP version | 7.1.52802 |
| HIP device | AMD Instinct MI300X VF |
| `find_spec` tensordict / rsl_rl | False / False |
| `find_spec` evoxc / tensorneat / torch | True / True / True |

Sources: `torch_cpu_boundary.json`

### T20 Torch/CPU boundary: cited committed numbers (never re-measured by the probe)

The probe cites these values from the records named in the `source record` column; the head column is the head that produced the cited record.

| task | family | source record | head | status | state_read ms/step | state_read share | numpy B/step | device B/step | device sync/step |
|---|---|---|---|---|---|---|---|---|---|
| go2 | replay | results/env_go2_replay_seed0.json | 7c567d49 | ok | 0.4738 | 5.81% | n/a | n/a | n/a |
| go2 | residency_gpu | results/residency_go2_gpu.json | cdda51ac | ok | n/a | n/a | 95553.2 | 395.7 | 2.005 |
| go2 | throughput_numpy_cpu_pop8 | results/env_go2_throughput_numpy_cpu_pop8.json | cdda51ac | ok | n/a | n/a | n/a | n/a | n/a |
| hover | replay | results/env_hover_replay_seed0.json | 7c567d49 | ok | 0.2044 | 4.38% | n/a | n/a | n/a |
| hover | residency_gpu | results/residency_hover_gpu.json | cdda51ac | ok | n/a | n/a | 85450.6 | 1109.5 | 1.000 |
| hover | throughput_numpy_cpu_pop8 | results/env_hover_throughput_numpy_cpu_pop8.json | cdda51ac | ok | n/a | n/a | n/a | n/a | n/a |

Sources: `torch_cpu_boundary.json`

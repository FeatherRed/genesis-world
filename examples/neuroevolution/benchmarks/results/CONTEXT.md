# examples/neuroevolution/benchmarks/results

Committed raw records of the Stage-3 benchmark harness: one JSON file per measurement, written by `bench_envs.py` through `common.write_record` (sorted keys, sample lists capped so a file stays below ~150 KB).
`env_<task>_replay_seed<seed>.json` is the primary run; `env_<task>_replay_seed<seed>_r<repeat>.json` is an independent process running the same command, and the two must agree on `measurements.fingerprint.sha256` while their timings differ.
Records are never edited by hand — rerun the command named in `command` to regenerate one, and read `host`/`contended` before trusting a timing.

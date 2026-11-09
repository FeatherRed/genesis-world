"""Minimal process-start worker of the ``bench_probe.py startup`` subcommand.

The Stage-3 report explains that ``gs.init(backend=gs.gpu)`` repairs the ROCm HSA runtime by
re-executing the interpreter (``genesis/utils/hsa.py``), but no committed record measured what that
repair costs. This worker isolates the one region that no other record covers: *process start up to
``gs.init`` returning*, with nothing else in the process.

It must be a real script file invoked by path, because the repair re-executes the interpreter with
``os.execve(sys.executable, [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]], env)``;
``python -c`` or stdin has no script to re-run and would silently skip the repair. It imports no
torch and touches no ``torch.cuda``: the repair is skipped once torch's GPU support is initialized.

Timing marks (all ``time.perf_counter``, in the process that prints the payload -- i.e. the
re-executed child when a repair happens):

* ``t0`` -- module top, before any third-party import;
* ``t1`` -- after ``import genesis as gs``;
* ``t2`` -- after ``gs.init(...)`` returns.

Output protocol (consumed by ``bench_probe._startup_run``), one marked JSON line, like
``_tell_worker.py``'s ``PHASE``/``RESULT`` lines: ``STARTUP {json}``. Everything else the process
writes (warnings) is ignored by the parent.

A bare invocation prints the usage and exits 0, like the other benchmark CLIs and workers, so the
``examples/**/*.py`` smoke harness does not treat this worker as a failing example.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# The measured region starts here: no third-party module has been imported yet.
_T0 = time.perf_counter()

import genesis as gs  # noqa: E402 - this import is part of the measured region

_T1 = time.perf_counter()

MARKER = "STARTUP"

#: ``--backend`` values accepted by the worker, mapped to the ``genesis`` module constants.
BACKENDS = {"cpu": "cpu", "gpu": "gpu"}


def _status_payload(status) -> dict:
    """Serialise an ``HsaRuntimeStatus`` to plain JSON (paths and versions as strings)."""
    import dataclasses

    payload = {}
    for field in dataclasses.fields(status):
        value = getattr(status, field.name)
        payload[field.name] = value if value is None or isinstance(value, bool) else str(value)
    return payload


def main(argv=None) -> int:
    """Import genesis, run ``gs.init`` and print the one JSON payload line."""
    parser = argparse.ArgumentParser(prog="_startup_worker.py", description="process-start probe worker")
    parser.add_argument("--backend", default="gpu", choices=sorted(BACKENDS), help="genesis backend (default: gpu)")
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    gs.init(backend=getattr(gs, BACKENDS[args.backend]), logging_level="warning")
    init_done = time.perf_counter()

    from genesis.utils.hsa import inspect_hsa_runtime

    status = inspect_hsa_runtime()
    payload = {
        "pid": os.getpid(),
        "argv0": sys.argv[0],
        "backend": args.backend,
        "import_s": round(_T1 - _T0, 6),
        "init_s": round(init_done - _T1, 6),
        "module_to_init_s": round(init_done - _T0, 6),
        "reexec_done": os.environ.get("GS_HSA_REEXEC_DONE"),
        "opt_out": os.environ.get("GS_HSA_REEXEC"),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        # ``gs.init`` skips the HSA inspection entirely for the cpu/metal backends, so the CPU
        # control never re-execs even though ``inspect_hsa_runtime`` still reports the mismatch.
        "hsa_check_applies": args.backend != "cpu",
        "device": str(gs.device),
        "hsa_status": _status_payload(status),
    }
    print(f"{MARKER} {json.dumps(payload, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

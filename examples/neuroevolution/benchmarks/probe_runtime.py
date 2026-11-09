"""Process transport of the Phase-4 probe: hard-timeout subprocesses, the tell worker and GPU state.

Every measurement in ``bench_probe.py`` that can hang or fail in a foreign process is run here:

* :func:`run_capture` starts a command in its own session, redirects stdout/stderr to temporary
  files (a pipe could fill and block before the timeout), waits with a hard timeout and kills the
  whole process group on expiry -- the IREE/ROCm compile spawns children;
* :func:`import_case` is one baseline import/adapter attempt (the failure is the measurement);
* :func:`install_attempt` makes the single offline dependency attempt (pip, then uv), writing only
  under ``$TMPDIR``;
* :func:`run_tell_entry` runs ``_tell_worker.py`` and parses its phase trail, so a timeout records
  how far the tell-graph build got;
* :func:`gpu_after` and :func:`tell_worker_processes` prove nothing was left running.

Nothing here imports ``genesis``, ``torch``, ``tensordict`` or a pilot module.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import common
from runtime_measure import _process_tree

#: Hard timeout of one baseline import/adapter subprocess.
CASE_TIMEOUT_S = 60.0

#: Hard timeout of the single offline dependency install attempt (``timeout 90`` in the command).
INSTALL_TIMEOUT_S = 90.0

#: Hard timeout of one ``tell-backend`` worker run (the IREE/ROCm compile is expected to hit it).
TELL_TIMEOUT_S = 240.0

#: Phase-3 measured XLA/ROCm PJRT plugin; the positive GPU control for the tell graph.
DEFAULT_XLA_PLUGIN = "/tmp/xla_rocm_whl/x/jax_plugins/xla_rocm7/xla_rocm_plugin.so"

#: ``ExceptionType: message`` shape of the last traceback line (also ``ImportError`` without colon).
_EXCEPTION_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit))(?::\s*(.*))?$")


def _tail(text: str, lines: int = 30, width: int = 400) -> list[str]:
    """Return the last ``lines`` non-empty lines of ``text``, each clipped to ``width``."""
    kept = [line.rstrip()[:width] for line in text.splitlines() if line.strip()]
    return kept[-lines:]


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill ``proc``'s whole process group (an IREE compile can spawn children)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run_capture(command: list[str], *, timeout: float, env=None, cwd=None) -> dict:
    """Run ``command`` in its own session with a hard timeout and a process-group kill."""
    started = time.perf_counter()
    returncode = None
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="bench_probe_") as tmp:
        out_path = Path(tmp) / "stdout.txt"
        err_path = Path(tmp) / "stderr.txt"
        with out_path.open("wb") as out, err_path.open("wb") as err:
            proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                command,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
            )
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                try:
                    returncode = proc.wait(timeout=15)
                except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL is not ignorable
                    returncode = None
        stdout_text = out_path.read_text(encoding="utf-8", errors="replace")
        stderr_text = err_path.read_text(encoding="utf-8", errors="replace")
    return {
        "command": [str(part) for part in command],
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": round(time.perf_counter() - started, 6),
        "stdout_tail": _tail(stdout_text),
        "stderr_tail": _tail(stderr_text, lines=40),
        "stdout_bytes": len(stdout_text),
        "stderr_bytes": len(stderr_text),
        "_stdout": stdout_text,
        "_stderr": stderr_text,
    }


def _exception_from_stderr(text: str):
    """Return ``{"type", "message"}`` of the last traceback line, or ``None``."""
    for line in reversed(_tail(text, lines=8)):
        match = _EXCEPTION_LINE.match(line.strip())
        if match:
            return {"type": match.group(1), "message": (match.group(2) or "").strip()}
    return None


def probe_env(extra_dir: str | None = None) -> dict:
    """Environment for a probe subprocess: repository root and the case directory on ``PYTHONPATH``."""
    root = common.repo_root()
    parts = [str(root)]
    if extra_dir:
        parts.append(str(root / extra_dir))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl"))
    return env


def import_case(name: str, code: str, extra_dir: str | None, timeout: float) -> dict:
    """Run one ``-c`` import/construction case in its own process and record the outcome."""
    run = run_capture(
        [sys.executable, "-c", code],
        timeout=timeout,
        env=probe_env(extra_dir),
        cwd=common.repo_root(),
    )
    exception = _exception_from_stderr(run.pop("_stderr"))
    run.pop("_stdout")
    verdict = "timeout" if run["timed_out"] else ("ok" if run["returncode"] == 0 else "failed")
    return {
        "name": name,
        "code": code,
        "extra_pythonpath": extra_dir,
        "python": sys.executable,
        "verdict": verdict,
        "exception": exception,
        **run,
    }


def install_attempt(python: str) -> dict:
    """Attempt the offline install once (pip, then uv), writing only under ``$TMPDIR``."""
    target = Path(tempfile.gettempdir()) / "bench_deps"
    command = [
        "timeout",
        str(int(INSTALL_TIMEOUT_S)),
        python,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(target),
        "tensordict",
        "rsl-rl-lib",
    ]
    pip_run = run_capture(command, timeout=INSTALL_TIMEOUT_S + 30.0, env=os.environ.copy())
    pip_run.pop("_stdout")
    pip_stderr = pip_run.pop("_stderr")
    pip_available = "No module named pip" not in pip_stderr
    attempt = {
        "target": str(target),
        "timeout_s": INSTALL_TIMEOUT_S,
        "target_is_outside_venv": str(target).startswith(tempfile.gettempdir()) and "site-packages" not in str(target),
        "pip": {
            "command": pip_run["command"],
            "pip_module_available": pip_available,
            "returncode": pip_run["returncode"],
            "timed_out": pip_run["timed_out"],
            "elapsed_s": pip_run["elapsed_s"],
            "stderr_tail": pip_run["stderr_tail"],
        },
        "uv": {"available": shutil.which("uv") is not None, "which": shutil.which("uv")},
    }
    if pip_available and pip_run["returncode"] == 0 and not pip_run["timed_out"]:
        attempt.update(_probe_installed_target(python, target, "pip"))
        return attempt
    if not pip_available and shutil.which("uv"):
        uv_command = [shutil.which("uv"), "pip", "install", "--no-deps", "--target", str(target), "tensordict", "rsl-rl-lib"]
        uv_run = run_capture(uv_command, timeout=INSTALL_TIMEOUT_S + 30.0, env=os.environ.copy())
        uv_run.pop("_stdout")
        uv_run.pop("_stderr")
        attempt["uv"].update(
            {
                "command": uv_run["command"],
                "returncode": uv_run["returncode"],
                "timed_out": uv_run["timed_out"],
                "elapsed_s": uv_run["elapsed_s"],
                "stderr_tail": uv_run["stderr_tail"],
            }
        )
        if uv_run["returncode"] == 0 and not uv_run["timed_out"]:
            attempt.update(_probe_installed_target(python, target, "uv"))
            return attempt
    attempt.update(
        {
            "method_used": None,
            "status": "unavailable" if not pip_available and not attempt["uv"]["available"] else "failed",
            "target_imports": None,
            "note": (
                "this venv is uv-managed and has no pip module; uv is not on PATH, so the offline "
                "install could not start. Nothing was installed and nothing outside $TMPDIR was written."
                if not pip_available and not attempt["uv"]["available"]
                else "the offline install attempt failed; see the stderr tails"
            ),
            "ppo_training_run": False,
        }
    )
    return attempt


def _probe_installed_target(python: str, target: Path, method: str) -> dict:
    """Record which modules import from an unexpectedly successful install target."""
    code = (
        "import json, sys\n"
        "out = {}\n"
        "for name in ('tensordict', 'rsl_rl'):\n"
        "    try:\n"
        "        __import__(name)\n"
        "        out[name] = {'imported': True, 'file': sys.modules[name].__file__}\n"
        "    except Exception as exc:\n"
        "        out[name] = {'imported': False, 'error': f'{type(exc).__name__}: {exc}'}\n"
        "print('INSTALLED ' + json.dumps(out))\n"
    )
    env = probe_env()
    env["PYTHONPATH"] = os.pathsep.join([str(target), env.get("PYTHONPATH", "")])
    run = run_capture([python, "-c", code], timeout=120.0, env=env, cwd=common.repo_root())
    run.pop("_stderr")
    stdout = run.pop("_stdout")
    payload = None
    for line in stdout.splitlines():
        if line.startswith("INSTALLED "):
            payload = json.loads(line[len("INSTALLED ") :])
    return {
        "method_used": method,
        "status": "succeeded",
        "target_imports": payload,
        "target_import_command": run["command"],
        "ppo_training_run": False,
        "note": (
            "the offline install unexpectedly succeeded and the target imports; PPO training/eval was "
            "still NOT run (needs a GPU scene and a training budget), so PPO stays an unmatched algorithm baseline"
        ),
    }


# --- tell worker ------------------------------------------------------------------
def run_tell_entry(spec: dict, timeout: float) -> dict:
    """Run the tell worker for one backend/device spec and parse its phase trail."""
    entry = {
        "backend": spec["backend"],
        "device": spec["device"],
        "plugin_path": spec["plugin_path"],
        "timeout_s": timeout,
        "status": "failed",
        "phases_reached": [],
        "phase_timings": {},
        "result": None,
        "stdout_tail": [],
        "stderr_tail": [],
    }
    if spec["backend"] == "xla" and spec["plugin_path"] is None:
        entry.update(
            {
                "status": "skipped",
                "reason": f"no XLA/PJRT plugin: {DEFAULT_XLA_PLUGIN} is missing and --pjrt-plugin was not given",
            }
        )
        return entry
    command = [sys.executable, str(Path(__file__).resolve().parent / "_tell_worker.py"), "--backend", str(spec["backend"])]
    if spec["device"] is not None:
        command += ["--device", str(spec["device"])]
    if spec["plugin_path"]:
        command += ["--pjrt-plugin", str(spec["plugin_path"])]
    env = probe_env("examples/neuroevolution")
    if spec["plugin_path"]:
        env.setdefault("EVOXC_PJRT_PLUGIN", str(spec["plugin_path"]))
    run = run_capture(command, timeout=timeout, env=env, cwd=common.repo_root())
    stdout = run.pop("_stdout")
    stderr = run.pop("_stderr")
    entry.update(
        {
            "command": run["command"],
            "returncode": run["returncode"],
            "timed_out": run["timed_out"],
            "elapsed_s": run["elapsed_s"],
            "stdout_tail": run["stdout_tail"],
            "stderr_tail": run["stderr_tail"],
            "stdout_bytes": run["stdout_bytes"],
            "stderr_bytes": run["stderr_bytes"],
        }
    )
    for line in stdout.splitlines():
        if line.startswith("PHASE "):
            try:
                payload = json.loads(line[len("PHASE ") :])
            except json.JSONDecodeError:
                continue
            entry["phases_reached"].append(payload.get("phase"))
            entry["phase_timings"][payload.get("phase")] = payload
        elif line.startswith("RESULT "):
            try:
                entry["result"] = json.loads(line[len("RESULT ") :])
            except json.JSONDecodeError:
                entry["result"] = {"raw": line[len("RESULT ") :]}
    if run["timed_out"]:
        entry["status"] = "timeout"
    elif run["returncode"] == 0 and (entry["result"] or {}).get("status") == "ok":
        entry["status"] = "ok"
    else:
        entry["status"] = "failed"
        if entry["result"] is None:
            entry["result"] = {"status": "failed", "error": _exception_from_stderr(stderr)}
    entry["last_phase"] = entry["phases_reached"][-1] if entry["phases_reached"] else None
    entry["worker_processes_left"] = tell_worker_processes()
    return entry


def tell_worker_processes() -> list[str]:
    """Return the command lines of any live ``_tell_worker.py`` process (expected: empty)."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["ps", "-eo", "pid,args"], capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return []
    own = {str(os.getpid())}
    found = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if "_tell_worker.py" in stripped and not any(stripped.startswith(pid) for pid in own):
            found.append(stripped[:300])
    return found


def gpu_after() -> dict:
    """Snapshot ``rocm-smi --showpids`` after the probes and say whether the GPU is free."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["rocm-smi", "--showpids"], capture_output=True, text=True, check=False, timeout=30
        )
        text = (completed.stdout or "") + (completed.stderr or "")
    except (OSError, subprocess.SubprocessError):
        text = ""
    pids = common.rocm_pids(text)
    own = {str(pid) for pid in _process_tree()}
    foreign = [pid for pid in pids if pid not in own]
    return {
        "rocm_pids": pids,
        "own_tree_pids": sorted(own & set(pids)),
        "foreign_pids": foreign,
        "gpu_free": not foreign,
        "note": (
            "this probe process holds a HIP context because common.gpu_info() queried torch.cuda, so "
            "its own PIDs are excluded; an empty foreign list means no compile or rollout is left running"
        ),
    }
